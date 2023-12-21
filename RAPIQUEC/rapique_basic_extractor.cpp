#include <vector>
#include <numeric>
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <execution>
#include <future>

std::vector<double> nakafit(const std::vector<double>& data);
std::pair<double, double> est_GGD_param(const std::vector<double>& vec);
std::tuple<double, double, double> est_AGGD_param(const std::vector<double>& vec);

void circularShift(const cv::Mat& src, cv::Mat& dst, cv::Point shift) {
    int shift_x = (shift.x % src.cols + src.cols) % src.cols;
    int shift_y = (shift.y % src.rows + src.rows) % src.rows;

    cv::Mat extended;
    cv::copyMakeBorder(src, extended, 0, shift_y, 0, shift_x, cv::BORDER_WRAP);
    cv::Rect roi(shift_x, shift_y, src.cols, src.rows);

    dst = extended(roi);
}

std::vector<double> rapique_basic_extractor(const cv::Mat& img) {
    std::vector<double> ftrs;
    ftrs.reserve(18); 

    const int filtlength = 7;
    const cv::Mat gaussianKernel = cv::getGaussianKernel(filtlength, filtlength / 6.0, CV_64F);
    cv::Mat window = gaussianKernel * gaussianKernel.t();
    window /= cv::sum(window)[0];

    cv::Mat mu, sigma_sq, sigma;
    cv::filter2D(img, mu, CV_64F, window, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
    cv::filter2D(img.mul(img), sigma_sq, CV_64F, window, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
    sigma_sq = cv::max(sigma_sq - mu.mul(mu), 0);
    cv::sqrt(sigma_sq, sigma);
    
    cv::Mat structdis = (img - mu) / (sigma + 1);

    std::vector<double> vec_struct(structdis.begin<double>(), structdis.end<double>());
    auto [gamparam, sigparam] = est_GGD_param(vec_struct);
    ftrs.push_back(gamparam);
    ftrs.push_back(sigparam);

    std::vector<double> sigmaVec(sigma.begin<double>(), sigma.end<double>());
    std::vector<double> sigmaParam = nakafit(sigmaVec);
    ftrs.insert(ftrs.end(), sigmaParam.begin(), sigmaParam.end());

    const std::vector<std::pair<int, int>> shifts = {{0, 1}, {1, 0}, {1, 1}, {-1, 1}};

    std::vector<cv::Mat> pairs(shifts.size());
    std::transform(std::execution::par, shifts.begin(), shifts.end(), pairs.begin(), [&structdis](const std::pair<int, int>& shift) {
        cv::Mat shifted_structdis;
        circularShift(structdis, shifted_structdis, cv::Point(shift.first, shift.second));
        return structdis.mul(shifted_structdis);
    });

    std::vector<std::future<std::tuple<double, double, double, double>>> futures;
        for (const auto& pair : pairs) {
            futures.push_back(std::async(std::launch::async, [&pair]() {
                std::vector<double> pairVec(pair.begin<double>(), pair.end<double>());
                auto [alpha, leftstd, rightstd] = est_AGGD_param(pairVec);
                double meanparam = (rightstd - leftstd) * (std::tgamma(2.0 / alpha) / std::tgamma(1.0 / alpha)) * 
                                (std::sqrt(std::tgamma(1.0 / alpha)) / std::sqrt(std::tgamma(3.0 / alpha)));
                return std::make_tuple(alpha, meanparam, leftstd, rightstd);
            }));
        }

        for (auto& future : futures) {
            auto [alpha, meanparam, leftstd, rightstd] = future.get();
            ftrs.push_back(alpha);
            ftrs.push_back(meanparam);
            ftrs.push_back(leftstd);
            ftrs.push_back(rightstd);
        }

    cv::Mat log_struct;
    cv::log(cv::abs(structdis) + 0.1, log_struct);
    std::vector<cv::Mat> shifted_structs(shifts.size());

    for (size_t i = 0; i < shifts.size(); ++i) {
        circularShift(log_struct, shifted_structs[i], cv::Point(shifts[i].first, shifts[i].second));
        cv::Mat structdis_diff = log_struct - shifted_structs[i];

        std::vector<double> structdis_diff_vec(structdis_diff.begin<double>(), structdis_diff.end<double>());
        auto [gamparam_diff, sigparam_diff] = est_GGD_param(structdis_diff_vec);
        ftrs.push_back(gamparam_diff);
        ftrs.push_back(sigparam_diff);
    }
    
    cv::Mat combined_structdis_diff = log_struct + shifted_structs[2] - shifted_structs[0] - shifted_structs[1];
    std::vector<double> combined_diff_vec(combined_structdis_diff.begin<double>(), combined_structdis_diff.end<double>());
    auto [gamparam_combined, sigparam_combined] = est_GGD_param(combined_diff_vec);
    ftrs.push_back(gamparam_combined);
    ftrs.push_back(sigparam_combined);

    static const cv::Mat win_tmp_1 = (cv::Mat_<double>(3,3) << 0, 1, 0, -1, 0, -1, 0, 1, 0);
    static const cv::Mat win_tmp_2 = (cv::Mat_<double>(3,3) << 1, 0, -1, 0, 0, 0, -1, 0, 1);
    
    cv::Mat structdis_diff_1, structdis_diff_2;
    cv::filter2D(log_struct, structdis_diff_1, CV_64F, win_tmp_1, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
    cv::filter2D(log_struct, structdis_diff_2, CV_64F, win_tmp_2, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);

    std::vector<double> structdis_diff_1_vec(structdis_diff_1.begin<double>(), structdis_diff_1.end<double>());
    std::vector<double> structdis_diff_2_vec(structdis_diff_2.begin<double>(), structdis_diff_2.end<double>());

    auto [gamparam1, sigparam1] = est_GGD_param(structdis_diff_1_vec);
    auto [gamparam2, sigparam2] = est_GGD_param(structdis_diff_2_vec);
    
    ftrs.push_back(gamparam1);
    ftrs.push_back(sigparam1);
    ftrs.push_back(gamparam2);
    ftrs.push_back(sigparam2);

    return ftrs;
}

//Documentation

// rapique_basic_extractor to extract features from an image using various image processing techniques. It employs the OpenCV library for handling images and standard C++ libraries for numeric operations. Line by line:
// 1-7. Include statements:
// •	These lines include necessary headers for the program. They bring in support for vectors, mathematical functions, I/O operations, OpenCV functionalities, OpenCL interface for GPU optimizations, execution policies for parallel algorithms, and future objects for asynchronous operations.
// 8-10. Forward declarations:
// •	Declares three functions that are used later in the code for statistical analysis.
// 11-14. Function circularShift:
// •	A utility function that performs a circular shift of an image (src) by a specified shift (x, y) and stores the result in dst.
// 15-83. Function rapique_basic_extractor:
// •	This function performs feature extraction from an input image img for image quality assessment.
// 16-18. Feature vector initialization:
// •	Initializes a vector ftrs to store features and reserves space for 18 elements.
// 19-25. Gaussian window creation:
// •	Creates a Gaussian window/kernel for local averaging operations. It is used to calculate local mean (mu) and local variance (sigma_sq).
// 26-30. Local mean and standard deviation computation:
// •	Applies a filter with the Gaussian window to the image and its square to compute local mean and variance. The standard deviation (sigma) is then calculated.
// 31-36. Structural distortion calculation:
// •	Computes the structural distortion structdis of the image, normalizing the difference between the image and its local mean by the local standard deviation.
// 37-43. Generalized Gaussian Distribution (GGD) parameter estimation:
// •	Uses the est_GGD_param function to estimate parameters of the GGD model from structdis.
// 44-48. Nakagami distribution fitting:
// •	Uses the nakafit function to estimate the Nakagami distribution parameters from the local standard deviation values.
// 49-58. Pairwise product of shifted structural distortions:
// •	For each shift defined in shifts, it circularly shifts structdis and multiplies it with the original structdis. These pairwise products are stored in pairs.
// 59-72. Asymmetric Generalized Gaussian Distribution (AGGD) parameter estimation:
// •	Estimates AGGD parameters for each element in pairs asynchronously using std::async. The results are stored in futures.
// 73-81. Feature aggregation:
// •	Retrieves the results from futures and appends them to the feature vector ftrs.
// 82-92. Logarithmic structural distortion calculation:
// •	Computes the logarithm of the absolute structural distortion and performs circular shifts for the calculation of additional features.
// 93-106. Combined difference feature extraction:
// •	Computes a combined structural distortion difference and estimates its GGD parameters.
// 107-115. Directional derivative computation:
// •	Calculates directional derivatives of the logarithmic structural distortion using predefined kernels (win_tmp_1 and win_tmp_2) and estimates their GGD parameters.
// 116-123. Feature vector return:
// •	Appends the GGD parameters of directional derivatives to ftrs and returns the complete feature vector.
// This function is a comprehensive implementation of feature extraction techniques commonly used in image quality assessment. It combines various statistical models and image processing operations to extract features that can be indicative of image quality. The use of parallel execution and asynchronous operations helps improve the efficiency of the feature extraction process.






