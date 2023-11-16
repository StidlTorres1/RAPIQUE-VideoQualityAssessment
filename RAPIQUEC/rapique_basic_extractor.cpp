// Necessary libraries are included: <vector> to use the std::vector data type, <numeric> and <cmath>
// for mathematical operations, <iostream> for standard input and output, and <opencv2/opencv.hpp> and
// <opencv2/core/ocl.hpp> for OpenCV functions.
#include <vector>
#include <numeric>
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>

// Prototypes of functions are declared that will be used later to fit statistical models and perform operations with vectors and matrices.
// operations with vectors and matrices.
// nakafit calculates the mean and a related measure of the standard deviation of a data set.
std::vector<double> nakafit(const std::vector<double>& data);
std::pair<double, double> est_GGD_param(const std::vector<double>& vec);
// estimate parameters of an AGGD distribution
std::tuple<double, double, double> est_AGGD_param(const std::vector<double>& vec);
// Definition of the circularShift function, which performs a circular shift of an src array and stores it in dst.
// Uses the cv::Point class to define the shift in x and y.
void circularShift(const cv::Mat& src, cv::Mat& dst, cv::Point shift);
// circularShift that performs a circular shift of an image using OpenCV. The function receives an input image and applies a specified shift to this image, so that pixels that move off the edges of the image reappear on the opposite side.
// Function parameters:

// src: Input image (source), of type cv::Mat.
// dst: Output image (destination), of type cv::Mat& (reference).
// shift: Point indicating the displacement in x and y, of type cv::Point.
void circularShift(const cv::Mat& src, cv::Mat& dst, cv::Point shift) {
    // shift_x and shift_y are calculated as the modulus of the shift with respect to the number of columns and rows in the image, respectively. This ensures that the shift is within the image boundaries.
    // If shift_x or shift_y are negative (shift to the left or up), they are adjusted to ensure a positive shift.
    int shift_x = shift.x % src.cols;
    int shift_y = shift.y % src.rows;
    if (shift_x < 0) {
        shift_x += src.cols;
    }
    if (shift_y < 0) {
        shift_y += src.rows;
    }

    // cv::copyMakeBorder is used to create an extended image from the source image (src).
    // The function adds borders to the image that are repeated according to the specified offset (shift_x and shift_y). This step is crucial to achieve the circular shift effect.
    cv::Mat extended;
    cv::copyMakeBorder(src, extended, 0, shift_y, 0, shift_x, cv::BORDER_WRAP);
    
    // Definition of the area of interest (ROI):

    // A rectangle (cv::Rect) representing the area of interest (ROI) in the extended image is defined.
    // The ROI has the size of the original image (src.cols, src.rows) and starts at the point (shift_x, shift_y).
    cv::Rect roi(shift_x, shift_y, src.cols, src.rows);

    // Extraction of the ROI in the target image:

    // The target image (dst) is set as the region specified by the ROI in the extended image.
    // This effectively cuts off the relevant part of the extended image, resulting in the circularly shifted image.

    dst = extended(roi);
}


std::vector<double> rapique_basic_extractor(const cv::Mat& img) {
// This line initiates the definition of the rapique_basic_extractor function, which takes an image (cv::Mat) as argument
// and returns a vector (std::vector<double>) of extracted features.

// Here we declare a vector of type double called ftrs, which will be used to store the features extracted from the image.
    std::vector<double> ftrs;

// The filter length (filtlength) is defined and a Gaussian window (window) is created. This window is used for smoothing the image and
// is normalized so that the sum of its elements is 1
    int filtlength = 7;
    cv::Mat window = cv::getGaussianKernel(filtlength, filtlength / 6, CV_64F) * 
                     cv::getGaussianKernel(filtlength, filtlength / 6, CV_64F).t();
    window /= cv::sum(window)[0];

// The local mean (mu) and the local variance (sigma_sq) of the image are calculated. The function cv::filter2D applies the Gaussian window to the image.
// The variance is calculated from the mean and it is ensured that there are no negative values using cv::max.
    cv::Mat mu;
    cv::filter2D(img, mu, CV_64F, window, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
    cv::Mat mu_sq = mu.mul(mu);

    
    cv::Mat sigma_sq;
    cv::filter2D(img.mul(img), sigma_sq, CV_64F, window, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
    sigma_sq -= mu_sq;
    cv::max(sigma_sq, 0, sigma_sq);

// The local standard deviation (sigma) and the structural distortion matrix (structdis) are calculated. The structdis matrix is a normalized representation
// normalized representation of the differences between the original image and its local mean.

    cv::Mat sigma;
    cv::sqrt(sigma_sq, sigma);
    cv::Mat structdis = (img - mu) / (sigma + 1);

//The structdis matrix is converted to a vector (vec_struct) for further processing. This is done to facilitate statistical calculations.
    std::vector<double> vec_struct;
    if (structdis.isContinuous()) {
        vec_struct.assign((double*)structdis.datastart, (double*)structdis.dataend);
    } else {
        cv::Mat structdis_cont = structdis.clone();
        vec_struct.assign((double*)structdis_cont.datastart, (double*)structdis_cont.dataend);
    }

// The parameters of the Generalized Gaussian distribution (GGD) of structdis estimated and added to the vector ftrs.
    auto [gamparam, sigparam] = est_GGD_param(vec_struct);
    ftrs.push_back(gamparam);
    ftrs.push_back(sigparam);

// The sigma matrix is converted to a vector and the Nakagami adjustment (nakafit) is calculated for this vector. The results are added to ftrs.
    std::vector<double> sigmaVec(sigma.begin<double>(), sigma.end<double>());
    std::vector<double> sigmaParam = nakafit(sigmaVec);
    ftrs.insert(ftrs.end(), sigmaParam.begin(), sigmaParam.end());

// A set of circular displacements is performed on the structdis matrix. For each displacement, we calculate the parameters of the
// Generalized Gaussian Asymmetric Distribution (AGGD) are calculated and added to the vector ftrs.
    std::vector<std::pair<int, int>> shifts = {{0, 1}, {1, 0}, {1, 1}, {-1, 1}};
    for (const auto& shift : shifts) {
        cv::Mat shifted_structdis;
        // Convert shift torque to cv::Point and pass it to circularShift
        circularShift(structdis, shifted_structdis, cv::Point(shift.first, shift.second));

        cv::Mat pair = structdis.mul(shifted_structdis);

        // Convert the pair matrix to a vector for processing
        std::vector<double> pairVec(pair.begin<double>(), pair.end<double>());

        // Estimate and add pair AGGD parameters to the feature vector
        auto [alpha, leftstd, rightstd] = est_AGGD_param(pairVec);
        double meanparam = (rightstd - leftstd) * (tgamma(2.0 / alpha) / tgamma(1.0 / alpha)) *
                           (std::sqrt(tgamma(1.0 / alpha)) / std::sqrt(tgamma(3.0 / alpha)));
        ftrs.push_back(alpha);
        ftrs.push_back(meanparam);
        ftrs.push_back(leftstd);
        ftrs.push_back(rightstd);
    }

// The logarithm of the structural distortion matrix is calculated and the same circular displacement process is performed, and
// calculation of GGD parameters for structural differences.
    cv::Mat log_struct;
    cv::log(cv::abs(structdis) + 0.1, log_struct);

    // Perform circular displacements and calculate structural differences
    for (const auto& shift : shifts) {
        cv::Mat shifted_structdis;
        // Convert shift torque to cv::Point and pass it to circularShift
        circularShift(log_struct, shifted_structdis, cv::Point(shift.first, shift.second));
        cv::Mat structdis_diff = log_struct - shifted_structdis;

        // Convert the matrix of structural differences to a vector
        std::vector<double> structdis_diff_vec(structdis_diff.begin<double>(), structdis_diff.end<double>());

        // Estimate and add the GGD parameters of the structural differences to the vector of characteristics.
        auto [gamparam_diff, sigparam_diff] = est_GGD_param(structdis_diff_vec);
        ftrs.push_back(gamparam_diff);
        ftrs.push_back(sigparam_diff);
    }

    // Edge filters are applied to the log_struct matrix and structural differences are calculated.
    // The GGD parameters of these differences are added to ftrs.
    cv::Mat win_tmp_1 = (cv::Mat_<double>(3,3) << 0, 1, 0, -1, 0, -1, 0, 1, 0);
    cv::Mat win_tmp_2 = (cv::Mat_<double>(3,3) << 1, 0, -1, 0, 0, 0, -1, 0, 1);
    
    cv::Mat structdis_diff_1, structdis_diff_2;
    cv::filter2D(log_struct, structdis_diff_1, CV_64F, win_tmp_1, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
    cv::filter2D(log_struct, structdis_diff_2, CV_64F, win_tmp_2, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);

    // Convert filtering results to vectors
    std::vector<double> structdis_diff_1_vec(structdis_diff_1.begin<double>(), structdis_diff_1.end<double>());
    std::vector<double> structdis_diff_2_vec(structdis_diff_2.begin<double>(), structdis_diff_2.end<double>());

    // Estimate and add the GGD parameters of the filtered structural differences to the vector of characteristics.
    auto [gamparam1, sigparam1] = est_GGD_param(structdis_diff_1_vec);
    auto [gamparam2, sigparam2] = est_GGD_param(structdis_diff_2_vec);
    
    ftrs.push_back(gamparam1);
    ftrs.push_back(sigparam1);
    ftrs.push_back(gamparam2);
    ftrs.push_back(sigparam2);

    // The function ends by returning the vector ftrs containing all the extracted features.
    return ftrs;
}

// int main() {
//     // Example of how to call the rapique_basic_extractor function
//     cv::Mat image; // Assuming we have an OpenCV image loaded here.
//     std::vector<double> features = rapique_basic_extractor(image);

//     //Print features.
//     for (const auto& feature : features) {
//         std::cout << feature << std::endl;
//     }

//     return 0; // Indicates that the program was successfully completed.
// }