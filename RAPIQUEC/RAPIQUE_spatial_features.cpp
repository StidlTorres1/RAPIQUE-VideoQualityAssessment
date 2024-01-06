#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <omp.h>

std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>> gen_DoG(const cv::Mat& img, int kband);
std::pair<cv::Mat, cv::Mat> gauDerivative(double sigma);
std::vector<double> rapique_basic_extractor(const cv::Mat& src);
cv::Mat convertRGBToLAB(const cv::Mat& I);

std::vector<double> RAPIQUE_spatial_features(const cv::Mat& RGB) {
    if (RGB.channels() != 3) {
        throw std::invalid_argument("The input should be an RGB image");
    }

    // Constantes precalculadas para mejorar el rendimiento
    const int kscale = 2;
    const int kband = 4;
    const double sigmaForGauDerivative = 1.66;
    const double scaleFactorForGaussianDer = 0.28;
    const double powValue = 1; 

    cv::Mat Y, GM, LOG, RGBDouble;
    cv::cvtColor(RGB, Y, cv::COLOR_BGR2GRAY);
    Y.convertTo(Y, CV_64F);

    cv::Sobel(Y, GM, CV_64F, 1, 1);
    cv::absdiff(GM, cv::Scalar(0), GM);

    cv::Mat window2 = cv::getGaussianKernel(9, 1.5, CV_64F);
    window2 /= cv::sum(window2)[0];
    cv::filter2D(Y, LOG, CV_64F, window2);
    cv::absdiff(LOG, cv::Scalar(0), LOG);

    auto [_, DOG] = gen_DoG(Y, kband);

    RGB.convertTo(RGBDouble, CV_64F);
    std::vector<cv::Mat> channels(3);
    cv::split(RGBDouble, channels);

    cv::Mat O1 = 0.30 * channels[0] + 0.04 * channels[1] - 0.35 * channels[2];
    cv::Mat O2 = 0.34 * channels[0] - 0.60 * channels[1] + 0.17 * channels[2];

    auto [dx, dy] = gauDerivative(sigmaForGauDerivative);

    // Uso de funciones para reducir la duplicación de código
    auto computeMagnitude = [&dx, &dy](cv::Mat& Ix, cv::Mat& Iy, const cv::Mat& src) -> cv::Mat {
        if (Ix.empty() || Ix.size() != src.size() || Ix.type() != src.type()) {
            Ix.create(src.size(), src.type());
        }
        if (Iy.empty() || Iy.size() != src.size() || Iy.type() != src.type()) {
            Iy.create(src.size(), src.type());
        }

        #pragma omp parallel sections
        {
            #pragma omp section
            {
                cv::filter2D(src, Ix, CV_64F, dx);
            }
            #pragma omp section
            {
                cv::filter2D(src, Iy, CV_64F, dy);
            }
        }

        cv::Mat magnitude;
        cv::magnitude(Ix, Iy, magnitude);
        return magnitude;
    };
    

    cv::Mat Ix, Iy;
    cv::Mat GMO1 = computeMagnitude(Ix, Iy, O1);
    cv::Mat GMO2 = computeMagnitude(Ix, Iy, O2);

    std::vector<cv::Mat> logChannels(3);
    #pragma omp parallel for
    for (int i = 0; i < 3; ++i) {
        cv::log(channels[i] + 0.1, logChannels[i]);
    }

    // Reducción de cálculos redundantes en BY y RG
    cv::Mat BY = (logChannels[0] - cv::mean(logChannels[0])[0] + logChannels[1] - cv::mean(logChannels[1])[0] - 2 * (logChannels[2] - cv::mean(logChannels[2])[0])) / std::sqrt(6);
    cv::Mat RG = (logChannels[0] - cv::mean(logChannels[0])[0] - (logChannels[1] - cv::mean(logChannels[1])[0])) / std::sqrt(2);

    
    cv::Mat GMBY = computeMagnitude(Ix, Iy, BY);
    cv::Mat GMRG = computeMagnitude(Ix, Iy,RG);

    cv::Mat LAB = convertRGBToLAB(RGB);
    LAB.convertTo(LAB, CV_64F);
    std::vector<cv::Mat> LABchannels(3);
    cv::split(LAB, LABchannels);
    
    cv::Mat GMA = computeMagnitude(Ix, Iy,LABchannels[1]);
    cv::Mat GMB = computeMagnitude(Ix, Iy,LABchannels[2]);

    std::vector<cv::Mat> compositeMat = {Y, GM, LOG};
    if (!DOG.empty()) {
        compositeMat.push_back(DOG[0]);
    }
    compositeMat.insert(compositeMat.end(), {O1, O2, GMO1, GMO2, BY, RG, GMBY, GMRG, LABchannels[1], LABchannels[2], GMA, GMB});

    std::vector<double> feats;
    feats.reserve(680);

    // Mejora en la paralelización y manejo de memoria en escalado
    std::vector<cv::Mat> scaledMats;
    scaledMats.reserve(compositeMat.size() * kscale);

    #pragma omp parallel
    {
        std::vector<cv::Mat> localScaledMats;
        localScaledMats.reserve(compositeMat.size() * kscale);

        #pragma omp for nowait
        for (size_t i = 0; i < compositeMat.size(); ++i) {
            const auto& mat = compositeMat[i];
            for (int scale = 1; scale <= kscale; ++scale) {
                if (i >= 4 && scale == 1) continue;
                cv::Mat y_scale;
                cv::resize(mat, y_scale, cv::Size(), std::pow(2, -(scale - 1)), std::pow(2, -(scale - 1)), cv::INTER_CUBIC);
                localScaledMats.push_back(std::move(y_scale));
            }
        }

        #pragma omp critical
        scaledMats.insert(scaledMats.end(), localScaledMats.begin(), localScaledMats.end());
    }

    // Mejora en la extracción de características
    #pragma omp parallel
    {
        std::vector<double> localFeats;
        localFeats.reserve(680);

        #pragma omp for nowait
        for (size_t idx = 0; idx < scaledMats.size(); ++idx) {
            std::vector<double> chFeats = rapique_basic_extractor(scaledMats[idx]);
            localFeats.insert(localFeats.end(), chFeats.begin(), chFeats.end());
        }

        #pragma omp critical
        feats.insert(feats.end(), localFeats.begin(), localFeats.end());
    }

    return feats;
}


//Documentation
// An implementation of a spatial feature extraction process for image quality assessment using the RAPIQUE (Rapid and Accurate Image Quality Evaluator) approach. It involves several steps of image processing using OpenCV, and the code is optimized with OpenMP for parallel computing. Line by line:
// 1-5. Include statements:
// •	#include <opencv2/opencv.hpp>: Includes the main OpenCV library.
// •	#include <opencv2/core/ocl.hpp>: Includes OpenCV's OpenCL interface for GPU optimizations.
// •	#include <opencv2/imgproc/imgproc.hpp>: Includes OpenCV's image processing functionalities.
// •	#include <omp.h>: Includes the OpenMP library for parallel programming.
// 6-9. Forward declarations:
// •	These lines declare four functions that will be used for image processing and feature extraction.
// 10-72. std::vector<double> RAPIQUE_spatial_features(const cv::Mat& RGB) { ... }:
// •	This function performs the RAPIQUE spatial feature extraction on an RGB image.
// 11-13. Input validation and parameter setup:
// •	Checks if the input image is in RGB format.
// •	Defines constants for scale (kscale), number of frequency bands (kband), and parameters related to Gaussian derivatives.
// 14-34. Initial image processing:
// •	Converts the RGB image to grayscale and then to a 64-bit floating-point format.
// •	Applies a Sobel operator to compute the gradient magnitude (GM) and a Gaussian low-pass filter (LOG).
// 35-38. Generation of Difference of Gaussian (DoG) images:
// •	Calls gen_DoG function to generate DoG images from the grayscale image.
// 39-42. Further processing on RGB channels:
// •	Converts RGB to a double-precision format and splits into individual channels.
// •	Creates O1 and O2 feature maps using specific channel combinations.
// 43-55. Gaussian derivative processing:
// •	Applies Gaussian derivative filters (dx, dy) to O1 and O2 feature maps.
// •	Computes gradient magnitude and combined responses for both O1 and O2.
// 56-64. Logarithmic processing of RGB channels:
// •	Applies logarithmic transformation to RGB channels and computes BY and RG feature maps.
// 65-78. LAB color space processing:
// •	Converts RGB to LAB color space and splits it into channels.
// •	Applies Gaussian derivative filters to A and B channels of LAB.
// 79-89. Composition of feature maps:
// •	Combines various feature maps into a single composite matrix for further processing.
// 90-104. Image scaling and feature extraction:
// •	Resizes each feature map to different scales.
// •	Extracts features from each scaled feature map using the rapique_basic_extractor function.
// 105-113. Parallel feature extraction:
// •	Utilizes OpenMP to parallelize the feature extraction process.
// •	Aggregates features from all scaled feature maps.
// 114.	Returns the extracted features.
// The code is designed to extract a comprehensive set of features from an image, which can be used for assessing its quality. The use of Gaussian derivatives, logarithmic transformations, and color space conversions are typical in image processing for capturing various aspects of image quality. The parallelization with OpenMP improves the performance of the feature extraction process, making it suitable for rapid evaluations.

