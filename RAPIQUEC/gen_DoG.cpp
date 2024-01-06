#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <execution>

std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>> gen_DoG(const cv::Mat& img, int kband) {
    constexpr double kval = 1.6;
    std::vector<cv::Mat> gspace_img(kband);
    std::vector<cv::Mat> ksplit_img(kband);
    std::vector<double> sigmas(kband);
    std::vector<int> wsizes(kband);

    gspace_img[0] = img;

    // Pre-calculate sigmas and window sizes
    for (int band = 1; band < kband; ++band) {
        sigmas[band] = std::pow(kval, band - 2);
        int ws = static_cast<int>(std::ceil(2 * (3 * sigmas[band] + 1)));
        wsizes[band] = ws + (ws % 2 == 0 ? 1 : 0);
    }

    // Parallel Gaussian Blur
    std::transform(std::execution::par, sigmas.begin() + 1, sigmas.end(), wsizes.begin() + 1, gspace_img.begin() + 1,
        [&img](double sigma, int ws) {
            cv::Mat result;
            cv::GaussianBlur(img, result, cv::Size(ws, ws), sigma, sigma, cv::BORDER_REPLICATE);
            return result;
        }
    );

    // Parallel Subtraction
    std::transform(std::execution::par, gspace_img.begin(), gspace_img.end() - 1, gspace_img.begin() + 1, ksplit_img.begin(),
        [](const cv::Mat &a, const cv::Mat &b) {
            cv::Mat result;
            cv::subtract(a, b, result);
            return result;
        }
    );

    ksplit_img[kband - 1] = gspace_img[kband - 1].clone();

    return {gspace_img, ksplit_img};
}


//Documentation
// A function for generating Difference of Gaussians (DoG) images, a process often used in image processing, particularly in scale-space representation and feature detection (like SIFT). Let's break down this code line by line:
// 1.	#include <vector>: This line includes the C++ Standard Library's <vector> header, which allows the use of the std::vector container class.
// 2.	#include <cmath>: Includes the C++ Standard Library's <cmath> header for mathematical functions, such as std::pow and std::ceil.
// 3.	#include <opencv2/opencv.hpp>: Includes the OpenCV library, a popular computer vision library used for image processing.
// 4.	#include <opencv2/core/ocl.hpp>: Includes OpenCV's OpenCL-based operations, which can be used for hardware-accelerated computer vision functions.
// 5.	std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>> gen_DoG(const cv::Mat& img, int kband) {: Defines a function gen_DoG that returns a std::pair of two vectors of cv::Mat (OpenCV matrix objects). The function takes a constant reference to a cv::Mat (named img, representing an image) and an integer kband.
// 6.	constexpr double kval = 1.6;: Declares a constant kval used in the Gaussian blur calculations. This value is a typical choice in scale-space theory.
// 7.	std::vector<cv::Mat> gspace_img(kband);: Declares a vector of cv::Mat objects (gspace_img) with size kband, to store Gaussian-blurred images.
// 8.	std::vector<cv::Mat> ksplit_img(kband);: Declares another vector of cv::Mat objects (ksplit_img) with size kband, to store the DoG images.
// 9.	gspace_img[0] = img;: Initializes the first element of gspace_img with the input image.
// 10.	for (int band = 1; band < kband; ++band) {: Starts a loop to calculate Gaussian-blurred images at different scales.
// 11.	double sigma = std::pow(kval, band - 2);: Calculates the standard deviation (sigma) for the Gaussian blur, increasing exponentially with each scale (band).
// 12.	int ws = static_cast<int>(std::ceil(2 * (3 * sigma + 1)));: Determines the window size (ws) for the Gaussian blur, proportional to sigma.
// 13.	ws += ws % 2 == 0 ? 1 : 0;: Ensures the window size is odd (Gaussian blur typically requires an odd-sized kernel).
// 14.	cv::GaussianBlur(img, gspace_img[band], cv::Size(ws, ws), sigma, sigma, cv::BORDER_REPLICATE);: Applies Gaussian blur to the input image with the computed sigma and window size, storing the result in gspace_img.
// 15.	for (int band = 0; band < kband - 1; ++band) {: Starts another loop to compute the DoG images.
// 16.	cv::subtract(gspace_img[band], gspace_img[band + 1], ksplit_img[band]);: Subtracts each blurred image from the next finer scale blurred image, storing the result in ksplit_img.
// 17.	ksplit_img[kband - 1] = gspace_img[kband - 1].clone();: Sets the last element of ksplit_img to a clone of the last Gaussian-blurred image.
// 18.	return {gspace_img, ksplit_img};: Returns the pair of vectors: one containing the Gaussian-blurred images and the other containing the DoG images.
// The function gen_DoG essentially performs a series of Gaussian blurs on an input image at increasing scales and then computes the difference between successive scales to produce DoG images. These DoG images are useful for detecting features across scales and are a key component in scale-invariant feature detection algorithms.

