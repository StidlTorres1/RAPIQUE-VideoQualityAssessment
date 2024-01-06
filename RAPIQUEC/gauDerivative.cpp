#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <future>

std::pair<cv::Mat, cv::Mat> gauDerivative(double sigma) {
    const int halfLength = static_cast<int>(std::ceil(3 * sigma));
    const int size = 2 * halfLength + 1;
    const double sigmaSquared = 2 * sigma * sigma;
    const double inverseSigmaSquared = 1 / sigmaSquared;

    cv::Mat gauDerX(size, size, CV_64F);
    cv::Mat gauDerY(size, size, CV_64F);

    auto compute = [&](bool isX) -> void {
        for (int i = -halfLength; i <= halfLength; ++i) {
            const double iSquared = i * i;
            for (int j = -halfLength; j <= halfLength; ++j) {
                const double jSquared = j * j;
                const double commonFactor = std::exp(-(iSquared + jSquared) * inverseSigmaSquared);

                if (isX) {
                    gauDerX.at<double>(i + halfLength, j + halfLength) = i * commonFactor;
                } else {
                    gauDerY.at<double>(i + halfLength, j + halfLength) = j * commonFactor;
                }
            }
        }
    };

    std::future<void> futureX = std::async(std::launch::async, compute, true);
    std::future<void> futureY = std::async(std::launch::async, compute, false);

    futureX.wait();
    futureY.wait();

    return {gauDerX, gauDerY};
}




//Documentation

// Is a function for generating Gaussian derivative filters in both the X and Y directions. These filters are used in image processing for edge detection and feature extraction. Let's go through the code line by line:
// 1.	#include <vector>: Includes the C++ Standard Library's <vector> header for using the std::vector container.
// 2.	#include <cmath>: Includes the C++ Standard Library's <cmath> header for mathematical functions such as std::ceil and std::exp.
// 3.	#include <opencv2/opencv.hpp>: Includes the OpenCV library, a widely used library for computer vision and image processing.
// 4.	#include <opencv2/core/ocl.hpp>: Includes OpenCV's OpenCL-based operations for potential hardware acceleration.
// 5.	std::pair<cv::Mat, cv::Mat> gauDerivative(double sigma) {: Declares a function gauDerivative that takes a double representing the standard deviation of the Gaussian (sigma) and returns a pair of cv::Mat objects, which are matrices in OpenCV. These matrices will represent the Gaussian derivatives in the X and Y directions.
// 6.	const int halfLength = static_cast<int>(std::ceil(3 * sigma));: Calculates halfLength, which is half the size of the Gaussian filter. The size of the filter is typically chosen to be about 3 times the standard deviation (sigma) on either side of the center.
// 7.	const int size = 2 * halfLength + 1;: Determines the full size of the filter. Since the filter is symmetric and centered, the total size is 2 * halfLength + 1.
// 8.	const double sigmaSquared = 2 * sigma * sigma;: Calculates sigmaSquared, which is twice the square of sigma. This is used in the Gaussian function.
// 9.	const double inverseSigmaSquared = 1 / sigmaSquared;: Computes the inverse of sigmaSquared, which will be used in the exponential part of the Gaussian function.
// 10.	cv::Mat gauDerX(size, size, CV_64F);: Initializes a cv::Mat object gauDerX to store the Gaussian derivative in the X direction. CV_64F indicates that each element is a double-precision floating-point number.
// 11.	cv::Mat gauDerY(size, size, CV_64F);: Similarly, initializes a cv::Mat object gauDerY for the Y direction.
// 12.	The nested for loops iterate over the filter dimensions. The outer loop runs over rows (i), and the inner loop runs over columns (j), both ranging from -halfLength to halfLength.
// 13.	const double iSquared = i * i; and const double jSquared = j * j;: These lines compute the square of the current row and column indices, which are used in the Gaussian formula.
// 14.	const double commonFactor = std::exp(-(iSquared + jSquared) * inverseSigmaSquared);: Calculates the common exponential factor of the Gaussian function for the current cell.
// 15.	gauDerX.at<double>(i + halfLength, j + halfLength) = i * commonFactor;: For the X derivative, the value is proportional to i (the row index) multiplied by the common Gaussian factor. This line populates the gauDerX matrix.
// 16.	gauDerY.at<double>(i + halfLength, j + halfLength) = j * commonFactor;: For the Y derivative, it's proportional to j (the column index). This populates the gauDerY matrix.
// 17.	return {gauDerX, gauDerY};: Returns a pair of matrices, gauDerX and gauDerY, which are the Gaussian derivatives in the X and Y directions, respectively.
// The function gauDerivative generates matrices that can be used to convolve with an image to detect edges and gradients, which are critical operations in many computer vision and image processing applications. The use of Gaussian derivatives smooths the image while computing gradients, reducing the impact of noise.
