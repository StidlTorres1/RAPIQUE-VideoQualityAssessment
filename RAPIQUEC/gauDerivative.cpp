#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>

std::pair<cv::Mat, cv::Mat> gauDerivative(double sigma) {
    // halfLength: Half the length of the Gaussian kernel, calculated from sigma.
    // size: Total size of the kernel, which is odd.
    int halfLength = static_cast<int>(std::ceil(3 * sigma));
    int size = 2 * halfLength + 1;
    // gauDerX and gauDerY: Arrays to store the Gaussian derivatives in X and Y. They are initialized with the calculated size and data type CV_64F (double precision float).
    cv::Mat gauDerX(size, size, CV_64F);
    cv::Mat gauDerY(size, size, CV_64F);

    // Calculation of Gaussian derivatives:

    // The code iterates over each point in the filter window (from -halfLength to +halfLength for both X and Y).
    // It calculates the values of the Gaussian derivatives using the mathematical formula for the derivative of a Gaussian, taking into account the distance from the center of the window and the sigma value.
    // commonFactor: Common factor in the derivatives, based on the formula for the Gaussian function.
    // gauDerX and gauDerY are updated at each iteration with these values.
    for (int i = -halfLength; i <= halfLength; ++i) {
        for (int j = -halfLength; j <= halfLength; ++j) {
            double x = static_cast<double>(i);
            double y = static_cast<double>(j);
            double commonFactor = std::exp(-(x * x + y * y) / (2 * sigma * sigma));
            gauDerX.at<double>(i + halfLength, j + halfLength) = x * commonFactor;
            gauDerY.at<double>(i + halfLength, j + halfLength) = y * commonFactor;
        }
    }
    // Returns a pair containing the matrices gauDerX and gauDerY, representing the Gaussian derivatives in the X and Y directions, respectively.
    return {gauDerX, gauDerY};
}
