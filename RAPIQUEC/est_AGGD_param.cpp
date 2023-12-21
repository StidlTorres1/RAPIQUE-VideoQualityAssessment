#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>

double mean(const std::vector<double>& v) {
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

std::pair<double, double> calculateStdDev(const std::vector<double>& vec) {
    double sumLeft = 0.0, sumRight = 0.0;
    int countLeft = 0, countRight = 0;

    for (double val : vec) {
        if (val < 0) {
            sumLeft += std::abs(val);
            ++countLeft;
        } else if (val > 0) {
            sumRight += std::abs(val);
            ++countRight;
        }
    }

    double leftMean = countLeft == 0 ? 0.0 : std::sqrt(sumLeft / countLeft);
    double rightMean = countRight == 0 ? 0.0 : std::sqrt(sumRight / countRight);

    return {leftMean, rightMean};
}

double gammaFunction(double x) {
    return std::tgamma(x);
}

std::tuple<double, double, double> est_AGGD_param(const std::vector<double>& vec) {
    std::vector<double> gam(5901);
    std::iota(gam.begin(), gam.end(), 100);  
    std::transform(gam.begin(), gam.end(), gam.begin(), [](double x) { return x / 1000.0; });

    double sumAbs = 0.0;
    for (double val : vec) {
        sumAbs += std::abs(val);
    }
    double meanAbs = sumAbs / vec.size();

    auto [leftstd, rightstd] = calculateStdDev(vec);
    double gammahat = leftstd / rightstd;
    double rhat = std::pow(meanAbs, 2) / mean(vec);
    double rhatnorm = (rhat * (std::pow(gammahat, 3) + 1) * (gammahat + 1)) / std::pow((std::pow(gammahat, 2) + 1), 2);

    double minDiff = std::numeric_limits<double>::max();
    double alpha = 0.0;
    for (double x : gam) {
        double r_gam_val = std::pow(gammaFunction(2.0 / x), 2) / (gammaFunction(1.0 / x) * gammaFunction(3.0 / x));
        double diff = std::pow(r_gam_val - rhatnorm, 2);
        if (diff < minDiff) {
            minDiff = diff;
            alpha = x;
        }
    }

    return {alpha, leftstd, rightstd};
}


//Documentation
// Estimating the parameters of an Asymmetric Generalized Gaussian Distribution (AGGD). Let's break it down line by line:
// 1-5. Include statements:
// •	#include <vector>: Includes support for the std::vector container.
// •	#include <cmath>: Includes standard mathematical functions.
// •	#include <algorithm>: Provides various algorithms like std::transform.
// •	#include <numeric>: Contains numeric processing functions like std::accumulate.
// •	#include <iostream>: Includes standard I/O stream objects.
// 6-8. double mean(const std::vector<double>& v) { ... }:
// •	Defines a function mean that calculates and returns the mean (average) of elements in a vector v using std::accumulate.
// 9-19. std::pair<double, double> calculateStdDev(const std::vector<double>& vec) { ... }:
// •	This function computes two values: the standard deviation of negative values (leftMean) and positive values (rightMean) in the input vector vec.
// •	It separates the sum and count of absolute values for negative and positive elements.
// •	It calculates the mean of the absolute values for negative and positive numbers separately, applying a square root.
// 20-22. double gammaFunction(double x) { ... }:
// •	A simple wrapper function for the std::tgamma function, which computes the gamma function of x.
// 23-43. std::tuple<double, double, double> est_AGGD_param(const std::vector<double>& vec) { ... }:
// •	This function estimates parameters for an Asymmetric Generalized Gaussian Distribution (AGGD) given a vector of data.
// •	It initializes a vector gam with 5901 elements, filling it with values from 100 to 6000, scaled down by 1000.
// •	It calculates the mean absolute value (meanAbs) of the elements in vec.
// •	The function calculateStdDev is used to get the standard deviation of negative (leftstd) and positive (rightstd) values separately.
// •	gammahat is the ratio of these two standard deviations.
// •	rhat and rhatnorm are intermediate values used for estimating AGGD parameters.
// •	It iterates through the gam vector to find the value of alpha that minimizes the difference between a calculated ratio (r_gam_val) and rhatnorm.
// •	Finally, it returns a tuple with the estimated parameters: alpha, leftstd, and rightstd.
// implementation for estimating AGGD parameters, which are useful in various fields like signal processing and image analysis. AGGD is particularly suited for modeling asymmetric data distributions. The functions are designed to separately handle positive and negative values, which is a key aspect of the asymmetry in AGGD.
