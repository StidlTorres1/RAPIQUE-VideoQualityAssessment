// Included are the necessary libraries for handling vectors (<vector>), mathematical operations (<cmath>), algorithms (<algorithm>), numerical operations (<numeric>) and standard input/output (<iostream>).
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>

// Defines the function mean, which calculates the mean of a vector v of type double. Use std::accumulate to add the elements of the vector and divide the total by the size of the vector.
double mean(const std::vector<double>& v) {
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

// Defines calculateStdDev, which calculates the standard deviations of the negative and positive values ​​(considered in absolute value) in a vec vector. Separates vec into two vectors (left and right) 
// based on whether their elements are less or greater than zero. It then returns a pair of values ​​that are the square roots of the means of these vectors.
std::pair<double, double> calculateStdDev(const std::vector<double>& vec) {
    std::vector<double> left, right;
    for (double val : vec) {
        if (val < 0) left.push_back(std::abs(val)); // Use the absolute value
        else if (val > 0) right.push_back(std::abs(val)); // Use the absolute value
    }
    return {std::sqrt(mean(left)), std::sqrt(mean(right))};
}

// Defines gammaFunction, which simply calculates the gamma function for a given x value, using std::tgamma.
double gammaFunction(double x) {
    return std::tgamma(x);
}

// Defines the function est_AGGD_param, which estimates the AGGD parameters for a vec vector.
std::tuple<double, double, double> est_AGGD_param(const std::vector<double>& vec) {
    std::vector<double> gam(5901);
    std::iota(gam.begin(), gam.end(), 100);  //Creates a gam vector containing values ​​from 0.1 to 6 in increments of 0.001.
    std::transform(gam.begin(), gam.end(), gam.begin(), [](double x) { return x / 1000.0; });

    std::vector<double> r_gam(gam.size());
    // Transform gam into r_gam by applying a function that uses the gamma function to calculate a value based on the AGGD formula.
    std::transform(gam.begin(), gam.end(), r_gam.begin(), [](double x) {
        return std::pow(gammaFunction(2.0 / x), 2) / (gammaFunction(1.0 / x) * gammaFunction(3.0 / x));
    });

    // Calculates leftstd and rightstd, the standard deviations of the negative and positive values ​​of vec.
    auto [leftstd, rightstd] = calculateStdDev(vec);
    double gammahat = leftstd / rightstd;
    double rhat = std::pow(std::accumulate(vec.begin(), vec.end(), 0.0, [](double a, double b) {
        return a + std::abs(b);  // Use the absolute value
    }) / vec.size(), 2) / mean(vec);
    // Calculates gammahat, an estimate of the ratio of standard deviations, and rhat, a value based on the mean of the absolute values ​​of vec.
    double rhatnorm = (rhat * (std::pow(gammahat, 3) + 1) * (gammahat + 1)) / std::pow((std::pow(gammahat, 2) + 1), 2);

    // Normalize rhat for use in estimating AGGD parameters.
    // Find the position in r_gam that is closest to rhatnorm and use that position to determine the value of alpha. Finally, it returns alpha, leftstd and rightstd as a trio of values
    auto array_position = std::distance(r_gam.begin(), std::min_element(r_gam.begin(), r_gam.end(), [rhatnorm](double a, double b) {
        return std::abs(a - rhatnorm) < std::abs(b - rhatnorm); // Compare absolute differences
    }));

    double alpha = gam[array_position];

    return {alpha, leftstd, rightstd};
}