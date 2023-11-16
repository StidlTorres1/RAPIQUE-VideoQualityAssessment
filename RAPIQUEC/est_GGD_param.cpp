// Included are the standard libraries needed to handle vectors (<vector>), perform mathematical calculations (<cmath>), numerical operations (<numeric>), and algorithms (<algorithm>), as well as for standard input and output (<iostream >).
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iostream>

// Starts defining the function est_GGD_param, which takes a vector of doubles as an argument and returns a pair of doubles
std::pair<double, double> est_GGD_param(const std::vector<double>& vec) {
    // It checks whether the vector vec is empty. If it is, a pair of zeros is returned, thus avoiding calculation errors in the case of a vector without elements.
    if (vec.empty()) {
        return {0.0, 0.0};
    }

    // A gam vector is created with 5901 elements, filling it with values ​​from 0.1 to 6 in increments of 0.001
    std::vector<double> gam(5901);
    std::iota(gam.begin(), gam.end(), 100); 
    // gam is transformed into r_gam, applying a function that calculates a value based on the gamma function. This transformation is part of the GGD parameter estimation process.
    std::transform(gam.begin(), gam.end(), gam.begin(), [](double x) { return x / 1000.0; });

    std::vector<double> r_gam(gam.size());
    std::transform(gam.begin(), gam.end(), r_gam.begin(), [](double x) {
        return ::tgamma(1.0 / x) * ::tgamma(3.0 / x) / std::pow(::tgamma(2.0 / x), 2);
    });

    // Sigma_sq is calculated, which is the variance of the vec elements. std::accumulate is used to sum the squares of the elements of the vector and then divide by the size of the vector.
    double sigma_sq = std::accumulate(vec.begin(), vec.end(), 0.0,
                                      [](double acc, double x) { return acc + x * x; }) / vec.size();

    // Alpha_par is defined as the square root of sigma_sq. This will be one of the returned parameters.
    double alpha_par = std::sqrt(sigma_sq);
    // E is calculated, which is the average of the absolute values ​​of the vec elements.
    double E = std::accumulate(vec.begin(), vec.end(), 0.0,
                               [](double sum, double x) { return sum + std::abs(x); }) / vec.size();
    // Rho, a measure related to the dispersion of the data, is calculated.
    double rho = sigma_sq / std::pow(E, 2);

    // The value in r_gam closest to rho is found and the position of this value in the vector is determined. This is used to find the beta_par parameter.

    auto array_position = std::distance(r_gam.begin(), std::min_element(r_gam.begin(), r_gam.end(),
                                                                        [rho](double a, double b) {
                                                                            return std::abs(rho - a) < std::abs(rho - b);
                                                                        }));
    double beta_par = gam[array_position];

    //The function returns a pair containing beta_par and alpha_par.
    return {beta_par, alpha_par};
}