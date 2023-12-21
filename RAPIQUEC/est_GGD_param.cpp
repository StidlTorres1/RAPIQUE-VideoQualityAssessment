#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iostream>

std::pair<double, double> est_GGD_param(const std::vector<double>& vec) {
    if (vec.empty()) {
        return {0.0, 0.0};
    }

    const size_t gamSize = 5901;
    std::vector<double> gam(gamSize);
    std::vector<double> r_gam(gamSize);

    double sumAbsX = 0.0, sumX2 = 0.0;
    for (double x : vec) {
        sumAbsX += std::abs(x);
        sumX2 += x * x;
    }

    double sigma_sq = sumX2 / vec.size();
    double E = sumAbsX / vec.size();
    double rho = sigma_sq / (E * E);

    for (size_t i = 0; i < gamSize; ++i) {
        double gamVal = (i + 100) / 1000.0;
        gam[i] = gamVal;
        double tgamma1 = ::tgamma(1.0 / gamVal);
        double tgamma2 = ::tgamma(2.0 / gamVal);
        double tgamma3 = ::tgamma(3.0 / gamVal);
        r_gam[i] = tgamma1 * tgamma3 / (tgamma2 * tgamma2);
    }

    auto array_position = std::distance(r_gam.begin(), std::min_element(r_gam.begin(), r_gam.end(),
                                                                        [rho](double a, double b) {
                                                                            return std::abs(rho - a) < std::abs(rho - b);
                                                                        }));
    double beta_par = gam[array_position];
    double alpha_par = std::sqrt(sigma_sq);

    return {beta_par, alpha_par};
}

//Documentation

// A function for estimating the parameters of a Generalized Gaussian Distribution (GGD) given a vector of data. The function calculates two parameters of the GGD: beta_par (shape parameter) and alpha_par (scale parameter). Line by line:
// 1.	#include <vector>: This line includes the C++ Standard Library's <vector> header for using the std::vector container.
// 2.	#include <cmath>: Includes the C++ Standard Library's <cmath> header for mathematical functions such as std::abs, std::sqrt, and ::tgamma (gamma function).
// 3.	#include <numeric>: Includes the <numeric> header, which provides functions for numeric operations like std::accumulate.
// 4.	#include <algorithm>: Includes the C++ Standard Library's <algorithm> header for algorithms like std::min_element.
// 5.	#include <iostream>: Includes the <iostream> header for input and output operations, although it's not directly used in this function.
// 6.	std::pair<double, double> est_GGD_param(const std::vector<double>& vec) {: Defines a function est_GGD_param that takes a constant reference to a std::vector<double> (named vec) and returns a std::pair<double, double>.
// 7.	if (vec.empty()) { return {0.0, 0.0}; }: Checks if the input vector is empty. If so, it returns a pair of zeros, which is a sensible default when no data is available.
// 8.	const size_t gamSize = 5901;: Defines a constant gamSize for the size of two vectors that will be used later. The size is set to 5901.
// 9.	std::vector<double> gam(gamSize); std::vector<double> r_gam(gamSize);: Declares two vectors, gam and r_gam, each of size gamSize. These will store computed values for gamma function-related calculations.
// 10.	double sumAbsX = 0.0, sumX2 = 0.0;: Initializes two variables for summing the absolute values and squares of the elements in vec.
// 11.	The for loop iterates through each element x in vec, summing the absolute value of x to sumAbsX and the square of x to sumX2.
// 12.	double sigma_sq = sumX2 / vec.size();: Calculates the squared scale parameter (sigma_sq) as the average of the squares of the elements.
// 13.	double E = sumAbsX / vec.size();: Calculates the expected value (E) as the average of the absolute values of the elements.
// 14.	double rho = sigma_sq / (E * E);: Computes rho, a ratio used in the estimation of the GGD parameters.
// 15.	The for loop computes values related to the gamma function for different gamVal values, stores them in gam and calculates r_gam as a function of the gamma values.
// 16.	The std::min_element function along with std::distance is used to find the position in the r_gam array that is closest to rho.
// 17.	double beta_par = gam[array_position];: Determines the beta_par (shape parameter) of the GGD.
// 18.	double alpha_par = std::sqrt(sigma_sq);: Calculates the alpha_par (scale parameter) of the GGD.
// 19.	return {beta_par, alpha_par};: Returns a pair of values, beta_par and alpha_par, which are the estimated parameters of the GGD.
// This function is useful in statistical analysis and signal processing, particularly where modeling data distributions as Generalized Gaussian is appropriate. The function iteratively calculates and selects the best fitting GGD parameters based on the input data.
