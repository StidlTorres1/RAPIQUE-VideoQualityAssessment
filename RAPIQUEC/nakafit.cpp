#include <vector>
#include <numeric>
#include <cmath>
#include <iostream>

std::vector<double> nakafit(const std::vector<double>& data) {
    if (data.empty()) {
        return {0.0, 0.0};
    }

    const auto n = data.size();
    const double mean = std::accumulate(data.begin(), data.end(), 0.0) / n;

    const double sq_sum = std::inner_product(data.begin(), data.end(), data.begin(), 0.0,
                                             std::plus<>(),
                                             [mean](double a, double b) { return (a - mean) * (b - mean); });

    const double stdev = std::sqrt(sq_sum / (n - 1));
    const double mean_over_stdev_sq = (mean / stdev) * (mean / stdev);

    return {mean, mean_over_stdev_sq};
}


//Documentation
// 1.	#include <vector>: This line includes the C++ Standard Library's <vector> header, which allows the use of the std::vector container class.
// 2.	#include <numeric>: Includes the <numeric> header, providing access to numerical algorithms like std::accumulate and std::inner_product.
// 3.	#include <cmath>: Includes the <cmath> header for mathematical functions, specifically std::sqrt in this case.
// 4.	#include <iostream>: Includes the <iostream> header for input and output operations, although it's not directly used in this function.
// 5.	std::vector<double> nakafit(const std::vector<double>& data) {: Defines a function nakafit that takes a constant reference to a std::vector<double> (named data) and returns a std::vector<double>.
// 6.	if (data.empty()) { return {0.0, 0.0}; }: This checks if the input vector data is empty. If it is, the function returns a vector with two zeros. This serves as a guard clause to handle the edge case of an empty dataset.
// 7.	const auto n = data.size();: Determines the size of the input data vector and stores it in n. The auto keyword automatically deduces the type of n (which would be std::vector<double>::size_type).
// 8.	const double mean = std::accumulate(data.begin(), data.end(), 0.0) / n;: Calculates the mean (average) of the elements in data. std::accumulate sums up the elements starting with an initial value of 0.0. The sum is then divided by n to get the mean.
// 9.	const double sq_sum = std::inner_product(data.begin(), data.end(), data.begin(), 0.0, std::plus<>(), [mean](double a, double b) { return (a - mean) * (b - mean); });: This line calculates the sum of the squares of the differences from the mean. std::inner_product is used here in a non-standard way: instead of multiplying each pair of elements from two sequences, it computes the sum of (a - mean) * (b - mean) for each element a and b in data. This is effectively calculating the sum of squared deviations from the mean.
// 10.	const double stdev = std::sqrt(sq_sum / (n - 1));: Calculates the standard deviation. The square root of the average of the squared deviations (calculated in the previous step) is taken. The average is found by dividing sq_sum by (n - 1) (using Bessel's correction for sample standard deviation).
// 11.	const double mean_over_stdev_sq = (mean / stdev) * (mean / stdev);: Computes the square of the ratio of the mean to the standard deviation.
// 12.	return {mean, mean_over_stdev_sq};: Returns a vector containing two elements: the mean and the square of the mean over the standard deviation.
// This function is a statistical utility that takes a dataset and calculates two key metrics: the mean of the data and the square of the ratio of the mean to the standard deviation. 
// This is useful in statistical analysis or data normalization processes.
