// Included are the :
// <vector> for using the STL container std::vector.
// <numeric> for numeric operations such as std::accumulate and std::inner_product.
// <cmath> for mathematical functions, specifically std::sqrt.
// <iostream> for input and output operations.
#include <vector>
#include <numeric>
#include <cmath>
#include <iostream>


// This line begins the definition of the nakafit function, which takes a vector of double
// as an argument (passed by constant reference to avoid unnecessary copies and ensure that the data is not modified).
std::vector<double> nakafit(const std::vector<double>& data) {
    // Calculates the average of the elements of the vector data. It uses std::accumulate to sum all the elements of the vector and then
    // divides by the size of the vector to get the average.
    double mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();

    // The sum of squares of the differences of each element with respect to the mean is calculated. This is done with std::inner_product,
    // which here performs an operation similar to the sum of squares in statistics. The lambda [mean](double a, double b) { return (a - mean) * (b - mean); } 
    // defines how the product of the differences of each pair of elements with respect to the mean is calculated.
    double sq_sum = std::inner_product(data.begin(), data.end(), data.begin(), 0.0,
                                       std::plus<double>(),
                                       [mean](double a, double b) { return (a - mean) * (b - mean); });

    // Calculates the standard deviation of the elements in data. First, we check if the vector size is greater than 1 to avoid division by zero. Then, std::sqrt
    //  is used to obtain the square root of the sum of squares divided by N-1 (where N is the vector size), which is the formula for the standard deviation in a sample.
    double stdev;
    if (data.size() > 1) {
        stdev = std::sqrt(sq_sum / (data.size() - 1));
    } else {
        stdev = 0.0; // Case handling when there is only one element in 'data'.
    }

    // Two parameters are calculated and stored in a new param vector:
    // param[0] is the mean of the data.
    // param[1] is a derived measure of the standard deviation, calculated as the square of the quotient of the mean and the standard deviation.
    // The function returns param, a vector containing these two calculated parameters.
    std::vector<double> param(2);
    param[0] = mean;
    param[1] = std::pow(mean / stdev, 2);

    return param;
}