#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>

std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>> gen_DoG(const cv::Mat& img, int kband) {
    // kval: Base value for sigma calculation in the Gaussian filter.
    // gspace_img: Vector to store the images after applying the Gaussian filter.
    // ksplit_img: Vector to store DoG images.
    double kval = 1.6;
    std::vector<cv::Mat> gspace_img(kband);
    std::vector<cv::Mat> ksplit_img(kband);

    gspace_img[0] = img.clone();

    //  Generation of Gaussian pyramids:

    // The for loop runs through each band from 1 to kband - 1.
    // For each band, it calculates sigma based on kval and the band level.
    // Calculates the filter window size (ws) and ensures that it is odd.
    // Applies cv::GaussianBlur to smooth the original image with the calculated sigma and stores it in gspace_img.

    for (int band = 1; band < kband; ++band) {
        double sigma = std::pow(kval, band - 2);  // Corrected to match MATLAB
        // int ws = static_cast<int>(std::ceil(2 * (3 * sigma + 1)));
        int ws = static_cast<int>(std::ceil(2 * (3 * sigma + 1)));
        if (ws % 2 == 0) {
            ws += 1; // Ensures that ws is odd
        }
        cv::GaussianBlur(img, gspace_img[band], cv::Size(ws, ws), sigma, sigma, cv::BORDER_REPLICATE);
    }

    // DoG calculation:

    // Performs a for loop to calculate the difference between each consecutive Gaussian image level and stores it in ksplit_img.
    // The last element of ksplit_img is simply the last Gaussian image.
    // Calculate DoG (Difference of Gaussian)
    for (int band = 0; band < kband - 1; ++band) {
        ksplit_img[band] = gspace_img[band] - gspace_img[band + 1];
    }
    ksplit_img[kband - 1] = gspace_img[kband - 1];

    // Returns a pair of vectors: one with the Gaussian images and the other with the DoG images.
    return {gspace_img, ksplit_img};
}