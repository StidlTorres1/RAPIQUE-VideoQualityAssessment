#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>


cv::Mat YUVread(const std::string &filename, int width, int height, int frameNum) {
    // frameSize: Size of the Y component (luminance).
    // uvFrameSize: Size of the U and V (chrominance) components, which is a quarter of the size of Y due to chrominance subsampling (usual in YUV formats).
    const int frameSize = width * height;
    const int uvFrameSize = frameSize / 4;
    const cv::Size frameDimensions(width, height);

    // Opening the YUV file:

    // Uses std::ifstream to open the file in binary mode.
    // Check if the file opens correctly.
    std::ifstream yuvFile(filename, std::ios::binary);
    if (!yuvFile) {
        std::cerr << "Cannot open file!" << std::endl;
        return cv::Mat();
    }

    // Reading the frame position:

    // Use seekg to position the read pointer at the start of the desired frame. Each frame has a size of 1.5 * frameSize (Y plus U and V).

    // Calculate the frame position
    yuvFile.seekg(frameSize * 1.5 * frameNum, std::ios::beg);


    // Reading and processing of Y, U, V components:

    // Y (luminance): Reads the Y component in an OpenCV matrix (cv::Mat), converts the data to double for greater accuracy in processing.
    // U and V (chrominance): Reads the U and V components, which are subsampled to half the resolution of Y in each dimension. It then converts the data to double and resizes these matrices to the original frame size using linear interpolation.
    // Read Y component
    cv::Mat Y(frameDimensions, CV_8UC1);
    yuvFile.read(reinterpret_cast<char*>(Y.data), frameSize);
    if (!yuvFile) {
        std::cerr << "Error reading the Y component" << std::endl;
        return cv::Mat();
    }

    // Convert Y to double
    Y.convertTo(Y, CV_64F);

    // Read U component
    cv::Mat U(height / 2, width / 2, CV_8UC1);
    yuvFile.read(reinterpret_cast<char*>(U.data), uvFrameSize);
    if (!yuvFile) {
        std::cerr << "Error reading the U component" << std::endl;
        return cv::Mat();
    }

    // Convert U to double and resize
    U.convertTo(U, CV_64F);
    cv::resize(U, U, frameDimensions, 0, 0, cv::INTER_LINEAR);

    // Read V component
    cv::Mat V(height / 2, width / 2, CV_8UC1);
    yuvFile.read(reinterpret_cast<char*>(V.data), uvFrameSize);
    if (!yuvFile) {
        std::cerr << "Error reading the V component" << std::endl;
        return cv::Mat();
    }

    // Convert V to double and resize
    V.convertTo(V, CV_64F);
    cv::resize(V, V, frameDimensions, 0, 0, cv::INTER_LINEAR);

    // Combining Y, U, V into a single cv::Mat:
    // Uses cv::merge to combine the three matrices (Y, U, V) into a single matrix (YUV), which represents the entire frame in the YUV color space.
    // Merge Y, U, and V into one Mat
    std::vector<cv::Mat> channels = {Y, U, V};
    cv::Mat YUV;
    cv::merge(channels, YUV);

    // The function returns the YUV matrix, which contains the processed frame.
    return YUV;
}