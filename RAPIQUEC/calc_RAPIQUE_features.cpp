#include <vector>
#include <string>
#include <cmath> 
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <cstdio>

// YUVread:

// Parameters: receives a string with the file name, the frame dimensions (width and height), and a frame number (frameNum).
// Process: Reads and processes a specific frame from the YUV file.
cv::Mat YUVread(const std::string& filename, int width, int height, int frameNum);
// RAPIQUE_spatial_features that extracts a set of spatial features from an RGB image for image quality evaluation
std::vector<double> RAPIQUE_spatial_features(const cv::Mat& RGB);

std::vector<double> calc_RAPIQUE_features(const std::string& yuv_name, int width, int height,
                                          int framerate, float minside, const std::string& net,
                                          const std::string& layer, int log_level) {
    
        std::vector<double> feats_frames;
        std::vector<double> feats_frames2;

    // Opens the YUV file and calculates the total number of frames (nb_frames) based on the file size and frame dimensions.
    std::ifstream test_file(yuv_name, std::ios::binary | std::ios::ate);
    if (!test_file.is_open()) {
        std::cerr << "Test YUV file not found.\n";
        return feats_frames;
    }

    std::streamsize file_length = test_file.tellg();
    int nb_frames = static_cast<int>(std::floor(file_length / (width * height * 1.5)));

    // int nb_frames = std::floor(file_length / (width * height * 1.5));

    test_file.seekg(0, std::ios::beg);
    
    // A for loop iterates over the selected frames of the video (based on framerate).
    // For each iteration, it reads the current frame (this_YUV_frame), the previous frame (prev_YUV_frame), and the next frame (next_YUV_frame) using the YUVread function.

    // Converts YUV frames to RGB and resizes them if necessary to maintain a minimum size (minside).
    // Calculation of spatial features:

    // Uses the RAPIQUE_spatial_features function to calculate spatial features of the previous and next RGB frames.
    // Adds these features to a vector (feats_frames).

    // for (int fr = std::floor(framerate / 2); fr < nb_frames - 2; fr += framerate) {
    for (int fr = static_cast<int>(std::floor(framerate / 2)); fr < nb_frames - 2; fr += framerate) {
        cv::Mat this_YUV_frame = YUVread(yuv_name, width, height, fr);
        int calculatedFrame = std::max(1, fr - static_cast<int>(std::floor(framerate / 3)));
        cv::Mat prev_YUV_frame = YUVread(yuv_name, width, height, calculatedFrame);
        int frameIndex = std::min(nb_frames - 2, fr + static_cast<int>(std::floor(framerate / 3)));
        cv::Mat next_YUV_frame = YUVread(yuv_name, width, height, frameIndex);

        cv::Mat this_rgb, prev_rgb, next_rgb;
        if (this_YUV_frame.empty() || prev_YUV_frame.empty() || next_YUV_frame.empty()) {
            std::cerr << "Error: One or more YUV frames are empty.\n";
            // Handle error appropriately
        }
        
        this_YUV_frame.convertTo(this_YUV_frame, CV_8U);
        prev_YUV_frame.convertTo(prev_YUV_frame, CV_8U);
        next_YUV_frame.convertTo(next_YUV_frame, CV_8U);
        cv::cvtColor(this_YUV_frame, this_rgb, cv::COLOR_YUV2BGR);
        cv::cvtColor(prev_YUV_frame, prev_rgb, cv::COLOR_YUV2BGR);
        cv::cvtColor(next_YUV_frame, next_rgb, cv::COLOR_YUV2BGR);

        double sside = std::min(this_rgb.rows, this_rgb.cols);
        double ratio = minside / sside;
        if (ratio < 1) {
            cv::resize(prev_rgb, prev_rgb, cv::Size(), ratio, ratio, cv::INTER_AREA);
            cv::resize(next_rgb, next_rgb, cv::Size(), ratio, ratio, cv::INTER_AREA);
            cv::resize(this_rgb, this_rgb, cv::Size(), ratio, ratio, cv::INTER_AREA);
        }

        std::vector<double> prev_feats_spt = RAPIQUE_spatial_features(prev_rgb);
        std::vector<double> next_feats_spt = RAPIQUE_spatial_features(next_rgb);

        // Add extracted features to feats_frames
        feats_frames.insert(feats_frames.end(), prev_feats_spt.begin(), prev_feats_spt.end());
        feats_frames.insert(feats_frames.end(), next_feats_spt.begin(), next_feats_spt.end());
    }
    // Dummy feature generation:

    // A for loop adds example values to the feats_frames2 vector.
        for (int i = 0; i < 10; ++i) {
            feats_frames2.push_back(i * 0.1); // Add example values to the vector.
        }
    // Retorna el vector feats_frames2, que contiene las características calculadas o los valores de ejemplo.
    return feats_frames2;
}