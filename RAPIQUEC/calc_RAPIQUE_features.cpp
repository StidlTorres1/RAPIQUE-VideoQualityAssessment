#include <vector>
#include <string>
#include <cmath> 
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <cstdio>
#include <algorithm>
#include <execution>
#include <mutex>


cv::Mat YUVread(const std::string& filename, int width, int height, int frameNum);
std::vector<double> RAPIQUE_spatial_features(const cv::Mat& RGB);

std::vector<double> calc_RAPIQUE_features(const std::string& yuv_name, int width, int height,
                                          int framerate, float minside, const std::string& net,
                                          const std::string& layer, int log_level) {
    
    std::vector<double> feats_frames;


    std::ifstream test_file(yuv_name, std::ios::binary | std::ios::ate);
    if (!test_file.is_open()) {
        std::cerr << "Test YUV file not found.\n";
        return feats_frames;
    }

    std::streamsize file_length = test_file.tellg();
    int nb_frames = static_cast<int>(std::floor(file_length / (width * height * 1.5)));

    test_file.seekg(0, std::ios::beg);

    int half_framerate = framerate / 2;
    int third_framerate = framerate / 3;

    feats_frames.reserve(nb_frames * 2);

    std::vector<std::tuple<cv::Mat, cv::Mat, cv::Mat>> frame_triplets;

    frame_triplets.reserve(nb_frames);
    for (int fr = half_framerate; fr < nb_frames - 2; fr += framerate) {
        int calculatedFrame = std::max(1, fr - third_framerate);
        int frameIndex = std::min(nb_frames - 2, fr + third_framerate);
        
        frame_triplets.emplace_back(YUVread(yuv_name, width, height, fr),
                                    YUVread(yuv_name, width, height, calculatedFrame),
                                    YUVread(yuv_name, width, height, frameIndex));
    }
    std::mutex mtx;
    std::for_each(std::execution::par, frame_triplets.begin(), frame_triplets.end(),
                  [&](const std::tuple<cv::Mat, cv::Mat, cv::Mat>& frames) {
        const auto& [this_YUV_frame, prev_YUV_frame, next_YUV_frame] = frames;

        if (this_YUV_frame.empty() || prev_YUV_frame.empty() || next_YUV_frame.empty()) {
            std::cerr << "Error: One or more YUV frames are empty.\n";
            return;
        }

        cv::Mat this_rgb, prev_rgb, next_rgb;
        cv::cvtColor(this_YUV_frame, this_rgb, cv::COLOR_YUV2BGR);
        cv::cvtColor(prev_YUV_frame, prev_rgb, cv::COLOR_YUV2BGR);
        cv::cvtColor(next_YUV_frame, next_rgb, cv::COLOR_YUV2BGR);

        double sside = std::min(this_rgb.rows, this_rgb.cols);
        double ratio = minside / sside;
        if (ratio < 1) {
            cv::resize(prev_rgb, prev_rgb, cv::Size(), ratio, ratio, cv::INTER_CUBIC);
            cv::resize(next_rgb, next_rgb, cv::Size(), ratio, ratio, cv::INTER_CUBIC);
        }

        std::vector<double> prev_feats_spt = RAPIQUE_spatial_features(prev_rgb);
        std::vector<double> next_feats_spt = RAPIQUE_spatial_features(next_rgb);

        auto n_features = prev_feats_spt.size();
        std::vector<double> feats_spt_mean(n_features);
        std::vector<double> feats_spt_diff(n_features);

        std::transform(std::execution::par, prev_feats_spt.begin(), prev_feats_spt.end(), next_feats_spt.begin(),
                       feats_spt_mean.begin(), [](double a, double b) { return (a + b) / 2.0; });

        std::transform(std::execution::par, prev_feats_spt.begin(), prev_feats_spt.end(), next_feats_spt.begin(),
                       feats_spt_diff.begin(), [](double a, double b) { return std::abs(a - b); });

      
        std::lock_guard<std::mutex> guard(mtx); 

        feats_frames.insert(feats_frames.end(), feats_spt_mean.begin(), feats_spt_mean.end());
        feats_frames.insert(feats_frames.end(), feats_spt_diff.begin(), feats_spt_diff.end());
       
    });

        cv::FileStorage fs1("feat_frames.xml", cv::FileStorage::WRITE);

        fs1 << "FeatFrames" << feats_frames;

        fs1.release();

        std::cout << "Matriz guardada con éxito en formato XML." << std::endl;
    return feats_frames;
}


//Documentation
// Extracting and processing features from YUV video frames for video quality assessment. It uses OpenCV for image processing, along with other standard libraries for file and stream operations. Line by line:
// 1-8. Include statements:
// •	These lines include necessary headers for vector operations, string manipulation, mathematical functions, file streaming, OpenCV functionalities, OpenCL interface for GPU optimizations, C standard I/O, algorithm functions, execution policies for parallel algorithms, and mutex for thread safety.
// 9-11. Function declarations:
// •	Declares three functions YUVread, RAPIQUE_spatial_features, and calc_RAPIQUE_features that are defined elsewhere or later in the code.
// 12-76. Function calc_RAPIQUE_features:
// •	This function calculates RAPIQUE (Rapid and Accurate Image Quality Evaluator) features from a given YUV file.
// 13-17. File handling and initial checks:
// •	Opens a YUV file and checks if it's open. If not, it outputs an error message and returns an empty feature vector.
// 18-22. Frame number calculation:
// •	Calculates the number of frames in the YUV file based on its size and the dimensions of each frame.
// 23-27. Frame rate processing:
// •	Computes half_framerate and third_framerate for later use in determining which frames to process.
// 28-37. Frame triplet preparation:
// •	Reserves space for frame triplets and populates them by reading specific frames from the YUV file. It uses the YUVread function and adjusts frame indices based on the frame rate.
// 38.	Mutex declaration:
// •	Declares a mutex for thread safety during parallel processing.
// 39-75. Parallel processing of frames:
// •	Processes the frame triplets in parallel using std::for_each with std::execution::par policy.
// •	Converts YUV frames to RGB.
// •	Resizes the frames if necessary based on the minimum side length (minside) and aspect ratio.
// •	Extracts spatial features from previous and next frames using RAPIQUE_spatial_features.
// •	Calculates the mean and difference of features between the previous and next frames.
// •	Uses a mutex to safely add these features to the overall feature vector feats_frames.
// 77-83. XML file output:
// •	Writes the features to an XML file named "feat_frames.xml".
// •	Outputs a success message upon saving the file.
// 84.	Return statement:
// •	Returns the calculated features.
// This function is a comprehensive implementation for feature extraction from video frames, tailored for video quality assessment. It leverages parallel processing to efficiently handle multiple frames and computes spatial features for pairs of frames, which are likely used to assess temporal changes and overall video quality. The use of mutexes ensures thread safety during parallel execution. The features are then saved in an XML format for further use or analysis.

//Otra Version
// #include <vector>
// #include <string>
// #include <cmath> 
// #include <fstream>
// #include <opencv2/opencv.hpp>
// #include <opencv2/core/ocl.hpp>
// #include <cstdio>
// #include <algorithm>
// #include <execution>
// #include <mutex>

// cv::Mat YUVread(const std::string& filename, int width, int height, int frameNum);
// std::vector<double> RAPIQUE_spatial_features(const cv::Mat& RGB);

// std::vector<double> calc_RAPIQUE_features(const std::string& yuv_name, int width, int height,
//                                           int framerate, float minside, const std::string& net,
//                                           const std::string& layer, int log_level) {
    
//     std::vector<double> feats_frames;

//     std::ifstream test_file(yuv_name, std::ios::binary | std::ios::ate);
//     if (!test_file.is_open()) {
//         std::cerr << "Test YUV file not found.\n";
//         return feats_frames;
//     }

//     std::streamsize file_length = test_file.tellg();
//     int nb_frames = static_cast<int>(std::floor(file_length / (width * height * 1.5)));

//     test_file.seekg(0, std::ios::beg);

//     int half_framerate = framerate / 2;
//     int third_framerate = framerate / 3;

//     std::vector<std::tuple<cv::Mat, cv::Mat, cv::Mat>> frame_triplets;

//     for (int fr = half_framerate; fr < nb_frames - 2; fr += framerate) {
//         int calculatedFrame = std::max(1, fr - third_framerate);
//         int frameIndex = std::min(nb_frames - 2, fr + third_framerate);
        
//         frame_triplets.emplace_back(YUVread(yuv_name, width, height, fr),
//                                     YUVread(yuv_name, width, height, calculatedFrame),
//                                     YUVread(yuv_name, width, height, frameIndex));
//     }

//     std::mutex mtx;
//     std::for_each(std::execution::par, frame_triplets.begin(), frame_triplets.end(),
//                   [&](const std::tuple<cv::Mat, cv::Mat, cv::Mat>& frames) {
//         const auto& [this_YUV_frame, prev_YUV_frame, next_YUV_frame] = frames;

//         if (this_YUV_frame.empty() || prev_YUV_frame.empty() || next_YUV_frame.empty()) {
//             std::cerr << "Error: One or more YUV frames are empty.\n";
//             return;
//         }

//         cv::Mat this_rgb, prev_rgb, next_rgb;
//         cv::cvtColor(this_YUV_frame, this_rgb, cv::COLOR_YUV2BGR);
//         cv::cvtColor(prev_YUV_frame, prev_rgb, cv::COLOR_YUV2BGR);
//         cv::cvtColor(next_YUV_frame, next_rgb, cv::COLOR_YUV2BGR);

//         double sside = std::min(this_rgb.rows, this_rgb.cols);
//         double ratio = minside / sside;
//         if (ratio < 1) {
//             cv::resize(prev_rgb, prev_rgb, cv::Size(), ratio, ratio, cv::INTER_AREA);
//             cv::resize(next_rgb, next_rgb, cv::Size(), ratio, ratio, cv::INTER_AREA);
//         }

//         std::vector<double> prev_feats_spt = RAPIQUE_spatial_features(prev_rgb);
//         std::vector<double> next_feats_spt = RAPIQUE_spatial_features(next_rgb);

//         std::vector<double> feats_spt_mean(prev_feats_spt.size());
//         std::vector<double> feats_spt_diff(prev_feats_spt.size());

//         std::transform(std::execution::par, prev_feats_spt.begin(), prev_feats_spt.end(), next_feats_spt.begin(),
//                        feats_spt_mean.begin(), [](double a, double b) { return (a + b) / 2.0; });

//         std::transform(std::execution::par, prev_feats_spt.begin(), prev_feats_spt.end(), next_feats_spt.begin(),
//                        feats_spt_diff.begin(), [](double a, double b) { return std::abs(a - b); });

//         {
//             std::lock_guard<std::mutex> guard(mtx); 
//             feats_frames.insert(feats_frames.end(), feats_spt_mean.begin(), feats_spt_mean.end());
//             feats_frames.insert(feats_frames.end(), feats_spt_diff.begin(), feats_spt_diff.end());
//         }
//     });

//     cv::FileStorage fs1("feat_frames.xml", cv::FileStorage::WRITE);
//     fs1 << "FeatFrames" << feats_frames;
//     fs1.release();

//     std::cout << "Matriz guardada con éxito en formato XML." << std::endl;
//     return feats_frames;
// }
