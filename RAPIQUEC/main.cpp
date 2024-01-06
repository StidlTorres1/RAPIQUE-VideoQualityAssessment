#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <fstream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <chrono>
#include <future>
using namespace std::chrono;
// Inclusion of Libraries and Namespaces:

// Libraries are included to handle files, strings, data streams, and OpenCV is used for image and video processing.
// The std::chrono namespace is used to measure program execution time.
// calc_RAPIQUE_features Function Declaration:

// A function is declared to calculate the RAPIQUE characteristics of a video file in YUV format.
// DataRow Structure:

// A structure is defined to store the data for each video, such as Flickr ID, quality score (MOS), dimensions, pixel format, etc.
// Main function main:

// OpenCL Support Detection: verifies and enables GPU support through OpenCL.
// Paths and Parameters Definition: Sets file and directory paths, and defines parameters such as algorithm name, data name, log level, etc.
// calc_RAPIQUE_features which is used to calculate spatial features of a video file in YUV format for image/video quality evaluation.
// Parameters: Receives the YUV file name (yuv_name), the frame dimensions (width and height), the frame rate (framerate), a minimum size (minside), and additional parameters (net, layer, log_level) that are not used in the body of the function.
// Process: Calculates spatial characteristics of the video frames.

// Read List of Video Files Reading: Reads a list of videos from a CSV file and stores the data in a DataRow vector.
// Directory Creation: Creates directories to store temporary files and results if they do not exist.
// Video Processing:
// Converts each video from MP4 to YUV using FFmpeg.
// Calculate RAPIQUE features for each YUV file.
// Calculates the average of the features, excluding NaN values.
// Adds the features to a feats_mat array.
// Deletes YUV files to free up space.
// Execution Time Measurement: Measures and displays the total execution time of the program.
// Program Termination:

// The program terminates by returning 0, indicating a successful execution.



std::vector<double> calc_RAPIQUE_features(const std::string& yuv_name, int width, int height, 
                                          int framerate, float minside, const std::string& net, 
                                          const std::string& layer, int log_level);
struct DataRow {
    long long flickr_id;
    double mos;
    int width;
    int height;
    std::string pixfmt;
    double framerate;
    int nb_frames;
    int bitdepth;
    int bitrate;

    DataRow(const std::string& line) {
        std::istringstream iss(line);
        std::string token;

        std::getline(iss, token, ','); flickr_id = std::stoll(token);
        std::getline(iss, token, ','); mos = std::stod(token);
        std::getline(iss, token, ','); width = std::stoi(token);
        std::getline(iss, token, ','); height = std::stoi(token);
        std::getline(iss, token, ','); pixfmt = token;
        std::getline(iss, token, ','); framerate = std::stod(token);
        std::getline(iss, token, ','); nb_frames = std::stoi(token);
        std::getline(iss, token, ','); bitdepth = std::stoi(token);
        std::getline(iss, token, ','); bitrate = std::stoi(token);
    }
};
int main(int, char**){

    
     auto t1 = std::chrono::high_resolution_clock::now();
        // Verify and enable GPU support
    if(cv::ocl::haveOpenCL()) {
        cv::ocl::setUseOpenCL(true);
        std::cout << "OpenCL support detected. Using GPU..." << std::endl;
    } else {
        std::cout << "OpenCL support not detected. Using CPU..." << std::endl;
    }
     // Verify operating system on Windows, Linux or macOS
    const std::string path_separator = 
    #ifdef _WIN32
        "\\";
    #else
        "/";
    #endif
        // Parameters
    const std::string algo_name = "RAPIQUE";
    const std::string data_name = "KONVID_1K";
    const int log_level = 0;  // 1=verbose, 0=quite
    const bool write_file = true;
    std::string root_path_data,data_path;
    std::filesystem::path currentPath = std::filesystem::current_path();
    
    // Get the directory that is two levels up
    std::filesystem::path desiredPath = currentPath.parent_path().parent_path();
    std::string root_path = desiredPath.string()+path_separator+"dataBase"+path_separator;

    // Definir paths
    if(data_name == "KONVID_1K") {
    root_path_data = root_path+"KONVID_1K"+path_separator;
    data_path = root_path_data + "KoNViD_1k_videos";
    }
    else if(data_name == "LIVE_VQC") {
        root_path_data = root_path+"LIVE_VQC"+path_separator;
        data_path = root_path_data + "VideoDatabase";
    }
    else if(data_name == "YOUTUBE_UGC") {
        root_path_data = root_path+"YT_UGC"+path_separator;
        root_path_data = root_path_data + "original_videos";
    }
    else if(data_name == "LIVE_HFR") {
        root_path_data = root_path+"LIVE_HFR"+path_separator;
        data_path = root_path_data;
    }
    else if(data_name == "LIVE_VQA") {
        root_path_data = root_path+"LIVE_VQA"+path_separator;
        data_path = root_path_data + "videos";
    }

    const std::string feat_path = "mos_files";
    std::string filelist_csv = root_path+ feat_path + path_separator + data_name + "_metadata.csv"; 

    std::vector<DataRow> filelist; 
    std::ifstream inFile(filelist_csv);

    if (inFile.is_open()) {
        std::string line;
        // Skip the first line (headers).
    std::getline(inFile, line);

    while (std::getline(inFile, line)) {
        try {
            filelist.push_back(DataRow(line));
        } catch (const std::exception& e) {
            std::cerr << "Error processing line: " << line << ". Cause: " << e.what() << std::endl;
        }
    }
        inFile.close();
    } else {
        std::cerr << "Unable to open file " << filelist_csv << std::endl;
    }
    const std::string out_path = root_path+"feat_files";
    if(!std::filesystem::exists(out_path)) {
        std::filesystem::create_directory(out_path);
    }
    const std::string out_path_temp = root_path+"tmp";
    if(!std::filesystem::exists(out_path_temp)) {
        std::filesystem::create_directory(out_path_temp);
    }
    const std::string out_mat_name = out_path + path_separator + data_name + "_" + algo_name + "_feats.mat";
    cv::Mat feats_mat;
    std::vector<std::vector<double>> feats_mat_frames(filelist.size());

    // init deep learning models
    const float minside = 512.0f;
    const std::string net = "resnet50";
    const std::string layer = "avg_pool";
    std::mutex mtx;
    
    
    for(const auto& entry : filelist) {
        std::string video_name = data_path + path_separator + std::to_string(entry.flickr_id) + ".mp4";
        std::string yuv_name = out_path_temp + path_separator + std::to_string(entry.flickr_id) + ".yuv";

        if(video_name != yuv_name) {
            std::string cmd = "ffmpeg -loglevel error -y -i " + video_name + " -pix_fmt yuv420p -vsync 0 " + yuv_name;
            system(cmd.c_str());
        }

        int width = entry.width;
        int height = entry.height;
        int framerate = std::round(entry.framerate);
        
        std::vector<double> feats_frames = calc_RAPIQUE_features(yuv_name, width, height, 
                                                                 framerate, minside, net, 
                                                                 layer, log_level);
    
        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span = t2 - t1;
        std::cout << "The code was executed in: " << time_span.count() << " seconds." << std::endl;
        // Calculation of the mean of the characteristics, omitting NaN values.
        double sum = 0;
        int valid_count = 0;
        for (const auto& value : feats_frames) {
            if (!std::isnan(value)) {
                sum += value;
                ++valid_count;
            }
        }
        double mean = valid_count > 0 ? sum / valid_count : 0;

        // Add to feature matrix
        feats_mat.push_back(mean);
        feats_mat_frames.push_back(feats_frames);

        // Clear cache by deleting the YUV file
        std::remove(yuv_name.c_str());

        // (Omitted: Writing results to file)
    }
    
    
    // auto t2 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> time_span = t2 - t1;
    // std::cout << "The code was executed in: " << time_span.count() << " seconds." << std::endl;
    return 0;
}




