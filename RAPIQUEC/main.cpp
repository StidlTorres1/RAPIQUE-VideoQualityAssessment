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
        // Verificar y activar soporte para GPU
    if(cv::ocl::haveOpenCL()) {
        cv::ocl::setUseOpenCL(true);
        std::cout << "OpenCL support detected. Using GPU..." << std::endl;
    } else {
        std::cout << "OpenCL support not detected. Using CPU..." << std::endl;
    }
     // Verificar sistema operativo en Windows, Linux o macOS
    const std::string path_separator = 
    #ifdef _WIN32
        "\\";
    #else
        "/";
    #endif
        // Parametros
    const std::string algo_name = "RAPIQUE";
    const std::string data_name = "KONVID_1K";
    const bool write_file = true;
    std::string root_path_data,data_path;
    std::filesystem::path currentPath = std::filesystem::current_path();
    
    // Obtener el directorio que está dos niveles arriba
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
        // Saltar la primera línea (encabezados).
    std::getline(inFile, line);

    while (std::getline(inFile, line)) {
        try {
            filelist.push_back(DataRow(line));
        } catch (const std::exception& e) {
            std::cerr << "Error procesando línea: " << line << ". Razón: " << e.what() << std::endl;
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

    const std::string net = "resnet50";
    const std::string layer = "avg_pool";
    
    // Paralelización usando std::async
    std::vector<std::future<void>> futures;

    for(const auto& entry : filelist) {
        futures.push_back(std::async(std::launch::async, [&entry, &path_separator, &data_path, &out_path_temp](){
            std::string video_name = data_path + path_separator + std::to_string(entry.flickr_id) + ".mp4";
            std::string yuv_name = out_path_temp + path_separator + std::to_string(entry.flickr_id) + ".yuv";

            if(video_name != yuv_name) {
                std::string cmd = "ffmpeg -loglevel error -y -i " + video_name + " -pix_fmt yuv420p -vsync 0 " + yuv_name;
                system(cmd.c_str());
            }
        }));
    }

    for(auto& fut : futures) {
        fut.get();
    }
    
    
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span = t2 - t1;
    std::cout << "El código se ejecutó en: " << time_span.count() << " segundos." << std::endl;
    return 0;
}
