cmake_minimum_required(VERSION 3.0.0) 
project(RAPIQUEC VERSION 0.1.0 LANGUAGES C CXX) 
include(CTest) 
enable_testing() 


find_package(OpenCV REQUIRED) 
find_package(Threads REQUIRED) 

add_executable(RAPIQUEC main.cpp calc_RAPIQUE_features.cpp YUVread.cpp rapique_basic_extractor.cpp nakafit.cpp est_GGD_param.cpp est_AGGD_param.cpp gen_DoG.cpp gauDerivative.cpp convertRGBtoLAB.cpp RAPIQUE_spatial_features.cpp)




target_compile_features(RAPIQUEC PRIVATE cxx_std_17)

set(CPACK_PROJECT_NAME ${PROJECT_NAME}) 
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION}) 
include(CPack) 

set (CMAKE_CXX_STANDARD 17) 
set (CMAKE_CXX_STANDARD_REQUIRED ON) 

include_directories(${OpenCV_Include_DIRS} ) 
target_link_libraries(RAPIQUEC ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT}) 



