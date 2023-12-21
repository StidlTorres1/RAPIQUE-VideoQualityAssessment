cmake_minimum_required(VERSION 3.0.0) #Specifies the minimum required version of CMake for this project.
project(RAPIQUEC VERSION 0.1.0 LANGUAGES C CXX) #It defines the project with the name RAPIQUEC, version 0.1.0, and specifies that C and C++ languages will be used.

include(CTest) #It includes the functionality of CTest, a testing tool for CMake.
enable_testing() #Enables the ability to run tests.


find_package(OpenCV REQUIRED) #It looks for the OpenCV package, which is required for the project. If OpenCV is not found, CMake will generate an error.
find_package(Threads REQUIRED) #Search for the Threads package, which is required for concurrent programming.
#find_package(Boost REQUIRED COMPONENTS math)

# Agrega las siguientes líneas para comprobar si OpenCV fue compilado con CUDA
#if(NOT OpenCV_CUDA_VERSION)
#  message(FATAL_ERROR "OpenCV no fue compilado con CUDA, se requiere CUDA para este proyecto.")
#endif()

# add_executable(RAPIQUEC rapique_basic_extractor.cpp nakafit.cpp est_GGD_param.cpp est_AGGD_param.cpp)
#Compiles the specified files and links them into an executable called RAPIQUEC. It includes a variety of source files that are part of the RAPIQUEC project.
add_executable(RAPIQUEC main.cpp calc_RAPIQUE_features.cpp YUVread.cpp rapique_basic_extractor.cpp nakafit.cpp est_GGD_param.cpp est_AGGD_param.cpp gen_DoG.cpp gauDerivative.cpp convertRGBtoLAB.cpp RAPIQUE_spatial_features.cpp)

#add_executable(RAPIQUEC main.cpp calc_RAPIQUE_features.cpp YUVread.cpp rapique_basic_extractor.cpp nakafit.cpp est_GGD_param.cpp est_AGGD_param.cpp)

# add_executable(RAPIQUEC main.cpp rapique_basic_extractor.cpp nakafit.cpp est_GGD_param.cpp est_AGGD_param.cpp)

#Sets the C++ compiler standard in C++17 for the project.
target_compile_features(RAPIQUEC PRIVATE cxx_std_17)

set(CPACK_PROJECT_NAME ${PROJECT_NAME}) #Set the project name for CPack, a CMake tool for package generation.
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION}) #Sets the project version for CPack.
include(CPack) #Incluye el módulo CPack

set (CMAKE_CXX_STANDARD 17) #Sets the C++ compiler standard to C++17.
set (CMAKE_CXX_STANDARD_REQUIRED ON) #Indicates that the established C++ standard is mandatory.

include_directories(${OpenCV_Include_DIRS} ) #It includes the directories of the OpenCV headers.
target_link_libraries(RAPIQUEC ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT}) #Enlaza las bibliotecas de OpenCV y las bibliotecas de hilos al ejecutable

#include_directories(${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS}) # Añade ${Boost_INCLUDE_DIRS} aquí

#target_link_libraries(RAPIQUEC ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT} ${Boost_LIBRARIES}) # Añade ${Boost_LIBRARIES} aquí
