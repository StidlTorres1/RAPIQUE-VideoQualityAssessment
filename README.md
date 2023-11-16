# RAPIQUEC Project README

## NOTE
To run the first version of the project preview you must go to the RAPIQUEC folder and there generate the build and the cmake configuration.
## Overview
RAPIQUEC is a sophisticated C++ project designed for processing a large volume of videos to extract relevant features for image/video quality assessment. It is a C++ re-implementation of the RAPIQUE (Rapid and Accurate Image Quality Estimation) algorithm, an advanced model for image quality evaluation based on machine learning techniques. The project efficiently handles extensive video datasets and leverages FFmpeg for format conversion and temporary file management.

## System Requirements
- Visual Studio Code
- C++17 compatible environment
- OpenCV 4.8.0
- FFmpeg for format conversion

## Installation Guide

### Visual Studio Code
Download and install Visual Studio Code from [here](https://code.visualstudio.com/download).

### Installing C++17 Dependencies
- **Option 1**: Install MinGW using instructions provided [here](https://nuwen.net/mingw.html#install).
- **Option 2**: For Windows, follow the C++17 installation guide available [here](https://www.geeksforgeeks.org/complete-guide-to-install-c17-in-windows/).

### OpenCV Installation
Install OpenCV 4.8.0 and add it to your system path. Detailed instructions and downloads are available [here](https://opencv.org/releases/). A helpful video guide can also be found [here](https://www.youtube.com/watch?v=m9HBM1m_EMU).

## Project Setup in Visual Studio Code

### Opening the Project
- Launch Visual Studio Code.
- Select `File -> Open Folder...` and navigate to your project's CMake folder.

### Project Configuration
- Ensure CMake and CMake Tools extensions are installed in VS Code. Install them from the extension marketplace if not already present.
- After installation, the CMake configuration should be visible in the VS Code status bar.
- Choose the appropriate C++ compiler when prompted.

### Building the Project
- Click on the Build button in the CMake Tools status bar, or use the `CMake: Build` command from the command palette (`Ctrl+Shift+P`).
- This step will generate necessary files and compile the project.

### Running the Project
- Post-build, run the project directly from VS Code using `CMake: Run Without Debugging` for a regular run, or `CMake: Debug` for a debug session.

### Executing from VS Code Terminal
- Alternatively, open a terminal in VS Code (`Terminal -> New Terminal`).
- Navigate to the directory containing the build files (usually `build` or similar).
- Execute the compiled executable, e.g., `./RAPIQUEC` on Linux/Mac or `.\RAPIQUEC.exe` on Windows.

### Troubleshooting
- Ensure all dependencies (like OpenCV and Threads) are correctly installed and accessible to CMake.
- For build or execution errors, refer to the console output for detailed error information.

## Additional Information
For more details and information about previous installations, refer to the OpenCV release page and the video guide linked above.
