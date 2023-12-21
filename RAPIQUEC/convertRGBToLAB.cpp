#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgproc/imgproc.hpp>

cv::Mat convertRGBToLAB(const cv::Mat& inputImage) {
    if (inputImage.empty()) {
        return cv::Mat(); 
    }

    cv::Mat blurredImage;
    cv::GaussianBlur(inputImage, blurredImage, cv::Size(3, 3), 3, 3, cv::BORDER_REFLECT_101); 

    cv::Mat labImage;
    cv::cvtColor(blurredImage, labImage, cv::COLOR_BGR2Lab);

    return labImage;
}

//Documentation

// To convert an image from the RGB color space to the LAB color space. Line by line:
// 1.	#include <opencv2/opencv.hpp>: This line includes the main OpenCV header file, which provides access to a wide range of computer vision functionalities.
// 2.	#include <opencv2/core/ocl.hpp>: Includes OpenCV's OpenCL (Open Computing Language) interface. This is used for GPU acceleration of OpenCV operations, though it's not explicitly utilized in this function.
// 3.	#include <opencv2/imgproc/imgproc.hpp>: Includes the image processing module of OpenCV, providing access to various image processing functions.
// 4.	cv::Mat convertRGBToLAB(const cv::Mat& inputImage) {: This line defines a function convertRGBToLAB that takes a constant reference to a cv::Mat object (representing an image) as its parameter and returns a cv::Mat object. cv::Mat is a matrix data type in OpenCV used to store images.
// 5.	if (inputImage.empty()) { return cv::Mat(); }: This line checks if the input image is empty (i.e., it has no data). If it is, the function returns an empty cv::Mat object. This is a safety check to handle edge cases where the input image might not be valid.
// 6.	cv::Mat blurredImage;: Declares a cv::Mat object blurredImage to store the blurred version of the input image.
// 7.	cv::GaussianBlur(inputImage, blurredImage, cv::Size(3, 3), 3, 3, cv::BORDER_REFLECT_101);: Applies Gaussian blur to the input image. The function cv::GaussianBlur takes several arguments:
// a.	inputImage: The input image to be blurred.
// b.	blurredImage: The output image.
// c.	cv::Size(3, 3): The size of the Gaussian kernel (3x3 in this case).
// d.	3, 3: The standard deviation in the X and Y directions. Here, both are set to 3.
// e.	cv::BORDER_REFLECT_101: Specifies how the image border is handled during the blur operation.
// 8.	cv::Mat labImage;: Declares a cv::Mat object labImage to store the converted LAB image.
// 9.	cv::cvtColor(blurredImage, labImage, cv::COLOR_BGR2Lab);: Converts the blurred image from BGR (Blue, Green, Red - standard in OpenCV) to LAB color space. The function cv::cvtColor is used for color space conversions.
// 10.	return labImage;: Returns the LAB color space image.
// The function convertRGBToLAB is a utility for image preprocessing, converting an RGB image to the LAB color space after applying a Gaussian blur. LAB color space is often used in image processing applications because it is more consistent with human vision compared to RGB space. The Gaussian blur is likely applied to reduce noise and detail in the image before the color space conversion.


