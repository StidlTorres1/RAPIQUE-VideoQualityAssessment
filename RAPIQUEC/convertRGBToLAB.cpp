#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>

cv::Mat convertRGBToLAB(const cv::Mat& I) {
    // Application of the Gaussian filter:

    // The original image I is smoothed using a Gaussian filter. The kernel size of the filter is 3x3 and the standard deviation is 3.
    // The cv::GaussianBlur is used for this purpose.
    // The smoothed image is stored in the variable gfrgb.
    cv::Mat gfrgb;
    GaussianBlur(I, gfrgb, cv::Size(3, 3), 3, 3, cv::BORDER_REFLECT);

    
    // BGR to LAB conversion:

    // Although the original image I is mentioned as RGB, OpenCV generally works with the BGR format, so the function assumes that I is in BGR.
    // It uses cv::cvtColor to convert the smoothed image (gfrgb) from the BGR color space to the LAB color space.
    // The converted image is stored in the variable lab.
    cv::Mat lab;
    cv::cvtColor(gfrgb, lab, cv::COLOR_BGR2Lab);

    // The function returns the lab image, which is the converted and smoothed version of the original image in the LAB color space.
    return lab;
}

// cv::Mat convertRGBToLAB(const cv::Mat& I) {
//     // Aplica un filtro gaussiano con un kernel de tamaño 3x3 y desviación estándar de 3
//     cv::Mat gfrgb;
//     GaussianBlur(I, gfrgb, cv::Size(3, 3), 3, 3, cv::BORDER_REFLECT);

//     // Convierte la imagen de RGB a CIE XYZ
//     cv::Mat xyz;
//     cv::cvtColor(gfrgb, xyz, cv::COLOR_BGR2XYZ);

//     // Ajuste manual para el punto blanco 'D65' en la conversión de XYZ a LAB
//     // Constantes para D65
//     const double Xn = 95.047;
//     const double Yn = 100.000;
//     const double Zn = 108.883;

//     cv::Mat lab(xyz.size(), CV_32FC3);
//     for (int i = 0; i < xyz.rows; ++i) {
//         for (int j = 0; j < xyz.cols; ++j) {
//             cv::Vec3f xyzPixel = xyz.at<cv::Vec3f>(i, j);

//             double X = xyzPixel[0] / Xn;
//             double Y = xyzPixel[1] / Yn;
//             double Z = xyzPixel[2] / Zn;

//             // Fórmula de conversión de XYZ a LAB
//             X = (X > 0.008856) ? pow(X, 1.0/3.0) : (7.787 * X) + (16.0 / 116.0);
//             Y = (Y > 0.008856) ? pow(Y, 1.0/3.0) : (7.787 * Y) + (16.0 / 116.0);
//             Z = (Z > 0.008856) ? pow(Z, 1.0/3.0) : (7.787 * Z) + (16.0 / 116.0);

//             lab.at<cv::Vec3f>(i, j)[0] = (116.0 * Y) - 16.0;
//             lab.at<cv::Vec3f>(i, j)[1] = 500.0 * (X - Y);
//             lab.at<cv::Vec3f>(i, j)[2] = 200.0 * (Y - Z);
//         }
//     }

//     return lab;
// }