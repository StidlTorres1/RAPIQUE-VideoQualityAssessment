#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>

// gen_DoG which generates pairs of Gaussian images and their differences (Difference of Gaussian (DoG) for image processing
// Parameters: Receives an image (img) and an integer (kband) that defines the number of bands (levels) to generate.
// Return: Returns a pair of cv::Mat vectors. The first one contains the Gaussian images and the second one the Gaussian differences (DoG).
std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>> gen_DoG(const cv::Mat& img, int kband);
// gauDerivative, which generates two matrices representing the derivatives on the X-axis and on the Y-axis of a Gaussian filter
// Parameter: A sigma value which is the standard deviation of the Gaussian filter.
// Return: A pair of matrices (cv::Mat) corresponding to the Gaussian derivatives on the X and Y axes.
std::pair<cv::Mat, cv::Mat> gauDerivative(double sigma);
// rapique_basic_extractor that extracts a set of features from an image using image processing and statistical methods
// The rapique_basic_extractor function takes an image (cv::Mat) as argument and returns a vector (std::vector<double>) of extracted features.
std::vector<double> rapique_basic_extractor(const cv::Mat& src);
// convertRGBToLAB which converts an image from RGB color space to LAB after applying a Gaussian filter
// Parameter: The function takes a cv::Mat object (an image) as parameter, named I.
// Return: Returns a processed image in the LAB color space.
cv::Mat convertRGBToLAB(const cv::Mat& I);

std::vector<double> RAPIQUE_spatial_features(const cv::Mat& RGB) {
    // The function first checks if the input image (RGB) is really an RGB image (with 3 channels). If it is not, it throws an exception.
    if (RGB.channels() != 3) {
        throw std::invalid_argument("The input should be an RGB image");
    }
    // Processing Parameters:

    // Defines various constants to be used for image processing, such as scales, bands and parameters for Gaussian derivatives.
    const int kscale = 2;
    const int kband = 4;
    const double sigmaForGauDerivative = 1.66;
    const double scaleFactorForGaussianDer = 0.28;

    // RGB to Grayscale Conversion and Processing:

    // Converts the RGB image to grayscale (Y), and then applies a Sobel filter to obtain the magnitude gradient (GM) and a Gaussian filter to obtain the logarithmic representation (LOG).
    cv::Mat Y;
    cv::cvtColor(RGB, Y, cv::COLOR_BGR2GRAY);
    Y.convertTo(Y, CV_64F);

    cv::Mat GM;
    cv::Sobel(Y, GM, CV_64F, 1, 1);
    GM = cv::abs(GM);

    cv::Mat LOG;
    cv::Mat window2 = cv::getGaussianKernel(9, 9 / 6, CV_64F);
    window2 = window2 / cv::sum(cv::abs(window2))[0];
    cv::filter2D(Y, LOG, CV_64F, window2);
    LOG = cv::abs(LOG);

    // Difference of Gaussians (DoG) generation:

    // Uses the gen_DoG function to generate a series of DoG images from the grayscale image.
    auto [_, DOG] = gen_DoG(Y, kband);

    cv::Mat RGBDouble;
    RGB.convertTo(RGBDouble, CV_64F);

    // Calculation of O1 and O2 Chroma Components:

    // Calculates two chroma components (O1 and O2) from the RGB channels of the original image.
    
    std::vector<cv::Mat> channels(3);
    cv::split(RGBDouble, channels);
    cv::Mat O1 = 0.30 * channels[0] + 0.04 * channels[1] - 0.35 * channels[2];
    cv::Mat O2 = 0.34 * channels[0] - 0.60 * channels[1] + 0.17 * channels[2];

    // Application of Gaussian derivatives:

    // Uses the gauDerivative function to obtain Gaussian derivatives and applies them to the chromatic components and logarithmic images of RGB channels, calculating their magnitude gradients.
    auto [dx, dy] = gauDerivative(sigmaForGauDerivative / std::pow(1, scaleFactorForGaussianDer));
    // cv::Mat compResO1 = cv::filter2D(O1, CV_64F, dx) + cv::filter2D(O1, CV_64F, dy);
    // cv::Mat GMO1 = cv::abs(compResO1);



    cv::Mat compResO1, tempMat;
    cv::filter2D(O1, compResO1, CV_64F, dx, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
    cv::filter2D(O1, tempMat, CV_64F, dy, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
    cv::add(compResO1, tempMat, compResO1);
    cv::Mat GMO1 = cv::abs(compResO1);


    // cv::Mat compResO2 = cv::filter2D(O2, CV_64F, dx) + cv::filter2D(O2, CV_64F, dy);
    // cv::Mat GMO2 = cv::abs(compResO2);

    cv::Mat compResO2, tempMat2;
    cv::filter2D(O2, compResO2, CV_64F, dx, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
    cv::filter2D(O2, tempMat2, CV_64F, dy, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
    cv::add(compResO2, tempMat2, compResO2);
    cv::Mat GMO2 = cv::abs(compResO2);


    // cv::Mat logR = cv::log(channels[0] + 0.1);
    // cv::Mat logG = cv::log(channels[1] + 0.1);
    // cv::Mat logB = cv::log(channels[2] + 0.1);

    cv::Mat tempR, logR;
    cv::add(channels[0], cv::Scalar(0.1), tempR); // Suma 0.1 a cada elemento de channels[0]
    cv::log(tempR, logR); // Aplica log

    cv::Mat tempG, logG;
    cv::add(channels[1], cv::Scalar(0.1), tempG); // Suma 0.1 a cada elemento de channels[1]
    cv::log(tempG, logG); // Aplica log

    cv::Mat tempB, logB;
    cv::add(channels[2], cv::Scalar(0.1), tempB); // Suma 0.1 a cada elemento de channels[2]
    cv::log(tempB, logB); // Aplica log





    cv::Scalar meanLogR = cv::mean(logR);
    cv::Scalar meanLogG = cv::mean(logG);
    cv::Scalar meanLogB = cv::mean(logB);

    cv::Mat logRMS = logR - meanLogR[0];
    cv::Mat logGMS = logG - meanLogG[0];
    cv::Mat logBMS = logB - meanLogB[0];

    cv::Mat BY = (logRMS + logGMS - 2 * logBMS) / sqrt(6);
    cv::Mat RG = (logRMS - logGMS) / sqrt(2);

    auto [dxBY, dyBY] = gauDerivative(sigmaForGauDerivative / std::pow(1, scaleFactorForGaussianDer));
    // cv::Mat compResBY = cv::filter2D(BY, CV_64F, dxBY) + cv::filter2D(BY, CV_64F, dyBY);
    // cv::Mat GMBY = cv::abs(compResBY);

    cv::Mat compResBY, tempMat3;
    cv::filter2D(BY, compResBY, CV_64F, dxBY, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
    cv::filter2D(BY, tempMat3, CV_64F, dyBY, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
    cv::add(compResBY, tempMat3, compResBY);
    cv::Mat  GMBY = cv::abs(compResBY);


    auto [dxRG, dyRG] = gauDerivative(sigmaForGauDerivative / std::pow(1, scaleFactorForGaussianDer));
    // cv::Mat compResRG = cv::filter2D(RG, CV_64F, dxRG) + cv::filter2D(RG, CV_64F, dyRG);
    // cv::Mat GMRG = cv::abs(compResRG);

    cv::Mat compResRG, tempMat4;
    cv::filter2D(RG, compResRG, CV_64F, dxRG, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
    cv::filter2D(RG, tempMat4, CV_64F, dyRG, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
    cv::add(compResRG, tempMat4, compResRG);
    cv::Mat  GMRG = cv::abs(compResRG);



    // RGB to LAB Conversion and Processing:

    // Converts the RGB image to LAB, then extracts and processes the A and B channels in the same way as the O1 and O2 chroma components.
    cv::Mat LAB = convertRGBToLAB(RGB);
    LAB.convertTo(LAB, CV_64F);
    std::vector<cv::Mat> LABchannels(3);
    cv::split(LAB, LABchannels);

    cv::Mat A = LABchannels[1];
    cv::Mat B = LABchannels[2];
    // Preparation of Composite Matrices for Feature Extraction:

    // Creates a set of matrices (compositeMat) that includes the previously processed images and the applied Gaussian derivatives.

    auto [dxA, dyA] = gauDerivative(sigmaForGauDerivative / std::pow(1, scaleFactorForGaussianDer));
    // cv::Mat compResA = cv::filter2D(A, CV_64F, dxA) + cv::filter2D(A, CV_64F, dyA);
    // cv::Mat GMA = cv::abs(compResA);

    cv::Mat compResA, tempMat5;
    cv::filter2D(A, compResA, CV_64F, dxA, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
    cv::filter2D(A, tempMat5, CV_64F, dyA, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
    cv::add(compResA, tempMat5, compResA);
    cv::Mat  GMA = cv::abs(compResA);



    auto [dxB, dyB] = gauDerivative(sigmaForGauDerivative / std::pow(1, scaleFactorForGaussianDer));
    // cv::Mat compResB = cv::filter2D(B, CV_64F, dxB) + cv::filter2D(B, CV_64F, dyB);
    // cv::Mat GMB = cv::abs(compResB);


    cv::Mat compResB, tempMat6;
    cv::filter2D(B, compResB, CV_64F, dxB, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
    cv::filter2D(B, tempMat6, CV_64F, dyB, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
    cv::add(compResB, tempMat6, compResB);
    cv::Mat  GMB = cv::abs(compResB);


    // Feature Extraction at Multiple Scales:

    // For each matrix in compositeMat, apply resizing for different scales and then extract features using the rapique_basic_extractor function. The extracted features are added to the feats vector.

    std::vector<cv::Mat> compositeMat = {Y, GM, LOG};
    compositeMat.insert(compositeMat.end(), DOG.begin(), DOG.end());
    compositeMat.insert(compositeMat.end(), {O1, O2, GMO1, GMO2, BY, RG, GMBY, GMRG, A, B, GMA, GMB});

    std::vector<double> feats;
    for (int ch = 0; ch < compositeMat.size(); ++ch) {
        for (int scale = 1; scale <= kscale; ++scale) {
            if (ch >= 5 && scale == 1) {
                continue;
            }
            cv::Mat y_scale;
            cv::resize(compositeMat[ch], y_scale, cv::Size(), std::pow(2, -(scale - 1)), std::pow(2, -(scale - 1)), cv::INTER_LINEAR);
            std::vector<double> chFeats = rapique_basic_extractor(y_scale);
            feats.insert(feats.end(), chFeats.begin(), chFeats.end());
        }
    }

    // The function returns the vector feats containing all the spatial features extracted from the image.

    return feats;
}