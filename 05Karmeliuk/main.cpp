#include <opencv2/highgui.hpp>
#include<opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp>
#include <iostream>

using namespace cv;

double *convolution(const Mat &in_image, double **mask, int ksize, double koef, Mat &out_image) {
    int middleK = ksize / 2;
    double average = 0;
    double *non_rounded = new double[in_image.rows * in_image.cols]{0};
    for (int i = middleK; i < in_image.rows - middleK; ++i) {
        for (int j = middleK; j < in_image.cols - middleK; ++j) {
            for (int k = 0; k < ksize; ++k) {
                for (int l = 0; l < ksize; ++l) {
                    average += in_image.at<uchar>(i + k - middleK, j + l - middleK) * mask[k][l] * koef;
                }
            }
            non_rounded[i * in_image.cols + j] = average;
            average = abs(average);
            if (average > 255)
                average = 255;
            out_image.at<uchar>(i, j) = cvRound(average);
            average = 0;
        }
    }
    return non_rounded;
};

double calculateMask(const int ksize, double **mask, double sigma) {
    double sum;
    int center = ksize / 2;
    double sigmaPow = sigma * sigma;
    for (int i = -center; i <= center; ++i) {
        for (int j = -center; j <= center; ++j) {
            mask[i + center][j + center] =
                    1. / (2 * M_PI * sigmaPow)
                    * exp(-((i * i + j * j) / (2 * sigmaPow)));
            sum += mask[i + center][j + center];
        }
    }
    return sum;
}

void Gaussian(const Mat &in_image, Mat &out_image, int ksize, double sigma) {
    double **mask;
    mask = new double *[ksize];
    for (int i = 0; i < ksize; ++i) {
        mask[i] = new double[ksize];
    }
    double sum = calculateMask(ksize, mask, sigma);

    convolution(in_image, mask, ksize, 1. / sum, out_image);
}

void SobelOperator(const Mat &in_image, Mat &out_image, double **gradPath) {
    double **Gx = new double *[3];
    Gx[0] = new double[3]{-1, 0, 1};
    Gx[1] = new double[3]{-2, 0, 2};
    Gx[2] = new double[3]{-1, 0, 1};


    double **Gy = new double *[3];
    Gy[0] = new double[3]{-1, -2, -1};
    Gy[1] = new double[3]{0, 0, 0};
    Gy[2] = new double[3]{1, 2, 1};
    double k = 1. / 2.;

    Mat outGx(in_image.clone());
    Mat outGy(in_image.clone());

    double *convX = convolution(in_image, Gx, 3, k, outGx);
//    imshow("Gx image", outGx);
    double *convY = convolution(in_image, Gy, 3, k, outGy);
//    imshow("Gy image", outGy);

    for (int i = 0; i < in_image.rows; ++i) {
        for (int j = 0; j < in_image.cols; ++j) {
            out_image.at<uchar>(i, j) = cvRound(
                    sqrt(
                            outGx.at<uchar>(i, j) * outGx.at<uchar>(i, j) +
                            outGy.at<uchar>(i, j) * outGy.at<uchar>(i, j)
                    )
            );
            gradPath[i][j] = atan2(convY[i * in_image.cols + j], convX[i * in_image.cols + j]);
        }
    }
}

void suppression(const Mat &in_image, Mat &out_image, double **gradPath) {
    double angle;
    uchar value;
    uchar value1, value2;
    for (int x = 1; x < in_image.rows - 1; x++) {
        for (int y = 1; y < in_image.cols - 1; y++) {
            angle = gradPath[x][y] * 180 / M_PI;
            if (angle < 0) {
                angle += 180;
            }
            value = in_image.at<uchar>(x, y);
            if (angle < 22.5 || angle >= 157.5) {
                value1 = in_image.at<uchar>(x, y + 1);
                value2 = in_image.at<uchar>(x, y - 1);
            } else if (angle >= 22.5 && angle < 67.5) {
                value1 = in_image.at<uchar>(x + 1, y - 1);
                value2 = in_image.at<uchar>(x - 1, y + 1);
            } else if (angle >= 67.5 && angle < 112.5) {
                value1 = in_image.at<uchar>(x + 1, y);
                value2 = in_image.at<uchar>(x - 1, y);
            } else if (angle >= 112.5 && angle < 157.5) {
                value1 = in_image.at<uchar>(x - 1, y - 1);
                value2 = in_image.at<uchar>(x + 1, y + 1);
            }

            if (value >= value1 && value >= value2) {
                out_image.at<uchar>(x, y) = value;
            } else {
                out_image.at<uchar>(x, y) = 0;
            }

        }
    }
}

void threshold(const Mat &in_image, Mat &out_image, double lowThreshold, double highThreshold) {
    uchar value;
    for (int i = 0; i < in_image.rows; ++i) {
        for (int j = 0; j < in_image.cols; ++j) {
            value = in_image.at<uchar>(i, j);
            if (value < lowThreshold) {
                out_image.at<uchar>(i, j) = 0;
            } else if (value > highThreshold) {
                out_image.at<uchar>(i, j) = 255;
            } else {
                out_image.at<uchar>(i, j) = 90;
            }

        }
    }
}

void hysteresis(const Mat &in_image, Mat &out_image) {
    uchar value;
    for (int i = 1; i < in_image.rows - 1; ++i) {
        for (int j = 1; j < in_image.cols - 1; ++j) {
            value = in_image.at<uchar>(i, j);
            if (value == 90) {
                if (in_image.at<uchar>(i - 1, j - 1) == 255 || in_image.at<uchar>(i, j - 1) == 255 ||
                    in_image.at<uchar>(i + 1, j - 1) == 255 ||
                    in_image.at<uchar>(i - 1, j) == 255 || in_image.at<uchar>(i + 1, j) == 255 ||
                    in_image.at<uchar>(i - 1, j + 1) == 255 ||
                    in_image.at<uchar>(i, j + 1) == 255 || in_image.at<uchar>(i + 1, j + 1) == 255) {
                    out_image.at<uchar>(i, j) = 255;
                } else {
                    out_image.at<uchar>(i, j) = 0;
                }
            }
        }
    }
}

int main() {
    Mat img = imread("../img.png", IMREAD_GRAYSCALE);
//    Mat img = imread("../sebek.jpg", IMREAD_GRAYSCALE);
//    Mat img = imread("../dandelion.jpeg", IMREAD_GRAYSCALE);
//    Mat img = imread("../chair.tif", IMREAD_GRAYSCALE);
    Mat filteredIMG(img.clone());


    double **gradPath = new double *[img.rows];
    for (int i = 0; i < img.rows; i++) {
        gradPath[i] = new double[img.cols]{0};
    }

    if (img.data == nullptr) {
        std::cout << "Could not open or find the image" << std::endl;
        return -1;
    }

    Gaussian(img, filteredIMG, 5, 1.1);

    Mat gradientIMG(filteredIMG.clone());
    SobelOperator(filteredIMG, gradientIMG, gradPath);

    Mat suppressedIMG(gradientIMG.clone());
    suppression(gradientIMG, suppressedIMG, gradPath);

    Mat thresholdedIMG(gradientIMG.clone());
    threshold(suppressedIMG, thresholdedIMG, 5, 15);

    Mat HysteresisIMG(thresholdedIMG.clone());
    hysteresis(thresholdedIMG, HysteresisIMG);

    imshow("Original image", img);
    imshow("Filtered image", filteredIMG);
    imshow("Gradient image", gradientIMG);
    imshow("Suppressed image", suppressedIMG);
    imshow("Threshold image", thresholdedIMG);
    imshow("Hysteresis image", HysteresisIMG);

    Mat CVCanny(img.clone());
    cv::GaussianBlur(CVCanny, CVCanny, Size (5,5), 1.5);
    Canny(img, CVCanny, 30, 70, 3);
    imshow("CV Canny image", CVCanny);
    waitKey(0);
//    std::cin.get();
    return 0;
}
