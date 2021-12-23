#include <opencv2/highgui.hpp>
#include<opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp>
#include <iostream>

using namespace cv;


void dilatation(Mat &in_img, Mat &out_img, const double *structured, int k) {
    double max = 0;
    int center = k / 2;
    for (int i = center; i < in_img.rows - center; ++i) {
        for (int j = center; j < in_img.cols - center; ++j) {
            for (int m = 0; m < k; ++m) {
                for (int l = 0; l < k; ++l) {
                    if (structured[l * k + m] == 1) {
                        uchar value = in_img.at<uchar>(i + l - center, j + m - center) *
                                      structured[l * k + m];

                        if (value > max) {
                            max = value;
                        }
                    }
                }
            }
            out_img.at<uchar>(i, j) = max;
            max = 0;
        }
    }
}

void erosion(Mat &in_img, Mat &out_img, const double *structured, int k) {
    double min = 255;
    int center = k / 2;
    for (int i = center; i < in_img.rows - center; ++i) {
        for (int j = center; j < in_img.cols - center; ++j) {
            for (int m = 0; m < k; ++m) {
                for (int l = 0; l < k; ++l) {
                    if (structured[l * k + m] == 1) {
                        uchar value = in_img.at<uchar>(i + l - center, j + m - center) *
                                      structured[l * k + m];
                        if (value < min) {
                            min = value;
                        }
                    }
                }
            }
            out_img.at<uchar>(i, j) = min;
            min = 255;
        }
    }
};

void difference(Mat &in_img1, Mat &in_img2, Mat &out_img) {

    for (int i = 0; i < in_img1.rows; ++i) {
        for (int j = 0; j < in_img1.cols; ++j) {
            out_img.at<uchar>(i, j) = in_img1.at<uchar>(i, j) - in_img2.at<uchar>(i, j);
        }
    }
};

int main() {
//    Mat img = imread("../img.png", IMREAD_GRAYSCALE);
    Mat img = imread("../text.png", IMREAD_GRAYSCALE);
//    Mat img = imread("../contour.tif", IMREAD_GRAYSCALE);
//    Mat img = imread("../sebek.jpg", IMREAD_GRAYSCALE);
//    Mat img = imread("../chair.tif", IMREAD_GRAYSCALE);

    Mat dilImage(img.clone());
    Mat erosImage(img.clone());
    Mat differenceImage(img.clone());
    double structured[9] = {
            0, 1, 0,
            1, 1, 1,
            0, 1, 0
    };
    if (img.data == nullptr) {
        std::cout << "Could not open or find the image" << std::endl;
        return -1;
    }
    dilatation(img, dilImage, structured, 3);
    erosion(img, erosImage, structured, 3);
    difference(dilImage, erosImage, differenceImage);
    imshow("Original", img);
    imshow("Dilatation", dilImage);
    imshow("Erosion", erosImage);
    imshow("Difference", differenceImage);
    waitKey(0);
//    std::cin.get();
    return 0;
}
