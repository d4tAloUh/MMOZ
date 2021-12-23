#include <opencv2/highgui.hpp>
#include<opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp>
#include <iostream>

using namespace cv;

void convolution(const Mat &in_image, double **mask, int ksize, double koef, Mat &out_image) {
    int middleK = ksize / 2;
    double average = 0;

    for (int i = middleK; i < in_image.rows - middleK; ++i) {
        for (int j = middleK; j < in_image.cols - middleK; ++j) {
            for (int k = 0; k < ksize; ++k) {
                for (int l = 0; l < ksize; ++l) {
                    average += in_image.at<uchar>(i + k - middleK, j + l - middleK) * mask[k][l] * koef;
                }
            }
            average = abs(average);
            if (average > 255)
                average = 255;
            out_image.at<uchar>(i, j) = cvRound(average);
            average = 0;
        }
    }
};

void SobelOperator(const Mat &in_image, Mat &out_image, bool abs = false) {
    double **Gx = new double *[3];
    Gx[0] = new double[3]{-1, 0, 1};
    Gx[1] = new double[3]{-2, 0, 2};
    Gx[2] = new double[3]{-1, 0, 1};


    double **Gy = new double *[3];
    Gy[0] = new double[3]{-1, -2, -1};
    Gy[1] = new double[3]{0, 0, 0};
    Gy[2] = new double[3]{1, 2, 1};
    double k = 1. / 4.;

    Mat outGx(in_image.clone());
    Mat outGy(in_image.clone());

    convolution(in_image, Gx, 3, k, outGx);
//    imshow("Gx image", outGx);
    convolution(in_image, Gy, 3, k, outGy);
//    imshow("Gy image", outGy);

    for (int i = 0; i < in_image.rows; ++i) {
        for (int j = 0; j < in_image.cols; ++j) {
            if (abs) {
                out_image.at<uchar>(i, j) = cvRound(fabs(
                        outGx.at<uchar>(i, j) +
                        outGy.at<uchar>(i, j)
                ));
            } else {
                out_image.at<uchar>(i, j) = sqrt(
                        outGx.at<uchar>(i, j) * outGx.at<uchar>(i, j) +
                        outGy.at<uchar>(i, j) * outGy.at<uchar>(i, j)
                );
            }
        }
    }
};

int main() {
    Mat img = imread("../img.png", IMREAD_GRAYSCALE);
//    Mat img = imread("../sebek.jpg", IMREAD_GRAYSCALE);
//    Mat img = imread("../chair.tif", IMREAD_GRAYSCALE);
    Mat outImg(img.clone());
    Mat outImgABS(img.clone());
    Mat outImgCV(img.clone());
    Mat outImgCVX(img.clone());
    Mat outImgCVY(img.clone());


    if (img.data == nullptr) {
        std::cout << "Could not open or find the image" << std::endl;
        return -1;
    }

    SobelOperator(img, outImg);
    SobelOperator(img, outImgABS, true);
    imshow("Original image", img);
    imshow("ABS image", outImgABS);
    imshow("Out image", outImg);

//  CV PART
    Sobel(img, outImgCVX, CV_32F, 1, 0);
    Sobel(img, outImgCVY, CV_32F, 0, 1);
    convertScaleAbs(outImgCVX, outImgCVX);
    convertScaleAbs(outImgCVY, outImgCVY);
    addWeighted(outImgCVX, 0.5, outImgCVY, 0.5, 0, outImgCV);
//    imshow("CV image", outImgCV);

    waitKey(0);
//    std::cin.get();
    return 0;
}
