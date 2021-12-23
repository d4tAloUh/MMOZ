#include <opencv2/highgui.hpp>
#include<opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp>
#include <iostream>

using namespace cv;

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

double calculateSum(double** mask, int ksize){
    double result = 0;
    for (int i = 0; i < ksize; ++i) {
        for (int j = 0; j < ksize; ++j) {
            result += mask[i][j];
        }
    }
    return result;
}

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
            out_image.at<uchar>(i, j) = cvRound(average);
            average = 0;
        }
    }
};

void GaussianBlur(const Mat &in_image, Mat &out_image, int ksize, double sigma) {
    double **mask;
    mask = new double *[ksize];
    for (int i = 0; i < ksize; ++i) {
        mask[i] = new double[ksize];
    }
    double sum = calculateMask(ksize, mask, sigma);

    convolution(in_image, mask, ksize, 1./sum, out_image);
}

int main() {
    Mat img = imread("../sebek.jpg", IMREAD_GRAYSCALE);
//    Mat img = imread("../chair.tif", IMREAD_GRAYSCALE);
    Mat outImg(img.clone());
    Mat convolutionImg(img.clone());


    if (img.data == nullptr)  {
        std::cout << "Could not open or find the image" << std::endl;
        return -1;
    }
    int ksize = 11;
    double sigma = 0.5;
    GaussianBlur(img, outImg, ksize, sigma);

    int kernelSize = 5;
    double **kernel = new double *[kernelSize];
    kernel[0] = new double[kernelSize]{1, 4, 7, 4, 1};
    kernel[1] = new double[kernelSize]{4, 16, 26, 16, 4};
    kernel[2] = new double[kernelSize]{7, 26, 41, 26, 7};
    kernel[3] = new double[kernelSize]{4, 16, 26, 16, 4};
    kernel[4] = new double[kernelSize]{1, 4, 7, 4, 1};


//    double **kernel = new double *[kernelSize];
//    kernel[0] = new double[kernelSize]{-1., 0., -1.};
//    kernel[1] = new double[kernelSize]{0., -4., 0.};
//    kernel[2] = new double[kernelSize]{-1, 0., -1.};
    double koef = calculateSum(kernel, kernelSize);
    convolution(img, kernel, kernelSize, 1./koef, convolutionImg);

    imshow("Original image", img);
    imshow("Blurred image", outImg);
    imshow("Convolution image", convolutionImg);

    waitKey(0);
//    std::cin.get();
    return 0;
}
