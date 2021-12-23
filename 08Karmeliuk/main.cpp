#include <opencv2/highgui.hpp>
#include<opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp>
#include <iostream>
#include <stack>

using namespace cv;

void binarize(Mat &img, int threshold) {
    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            if (img.at<uchar>(i, j) < threshold) {
                img.at<uchar>(i, j) = 0;
            } else {
                img.at<uchar>(i, j) = 255;
            }
        }
    }
}

void calculateDistribution(int dist[], const Mat &img) {
    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            dist[img.at<uchar>(i, j)]++;
        }
    }
}

int otsuMethod(Mat &in_img) {
    int dist[256] = {0};
    calculateDistribution(dist, in_img);

    int sum = 0;
    int n = 0;
    for (int t = 0; t < 256; t++) {
        sum += t * dist[t];
        n += dist[t];
    }
    int threshold = 0;
    float maxSigma = -1;
    int q1 = 0;
    int q2;
    int sumB = 0;
    float u1, u2, sigma;
    for (int t = 0; t < 256; t++) {
        q1 += dist[t];
        if (q1 == 0)
            continue;
        q2 = n - q1;

        sumB += t * dist[t];

        u1 = (float) sumB / (float) q1;
        u2 = (float) (sum - sumB) / (float) q2;

        sigma = q1 * q2 * (u1 - u2) * (u1 - u2);

        if (sigma > maxSigma) {
            threshold = t;
            maxSigma = sigma;
        }
    }

    return threshold;
}


int main() {
//    Mat img = imread("../img.png", IMREAD_GRAYSCALE);
//    Mat img = imread("../coins.jpg", IMREAD_GRAYSCALE);
//    Mat img = imread("../print.jpg", IMREAD_GRAYSCALE);
//    Mat img = imread("../chair.tif", IMREAD_GRAYSCALE);
    Mat img = imread("../otsu.jpg", IMREAD_GRAYSCALE);

    Mat binarized(img.clone());

    if (img.data == nullptr) {
        std::cout << "Could not open or find the image" << std::endl;
        return -1;
    }

    int result = otsuMethod(img);
    binarize(binarized, result);

    std::cout << result << std::endl;

    imshow("Original", img);
    imshow("Binarized", binarized);

    waitKey(0);
//    std::cin.get();
    return 0;
}
