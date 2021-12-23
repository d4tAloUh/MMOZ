#include <opencv2/highgui.hpp>
#include<opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp>
#include <iostream>

using namespace cv;

void decrease(const Mat &in_image, Mat &out_image, int n = 2) {
    int rows = in_image.rows;
    int cols = in_image.cols;
    out_image = Mat(rows / n, cols / n, CV_8UC3);
    Vec3b pixel1, pixel2, pixel3, pixel4, average_pixel;
    for (int i = 0; i < rows / 2; ++i) {
        for (int j = 0; j < cols / 2; ++j) {
            pixel1 = in_image.at<Vec3b>(i * 2 + 1, j * 2 + 1);
            pixel2 = in_image.at<Vec3b>(i * 2, j * 2 + 1);
            pixel3 = in_image.at<Vec3b>(i * 2 + 1, j * 2);
            pixel4 = in_image.at<Vec3b>(i * 2, j * 2);
//          find average color
            average_pixel = Vec3b(
                    (pixel1[0] + pixel2[0] + pixel3[0] + pixel4[0] )/ 4,
                    (pixel1[1] + pixel2[1] + pixel3[1] + pixel4[1] )/ 4,
                    (pixel1[2] + pixel2[2] + pixel3[2] + pixel4[2] )/ 4
                    );
            out_image.at<Vec3b>(i, j) = average_pixel;
        }
    }
}

void increase(const Mat &in_image, Mat &out_image, int n = 2) {
    int rows = in_image.rows;
    int cols = in_image.cols;
    out_image = Mat(rows * n, cols * n, CV_8UC3);
    Vec3b pixel;
    for (int i = 1; i < rows; ++i) {
        for (int j = 1; j < cols; ++j) {
            pixel = in_image.at<Vec3b>(i - 1, j - 1);
            out_image.at<Vec3b>(i * 2 - 1, j * 2 - 1) = pixel;
            out_image.at<Vec3b>(i * 2, j * 2 - 1) = pixel;
            out_image.at<Vec3b>(i * 2 - 1, j * 2) = pixel;
            out_image.at<Vec3b>(i * 2, j * 2) = pixel;
        }
    }
}

int main() {
    Mat img = imread("../sebek.jpg", IMREAD_COLOR);
    if (img.data == nullptr) {
        std::cout << "Could not open or find the image" << std::endl;
        return -1;
    }
    Mat decreasedImg;
    Mat increasedImg;
    increase(img, increasedImg);
    decrease(img, decreasedImg);

    imshow("Original image", img);
    imshow("Decreased image", decreasedImg);
    imshow("Increased image", increasedImg);
    waitKey(0);
    return 0;
}
