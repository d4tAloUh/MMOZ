#include <opencv2/highgui.hpp>
#include<opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp>
#include <iostream>
#include <stack>

using namespace cv;
const Point pointShift[8] =
        {
                Point(-1, -1),Point(0, -1),Point(1, -1),
                Point(-1, 0),Point(1, 0),
                Point(-1, 1), Point(0, 1),Point(1, 1),
        };
//const Point pointShift[4] =
//        {
//                Point(0, -1),
//                Point(-1, 0),
//                Point(1, 0),
//                Point(0, 1),
//        };

void label(Mat &in_img, Mat &segments, const Point &seed, int label) {
    std::stack<Point> queue;
    queue.push(seed);
    Point shifted;
    Point current;
    while (!queue.empty()) {
        current = queue.top();
        queue.pop();
        segments.at<uchar>(current.y, current.x) = label;
        for (int i = 0; i < 8; ++i) {
            shifted = current + pointShift[i];
//          Skip out of bounds
            if (shifted.x < 0 || shifted.y < 0 ||
                shifted.y >= in_img.rows ||
                shifted.x >= in_img.cols)
                continue;

            if (in_img.at<uchar>(shifted.y, shifted.x) == 255 &&
                    segments.at<uchar>(shifted.y, shifted.x) == 0) {
                queue.push(shifted);
            }
        }
    }
}

void printMatrix(Mat &in_img){
    for (int i = 0; i < in_img.rows; ++i) {
        for (int j = 0; j < in_img.cols; ++j) {
            std::cout << int(in_img.at<uchar>(i, j)) << " ";
        }
        std::cout << std::endl;
    }
}

void find_components(Mat &in_img, Mat &segments) {
    int mark = 10;
    for (int i = 0; i < in_img.rows; ++i) {
        for (int j = 0; j < in_img.cols; ++j) {
            if (in_img.at<uchar>(i, j) == 255 && segments.at<uchar>(i, j) == 0) {
                label(in_img, segments, Point(j, i), mark);
                if (mark > 255){
                    mark = 10;
                }
                mark += 10;
//                printMatrix(mask);
//                std::cout << std::endl << std::endl;
            }
        }
    }
}


void convertToBinary(Mat &in_img) {
    for (int i = 0; i < in_img.rows; ++i) {
        for (int j = 0; j < in_img.cols; ++j) {
            if (in_img.at<uchar>(i, j) >= 110) {
                in_img.at<uchar>(i, j) = 255;
            } else {
                in_img.at<uchar>(i, j) = 0;
            }
        }
    }
}


int main() {
    Mat img = imread("../img.png", IMREAD_GRAYSCALE);
//    Mat img = imread("../im_kola.png", IMREAD_GRAYSCALE);
//    Mat img = imread("../rsz_im_kola.png", IMREAD_GRAYSCALE);
//    Mat img = imread("../scenery.png", IMREAD_GRAYSCALE);
//    Mat img = imread("../coins.jpg", IMREAD_GRAYSCALE);
//    Mat img = imread("../chair.tif", IMREAD_GRAYSCALE);
    Mat test(img.clone());
    convertToBinary(test);
    Mat segments = Mat::zeros(img.rows, img.cols, img.type());
    if (img.data == nullptr) {
        std::cout << "Could not open or find the image" << std::endl;
        return -1;
    }

    find_components(test, segments);
    imshow("Mask", segments);
    imshow("Original", img);
    imshow("Binary", test);
    waitKey(0);
//    std::cin.get();
    return 0;
}
