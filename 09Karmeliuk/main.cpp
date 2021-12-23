#include <opencv2/highgui.hpp>
#include<opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp>
#include <iostream>

using namespace cv;

//const Point pointShift[8] =
//        {
//                Point(-1, -1), Point(0, -1), Point(1, -1),
//                Point(-1, 0),                       Point(1, 0),
//                Point(-1, 1),  Point(0, 1),  Point(1, 1),
//        };

const Point pointShift[8] =
        {
                Point(-1, -1), Point(0, -1),
                Point(1, -1), Point(1, 0),
                Point(1, 1), Point(0, 1),
                Point(-1, 1), Point(-1, 0),
        };

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

void boundaryTracing(Mat &img, Mat &out) {
    int startX = -1, startY = -1;

    for (int i = 0; i < img.rows; ++i) {
        if (startX != -1 || startY != -1)
            break;
        for (int j = 0; j < img.cols; ++j) {
            if (img.at<uchar>(i, j) == 255) {
                startY = i;
                startX = j;
                break;
            }
        }
    }


    Point shifted, current, prev;
    current = Point(startX, startY);
    out.at<uchar>(current.y, current.x) = 255;
    prev = Point(startX - 1, startY);
    int i;
    do {
        i = 0;
        while (shifted != prev) {
            shifted = current + pointShift[i];
            i = (i + 1) % 8;
        }

        for (int j = 0; j < 8; ++j) {
            shifted = current + pointShift[(i + j) % 8];
            if (img.at<uchar>(shifted.y, shifted.x) == 255) {
                out.at<uchar>(shifted.y, shifted.x) = 255;
                prev = current + pointShift[(i + j + 7) % 8];
                current = shifted;
                break;
            }
        }
    } while (current.y != startY || current.x != startX);
}

int main() {
//    Mat img = imread("../test.png", IMREAD_GRAYSCALE);
//    Mat img = imread("../moon.png", IMREAD_GRAYSCALE);
    Mat img = imread("../triangle.png", IMREAD_GRAYSCALE);
//    Mat img = imread("../wierd.png", IMREAD_GRAYSCALE);

    Mat binarized(img.clone());

    if (img.data == nullptr) {
        std::cout << "Could not open or find the image" << std::endl;
        return -1;
    }
    binarize(binarized, 125);
    imshow("Original", img);
    imshow("Binarized", binarized);
    Mat boundary = Mat::zeros(img.rows, img.cols, img.type());
    boundaryTracing(binarized, boundary);

    imshow("Result", boundary);

    waitKey(0);
//    std::cin.get();
    return 0;
}
