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

std::vector<Point> boundaryTracing(Mat &img, Mat &out) {
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

    std::vector<Point> contour;
    Point shifted, current, prev;
    current = Point(startX, startY);
    out.at<uchar>(current.y, current.x) = 255;
    prev = Point(startX - 1, startY);
    int i;
    do {
        contour.push_back(current);
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
    return contour;
}

//double find_degree(const Point &x1, const Point &x2) {
//    return 0;
//}

void contourCurve(Mat &img, int k=1) {
    Mat boundary = Mat::zeros(img.rows, img.cols, img.type());
    std::vector<Point> contour = boundaryTracing(img, boundary);
    int contourSize = contour.size();
    Point prev, current, next, fv, bv;
    double d_b, d_f, deg_f, deg_b, deg_avg, diff_f;
    imshow("Contour", boundary);
    std::vector<double> result;
    for (int i = k; i < contourSize - k; ++i) {
        prev = contour[i - k];
        current = contour[i];
        next = contour[i + k];
        fv = Point(current.x - next.x, current.y - next.y);
        bv = Point(current.x - prev.x, current.y - prev.y);
//        std::cout << prev <<" " << current<< " " << next << " " <<  result[i-1]<< std::endl;
//        std::cout << fv << " " << bv << std::endl;
        d_f = sqrt(fv.x * fv.x + fv.y * fv.y);
        d_b = sqrt(bv.x * bv.x + bv.y * bv.y);
        deg_f = atan2(abs(fv.x), abs(fv.y));
        deg_b = atan2(abs(bv.x), abs(bv.y));
        deg_avg = (deg_f + deg_b) / 2;
        diff_f = deg_f - deg_avg;
        result.push_back(diff_f * (d_b + d_f) / (2 * d_b * d_f));
    }
    int hist_h = 512;
    int hist_w = result.size();
    Mat histogram = Mat(hist_h, hist_w, CV_8UC3);
    for (int i = 0; i < result.size(); i++) {
        line(histogram,
             Point(i + 1, hist_h / 2),
             Point(i + 1, hist_h / 2 + result[i] * 700),
             Scalar(255, 255, 255), 1, LINE_8, 0
        );
    }
    imshow("Histogram", histogram);
}

int main() {
//    Mat img = imread("../test.png", IMREAD_GRAYSCALE);
//    Mat img = imread("../moon.png", IMREAD_GRAYSCALE);
//    Mat img = imread("../triangle.png", IMREAD_GRAYSCALE);
//    Mat img = imread("../wierd.png", IMREAD_GRAYSCALE);
//    Mat img = imread("../contour.tif", IMREAD_GRAYSCALE);
    Mat img = imread("../rect.png", IMREAD_GRAYSCALE);

    Mat binarized(img.clone());

    if (img.data == nullptr) {
        std::cout << "Could not open or find the image" << std::endl;
        return -1;
    }
    binarize(binarized, 125);
//    imshow("Original", img);
//    imshow("Binarized", binarized);
    contourCurve(binarized);

    waitKey(0);
//    std::cin.get();
    return 0;
}
