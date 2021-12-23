#include <opencv2/highgui.hpp>
#include<opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp>
#include <iostream>

using namespace cv;
using namespace std;

const double qualityLevel = 0.04;
const double minDistance = 10;
const int maxCorners = 150;

Mat filterR(Mat &R, Mat &img, int wsize) {
    Mat result = Mat::zeros(R.size(), CV_64F);
    for (int i = wsize / 2; i < (R.rows - wsize / 2); i++)
        for (int j = wsize / 2; j < (R.cols - wsize / 2); j++) {
            double origin = R.at<double>(i, j);
            bool found = false;
            for (int ii = i - wsize / 2; ii <= (i + wsize / 2) && !found; ii++)
                for (int jj = j - wsize / 2; jj <= (j + wsize / 2); jj++)
                    if (origin < R.at<double>(ii, jj)) {
                        origin = 0;
                        found = true;
                        break;
                    }
            if (origin == 0)
                result.at<double>(i, j) = 0;
            else {
                result.at<double>(i, j) = 255;
                circle(img, Point(j, i), 5, Scalar(0, 0, 255), 2, 8, 0);
            }
        }

    return result;
}

void takasi_sobel(Mat &in_img) {
    Mat blurred(in_img.clone());
    GaussianBlur(in_img, blurred, Size(5, 5), 0.3);
//    imshow("in_img", in_img);
//    imshow("blurred", blurred);

//  derivatives
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    int ddepth = CV_32F;
    int ksize = 3;
    Sobel(blurred, grad_x, ddepth, 1, 0, ksize);
    Sobel(blurred, grad_y, ddepth, 0, 1, ksize);
    convertScaleAbs(grad_x, abs_grad_x);
    convertScaleAbs(grad_y, abs_grad_y);

//    imshow("grad_x", abs_grad_x);
//    imshow("grad_y", abs_grad_y);

//  multiply
    Mat pxy(abs_grad_x.rows, abs_grad_x.cols, CV_64FC1);
    Mat px(abs_grad_x.rows, abs_grad_x.cols, CV_64FC1);
    Mat py(abs_grad_x.rows, abs_grad_x.cols, CV_64FC1);
    for (int i = 0; i < abs_grad_x.rows; ++i) {
        for (int j = 0; j < abs_grad_x.cols; ++j) {
            px.at<uchar>(i, j) = abs_grad_x.at<uchar>(i, j) * abs_grad_x.at<uchar>(i, j);
            py.at<uchar>(i, j) = abs_grad_y.at<uchar>(i, j) * abs_grad_y.at<uchar>(i, j);
            pxy.at<uchar>(i, j) = abs_grad_x.at<uchar>(i, j) * abs_grad_y.at<uchar>(i, j);
        }
    }
//    imshow("grad_xy", pxy);
//    imshow("grad_x", px);
//    imshow("grad_y", py);


    Mat Ixy(abs_grad_x.rows, abs_grad_x.cols, CV_64FC1);
    Mat Ix(abs_grad_x.rows, abs_grad_x.cols, CV_64FC1);
    Mat Iy(abs_grad_x.rows, abs_grad_x.cols, CV_64FC1);
    double sigma = 1.0;
    GaussianBlur(pxy, Ixy, Size(5, 5), sigma);
    GaussianBlur(px, Ix, Size(5, 5), sigma);
    GaussianBlur(py, Iy, Size(5, 5), sigma);
    double r, trace, det;
    double k = 0.15;
//    double abs_threshold = 2500000;
    int window = 0;
    int offset = window / 2;
//    Mat result(in_img.clone());
//    cvtColor(result, result, COLOR_GRAY2RGB);
    Mat result = Mat::zeros(in_img.size(), CV_64FC1);
    double sumX, sumY, sumXY;
//  calculate harris value
    for (int i = offset; i < in_img.rows - offset; ++i) {
        for (int j = offset; j < in_img.cols - offset; ++j) {
            sumX = 0, sumY = 0, sumXY = 0;
            for (int l = -offset; l <= offset; ++l) {
                for (int m = -offset; m <= offset; ++m) {
                    sumX += Ix.at<uchar>(i + l, j + m);
                    sumY += Iy.at<uchar>(i + l, j + m);
                    sumXY += Ixy.at<uchar>(i + l, j + m);
                }
            }
            trace = (sumX + sumY);
            det = sumX * sumY - sumXY * sumXY;
            r = det - k * (trace * trace);
//            r = (sumX + sumY - sqrt((sumX - sumY) * (sumX - sumY) + 4 * sumXY * sumXY)) / 2;
//            r = (trace - sqrt(trace * trace + 4 * det * det)) / 2;
//            if (r > abs_threshold) {
            result.at<double>(i, j) = r;
//            }
        }
    }

    Mat res = filterR(result, in_img, 10);
    imshow("Result after filter", res);
    imshow("orig", in_img);
}

double euclidianDistance(int x1, int x2, int y1, int y2) {
    return sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}

bool comp(Point3d &first, Point3d &second) {
    return first.z >= second.z;
}

vector<Point3d> selectBest(vector<Point3d> &corners, double maxDistance) {
    vector<Point3d> bestCorners;
    sort(corners.begin(), corners.end(), comp);
    bestCorners.push_back(corners.at(0));
    bool flag;
    for (auto &corner: corners) {
        flag = false;
        for (auto &bestCorner: bestCorners) {
            if (euclidianDistance(corner.x, bestCorner.x, corner.y, bestCorner.y) < maxDistance) {
                flag = true;
                break;
            }
        }
        if (!flag) {
            bestCorners.push_back(corner);
        }
    }
    return bestCorners;
}

void convolution(const Mat &in_image, double **mask, int ksize, Mat &out_image) {
    int middleK = ksize / 2;
    double average;
    for (int i = middleK; i < in_image.rows - middleK; ++i) {
        for (int j = middleK; j < in_image.cols - middleK; ++j) {
            average = 0;
            for (int k = 0; k < ksize; ++k) {
                for (int l = 0; l < ksize; ++l) {
                    average += in_image.at<uchar>(i - middleK + k, j - middleK + l) * mask[k][l];
                }
            }
            out_image.at<uchar>(i, j) = abs(average);
        }
    }
};

void calculateMask(const int ksize, double **mask, double sigma) {
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
    for (int i = -center; i <= center; ++i) {
        for (int j = -center; j <= center; ++j) {
            {
                mask[i + center][j + center] /= sum;
            }
        }
    }
}

void tomasi_test(Mat &in_img) {
    auto **Gx = new double *[3];
    Gx[0] = new double[3]{-1, 0, 1};
    Gx[1] = new double[3]{-2, 0, 2};
    Gx[2] = new double[3]{-1, 0, 1};


    auto **Gy = new double *[3];
    Gy[0] = new double[3]{-1, -2, -1};
    Gy[1] = new double[3]{0, 0, 0};
    Gy[2] = new double[3]{1, 2, 1};

//  find Sobel
    Mat outGx(in_img.clone());
    Mat outGy(in_img.clone());
    convolution(in_img, Gx, 3, outGx);
    convolution(in_img, Gy, 3, outGy);



    Mat imageSobelX = Mat::zeros(in_img.size(), CV_64F);
    Mat imageSobelY = Mat::zeros(in_img.size(), CV_64F);
    convertScaleAbs(outGx, imageSobelX);
    convertScaleAbs(outGy, imageSobelY);

    Mat XX = Mat::zeros(in_img.size(), CV_64F);
    Mat YY = Mat::zeros(in_img.size(), CV_64F);
    Mat XY = Mat::zeros(in_img.size(), CV_64F);

    for (int i = 0; i < imageSobelX.rows; i++) {
        for (int j = 0; j < imageSobelX.cols; j++) {
            XX.at<double>(i, j) = imageSobelX.at<uchar>(i, j) * imageSobelX.at<uchar>(i, j);
            YY.at<double>(i, j) = imageSobelY.at<uchar>(i, j) * imageSobelY.at<uchar>(i, j);
            XY.at<double>(i, j) = imageSobelX.at<uchar>(i, j) * imageSobelY.at<uchar>(i, j);
        }
    }

    double **mask;
    int ksize = 5;
    mask = new double *[ksize];
    for (int i = 0; i < ksize; ++i) {
        mask[i] = new double[ksize];
    }
    calculateMask(ksize, mask, 1.0);

//    double k = 0.05;
//    int threshold = 50000000;

    int threshold = 100;
    int offset = ksize / 2;
    double a, b, c, det, trace;
    Mat resultData = Mat(XX.size(), CV_64F);
    for (int i = offset; i < resultData.rows - offset; i++) {
        for (int j = offset; j < resultData.cols - offset; j++) {
            a = 0;
            b = 0;
            c = 0;
            for (int m = -offset; m <= offset; m++) {
                for (int n = -offset; n <= offset; n++) {
                    a += XX.at<double>(i + m, j + n) * mask[offset + m][offset + n];
                    c += YY.at<double>(i + m, j + n) * mask[offset + m][offset + n];
                    b += XY.at<double>(i + m, j + n) * mask[offset + m][offset + n];
                }
            }
            det = a * b - c * c;
            trace = (a + b);
//            resultData.at<double>(i, j) = det - k * trace * trace;
            resultData.at<double>(i, j) = (trace - sqrt(trace * trace - 4 * det)) / 2;
        }
    }

    vector<Point3d> corners;
    Mat ResultImage = in_img.clone();

    cvtColor(ResultImage, ResultImage, COLOR_GRAY2RGB);
    for (int i = 1; i < ResultImage.rows - 1; i++) {
        for (int j = 1; j < ResultImage.cols - 1; j++) {
            if (resultData.at<double>(i, j) > resultData.at<double>(i - 1, j - 1) &&
                resultData.at<double>(i, j) > resultData.at<double>(i - 1, j) &&
                resultData.at<double>(i, j) > resultData.at<double>(i - 1, j - 1) &&
                resultData.at<double>(i, j) > resultData.at<double>(i - 1, j + 1) &&
                resultData.at<double>(i, j) > resultData.at<double>(i, j - 1) &&
                resultData.at<double>(i, j) > resultData.at<double>(i, j + 1) &&
                resultData.at<double>(i, j) > resultData.at<double>(i + 1, j - 1) &&
                resultData.at<double>(i, j) > resultData.at<double>(i + 1, j) &&
                resultData.at<double>(i, j) > resultData.at<double>(i + 1, j + 1)) {
                if (resultData.at<double>(i, j) > threshold) {
                    circle(ResultImage, Point(j, i), 4,
                           Scalar(0, 125, 255), 2, 8, 0);
                    corners.emplace_back(j, i, resultData.at<double>(i, j));
                }
            }
        }
    }
    Mat bestCornersIMG = in_img.clone();
    cvtColor(bestCornersIMG, bestCornersIMG, COLOR_GRAY2RGB);
    vector<Point3d> bestCorners = selectBest(corners, 15);
    for (auto & bestCorner : bestCorners) {
        circle(bestCornersIMG, Point(bestCorner.x, bestCorner.y),
               4, Scalar(0, 125, 255), 2, 8, 0);
    }

    imshow("best takasi", bestCornersIMG);
    imshow("result takasi", ResultImage);
}

void tomasi_cv(Mat &in_img) {
    Mat result(in_img.clone());
    std::vector<cv::Point2f> corners;
    int blockSize = 3;
    bool useHarrisDetector = false;

    goodFeaturesToTrack(in_img,
                        corners,
                        maxCorners,
                        qualityLevel,
                        minDistance,
                        Mat(),
                        blockSize,
                        useHarrisDetector);

    cvtColor(result, result, COLOR_GRAY2RGB);
    for (auto &corner: corners) {
        circle(result, corner, 4, Scalar(0, 125, 255), 2, 8, 0);
    }

    imshow("Tokasi CV", result);
}

void harris_cv(Mat &in_img) {
    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;

    Mat result(in_img.clone());
    cvtColor(result, result, COLOR_GRAY2RGB);
    Mat dst = Mat::zeros(in_img.size(), CV_32FC1);
    cornerHarris(in_img, dst, blockSize, apertureSize, k);
    double max = -999999999999999;
    for (int i = 0; i < dst.rows; i++) {
        for (int j = 0; j < dst.cols; j++) {
            if (dst.at<float>(i, j) > max) {
                max = dst.at<float>(i, j);
            }
        }
    }

    max = max * 0.01;
    for (int i = 0; i < dst.rows; i++) {
        for (int j = 0; j < dst.cols; j++) {
            if (dst.at<float>(i, j) > max) {
                circle(result, Point(j, i), 5, Scalar(0, 125, 255), 2, 8, 0);
            }
        }
    }

    imshow("harris cv", result);
}


int main() {
//    Mat img = imread("../test.png", IMREAD_GRAYSCALE);
//    Mat img = imread("../1.jpg", IMREAD_GRAYSCALE);
//    Mat img = imread("../chair.tif", IMREAD_GRAYSCALE);
//    Mat img = imread("../chess.jpg", IMREAD_GRAYSCALE);
    Mat img = imread("../hand.jpg", IMREAD_GRAYSCALE);

    tomasi_test(img);
//    harris_cv(img);
    tomasi_cv(img);

    if (img.data == nullptr) {
        std::cout << "Could not open or find the image" << std::endl;
        return -1;
    }

    waitKey(0);
//    std::cin.get();
    return 0;
}