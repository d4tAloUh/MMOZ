#include <opencv2/highgui.hpp>
#include<opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp>
#include <iostream>

using namespace cv;

void calculateDistribution(int dist[], const Mat &img) {
    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            dist[img.at<uchar>(i, j)]++;
        }
    }
}

// cumulative distribution function
void cdf(int cum_dist[], const int dist[]) {
    cum_dist[0] = dist[0];
    for (int i = 1; i < 256; i++) {
        cum_dist[i] = dist[i] + cum_dist[i - 1];
    }
}

void optimizeHeight(int array[], int height){
    int maxVal = 0;
    //  find maximum
    for (int i = 0; i < 256; ++i) {
        if (array[i] > maxVal) {
            maxVal = array[i];
        }
    }
    //  normalize height to hist_h
    for (int i = 0; i < 256; ++i) {
        array[i] = array[i] * height / (maxVal + 1);
    }

}

void drawHistogram(const Mat &img, Mat &histogram) {
    int hist_w = 512;
    int hist_h = 512;

    histogram = Mat(hist_h, hist_w, CV_8UC3);

//  distribution
    int dist[256] = {0};
    int cum_dist[256] = {0};

    calculateDistribution(dist, img);
    optimizeHeight(dist, hist_h);

    cdf(cum_dist, dist);
    optimizeHeight(cum_dist, hist_h);

    for (int i = 0; i < 256; i++) {
        line(histogram,
             Point(i * 2, hist_h),
             Point(i * 2, hist_h - dist[i]),
             Scalar(255, 255, 255), 1, LINE_8, 0
        );
    }
    for (int i = 1; i < 256; i++) {
        line(histogram,
             Point((i - 1)* 2, hist_h - cum_dist[i-1]),
             Point(i * 2, hist_h - cum_dist[i]),
             Scalar(0, 255, 0), 1, LINE_8, 0
        );
    }
}

void HistogramEqualization(const Mat &in_image, Mat &out_image, Mat &in_histogram, Mat &out_histogram) {
    out_image = Mat(in_image.rows, in_image.cols, CV_8UC1);
    int dist[256] = {0};
    int cum_dist[256] = {0};

    calculateDistribution(dist, in_image);
    cdf(cum_dist, dist);
//  coefficient to normalize
    double alpha = 255.0 / (in_image.rows * in_image.cols);

    for (int i = 0; i < in_image.rows; ++i) {
        for (int j = 0; j < in_image.cols; ++j) {
            uchar val = in_image.at<uchar>(i, j);
            out_image.at<uchar>(i, j) = cvRound(
                    cum_dist[val] * alpha
            );
        }
    }
    drawHistogram(in_image, in_histogram);
    drawHistogram(out_image, out_histogram);
};

int main() {
//    Mat img = imread("../chair.tif", IMREAD_GRAYSCALE);
    Mat img = imread("../sebek.jpg", IMREAD_GRAYSCALE);
//    Mat img = imread("../test.jpg", IMREAD_GRAYSCALE);
    if (img.data == nullptr) {
        std::cout << "Could not open or find the image" << std::endl;
        return -1;
    }
    Mat outImage;
    Mat inHistogram;
    Mat outHistogram;
    HistogramEqualization(img, outImage, inHistogram, outHistogram);
    imshow("Original image", img);
    imshow("Normalized image", outImage);
    imshow("Original histogram", inHistogram);
    imshow("Normalized histogram", outHistogram);
    waitKey(0);
//    std::cin.get();
    return 0;
}
