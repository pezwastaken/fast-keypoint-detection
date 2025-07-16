#include "opencv2/core/base.hpp"
#include "opencv2/core/hal/interface.h"
#include <climits>
#include <cstdlib>
#include <iostream>
#include <future>
#include <string>
#include <opencv2/opencv.hpp>
#include <vector>

#define TODO (exit(1))

#define PURE [[gnu::pure]]

struct Pixel {
    int x;
    int y;
    int score;
};

inline void load_circle(std::array<uchar, 16>& b, const cv::Mat& img, int i, int j) {

    b[1] = img.at<uchar>(i - 3, j + 1);
    b[2] = img.at<uchar>(i - 2, j + 2);

    b[3] = img.at<uchar>(i - 1, j + 3);
    //skip the 5th
    b[5] = img.at<uchar>(i + 1, j + 3);

    b[6] = img.at<uchar>(i + 2, j + 2);

    b[7] = img.at<uchar>(i + 3, j + 1);
    //skip the 9th
    b[9] = img.at<uchar>(i + 3, j - 1);

    b[10] = img.at<uchar>(i + 2, j - 2);

    b[11] = img.at<uchar>(i + 1, j - 3);
    //skip the 12th
    b[13] = img.at<uchar>(i - 1, j - 3);

    b[14] = img.at<uchar>(i - 2, j - 2);

    b[15] = img.at<uchar>(i - 3, j - 1);
}

std::vector<Pixel> fast_kp(const cv::Mat& img) {

    const int n_rows = img.rows - 3;
    const int n_cols = img.cols - 3;

    const int THRESHOLD = 50;

    std::array<uchar, 16> circle {{ }};
    std::vector<Pixel> keypoints;
    keypoints.reserve(100);

    for (int i = 3; i < n_rows; ++i) {
        for (int j = 3; j < n_cols; ++j) {

            uchar v = img.at<uchar>(i, j);

            uchar v_1 = img.at<uchar>(i - 3, j);
            uchar v_9 = img.at<uchar>(i + 3, j);
            uchar v_5 = img.at<uchar>(i, j + 3);
            uchar v_13 = img.at<uchar>(i, j - 3);

            int brighter = 0;
            int darker = 0;

            if (v_1 > v + THRESHOLD) ++brighter;
            else if (v_1 < v - THRESHOLD) ++darker;

            if (v_9 > v + THRESHOLD) ++brighter;
            else if (v_9 < v - THRESHOLD) ++darker;

            if (v_5 > v + THRESHOLD) ++brighter;
            else if (v_5 < v - THRESHOLD) ++darker;

            if (v_13 > v + THRESHOLD) ++brighter;
            else if (v_13 < v - THRESHOLD) ++darker;

            if (brighter < 3 && darker < 3) continue;

            circle[0] = v_1;
            circle[8] = v_9;
            circle[4] = v_5;
            circle[12] = v_13;

            const int circle_size = 16;
            load_circle(circle, img, i, j);

            bool found = false;
            bool contig = true;

            int pixel_score = INT_MAX;

            if (brighter > darker) {

                for (int i_circle = 0; i_circle < circle_size && !found; ++i_circle) {

                    contig = true;
                    int current_score = INT_MAX;

                    for (int j_circle = 0; j_circle < 12 && contig; ++j_circle) {
                        int idx = (i_circle + j_circle) & 15;
                        contig = contig && (circle[idx] > v + THRESHOLD);

                        const int tmp = circle[idx] - v;
                        if (tmp < current_score) current_score = tmp;

                    }

                    if (contig) {
                        if (current_score < pixel_score) pixel_score = current_score;
                        found = true;
                    }

                }
            }

            else {

                for (int i_circle = 0; i_circle < circle_size && !found; ++i_circle) {

                    contig = true;
                    int current_score = INT_MAX;

                    for (int j_circle = 0; j_circle < 12 && contig; ++j_circle) { 
                        int idx = (i_circle + j_circle) & 15;
                        contig = contig && (circle[idx] < v - THRESHOLD);

                        const int tmp = v - circle[idx];
                        if (tmp < current_score) current_score = tmp;
                    }

                    if (contig) {
                        if (current_score < pixel_score) pixel_score = current_score;
                        found = true;
                    }
                }
            }

            if (found) {
                Pixel px = Pixel { i, j, pixel_score};
                keypoints.push_back(px);
            }

        }
    }

    return keypoints;
}

void nms(const std::vector<Pixel> keypoints) {

    std::vector<Pixel> res;
    res.reserve(keypoints.size() / 2);

    // for (range-declaration : range-expression) {
    //
    // }

}

int main() {

    // std::string image_path = "squares.png";
    std::string image_path = "11.jpg";

    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);

    if (image.empty()) {
        std::cerr << "Error: Could not open or find the image at " << image_path
            << std::endl;
        return -1;
    }

    cv::Mat gray_image;
    cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);


    auto keypoints = fast_kp(gray_image);

    for (const auto& k : keypoints) {
        std::cout << "keypoint at: " << k.x << ", " << k.y << " (" << k.score << ")" << std::endl;
    }

    std::cout << "detected " << keypoints.size() << " keypoints" << std::endl;

    // cv::imwrite("blurred_image.jpg", blurred);

    return 0;
}
