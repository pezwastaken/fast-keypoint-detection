#include "opencv2/core/base.hpp"
#include "opencv2/core/hal/interface.h"
#include "opencv2/core/types.hpp"
#include <algorithm>
#include <bitset>
#include <climits>
#include <random>
#include <cstdlib>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <vector>

#define TODO (exit(1))

#define PURE [[gnu::pure]]

using Image = cv::Mat;

void debug_display_image(std::string filename, const cv::Mat& img, const std::vector<cv::KeyPoint>& keypoints) {
    cv::Mat output_image;
    cv::drawKeypoints(img, keypoints, output_image);

    cv::imwrite("keypoints.jpg", output_image);

    std::cout << "detected " << keypoints.size() << " keypoints (not filtered)" << std::endl;
}

struct Pixel {
    int x;
    int y;

    explicit Pixel (int x_, int y_) : x(x_), y(y_) {}
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

    const int THRESHOLD = 40;

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
                        contig = contig && (circle[idx] >= v + THRESHOLD);

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
                        contig = contig && (circle[idx] <= v - THRESHOLD);

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
                Pixel px = Pixel { j, i };
                // cv::KeyPoint kp(cv::Point2f(j, i), 1.0f);  // size = 1.0f
                keypoints.push_back(px);
            }

        }
    }

    return keypoints;
}

inline bool in_bounds(const Pixel& c, const Pixel& other) {
    return abs(c.x - other.x) <= 2 && abs(c.y - other.y) <= 2;
}

inline int clamp_coordinate(int a, const int hi) {
    return std::clamp(a, 0, hi);
}

#define N 256

std::vector<Pixel> gen_brief_points(const cv::Mat& image) {
    const int S = 31;
    const int number_of_pairs = 2 * N;

    const int im_width = image.cols;
    const int im_height = image.rows;

    std::default_random_engine generator(std::random_device{}());

    const double std_dev = static_cast<double>(S) / 5.0;
    std::normal_distribution<double> distribution(0.0, std_dev);

    std::vector<Pixel> points;
    points.reserve(number_of_pairs);

    //TODO no guarantee that there are no overlapping points
    //
    for (int i = 0; i < number_of_pairs; ++i) {
        double x = distribution(generator);
        double y = distribution(generator);

        int px = std::round(x);
        int py = std::round(y);

        px = std::clamp(px, -15, 15);
        py = std::clamp(py, -15, 15);

        points.emplace_back(px, py);
    }

    assert(points.size() == number_of_pairs);
    return points;
}

inline Pixel add_and_clamp(Pixel current, Pixel other, int hi_x, int hi_y) {
    int x = std::clamp(current.x + other.x, 0, hi_x);
    int y = std::clamp(current.y + other.y, 0, hi_y);
    return Pixel(x, y);
}

inline bool lower_intensity(Pixel a, Pixel b, const Image& img) {

    auto intensity_a = img.at<uchar>(a.y, a.x);
    auto intensity_b = img.at<uchar>(b.y, b.x);

    return intensity_a < intensity_b;
}

std::vector<std::bitset<N>> brief_features(const Image& img, const std::vector<Pixel> keypoints) {

    //pairs of offsets
    const auto& brief_point_pairs = gen_brief_points(img);

    const int x_max = img.cols - 1;
    const int y_max = img.rows - 1;

    std::vector<std::bitset<N>> features;
    features.reserve(keypoints.size());

    for (const auto& kp : keypoints) {

        std::bitset<N> v(0);
        int bit = 0;

        for (int i = 0; i < N; ++i) {

            const auto offset_1 = brief_point_pairs[i];
            const auto offset_2 = brief_point_pairs[++i];

            const auto px = add_and_clamp(kp, offset_1, x_max, y_max);
            // const auto px = kp + offset_1;
            const auto py = add_and_clamp(kp, offset_2, x_max, y_max);
            // const auto py = kp + offset_2;

            if (lower_intensity(px, py, img)) {
                v[bit] = 1;
            }

            ++bit;
        }

        features.push_back(v);
    }

    return features;
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
    // auto filtered_keypoints = nms(keypoints);

    // debug_display_image("keypoints.jpg", gray_image, keypoints);

    brief_features(gray_image, keypoints);

    //COMPARISON
    // std::vector<cv::KeyPoint> opencv_fast_keypoints;
    // cv::FAST(gray_image, opencv_fast_keypoints, 40);  // threshold = 20
    //
    // cv::Mat fast_output;
    // cv::drawKeypoints(image, opencv_fast_keypoints, fast_output, cv::Scalar(0, 0, 255));
    // cv::imwrite("opencv_keypoints.jpg", fast_output);

    // for (const auto& k : filtered_keypoints) {
    //     std::cout << "keypoint at: " << k.x << ", " << k.y << " (" << k.score << ")" << std::endl;
    // }

    // std::cout << "detected " << filtered_keypoints.size() << " keypoints (filtered)" << std::endl;


    // cv::imwrite("blurred_image.jpg", blurred);

    return 0;
}


// std::vector<Pixel> nms(const std::vector<Pixel> keypoints) {
//
//     std::vector<Pixel> res;
//     res.reserve(keypoints.size() / 2);
//
//     for (const auto& px: keypoints) {
//
//         bool highest_score = true;
//
//         for (const auto& other: keypoints) {
//
//             if (in_bounds(px, other)) {
//                 highest_score = highest_score && px.score >= other.score ;
//             }
//         }
//
//         if (highest_score) res.push_back(px);
//
//     }
//
//     return res;
// }


