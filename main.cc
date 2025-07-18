#include "opencv2/core/base.hpp"
#include "opencv2/core/hal/interface.h"
#include "opencv2/core/types.hpp"
#include <algorithm>
#include <bitset>
#include <climits>
#include <cstdint>
#include <random>
#include <cstdlib>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <tuple>
#include <utility>
#include <vector>

#define TODO (exit(1))

#define PURE [[gnu::pure]]

struct Pixel {
    int x;
    int y;

    explicit Pixel (int x_, int y_) : x(x_), y(y_) {}
};

using Image = cv::Mat;
using Pixel_Vec = std::vector<Pixel>;
using Score_Vec = std::vector<int>;

#define N 256
using Feature = std::bitset<N>;
using Features = std::vector<Feature>;

struct Match {
    size_t idx_1;
    size_t idx_2;

    size_t distance;
};


void debug_display_image(std::string filename, const Image& img, const Pixel_Vec& keypoints) {

    std::string output_prefix = "output_image/keypoints_";

    std::vector<cv::KeyPoint> cv_keypoints;
    cv_keypoints.reserve(keypoints.size());

                // cv::KeyPoint kp(cv::Point2f(j, i), 1.0f);  // size = 1.0f

    for (const auto& k : keypoints) {
        cv::KeyPoint cv_kp(cv::Point2f(k.x, k.y), 1.0f);
        cv_keypoints.push_back(cv_kp);
    }

    cv::Mat output_image;
    cv::drawKeypoints(img, cv_keypoints, output_image);

    cv::imwrite(output_prefix + filename, output_image);
    std::cout << "detected " << keypoints.size() << " keypoints" << std::endl;
}

inline void load_circle(std::array<uchar, 16>& b, const Image& img, int i, int j) {

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

std::tuple<Pixel_Vec, Score_Vec> fast_kp(const Image& img) {

    const int n_rows = img.rows - 3;
    const int n_cols = img.cols - 3;

    const int THRESHOLD = 40;

    std::array<uchar, 16> circle {{ }};
    // std::vector<std::tuple <Pixel, int> > keypoints;

    Pixel_Vec keypoints;
    Score_Vec scores;

    keypoints.reserve(1000);

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
                Pixel px = Pixel { j, i};
                // cv::KeyPoint kp(cv::Point2f(j, i), 1.0f);  // size = 1.0f
                keypoints.push_back(px);
                scores.push_back(pixel_score);
            }

        }
    }

    return std::make_tuple(keypoints, scores);
}

inline bool in_bounds(const Pixel& c, const Pixel& other) {
    return abs(c.x - other.x) <= 2 && abs(c.y - other.y) <= 2;
}

std::vector<Pixel> nms(const std::tuple<Pixel_Vec, Score_Vec>& keypoints_with_score) {

    const auto& keypoints = std::get<Pixel_Vec>(keypoints_with_score);
    const auto& scores = std::get<Score_Vec>(keypoints_with_score);

    assert(keypoints.size() == scores.size());

    const auto n_keypoints = keypoints.size();

    Pixel_Vec res;
    res.reserve(keypoints.size() / 2);

    for (int i = 0; i < n_keypoints; ++i) {

        const auto& kp = keypoints[i];
        const auto& score = scores[i];

        bool highest_score = true;

        for (int j = 0; j < n_keypoints; ++j) {

            const auto& other_kp = keypoints[j];
            const auto& other_score = scores[j];

            if (in_bounds(kp, other_kp)) {
                highest_score = highest_score && score >= other_score ;
            }
        }

        if (highest_score) res.push_back(kp);

    }

    return res;
}

inline int clamp_coordinate(int a, const int hi) {
    return std::clamp(a, 0, hi);
}

std::vector<Pixel> gen_brief_points(const Image& image) {
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

std::vector<Feature> brief_features(const Image& img, const Pixel_Vec& keypoints) {

    Image blurred_img;
    cv::GaussianBlur(img, blurred_img, cv::Size(3, 3), 0);

    //pairs of offsets
    const auto& brief_point_pairs = gen_brief_points(blurred_img);

    const int x_max = blurred_img.cols - 1;
    const int y_max = blurred_img.rows - 1;

    assert(blurred_img.cols == img.cols);
    assert(blurred_img.rows == img.rows);

    std::vector<Feature> features;
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

            if (lower_intensity(px, py, blurred_img)) {
                v[bit] = 1;
            }

            ++bit;
        }

        features.push_back(v);
    }

    return features;
}

inline size_t h_distance(const Feature& a, const Feature& b) {
    return (a ^ b).count();
}

//returns index + distance
inline std::pair<size_t, size_t> best_match(const Feature& feat, const Features& haystack) {

    size_t best_match = SIZE_MAX;
    size_t best_match_idx = N;

    const size_t set_size = haystack.size();

    for (size_t j = 0; j < set_size; ++j) {

        const auto& f_2 = haystack[j];

        size_t distance = h_distance(feat, f_2);
        if (distance < best_match) {
            best_match = distance;
            best_match_idx = j;
        }
    }

    return std::make_pair(best_match_idx, best_match);
}


std::vector<Match> brute_force_match(const Features& set_1, const Features& set_2) {

    std::vector<Match> matches;
    matches.reserve(set_1.size());

    const size_t s1_size = set_1.size();
    const size_t s2_size = set_2.size();

    //NOTE surjective

    for (size_t i = 0; i < s1_size; ++i) {

        const auto& f_1 = set_1[i];

        auto [ idx_set_2, distance ] = best_match(f_1, set_2);

        //cross check here
        const auto& f_2 = set_2[idx_set_2];

        auto [ idx_set_1, _ ] = best_match(f_2, set_1);

        //matched in both ways
        if (idx_set_1 == i) {
            Match m = Match { .idx_1 = i, .idx_2 = idx_set_2, .distance = distance};
            matches.push_back(m);
        }

    }

    return matches;

}

int main() {

    // std::string image_path = "squares.png";
    std::string image_dir = "input_image/library/";
    std::string filename = "2.jpg";

    cv::Mat image = cv::imread(image_dir + filename, cv::IMREAD_COLOR);

    if (image.empty()) {
        std::cerr << "Error: Could not open or find the image at " << image_dir + filename
                  << std::endl;
        return -1;
    }

    cv::Mat gray_image;
    cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);

    auto keypoints_with_score = fast_kp(gray_image);
    const auto filtered_keypoints = nms(keypoints_with_score);

    debug_display_image(filename, gray_image, filtered_keypoints);

    const Features features = brief_features(gray_image, filtered_keypoints);

    assert(features.size() == filtered_keypoints.size());

    //COMPARISON
    // std::vector<cv::KeyPoint> opencv_fast_keypoints;
    // cv::FAST(gray_image, opencv_fast_keypoints, 40);
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


