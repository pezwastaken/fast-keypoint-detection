#include "opencv2/core/base.hpp"
#include "opencv2/core/hal/interface.h"
#include "opencv2/core/types.hpp"
#include <algorithm>
#include <bitset>
#include <climits>
#include <cmath>
#include <cstdint>
#include <random>
#include <cstdlib>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <span>
#include <utility>
#include <vector>

#define TODO (exit(1))

using usize = size_t;

struct Pixel {
    int x;
    int y;

    int score;

    explicit Pixel (int x_, int y_, int score_) : x(x_), y(y_), score(score_) {}
};

using Image = cv::Mat;
using Pixel_Vec = std::vector<Pixel>;

using Pixel_Set = std::set<Pixel>;

#define N 256
using Feature = std::bitset<N>;
using Features = std::vector<Feature>;

struct Match {
    usize idx_1;
    usize idx_2;

    usize distance;
};

struct Vec2 {
    int x;
    int y;
};



void debug_display_image(std::string filename, const Image& img, std::span<const Pixel> keypoints) {

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

void dump_keypoints(auto top_keypoints, auto file_path, const auto& gray_image) {

    int n = std::min(top_keypoints.size(), usize(15));

    for (auto px: top_keypoints.subspan(0, n)) {
        std::cout << "intensity: " << px.score << std::endl;
    }

    debug_display_image(file_path, gray_image, top_keypoints);
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

Pixel_Vec fast_kp(const Image& img) {

    const int n_rows = img.rows - 3;
    const int n_cols = img.cols - 3;

    const int THRESHOLD = 40;

    std::array<uchar, 16> circle {{ }};

    Pixel_Vec keypoints;

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
                Pixel px = Pixel { j, i, pixel_score};
                keypoints.push_back(px);
            }

        }
    }

    return keypoints;
    // return std::make_tuple(keypoints, scores);
}

inline bool in_bounds(const Pixel& c, const Pixel& other) {
    return abs(c.x - other.x) <= 2 && abs(c.y - other.y) <= 2;
}

std::vector<Pixel> nms(const Pixel_Vec& keypoints) {

    const auto n_keypoints = keypoints.size();

    Pixel_Vec res;
    res.reserve(keypoints.size() / 2);

    for (int i = 0; i < n_keypoints; ++i) {

        const auto& kp = keypoints[i];

        bool highest_score = true;

        for (int j = 0; j < n_keypoints; ++j) {

            const auto& other_kp = keypoints[j];

            if (in_bounds(kp, other_kp)) {
                highest_score = highest_score && kp.score >= other_kp.score ;
            }
        }

        if (highest_score) res.push_back(kp);

    }

    return res;
}

inline int clamp_coordinate(int a, const int hi) {
    return std::clamp(a, 0, hi);
}

Vec2 intensity_centroid(Pixel kp, const Image& img)
{
    Vec2 intensity_orientation {0};

    const int R = 15;

    const int img_width = img.cols;
    const int img_height = img.rows;

    const Vec2 min = {
        .x = std::clamp(kp.x - R, 0, img_width),
        .y = std::clamp(kp.y - R, 0, img_height)
    };

    const Vec2 max = {
        .x = std::clamp(kp.x + R, 0, img_width),
        .y = std::clamp(kp.y + R, 0, img_height)
    };

    for (int i = min.y; i < max.y; ++i)
    {
        for (int j = min.x; j < max.x; ++j)
        {

            const int rows_offset = i - kp.y;
            const int cols_offset = j - kp.x;

            //within radius
            if (rows_offset*rows_offset + cols_offset*cols_offset <= R*R)
            {
                uchar intensity = img.at<uchar>(i, j);

                intensity_orientation.x += cols_offset * intensity;
                intensity_orientation.y += rows_offset * intensity;
            }

        }
    }

    return intensity_orientation;
}

[[gnu::pure]]
float vec_to_angle(Vec2 v)
{
    return atan2(v.y, v.x);
}

std::vector<Pixel> gen_brief_points(const Image& image) {
    const int S = 31;
    const int number_of_pairs = 2 * N;

    const int im_width = image.cols;
    const int im_height = image.rows;

    std::default_random_engine generator(42);

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

        points.emplace_back(px, py, 0);
    }

    assert(points.size() == number_of_pairs);
    return points;
}

std::vector<Pixel> rotate_points(const std::vector<Pixel>& original_points, float theta)
{
    std::vector<Pixel> points;
    points.reserve(original_points.size());

    const float cos_theta = std::cos(theta);
    const float sin_theta = std::sin(theta);

    for (auto& point : original_points)
    {
        const float rotated_x = std::round(point.x * cos_theta - point.y * sin_theta);
        const float rotated_y = std::round(point.x * sin_theta + point.y * cos_theta);

        const int px = std::clamp((int)rotated_x, -15, 15);
        const int py = std::clamp((int)rotated_y, -15, 15);

        Pixel rotated_point(px, py, 0);
        points.push_back(rotated_point);
    }

    return points;
}

// map angles to their corresponding rotated brief pattern
using OrientedBriefKeypointsMap = std::vector<std::vector<Pixel>>;

OrientedBriefKeypointsMap gen_orb_map(const std::vector<Pixel>& brief_points)
{

    const int bins = 360 / 12;
    const float angle_increment_rad = 12.0F * M_PI / 180.0F;

    OrientedBriefKeypointsMap rotated_patterns;
    rotated_patterns.reserve(bins);

    for (int i = 0; i < bins; ++i)
    {
        const float angle = (i + 0.5F) * angle_increment_rad;

        rotated_patterns.push_back(rotate_points(brief_points, angle));
    }

    return rotated_patterns;
}

inline Pixel add_and_clamp(Pixel current, Pixel other, int hi_x, int hi_y) {
    int x = std::clamp(current.x + other.x, 0, hi_x);
    int y = std::clamp(current.y + other.y, 0, hi_y);
    return Pixel(x, y, current.score);
}

inline bool lower_intensity(Pixel a, Pixel b, const Image& img) {

    auto intensity_a = img.at<uchar>(a.y, a.x);
    auto intensity_b = img.at<uchar>(b.y, b.x);

    return intensity_a < intensity_b;
}

std::vector<Feature> brief_features(const Image& img, std::span<const Pixel> keypoints) {

    Image blurred_img;
    cv::GaussianBlur(img, blurred_img, cv::Size(3, 3), 0);

    //pairs of offsets
    const auto brief_points = gen_brief_points(blurred_img);
    const auto orb_map = gen_orb_map(brief_points);

    const auto& brief_point_pairs = orb_map[20];

    const int x_max = blurred_img.cols - 1;
    const int y_max = blurred_img.rows - 1;

    assert(blurred_img.cols == img.cols);
    assert(blurred_img.rows == img.rows);

    std::vector<Feature> features;
    features.reserve(keypoints.size());

    for (const auto& kp : keypoints) {

        std::bitset<N> v(0);
        int bit = 0;

        // Vec2 intensity_vec = intensity_centroid(kp, blurred_img);
        // float intensity_angle = vec_to_angle(intensity_vec);
        // if (intensity_angle < 0) intensity_angle += 2.0F * M_PI;
        //
        // int rotated_pattern_bin = (intensity_angle * 180.0F / M_PI) / 12;
        // assert(rotated_pattern_bin < 30);
        // assert(rotated_pattern_bin >= 0);
        // const auto& brief_point_pairs = orb_map[rotated_pattern_bin];

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

inline usize h_distance(const Feature& a, const Feature& b) {
    return (a ^ b).count();
}

//returns index + distance
inline std::pair<usize, usize> best_match(const Feature& feat, const Features& haystack) {

    usize best_match = SIZE_MAX;
    usize best_match_idx = N;

    const usize set_size = haystack.size();

    for (usize j = 0; j < set_size; ++j) {

        const auto& f_2 = haystack[j];

        usize distance = h_distance(feat, f_2);
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

    const usize s1_size = set_1.size();
    const usize s2_size = set_2.size();

    for (usize i = 0; i < s1_size; ++i) {

        const auto& f_1 = set_1[i];

        auto [ idx_set_2, distance ] = best_match(f_1, set_2);
        // std::cout << "matched " << i << " with " << idx_set_2 << std::endl;

        //cross check here
        const auto& f_2 = set_2[idx_set_2];

        auto [ idx_set_1, _ ] = best_match(f_2, set_1);
        // std::cout << "\t crosscheck: matched " << idx_set_2 << " with " << idx_set_1 << std::endl;

        //matched in both ways
        if (idx_set_1 == i) {
            Match m = Match { .idx_1 = i, .idx_2 = idx_set_2, .distance = distance};
            matches.push_back(m);
        }

    }

    std::sort(matches.begin(), matches.end(),
              [](const auto& match1, const auto& match2) { return match1.distance < match2.distance; });

    return matches;

}

Pixel_Vec extract_keypoints(const Image& img) {

    auto keypoints_with_score = fast_kp(img);
    auto filtered_keypoints = nms(keypoints_with_score);

    std::sort(filtered_keypoints.begin(),
              filtered_keypoints.end(),
              [](const auto& px1, const auto& px2) { return px1.score > px2.score; });

    return filtered_keypoints;

}

#define N_KEYPOINTS (usize(1500))

std::span<const Pixel> span_top_keypoints(const std::vector<Pixel>& keypoints) {
    const usize N_TOP_KEYPOINTS = std::min(N_KEYPOINTS, keypoints.size());
    std::cout << "N_TOP_KEYPOINTS: " << N_TOP_KEYPOINTS << std::endl;

    std::span<const Pixel> top_keypoints(keypoints);
    top_keypoints = top_keypoints.subspan(0, N_TOP_KEYPOINTS);

    return top_keypoints;
}

Features extract_features_from_keypoints(const Image& img, std::span<const Pixel> keypoints, std::string debug_output_path="") {

#ifndef NDEBUG
    dump_keypoints(keypoints, debug_output_path, img);
#endif

    Features features = brief_features(img, keypoints);

    assert(features.size() == keypoints.size());

    return features;
}

void display_matches(const Image& i1, const Image& i2, const auto& matches,
                     std::span<const Pixel> kps1, std::span<const Pixel> kps2) {

    Image output;
    cv::hconcat(i1, i2, output);

    for (int i = 0; i < 30; ++i) {

        const Match& match = matches[i];

        const auto& kp1 = kps1[match.idx_1];
        const auto& kp2 = kps2[match.idx_2];

        cv::Point2f pt1 (kp1.x, kp1.y);
        cv::Point2f pt2 (kp2.x, kp2.y);
        pt2.x += i1.cols; // Offset for second image

        // Draw line
        cv::line(output, pt1, pt2, cv::Scalar(0, 255, 0), 1);

        // Draw circles
        cv::circle(output, pt1, 3, cv::Scalar(0, 0, 255), -1);
        cv::circle(output, pt2, 3, cv::Scalar(0, 0, 255), -1);

    }

    cv::imwrite("matches.jpg", output);

}

Image read_image(std::string filepath) {

    cv::Mat image = cv::imread(filepath, cv::IMREAD_COLOR);

    if (image.empty()) {
        std::cerr << "Error: Could not open or find the image at " << filepath
                  << std::endl;
        exit(1);
    }

    return image;

}

void match_images(auto filename_1, auto filename_2) {

    std::string image_dir = "input_image/library/";

    auto img_1 = read_image(image_dir + filename_1);
    Image grayscale_image_1;
    cv::cvtColor(img_1, grayscale_image_1, cv::COLOR_BGR2GRAY);

    const auto kp_1 = extract_keypoints(grayscale_image_1);
    auto top_kp_1 = span_top_keypoints(kp_1);
    const auto features_1 = extract_features_from_keypoints(grayscale_image_1, top_kp_1, image_dir + filename_1);


    auto img_2 = read_image(image_dir + filename_2);
    Image grayscale_image_2;
    cv::cvtColor(img_2, grayscale_image_2, cv::COLOR_BGR2GRAY);

    const auto kp_2 = extract_keypoints(grayscale_image_2);
    const auto top_kp_2 = span_top_keypoints(kp_2);
    const auto features_2 = extract_features_from_keypoints(grayscale_image_2, top_kp_2, image_dir + filename_2);

    std::cout << " distance between 0th elements: " << h_distance(features_1[0], features_2[0]) << std::endl;
    std::cout << " distance between 1th elements: " << h_distance(features_1[1], features_2[1]) << std::endl;

    const auto matches = brute_force_match(features_1, features_2);

    display_matches(img_1, img_2, matches, top_kp_1, top_kp_2);

}

int main() {

    match_images("5.jpg", "6.jpg");


    //COMPARISON
    // std::vector<cv::KeyPoint> opencv_fast_keypoints;
    // cv::FAST(gray_image, opencv_fast_keypoints, 40);
    //
    // cv::Mat fast_output;
    // cv::drawKeypoints(image, opencv_fast_keypoints, fast_output, cv::Scalar(0, 0, 255));
    // cv::imwrite("opencv_keypoints.jpg", fast_output);

    return 0;
}


