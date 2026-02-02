/**
 * test_real_image.cpp - Real image processing tests
 */

#include "test_common.h"

TEST(RealImageTest, LoadAndProcess) {
    const std::string& image_path = test_image_path();
    if (image_path.empty()) {
        GTEST_SKIP() << "No test image path provided";
    }

    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    ASSERT_FALSE(img.empty()) << "Image not found or invalid: " << image_path;
    if (img.cols < 640 || img.rows < 640) {
        GTEST_SKIP() << "Image smaller than 640x640 (upscale not allowed)";
    }

    Frame frame(img.clone());
    cv_helpers::resize(frame, 640, 640);
    EXPECT_EQ(frame.width(), 640);
    EXPECT_EQ(frame.height(), 640);

    EXPECT_NO_THROW(cv_helpers::cvt_color(frame, ColorConversion::BGR2RGB));
}

TEST(RealImageTest, PreprocessPipeline) {
    const std::string& image_path = test_image_path();
    if (image_path.empty()) {
        GTEST_SKIP() << "No test image path provided";
    }

    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    ASSERT_FALSE(img.empty()) << "Image not found or invalid: " << image_path;
    if (img.cols < 640 || img.rows < 640) {
        GTEST_SKIP() << "Image smaller than 640x640 (upscale not allowed)";
    }

    Frame frame(img.clone());
    cv_helpers::resize(frame, 640, 640);
    cv_helpers::cvt_color(frame, ColorConversion::BGR2RGB);

    EXPECT_EQ(frame.width(), 640);
    EXPECT_EQ(frame.height(), 640);
}

TEST(RealImageTest, ToTensor) {
    const std::string& image_path = test_image_path();
    if (image_path.empty()) {
        GTEST_SKIP() << "No test image path provided";
    }

    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    ASSERT_FALSE(img.empty()) << "Image not found or invalid: " << image_path;
    if (img.cols < 640 || img.rows < 640) {
        GTEST_SKIP() << "Image smaller than 640x640 (upscale not allowed)";
    }

    Frame frame(img.clone());
    cv_helpers::resize(frame, 640, 640);
    cv_helpers::cvt_color(frame, ColorConversion::BGR2RGB);

    std::vector<double> mean = {0.0, 0.0, 0.0};
    std::vector<double> std = {1.0, 1.0, 1.0};
    auto tensor = cv_helpers::frame_to_tensor(frame, 1.0 / 255.0, mean, std);

    EXPECT_EQ(tensor.ndim(), 4);
    EXPECT_EQ(tensor.size(0), 1);
    EXPECT_EQ(tensor.size(1), 3);
    EXPECT_EQ(tensor.size(2), 640);
    EXPECT_EQ(tensor.size(3), 640);
}

TEST(RealImageTest, Crop) {
    const std::string& image_path = test_image_path();
    if (image_path.empty()) {
        GTEST_SKIP() << "No test image path provided";
    }

    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    ASSERT_FALSE(img.empty()) << "Image not found or invalid: " << image_path;

    Frame frame(img.clone());

    int crop_w = std::min(img.cols, 320);
    int crop_h = std::min(img.rows, 240);
    int x = (img.cols - crop_w) / 2;
    int y = (img.rows - crop_h) / 2;

    cv_helpers::crop(frame, x, y, crop_w, crop_h);
    EXPECT_EQ(frame.width(), crop_w);
    EXPECT_EQ(frame.height(), crop_h);
}

TEST(RealImageTest, InferencePreparation) {
    const std::string& image_path = test_image_path();
    if (image_path.empty()) {
        GTEST_SKIP() << "No test image path provided";
    }

    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    ASSERT_FALSE(img.empty()) << "Image not found or invalid: " << image_path;
    if (img.cols < 640 || img.rows < 640) {
        GTEST_SKIP() << "Image smaller than 640x640 (upscale not allowed)";
    }

    Frame frame(img.clone());
    cv_helpers::resize(frame, 640, 640);
    cv_helpers::cvt_color(frame, ColorConversion::BGR2RGB);

    std::vector<double> mean = {0.485, 0.456, 0.406};
    std::vector<double> std = {0.229, 0.224, 0.225};
    auto tensor = cv_helpers::frame_to_tensor(frame, 1.0 / 255.0, mean, std);

    EXPECT_EQ(tensor.ndim(), 4);
    EXPECT_EQ(tensor.size(0), 1);
    EXPECT_EQ(tensor.size(1), 3);
    EXPECT_EQ(tensor.size(2), 640);
    EXPECT_EQ(tensor.size(3), 640);
}
