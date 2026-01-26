/**
 * test_cv_helpers.cpp - cv_helpers smart backend selection tests
 */

#include "test_common.h"

TEST(CvHelpersTest, BackendDetectionForMat) {
    cv::Mat mat = create_test_image(640, 480);
    Frame frame(mat);

    const char* backend = cv_helpers::get_backend_name(frame);
    EXPECT_STREQ(backend, "opencv");
}

TEST(CvHelpersTest, ResizeCpuPath) {
    cv::Mat mat = create_test_image(1920, 1080);
    Frame frame(mat);

    const char* backend = cv_helpers::get_backend_name(frame);
    EXPECT_STREQ(backend, "opencv");

    cv_helpers::resize(frame, 640, 640);
    EXPECT_EQ(frame.width(), 640);
    EXPECT_EQ(frame.height(), 640);
}

TEST(CvHelpersTest, CvtColor) {
    cv::Mat mat = create_test_image(640, 480);
    Frame frame(mat);

    EXPECT_NO_THROW(cv_helpers::cvt_color(frame, ColorConversion::BGR2RGB));
}

TEST(CvHelpersTest, Crop) {
    cv::Mat mat = create_test_image(640, 480);
    Frame frame(mat);

    cv_helpers::crop(frame, 100, 100, 320, 240);
    EXPECT_EQ(frame.width(), 320);
    EXPECT_EQ(frame.height(), 240);
}

TEST(CvHelpersTest, EmptyFrameThrows) {
    Frame frame;
    EXPECT_THROW(cv_helpers::resize(frame, 640, 640), std::invalid_argument);
}

TEST(CvHelpersTest, ChainedOperations) {
    cv::Mat mat = create_test_image(1920, 1080);
    Frame frame(mat);

    cv_helpers::resize(frame, 640, 640);
    cv_helpers::cvt_color(frame, ColorConversion::BGR2RGB);
    cv_helpers::crop(frame, 50, 50, 540, 540);

    EXPECT_EQ(frame.width(), 540);
    EXPECT_EQ(frame.height(), 540);
}
