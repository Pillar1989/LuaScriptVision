/**
 * test_opencv_proc.cpp - OpenCvProcessor tests
 */

#include "test_common.h"

TEST(OpenCvProcessorTest, ResizeDownscale) {
    OpenCvProcessor processor;
    cv::Mat mat = create_test_image(1920, 1080);
    Frame frame(mat);

    processor.resize(frame, 640, 640);
    EXPECT_EQ(frame.width(), 640);
    EXPECT_EQ(frame.height(), 640);
}

TEST(OpenCvProcessorTest, ResizeUpscale) {
    OpenCvProcessor processor;
    cv::Mat mat = create_test_image(320, 240);
    Frame frame(mat);

    processor.resize(frame, 640, 480);
    EXPECT_EQ(frame.width(), 640);
    EXPECT_EQ(frame.height(), 480);
}

TEST(OpenCvProcessorTest, CvtColorBgrToRgb) {
    OpenCvProcessor processor;
    cv::Mat mat = create_test_image(640, 480);
    Frame frame(mat);

    EXPECT_NO_THROW(processor.cvtColor(frame, ColorConversion::BGR2RGB));
}

TEST(OpenCvProcessorTest, Crop) {
    OpenCvProcessor processor;
    cv::Mat mat = create_test_image(640, 480);
    Frame frame(mat);

    processor.crop(frame, 100, 100, 320, 240);
    EXPECT_EQ(frame.width(), 320);
    EXPECT_EQ(frame.height(), 240);
}

TEST(OpenCvProcessorTest, InvalidResizeThrows) {
    OpenCvProcessor processor;
    cv::Mat mat = create_test_image(640, 480);
    Frame frame(mat);

    EXPECT_THROW(processor.resize(frame, -100, -100), std::invalid_argument);
}

TEST(OpenCvProcessorTest, InvalidCropThrows) {
    OpenCvProcessor processor;
    cv::Mat mat = create_test_image(640, 480);
    Frame frame(mat);

    EXPECT_THROW(processor.crop(frame, 500, 500, 500, 500), std::invalid_argument);
}
