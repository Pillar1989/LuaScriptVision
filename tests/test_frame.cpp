/**
 * test_frame.cpp - Frame class tests
 */

#include "test_common.h"

TEST(FrameTest, EmptyConstructor) {
    Frame frame;
    EXPECT_TRUE(frame.empty());
}

TEST(FrameTest, FromMat) {
    cv::Mat mat = create_test_image(640, 480);
    Frame frame(mat);

    EXPECT_FALSE(frame.empty());
    EXPECT_EQ(frame.width(), 640);
    EXPECT_EQ(frame.height(), 480);
    EXPECT_EQ(frame.channels(), 3);
}

TEST(FrameTest, ToMat) {
    cv::Mat mat = create_test_image(320, 240);
    Frame frame(mat);

    const cv::Mat& result = frame.to_mat();
    EXPECT_EQ(result.cols, 320);
    EXPECT_EQ(result.rows, 240);
}

TEST(FrameTest, StorageType) {
    cv::Mat mat = create_test_image(640, 480);
    Frame frame(mat);

    EXPECT_EQ(frame.storage_type(), Frame::StorageType::OPENCV);
}

TEST(FrameTest, PhysicalAddressForMat) {
    cv::Mat mat = create_test_image(640, 480);
    Frame frame(mat);

    EXPECT_FALSE(frame.has_physical_addr());
}

TEST(FrameTest, MoveConstructor) {
    cv::Mat mat = create_test_image(640, 480);
    Frame frame1(mat);
    Frame frame2(std::move(frame1));

    EXPECT_TRUE(frame1.empty());
    EXPECT_FALSE(frame2.empty());
    EXPECT_EQ(frame2.width(), 640);
    EXPECT_EQ(frame2.height(), 480);
}

TEST(FrameTest, Clone) {
    cv::Mat mat = create_test_image(640, 480);
    Frame frame1(mat);
    Frame frame2 = frame1.clone();

    EXPECT_FALSE(frame1.empty());
    EXPECT_FALSE(frame2.empty());
    EXPECT_EQ(frame2.width(), 640);
    EXPECT_EQ(frame2.height(), 480);
}
