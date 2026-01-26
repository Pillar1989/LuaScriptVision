/**
 * test_input_source_image.cpp - ImageSource tests (VDEC JPEG)
 */

#include "test_common.h"

#ifdef USE_CVI_MPI

#include "cv/image_source.h"
#include "cv/cvi_vpss_processor.h"

using namespace lua_cv;

TEST(InputSourceImage, ReadAndLetterbox) {
    if (!is_cvi_ready()) {
        GTEST_SKIP() << "CVI system not initialized";
    }

    const std::string& image_path = test_image_path();
    if (image_path.empty()) {
        GTEST_SKIP() << "No test image path provided";
    }

    ImageSource source;
    ASSERT_TRUE(source.open(image_path));

    Frame frame;
    ASSERT_TRUE(source.read(frame));
    EXPECT_GT(frame.width(), 0);
    EXPECT_GT(frame.height(), 0);
    EXPECT_TRUE(frame.has_physical_addr());

    CviVpssProcessor processor;
    CviVpssProcessor::LetterboxMeta meta;
    processor.letterbox(frame, 640, 640, 114, &meta);

    EXPECT_EQ(frame.width(), 640);
    EXPECT_EQ(frame.height(), 640);
    EXPECT_EQ(frame.pixel_format(), PixelFormat::RGB_PLANAR);

    source.release(frame);
    source.close();
}

#else

TEST(InputSourceImage, Skipped) {
    GTEST_SKIP() << "USE_CVI_MPI not defined";
}

#endif
