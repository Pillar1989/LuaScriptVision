/**
 * test_vpss_proc.cpp - CviVpssProcessor hardware acceleration tests
 */

#include "test_common.h"
#include <stdexcept>

#ifdef USE_CVI_MPI

namespace {
void require_cvi_ready() {
    if (!is_cvi_ready()) {
        GTEST_SKIP() << "CVI system not initialized";
    }
}

uint8_t read_plane_byte(const VIDEO_FRAME_INFO_S& frame, int plane) {
    if (plane < 0 || plane >= 3) {
        throw std::invalid_argument("read_plane_byte - invalid plane index");
    }
    CVI_U64 phys = frame.stVFrame.u64PhyAddr[plane];
    if (phys == 0) {
        throw std::runtime_error("read_plane_byte - missing physical address");
    }
    CVI_U32 length = frame.stVFrame.u32Length[plane];
    if (length == 0) {
        CVI_U32 stride = frame.stVFrame.u32Stride[plane];
        if (stride == 0) {
            stride = frame.stVFrame.u32Stride[0];
        }
        length = stride * frame.stVFrame.u32Height;
    }
    if (length == 0) {
        throw std::runtime_error("read_plane_byte - invalid plane length");
    }
    void* mapped = CVI_SYS_MmapCache(phys, length);
    if (!mapped) {
        throw std::runtime_error("read_plane_byte - CVI_SYS_MmapCache failed");
    }
    CVI_SYS_IonInvalidateCache(phys, mapped, length);
    uint8_t value = static_cast<uint8_t*>(mapped)[0];
    CVI_SYS_Munmap(mapped, length);
    return value;
}
}

TEST(VpssProcessorTest, MatToVideoFrame) {
    require_cvi_ready();

    cv::Mat mat = create_test_image(640, 480);
    Frame frame(mat);

    EXPECT_EQ(frame.storage_type(), Frame::StorageType::OPENCV);
}

TEST(VpssProcessorTest, ResizeDownscale) {
    require_cvi_ready();

    CviVpssProcessor processor;
    cv::Mat mat = create_test_image(1280, 720);

    VB_POOL vb_pool = find_suitable_vb_pool(mat.cols, mat.rows, PIXEL_FORMAT_BGR_888);
    if (vb_pool == VB_INVALID_POOLID) {
        GTEST_SKIP() << "No suitable VB pool";
    }

    VB_BLK vb_block;
    VIDEO_FRAME_INFO_S video_frame = processor.mat_to_video_frame(mat, vb_block, vb_pool);
    VbBlockGuard vb_guard(vb_block);
    Frame frame(video_frame, false);

    processor.resize(frame, 640, 640);

    EXPECT_EQ(frame.width(), 640);
    EXPECT_EQ(frame.height(), 640);
}

TEST(VpssProcessorTest, ResizeUpscale) {
    require_cvi_ready();

    CviVpssProcessor processor;
    cv::Mat mat = create_test_image(320, 240);

    VB_POOL vb_pool = find_suitable_vb_pool(mat.cols, mat.rows, PIXEL_FORMAT_BGR_888);
    if (vb_pool == VB_INVALID_POOLID) {
        GTEST_SKIP() << "No suitable VB pool";
    }

    VB_BLK vb_block;
    VIDEO_FRAME_INFO_S video_frame = processor.mat_to_video_frame(mat, vb_block, vb_pool);
    VbBlockGuard vb_guard(vb_block);
    Frame frame(video_frame, false);

    // Upscale within VPSS MEM capacity (1280x720) should succeed
    EXPECT_NO_THROW(processor.resize(frame, 640, 480));

    // Upscale beyond VPSS MEM capacity should throw
    EXPECT_THROW(processor.resize(frame, 1920, 1080), std::invalid_argument);
}

TEST(VpssProcessorTest, CvtColorBgrToRgb) {
    require_cvi_ready();

    CviVpssProcessor processor;
    cv::Mat mat = create_test_image(640, 480);

    VB_POOL vb_pool = find_suitable_vb_pool(mat.cols, mat.rows, PIXEL_FORMAT_BGR_888);
    if (vb_pool == VB_INVALID_POOLID) {
        GTEST_SKIP() << "No suitable VB pool";
    }

    VB_BLK vb_block;
    VIDEO_FRAME_INFO_S video_frame = processor.mat_to_video_frame(mat, vb_block, vb_pool);
    VbBlockGuard vb_guard(vb_block);
    Frame frame(video_frame, false);

    processor.cvtColor(frame, ColorConversion::BGR2RGB);

    SUCCEED();
}

TEST(VpssProcessorTest, Crop) {
    require_cvi_ready();

    CviVpssProcessor processor;
    cv::Mat mat = create_test_image(640, 480);

    VB_POOL vb_pool = find_suitable_vb_pool(mat.cols, mat.rows, PIXEL_FORMAT_BGR_888);
    if (vb_pool == VB_INVALID_POOLID) {
        GTEST_SKIP() << "No suitable VB pool";
    }

    VB_BLK vb_block;
    VIDEO_FRAME_INFO_S video_frame = processor.mat_to_video_frame(mat, vb_block, vb_pool);
    VbBlockGuard vb_guard(vb_block);
    Frame frame(video_frame, false);

    processor.crop(frame, 100, 100, 320, 240);

    EXPECT_EQ(frame.width(), 320);
    EXPECT_EQ(frame.height(), 240);
}

TEST(VpssProcessorTest, PhysicalAddressAfterResize) {
    require_cvi_ready();

    CviVpssProcessor processor;
    cv::Mat mat = create_test_image(640, 480);
    Frame frame(mat);

    processor.resize(frame, 320, 240);

    EXPECT_TRUE(frame.has_physical_addr());
    if (frame.has_physical_addr()) {
        EXPECT_NE(frame.physical_addr(), 0u);
    }
}

TEST(VpssProcessorTest, ChainedOperations) {
    require_cvi_ready();

    CviVpssProcessor processor;
    uint32_t test_width = 1920;
    uint32_t test_height = 1080;
    VB_POOL vb_pool = find_suitable_vb_pool(test_width, test_height, PIXEL_FORMAT_BGR_888);
    if (vb_pool == VB_INVALID_POOLID) {
        test_width = 1280;
        test_height = 720;
        vb_pool = find_suitable_vb_pool(test_width, test_height, PIXEL_FORMAT_BGR_888);
    }
    if (vb_pool == VB_INVALID_POOLID) {
        GTEST_SKIP() << "No suitable VB pool";
    }

    cv::Mat mat = create_test_image(test_width, test_height);
    VB_BLK vb_block;
    VIDEO_FRAME_INFO_S video_frame = processor.mat_to_video_frame(mat, vb_block, vb_pool);
    VbBlockGuard vb_guard(vb_block);
    Frame frame(video_frame, false);

    processor.resize(frame, 640, 640);
    processor.cvtColor(frame, ColorConversion::BGR2RGB);
    processor.crop(frame, 50, 50, 540, 540);

    EXPECT_EQ(frame.width(), 540);
    EXPECT_EQ(frame.height(), 540);
}

TEST(VpssProcessorTest, LetterboxLandscape) {
    require_cvi_ready();

    const int in_w = 1280;
    const int in_h = 720;
    const int out_w = 640;
    const int out_h = 640;
    const uint8_t pad_value = 114;

    VB_POOL vb_pool = find_suitable_vb_pool(in_w, in_h, PIXEL_FORMAT_BGR_888);
    if (vb_pool == VB_INVALID_POOLID) {
        GTEST_SKIP() << "No suitable VB pool";
    }

    CviVpssProcessor processor;
    cv::Mat mat = create_test_image(in_w, in_h);
    VB_BLK vb_block;
    VIDEO_FRAME_INFO_S video_frame = processor.mat_to_video_frame(mat, vb_block, vb_pool);
    VbBlockGuard vb_guard(vb_block);
    Frame frame(video_frame, false);

    CviVpssProcessor::LetterboxMeta meta;
    processor.letterbox(frame, out_w, out_h, pad_value, &meta);

    EXPECT_EQ(frame.width(), out_w);
    EXPECT_EQ(frame.height(), out_h);
    EXPECT_EQ(frame.pixel_format(), PixelFormat::RGB_PLANAR);

    const float scale = std::min(static_cast<float>(out_w) / in_w,
                                 static_cast<float>(out_h) / in_h);
    const int new_w = static_cast<int>(in_w * scale + 0.5f);
    const int new_h = static_cast<int>(in_h * scale + 0.5f);
    EXPECT_NEAR(meta.scale, scale, 0.01f);
    EXPECT_EQ(meta.pad_x, (out_w - new_w) / 2);
    EXPECT_EQ(meta.pad_y, (out_h - new_h) / 2);
    EXPECT_EQ(meta.ori_w, in_w);
    EXPECT_EQ(meta.ori_h, in_h);

    const VIDEO_FRAME_INFO_S* out_frame = frame.video_frame();
    ASSERT_NE(out_frame, nullptr);
    EXPECT_EQ(read_plane_byte(*out_frame, 0), pad_value);
    EXPECT_EQ(read_plane_byte(*out_frame, 1), pad_value);
    EXPECT_EQ(read_plane_byte(*out_frame, 2), pad_value);
}

TEST(VpssProcessorTest, LetterboxPortrait) {
    require_cvi_ready();

    const int in_w = 480;
    const int in_h = 640;
    const int out_w = 640;
    const int out_h = 640;
    const uint8_t pad_value = 114;

    VB_POOL vb_pool = find_suitable_vb_pool(in_w, in_h, PIXEL_FORMAT_BGR_888);
    if (vb_pool == VB_INVALID_POOLID) {
        GTEST_SKIP() << "No suitable VB pool";
    }

    CviVpssProcessor processor;
    cv::Mat mat = create_test_image(in_w, in_h);
    VB_BLK vb_block;
    VIDEO_FRAME_INFO_S video_frame = processor.mat_to_video_frame(mat, vb_block, vb_pool);
    VbBlockGuard vb_guard(vb_block);
    Frame frame(video_frame, false);

    CviVpssProcessor::LetterboxMeta meta;
    processor.letterbox(frame, out_w, out_h, pad_value, &meta);

    EXPECT_EQ(frame.width(), out_w);
    EXPECT_EQ(frame.height(), out_h);
    EXPECT_EQ(frame.pixel_format(), PixelFormat::RGB_PLANAR);

    const float scale = std::min(static_cast<float>(out_w) / in_w,
                                 static_cast<float>(out_h) / in_h);
    const int new_w = static_cast<int>(in_w * scale + 0.5f);
    const int new_h = static_cast<int>(in_h * scale + 0.5f);
    EXPECT_NEAR(meta.scale, scale, 0.01f);
    EXPECT_EQ(meta.pad_x, (out_w - new_w) / 2);
    EXPECT_EQ(meta.pad_y, (out_h - new_h) / 2);
    EXPECT_EQ(meta.ori_w, in_w);
    EXPECT_EQ(meta.ori_h, in_h);

    const VIDEO_FRAME_INFO_S* out_frame = frame.video_frame();
    ASSERT_NE(out_frame, nullptr);
    EXPECT_EQ(read_plane_byte(*out_frame, 0), pad_value);
    EXPECT_EQ(read_plane_byte(*out_frame, 1), pad_value);
    EXPECT_EQ(read_plane_byte(*out_frame, 2), pad_value);
}

#else

TEST(VpssProcessorTest, Skipped) {
    GTEST_SKIP() << "USE_CVI_MPI not defined";
}

#endif
