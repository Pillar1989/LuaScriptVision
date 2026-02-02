#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <cstring>

#include "stream/venc_encoder.h"
#include "cv/mmf_context.h"

#ifdef USE_CVI_MPI
#include <cvi_sys.h>
#include <cvi_vb.h>
#include <cvi_vpss.h>
#include <cvi_buffer.h>
#endif

#ifndef ALIGN
#define ALIGN(x, a) (((x) + (a) - 1) & ~((a) - 1))
#endif

using namespace lua_cv;

class VencEncoderTest : public ::testing::Test {
protected:
    void SetUp() override {
#ifdef USE_CVI_MPI
        MmfContext::Config config;
        if (!MmfContext::build_default_config(&config)) {
            GTEST_SKIP() << "Failed to build MMF config";
        }
        config.force_reset = true;
        if (!MmfContext::instance().init(config)) {
            GTEST_SKIP() << "Failed to init MMF context";
        }
#else
        GTEST_SKIP() << "CVI MPI not available";
#endif
    }

    void TearDown() override {
#ifdef USE_CVI_MPI
        MmfContext::instance().shutdown();
#endif
    }
};

#ifdef USE_CVI_MPI

TEST_F(VencEncoderTest, H264BasicInit) {
    VencEncoder::Config config;
    config.codec = VencEncoder::CodecType::H264;
    config.width = 1920;
    config.height = 1080;
    config.fps = 30;
    config.bitrate_kbps = 2000;
    config.gop = 30;
    config.channel = 0;

    VencEncoder encoder(config);
    ASSERT_TRUE(encoder.init());
    EXPECT_TRUE(encoder.is_initialized());
    EXPECT_FALSE(encoder.is_bound());

    encoder.shutdown();
    EXPECT_FALSE(encoder.is_initialized());
}

TEST_F(VencEncoderTest, H265BasicInit) {
    VencEncoder::Config config;
    config.codec = VencEncoder::CodecType::H265;
    config.width = 1920;
    config.height = 1080;
    config.fps = 30;
    config.bitrate_kbps = 2000;
    config.gop = 30;
    config.channel = 0;

    VencEncoder encoder(config);
    ASSERT_TRUE(encoder.init());
    EXPECT_TRUE(encoder.is_initialized());

    encoder.shutdown();
    EXPECT_FALSE(encoder.is_initialized());
}

TEST_F(VencEncoderTest, JpegBasicInit) {
    VencEncoder::Config config;
    config.codec = VencEncoder::CodecType::JPEG;
    config.width = 1280;
    config.height = 720;
    config.quality = 80;
    config.channel = 1;

    VencEncoder encoder(config);
    ASSERT_TRUE(encoder.init());
    EXPECT_TRUE(encoder.is_initialized());

    encoder.shutdown();
    EXPECT_FALSE(encoder.is_initialized());
}

TEST_F(VencEncoderTest, DISABLED_MjpegBasicInit) {
    // Note: SG2002 SDK does not support MJPEG encoding
    VencEncoder::Config config;
    config.codec = VencEncoder::CodecType::MJPEG;
    config.width = 1920;
    config.height = 1080;
    config.fps = 30;
    config.bitrate_kbps = 8000;
    config.channel = 0;

    VencEncoder encoder(config);
    ASSERT_TRUE(encoder.init());
    EXPECT_TRUE(encoder.is_initialized());

    encoder.shutdown();
    EXPECT_FALSE(encoder.is_initialized());
}

TEST_F(VencEncoderTest, MultipleChannels) {
    VencEncoder::Config config1;
    config1.codec = VencEncoder::CodecType::H264;
    config1.width = 1920;
    config1.height = 1080;
    config1.channel = 0;

    VencEncoder::Config config2;
    config2.codec = VencEncoder::CodecType::JPEG;
    config2.width = 640;
    config2.height = 640;
    config2.channel = 1;

    VencEncoder encoder1(config1);
    VencEncoder encoder2(config2);

    ASSERT_TRUE(encoder1.init());
    ASSERT_TRUE(encoder2.init());

    EXPECT_TRUE(encoder1.is_initialized());
    EXPECT_TRUE(encoder2.is_initialized());
    EXPECT_EQ(encoder1.channel(), 0);
    EXPECT_EQ(encoder2.channel(), 1);

    encoder2.shutdown();
    encoder1.shutdown();
}

TEST_F(VencEncoderTest, JpegEncodeManualFrame) {
    // Use 1280x720 which matches the VB pool configuration
    VencEncoder::Config config;
    config.codec = VencEncoder::CodecType::JPEG;
    config.width = 1280;
    config.height = 720;
    config.quality = 80;
    config.channel = 0;

    VencEncoder encoder(config);
    ASSERT_TRUE(encoder.init());

    VIDEO_FRAME_INFO_S frame;
    std::memset(&frame, 0, sizeof(frame));

    uint32_t width = config.width;
    uint32_t height = config.height;
    uint32_t stride = ALIGN(width, 64);
    uint32_t y_size = stride * height;
    uint32_t uv_size = y_size / 2;
    uint32_t total_size = y_size + uv_size;

    VB_BLK blk = CVI_VB_GetBlock(VB_INVALID_POOLID, total_size);
    if (blk == VB_INVALID_HANDLE) {
        std::cout << "[TEST] Could not allocate VB block (size=" << total_size
                  << "), skipping encode test" << std::endl;
        encoder.shutdown();
        GTEST_SKIP() << "No VB block available";
        return;
    }

    CVI_U64 phys_addr = CVI_VB_Handle2PhysAddr(blk);
    void* virt_addr = CVI_SYS_Mmap(phys_addr, total_size);
    if (!virt_addr) {
        CVI_VB_ReleaseBlock(blk);
        encoder.shutdown();
        GTEST_SKIP() << "Failed to mmap VB block";
        return;
    }

    // Fill Y plane with gray
    std::memset(virt_addr, 128, y_size);
    // Fill UV plane with neutral chroma
    std::memset(static_cast<uint8_t*>(virt_addr) + y_size, 128, uv_size);

    frame.stVFrame.enPixelFormat = PIXEL_FORMAT_NV21;
    frame.stVFrame.u32Width = width;
    frame.stVFrame.u32Height = height;
    frame.stVFrame.u32Stride[0] = stride;
    frame.stVFrame.u32Stride[1] = stride;
    frame.stVFrame.u64PhyAddr[0] = phys_addr;
    frame.stVFrame.u64PhyAddr[1] = phys_addr + y_size;
    frame.stVFrame.pu8VirAddr[0] = static_cast<CVI_U8*>(virt_addr);
    frame.stVFrame.pu8VirAddr[1] = static_cast<CVI_U8*>(virt_addr) + y_size;
    frame.stVFrame.u32Length[0] = y_size;
    frame.stVFrame.u32Length[1] = uv_size;

    CVI_SYS_IonFlushCache(phys_addr, virt_addr, total_size);

    bool send_ok = encoder.send_frame(frame, 1000);
    std::cout << "[TEST] send_frame result: " << (send_ok ? "OK" : "FAILED") << std::endl;
    ASSERT_TRUE(send_ok);

    // Query channel status after send
    VENC_CHN_STATUS_S chn_status;
    CVI_S32 rc = CVI_VENC_QueryStatus(encoder.channel(), &chn_status);
    if (rc == CVI_SUCCESS) {
        std::cout << "[TEST] VENC status: CurPacks=" << chn_status.u32CurPacks
                  << " LeftRecvPics=" << chn_status.u32LeftRecvPics
                  << " LeftEncPics=" << chn_status.u32LeftEncPics
                  << " LeftStreamFrames=" << chn_status.u32LeftStreamFrames
                  << std::endl;
    }

    VencEncoder::EncodedStream stream;
    bool got_stream = encoder.get_stream(&stream, 2000);
    std::cout << "[TEST] get_stream result: " << (got_stream ? "OK" : "FAILED")
              << " size=" << stream.data.size() << std::endl;

    CVI_SYS_Munmap(virt_addr, total_size);
    CVI_VB_ReleaseBlock(blk);

    ASSERT_TRUE(got_stream);
    EXPECT_GT(stream.data.size(), 0u);
    EXPECT_TRUE(stream.is_keyframe);

    if (stream.data.size() >= 2) {
        EXPECT_EQ(stream.data[0], 0xFF);
        EXPECT_EQ(stream.data[1], 0xD8);
    }

    encoder.release_stream();
    encoder.shutdown();
}

#endif

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
