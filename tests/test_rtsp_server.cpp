#include <gtest/gtest.h>
#include <iostream>
#include <chrono>
#include <thread>
#include <atomic>
#include <cstring>

#include "stream/rtsp_server.h"
#include "stream/venc_encoder.h"
#include "cv/mmf_context.h"

#ifdef USE_CVI_MPI
#include <cvi_sys.h>
#include <cvi_vb.h>
#endif

#ifndef ALIGN
#define ALIGN(x, a) (((x) + (a) - 1) & ~((a) - 1))
#endif

using namespace lua_cv;

class RtspServerTest : public ::testing::Test {
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

TEST_F(RtspServerTest, StartStop) {
    RtspServer::Config config;
    config.port = 8554;
    config.session_name = "test";

    RtspServer server(config);
    EXPECT_FALSE(server.is_running());

    ASSERT_TRUE(server.start());
    EXPECT_TRUE(server.is_running());

    std::string url = server.get_url();
    std::cout << "[TEST] Server URL: " << url << std::endl;
    EXPECT_NE(url.find("rtsp://"), std::string::npos);
    EXPECT_NE(url.find(":8554"), std::string::npos);
    EXPECT_NE(url.find("/test"), std::string::npos);

    server.stop();
    EXPECT_FALSE(server.is_running());
}

TEST_F(RtspServerTest, H265Codec) {
    RtspServer::Config config;
    config.port = 8555;
    config.session_name = "h265";
    config.video_codec = RtspServer::VideoCodec::H265;

    RtspServer server(config);
    ASSERT_TRUE(server.start());
    EXPECT_TRUE(server.is_running());

    std::cout << "[TEST] H.265 Server URL: " << server.get_url() << std::endl;

    server.stop();
}

TEST_F(RtspServerTest, ClientCallback) {
    RtspServer::Config config;
    config.port = 8556;
    config.session_name = "callback";

    RtspServer server(config);

    std::atomic<bool> connected{false};
    std::atomic<bool> disconnected{false};

    server.set_on_client_connect([&](const std::string& ip) {
        std::cout << "[TEST] Client connected from: " << ip << std::endl;
        connected = true;
    });

    server.set_on_client_disconnect([&](const std::string& ip) {
        std::cout << "[TEST] Client disconnected from: " << ip << std::endl;
        disconnected = true;
    });

    ASSERT_TRUE(server.start());
    std::cout << "[TEST] Callback Server URL: " << server.get_url() << std::endl;

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    server.stop();
}

TEST_F(RtspServerTest, SendVideoFrame) {
    VencEncoder::Config venc_config;
    venc_config.codec = VencEncoder::CodecType::H264;
    venc_config.width = 1280;
    venc_config.height = 720;
    venc_config.fps = 30;
    venc_config.bitrate_kbps = 2000;
    venc_config.gop = 30;
    venc_config.channel = 0;

    VencEncoder encoder(venc_config);
    ASSERT_TRUE(encoder.init());

    RtspServer::Config rtsp_config;
    rtsp_config.port = 8557;
    rtsp_config.session_name = "video";
    rtsp_config.video_codec = RtspServer::VideoCodec::H264;
    rtsp_config.video_width = 1280;
    rtsp_config.video_height = 720;

    RtspServer server(rtsp_config);
    ASSERT_TRUE(server.start());

    std::cout << "[TEST] Video Server URL: " << server.get_url() << std::endl;

    VIDEO_FRAME_INFO_S frame;
    std::memset(&frame, 0, sizeof(frame));

    uint32_t width = venc_config.width;
    uint32_t height = venc_config.height;
    uint32_t stride = ALIGN(width, 64);
    uint32_t y_size = stride * height;
    uint32_t uv_size = y_size / 2;
    uint32_t total_size = y_size + uv_size;

    VB_BLK blk = CVI_VB_GetBlock(VB_INVALID_POOLID, total_size);
    if (blk == VB_INVALID_HANDLE) {
        std::cout << "[TEST] Could not allocate VB block, skipping" << std::endl;
        server.stop();
        encoder.shutdown();
        GTEST_SKIP() << "No VB block available";
        return;
    }

    CVI_U64 phys_addr = CVI_VB_Handle2PhysAddr(blk);
    void* virt_addr = CVI_SYS_Mmap(phys_addr, total_size);
    if (!virt_addr) {
        CVI_VB_ReleaseBlock(blk);
        server.stop();
        encoder.shutdown();
        GTEST_SKIP() << "Failed to mmap VB block";
        return;
    }

    std::memset(virt_addr, 128, y_size);
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

    int frames_sent = 0;
    for (int i = 0; i < 5; ++i) {
        if (encoder.send_frame(frame, 1000)) {
            VencEncoder::EncodedStream stream;
            if (encoder.get_stream(&stream, 1000)) {
                bool sent = server.send_video(stream.data.data(), stream.data.size(), i * 33333);
                if (sent) {
                    frames_sent++;
                    std::cout << "[TEST] Frame " << i << " sent, size=" << stream.data.size()
                              << " keyframe=" << stream.is_keyframe << std::endl;
                }
                encoder.release_stream();
            }
        }
    }

    CVI_SYS_Munmap(virt_addr, total_size);
    CVI_VB_ReleaseBlock(blk);

    EXPECT_GT(frames_sent, 0);
    std::cout << "[TEST] Total frames sent: " << frames_sent << std::endl;

    server.stop();
    encoder.shutdown();
}

#endif

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
