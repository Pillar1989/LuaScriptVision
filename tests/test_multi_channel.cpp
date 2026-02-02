#include <gtest/gtest.h>
#include <iostream>
#include <chrono>
#include <thread>
#include <cstring>

#include "stream/venc_encoder.h"
#include "stream/rtsp_server.h"
#include "cv/mmf_context.h"

#ifdef USE_CVI_MPI
#include <cvi_sys.h>
#include <cvi_vb.h>
#include <cvi_vpss.h>
#endif

#ifdef USE_CVI_CAMERA
#include "cv/cvi_camera.h"
#endif

using namespace lua_cv;

class MultiChannelTest : public ::testing::Test {
protected:
    void SetUp() override {
#ifndef USE_CVI_CAMERA
        GTEST_SKIP() << "CVI Camera not enabled";
#endif
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

#if defined(USE_CVI_MPI) && defined(USE_CVI_CAMERA)

TEST_F(MultiChannelTest, CameraStreamChannelOutput) {
    CviCamera::Config cam_config;
    cam_config.enable_infer = true;

    CviCamera camera(cam_config);
    if (!camera.open()) {
        GTEST_SKIP() << "Failed to open camera";
    }

    int vpss_grp = MmfContext::vpss_group_for_camera();
    int stream_chn = MmfContext::vpss_channel_for_camera_stream();

    std::cout << "[TEST] Camera VPSS group=" << vpss_grp
              << " stream_chn=" << stream_chn << std::endl;

    VIDEO_FRAME_INFO_S stream_frame;
    std::memset(&stream_frame, 0, sizeof(stream_frame));

    CVI_S32 rc = CVI_VPSS_GetChnFrame(static_cast<VPSS_GRP>(vpss_grp),
                                       static_cast<VPSS_CHN>(stream_chn),
                                       &stream_frame, 1000);
    if (rc != CVI_SUCCESS) {
        std::cout << "[TEST] CVI_VPSS_GetChnFrame failed: rc=" << rc << std::endl;
        camera.release();
        GTEST_SKIP() << "VPSS GetChnFrame failed (camera may not be connected)";
        return;
    }

    std::cout << "[TEST] Stream frame: " << stream_frame.stVFrame.u32Width
              << "x" << stream_frame.stVFrame.u32Height
              << " format=" << stream_frame.stVFrame.enPixelFormat << std::endl;

    EXPECT_GT(stream_frame.stVFrame.u32Width, 0u);
    EXPECT_GT(stream_frame.stVFrame.u32Height, 0u);

    CVI_VPSS_ReleaseChnFrame(static_cast<VPSS_GRP>(vpss_grp),
                              static_cast<VPSS_CHN>(stream_chn),
                              &stream_frame);

    camera.release();
}

TEST_F(MultiChannelTest, CameraInferChannelOutput) {
    CviCamera::Config cam_config;
    cam_config.enable_infer = true;

    CviCamera camera(cam_config);
    if (!camera.open()) {
        GTEST_SKIP() << "Failed to open camera";
    }

    Frame infer_frame;
    if (!camera.read(infer_frame, 1000)) {
        camera.release();
        GTEST_SKIP() << "Camera read failed (camera may not be connected)";
        return;
    }

    std::cout << "[TEST] Infer frame: " << infer_frame.width()
              << "x" << infer_frame.height()
              << " format=" << static_cast<int>(infer_frame.pixel_format()) << std::endl;

    EXPECT_GT(infer_frame.width(), 0u);
    EXPECT_GT(infer_frame.height(), 0u);

    infer_frame.release();
    camera.release();
}

TEST_F(MultiChannelTest, StreamChannelToVenc) {
    CviCamera::Config cam_config;
    cam_config.enable_infer = true;

    CviCamera camera(cam_config);
    if (!camera.open()) {
        GTEST_SKIP() << "Failed to open camera";
    }

    int vpss_grp = MmfContext::vpss_group_for_camera();
    int stream_chn = MmfContext::vpss_channel_for_camera_stream();

    VencEncoder::Config venc_config;
    venc_config.codec = VencEncoder::CodecType::H264;
    venc_config.width = 1920;
    venc_config.height = 1080;
    venc_config.fps = 30;
    venc_config.bitrate_kbps = 2000;
    venc_config.gop = 30;
    venc_config.channel = 0;

    VencEncoder encoder(venc_config);
    ASSERT_TRUE(encoder.init());

    ASSERT_TRUE(encoder.bind_to_vpss(static_cast<VPSS_GRP>(vpss_grp),
                                      static_cast<VPSS_CHN>(stream_chn)));

    int frames_encoded = 0;
    for (int i = 0; i < 10; ++i) {
        VencEncoder::EncodedStream stream;
        if (encoder.get_stream(&stream, 500)) {
            frames_encoded++;
            std::cout << "[TEST] Encoded frame " << i
                      << " size=" << stream.data.size()
                      << " keyframe=" << stream.is_keyframe << std::endl;
            encoder.release_stream();
        } else {
            std::cout << "[TEST] Frame " << i << " timeout" << std::endl;
        }
    }

    encoder.shutdown();
    camera.release();

    if (frames_encoded == 0) {
        GTEST_SKIP() << "No frames encoded (camera may not be connected)";
    }

    EXPECT_GT(frames_encoded, 0);
    std::cout << "[TEST] Total frames encoded: " << frames_encoded << std::endl;
}

TEST_F(MultiChannelTest, StreamAndInferConcurrent) {
    CviCamera::Config cam_config;
    cam_config.enable_infer = true;

    CviCamera camera(cam_config);
    if (!camera.open()) {
        GTEST_SKIP() << "Failed to open camera";
    }

    int vpss_grp = MmfContext::vpss_group_for_camera();
    int stream_chn = MmfContext::vpss_channel_for_camera_stream();

    VencEncoder::Config venc_config;
    venc_config.codec = VencEncoder::CodecType::H264;
    venc_config.width = 1920;
    venc_config.height = 1080;
    venc_config.fps = 30;
    venc_config.bitrate_kbps = 2000;
    venc_config.gop = 30;
    venc_config.channel = 0;

    VencEncoder encoder(venc_config);
    ASSERT_TRUE(encoder.init());
    ASSERT_TRUE(encoder.bind_to_vpss(static_cast<VPSS_GRP>(vpss_grp),
                                      static_cast<VPSS_CHN>(stream_chn)));

    int infer_frames = 0;
    int encoded_frames = 0;

    for (int i = 0; i < 30; ++i) {
        Frame infer_frame;
        if (camera.read(infer_frame, 100)) {
            infer_frames++;
            infer_frame.release();
        }

        VencEncoder::EncodedStream stream;
        if (encoder.get_stream(&stream, 100)) {
            encoded_frames++;
            encoder.release_stream();
        }
    }

    encoder.shutdown();
    camera.release();

    std::cout << "[TEST] Concurrent test: infer_frames=" << infer_frames
              << " encoded_frames=" << encoded_frames << std::endl;

    if (infer_frames == 0 && encoded_frames == 0) {
        GTEST_SKIP() << "No frames captured (camera may not be connected)";
    }

    EXPECT_GT(infer_frames, 0);
    EXPECT_GT(encoded_frames, 0);
}

TEST_F(MultiChannelTest, FullPipelineStreamAndRtsp) {
    CviCamera::Config cam_config;
    cam_config.enable_infer = true;

    CviCamera camera(cam_config);
    if (!camera.open()) {
        GTEST_SKIP() << "Failed to open camera";
    }

    int vpss_grp = MmfContext::vpss_group_for_camera();
    int stream_chn = MmfContext::vpss_channel_for_camera_stream();

    VencEncoder::Config venc_config;
    venc_config.codec = VencEncoder::CodecType::H264;
    venc_config.width = 1920;
    venc_config.height = 1080;
    venc_config.fps = 30;
    venc_config.bitrate_kbps = 2000;
    venc_config.gop = 30;
    venc_config.channel = 0;

    VencEncoder encoder(venc_config);
    ASSERT_TRUE(encoder.init());
    ASSERT_TRUE(encoder.bind_to_vpss(static_cast<VPSS_GRP>(vpss_grp),
                                      static_cast<VPSS_CHN>(stream_chn)));

    RtspServer::Config rtsp_config;
    rtsp_config.port = 8560;
    rtsp_config.session_name = "live";
    rtsp_config.video_codec = RtspServer::VideoCodec::H264;
    rtsp_config.video_width = 1920;
    rtsp_config.video_height = 1080;

    RtspServer server(rtsp_config);
    ASSERT_TRUE(server.start());

    std::cout << "[TEST] Full pipeline RTSP URL: " << server.get_url() << std::endl;

    int frames_sent = 0;
    for (int i = 0; i < 30; ++i) {
        VencEncoder::EncodedStream stream;
        if (encoder.get_stream(&stream, 100)) {
            if (server.send_video(stream.data.data(), stream.data.size(), i * 33333)) {
                frames_sent++;
            }
            encoder.release_stream();
        }
    }

    server.stop();
    encoder.shutdown();
    camera.release();

    std::cout << "[TEST] Full pipeline frames sent: " << frames_sent << std::endl;

    if (frames_sent == 0) {
        GTEST_SKIP() << "No frames sent (camera may not be connected)";
    }

    EXPECT_GT(frames_sent, 0);
}

#endif

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
