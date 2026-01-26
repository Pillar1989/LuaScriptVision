/**
 * test_camera_capture.cpp - Camera capture tests for gtest
 */

#include "test_common.h"

#ifdef USE_CVI_MPI
#ifdef USE_CVI_CAMERA

#include <unistd.h>
#include <sstream>
#include <memory>
#include "cv/cvi_camera.h"
#include "cv/cvi_sensor.h"
#include "cv/mmf_context.h"
#include <cvi_vpss.h>

using namespace lua_cv;

static constexpr int kBenchmarkFrames = 30;

class CameraCaptureTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        if (!is_cvi_ready()) {
            camera_ready_ = false;
            camera_error_ = "CVI system not initialized";
            return;
        }

        CviCamera::Config config;
        config.format = PixelFormat::BGR;
        config.enable_infer = true;
        camera_.reset(new CviCamera(config));

        if (!camera_->open()) {
            camera_ready_ = false;
            camera_error_ = "Failed to open camera";
            return;
        }

        camera_ready_ = true;
        usleep(500000);
    }

    static void TearDownTestSuite() {
        if (camera_) {
            camera_->release();
            camera_.reset();
        }
        camera_ready_ = false;
    }

    static void RequireCameraReady() {
        if (!camera_ready_) {
            GTEST_SKIP() << camera_error_;
        }
    }

    static std::unique_ptr<CviCamera> camera_;
    static bool camera_ready_;
    static std::string camera_error_;
};

std::unique_ptr<CviCamera> CameraCaptureTest::camera_;
bool CameraCaptureTest::camera_ready_ = false;
std::string CameraCaptureTest::camera_error_;

TEST_F(CameraCaptureTest, SensorDetect) {
    if (!is_cvi_ready()) {
        GTEST_SKIP() << "CVI system not initialized";
    }

    CviSensor sensor;
    bool ok = sensor.init();
    EXPECT_TRUE(ok);
    if (ok) {
        std::cout << "[Camera] Sensor: " << sensor.get_sensor_name() << " "
                  << sensor.get_width() << "x" << sensor.get_height() << std::endl;
        sensor.cleanup();
    }
}

TEST_F(CameraCaptureTest, Opened) {
    RequireCameraReady();
    EXPECT_TRUE(camera_->is_opened());
}

// 场景1: Camera→Stream - VI → VPSS → chn0 1080p NV21
TEST_F(CameraCaptureTest, StreamChannel) {
    RequireCameraReady();

    if (!camera_->wait_for_ready(3000)) {
        GTEST_SKIP() << "Camera not ready";
    }

    // 获取 stream channel 配置
    int stream_chn = MmfContext::vpss_channel_for_camera_stream();
    int vpss_grp = MmfContext::vpss_group_for_camera();
    ASSERT_GE(stream_chn, 0) << "Stream channel not configured";
    ASSERT_GE(vpss_grp, 0) << "VPSS group not configured";

    // 直接从 stream channel 获取帧
    VIDEO_FRAME_INFO_S frame{};
    CVI_S32 rc = CVI_VPSS_GetChnFrame(
        static_cast<VPSS_GRP>(vpss_grp),
        static_cast<VPSS_CHN>(stream_chn),
        &frame, 1000);

    ASSERT_EQ(rc, CVI_SUCCESS) << "GetChnFrame failed: 0x" << std::hex << rc;

    // 验证 stream 输出格式 (1080p NV21)
    std::cout << "[Stream] " << frame.stVFrame.u32Width << "x"
              << frame.stVFrame.u32Height
              << " fmt=" << frame.stVFrame.enPixelFormat << std::endl;

    EXPECT_EQ(frame.stVFrame.u32Width, 1920);
    EXPECT_EQ(frame.stVFrame.u32Height, 1080);
    EXPECT_EQ(frame.stVFrame.enPixelFormat, PIXEL_FORMAT_NV21);

    CVI_VPSS_ReleaseChnFrame(
        static_cast<VPSS_GRP>(vpss_grp),
        static_cast<VPSS_CHN>(stream_chn),
        &frame);
}

// 场景2: Camera→Infer - VI → VPSS → chn1 640×640 BGR
TEST_F(CameraCaptureTest, InferChannel) {
    RequireCameraReady();

    if (!camera_->wait_for_ready(3000)) {
        GTEST_SKIP() << "Camera not ready";
    }

    Frame frame;
    bool got_frame = false;
    for (int attempt = 0; attempt < 5; ++attempt) {
        if (camera_->read(frame)) {
            got_frame = true;
            break;
        }
        usleep(200000);
    }

    ASSERT_TRUE(got_frame) << "Failed to read from infer channel";

    // 验证 infer 输出格式 (640×640 BGR)
    std::cout << "[Infer] " << frame.width() << "x" << frame.height()
              << " fmt=" << static_cast<int>(frame.pixel_format())
              << (frame.has_physical_addr() ? " (zero-copy)" : "") << std::endl;

    EXPECT_EQ(frame.width(), 640);
    EXPECT_EQ(frame.height(), 640);
    EXPECT_EQ(frame.pixel_format(), PixelFormat::BGR);
    EXPECT_TRUE(frame.has_physical_addr());

    frame.release();
}

TEST_F(CameraCaptureTest, SingleFrameCapture) {
    RequireCameraReady();

    std::cout << "\n[Camera] Runtime status before capture" << std::endl;
    system("cat /proc/cvitek/sys | grep -A 10 'BIND RELATION'");
    system("cat /proc/cvitek/vi | grep -A 5 'VI CHN STATUS'");
    system("cat /proc/cvitek/vpss | grep -A 20 'WORK STATUS'");
    system("cat /proc/cvitek/vb | grep -A 15 'PoolId.*0'");

    if (!camera_->wait_for_ready(3000)) {
        GTEST_SKIP() << "Camera not ready";
    }

    Frame frame;
    bool result = false;
    int attempts_used = 0;
    const int max_attempts = 5;

    for (int attempt = 0; attempt < max_attempts; ++attempt) {
        if (camera_->read(frame)) {
            result = true;
            attempts_used = attempt + 1;
            break;
        }
        usleep(200000);
    }

    if (!result) {
        std::cout << "\n[Camera] Status after capture failure" << std::endl;
        system("cat /proc/cvitek/vpss | grep -A 5 'WORK STATUS'");
        system("cat /proc/cvitek/vb | grep -A 20 'PoolId.*: 0'");
    }

    EXPECT_TRUE(result);
    if (result) {
        std::cout << "[Camera] Frame: " << frame.width() << "x" << frame.height()
                  << ", attempts=" << attempts_used
                  << (frame.has_physical_addr() ? " (zero-copy)" : " (copy)")
                  << std::endl;
        frame.release();
    }
}

TEST_F(CameraCaptureTest, VpssPipelineToTensor) {
    RequireCameraReady();

    if (!camera_->wait_for_ready(3000)) {
        GTEST_SKIP() << "Camera not ready";
    }

    Frame frame;
    bool got_frame = false;
    for (int attempt = 0; attempt < 5; ++attempt) {
        if (camera_->read(frame)) {
            got_frame = true;
            break;
        }
        usleep(200000);
    }

    ASSERT_TRUE(got_frame) << "Failed to capture frame";

    CviVpssProcessor processor;
    processor.resize(frame, 640, 640);

    int crop_size = 600;
    int crop_x = (frame.width() - crop_size) / 2;
    int crop_y = (frame.height() - crop_size) / 2;
    processor.crop(frame, crop_x, crop_y, crop_size, crop_size);
    processor.resize(frame, 640, 640);
    processor.cvtColor(frame, ColorConversion::BGR2RGB);

    std::vector<double> mean = {0.485, 0.456, 0.406};
    std::vector<double> std = {0.229, 0.224, 0.225};
    auto tensor = cv_helpers::frame_to_tensor(frame, 1.0 / 255.0, mean, std);

    EXPECT_EQ(tensor.ndim(), 4);
    EXPECT_EQ(tensor.size(0), 1);
    EXPECT_EQ(tensor.size(1), 3);
    EXPECT_EQ(tensor.size(2), 640);
    EXPECT_EQ(tensor.size(3), 640);

    frame.release();
}

TEST_F(CameraCaptureTest, Benchmark) {
    RequireCameraReady();

    if (!camera_->wait_for_ready(3000)) {
        GTEST_SKIP() << "Camera not ready";
    }

    uint64_t total_us = 0;
    int successful = 0;

    for (int i = 0; i < kBenchmarkFrames; i++) {
        Frame frame;
        Timer timer;
        timer.start();

        if (camera_->read(frame, 200)) {
            total_us += static_cast<uint64_t>(timer.elapsed_ms() * 1000);
            successful++;
            frame.release();
        }
    }

    ASSERT_GT(successful, 0);

    double avg_ms = total_us / (1000.0 * successful);
    double actual_fps = 1000.0 / avg_ms;

    std::cout << "[Camera] Benchmark: " << successful << "/" << kBenchmarkFrames
              << " frames, " << std::fixed << std::setprecision(2)
              << actual_fps << " fps" << std::endl;
}

#else

TEST(CameraCaptureTest, Skipped) {
    GTEST_SKIP() << "USE_CVI_CAMERA not defined";
}

#endif
#else

TEST(CameraCaptureTest, Skipped) {
    GTEST_SKIP() << "USE_CVI_MPI not defined";
}

#endif
