/**
 * test_camera_capture.cpp - Camera Capture Tests for test_cv_func
 *
 * Tests Camera → VI → ISP → VPSS pipeline when ENABLE_CVI_CAMERA=ON
 * Requires MmfContext to be initialized by the test runner.
 *
 * Tests:
 * - Sensor auto-detection (OV5647, GC2053)
 * - Camera open/close
 * - Frame capture
 * - Capture benchmark (FPS measurement)
 */

#include "test_common.h"

#ifdef USE_CVI_MPI
#ifdef USE_CVI_CAMERA

#include <unistd.h>  // for usleep
#include <sstream>
#include "cv/cvi_camera.h"

using namespace lua_cv;

// Test configuration
static constexpr int BENCHMARK_FRAMES = 30;

static bool test_sensor_autodetect(TestSuite& suite) {
    Timer timer;
    timer.start();

    CviSensor sensor;
    bool result = sensor.init();

    double elapsed = timer.elapsed_ms();

    if (result) {
        std::string info = std::string(sensor.get_sensor_name()) + " " +
                          std::to_string(sensor.get_width()) + "x" +
                          std::to_string(sensor.get_height());
        suite.add_result("Camera: Sensor auto-detect", true, elapsed, info);
        sensor.cleanup();
        return true;
    } else {
        suite.add_result("Camera: Sensor auto-detect", false, elapsed,
                        "No sensor detected (OV5647/GC2053)");
        return false;
    }
}

static bool test_camera_open_close(TestSuite& suite, CviCamera& camera) {
    Timer timer;
    timer.start();

    bool result = camera.open();

    double elapsed = timer.elapsed_ms();

    if (result) {
        std::string info = std::string(camera.get_sensor_name()) + " " +
                          std::to_string(camera.width()) + "x" +
                          std::to_string(camera.height()) + " @ " +
                          std::to_string(static_cast<int>(camera.fps())) + " fps";
        suite.add_result("Camera: Open", true, elapsed, info);

        // Warmup delay - give ISP pipeline time to produce frames
        std::cout << "  [INFO] Waiting for ISP pipeline to stabilize..." << std::endl;
        usleep(500000);  // 500ms warmup

        return true;
    } else {
        suite.add_result("Camera: Open", false, elapsed, "Failed to open camera");
        return false;
    }
}

static bool test_camera_capture(TestSuite& suite, CviCamera& camera) {
    if (!camera.is_opened()) {
        suite.add_result("Camera: Single frame capture", false, 0, "Camera not opened");
        return false;
    }

    // Check runtime status BEFORE capture attempt
    std::cout << "\n=== Runtime Status Before Frame Capture ===" << std::endl;
    std::cout << "[SYS Binding Status]" << std::endl;
    system("cat /proc/cvitek/sys | grep -A 10 'BIND RELATION'");
    std::cout << "\n[VI CHN Status]" << std::endl;
    system("cat /proc/cvitek/vi | grep -A 5 'VI CHN STATUS'");
    std::cout << "\n[VPSS Work Status]" << std::endl;
    system("cat /proc/cvitek/vpss | grep -A 20 'WORK STATUS'");
    std::cout << "\n[VB Pool Usage]" << std::endl;
    system("cat /proc/cvitek/vb | grep -A 15 'PoolId.*0'");
    std::cout << "==========================================\n" << std::endl;

    if (!camera.wait_for_ready(3000)) {
        suite.add_result("Camera: Single frame capture", false, 0,
                         "Camera not ready (no frames)");
        return false;
    }

    Timer timer;
    Frame frame;

    timer.start();
    bool result = false;
    int attempts_used = 0;
    const int max_attempts = 5;
    for (int attempt = 0; attempt < max_attempts; ++attempt) {
        if (camera.read(frame)) {
            result = true;
            attempts_used = attempt + 1;
            break;
        }
        usleep(200000);
    }
    double elapsed = timer.elapsed_ms();

    if (result) {
        std::string info = std::to_string(frame.width()) + "x" +
                          std::to_string(frame.height()) +
                          (frame.has_physical_addr() ? " (zero-copy)" : " (copy)") +
                          ", attempts=" + std::to_string(attempts_used);
        suite.add_result("Camera: Single frame capture", true, elapsed, info);
        frame.release();
        return true;
    } else {
        suite.add_result("Camera: Single frame capture", false, elapsed, "Capture failed");

        // Check status AFTER failure to diagnose
        std::cout << "\n=== Status After Capture Failure ===" << std::endl;
        std::cout << "[VPSS Work Status]" << std::endl;
        system("cat /proc/cvitek/vpss | grep -A 5 'WORK STATUS'");
        std::cout << "\n[VB Pool 0 Detailed]" << std::endl;
        system("cat /proc/cvitek/vb | grep -A 20 'PoolId.*: 0'");
        std::cout << "=====================================\n" << std::endl;

        return false;
    }
}

static bool test_camera_vpss_tensor_pipeline(TestSuite& suite, CviCamera& camera) {
    if (!camera.is_opened()) {
        suite.add_result("Camera: VPSS pipeline to tensor", false, 0, "Camera not opened");
        return false;
    }

    Frame frame;
    bool got_frame = false;
    for (int attempt = 0; attempt < 5; ++attempt) {
        if (camera.read(frame)) {
            got_frame = true;
            break;
        }
        usleep(200000);
    }
    if (!got_frame) {
        suite.add_result("Camera: VPSS pipeline to tensor", false, 0, "Failed to capture frame");
        return false;
    }

    Timer timer;
    timer.start();

    std::string stage = "resize";
    double resize_ms = 0.0;
    double crop_ms = 0.0;
    double resize2_ms = 0.0;
    double cvt_ms = 0.0;
    double tensor_ms = 0.0;
    Timer stage_timer;
    try {
        CviVpssProcessor processor;
        stage_timer.start();
        processor.resize(frame, 640, 640);
        resize_ms = stage_timer.elapsed_ms();

        int crop_size = 600;
        int crop_x = (frame.width() - crop_size) / 2;
        int crop_y = (frame.height() - crop_size) / 2;
        stage = "crop";
        stage_timer.start();
        processor.crop(frame, crop_x, crop_y, crop_size, crop_size);
        crop_ms = stage_timer.elapsed_ms();
        stage = "resize_after_crop";
        stage_timer.start();
        processor.resize(frame, 640, 640);
        resize2_ms = stage_timer.elapsed_ms();
        stage = "cvtColor";
        stage_timer.start();
        processor.cvtColor(frame, ColorConversion::BGR2RGB);
        cvt_ms = stage_timer.elapsed_ms();

        std::vector<double> mean = {0.485, 0.456, 0.406};
        std::vector<double> std = {0.229, 0.224, 0.225};
        stage = "frame_to_tensor";
        stage_timer.start();
        auto tensor = cv_helpers::frame_to_tensor(frame, 1.0 / 255.0, mean, std);
        tensor_ms = stage_timer.elapsed_ms();

        bool ok = (tensor.ndim() == 4 &&
                   tensor.size(0) == 1 &&
                   tensor.size(1) == 3 &&
                   tensor.size(2) == 640 &&
                   tensor.size(3) == 640);

        std::ostringstream info;
        if (ok) {
            info << "Shape: [1,3,640,640], "
                 << "resize=" << std::fixed << std::setprecision(2) << resize_ms << "ms, "
                 << "crop=" << crop_ms << "ms, "
                 << "resize2=" << resize2_ms << "ms, "
                 << "cvt=" << cvt_ms << "ms, "
                 << "tensor=" << tensor_ms << "ms";
        } else {
            info << "Invalid tensor shape";
        }
        suite.add_result("Camera: VPSS pipeline to tensor", ok, timer.elapsed_ms(), info.str());
        frame.release();
        return ok;
    } catch (const std::exception& e) {
        std::string info = stage + ": " + e.what();
        if (info.find("CviVpssProcessor") != std::string::npos) {
            info = "VPSS stage failed: " + info;
        }
        suite.add_result("Camera: VPSS pipeline to tensor", false, timer.elapsed_ms(), info);
        frame.release();
        return false;
    }
}

static bool test_camera_benchmark(TestSuite& suite, CviCamera& camera) {
    if (!camera.is_opened()) {
        suite.add_result("Camera: Benchmark", false, 0, "Camera not opened");
        return false;
    }

    uint64_t total_us = 0;
    int successful = 0;

    for (int i = 0; i < BENCHMARK_FRAMES; i++) {
        Frame frame;
        Timer timer;
        timer.start();

        if (camera.read(frame)) {
            total_us += static_cast<uint64_t>(timer.elapsed_ms() * 1000);
            successful++;
            frame.release();
        }
    }

    if (successful > 0) {
        double avg_ms = total_us / (1000.0 * successful);
        double actual_fps = 1000.0 / avg_ms;

        std::string info = std::to_string(successful) + "/" +
                          std::to_string(BENCHMARK_FRAMES) + " frames, " +
                          std::to_string(actual_fps).substr(0, 5) + " fps";
        suite.add_result("Camera: Benchmark (" + std::to_string(BENCHMARK_FRAMES) + " frames)",
                        true, avg_ms, info);
        return true;
    } else {
        suite.add_result("Camera: Benchmark", false, 0, "No frames captured");
        return false;
    }
}

void run_camera_capture_tests(TestSuite& suite) {
    std::cout << "\n[Test] Camera Capture (VI → ISP → VPSS)" << std::endl;

    // NOTE: Camera tests require a single global VB/SYS init (MmfContext).
    // Do NOT call CVI_VB_Exit() here as it corrupts kernel state on fresh boot.

    // Test 1-3: Full camera pipeline (includes sensor detection + VB init)
    try {
        CviCamera::Config config;
        config.format = PixelFormat::NV21;

        CviCamera camera(config);

        if (test_camera_open_close(suite, camera)) {
            // Report sensor detection success as part of camera open
            suite.add_result("Camera: Sensor detected", true, 0,
                           std::string(camera.get_sensor_name()) + " " +
                           std::to_string(camera.width()) + "x" +
                           std::to_string(camera.height()));

            test_camera_capture(suite, camera);
            test_camera_vpss_tensor_pipeline(suite, camera);
            test_camera_benchmark(suite, camera);
            camera.release();
        } else {
            std::cout << "  ⊘  Skipping capture/benchmark (camera not opened)" << std::endl;
        }

    } catch (const std::exception& e) {
        suite.add_result("Camera: Exception", false, 0, e.what());
    }

    // Re-initialize VB for other tests (if any follow)
    std::cout << "  [INFO] Camera tests complete, VB will be re-initialized if needed" << std::endl;
}

#else // !USE_CVI_CAMERA

void run_camera_capture_tests(TestSuite& suite) {
    std::cout << "\n[Test] Camera Capture" << std::endl;
    std::cout << "  ⊘  Skipped (USE_CVI_CAMERA not defined)" << std::endl;
    std::cout << "       Build with: cmake -DENABLE_CVI_CAMERA=ON ..." << std::endl;
}

#endif // USE_CVI_CAMERA

#else // !USE_CVI_MPI

void run_camera_capture_tests(TestSuite& suite) {
    std::cout << "\n[Test] Camera Capture" << std::endl;
    std::cout << "  ⊘  Skipped (USE_CVI_MPI not defined, x86 build)" << std::endl;
}

#endif // USE_CVI_MPI
