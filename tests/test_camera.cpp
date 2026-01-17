/**
 * test_camera.cpp - Camera → VI → ISP → VPSS Pipeline Test
 *
 * Tests the camera capture pipeline:
 *   Camera Sensor → VI → ISP → VPSS → Output frame
 *
 * Resource allocation:
 *   - CviSensor: VI_DEV=0, VI_PIPE=0, VI_CHN=0
 *   - CviCamera: VPSS_GRP=0, VPSS_CHN=0
 *
 * Supported sensors: OV5647, GC2053 (auto-detected)
 *
 * Build:
 *   cmake -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-sg200x.cmake -DENABLE_CVI_CAMERA=ON -B build
 *   cmake --build build -j16
 *
 * Deploy & Run:
 *   sshpass -p '11' scp build/test_camera recamera@192.168.42.1:/tmp/
 *   sshpass -p '11' ssh recamera@192.168.42.1 'echo "11" | sudo -S /tmp/test_camera'
 *
 * Note: Requires camera sensor connected and proper ISP libraries linked.
 */

#include <iostream>
#include <cstring>
#include <chrono>

#ifdef USE_CVI_MPI
#ifdef USE_CVI_CAMERA

#include "modules/cv/cvi_camera.h"

#include <cvi_sys.h>
#include <cvi_vb.h>

using namespace lua_cv;

// Test configuration
static constexpr int CAPTURE_FRAMES = 30;

bool test_camera_open(CviCamera& camera) {
    std::cout << "\n=== Test: Camera Open ===" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    bool result = camera.open();
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (result) {
        std::cout << "[PASS] Camera opened successfully" << std::endl;
        std::cout << "  - Open time: " << duration.count() << " ms" << std::endl;
        std::cout << "  - Sensor: " << camera.get_sensor_name() << std::endl;
        std::cout << "  - Resolution: " << camera.width() << "x" << camera.height() << std::endl;
        std::cout << "  - FPS: " << camera.fps() << std::endl;
        return true;
    } else {
        std::cerr << "[FAIL] Failed to open camera" << std::endl;
        return false;
    }
}

bool test_camera_capture(CviCamera& camera) {
    std::cout << "\n=== Test: Camera Capture ===" << std::endl;

    if (!camera.is_opened()) {
        std::cerr << "[SKIP] Camera not opened" << std::endl;
        return false;
    }

    Frame frame;
    auto start = std::chrono::high_resolution_clock::now();
    bool result = camera.read(frame);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (result) {
        std::cout << "[PASS] Frame captured successfully" << std::endl;
        std::cout << "  - Capture time: " << duration.count() << " ms" << std::endl;
        std::cout << "  - Frame size: " << frame.width() << "x" << frame.height() << std::endl;
        std::cout << "  - Has physical addr: " << (frame.has_physical_addr() ? "yes" : "no") << std::endl;

        // Release frame
        frame.release();
        return true;
    } else {
        std::cerr << "[FAIL] Failed to capture frame" << std::endl;
        return false;
    }
}

bool test_camera_benchmark(CviCamera& camera, int frames = CAPTURE_FRAMES) {
    std::cout << "\n=== Test: Camera Benchmark (" << frames << " frames) ===" << std::endl;

    if (!camera.is_opened()) {
        std::cerr << "[SKIP] Camera not opened" << std::endl;
        return false;
    }

    uint64_t total_capture_us = 0;
    int successful = 0;

    for (int i = 0; i < frames; i++) {
        Frame frame;

        auto t1 = std::chrono::high_resolution_clock::now();
        bool result = camera.read(frame);
        auto t2 = std::chrono::high_resolution_clock::now();

        if (result) {
            total_capture_us += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
            successful++;
            frame.release();
        } else {
            std::cerr << "  - Frame " << i << " capture failed" << std::endl;
        }
    }

    if (successful > 0) {
        double avg_capture_ms = total_capture_us / (1000.0 * successful);
        double actual_fps = 1000.0 / avg_capture_ms;

        std::cout << "[PASS] Camera benchmark completed" << std::endl;
        std::cout << "  - Successful frames: " << successful << "/" << frames << std::endl;
        std::cout << "  - Avg capture time: " << avg_capture_ms << " ms" << std::endl;
        std::cout << "  - Actual FPS: " << actual_fps << std::endl;

        return true;
    } else {
        std::cerr << "[FAIL] No frames captured successfully" << std::endl;
        return false;
    }
}

bool test_sensor_autodetect() {
    std::cout << "\n=== Test: Sensor Auto-detect ===" << std::endl;

    CviSensor sensor;
    bool result = sensor.init();

    if (result) {
        std::cout << "[PASS] Sensor auto-detected" << std::endl;
        std::cout << "  - Type: " << sensor.get_sensor_name() << std::endl;
        std::cout << "  - Resolution: " << sensor.get_width() << "x" << sensor.get_height() << std::endl;
        std::cout << "  - VI Pipe: " << sensor.get_vi_pipe() << std::endl;
        std::cout << "  - VI Channel: " << sensor.get_vi_chn() << std::endl;

        sensor.cleanup();
        return true;
    } else {
        std::cerr << "[FAIL] No sensor detected" << std::endl;
        std::cerr << "  - Check if camera sensor is connected" << std::endl;
        std::cerr << "  - Supported: OV5647, GC2053" << std::endl;
        return false;
    }
}

int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "  Camera → VI → ISP → VPSS Test" << std::endl;
    std::cout << "========================================" << std::endl;

    int passed = 0;
    int failed = 0;

    // Test 1: Sensor auto-detect (standalone test, no camera init)
    // Note: This test initializes/cleanups sensor separately
    if (test_sensor_autodetect()) passed++; else failed++;

    try {
        // Test 2-4: Full camera pipeline
        CviCamera::Config config;
        config.format = PixelFormat::NV21;

        CviCamera camera(config);

        if (test_camera_open(camera)) {
            passed++;

            if (test_camera_capture(camera)) passed++; else failed++;
            if (test_camera_benchmark(camera, CAPTURE_FRAMES)) passed++; else failed++;

            camera.release();
        } else {
            failed++;
            std::cerr << "[SKIP] Skipping capture/benchmark tests (camera not opened)" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Unexpected exception: " << e.what() << std::endl;
        failed++;
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "  Test Results: " << passed << " passed, " << failed << " failed" << std::endl;
    std::cout << "========================================" << std::endl;

    return (failed > 0) ? 1 : 0;
}

#else // !USE_CVI_CAMERA

#include <iostream>

int main() {
    std::cerr << "[SKIP] This test requires USE_CVI_CAMERA" << std::endl;
    std::cerr << "       Build with: cmake -DENABLE_CVI_CAMERA=ON ..." << std::endl;
    return 0;
}

#endif // USE_CVI_CAMERA

#else // !USE_CVI_MPI

#include <iostream>

int main() {
    std::cerr << "[SKIP] This test requires CVI MPI (SG200X platform)" << std::endl;
    return 0;
}

#endif // USE_CVI_MPI
