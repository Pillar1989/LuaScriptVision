/**
 * test_common.cpp - Implementation of common test infrastructure
 */

#include "test_common.h"

// ============================================================
// TestSuite Implementation
// ============================================================

void TestSuite::add_result(const std::string& name, bool passed, double time_ms,
                          const std::string& message) {
    results_.push_back({name, passed, time_ms, message});

    std::cout << (passed ? "  ✓ " : "  ✗ ")
              << std::left << std::setw(55) << name;

    if (time_ms > 0) {
        std::cout << std::right << std::fixed << std::setprecision(3)
                  << std::setw(10) << time_ms << " ms";
    }

    if (!message.empty()) {
        std::cout << "  (" << message << ")";
    }

    std::cout << std::endl;
}

void TestSuite::print_summary() const {
    std::cout << "\n========== Test Summary ==========" << std::endl;

    int passed = 0;
    int failed = 0;
    double total_time = 0.0;

    for (const auto& r : results_) {
        if (r.passed) passed++;
        else failed++;
        total_time += r.time_ms;
    }

    std::cout << "Tests passed: " << passed << "/" << (passed + failed) << std::endl;

    if (failed > 0) {
        std::cout << "\nFailed tests:" << std::endl;
        for (const auto& r : results_) {
            if (!r.passed) {
                std::cout << "  - " << r.name;
                if (!r.message.empty()) {
                    std::cout << ": " << r.message;
                }
                std::cout << std::endl;
            }
        }
    }

    std::cout << "\nTotal time: " << std::fixed << std::setprecision(3)
              << total_time << " ms" << std::endl;
    std::cout << "==================================\n" << std::endl;
}

bool TestSuite::all_passed() const {
    for (const auto& r : results_) {
        if (!r.passed) return false;
    }
    return true;
}

// ============================================================
// Timer Implementation
// ============================================================

void Timer::start() {
    start_ = std::chrono::high_resolution_clock::now();
}

double Timer::elapsed_ms() const {
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start_).count();
}

// ============================================================
// Benchmark Implementation
// ============================================================

BenchmarkResult run_benchmark(const std::string& name,
                              std::function<void()> func,
                              int iterations) {
    std::vector<double> times;
    times.reserve(iterations);

    // Warmup (2 iterations)
    func();
    func();

    // Measure
    for (int i = 0; i < iterations; ++i) {
        Timer timer;
        timer.start();
        func();
        times.push_back(timer.elapsed_ms());
    }

    // Calculate statistics
    double sum = 0.0;
    double min_val = times[0];
    double max_val = times[0];

    for (double t : times) {
        sum += t;
        if (t < min_val) min_val = t;
        if (t > max_val) max_val = t;
    }

    double avg = sum / iterations;

    double variance = 0.0;
    for (double t : times) {
        variance += (t - avg) * (t - avg);
    }
    double stddev = std::sqrt(variance / iterations);

    return {name, avg, min_val, max_val, stddev, iterations};
}

void print_benchmark_result(const BenchmarkResult& result) {
    std::cout << "\n--- " << result.name << " ---" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  Average: " << result.avg_ms << " ms" << std::endl;
    std::cout << "  Min:     " << result.min_ms << " ms" << std::endl;
    std::cout << "  Max:     " << result.max_ms << " ms" << std::endl;
    std::cout << "  StdDev:  " << result.stddev_ms << " ms" << std::endl;
    std::cout << "  Iterations: " << result.iterations << std::endl;
}

// ============================================================
// Test Image Generation
// ============================================================

cv::Mat create_test_image(int width, int height) {
    cv::Mat img(height, width, CV_8UC3);

    // Create gradient pattern for visual verification
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            img.at<cv::Vec3b>(y, x) = cv::Vec3b(
                (y * 255) / height,           // B
                (x * 255) / width,            // G
                ((x + y) * 255) / (width + height)  // R
            );
        }
    }

    return img;
}

// ============================================================
// CVI System Initialization
// ============================================================

#ifdef USE_CVI_MPI
#include <cvi_buffer.h>  // For COMMON_GetPicBufferSize
#include "cv/mmf_context.h"
#ifdef USE_CVI_CAMERA
#include "cv/cvi_sensor.h"
#endif

// Helper to find suitable pool for a given frame size
VB_POOL find_suitable_vb_pool(uint32_t width, uint32_t height, PIXEL_FORMAT_E fmt) {
    uint32_t needed_size = COMMON_GetPicBufferSize(width, height, fmt,
                            DATA_BITWIDTH_8, COMPRESS_MODE_NONE, 0);

    if (CVI_VB_IsInited() != CVI_TRUE) {
        return VB_INVALID_POOLID;
    }

    VB_CONFIG_S vb_config;
    std::memset(&vb_config, 0, sizeof(vb_config));
    CVI_S32 rc = CVI_VB_GetConfig(&vb_config);
    if (rc != CVI_SUCCESS || vb_config.u32MaxPoolCnt == 0) {
        return VB_INVALID_POOLID;
    }

    VB_POOL best_pool = VB_INVALID_POOLID;
    uint32_t best_size = 0;

    for (uint32_t i = 0; i < vb_config.u32MaxPoolCnt; ++i) {
        uint32_t blk_size = vb_config.astCommPool[i].u32BlkSize;
        uint32_t blk_cnt = vb_config.astCommPool[i].u32BlkCnt;
        if (blk_size == 0 || blk_cnt == 0) {
            continue;
        }
        if (blk_size >= needed_size &&
            (best_pool == VB_INVALID_POOLID || blk_size < best_size)) {
            best_pool = static_cast<VB_POOL>(i);
            best_size = blk_size;
        }
    }

    return best_pool;
}

bool init_cvi_system() {
    std::cout << "[INIT] Initializing CVI VB and SYS modules..." << std::endl;

    uint32_t sensor_width = 1920;
    uint32_t sensor_height = 1080;
#ifdef USE_CVI_CAMERA
    CviSensor sensor;
    if (sensor.init()) {
        sensor_width = static_cast<uint32_t>(sensor.get_width());
        sensor_height = static_cast<uint32_t>(sensor.get_height());
        sensor.cleanup();
    }
#endif

    lua_cv::VbPoolPlan vb_plan;
    vb_plan.add_pool(sensor_width, sensor_height, lua_cv::PixelFormat::NV21, 3, VB_REMAP_MODE_CACHED);
    if (sensor_width != 1280 || sensor_height != 720) {
        vb_plan.add_pool(1280, 720, lua_cv::PixelFormat::NV21, 3, VB_REMAP_MODE_CACHED);
    }
    vb_plan.add_pool(1280, 720, lua_cv::PixelFormat::BGR, 2, VB_REMAP_MODE_CACHED);
    vb_plan.add_pool(640, 640, lua_cv::PixelFormat::BGR, 3, VB_REMAP_MODE_CACHED);
    vb_plan.add_pool(640, 480, lua_cv::PixelFormat::BGR, 2, VB_REMAP_MODE_CACHED);
    vb_plan.add_pool(320, 240, lua_cv::PixelFormat::BGR, 2, VB_REMAP_MODE_CACHED);

    lua_cv::MmfContext::Config config;
    config.vb_plan = vb_plan;
    config.force_reset = true;

    for (int i = 0; i < VI_MAX_PIPE_NUM; ++i) {
        config.mode_plan.vi_vpss_mode.aenMode[i] = VI_OFFLINE_VPSS_ONLINE;
    }

    config.mode_plan.vpss_mode.enMode = VPSS_MODE_DUAL;
    for (int i = 0; i < VPSS_IP_NUM; ++i) {
        config.mode_plan.vpss_mode.aenInput[i] = (i == 0) ? VPSS_INPUT_MEM : VPSS_INPUT_ISP;
        config.mode_plan.vpss_mode.ViPipe[i] = 0;
    }

    if (!lua_cv::MmfContext::instance().init(config)) {
        std::cerr << "[ERROR] MmfContext init failed" << std::endl;
        return false;
    }

    std::cout << "[INIT] VB and SYS modules initialized successfully\n" << std::endl;
    return true;
}

void cleanup_cvi_system() {
    std::cout << "\n[CLEANUP] Shutting down CVI VB and SYS modules..." << std::endl;
    lua_cv::MmfContext::instance().shutdown();
}
#endif
