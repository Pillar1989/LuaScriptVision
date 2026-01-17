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

// VB Pool ID definitions - used by VPSS to attach to correct pool
// These must match the order in init_cvi_system()
constexpr VB_POOL VB_POOL_1080P = 0;  // 1920x1080 RGB888
constexpr VB_POOL VB_POOL_720P = 1;   // 1280x720 RGB888
constexpr VB_POOL VB_POOL_AI = 2;     // 640x640 RGB888
constexpr VB_POOL VB_POOL_VGA = 3;    // 640x480 RGB888
constexpr VB_POOL VB_POOL_QVGA = 4;   // 320x240 RGB888

// Helper to find suitable pool for a given frame size
VB_POOL find_suitable_vb_pool(uint32_t width, uint32_t height, PIXEL_FORMAT_E fmt) {
    uint32_t needed_size = COMMON_GetPicBufferSize(width, height, fmt,
                            DATA_BITWIDTH_8, COMPRESS_MODE_NONE, 0);

    // Pool sizes (must match init_cvi_system configuration)
    static const struct { VB_POOL id; uint32_t size; } pools[] = {
        {VB_POOL_1080P, COMMON_GetPicBufferSize(1920, 1080, PIXEL_FORMAT_BGR_888, DATA_BITWIDTH_8, COMPRESS_MODE_NONE, 0)},
        {VB_POOL_720P,  COMMON_GetPicBufferSize(1280, 720,  PIXEL_FORMAT_BGR_888, DATA_BITWIDTH_8, COMPRESS_MODE_NONE, 0)},
        {VB_POOL_AI,    COMMON_GetPicBufferSize(640,  640,  PIXEL_FORMAT_BGR_888, DATA_BITWIDTH_8, COMPRESS_MODE_NONE, 0)},
        {VB_POOL_VGA,   COMMON_GetPicBufferSize(640,  480,  PIXEL_FORMAT_BGR_888, DATA_BITWIDTH_8, COMPRESS_MODE_NONE, 0)},
        {VB_POOL_QVGA,  COMMON_GetPicBufferSize(320,  240,  PIXEL_FORMAT_BGR_888, DATA_BITWIDTH_8, COMPRESS_MODE_NONE, 0)},
    };

    // Find smallest pool that can fit the frame (pools are sorted by size descending)
    for (int i = 4; i >= 0; --i) {
        if (pools[i].size >= needed_size) {
            return pools[i].id;
        }
    }

    // Default to largest pool
    return VB_POOL_1080P;
}

bool init_cvi_system() {
    std::cout << "[INIT] Initializing CVI VB and SYS modules..." << std::endl;

    // Step 1: Clean up any previous state (from crashed test or other process)
    CVI_SYS_Exit();
    CVI_VB_Exit();

    // Step 2: Configure VB pools using COMMON_GetPicBufferSize for correct size calculation
    // All pools are public pools - VPSS will attach to these, not create new dynamic pools
    VB_CONFIG_S vb_config;
    memset(&vb_config, 0, sizeof(vb_config));
    vb_config.u32MaxPoolCnt = 5;

    // Pool 0: 1920x1080 RGB888 (largest, for 1080p input/output)
    vb_config.astCommPool[0].u32BlkSize = COMMON_GetPicBufferSize(
        1920, 1080, PIXEL_FORMAT_BGR_888, DATA_BITWIDTH_8, COMPRESS_MODE_NONE, 0);
    vb_config.astCommPool[0].u32BlkCnt = 5;
    vb_config.astCommPool[0].enRemapMode = VB_REMAP_MODE_CACHED;

    // Pool 1: 1280x720 RGB888 (720p input)
    vb_config.astCommPool[1].u32BlkSize = COMMON_GetPicBufferSize(
        1280, 720, PIXEL_FORMAT_BGR_888, DATA_BITWIDTH_8, COMPRESS_MODE_NONE, 0);
    vb_config.astCommPool[1].u32BlkCnt = 3;
    vb_config.astCommPool[1].enRemapMode = VB_REMAP_MODE_CACHED;

    // Pool 2: 640x640 RGB888 (AI input size)
    vb_config.astCommPool[2].u32BlkSize = COMMON_GetPicBufferSize(
        640, 640, PIXEL_FORMAT_BGR_888, DATA_BITWIDTH_8, COMPRESS_MODE_NONE, 0);
    vb_config.astCommPool[2].u32BlkCnt = 5;
    vb_config.astCommPool[2].enRemapMode = VB_REMAP_MODE_CACHED;

    // Pool 3: 640x480 RGB888 (VGA output)
    vb_config.astCommPool[3].u32BlkSize = COMMON_GetPicBufferSize(
        640, 480, PIXEL_FORMAT_BGR_888, DATA_BITWIDTH_8, COMPRESS_MODE_NONE, 0);
    vb_config.astCommPool[3].u32BlkCnt = 3;
    vb_config.astCommPool[3].enRemapMode = VB_REMAP_MODE_CACHED;

    // Pool 4: 320x240 RGB888 (QVGA output)
    vb_config.astCommPool[4].u32BlkSize = COMMON_GetPicBufferSize(
        320, 240, PIXEL_FORMAT_BGR_888, DATA_BITWIDTH_8, COMPRESS_MODE_NONE, 0);
    vb_config.astCommPool[4].u32BlkCnt = 3;
    vb_config.astCommPool[4].enRemapMode = VB_REMAP_MODE_CACHED;

    CVI_S32 rc = CVI_VB_SetConfig(&vb_config);
    if (rc != CVI_SUCCESS) {
        std::cerr << "[ERROR] CVI_VB_SetConfig failed: " << rc << std::endl;
        return false;
    }

    rc = CVI_VB_Init();
    if (rc != CVI_SUCCESS) {
        std::cerr << "[ERROR] CVI_VB_Init failed (rc=" << rc << ")" << std::endl;
        return false;
    }

    rc = CVI_SYS_Init();
    if (rc != CVI_SUCCESS) {
        std::cerr << "[ERROR] CVI_SYS_Init failed: " << rc << std::endl;
        CVI_VB_Exit();
        return false;
    }

    std::cout << "[INIT] VB pools configured:" << std::endl;
    for (uint32_t i = 0; i < vb_config.u32MaxPoolCnt; ++i) {
        std::cout << "  Pool " << i << ": " << vb_config.astCommPool[i].u32BlkSize
                  << " bytes x " << vb_config.astCommPool[i].u32BlkCnt << " blocks" << std::endl;
    }

    std::cout << "[INIT] VB and SYS modules initialized successfully\n" << std::endl;
    return true;
}

void cleanup_cvi_system() {
    std::cout << "\n[CLEANUP] Shutting down CVI VB and SYS modules..." << std::endl;
    CVI_SYS_Exit();
    CVI_VB_Exit();
}
#endif
