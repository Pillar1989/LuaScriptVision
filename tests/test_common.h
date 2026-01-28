/**
 * test_common.h - Common test infrastructure for CV module tests
 *
 * Provides:
 * - gtest configuration helpers
 * - Timer and benchmark utilities
 * - Test image generation helpers
 * - Shared includes and forward declarations
 */

#pragma once

#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <string>
#include <cmath>
#include <cstring>
#include <functional>

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

// CV module headers
#include "cv/frame.h"
#include "cv/cv_types.h"
#include "cv/opencv_processor.h"
#include "cv/cv_helpers.h"

// Tensor module headers (needed for frame_to_tensor)
#include "tensor/tensor.h"

#ifdef USE_CVI_MPI
#include "cv/cvi_vpss_processor.h"
#include <cvi_sys.h>
#include <cvi_vb.h>
#endif

using namespace lua_cv;

struct TestConfig {
    std::string image_path;

    static TestConfig& instance();
};

void set_test_image_path(const std::string& path);
const std::string& test_image_path();

// ============================================================
// Performance Measurement
// ============================================================

class Timer {
public:
    void start();
    double elapsed_ms() const;

private:
    std::chrono::high_resolution_clock::time_point start_;
};

struct BenchmarkResult {
    std::string name;
    double avg_ms;
    double min_ms;
    double max_ms;
    double stddev_ms;
    int iterations;
};

BenchmarkResult run_benchmark(const std::string& name,
                              std::function<void()> func,
                              int iterations = 10);

void print_benchmark_result(const BenchmarkResult& result);

// ============================================================
// Test Image Generation
// ============================================================

cv::Mat create_test_image(int width, int height);

// ============================================================
// System Initialization (for USE_CVI_MPI builds)
// ============================================================

#ifdef USE_CVI_MPI
bool init_cvi_system();
void cleanup_cvi_system();
VB_POOL find_suitable_vb_pool(uint32_t width, uint32_t height, PIXEL_FORMAT_E fmt);
bool is_cvi_ready();
void register_cvi_environment();

class VbBlockGuard {
public:
    explicit VbBlockGuard(VB_BLK block = VB_INVALID_HANDLE) : block_(block) {}
    ~VbBlockGuard() { reset(); }

    VbBlockGuard(const VbBlockGuard&) = delete;
    VbBlockGuard& operator=(const VbBlockGuard&) = delete;

    VbBlockGuard(VbBlockGuard&& other) noexcept : block_(other.block_) {
        other.block_ = VB_INVALID_HANDLE;
    }
    VbBlockGuard& operator=(VbBlockGuard&& other) noexcept {
        if (this != &other) {
            reset();
            block_ = other.block_;
            other.block_ = VB_INVALID_HANDLE;
        }
        return *this;
    }

    void reset(VB_BLK block = VB_INVALID_HANDLE) {
        if (block_ != VB_INVALID_HANDLE) {
            CVI_S32 rc = CVI_VB_ReleaseBlock(block_);
            if (rc != CVI_SUCCESS) {
                std::cerr << "[WARN] VbBlockGuard: CVI_VB_ReleaseBlock failed: 0x"
                          << std::hex << rc << std::dec << std::endl;
            }
        }
        block_ = block;
    }

    VB_BLK get() const { return block_; }
    VB_BLK release() {
        VB_BLK tmp = block_;
        block_ = VB_INVALID_HANDLE;
        return tmp;
    }

private:
    VB_BLK block_ = VB_INVALID_HANDLE;
};
#endif
