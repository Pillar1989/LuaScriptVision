/**
 * test_common.h - Common test infrastructure for CV module tests
 *
 * Provides:
 * - TestSuite and TestResult for test management
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

#include <opencv2/opencv.hpp>

// CV module headers
#include "cv/cvi_frame.h"
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

// ============================================================
// Test Infrastructure
// ============================================================

struct TestResult {
    std::string name;
    bool passed;
    double time_ms;
    std::string message;
};

class TestSuite {
public:
    void add_result(const std::string& name, bool passed, double time_ms = 0.0,
                    const std::string& message = "");

    void print_summary() const;

    bool all_passed() const;

private:
    std::vector<TestResult> results_;
};

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
#endif
