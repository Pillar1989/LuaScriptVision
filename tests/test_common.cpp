/**
 * test_common.cpp - Implementation of common test infrastructure
 */

#include "test_common.h"
#include <cstdlib>
#include <fstream>
#include <sstream>

// ============================================================
// Test Config
// ============================================================

TestConfig& TestConfig::instance() {
    static TestConfig config;
    return config;
}

void set_test_image_path(const std::string& path) {
    TestConfig::instance().image_path = path;
}

const std::string& test_image_path() {
    return TestConfig::instance().image_path;
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

namespace {
bool g_cvi_ready = false;

bool vb_trace_enabled() {
    static int cached = -1;
    if (cached != -1) {
        return cached == 1;
    }
    const char* env = std::getenv("LUA_VB_TRACE");
    if (!env || env[0] == '\0') {
        cached = 1;
        return true;
    }
    if (env[0] == '0' || env[0] == 'n' || env[0] == 'N') {
        cached = 0;
        return false;
    }
    if (std::strcmp(env, "false") == 0 || std::strcmp(env, "FALSE") == 0) {
        cached = 0;
        return false;
    }
    cached = 1;
    return true;
}

void dump_vb_proc(const std::string& tag) {
    if (!vb_trace_enabled()) {
        return;
    }
    std::ifstream file("/proc/cvitek/vb");
    if (!file) {
        std::cerr << "[VBTRACE] " << tag << " (cannot open /proc/cvitek/vb)" << std::endl;
        return;
    }

    std::ostringstream oss;
    std::string line;
    size_t total = 0;
    const size_t max_bytes = 4096;
    while (std::getline(file, line)) {
        if (total + line.size() + 1 > max_bytes) {
            line.resize(max_bytes - total);
        }
        oss << line << "\n";
        total += line.size() + 1;
        if (total >= max_bytes) {
            break;
        }
    }
    std::cerr << "[VBTRACE] " << tag << "\n" << oss.str() << std::endl;
}
}

class VbTraceListener : public ::testing::EmptyTestEventListener {
public:
    void OnTestStart(const ::testing::TestInfo& info) override {
        if (!vb_trace_enabled() || !g_cvi_ready) {
            return;
        }
        std::ostringstream tag;
        tag << "START " << info.test_suite_name() << "." << info.name();
        dump_vb_proc(tag.str());
    }

    void OnTestEnd(const ::testing::TestInfo& info) override {
        if (!vb_trace_enabled() || !g_cvi_ready) {
            return;
        }
        std::ostringstream tag;
        tag << "END " << info.test_suite_name() << "." << info.name()
            << " (" << (info.result()->Passed() ? "PASS" : "FAIL") << ")";
        dump_vb_proc(tag.str());
    }
};

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

    if (!lua_cv::MmfContext::is_supported_camera_size(sensor_width, sensor_height)) {
        std::cerr << "[ERROR] Unsupported camera resolution " << sensor_width
                  << "x" << sensor_height << std::endl;
        return false;
    }

    lua_cv::MmfContext::Config config;
    config.force_reset = true;
    if (!lua_cv::MmfContext::build_default_config(&config)) {
        std::cerr << "[ERROR] Failed to build MMF default config" << std::endl;
        return false;
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

bool is_cvi_ready() {
    return g_cvi_ready;
}

class CviTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        g_cvi_ready = init_cvi_system();
        if (!g_cvi_ready) {
            std::cerr << "[WARN] CVI init failed; CVI-dependent tests will be skipped."
                      << std::endl;
        }
    }

    void TearDown() override {
        if (g_cvi_ready) {
            cleanup_cvi_system();
            g_cvi_ready = false;
        }
    }
};

void register_cvi_environment() {
    ::testing::AddGlobalTestEnvironment(new CviTestEnvironment());
    if (vb_trace_enabled()) {
        ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();
        listeners.Append(new VbTraceListener());
    }
}
#endif
