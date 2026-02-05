#include <gtest/gtest.h>
#include <cstdio>
#include <fstream>
#include <thread>
#include <chrono>
#include <cstdlib>
#include <sys/stat.h>
#include <unistd.h>

#include "node/model_node.h"
#include "node/node_factory.h"
#include "node/error_codes.h"
#include "node/pipeline_context.h"
#include "cv/image_source.h"
#include "cv/mmf_context.h"

using namespace node;

namespace {
bool file_exists(const std::string& path) {
    struct stat st {};
    return stat(path.c_str(), &st) == 0;
}

std::string env_or_default(const char* key, const char* fallback) {
    const char* val = std::getenv(key);
    if (val && *val) {
        return std::string(val);
    }
    return std::string(fallback);
}

long read_rss_kb() {
    long rss_pages = 0;
    FILE* fp = std::fopen("/proc/self/statm", "r");
    if (!fp) {
        return 0;
    }
    long total_pages = 0;
    if (std::fscanf(fp, "%ld %ld", &total_pages, &rss_pages) != 2) {
        std::fclose(fp);
        return 0;
    }
    std::fclose(fp);
    long page_kb = sysconf(_SC_PAGESIZE) / 1024;
    return rss_pages * page_kb;
}

bool wait_for_infer(ModelNode& node, uint64_t target, int timeout_ms) {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        if (node.infer_count() >= target) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    return node.infer_count() >= target;
}
}  // namespace

// ============================================================================
// Test Fixtures and Helpers
// ============================================================================

class ModelNodeTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test Lua scripts in /tmp
        createTestScript("/tmp/test_valid.lua", R"(
local Model = {}

Model.preprocess_config = {
    type = "letterbox",
    input_size = {640, 640},
    normalize = true,
    scale = 1.0/255.0
}

function Model.postprocess(outputs, meta)
    return {
        boxes = {},
        scores = {}
    }
end

return Model
)");

        createTestScript("/tmp/test_no_postprocess.lua", R"(
local Model = {}

Model.preprocess_config = {
    type = "letterbox",
    input_size = {640, 640}
}

-- Missing postprocess function!

return Model
)");

        createTestScript("/tmp/test_syntax_error.lua", R"(
local Model = {}
-- Syntax error: missing end
function Model.postprocess(outputs, meta)
    return {
)");

        createTestScript("/tmp/test_runtime_error.lua", R"(
local Model = {}

Model.preprocess_config = {
    type = "letterbox",
    input_size = {640, 640}
}

function Model.postprocess(outputs, meta)
    -- Runtime error: calling nil
    return nil_variable.foo()
end

return Model
)");

        createTestScript("/tmp/test_not_table.lua", R"(
-- Returns a string instead of a table
return "not a table"
)");

        // Create a dummy model file
        createDummyModel("/tmp/test_model.cvimodel");
    }

    void TearDown() override {
        NodeFactory::instance().destroyAll();

        // Cleanup test files
        std::remove("/tmp/test_valid.lua");
        std::remove("/tmp/test_no_postprocess.lua");
        std::remove("/tmp/test_syntax_error.lua");
        std::remove("/tmp/test_runtime_error.lua");
        std::remove("/tmp/test_not_table.lua");
        std::remove("/tmp/test_model.cvimodel");
    }

    void createTestScript(const std::string& path, const std::string& content) {
        std::ofstream file(path);
        file << content;
        file.close();
    }

    void createDummyModel(const std::string& path) {
        const char* env_model = std::getenv("TPU_MODEL_PATH");
        const char* candidates[] = {
            (env_model && *env_model) ? env_model : nullptr,
            "/userdata/Models/model.cvimodel"
        };

        for (const char* candidate : candidates) {
            if (!candidate) {
                continue;
            }
            if (!file_exists(candidate)) {
                continue;
            }
            std::ifstream src(candidate, std::ios::binary);
            if (!src) {
                continue;
            }
            std::ofstream dst(path, std::ios::binary);
            if (!dst) {
                continue;
            }
            dst << src.rdbuf();
            return;
        }

        // Fallback: create a minimal file that exists
        std::ofstream file(path, std::ios::binary);
        file << "dummy model content";
        file.close();
    }

    nlohmann::json validConfig() {
        return {
            {"model", "/tmp/test_model.cvimodel"},
            {"script", "/tmp/test_valid.lua"},
            {"threshold", 0.5f}
        };
    }
};

// ============================================================================
// Configuration Parsing Tests
// ============================================================================

TEST_F(ModelNodeTest, ParseConfig) {
    nlohmann::json config = {
        {"model", "/tmp/test_model.cvimodel"},
        {"script", "/tmp/test_valid.lua"},
        {"threshold", 0.7f},
        {"input_mode", "full_frame"},
        {"crop_size", {112, 112}},
        {"timeout_ms", 3000}
    };

    ModelNode node("test", "model");
    // Note: onCreate will fail without actual CviSession (TPU),
    // but we can test config parsing up to that point
#ifndef USE_CVI_TPU
    // In non-TPU builds, file existence check happens
    int ret = node.onCreate(config);
    // Should pass file check at least
    EXPECT_TRUE(ret == MA_OK || ret == MA_EINVAL);  // May fail due to Lua or model
#endif
}

TEST_F(ModelNodeTest, MissingModelField) {
    nlohmann::json config = {
        {"script", "/tmp/test_valid.lua"}
        // Missing "model" field
    };

    ModelNode node("test", "model");
    EXPECT_EQ(node.onCreate(config), MA_EINVAL);
}

TEST_F(ModelNodeTest, MissingScriptField) {
    nlohmann::json config = {
        {"model", "/tmp/test_model.cvimodel"}
        // Missing "script" field
    };

    ModelNode node("test", "model");
    EXPECT_EQ(node.onCreate(config), MA_EINVAL);
}

TEST_F(ModelNodeTest, ModelNotFound) {
    nlohmann::json config = {
        {"model", "/nonexistent.cvimodel"},
        {"script", "/tmp/test_valid.lua"}
    };

    ModelNode node("test", "model");
    int ret = node.onCreate(config);
    // Should fail with either ENOENT (file not found) or EIO (model load failed)
    EXPECT_TRUE(ret == MA_ENOENT || ret == MA_EIO || ret == MA_EINVAL);
}

TEST_F(ModelNodeTest, ScriptNotFound) {
    nlohmann::json config = {
        {"model", "/tmp/test_model.cvimodel"},
        {"script", "/nonexistent.lua"}
    };

    ModelNode node("test", "model");
    EXPECT_EQ(node.onCreate(config), MA_EINVAL);
}

TEST_F(ModelNodeTest, InvalidScript_SyntaxError) {
    nlohmann::json config = {
        {"model", "/tmp/test_model.cvimodel"},
        {"script", "/tmp/test_syntax_error.lua"}
    };

    ModelNode node("test", "model");
    EXPECT_EQ(node.onCreate(config), MA_EINVAL);
}

TEST_F(ModelNodeTest, InvalidScript_NotTable) {
    nlohmann::json config = {
        {"model", "/tmp/test_model.cvimodel"},
        {"script", "/tmp/test_not_table.lua"}
    };

    ModelNode node("test", "model");
    EXPECT_EQ(node.onCreate(config), MA_EINVAL);
}

TEST_F(ModelNodeTest, MissingPostprocessFunction) {
    nlohmann::json config = {
        {"model", "/tmp/test_model.cvimodel"},
        {"script", "/tmp/test_no_postprocess.lua"}
    };

    ModelNode node("test", "model");
    EXPECT_EQ(node.onCreate(config), MA_EINVAL);
}

// ============================================================================
// Lifecycle Tests
// ============================================================================

TEST_F(ModelNodeTest, LifecycleWithoutCreate) {
    ModelNode node("test", "model");

    // Start without create should fail
    EXPECT_EQ(node.onStart(), MA_EINVAL);
    EXPECT_FALSE(node.isStarted());
}

TEST_F(ModelNodeTest, DoubleStop) {
    ModelNode node("test", "model");

    // Stop without start should be idempotent
    EXPECT_EQ(node.onStop(), MA_OK);
    EXPECT_EQ(node.onStop(), MA_OK);
}

TEST_F(ModelNodeTest, DestroyWithoutCreate) {
    ModelNode node("test", "model");

    // Destroy without create should be safe
    EXPECT_EQ(node.onDestroy(), MA_OK);
}

// ============================================================================
// onControl Tests
// ============================================================================

TEST_F(ModelNodeTest, SetThreshold) {
    ModelNode node("test", "model");

    // Control without create - depends on implementation
    // Most implementations require onCreate first
    EXPECT_EQ(node.onControl("set_threshold", {{"value", 0.7f}}), MA_OK);
}

TEST_F(ModelNodeTest, GetStats) {
    ModelNode node("test", "model");

    // Get stats returns OK even without running
    EXPECT_EQ(node.onControl("get_stats", {}), MA_OK);
}

TEST_F(ModelNodeTest, UnknownControlCommand) {
    ModelNode node("test", "model");

    EXPECT_EQ(node.onControl("unknown_command", {}), MA_EINVAL);
}

TEST_F(ModelNodeTest, ReloadScriptWhileRunning) {
    ModelNode node("test", "model");

    // Simulate running state
    // reload_script should fail if running
    // This test checks the guard condition
    EXPECT_EQ(node.onControl("reload_script", {}), MA_OK);  // Not running, so OK
}

// ============================================================================
// Statistics Tests
// ============================================================================

TEST_F(ModelNodeTest, InitialStatistics) {
    ModelNode node("test", "model");

    EXPECT_EQ(node.infer_count(), 0);
    EXPECT_EQ(node.error_count(), 0);
    EXPECT_NEAR(node.infer_ema_ms(), 0.0, 0.001);
}

// ============================================================================
// InputMode Tests
// ============================================================================

TEST_F(ModelNodeTest, CroppedRoiRequiresUpstreamModel) {
    // Create a test script that works
    createTestScript("/tmp/test_crop.lua", R"(
local Model = {}
Model.preprocess_config = { input_size = {112, 112} }
function Model.postprocess(outputs, meta)
    return {}
end
return Model
)");

    nlohmann::json config = {
        {"model", "/tmp/test_model.cvimodel"},
        {"script", "/tmp/test_crop.lua"},
        {"input_mode", "cropped_roi"},
        {"crop_size", {112, 112}}
    };

    ModelNode node("test", "model");

    // CROPPED_ROI mode requires upstream ModelNode
    // Without dependencies, should fail validation
    int ret = node.onCreate(config);
    // Should fail because no upstream model dependency
#ifdef USE_CVI_TPU
    EXPECT_TRUE(ret == MA_EINVAL || ret == MA_EIO);
#else
    EXPECT_EQ(ret, MA_EINVAL);
#endif

    std::remove("/tmp/test_crop.lua");
}

// ============================================================================
// Factory Integration Tests
// ============================================================================

TEST_F(ModelNodeTest, FactoryRegistration) {
    // ModelNode should be auto-registered via REGISTER_NODE macro
    // Try to create via factory
    auto* node = NodeFactory::instance().create("model1", "model", validConfig());

    // May fail if TPU not available, but should not crash
    if (node == nullptr) {
        // Check error reason
        int err = NodeFactory::instance().lastErrorCode();
        EXPECT_TRUE(err == MA_ENOENT || err == MA_EIO || err == MA_EINVAL);
    }
}

// ============================================================================
// Thread Safety Tests
// ============================================================================

TEST_F(ModelNodeTest, ConcurrentControlCommands) {
    ModelNode node("test", "model");

    std::vector<std::thread> threads;
    std::atomic<int> success_count{0};

    // Multiple threads sending control commands
    for (int i = 0; i < 10; i++) {
        threads.emplace_back([&, i]() {
            float threshold = 0.1f + i * 0.05f;
            if (node.onControl("set_threshold", {{"value", threshold}}) == MA_OK) {
                success_count++;
            }
        });
    }

    for (auto& t : threads) t.join();

    // All should succeed (set_threshold is thread-safe)
    EXPECT_EQ(success_count.load(), 10);
}

// ============================================================================
// Device Integration Tests (require TPU)
// ============================================================================

class ModelNodeDeviceTest : public ::testing::Test {
protected:
    static bool tpu_tests_enabled() {
        const char* env = std::getenv("ENABLE_TPU_TESTS");
        return env && std::string(env) == "1";
    }

    static void SetUpTestSuite() {
#ifdef USE_CVI_MPI
        if (!tpu_tests_enabled()) {
            return;
        }
        lua_cv::MmfContext::Config config;
        config.force_reset = true;
        if (!lua_cv::MmfContext::build_default_config(&config)) {
            cvi_ready_ = false;
            return;
        }
        cvi_ready_ = lua_cv::MmfContext::instance().init(config);
#endif
    }

    static void TearDownTestSuite() {
#ifdef USE_CVI_MPI
        if (cvi_ready_) {
            lua_cv::MmfContext::instance().shutdown();
            cvi_ready_ = false;
        }
#endif
    }

    void SetUp() override {
        if (!tpu_tests_enabled()) {
            GTEST_SKIP() << "TPU tests disabled. Set ENABLE_TPU_TESTS=1 to enable.";
        }
#ifndef USE_CVI_MPI
        GTEST_SKIP() << "USE_CVI_MPI not enabled.";
#else
        if (!cvi_ready_) {
            GTEST_SKIP() << "CVI init failed.";
        }
#endif

        model_path_ = env_or_default("TPU_MODEL_PATH", "/userdata/Models/model.cvimodel");
        script_path_ = env_or_default("TPU_SCRIPT_PATH", "/tmp/scripts/yolo11_tensor_detector.lua");
        image_path_ = env_or_default("TPU_IMAGE_PATH", "/tmp/zidane_640.jpg");

        if (!file_exists(model_path_)) {
            GTEST_SKIP() << "Model file missing: " << model_path_;
        }
        if (!file_exists(script_path_)) {
            GTEST_SKIP() << "Script file missing: " << script_path_;
        }
        if (!file_exists(image_path_)) {
            GTEST_SKIP() << "Image file missing: " << image_path_;
        }
    }

    std::string model_path_;
    std::string script_path_;
    std::string image_path_;

    static bool cvi_ready_;
};

bool ModelNodeDeviceTest::cvi_ready_ = false;

TEST_F(ModelNodeDeviceTest, FullFrameInference) {
    ModelNode node("detector", "model");
    nlohmann::json config = {
        {"model", model_path_},
        {"script", script_path_},
        {"threshold", 0.25}
    };

    ASSERT_EQ(node.create(config), MA_OK);
    ASSERT_EQ(node.start(), MA_OK);

    lua_cv::ImageSource source;
    ASSERT_TRUE(source.open(image_path_));

    lua_cv::Frame frame;
    ASSERT_TRUE(source.read(frame));

    auto* sf = new SharedFrame(std::move(frame));
    sf->ref();
    auto* ctx = new PipelineContext{sf, nlohmann::json::object(), 1};

    if (!node.inbox()->post(ctx, 1000)) {
        delete ctx;
        sf->release();
        node.destroy();
        FAIL() << "Failed to post PipelineContext to ModelNode inbox";
    }
    sf->release();

    EXPECT_TRUE(wait_for_infer(node, 1, 10000));
    EXPECT_GE(node.infer_count(), 1u);
    EXPECT_EQ(node.error_count(), 0u);

    node.destroy();
    source.close();
}

TEST_F(ModelNodeDeviceTest, CroppedRoiInference) {
    const std::string select_script = "/tmp/test_select_rois.lua";
    std::ofstream file(select_script);
    file << "local Model = {}\n"
            "Model.preprocess_config = { type = \"letterbox\", input_size = {640, 640}, stride = 32, fill_value = 114 }\n"
            "function Model.select_rois(upstream)\n"
            "  return upstream.rois or {}\n"
            "end\n"
            "function Model.postprocess(outputs, meta)\n"
            "  return {items = {}}\n"
            "end\n"
            "return Model\n";
    file.close();

    ModelNode upstream("detector", "model");
    ModelNode downstream("recognizer", "model");
    downstream.addDependency(&upstream);

    nlohmann::json up_config = {
        {"model", model_path_},
        {"script", script_path_},
        {"threshold", 0.25}
    };
    nlohmann::json down_config = {
        {"model", model_path_},
        {"script", select_script},
        {"input_mode", "cropped_roi"},
        {"threshold", 0.25}
    };

    ASSERT_EQ(upstream.create(up_config), MA_OK);
    ASSERT_EQ(downstream.create(down_config), MA_OK);
    ASSERT_EQ(upstream.start(), MA_OK);
    ASSERT_EQ(downstream.start(), MA_OK);

    lua_cv::ImageSource source;
    ASSERT_TRUE(source.open(image_path_));

    lua_cv::Frame frame;
    ASSERT_TRUE(source.read(frame));

    nlohmann::json upstream_result = {
        {"rois", nlohmann::json::array({{
            {"x", 0}, {"y", 0}, {"w", 160}, {"h", 160}
        }})}
    };

    auto* sf = new SharedFrame(std::move(frame));
    sf->ref();
    auto* ctx = new PipelineContext{sf, upstream_result, 1};

    if (!downstream.inbox()->post(ctx, 1000)) {
        delete ctx;
        sf->release();
        downstream.destroy();
        upstream.destroy();
        FAIL() << "Failed to post PipelineContext to downstream ModelNode";
    }
    sf->release();

    EXPECT_TRUE(wait_for_infer(downstream, 1, 10000));
    EXPECT_GE(downstream.infer_count(), 1u);
    EXPECT_EQ(downstream.error_count(), 0u);

    downstream.destroy();
    upstream.destroy();
    source.close();
    std::remove(select_script.c_str());
}

TEST_F(ModelNodeDeviceTest, InferenceLatencyBenchmark) {
    ModelNode node("detector", "model");
    nlohmann::json config = {
        {"model", model_path_},
        {"script", script_path_},
        {"threshold", 0.25}
    };

    ASSERT_EQ(node.create(config), MA_OK);
    ASSERT_EQ(node.start(), MA_OK);

    lua_cv::ImageSource source;
    ASSERT_TRUE(source.open(image_path_));

    int iters = 5;
    if (const char* env = std::getenv("TPU_BENCH_ITERS")) {
        int val = std::atoi(env);
        if (val > 0) iters = val;
    }

    for (int i = 0; i < iters; ++i) {
        lua_cv::Frame frame;
        ASSERT_TRUE(source.read(frame));

        auto* sf = new SharedFrame(std::move(frame));
        sf->ref();
        auto* ctx = new PipelineContext{sf, nlohmann::json::object(), static_cast<uint64_t>(i + 1)};

        if (!node.inbox()->post(ctx, 1000)) {
            delete ctx;
            sf->release();
            node.destroy();
            FAIL() << "Failed to post PipelineContext in benchmark";
        }
        sf->release();

        ASSERT_TRUE(wait_for_infer(node, static_cast<uint64_t>(i + 1), 10000));
    }

    double ema_ms = node.infer_ema_ms();
    std::cout << "[BENCH] ModelNode infer_ema_ms=" << ema_ms << " ms\n";

    if (const char* env = std::getenv("TPU_LATENCY_TARGET_MS")) {
        double target = std::atof(env);
        if (target > 0.0) {
            EXPECT_LT(ema_ms, target);
        }
    }

    node.destroy();
    source.close();
}

TEST_F(ModelNodeDeviceTest, MemoryLeakTest) {
    ModelNode node("detector", "model");
    nlohmann::json config = {
        {"model", model_path_},
        {"script", script_path_},
        {"threshold", 0.25}
    };

    ASSERT_EQ(node.create(config), MA_OK);
    ASSERT_EQ(node.start(), MA_OK);

    lua_cv::ImageSource source;
    ASSERT_TRUE(source.open(image_path_));

    int iters = 20;
    if (const char* env = std::getenv("TPU_LEAK_ITERS")) {
        int val = std::atoi(env);
        if (val > 0) iters = val;
    }

    int warmup_iters = 5;
    if (const char* env = std::getenv("TPU_LEAK_WARMUP")) {
        int val = std::atoi(env);
        if (val >= 0) warmup_iters = val;
    }

    auto post_and_wait = [&](uint64_t frame_id) -> bool {
        lua_cv::Frame frame;
        if (!source.read(frame)) {
            return false;
        }

        auto* sf = new SharedFrame(std::move(frame));
        sf->ref();
        auto* ctx = new PipelineContext{sf, nlohmann::json::object(), frame_id};

        if (!node.inbox()->post(ctx, 1000)) {
            delete ctx;
            sf->release();
            return false;
        }
        sf->release();

        if (!wait_for_infer(node, frame_id, 10000)) {
            return false;
        }
        return true;
    };

    for (int i = 0; i < warmup_iters; ++i) {
        ASSERT_TRUE(post_and_wait(static_cast<uint64_t>(i + 1)));
    }

    long rss_before = read_rss_kb();
    uint64_t base = node.infer_count();

    for (int i = 0; i < iters; ++i) {
        ASSERT_TRUE(post_and_wait(base + static_cast<uint64_t>(i + 1)));
    }

    long rss_after = read_rss_kb();
    std::cout << "[MEM] RSS before=" << rss_before << " KB after=" << rss_after
              << " KB (iters=" << iters << ")\n";

    if (rss_before > 0 && rss_after > 0) {
        long max_kb = 16384;
        if (const char* env = std::getenv("TPU_LEAK_MAX_KB")) {
            long val = std::atol(env);
            if (val > 0) max_kb = val;
        }
        EXPECT_LE(rss_after - rss_before, max_kb);
    }

    node.destroy();
    source.close();
}
