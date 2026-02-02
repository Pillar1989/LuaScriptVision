#include <gtest/gtest.h>
#include <iostream>
#include <chrono>
#include <thread>
#include <cstring>
#include <atomic>
#include <csignal>
#include <cstdlib>
#include <unistd.h>

#include "pipeline/parallel_pipeline.h"
#include "cv/mmf_context.h"

using namespace lua_cv;

namespace {
constexpr unsigned int kTestTimeoutSec = 30;
bool g_mmf_ready = false;
const char* g_skip_reason = "MMF context not initialized";

void handle_alarm(int) {
    std::cerr << "[TEST] Timeout after " << kTestTimeoutSec << " seconds" << std::endl;
    std::_Exit(124);
}

class TestTimeoutGuard {
public:
    TestTimeoutGuard() {
        std::signal(SIGALRM, handle_alarm);
        alarm(kTestTimeoutSec);
    }

    ~TestTimeoutGuard() {
        alarm(0);
    }
};

// Shared state for inference callback
std::atomic<int> g_callback_count{0};
std::atomic<uint64_t> g_total_infer_time_us{0};

} // namespace

class ParallelPipelineTest : public ::testing::Test {
protected:
    static ParallelPipeline* pipeline_;

    static void SetUpTestSuite() {
#ifndef USE_CVI_CAMERA
        g_mmf_ready = false;
        g_skip_reason = "CVI Camera not enabled";
        return;
#endif
#ifndef USE_CVI_MPI
        g_mmf_ready = false;
        g_skip_reason = "CVI MPI not available";
        return;
#endif
        MmfContext::Config config;
        if (!MmfContext::build_default_config(&config)) {
            g_mmf_ready = false;
            g_skip_reason = "Failed to build MMF config";
            return;
        }
        config.force_reset = true;
        if (!MmfContext::instance().init(config)) {
            g_mmf_ready = false;
            g_skip_reason = "Failed to init MMF context";
            return;
        }

        // Create pipeline instance (once for all tests)
        ParallelPipeline::Config pipeline_config;
        pipeline_config.rtsp_port = 8560;
        pipeline_config.rtsp_session = "test_suite";
        pipeline_config.result_queue_size = 10;
        pipeline_config.stream_fps = 30;
        pipeline_config.enable_audio = true;  // Enable audio capture
        pipeline_config.audio_volume = 80;     // Set volume to 80%

        pipeline_ = new ParallelPipeline(pipeline_config);

        // Shared inference callback for all tests
        auto callback = [](const Frame& frame, ParallelPipeline::InferenceResult* result) {
            if (frame.width() == 0 || frame.height() == 0) {
                return false;
            }

            auto start = std::chrono::high_resolution_clock::now();

            // Simulate inference work
            std::this_thread::sleep_for(std::chrono::milliseconds(10));

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

            g_total_infer_time_us += duration;
            g_callback_count++;

            // Fill result data
            result->data.resize(4);
            result->data[0] = static_cast<uint8_t>(g_callback_count.load() & 0xFF);
            result->data[1] = static_cast<uint8_t>((frame.width() >> 8) & 0xFF);
            result->data[2] = static_cast<uint8_t>((frame.height() >> 8) & 0xFF);
            result->data[3] = 0xFF;

            return true;
        };

        // Start pipeline (once for all tests)
        if (!pipeline_->start(callback)) {
            delete pipeline_;
            pipeline_ = nullptr;
            g_mmf_ready = false;
            g_skip_reason = "Failed to start pipeline (camera not ready)";
            return;
        }

        // Wait for pipeline to stabilize
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));

        g_mmf_ready = true;
        std::cout << "[TEST] Pipeline started successfully" << std::endl;
    }

    static void TearDownTestSuite() {
#ifdef USE_CVI_MPI
        if (pipeline_) {
            std::cout << "[TEST] Stopping pipeline..." << std::endl;
            pipeline_->stop();
            delete pipeline_;
            pipeline_ = nullptr;
        }
        if (g_mmf_ready) {
            MmfContext::instance().shutdown();
        }
#endif
    }

    void SetUp() override {
        if (!g_mmf_ready) {
            GTEST_SKIP() << g_skip_reason;
        }
    }
};

ParallelPipeline* ParallelPipelineTest::pipeline_ = nullptr;

#if defined(USE_CVI_MPI) && defined(USE_CVI_CAMERA)

TEST_F(ParallelPipelineTest, PipelineIsRunning) {
    TestTimeoutGuard timeout;

    ASSERT_NE(pipeline_, nullptr);
    EXPECT_TRUE(pipeline_->is_running());

    std::cout << "[TEST] Pipeline is running: PASS" << std::endl;
}

TEST_F(ParallelPipelineTest, InferenceCallback) {
    TestTimeoutGuard timeout;

    ASSERT_NE(pipeline_, nullptr);

    int initial_count = g_callback_count.load();
    std::cout << "[TEST] Initial callback count: " << initial_count << std::endl;

    // Wait for callbacks to accumulate
    std::this_thread::sleep_for(std::chrono::seconds(2));

    int final_count = g_callback_count.load();
    std::cout << "[TEST] Final callback count: " << final_count << std::endl;

    int callbacks_received = final_count - initial_count;
    std::cout << "[TEST] Callbacks during test: " << callbacks_received << std::endl;

    EXPECT_GT(callbacks_received, 0) << "No inference callbacks received";
}

TEST_F(ParallelPipelineTest, ResultQueue) {
    TestTimeoutGuard timeout;

    ASSERT_NE(pipeline_, nullptr);

    // Wait for results to queue up
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    int results_received = 0;
    ParallelPipeline::InferenceResult result;

    // Drain queue with max attempts to avoid infinite loop
    int max_attempts = 20;
    int empty_attempts = 0;
    while (empty_attempts < 3 && max_attempts-- > 0) {
        if (pipeline_->pop_result(&result, 100)) {
            results_received++;
            empty_attempts = 0;  // Reset counter on success
            EXPECT_GT(result.frame_id, 0u);
            EXPECT_GT(result.timestamp_ms, 0u);
            EXPECT_EQ(result.data.size(), 4u);
        } else {
            empty_attempts++;  // Queue is empty, increment counter
        }
    }

    std::cout << "[TEST] ResultQueue received: " << results_received << " results" << std::endl;
    EXPECT_GT(results_received, 0) << "No results in queue";
}

TEST_F(ParallelPipelineTest, StreamStats) {
    TestTimeoutGuard timeout;

    ASSERT_NE(pipeline_, nullptr);

    // Wait for RTSP client to connect (if any)
    std::cout << "[TEST] Waiting for RTSP client connection..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(3));

    auto stats = pipeline_->get_stats();

    std::cout << "[TEST] StreamStats:" << std::endl;
    std::cout << "  infer_frames: " << stats.infer_frames << std::endl;
    std::cout << "  stream_frames: " << stats.stream_frames << std::endl;
    std::cout << "  infer_fps: " << stats.infer_fps << std::endl;
    std::cout << "  stream_fps: " << stats.stream_fps << std::endl;
    std::cout << "  dropped_results: " << stats.dropped_results << std::endl;

    EXPECT_GT(stats.infer_frames, 0u);

    if (stats.stream_frames == 0) {
        std::cout << "[WARN] stream_frames=0 (no RTSP client connected?)" << std::endl;
    }
    // Stream frames depend on RTSP client connection
    // EXPECT_GT(stats.stream_frames, 0u);  // Optional: requires RTSP client
}

TEST_F(ParallelPipelineTest, ConcurrentOperation) {
    TestTimeoutGuard timeout;

    ASSERT_NE(pipeline_, nullptr);

    // Let pipeline run
    std::this_thread::sleep_for(std::chrono::seconds(3));

    auto stats = pipeline_->get_stats();

    std::cout << "[TEST] ConcurrentOperation:" << std::endl;
    std::cout << "  infer_frames: " << stats.infer_frames << std::endl;
    std::cout << "  stream_frames: " << stats.stream_frames << std::endl;
    std::cout << "  infer_fps: " << stats.infer_fps << std::endl;
    std::cout << "  stream_fps: " << stats.stream_fps << std::endl;

    EXPECT_GT(stats.infer_frames, 0u);

    if (stats.stream_frames > 0) {
        std::cout << "[TEST] Verified: inference and streaming run in parallel" << std::endl;
        EXPECT_GT(stats.stream_fps, 0);
    } else {
        std::cout << "[WARN] stream_frames=0 (no RTSP client connected)" << std::endl;
    }

    uint64_t total_time = g_total_infer_time_us.load();
    int total_count = g_callback_count.load();
    if (total_count > 0) {
        uint64_t avg_time_us = total_time / total_count;
        std::cout << "[TEST] Average inference time: " << avg_time_us << " us" << std::endl;
    }
}

#endif

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
