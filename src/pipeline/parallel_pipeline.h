#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "../modules/cv/frame.h"

#ifdef USE_CVI_MPI
#include "../stream/venc_encoder.h"
#include "../stream/rtsp_server.h"
#endif

#ifdef USE_CVI_CAMERA
#include "../modules/cv/cvi_camera.h"
#endif

namespace lua_cv {

#if defined(USE_CVI_MPI) && defined(USE_CVI_CAMERA)

class ParallelPipeline {
public:
    struct Config {
        int stream_width = 1920;
        int stream_height = 1080;
        int stream_fps = 30;
        int stream_bitrate_kbps = 2000;
        int stream_gop = 30;
        VencEncoder::CodecType stream_codec = VencEncoder::CodecType::H264;

        int rtsp_port = 554;
        std::string rtsp_session = "live";

        int infer_width = 640;
        int infer_height = 640;

        int result_queue_size = 8;
        int camera_ready_timeout_ms = 3000;
    };

    struct InferenceResult {
        uint64_t frame_id = 0;
        uint64_t timestamp_ms = 0;
        std::vector<uint8_t> data;
    };

    using InferenceCallback = std::function<bool(const Frame& frame, InferenceResult* result)>;

    explicit ParallelPipeline(const Config& config);
    ~ParallelPipeline();

    ParallelPipeline(const ParallelPipeline&) = delete;
    ParallelPipeline& operator=(const ParallelPipeline&) = delete;
    ParallelPipeline(ParallelPipeline&&) = delete;
    ParallelPipeline& operator=(ParallelPipeline&&) = delete;

    bool start(InferenceCallback callback);
    void stop();
    bool is_running() const { return running_.load(); }

    bool pop_result(InferenceResult* result, int timeout_ms = 0);

    std::string get_rtsp_url() const;

    struct Stats {
        uint64_t infer_frames = 0;
        uint64_t stream_frames = 0;
        uint64_t dropped_results = 0;
        double infer_fps = 0.0;
        double stream_fps = 0.0;
    };
    Stats get_stats() const;

    const Config& config() const { return config_; }

private:
    void stream_thread_func();
    void infer_thread_func();

    void push_result(InferenceResult result);

    Config config_;
    InferenceCallback infer_callback_;

    std::unique_ptr<CviCamera> camera_;
    std::unique_ptr<VencEncoder> encoder_;
    std::unique_ptr<RtspServer> rtsp_;

    std::thread stream_thread_;
    std::thread infer_thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> stop_requested_{false};

    std::queue<InferenceResult> result_queue_;
    mutable std::mutex result_mutex_;
    std::condition_variable result_cv_;

    mutable std::mutex stats_mutex_;
    Stats stats_;
    uint64_t infer_start_time_ = 0;
    uint64_t stream_start_time_ = 0;
};

#endif

} // namespace lua_cv
