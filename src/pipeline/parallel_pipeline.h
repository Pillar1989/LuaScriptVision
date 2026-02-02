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
#include "../stream/audio_capture.h"
#endif

#ifdef USE_CVI_CAMERA
#include "../modules/cv/cvi_camera.h"
#endif

namespace lua_cv {

#if defined(USE_CVI_MPI) && defined(USE_CVI_CAMERA)

class ParallelPipeline {
public:
    struct Config {
        // Video/Stream config
        int stream_width = 1920;
        int stream_height = 1080;
        int stream_fps = 30;
        int stream_bitrate_kbps = 4000;  // Increased from 2000 for better quality
        int stream_gop = 30;
        int stream_quality = 90;  // Quality parameter (1-100, higher is better)
        int stream_ip_qp_delta = -1;  // I-frame QP delta (negative = higher quality I-frames)
        VencEncoder::CodecType stream_codec = VencEncoder::CodecType::H264;

        int rtsp_port = 554;
        std::string rtsp_session = "live";

        // Inference config
        int infer_width = 640;
        int infer_height = 640;

        // Audio config
        bool enable_audio = false;
        std::string audio_device = "hw:0";
        int audio_sample_rate = 16000;
        int audio_channels = 1;
        int audio_bit_width = 16;
        int audio_volume = 80;  // 0-100

        // Other config
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

        // Audio stats
        uint64_t audio_frames = 0;
        double audio_fps = 0.0;
        uint64_t audio_pts = 0;      // Latest audio PTS (us)
        uint64_t video_pts = 0;      // Latest video PTS (us)
    };
    Stats get_stats() const;

    const Config& config() const { return config_; }

private:
    void stream_thread_func();
    void infer_thread_func();
    void audio_thread_func();  // Audio capture and RTSP send

    void push_result(InferenceResult result);

    Config config_;
    InferenceCallback infer_callback_;

    std::unique_ptr<CviCamera> camera_;
    std::unique_ptr<VencEncoder> encoder_;
    std::unique_ptr<RtspServer> rtsp_;
    std::unique_ptr<AudioCapture> audio_capture_;  // Audio capture

    std::thread stream_thread_;
    std::thread infer_thread_;
    std::thread audio_thread_;  // Audio thread
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
