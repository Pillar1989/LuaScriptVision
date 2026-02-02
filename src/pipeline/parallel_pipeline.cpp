#include "parallel_pipeline.h"

#include <chrono>
#include <iostream>

#ifdef USE_CVI_CAMERA
#include "../modules/cv/mmf_context.h"
#endif

namespace lua_cv {

#if defined(USE_CVI_MPI) && defined(USE_CVI_CAMERA)

ParallelPipeline::ParallelPipeline(const Config& config)
    : config_(config) {
}

ParallelPipeline::~ParallelPipeline() {
    // NOTE: stop() is NOT called here by design.
    // The caller is responsible for calling stop() before destruction.
    // This avoids double-cleanup and accessing resources after MMF shutdown.
}

bool ParallelPipeline::start(InferenceCallback callback) {
    if (running_.load()) {
        std::cerr << "[ParallelPipeline] Already running" << std::endl;
        return false;
    }

    if (!callback) {
        std::cerr << "[ParallelPipeline] No inference callback provided" << std::endl;
        return false;
    }

    infer_callback_ = std::move(callback);

    // Step 1: Get VPSS group and channel info
    int vpss_grp = MmfContext::vpss_group_for_camera();
    int stream_chn = MmfContext::vpss_channel_for_camera_stream();

    // Step 2: Create and init VENC encoder
    VencEncoder::Config venc_config;
    venc_config.codec = config_.stream_codec;
    venc_config.width = config_.stream_width;
    venc_config.height = config_.stream_height;
    venc_config.fps = config_.stream_fps;
    venc_config.bitrate_kbps = config_.stream_bitrate_kbps;
    venc_config.gop = config_.stream_gop;
    venc_config.quality = config_.stream_quality;
    venc_config.ip_qp_delta = config_.stream_ip_qp_delta;
    venc_config.channel = 0;

    encoder_ = std::make_unique<VencEncoder>(venc_config);
    if (!encoder_->init()) {
        std::cerr << "[ParallelPipeline] Failed to init VENC encoder" << std::endl;
        return false;
    }

    // Step 3: Bind VENC to VPSS BEFORE camera opens
    if (!encoder_->bind_to_vpss(static_cast<VPSS_GRP>(vpss_grp),
                                 static_cast<VPSS_CHN>(stream_chn))) {
        std::cerr << "[ParallelPipeline] Failed to bind VENC to VPSS" << std::endl;
        encoder_->shutdown();
        encoder_.reset();
        return false;
    }

    // Step 4: Now open camera (this will bind VI → VPSS)
    CviCamera::Config cam_config;
    cam_config.enable_infer = true;

    camera_ = std::make_unique<CviCamera>(cam_config);
    if (!camera_->open()) {
        std::cerr << "[ParallelPipeline] Failed to open camera" << std::endl;
        encoder_->shutdown();
        encoder_.reset();
        return false;
    }

    RtspServer::Config rtsp_config;
    rtsp_config.port = config_.rtsp_port;
    rtsp_config.session_name = config_.rtsp_session;
    rtsp_config.video_width = config_.stream_width;
    rtsp_config.video_height = config_.stream_height;
    rtsp_config.video_bitrate_kbps = config_.stream_bitrate_kbps;

    switch (config_.stream_codec) {
        case VencEncoder::CodecType::H264:
            rtsp_config.video_codec = RtspServer::VideoCodec::H264;
            break;
        case VencEncoder::CodecType::H265:
            rtsp_config.video_codec = RtspServer::VideoCodec::H265;
            break;
        default:
            rtsp_config.video_codec = RtspServer::VideoCodec::H264;
            break;
    }

    // Audio config (PCM only)
    rtsp_config.audio_codec = RtspServer::AudioCodec::PCM_L16;
    rtsp_config.audio_sample_rate = config_.audio_sample_rate;

    rtsp_ = std::make_unique<RtspServer>(rtsp_config);
    if (!rtsp_->start()) {
        std::cerr << "[ParallelPipeline] Failed to start RTSP server" << std::endl;
        camera_->release();
        camera_.reset();
        encoder_->shutdown();
        encoder_.reset();
        return false;
    }

    // Step 5: Initialize audio capture if enabled
    if (config_.enable_audio) {
        AudioCapture::Config audio_config;
        audio_config.device = config_.audio_device;
        audio_config.sample_rate = config_.audio_sample_rate;
        audio_config.channels = config_.audio_channels;
        audio_config.bit_width = config_.audio_bit_width;
        audio_config.volume = config_.audio_volume;

        audio_capture_ = std::make_unique<AudioCapture>(audio_config);
        if (!audio_capture_->start()) {
            std::cerr << "[ParallelPipeline] Failed to start audio capture" << std::endl;
            stop();  // Cleanup RTSP, camera, encoder
            return false;
        }
    }

    stop_requested_.store(false);
    running_.store(true);

    auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_ = Stats{};
        infer_start_time_ = now_ms;
        stream_start_time_ = now_ms;
    }

    if (!camera_->wait_for_ready(config_.camera_ready_timeout_ms)) {
        std::cerr << "[ParallelPipeline] Camera not ready within "
                  << config_.camera_ready_timeout_ms << " ms" << std::endl;
        stop();
        return false;
    }

    stream_thread_ = std::thread(&ParallelPipeline::stream_thread_func, this);
    infer_thread_ = std::thread(&ParallelPipeline::infer_thread_func, this);

    // Start audio thread if enabled
    if (config_.enable_audio && audio_capture_) {
        audio_thread_ = std::thread(&ParallelPipeline::audio_thread_func, this);
    }

    return true;
}

void ParallelPipeline::stop() {
    if (!running_.load()) {
        return;
    }

    // Step 1: Signal threads to stop
    stop_requested_.store(true);
    running_.store(false);
    result_cv_.notify_all();

    // Step 2: Stop RTSP and encoder BEFORE joining threads
    // This breaks the stream thread out of blocking get_stream() calls
    if (rtsp_) {
        rtsp_->stop();
        rtsp_.reset();
    }

    if (encoder_) {
        encoder_->shutdown();
        encoder_.reset();
    }

    // Step 3: Stop audio capture BEFORE joining audio thread
    if (audio_capture_) {
        audio_capture_->stop();
        audio_capture_.reset();
    }

    // Step 4: Now threads can exit cleanly
    if (stream_thread_.joinable()) {
        stream_thread_.join();
    }
    if (infer_thread_.joinable()) {
        infer_thread_.join();
    }
    if (audio_thread_.joinable()) {
        audio_thread_.join();
    }

    if (camera_) {
        camera_->release();
        camera_.reset();
    }
}

void ParallelPipeline::stream_thread_func() {
    uint64_t frame_count = 0;
    while (!stop_requested_.load()) {
        VencEncoder::EncodedStream stream;
        if (encoder_->get_stream(&stream, 100)) {
            // Check stop again before using encoder/rtsp (they might be reset during shutdown)
            if (stop_requested_.load()) {
                break;
            }

            uint64_t pts = frame_count * (1000000 / config_.stream_fps);
            if (rtsp_ && rtsp_->send_video(stream.data.data(), stream.data.size(), pts)) {
                frame_count++;
                {
                    std::lock_guard<std::mutex> lock(stats_mutex_);
                    stats_.stream_frames = frame_count;

                    auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now().time_since_epoch()).count();
                    double elapsed_sec = (now_ms - stream_start_time_) / 1000.0;
                    if (elapsed_sec > 0) {
                        stats_.stream_fps = frame_count / elapsed_sec;
                    }
                }
            }
            if (encoder_) {
                encoder_->release_stream();
            }
        }
    }
}

void ParallelPipeline::infer_thread_func() {
    uint64_t frame_count = 0;
    while (!stop_requested_.load()) {
        Frame infer_frame;
        if (!camera_->read(infer_frame, 100)) {
            continue;
        }

        frame_count++;
        auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();

        InferenceResult result;
        result.frame_id = frame_count;
        result.timestamp_ms = now_ms;

        if (infer_callback_(infer_frame, &result)) {
            push_result(std::move(result));
        }

        infer_frame.release();

        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.infer_frames = frame_count;

            double elapsed_sec = (now_ms - infer_start_time_) / 1000.0;
            if (elapsed_sec > 0) {
                stats_.infer_fps = frame_count / elapsed_sec;
            }
        }
    }

    std::cout << "[ParallelPipeline] Inference thread exiting, frames=" << frame_count << std::endl;
}

void ParallelPipeline::push_result(InferenceResult result) {
    std::lock_guard<std::mutex> lock(result_mutex_);

    if (result_queue_.size() >= static_cast<size_t>(config_.result_queue_size)) {
        result_queue_.pop();
        std::lock_guard<std::mutex> stats_lock(stats_mutex_);
        stats_.dropped_results++;
    }

    result_queue_.push(std::move(result));
    result_cv_.notify_one();
}

bool ParallelPipeline::pop_result(InferenceResult* result, int timeout_ms) {
    std::unique_lock<std::mutex> lock(result_mutex_);

    if (timeout_ms > 0) {
        if (!result_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                                  [this]() { return !result_queue_.empty() || !running_.load(); })) {
            return false;
        }
    }

    if (result_queue_.empty()) {
        return false;
    }

    *result = std::move(result_queue_.front());
    result_queue_.pop();
    return true;
}

std::string ParallelPipeline::get_rtsp_url() const {
    if (rtsp_) {
        return rtsp_->get_url();
    }
    return "";
}

ParallelPipeline::Stats ParallelPipeline::get_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

void ParallelPipeline::audio_thread_func() {
    uint64_t frame_count = 0;
    uint64_t start_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();

    std::cout << "[ParallelPipeline] Audio thread started" << std::endl;

    while (!stop_requested_.load()) {
        if (!audio_capture_) {
            break;
        }

        AudioCapture::AudioFrame audio_frame;
        if (audio_capture_->get_frame(&audio_frame, 50)) {
            // Check stop again before using rtsp_ (might be reset during shutdown)
            if (stop_requested_.load()) {
                break;
            }

            // Send audio to RTSP
            if (rtsp_) {
                rtsp_->send_audio(audio_frame.data.data(), audio_frame.data.size(), audio_frame.pts);
            }

            frame_count++;

            // Update stats
            {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                stats_.audio_frames = frame_count;
                stats_.audio_pts = audio_frame.pts;

                auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count();
                double elapsed_sec = (now_ms - start_time_ms) / 1000.0;
                if (elapsed_sec > 0) {
                    stats_.audio_fps = frame_count / elapsed_sec;
                }
            }
        }
    }

    std::cout << "[ParallelPipeline] Audio thread exiting, frames=" << frame_count << std::endl;
}

#endif

} // namespace lua_cv
