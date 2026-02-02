#pragma once

#include <cstdint>
#include <functional>
#include <string>

#ifdef USE_CVI_MPI
#include <cvi_rtsp/rtsp.h>
#endif

namespace lua_cv {

#ifdef USE_CVI_MPI

class RtspServer {
public:
    enum class VideoCodec {
        H264,
        H265,
        JPEG
    };

    enum class AudioCodec {
        None,
        PCM_L16,
        PCM_L24,
        AAC
    };

    struct Config {
        int port = 554;
        std::string session_name = "live";
        VideoCodec video_codec = VideoCodec::H264;
        int video_width = 1920;
        int video_height = 1080;
        int video_bitrate_kbps = 2000;
        AudioCodec audio_codec = AudioCodec::None;
        int audio_sample_rate = 16000;
        int audio_bitrate_kbps = 64;
        int max_connections = 4;
    };

    explicit RtspServer(const Config& config);
    ~RtspServer();

    RtspServer(const RtspServer&) = delete;
    RtspServer& operator=(const RtspServer&) = delete;
    RtspServer(RtspServer&&) = delete;
    RtspServer& operator=(RtspServer&&) = delete;

    bool start();
    void stop();
    bool is_running() const { return running_; }

    std::string get_url() const;

    bool send_video(const uint8_t* data, size_t length, uint64_t pts = 0);
    bool send_audio(const uint8_t* data, size_t length, uint64_t pts = 0);

    using ClientCallback = std::function<void(const std::string& client_ip)>;
    void set_on_client_connect(ClientCallback cb) { on_connect_ = std::move(cb); }
    void set_on_client_disconnect(ClientCallback cb) { on_disconnect_ = std::move(cb); }

    const Config& config() const { return config_; }

private:
    static void on_connect_callback(const char* ip, void* arg);
    static void on_disconnect_callback(const char* ip, void* arg);

    Config config_;
    CVI_RTSP_CTX* ctx_ = nullptr;
    CVI_RTSP_SESSION* session_ = nullptr;
    bool running_ = false;

    ClientCallback on_connect_;
    ClientCallback on_disconnect_;
};

#endif

} // namespace lua_cv
