#include "rtsp_server.h"

#include <cstring>
#include <iostream>
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <netinet/in.h>

namespace lua_cv {

#ifdef USE_CVI_MPI

RtspServer::RtspServer(const Config& config)
    : config_(config) {
}

RtspServer::~RtspServer() {
    stop();
}

bool RtspServer::start() {
    if (running_) {
        return true;
    }

    CVI_RTSP_CONFIG rtsp_config;
    std::memset(&rtsp_config, 0, sizeof(rtsp_config));
    rtsp_config.port = config_.port;
    rtsp_config.maxConnNum = config_.max_connections;

    int rc = CVI_RTSP_Create(&ctx_, &rtsp_config);
    if (rc != 0) {
        std::cerr << "[RTSP] CVI_RTSP_Create failed: rc=" << rc
                  << " port=" << config_.port << std::endl;
        return false;
    }

    CVI_RTSP_SESSION_ATTR attr;
    std::memset(&attr, 0, sizeof(attr));
    snprintf(attr.name, sizeof(attr.name), "%s", config_.session_name.c_str());
    attr.maxConnNum = config_.max_connections;

    switch (config_.video_codec) {
        case VideoCodec::H264:
            attr.video.codec = RTSP_VIDEO_H264;
            break;
        case VideoCodec::H265:
            attr.video.codec = RTSP_VIDEO_H265;
            break;
        case VideoCodec::JPEG:
            attr.video.codec = RTSP_VIDEO_JPEG;
            break;
    }
    attr.video.bitrate = config_.video_bitrate_kbps;

    switch (config_.audio_codec) {
        case AudioCodec::None:
            attr.audio.codec = RTSP_AUDIO_NONE;
            break;
        case AudioCodec::PCM_L16:
            attr.audio.codec = RTSP_AUDIO_PCM_L16;
            break;
        case AudioCodec::PCM_L24:
            attr.audio.codec = RTSP_AUDIO_PCM_L24;
            break;
        case AudioCodec::AAC:
            attr.audio.codec = RTSP_AUDIO_AAC;
            break;
    }
    attr.audio.sampleRate = config_.audio_sample_rate;
    attr.audio.bitrate = config_.audio_bitrate_kbps;

    rc = CVI_RTSP_CreateSession(ctx_, &attr, &session_);
    if (rc != 0) {
        std::cerr << "[RTSP] CVI_RTSP_CreateSession failed: rc=" << rc
                  << " session=" << config_.session_name << std::endl;
        CVI_RTSP_Destroy(&ctx_);
        ctx_ = nullptr;
        return false;
    }

    CVI_RTSP_STATE_LISTENER listener;
    std::memset(&listener, 0, sizeof(listener));
    listener.onConnect = on_connect_callback;
    listener.argConn = this;
    listener.onDisconnect = on_disconnect_callback;
    listener.argDisconn = this;

    rc = CVI_RTSP_SetListener(ctx_, &listener);
    if (rc != 0) {
        std::cerr << "[RTSP] CVI_RTSP_SetListener failed: rc=" << rc << std::endl;
    }

    rc = CVI_RTSP_Start(ctx_);
    if (rc != 0) {
        std::cerr << "[RTSP] CVI_RTSP_Start failed: rc=" << rc
                  << " port=" << config_.port << std::endl;
        CVI_RTSP_DestroySession(ctx_, session_);
        session_ = nullptr;
        CVI_RTSP_Destroy(&ctx_);
        ctx_ = nullptr;
        return false;
    }

    running_ = true;
    std::cout << "[RTSP] Server started: " << get_url() << std::endl;
    return true;
}

void RtspServer::stop() {
    if (!running_) {
        return;
    }

    if (ctx_) {
        CVI_RTSP_Stop(ctx_);

        if (session_) {
            CVI_RTSP_DestroySession(ctx_, session_);
            session_ = nullptr;
        }

        CVI_RTSP_Destroy(&ctx_);
        ctx_ = nullptr;
    }

    running_ = false;
    std::cout << "[RTSP] Server stopped" << std::endl;
}

std::string RtspServer::get_url() const {
    std::string ip = "127.0.0.1";

    struct ifaddrs* ifaddr = nullptr;
    if (getifaddrs(&ifaddr) == 0) {
        for (struct ifaddrs* ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
            if (ifa->ifa_addr == nullptr) continue;
            if (ifa->ifa_addr->sa_family != AF_INET) continue;

            std::string name(ifa->ifa_name);
            if (name == "lo") continue;

            char addr_buf[INET_ADDRSTRLEN];
            struct sockaddr_in* addr = reinterpret_cast<struct sockaddr_in*>(ifa->ifa_addr);
            inet_ntop(AF_INET, &addr->sin_addr, addr_buf, sizeof(addr_buf));
            ip = addr_buf;
            break;
        }
        freeifaddrs(ifaddr);
    }

    return "rtsp://" + ip + ":" + std::to_string(config_.port) + "/" + config_.session_name;
}

bool RtspServer::send_video(const uint8_t* data, size_t length, uint64_t pts) {
    if (!running_ || !session_ || !session_->video) {
        return false;
    }

    CVI_RTSP_DATA frame;
    std::memset(&frame, 0, sizeof(frame));
    frame.blockCnt = 1;
    frame.dataPtr[0] = const_cast<uint8_t*>(data);
    frame.dataLen[0] = static_cast<uint32_t>(length);
    frame.timestamp = pts;

    int rc = CVI_RTSP_WriteFrame(ctx_, session_->video, &frame);
    if (rc != 0) {
        return false;
    }

    return true;
}

bool RtspServer::send_audio(const uint8_t* data, size_t length, uint64_t pts) {
    if (!running_ || !session_ || !session_->audio) {
        return false;
    }

    if (config_.audio_codec == AudioCodec::None) {
        return false;
    }

    CVI_RTSP_DATA frame;
    std::memset(&frame, 0, sizeof(frame));
    frame.blockCnt = 1;
    frame.dataPtr[0] = const_cast<uint8_t*>(data);
    frame.dataLen[0] = static_cast<uint32_t>(length);
    frame.timestamp = pts;

    int rc = CVI_RTSP_WriteFrame(ctx_, session_->audio, &frame);
    if (rc != 0) {
        return false;
    }

    return true;
}

void RtspServer::on_connect_callback(const char* ip, void* arg) {
    RtspServer* server = static_cast<RtspServer*>(arg);
    if (server && server->on_connect_) {
        server->on_connect_(ip ? ip : "");
    }
    std::cout << "[RTSP] Client connected: " << (ip ? ip : "unknown") << std::endl;
}

void RtspServer::on_disconnect_callback(const char* ip, void* arg) {
    RtspServer* server = static_cast<RtspServer*>(arg);
    if (server && server->on_disconnect_) {
        server->on_disconnect_(ip ? ip : "");
    }
    std::cout << "[RTSP] Client disconnected: " << (ip ? ip : "unknown") << std::endl;
}

#endif

} // namespace lua_cv
