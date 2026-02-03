#pragma once

#include <cstdint>
#include <vector>
#include <memory>

#ifdef USE_CVI_MPI
#include <cvi_venc.h>
#include <cvi_sys.h>
#include <linux/cvi_comm_video.h>
#endif

namespace lua_cv {

#ifdef USE_CVI_MPI

class VencEncoder {
public:
    enum class CodecType {
        H264,
        H265,
        JPEG,
        MJPEG
    };

    struct Config {
        CodecType codec = CodecType::H264;
        uint32_t width = 1920;
        uint32_t height = 1080;
        uint32_t fps = 30;
        uint32_t bitrate_kbps = 4000;  // Increased from 2000 for better quality
        uint32_t gop = 30;
        int32_t ip_qp_delta = -1;  // Added: I-frame QP delta (negative = higher quality I-frames)
        uint32_t quality = 80;
        VENC_CHN channel = 0;
    };

    struct EncodedStream {
        std::vector<uint8_t> data;
        bool is_keyframe = false;
        uint64_t pts = 0;
    };

    explicit VencEncoder(const Config& config);
    ~VencEncoder();

    VencEncoder(const VencEncoder&) = delete;
    VencEncoder& operator=(const VencEncoder&) = delete;
    VencEncoder(VencEncoder&&) = delete;
    VencEncoder& operator=(VencEncoder&&) = delete;

    bool init();
    void shutdown();

    bool bind_to_vpss(VPSS_GRP grp, VPSS_CHN chn);
    void unbind_from_vpss();

    bool send_frame(const VIDEO_FRAME_INFO_S& frame, int timeout_ms = 100);

    bool get_stream(EncodedStream* stream, int timeout_ms = 100);
    void release_stream();

    bool request_idr();

    const Config& config() const { return config_; }
    bool is_initialized() const { return initialized_; }
    bool is_bound() const { return bound_; }
    VENC_CHN channel() const { return config_.channel; }

private:
    bool start_recv();
    bool init_h264();
    bool init_h265();
    bool init_jpeg();
    bool init_mjpeg();

    Config config_;
    bool initialized_ = false;
    bool bound_ = false;
    bool receiving_ = false;
    bool stream_acquired_ = false;
    VENC_STREAM_S current_stream_{};
    std::vector<VENC_PACK_S> current_packs_;

    VPSS_GRP bound_vpss_grp_ = -1;
    VPSS_CHN bound_vpss_chn_ = -1;
};

#endif

} // namespace lua_cv
