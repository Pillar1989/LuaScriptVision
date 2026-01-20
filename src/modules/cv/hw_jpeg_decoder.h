#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#ifdef USE_CVI_MPI
#include <cvi_vdec.h>
#include <linux/cvi_comm_video.h>
#endif

namespace lua_cv {

class HwJpegDecoder {
public:
    HwJpegDecoder();
    ~HwJpegDecoder();

    HwJpegDecoder(const HwJpegDecoder&) = delete;
    HwJpegDecoder& operator=(const HwJpegDecoder&) = delete;

#ifdef USE_CVI_MPI
    bool init(uint32_t max_width, uint32_t max_height);
    bool ensure_initialized(uint32_t width, uint32_t height);
    void cleanup();
    bool is_initialized() const { return initialized_; }

    VIDEO_FRAME_INFO_S decode(const uint8_t* data, size_t size);
    VIDEO_FRAME_INFO_S decode_sync(const uint8_t* data, size_t size);
    void release_frame(const VIDEO_FRAME_INFO_S& frame);
#endif

private:
#ifdef USE_CVI_MPI
    bool find_jpeg_range(const uint8_t* data, size_t size, size_t& start, size_t& end) const;
    bool configure_channel(uint32_t max_width, uint32_t max_height);
    VB_POOL select_pool(uint32_t width, uint32_t height) const;

    VDEC_CHN chn_ = 0;
    bool initialized_ = false;
    uint32_t max_width_ = 0;
    uint32_t max_height_ = 0;
#endif
};

} // namespace lua_cv
