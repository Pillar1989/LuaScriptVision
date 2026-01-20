#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "cv_types.h"

#ifdef USE_CVI_MPI
#include <cvi_vb.h>
#include <linux/cvi_comm_video.h>
#endif

namespace cv {
class Mat;
}

namespace lua_cv {

class HwJpegDecoder;

class IonImageLoader {
public:
    IonImageLoader();
    ~IonImageLoader();

    IonImageLoader(const IonImageLoader&) = delete;
    IonImageLoader& operator=(const IonImageLoader&) = delete;

#ifdef USE_CVI_MPI
    VIDEO_FRAME_INFO_S load(const std::string& filepath);
    VIDEO_FRAME_INFO_S load_from_memory(const uint8_t* data, size_t size);
    VIDEO_FRAME_INFO_S load_from_memory_fast(const uint8_t* data, size_t size);

    bool preallocate(uint32_t width, uint32_t height, uint32_t channels);
    void release();
#endif

private:
#ifdef USE_CVI_MPI
    bool is_jpeg_format(const uint8_t* data, size_t size) const;
    bool get_jpeg_dimensions(const uint8_t* data, size_t size,
                             uint32_t& width, uint32_t& height) const;
    bool decode_with_opencv(const uint8_t* data, size_t size);
    bool decode_into_vb(const cv::Mat& mat);
    VB_POOL select_pool(uint32_t width, uint32_t height, PixelFormat format) const;
    void release_last_frame();

    uint32_t prealloc_width_ = 0;
    uint32_t prealloc_height_ = 0;
    uint32_t prealloc_channels_ = 0;

    VB_BLK vb_block_ = VB_INVALID_HANDLE;
    bool has_vb_block_ = false;
    uint32_t vb_block_size_ = 0;
    VIDEO_FRAME_INFO_S last_frame_{};

    HwJpegDecoder* hw_decoder_ = nullptr;
    bool has_hw_frame_ = false;
    VIDEO_FRAME_INFO_S last_hw_frame_{};
#endif
};

} // namespace lua_cv
