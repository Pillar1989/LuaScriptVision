#pragma once

#include <cstdint>
#include <memory>
#include <opencv2/opencv.hpp>

#include "cv_types.h"
#include "device_buffer.h"
#include "image.h"

#ifdef USE_CVI_MPI
#include <cvi_vb.h>
#include <cvi_vpss.h>
#include <linux/cvi_comm_video.h>
#endif

namespace lua_cv {

class Frame {
public:
    enum class StorageType {
        NONE = 0,
        OPENCV,
        CVI,
    };

    Frame();
    explicit Frame(const cv::Mat& mat);
    Frame(const cv::Mat& mat, PixelFormat format);
    explicit Frame(cv::Mat&& mat);
    Frame(cv::Mat&& mat, PixelFormat format);

#ifdef USE_CVI_MPI
    Frame(const VIDEO_FRAME_INFO_S& frame, bool owns_memory);
    Frame(const VIDEO_FRAME_INFO_S& frame, int vpss_grp, int vpss_chn);
#endif

    Frame(const Frame& other);
    Frame(Frame&& other) noexcept;
    Frame& operator=(const Frame& other);
    Frame& operator=(Frame&& other) noexcept;

    ~Frame();

    bool empty() const;
    int width() const;
    int height() const;
    int channels() const;
    StorageType storage_type() const { return storage_type_; }

    bool has_physical_addr() const;
    uint64_t physical_addr() const;
    PixelFormat pixel_format() const { return format_; }

    const cv::Mat& to_mat() const;
    cv::Mat& to_mat();

    Frame clone() const;
    void release();

#ifdef USE_CVI_MPI
    const VIDEO_FRAME_INFO_S* video_frame() const;
    VIDEO_FRAME_INFO_S* video_frame();
    void set_vpss_owner(int vpss_grp, int vpss_chn);
    void set_vb_owner(VB_BLK vb_block);
#endif

private:
    void reset();

    StorageType storage_type_ = StorageType::NONE;
    PixelFormat format_ = PixelFormat::UNKNOWN;

    cv::Mat mat_;
    mutable cv::Mat mat_cache_;
    mutable bool mat_cache_valid_ = false;

    CvImage image_;

#ifdef USE_CVI_MPI
    VIDEO_FRAME_INFO_S cvi_frame_{};
    bool owns_vpss_frame_ = false;
    bool owns_vb_block_ = false;
    int vpss_grp_ = -1;
    int vpss_chn_ = -1;
    VB_BLK vb_block_ = VB_INVALID_HANDLE;
#endif
};

} // namespace lua_cv
