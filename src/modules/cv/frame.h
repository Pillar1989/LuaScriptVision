#pragma once

#include <cstdint>
#include <memory>
#include <opencv2/opencv.hpp>

#include "cv_types.h"
#include "vb_memory.h"

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

    // Resource ownership - mutually exclusive (only one can be set)
    enum class FrameOwner {
        EXTERNAL,   // Does not own resources, do not release
        VPSS,       // Release via CVI_VPSS_ReleaseChnFrame
        VB,         // Release via CVI_VB_ReleaseBlock
        VDEC        // Release via CVI_VDEC_ReleaseFrame
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

    // Zero-copy view of CVI frame data (only valid while Frame is alive)
    // Returns mmap'd view without copying. Caller must ensure Frame outlives usage.
    // For NV12/NV21 formats, returns raw YUV data as single-channel Mat.
    const cv::Mat& to_mat_view() const;

    // Copy semantics - returns clone of frame data
    // For NV12/NV21, converts to BGR format.
    cv::Mat to_mat_copy() const;

    // Legacy API - delegates to to_mat_copy() for compatibility
    const cv::Mat& to_mat() const;
    cv::Mat& to_mat();

    Frame clone() const;
    void release();

#ifdef USE_CVI_MPI
    const VIDEO_FRAME_INFO_S* video_frame() const;
    VIDEO_FRAME_INFO_S* video_frame();
    void set_vpss_owner(int vpss_grp, int vpss_chn);
    void set_vb_owner(VB_BLK vb_block);
    void set_vdec_owner(int vdec_chn);

    // Bridge to tensor module - returns VbMemory wrapping frame's VB_BLK
    // Returns nullptr if frame is not CVI storage type or has no physical address.
    // The returned VbMemory does NOT own the block (caller must keep Frame alive).
    std::shared_ptr<tensor::VbMemory> as_vb_memory() const;
#endif

private:
    void reset();
#ifdef USE_CVI_MPI
    void ensure_mapped() const;
    void unmap() const;
#endif

    StorageType storage_type_ = StorageType::NONE;
    PixelFormat format_ = PixelFormat::UNKNOWN;

    cv::Mat mat_;
    mutable cv::Mat mat_cache_;
    mutable bool mat_cache_valid_ = false;
    mutable cv::Mat mapped_view_;      // Zero-copy view of mmap'd CVI memory
    mutable void* mapped_ptr_ = nullptr;
    mutable size_t mapped_size_ = 0;

#ifdef USE_CVI_MPI
    VIDEO_FRAME_INFO_S cvi_frame_{};
    FrameOwner owner_ = FrameOwner::EXTERNAL;
    int vpss_grp_ = -1;
    int vpss_chn_ = -1;
    int vdec_chn_ = -1;
    VB_BLK vb_block_ = VB_INVALID_HANDLE;
#endif
};

} // namespace lua_cv
