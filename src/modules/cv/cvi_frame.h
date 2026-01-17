#pragma once

#include "cv_types.h"
#include <opencv2/opencv.hpp>
#include <variant>
#include <memory>

#ifdef USE_CVI_MPI
#include <linux/cvi_comm_video.h>
#endif

namespace lua_cv {

/**
 * Frame - Unified frame representation supporting both CPU and hardware backends
 *
 * This class provides a unified interface for image frames that can be backed by either:
 * 1. cv::Mat (CPU/OpenCV backend)
 * 2. VIDEO_FRAME_INFO_S (SG200X hardware backend with zero-copy support)
 *
 * The class automatically handles conversions between formats when needed (lazy conversion),
 * and properly manages memory ownership to avoid leaks.
 *
 * Zero-copy workflow:
 *   Camera → VI → VPSS → CvFrame(VIDEO_FRAME_INFO_S) → TPU
 *   (all using physical addresses, no memcpy)
 *
 * CPU fallback workflow:
 *   imread → CvFrame(cv::Mat) → OpenCV operations → Tensor
 *   (standard memory copies)
 */
class Frame {
public:
    /**
     * StorageType - Identifies which backend is currently active
     */
    enum class StorageType {
        EMPTY,          // No data (default constructed)
        OPENCV,         // Backed by cv::Mat (CPU)
        VIDEO_FRAME     // Backed by VIDEO_FRAME_INFO_S (hardware, zero-copy)
    };

    // ========== Constructors ==========

    /**
     * Default constructor - creates empty frame
     */
    Frame();

    /**
     * Construct from OpenCV Mat (CPU backend)
     * @param mat OpenCV matrix (will be cloned)
     */
    explicit Frame(const cv::Mat& mat);

#ifdef USE_CVI_MPI
    /**
     * Construct from VIDEO_FRAME_INFO_S (hardware backend)
     * @param frame Video frame from VI/VPSS
     * @param owns Whether this Frame owns the memory and should release it
     *             - true: Frame calls CVI_VPSS_ReleaseChnFrame() on destruction
     *             - false: External code manages lifetime (e.g., VI driver)
     */
    explicit Frame(const VIDEO_FRAME_INFO_S& frame, bool owns);

    /**
     * Construct from VIDEO_FRAME_INFO_S with VPSS context (for proper release)
     * @param frame Video frame from VPSS
     * @param grp VPSS group ID (for releasing frame)
     * @param chn VPSS channel ID (for releasing frame)
     *
     * This constructor automatically sets owns=true and will call
     * CVI_VPSS_ReleaseChnFrame(grp, chn, &frame) on destruction.
     */
    explicit Frame(const VIDEO_FRAME_INFO_S& frame, VPSS_GRP grp, VPSS_CHN chn);
#endif

    /**
     * Destructor - releases resources if owns_memory_ is true
     */
    ~Frame();

    // Disable copy (use explicit clone if needed)
    Frame(const Frame&) = delete;
    Frame& operator=(const Frame&) = delete;

    // Enable move
    Frame(Frame&& other) noexcept;
    Frame& operator=(Frame&& other) noexcept;

    // ========== Properties ==========

    /**
     * Get frame width in pixels
     */
    int width() const;

    /**
     * Get frame height in pixels
     */
    int height() const;

    /**
     * Get number of channels (e.g., 1 for GRAY, 3 for RGB/BGR)
     */
    int channels() const;

    /**
     * Get pixel format
     */
    PixelFormat format() const;

    /**
     * Get current storage type
     */
    StorageType storage_type() const { return type_; }

    /**
     * Check if frame is empty
     */
    bool empty() const { return type_ == StorageType::EMPTY; }

    // ========== Zero-Copy Support ==========

    /**
     * Check if frame has physical address (hardware backend)
     * Returns true only for VIDEO_FRAME_INFO_S backend
     */
    bool has_physical_addr() const;

    /**
     * Get physical address (for TPU zero-copy)
     * @throws std::runtime_error if frame doesn't have physical address
     */
    uint64_t physical_addr() const;

    /**
     * Get physical memory size in bytes
     * @throws std::runtime_error if frame doesn't have physical address
     */
    size_t physical_size() const;

    // ========== Conversions ==========

    /**
     * Convert to cv::Mat (lazy conversion, may cache result)
     * @return const reference to cv::Mat
     * @throws std::runtime_error if conversion fails
     */
    const cv::Mat& to_mat() const;

#ifdef USE_CVI_MPI
    /**
     * Get VIDEO_FRAME_INFO_S (only valid for hardware backend)
     * @return const reference to VIDEO_FRAME_INFO_S
     * @throws std::runtime_error if not a VIDEO_FRAME backend
     */
    const VIDEO_FRAME_INFO_S& to_video_frame() const;
#endif

    // ========== Resource Management ==========

    /**
     * Explicitly release resources
     * For VIDEO_FRAME_INFO_S: calls CVI_VPSS_ReleaseChnFrame if owns_memory_
     * For cv::Mat: releases mat
     */
    void release();

    /**
     * Check if this frame owns its memory
     */
    bool owns_memory() const { return owns_memory_; }

    /**
     * Clone this frame (deep copy)
     */
    Frame clone() const;

private:
    // ========== Internal State ==========

#ifdef USE_CVI_MPI
    std::variant<std::monostate, cv::Mat, VIDEO_FRAME_INFO_S> data_;
#else
    std::variant<std::monostate, cv::Mat> data_;
#endif

    StorageType type_;
    bool owns_memory_;

    // Lazy conversion cache (mutable because to_mat() is const)
    mutable cv::Mat mat_cache_;
    mutable bool mat_cache_valid_;

#ifdef USE_CVI_MPI
    // VPSS context for proper frame release (only when owns_memory_ is true)
    // If vpss_grp_ >= 0, this frame came from VPSS and should be released via CVI_VPSS_ReleaseChnFrame
    VPSS_GRP vpss_grp_ = -1;
    VPSS_CHN vpss_chn_ = -1;
#endif

    // ========== Internal Helpers ==========

#ifdef USE_CVI_MPI
    /**
     * Convert VIDEO_FRAME_INFO_S to cv::Mat (internal helper)
     */
    cv::Mat video_frame_to_mat(const VIDEO_FRAME_INFO_S& frame) const;
#endif

    /**
     * Helper to release VIDEO_FRAME resources
     */
    void release_video_frame();
};

} // namespace lua_cv
