#pragma once

#include <string>
#include <opencv2/core.hpp>

#ifdef USE_CVI_MPI
#include <linux/cvi_comm_video.h>
#include <cvi_sys.h>
#include <cvi_vb.h>
#include "hw_jpeg_decoder.h"
#endif

namespace lua_cv {

#ifdef USE_CVI_MPI

/**
 * IonImageLoader - Zero-copy image loading for VPSS acceleration
 *
 * This class implements architecture-level optimization for file input:
 * - For JPEG: Optional VDEC hardware decoding (~10ms vs ~120ms software)
 * - For other formats: Software decode with VB buffer (eliminates memcpy)
 * - Uses VB pools (CVI_VB_GetBlock) to avoid Ion allocation conflicts
 *
 * Performance comparison (1920x1080 JPEG image):
 * - VDEC hardware: file read + VDEC decode = ~15ms total
 * - Software decode: file read + imdecode to VB + cache flush = ~125ms
 *
 * Memory architecture:
 * - VDEC mode: Uses VB pool (managed by VDEC driver, VB_SOURCE_COMMON)
 * - Software mode: Uses VB pool (CPU writes, VPSS reads via DMA)
 */
class IonImageLoader {
public:
    /**
     * JpegDecoder - JPEG decoder selection
     */
    enum JpegDecoder {
        JPEG_DECODER_OPENCV,    // Software decoding (default, compatible)
        JPEG_DECODER_VDEC       // Hardware VDEC decoding (fast, JPEG only)
    };

    IonImageLoader();
    ~IonImageLoader();

    // Disable copy (Ion resources are not copyable)
    IonImageLoader(const IonImageLoader&) = delete;
    IonImageLoader& operator=(const IonImageLoader&) = delete;

    // Enable move
    IonImageLoader(IonImageLoader&& other) noexcept;
    IonImageLoader& operator=(IonImageLoader&& other) noexcept;

    /**
     * Set JPEG decoder mode
     * @param decoder JPEG decoder to use (OPENCV or VDEC)
     *
     * Note: VDEC mode only works for JPEG format. Other formats always use OpenCV.
     */
    void set_jpeg_decoder(JpegDecoder decoder) { jpeg_decoder_ = decoder; }

    /**
     * Get current JPEG decoder mode
     */
    JpegDecoder get_jpeg_decoder() const { return jpeg_decoder_; }

    /**
     * Load image from file directly to VB buffer
     *
     * @param filepath Path to image file (jpg, png, etc.)
     * @return VIDEO_FRAME_INFO_S with physical address ready for VPSS
     * @throws std::runtime_error if file cannot be loaded
     *
     * Note: The returned frame shares VB buffer with this loader.
     * Call release() only after VPSS processing is complete.
     */
    VIDEO_FRAME_INFO_S load(const std::string& filepath);

    /**
     * Load image from memory buffer directly to VB buffer
     *
     * @param data Encoded image data (jpg, png, etc.)
     * @param size Size of data in bytes
     * @return VIDEO_FRAME_INFO_S with physical address ready for VPSS
     * @throws std::runtime_error if decode fails
     */
    VIDEO_FRAME_INFO_S load_from_memory(const uint8_t* data, size_t size);

    /**
     * Pre-allocate VB buffer for known dimensions (optimization)
     *
     * Call this before load_from_memory_fast() when you know the image dimensions.
     * This enables true zero-copy decoding for subsequent images of the same size.
     *
     * @param width Image width
     * @param height Image height
     * @param channels Number of channels (3 for BGR, 1 for grayscale)
     * @return true if buffer allocated successfully
     */
    bool preallocate(uint32_t width, uint32_t height, uint32_t channels = 3);

    /**
     * Load image from memory with pre-allocated buffer (optimized path)
     *
     * Requires preallocate() to be called first with matching dimensions.
     * If image dimensions don't match, falls back to standard load_from_memory().
     *
     * Performance: ~3x faster than load_from_memory() when dimensions match
     * because imdecode writes directly to VB buffer (no intermediate copy).
     *
     * @param data Encoded image data (jpg, png, etc.)
     * @param size Size of data in bytes
     * @return VIDEO_FRAME_INFO_S with physical address ready for VPSS
     * @throws std::runtime_error if decode fails
     */
    VIDEO_FRAME_INFO_S load_from_memory_fast(const uint8_t* data, size_t size);

    /**
     * Release VB buffer resources
     *
     * Call this after VPSS processing is complete.
     * Also called automatically by destructor.
     */
    void release();

    /**
     * Check if loader has valid VB buffer
     */
    bool is_valid() const { return vb_virt_addr_ != nullptr; }

    /**
     * Get physical address (for debugging/testing)
     */
    CVI_U64 get_phys_addr() const { return vb_phys_addr_; }

    /**
     * Get current buffer size
     */
    uint32_t get_buffer_size() const { return vb_size_; }

private:
    /**
     * Ensure VB buffer is allocated with sufficient size
     *
     * @param width Image width
     * @param height Image height
     * @param channels Number of channels (3 for BGR, 1 for grayscale)
     * @return true if buffer is ready, false on allocation failure
     */
    bool ensure_buffer(uint32_t width, uint32_t height, uint32_t channels);

    /**
     * Create VIDEO_FRAME_INFO_S from current VB buffer
     */
    VIDEO_FRAME_INFO_S create_video_frame(uint32_t width, uint32_t height,
                                           PIXEL_FORMAT_E format, uint32_t stride);

    /**
     * Check if data is JPEG format (by magic number)
     */
    static bool is_jpeg_format(const uint8_t* data, size_t size);

    /**
     * Decode using OpenCV software decoder
     */
    VIDEO_FRAME_INFO_S decode_with_opencv(const uint8_t* data, size_t size);

    /**
     * Decode using VDEC hardware decoder
     */
    VIDEO_FRAME_INFO_S decode_with_vdec(const uint8_t* data, size_t size);

    /**
     * Parse JPEG header to extract image dimensions (fast, no full decode)
     */
    static bool parse_jpeg_dimensions(const uint8_t* data, size_t size, int& width, int& height);

    // VB buffer resources (used for software decoding)
    // Changed from Ion to VB to avoid allocation conflicts with VB pools
    VB_BLK vb_block_;          // VB block handle
    CVI_U64 vb_phys_addr_;     // Physical address for DMA
    void* vb_virt_addr_;       // Virtual address for CPU access (mapped)
    uint32_t vb_size_;         // Current buffer size

    // Buffer configuration cache (for reuse optimization)
    uint32_t cached_width_;
    uint32_t cached_height_;
    uint32_t cached_channels_;

    // JPEG decoder selection
    JpegDecoder jpeg_decoder_;

    // VDEC decoder instance (lazy initialization)
    HwJpegDecoder* vdec_decoder_;
    bool vdec_initialized_;
    uint32_t vdec_max_width_;
    uint32_t vdec_max_height_;

    // VDEC frame lifecycle management
    bool has_vdec_frame_;
    VIDEO_FRAME_INFO_S current_vdec_frame_;
};

#else

// Stub implementation when CVI MPI is not available
class IonImageLoader {
public:
    IonImageLoader() = default;
    ~IonImageLoader() = default;

    // All operations throw when CVI MPI is not available
    void* load(const std::string& filepath) {
        throw std::runtime_error("IonImageLoader requires USE_CVI_MPI");
    }

    void* load_from_memory(const uint8_t* data, size_t size) {
        throw std::runtime_error("IonImageLoader requires USE_CVI_MPI");
    }

    void release() {}
    bool is_valid() const { return false; }
};

#endif // USE_CVI_MPI

} // namespace lua_cv
