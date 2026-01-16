#pragma once

#include <string>
#include <vector>
#include <cstring>

#ifdef USE_CVI_MPI
#include <linux/cvi_comm_video.h>
#include <cvi_vb.h>  // Must be before cvi_comm_vdec.h for VB_POOL type
#include <linux/cvi_comm_vdec.h>
#include <cvi_vdec.h>
#include <cvi_sys.h>
#include <pthread.h>
#endif

namespace lua_cv {

#ifdef USE_CVI_MPI

/**
 * HwJpegDecoder - Hardware-accelerated JPEG decoding using VDEC
 *
 * This class uses the SG200X VDEC (Video Decoder) hardware to decode JPEG images.
 * The output is a VIDEO_FRAME_INFO_S with physical address, ready for VPSS processing.
 *
 * Performance advantage over software decoding (cv::imdecode):
 * - Hardware JPEG decoder is significantly faster (~10x for large images)
 * - Output directly in VIDEO_FRAME_INFO_S format with physical address
 * - Zero-copy path to VPSS: VDEC → VPSS (no CPU involvement)
 *
 * Typical decode times (compared to OpenCV software decoding):
 * - 1920x1080 JPEG: ~10ms (hardware) vs ~120ms (software)
 * - 1280x720 JPEG: ~5ms (hardware) vs ~80ms (software)
 *
 * Usage:
 *   HwJpegDecoder decoder;
 *   decoder.init(1920, 1080);  // Set max dimensions
 *   VIDEO_FRAME_INFO_S frame = decoder.decode_file("image.jpg");
 *   // Use frame with VPSS...
 *   decoder.release_frame(frame);
 */
class HwJpegDecoder {
public:
    HwJpegDecoder();
    ~HwJpegDecoder();

    // Disable copy
    HwJpegDecoder(const HwJpegDecoder&) = delete;
    HwJpegDecoder& operator=(const HwJpegDecoder&) = delete;

    /**
     * Initialize VDEC channel for JPEG decoding
     *
     * @param max_width Maximum image width to decode
     * @param max_height Maximum image height to decode
     * @return true on success
     */
    bool init(uint32_t max_width, uint32_t max_height);

    /**
     * Decode JPEG from memory buffer (async with thread pool)
     *
     * @param data JPEG data buffer
     * @param size Size of data in bytes
     * @return VIDEO_FRAME_INFO_S with decoded frame (NV21 format)
     * @throws std::runtime_error on decode failure
     */
    VIDEO_FRAME_INFO_S decode(const uint8_t* data, size_t size);

    /**
     * Decode JPEG from memory buffer (synchronous, blocking mode)
     *
     * This is an optimized version that uses blocking API calls instead of
     * the thread pool. It avoids the 50ms thread synchronization delay.
     *
     * Performance: ~10-15ms vs ~50ms for async decode()
     *
     * @param data JPEG data buffer
     * @param size Size of data in bytes
     * @return VIDEO_FRAME_INFO_S with decoded frame (NV21 format)
     * @throws std::runtime_error on decode failure
     */
    VIDEO_FRAME_INFO_S decode_sync(const uint8_t* data, size_t size);

    /**
     * Decode JPEG file to VIDEO_FRAME_INFO_S (async)
     *
     * @param filepath Path to JPEG file
     * @return VIDEO_FRAME_INFO_S with decoded frame (NV21 format)
     * @throws std::runtime_error on decode failure
     */
    VIDEO_FRAME_INFO_S decode_file(const std::string& filepath);

    /**
     * Decode JPEG file to VIDEO_FRAME_INFO_S (synchronous, blocking mode)
     *
     * This is an optimized version using blocking API calls.
     *
     * @param filepath Path to JPEG file
     * @return VIDEO_FRAME_INFO_S with decoded frame (NV21 format)
     * @throws std::runtime_error on decode failure
     */
    VIDEO_FRAME_INFO_S decode_file_sync(const std::string& filepath);

    /**
     * Release decoded frame back to VDEC
     *
     * Must be called after processing is complete.
     *
     * @param frame Frame to release
     */
    void release_frame(const VIDEO_FRAME_INFO_S& frame);

    /**
     * Cleanup VDEC resources
     */
    void cleanup();

    /**
     * Check if decoder is initialized
     */
    bool is_initialized() const { return initialized_; }

private:
    // Task state for permanent thread pool
    enum TaskState {
        TASK_IDLE,       // No task
        TASK_READY,      // Task submitted, waiting for threads
        TASK_SENDING,    // SendStream thread working
        TASK_GETTING,    // GetFrame thread working
        TASK_COMPLETE,   // Task done, result ready
        TASK_ERROR       // Task failed
    };

    // Decoding context for permanent threads
    struct TaskContext {
        const uint8_t* data;
        size_t size;
        VIDEO_FRAME_INFO_S frame;
        std::string error_msg;

        TaskContext() : data(nullptr), size(0) {
            std::memset(&frame, 0, sizeof(frame));
        }
    };

    // Thread functions (static for pthread_create)
    static void* send_stream_thread_func(void* arg);
    static void* get_frame_thread_func(void* arg);

    // Worker thread implementations
    void send_stream_worker();
    void get_frame_worker();

    // VDEC resources
    VDEC_CHN vdec_chn_;
    VB_POOL vb_pool_;
    bool initialized_;
    uint32_t max_width_;
    uint32_t max_height_;

    // First decode flag (for one-time delay)
    bool first_decode_;

    // ========== Permanent Thread Pool ==========

    // Permanent threads
    pthread_t send_thread_;
    pthread_t get_thread_;
    bool threads_running_;

    // Task queue (single task at a time)
    TaskContext task_;
    TaskState task_state_;

    // Task synchronization
    pthread_mutex_t task_mutex_;
    pthread_cond_t task_cond_;
};

#else

// Stub when CVI MPI not available
class HwJpegDecoder {
public:
    HwJpegDecoder() = default;
    ~HwJpegDecoder() = default;

    bool init(uint32_t, uint32_t) {
        return false;
    }

    void* decode(const uint8_t*, size_t) {
        throw std::runtime_error("HwJpegDecoder requires USE_CVI_MPI");
    }

    void* decode_sync(const uint8_t*, size_t) {
        throw std::runtime_error("HwJpegDecoder requires USE_CVI_MPI");
    }

    void* decode_file(const std::string&) {
        throw std::runtime_error("HwJpegDecoder requires USE_CVI_MPI");
    }

    void* decode_file_sync(const std::string&) {
        throw std::runtime_error("HwJpegDecoder requires USE_CVI_MPI");
    }

    void release_frame(const void*) {}
    void cleanup() {}
    bool is_initialized() const { return false; }
};

#endif // USE_CVI_MPI

} // namespace lua_cv
