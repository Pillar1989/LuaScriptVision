#include "hw_jpeg_decoder.h"

#ifdef USE_CVI_MPI

#include <fstream>
#include <stdexcept>
#include <cstring>
#include <iostream>
#include <sstream>
#include <unistd.h>
#include <cvi_buffer.h>

namespace lua_cv {

// Default VDEC channel for JPEG decoding
// VDEC has 64 channels (0-63), use channel 0 for JPEG by default
// Different from VPSS_GRP which is used for video processing
static constexpr VDEC_CHN DEFAULT_JPEG_VDEC_CHN = 0;

// Alignment macro (same as SDK)
#ifndef ALIGN
#define ALIGN(x, a) (((x) + ((a) - 1)) & ~((a) - 1))
#endif

HwJpegDecoder::HwJpegDecoder()
    : vdec_chn_(DEFAULT_JPEG_VDEC_CHN)
    , vb_pool_(VB_INVALID_POOLID)
    , initialized_(false)
    , max_width_(0)
    , max_height_(0)
    , first_decode_(true)
    , threads_running_(false)
    , task_state_(TASK_IDLE) {
    pthread_mutex_init(&task_mutex_, nullptr);
    pthread_cond_init(&task_cond_, nullptr);
}

HwJpegDecoder::~HwJpegDecoder() {
    cleanup();
    pthread_mutex_destroy(&task_mutex_);
    pthread_cond_destroy(&task_cond_);
}

bool HwJpegDecoder::init(uint32_t max_width, uint32_t max_height) {
    if (initialized_) {
        // Already initialized, check if dimensions match
        if (max_width <= max_width_ && max_height <= max_height_) {
            return true;
        }
        // Need to reinitialize with larger dimensions
        cleanup();
    }

    CVI_S32 rc;

    // CRITICAL: Do NOT call SetModParam - let VDEC use default VB_SOURCE_COMMON mode
    // SDK sample only calls SetModParam when g_VdecVbSrc != VB_SOURCE_COMMON

    // Step 1: Configure VDEC channel attributes for JPEG
    VDEC_CHN_ATTR_S chn_attr;
    std::memset(&chn_attr, 0, sizeof(chn_attr));

    chn_attr.enType = PT_JPEG;                    // JPEG payload type
    chn_attr.enMode = VIDEO_MODE_FRAME;           // Frame mode (complete JPEG per call)
    chn_attr.u32PicWidth = max_width;             // Max picture width
    chn_attr.u32PicHeight = max_height;           // Max picture height
    // Stream buffer size: aligned to 0x4000 (16KB) as per SDK requirement
    chn_attr.u32StreamBufSize = ALIGN(max_width * max_height, 0x4000);
    // For JPEG: FrameBufCnt = 1 (as per SDK sample vdecInitAttr line 299-301)
    chn_attr.u32FrameBufCnt = 1;
    // Calculate frame buffer size for JPEG
    // SDK sample uses YUV_PLANAR_444 as default for JPEG (vdecInitAttr line 259-261)
    uint32_t frame_buf_size = VDEC_GetPicBufferSize(
        PT_JPEG, max_width, max_height,
        PIXEL_FORMAT_YUV_PLANAR_444, DATA_BITWIDTH_8, COMPRESS_MODE_NONE);
    chn_attr.u32FrameBufSize = frame_buf_size;

    // Step 2: Create VDEC channel
    rc = CVI_VDEC_CreateChn(vdec_chn_, &chn_attr);
    if (rc != CVI_SUCCESS) {
        std::cerr << "[ERROR] CVI_VDEC_CreateChn failed: 0x" << std::hex << rc << std::dec << std::endl;
        return false;
    }

    // Step 3: Configure channel parameters (BEFORE StartRecvStream)
    VDEC_CHN_PARAM_S chn_param;
    std::memset(&chn_param, 0, sizeof(chn_param));

    rc = CVI_VDEC_GetChnParam(vdec_chn_, &chn_param);
    if (rc != CVI_SUCCESS) {
        std::cerr << "[ERROR] CVI_VDEC_GetChnParam failed: 0x" << std::hex << rc << std::dec << std::endl;
        CVI_VDEC_DestroyChn(vdec_chn_);
        return false;
    }

    // SDK sample_common_vdec.c line 797-801:
    // For JPEG: set alpha, pixel format, display frame num
    chn_param.stVdecPictureParam.u32Alpha = 255;
    chn_param.enPixelFormat = PIXEL_FORMAT_YUV_PLANAR_444;
    // For JPEG: u32DisplayFrameNum = 0 (as per SDK sample vdecInitAttr line 300)
    chn_param.u32DisplayFrameNum = 0;

    rc = CVI_VDEC_SetChnParam(vdec_chn_, &chn_param);
    if (rc != CVI_SUCCESS) {
        std::cerr << "[ERROR] CVI_VDEC_SetChnParam failed: 0x" << std::hex << rc << std::dec << std::endl;
        CVI_VDEC_DestroyChn(vdec_chn_);
        return false;
    }

    // Step 4: Start receiving stream
    rc = CVI_VDEC_StartRecvStream(vdec_chn_);
    if (rc != CVI_SUCCESS) {
        std::cerr << "[ERROR] CVI_VDEC_StartRecvStream failed: 0x" << std::hex << rc << std::dec << std::endl;
        CVI_VDEC_DestroyChn(vdec_chn_);
        return false;
    }

    // CRITICAL: Give VDEC more time to actually start receiving
    // Reduced from 500ms to 100ms - threads will handle synchronization
    usleep(100000);  // 100ms delay

    max_width_ = max_width;
    max_height_ = max_height;
    initialized_ = true;
    vb_pool_ = VB_INVALID_POOLID;  // Not used in COMMON mode

    // Start permanent worker threads
    if (!threads_running_) {
        // Set flag BEFORE creating threads to avoid race condition
        threads_running_ = true;

        pthread_attr_t attr;
        struct sched_param param;
        param.sched_priority = 80;
        pthread_attr_init(&attr);
        pthread_attr_setschedpolicy(&attr, SCHED_RR);
        pthread_attr_setschedparam(&attr, &param);
        pthread_attr_setinheritsched(&attr, PTHREAD_EXPLICIT_SCHED);

        int ret = pthread_create(&send_thread_, &attr, send_stream_thread_func, this);
        if (ret != 0) {
            std::cerr << "[ERROR] Failed to create SendStream thread: " << ret << std::endl;
            pthread_attr_destroy(&attr);
            threads_running_ = false;
            CVI_VDEC_DestroyChn(vdec_chn_);
            initialized_ = false;
            return false;
        }

        ret = pthread_create(&get_thread_, &attr, get_frame_thread_func, this);
        if (ret != 0) {
            std::cerr << "[ERROR] Failed to create GetFrame thread: " << ret << std::endl;
            pthread_attr_destroy(&attr);
            threads_running_ = false;
            pthread_cancel(send_thread_);
            pthread_join(send_thread_, nullptr);
            CVI_VDEC_DestroyChn(vdec_chn_);
            initialized_ = false;
            return false;
        }

        pthread_attr_destroy(&attr);
    }

    return true;
}

VIDEO_FRAME_INFO_S HwJpegDecoder::decode(const uint8_t* data, size_t size) {
    if (!initialized_) {
        throw std::runtime_error("HwJpegDecoder::decode - decoder not initialized");
    }

    // Only delay on first decode (give VDEC channel time to stabilize)
    if (first_decode_) {
        usleep(200000);  // 200ms, only on first decode
        first_decode_ = false;
    }

    // Submit task to permanent thread pool
    pthread_mutex_lock(&task_mutex_);

    // Wait for any previous task to complete
    while (task_state_ != TASK_IDLE && task_state_ != TASK_COMPLETE && task_state_ != TASK_ERROR) {
        pthread_cond_wait(&task_cond_, &task_mutex_);
    }

    // Setup new task
    task_.data = data;
    task_.size = size;
    std::memset(&task_.frame, 0, sizeof(task_.frame));
    task_.error_msg.clear();

    task_state_ = TASK_READY;
    pthread_cond_broadcast(&task_cond_);

    // Wait for task to complete
    while (task_state_ != TASK_COMPLETE && task_state_ != TASK_ERROR) {
        pthread_cond_wait(&task_cond_, &task_mutex_);
    }

    // Capture result
    VIDEO_FRAME_INFO_S result = task_.frame;
    std::string error = task_.error_msg;

    // Reset task state
    task_state_ = TASK_IDLE;
    pthread_cond_broadcast(&task_cond_);

    pthread_mutex_unlock(&task_mutex_);

    if (!error.empty()) {
        throw std::runtime_error(error);
    }

    return result;
}

// ========== Permanent thread pool implementation ==========

void* HwJpegDecoder::send_stream_thread_func(void* arg) {
    HwJpegDecoder* decoder = static_cast<HwJpegDecoder*>(arg);
    decoder->send_stream_worker();
    return nullptr;
}

void* HwJpegDecoder::get_frame_thread_func(void* arg) {
    HwJpegDecoder* decoder = static_cast<HwJpegDecoder*>(arg);
    decoder->get_frame_worker();
    return nullptr;
}

void HwJpegDecoder::send_stream_worker() {
    while (threads_running_) {
        pthread_mutex_lock(&task_mutex_);

        // Wait for task to be ready
        while (task_state_ != TASK_READY && threads_running_) {
            pthread_cond_wait(&task_cond_, &task_mutex_);
        }

        // Check if we should exit
        if (!threads_running_) {
            pthread_mutex_unlock(&task_mutex_);
            break;
        }

        // Transition to SENDING state
        task_state_ = TASK_SENDING;
        pthread_cond_signal(&task_cond_);  // Notify get_frame_thread

        const uint8_t* data = task_.data;
        size_t size = task_.size;
        pthread_mutex_unlock(&task_mutex_);

        // Prepare stream structure
        VDEC_STREAM_S stream;
        std::memset(&stream, 0, sizeof(stream));
        stream.pu8Addr = const_cast<CVI_U8*>(data);
        stream.u32Len = static_cast<CVI_U32>(size);
        stream.bEndOfFrame = CVI_TRUE;
        stream.bEndOfStream = CVI_FALSE;
        stream.bDisplay = CVI_TRUE;
        stream.u64PTS = 0;

        // Send stream with retry logic
        CVI_S32 rc;
        int retry_count = 0;
        const int max_retries = 500;  // 5 seconds total

        do {
            rc = CVI_VDEC_SendStream(vdec_chn_, &stream, -1);

            if (rc == CVI_SUCCESS) {
                break;
            } else {
                retry_count++;
                if (retry_count >= max_retries) {
                    pthread_mutex_lock(&task_mutex_);
                    task_.error_msg = "SendStream timeout after " +
                                    std::to_string(max_retries) + " retries";
                    task_state_ = TASK_ERROR;
                    pthread_cond_broadcast(&task_cond_);
                    pthread_mutex_unlock(&task_mutex_);
                    // Continue to next task (don't exit thread)
                    break;
                }
                usleep(10000);  // 10ms
            }
        } while (rc != CVI_SUCCESS);

        // Loop back to wait for next task
    }
}

void HwJpegDecoder::get_frame_worker() {
    while (true) {
        pthread_mutex_lock(&task_mutex_);

        // Wait for SENDING state (SendStream has started)
        while (task_state_ != TASK_SENDING && threads_running_) {
            pthread_cond_wait(&task_cond_, &task_mutex_);
        }

        // Check if we should exit
        if (!threads_running_) {
            pthread_mutex_unlock(&task_mutex_);
            break;
        }

        pthread_mutex_unlock(&task_mutex_);

        // Wait a bit for SendStream to complete (50ms like SDK)
        usleep(50000);

        // Get decoded frame with retry logic
        CVI_S32 rc;
        int retry_count = 0;
        const int max_retries = 100;  // 1 second total

        while (retry_count < max_retries) {
            rc = CVI_VDEC_GetFrame(vdec_chn_, &task_.frame, 10);  // 10ms timeout

            if (rc == CVI_SUCCESS) {
                // CRITICAL: Invalidate cache for all planes
                for (int i = 0; i < 3; i++) {
                    if (task_.frame.stVFrame.pu8VirAddr[i]) {
                        CVI_SYS_IonInvalidateCache(
                            task_.frame.stVFrame.u64PhyAddr[i],
                            task_.frame.stVFrame.pu8VirAddr[i],
                            task_.frame.stVFrame.u32Stride[i] * task_.frame.stVFrame.u32Height);
                    }
                }

                pthread_mutex_lock(&task_mutex_);
                task_state_ = TASK_COMPLETE;
                pthread_cond_broadcast(&task_cond_);
                pthread_mutex_unlock(&task_mutex_);
                break;
            } else if (rc == CVI_ERR_VDEC_BUF_EMPTY || rc == CVI_ERR_VDEC_BUSY) {
                retry_count++;
                if (retry_count >= max_retries) {
                    pthread_mutex_lock(&task_mutex_);
                    task_.error_msg = "GetFrame timeout after " +
                                    std::to_string(max_retries) + " retries";
                    task_state_ = TASK_ERROR;
                    pthread_cond_broadcast(&task_cond_);
                    pthread_mutex_unlock(&task_mutex_);
                    break;
                }
                usleep(10000);  // 10ms
                continue;
            } else {
                pthread_mutex_lock(&task_mutex_);
                task_.error_msg = "GetFrame failed: 0x" + std::to_string(rc);
                task_state_ = TASK_ERROR;
                pthread_cond_broadcast(&task_cond_);
                pthread_mutex_unlock(&task_mutex_);
                break;
            }
        }
    }
}

VIDEO_FRAME_INFO_S HwJpegDecoder::decode_file(const std::string& filepath) {
    // Read file into memory
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("HwJpegDecoder::decode_file - cannot open file: " + filepath);
    }

    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> file_data(file_size);
    if (!file.read(reinterpret_cast<char*>(file_data.data()), file_size)) {
        throw std::runtime_error("HwJpegDecoder::decode_file - failed to read file: " + filepath);
    }
    file.close();

    return decode(file_data.data(), file_data.size());
}

VIDEO_FRAME_INFO_S HwJpegDecoder::decode_sync(const uint8_t* data, size_t size) {
    if (!initialized_) {
        throw std::runtime_error("HwJpegDecoder::decode_sync - decoder not initialized");
    }

    // Only delay on first decode (give VDEC channel time to stabilize)
    if (first_decode_) {
        usleep(200000);  // 200ms, only on first decode
        first_decode_ = false;
    }

    // Prepare stream structure
    VDEC_STREAM_S stream;
    std::memset(&stream, 0, sizeof(stream));
    stream.pu8Addr = const_cast<CVI_U8*>(data);
    stream.u32Len = static_cast<CVI_U32>(size);
    stream.bEndOfFrame = CVI_TRUE;
    stream.bEndOfStream = CVI_FALSE;
    stream.bDisplay = CVI_TRUE;
    stream.u64PTS = 0;

    // SendStream with blocking mode (-1)
    CVI_S32 rc = CVI_VDEC_SendStream(vdec_chn_, &stream, -1);
    if (rc != CVI_SUCCESS) {
        std::ostringstream oss;
        oss << "HwJpegDecoder::decode_sync - SendStream failed: 0x" << std::hex << rc;
        throw std::runtime_error(oss.str());
    }

    // GetFrame with blocking mode (-1)
    // No need for 50ms delay - blocking mode handles synchronization
    VIDEO_FRAME_INFO_S frame;
    std::memset(&frame, 0, sizeof(frame));

    rc = CVI_VDEC_GetFrame(vdec_chn_, &frame, -1);
    if (rc != CVI_SUCCESS) {
        std::ostringstream oss;
        oss << "HwJpegDecoder::decode_sync - GetFrame failed: 0x" << std::hex << rc;
        throw std::runtime_error(oss.str());
    }

    // Invalidate cache for all planes (CPU needs to see DMA-written data)
    for (int i = 0; i < 3; i++) {
        if (frame.stVFrame.pu8VirAddr[i]) {
            CVI_SYS_IonInvalidateCache(
                frame.stVFrame.u64PhyAddr[i],
                frame.stVFrame.pu8VirAddr[i],
                frame.stVFrame.u32Stride[i] * frame.stVFrame.u32Height);
        }
    }

    return frame;
}

VIDEO_FRAME_INFO_S HwJpegDecoder::decode_file_sync(const std::string& filepath) {
    // Read file into memory
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("HwJpegDecoder::decode_file_sync - cannot open file: " + filepath);
    }

    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> file_data(file_size);
    if (!file.read(reinterpret_cast<char*>(file_data.data()), file_size)) {
        throw std::runtime_error("HwJpegDecoder::decode_file_sync - failed to read file: " + filepath);
    }
    file.close();

    return decode_sync(file_data.data(), file_data.size());
}

void HwJpegDecoder::release_frame(const VIDEO_FRAME_INFO_S& frame) {
    if (!initialized_) {
        return;
    }
    CVI_VDEC_ReleaseFrame(vdec_chn_, &frame);
}

void HwJpegDecoder::cleanup() {
    if (!initialized_) {
        return;
    }

    // Stop permanent worker threads first
    if (threads_running_) {
        pthread_mutex_lock(&task_mutex_);
        threads_running_ = false;
        task_state_ = TASK_IDLE;
        pthread_cond_broadcast(&task_cond_);
        pthread_mutex_unlock(&task_mutex_);

        // Wait for threads to exit
        pthread_join(send_thread_, nullptr);
        pthread_join(get_thread_, nullptr);

        threads_running_ = false;
    }

    // Stop receiving stream
    CVI_VDEC_StopRecvStream(vdec_chn_);

    // Reset channel (as per SDK sample - must be done before destroy)
    CVI_VDEC_ResetChn(vdec_chn_);

    // Destroy channel
    CVI_VDEC_DestroyChn(vdec_chn_);

    // For COMMON mode, no VB pool to detach/destroy

    initialized_ = false;
    max_width_ = 0;
    max_height_ = 0;
}

} // namespace lua_cv

#endif // USE_CVI_MPI
