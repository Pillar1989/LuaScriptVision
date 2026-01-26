/**
 * test_vdec_standalone.cpp - Standalone VDEC JPEG decoder test
 *
 * Completely mimics SDK sample_vdec initialization flow to isolate VDEC issues.
 * Does NOT use existing test_common.cpp infrastructure.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <iomanip>
#include <cstdlib>
#include <string>
#include <pthread.h>
#include <unistd.h>
#include <gtest/gtest.h>

#ifdef USE_CVI_MPI

#include <linux/cvi_comm_video.h>
#include <cvi_vb.h>
#include <linux/cvi_comm_vdec.h>
#include <cvi_vdec.h>
#include <cvi_sys.h>
#include <cvi_buffer.h>  // For VDEC_GetPicBufferSize

// Alignment macro (same as SDK)
#ifndef ALIGN
#define ALIGN(x, a) (((x) + ((a) - 1)) & ~((a) - 1))
#endif

// Execution mode
enum ExecutionMode {
    MODE_THREAD,      // Use separate threads for SendStream and GetFrame (default, like SDK)
    MODE_SEQUENTIAL   // Sequential execution: SendStream → Wait → GetFrame (simplified)
};

// Thread control
enum ThreadCtrl {
    THREAD_CTRL_START,
    THREAD_CTRL_PAUSE,
    THREAD_CTRL_STOP
};

// Thread parameters for SendStream
struct SendStreamParam {
    VDEC_CHN vdec_chn;
    std::vector<uint8_t>* file_data;
    bool* thread_started;
    ThreadCtrl* ctrl;
};

// Thread parameters for GetFrame
struct GetFrameParam {
    VDEC_CHN vdec_chn;
    bool* frame_received;
    ThreadCtrl* ctrl;
    VIDEO_FRAME_INFO_S* frame;
    const char* output_filepath;  // YUV output file path
};

// SendStream thread function
static void* send_stream_thread(void* arg) {
    SendStreamParam* param = static_cast<SendStreamParam*>(arg);

    std::cout << "[THREAD-SEND] SendStream thread started" << std::endl;
    *(param->thread_started) = true;

    // Give VDEC channel MORE time to stabilize (increased from 50ms to 200ms)
    // The channel needs time to be ready for receiving data
    usleep(200000);  // 200ms

    // Find JPEG start/end markers (0xFF 0xD8 and 0xFF 0xD9)
    uint8_t* data = param->file_data->data();
    size_t data_size = param->file_data->size();
    size_t jpeg_start = 0;
    size_t jpeg_end = data_size;

    // Find JPEG start (SOI marker)
    for (size_t i = 0; i < data_size - 1; i++) {
        if (data[i] == 0xFF && data[i + 1] == 0xD8) {
            jpeg_start = i;
            break;
        }
    }

    // Find JPEG end (EOI marker)
    for (size_t i = jpeg_start + 2; i < data_size - 1; i++) {
        if (data[i] == 0xFF && data[i + 1] == 0xD9) {
            jpeg_end = i + 2;
            break;
        }
    }

    size_t jpeg_size = jpeg_end - jpeg_start;
    std::cout << "[THREAD-SEND] JPEG data: offset=" << jpeg_start
              << ", size=" << jpeg_size << " bytes" << std::endl;

    // Prepare stream structure
    VDEC_STREAM_S stream;
    memset(&stream, 0, sizeof(stream));
    stream.pu8Addr = data + jpeg_start;
    stream.u32Len = static_cast<CVI_U32>(jpeg_size);
    stream.bEndOfFrame = CVI_TRUE;
    stream.bEndOfStream = CVI_FALSE;
    stream.bDisplay = CVI_TRUE;
    stream.u64PTS = 0;

    std::cout << "[THREAD-SEND] Sending stream to VDEC..." << std::endl;

    // Send stream with retry logic (same as SDK - retry on ALL errors while thread is running)
    CVI_S32 rc;
    int retry_count = 0;
    const int max_retries = 500;  // Increased max retries (5 seconds total)

    do {
        rc = CVI_VDEC_SendStream(param->vdec_chn, &stream, -1);

        if (rc == CVI_SUCCESS) {
            std::cout << "[THREAD-SEND] SendStream succeeded" << std::endl;
            break;
        } else {
            retry_count++;
            if (retry_count >= max_retries) {
                std::cerr << "[THREAD-SEND] SendStream timeout after "
                          << max_retries << " retries, error: 0x"
                          << std::hex << rc << std::dec << std::endl;
                *(param->ctrl) = THREAD_CTRL_STOP;
                return reinterpret_cast<void*>(-1);
            }
            std::cout << "[THREAD-SEND] SendStream failed (0x" << std::hex << rc << std::dec
                      << "), retry " << retry_count << std::endl;
            usleep(10000);  // 10ms delay between retries
        }
    } while (rc != CVI_SUCCESS && *(param->ctrl) != THREAD_CTRL_STOP);

    // Signal end of stream
    memset(&stream, 0, sizeof(stream));
    stream.bEndOfStream = CVI_TRUE;
    CVI_VDEC_SendStream(param->vdec_chn, &stream, -1);

    std::cout << "[THREAD-SEND] SendStream thread exiting" << std::endl;
    return nullptr;
}

// GetFrame thread function
static void* get_frame_thread(void* arg) {
    GetFrameParam* param = static_cast<GetFrameParam*>(arg);

    std::cout << "[THREAD-GET] GetFrame thread started" << std::endl;

    // Wait longer for SendStream thread to complete (it waits 200ms + send time)
    usleep(500000);  // 500ms - wait for SendStream to complete

    while (*(param->ctrl) != THREAD_CTRL_STOP) {
        // Try to get frame (with timeout)
        CVI_S32 rc = CVI_VDEC_GetFrame(param->vdec_chn, param->frame, 1000);  // 1s timeout

        if (rc == CVI_SUCCESS) {
            std::cout << "[THREAD-GET] GetFrame succeeded!" << std::endl;

            // CRITICAL: Invalidate cache for all planes to ensure memory coherency
            // This is required on RISC-V systems with non-coherent caches
            for (int i = 0; i < 3; i++) {
                if (param->frame->stVFrame.pu8VirAddr[i]) {
                    CVI_SYS_IonInvalidateCache(
                        param->frame->stVFrame.u64PhyAddr[i],
                        param->frame->stVFrame.pu8VirAddr[i],
                        param->frame->stVFrame.u32Stride[i] * param->frame->stVFrame.u32Height);
                }
            }

            std::cout << "[THREAD-GET] Frame info:" << std::endl;
            std::cout << "  - Width: " << param->frame->stVFrame.u32Width << std::endl;
            std::cout << "  - Height: " << param->frame->stVFrame.u32Height << std::endl;
            std::cout << "  - PixelFormat: " << (int)param->frame->stVFrame.enPixelFormat << std::endl;
            std::cout << "  - PhysAddr[0]: 0x" << std::hex
                      << param->frame->stVFrame.u64PhyAddr[0] << std::dec << std::endl;

            // Save YUV data to file
            if (param->output_filepath) {
                FILE* fp = fopen(param->output_filepath, "wb");
                if (fp) {
                    VIDEO_FRAME_S* pstVFrame = &param->frame->stVFrame;

                    // Calculate total YUV size (Y plane + U plane + V plane)
                    size_t total_size = 0;
                    for (int i = 0; i < 3; i++) {
                        if (pstVFrame->pu8VirAddr[i]) {
                            size_t plane_size = pstVFrame->u32Stride[i] * pstVFrame->u32Height;
                            // For chroma planes in YUV420, height is halved
                            if (i > 0) {
                                plane_size = pstVFrame->u32Stride[i] * (pstVFrame->u32Height / 2);
                            }
                            total_size += plane_size;
                        }
                    }

                    std::cout << "[THREAD-GET] Saving YUV to " << param->output_filepath
                              << " (" << total_size << " bytes)" << std::endl;

                    // Write each plane
                    for (int i = 0; i < 3; i++) {
                        if (pstVFrame->pu8VirAddr[i]) {
                            size_t plane_size = pstVFrame->u32Stride[i] * pstVFrame->u32Height;
                            if (i > 0) {
                                plane_size = pstVFrame->u32Stride[i] * (pstVFrame->u32Height / 2);
                            }
                            fwrite(pstVFrame->pu8VirAddr[i], 1, plane_size, fp);
                        }
                    }

                    fclose(fp);
                    std::cout << "[THREAD-GET] YUV file saved successfully" << std::endl;
                } else {
                    std::cerr << "[THREAD-GET] Failed to open output file: "
                              << param->output_filepath << std::endl;
                }
            }

            *(param->frame_received) = true;
            *(param->ctrl) = THREAD_CTRL_STOP;
            break;
        } else if (rc == CVI_ERR_VDEC_BUF_EMPTY) {
            // No frame available yet, continue waiting
            usleep(10000);  // 10ms
            continue;
        } else if (rc == CVI_ERR_VDEC_BUSY) {
            // Timeout, retry
            usleep(10000);  // 10ms
            continue;
        } else {
            std::cerr << "[THREAD-GET] CVI_VDEC_GetFrame failed: 0x"
                      << std::hex << rc << std::dec << std::endl;
            *(param->ctrl) = THREAD_CTRL_STOP;
            break;
        }
    }

    std::cout << "[THREAD-GET] GetFrame thread exiting" << std::endl;
    return nullptr;
}

static VB_POOL create_vdec_vb_pool(uint32_t max_width, uint32_t max_height) {
    // Calculate frame buffer size for JPEG (YUV444)
    uint32_t frame_buf_size = VDEC_GetPicBufferSize(
        PT_JPEG, max_width, max_height,
        PIXEL_FORMAT_YUV_PLANAR_444, DATA_BITWIDTH_8, COMPRESS_MODE_NONE);

    std::cout << "[INIT] Creating private VB pool for VDEC..." << std::endl;
    std::cout << "[INIT] Frame buffer size: 0x" << std::hex << frame_buf_size << std::dec
              << " (" << (frame_buf_size / 1024.0 / 1024.0) << " MB)" << std::endl;

    // Create private VB pool for VDEC (VB_SOURCE_USER mode)
    VB_POOL_CONFIG_S pool_config;
    memset(&pool_config, 0, sizeof(pool_config));
    pool_config.u32BlkSize = frame_buf_size;  // Use u32BlkSize, not u64BlkSize
    pool_config.u32BlkCnt = 3;  // 3 blocks for JPEG decode
    pool_config.enRemapMode = VB_REMAP_MODE_NONE;
    snprintf(pool_config.acName, sizeof(pool_config.acName), "vdec_jpeg");

    VB_POOL pool = CVI_VB_CreatePool(&pool_config);
    if (pool == VB_INVALID_POOLID) {
        std::cerr << "[ERROR] CVI_VB_CreatePool failed" << std::endl;
        return VB_INVALID_POOLID;
    }

    std::cout << "[INIT] Created private VB pool " << pool << ": "
              << (frame_buf_size / 1024) << " KB x " << pool_config.u32BlkCnt << " blocks" << std::endl;

    return pool;
}

static bool init_vb_and_sys_for_vdec(uint32_t max_width, uint32_t max_height) {
    std::cout << "[INIT] Initializing VB and SYS for VDEC..." << std::endl;

    // CRITICAL: Follow SDK pattern - Exit first to ensure clean state
    CVI_SYS_Exit();
    CVI_VB_Exit();
    usleep(100000);  // 100ms delay after exit

    // Calculate required VB pool size for JPEG decoding
    uint32_t frame_buf_size = VDEC_GetPicBufferSize(
        PT_JPEG, max_width, max_height,
        PIXEL_FORMAT_YUV_PLANAR_444, DATA_BITWIDTH_8, COMPRESS_MODE_NONE);

    std::cout << "[INIT] Required frame buffer size: 0x" << std::hex << frame_buf_size << std::dec
              << " (" << (frame_buf_size / 1024.0 / 1024.0) << " MB)" << std::endl;

    // Configure VB pool for VDEC (VB_SOURCE_COMMON mode)
    VB_CONFIG_S vb_config;
    memset(&vb_config, 0, sizeof(vb_config));
    vb_config.u32MaxPoolCnt = 1;
    vb_config.astCommPool[0].u32BlkSize = frame_buf_size;
    vb_config.astCommPool[0].u32BlkCnt = 3;  // 3 blocks for JPEG
    vb_config.astCommPool[0].enRemapMode = VB_REMAP_MODE_NONE;

    CVI_S32 rc = CVI_VB_SetConfig(&vb_config);
    if (rc != CVI_SUCCESS) {
        std::cerr << "[ERROR] CVI_VB_SetConfig failed: 0x" << std::hex << rc << std::dec << std::endl;
        return false;
    }

    rc = CVI_VB_Init();
    if (rc != CVI_SUCCESS) {
        std::cerr << "[ERROR] CVI_VB_Init failed: 0x" << std::hex << rc << std::dec << std::endl;
        return false;
    }

    rc = CVI_SYS_Init();
    if (rc != CVI_SUCCESS) {
        std::cerr << "[ERROR] CVI_SYS_Init failed: 0x" << std::hex << rc << std::dec << std::endl;
        CVI_VB_Exit();
        return false;
    }

    std::cout << "[INIT] VB pool configured: "
              << (frame_buf_size / 1024) << " KB x " << vb_config.astCommPool[0].u32BlkCnt
              << " blocks" << std::endl;
    std::cout << "[INIT] VB and SYS initialized successfully" << std::endl;

    return true;
}

static void cleanup_vb_and_sys() {
    std::cout << "[CLEANUP] Shutting down SYS and VB..." << std::endl;
    CVI_SYS_Exit();
    CVI_VB_Exit();
}

static bool init_vdec_channel(VDEC_CHN vdec_chn, uint32_t max_width, uint32_t max_height) {
    std::cout << "[VDEC] Initializing channel " << vdec_chn << " (COMMON mode)..." << std::endl;

    CVI_S32 rc;

    // Do NOT call SetModParam - let VDEC use default VB_SOURCE_COMMON mode
    // SDK sample only calls SetModParam when g_VdecVbSrc != VB_SOURCE_COMMON

    // Step 1: Configure VDEC channel attributes for JPEG
    VDEC_CHN_ATTR_S chn_attr;
    memset(&chn_attr, 0, sizeof(chn_attr));

    chn_attr.enType = PT_JPEG;
    chn_attr.enMode = VIDEO_MODE_FRAME;
    chn_attr.u32PicWidth = max_width;
    chn_attr.u32PicHeight = max_height;
    chn_attr.u32StreamBufSize = ALIGN(max_width * max_height, 0x4000);

    // CRITICAL: For JPEG, set FrameBufCnt to 1 and calculate FrameBufSize
    chn_attr.u32FrameBufCnt = 1;
    chn_attr.u32FrameBufSize = VDEC_GetPicBufferSize(
        PT_JPEG, max_width, max_height,
        PIXEL_FORMAT_YUV_PLANAR_444, DATA_BITWIDTH_8, COMPRESS_MODE_NONE);

    std::cout << "[VDEC] Channel attributes:" << std::endl;
    std::cout << "  - Dimensions: " << max_width << "x" << max_height << std::endl;
    std::cout << "  - StreamBufSize: 0x" << std::hex << chn_attr.u32StreamBufSize << std::dec << std::endl;
    std::cout << "  - FrameBufSize: 0x" << std::hex << chn_attr.u32FrameBufSize << std::dec << std::endl;
    std::cout << "  - FrameBufCnt: " << chn_attr.u32FrameBufCnt << std::endl;

    rc = CVI_VDEC_CreateChn(vdec_chn, &chn_attr);
    if (rc != CVI_SUCCESS) {
        std::cerr << "[ERROR] CVI_VDEC_CreateChn failed: 0x" << std::hex << rc << std::dec << std::endl;
        return false;
    }
    std::cout << "[VDEC] CreateChn succeeded" << std::endl;

    // Do NOT call AttachVbPool in COMMON mode - VDEC uses common VB pools automatically

    // Step 2: Set channel parameters
    VDEC_CHN_PARAM_S chn_param;
    memset(&chn_param, 0, sizeof(chn_param));

    rc = CVI_VDEC_GetChnParam(vdec_chn, &chn_param);
    if (rc != CVI_SUCCESS) {
        std::cerr << "[ERROR] CVI_VDEC_GetChnParam failed: 0x" << std::hex << rc << std::dec << std::endl;
        CVI_VDEC_DestroyChn(vdec_chn);
        return false;
    }

    // For JPEG, set alpha and pixel format (stVdecPictureParam)
    chn_param.stVdecPictureParam.u32Alpha = 255;
    chn_param.enPixelFormat = PIXEL_FORMAT_YUV_PLANAR_444;
    chn_param.u32DisplayFrameNum = 0;

    rc = CVI_VDEC_SetChnParam(vdec_chn, &chn_param);
    if (rc != CVI_SUCCESS) {
        std::cerr << "[ERROR] CVI_VDEC_SetChnParam failed: 0x" << std::hex << rc << std::dec << std::endl;
        CVI_VDEC_DestroyChn(vdec_chn);
        return false;
    }
    std::cout << "[VDEC] SetChnParam succeeded" << std::endl;

    // Step 3: Start receiving stream
    rc = CVI_VDEC_StartRecvStream(vdec_chn);
    if (rc != CVI_SUCCESS) {
        std::cerr << "[ERROR] CVI_VDEC_StartRecvStream failed: 0x" << std::hex << rc << std::dec << std::endl;
        CVI_VDEC_DestroyChn(vdec_chn);
        return false;
    }
    std::cout << "[VDEC] StartRecvStream succeeded" << std::endl;

    // Give VDEC more time to actually start receiving
    // The channel might take longer to be ready
    usleep(500000);  // 500ms delay

    // Step 4: Query status to verify (same as SDK sample)
    VDEC_CHN_STATUS_S status;
    memset(&status, 0, sizeof(status));
    rc = CVI_VDEC_QueryStatus(vdec_chn, &status);
    if (rc == CVI_SUCCESS) {
        std::cout << "[VDEC] Channel status after StartRecvStream:" << std::endl;
        std::cout << "  - bStartRecvStream: " << (int)status.bStartRecvStream << std::endl;
        std::cout << "  - enType: " << (int)status.enType << std::endl;
    }

    return true;
}

static void cleanup_vdec_channel(VDEC_CHN vdec_chn) {
    std::cout << "[VDEC] Cleaning up channel " << vdec_chn << "..." << std::endl;
    CVI_VDEC_StopRecvStream(vdec_chn);
    CVI_VDEC_ResetChn(vdec_chn);
    CVI_VDEC_DestroyChn(vdec_chn);
    // No VB pool to destroy in COMMON mode
}

static bool decode_jpeg_file(VDEC_CHN vdec_chn, const std::string& filepath,
                             const char* output_filepath = nullptr) {
    std::cout << "[DECODE] Loading JPEG file: " << filepath << std::endl;

    // Read file into memory
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Cannot open file: " << filepath << std::endl;
        return false;
    }

    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> file_data(file_size);
    if (!file.read(reinterpret_cast<char*>(file_data.data()), file_size)) {
        std::cerr << "[ERROR] Failed to read file" << std::endl;
        return false;
    }
    file.close();

    std::cout << "[DECODE] File size: " << (file_size / 1024.0) << " KB" << std::endl;

    // Thread control variables
    ThreadCtrl ctrl = THREAD_CTRL_START;
    bool thread_started = false;
    bool frame_received = false;
    VIDEO_FRAME_INFO_S frame;
    memset(&frame, 0, sizeof(frame));

    // Setup thread parameters
    SendStreamParam send_param = {
        vdec_chn,
        &file_data,
        &thread_started,
        &ctrl
    };

    GetFrameParam get_param = {
        vdec_chn,
        &frame_received,
        &ctrl,
        &frame,
        output_filepath
    };

    // Create threads - use same thread attributes as SDK
    pthread_t send_thread, get_thread;
    pthread_attr_t attr;

    std::cout << "[DECODE] Creating threads..." << std::endl;

    // Set thread attributes (same as SDK: SCHED_RR, priority 80)
    struct sched_param param;
    param.sched_priority = 80;
    pthread_attr_init(&attr);
    pthread_attr_setschedpolicy(&attr, SCHED_RR);
    pthread_attr_setschedparam(&attr, &param);
    pthread_attr_setinheritsched(&attr, PTHREAD_EXPLICIT_SCHED);

    // CRITICAL: Create SendStream thread FIRST (same as SDK sample)
    int ret = pthread_create(&send_thread, &attr, send_stream_thread, &send_param);
    if (ret != 0) {
        std::cerr << "[ERROR] Failed to create SendStream thread: " << ret << std::endl;
        pthread_attr_destroy(&attr);
        return false;
    }

    // Give SendStream thread a head start to begin sending data
    usleep(50000);  // 50ms

    ret = pthread_create(&get_thread, &attr, get_frame_thread, &get_param);
    if (ret != 0) {
        std::cerr << "[ERROR] Failed to create GetFrame thread: " << ret << std::endl;
        ctrl = THREAD_CTRL_STOP;
        pthread_join(send_thread, nullptr);
        pthread_attr_destroy(&attr);
        return false;
    }

    pthread_attr_destroy(&attr);

    std::cout << "[DECODE] Threads created, waiting for completion..." << std::endl;

    // Wait for threads to complete
    pthread_join(send_thread, nullptr);
    pthread_join(get_thread, nullptr);

    std::cout << "[DECODE] Threads joined" << std::endl;

    // Check results
    if (frame_received) {
        std::cout << "[DECODE] Successfully decoded JPEG!" << std::endl;
        // Release frame
        CVI_VDEC_ReleaseFrame(vdec_chn, &frame);
        std::cout << "[DECODE] Frame released" << std::endl;
        return true;
    } else {
        std::cerr << "[ERROR] Failed to receive decoded frame" << std::endl;
        return false;
    }
}

// Sequential mode: SendStream → Wait → GetFrame (no threads)
static bool decode_jpeg_file_sequential(VDEC_CHN vdec_chn, const std::string& filepath,
                                        const char* output_filepath = nullptr) {
    std::cout << "[DECODE] Loading JPEG file (SEQUENTIAL mode): " << filepath << std::endl;

    // Read file into memory
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Cannot open file: " << filepath << std::endl;
        return false;
    }

    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> file_data(file_size);
    if (!file.read(reinterpret_cast<char*>(file_data.data()), file_size)) {
        std::cerr << "[ERROR] Failed to read file" << std::endl;
        return false;
    }
    file.close();

    std::cout << "[DECODE] File size: " << (file_size / 1024.0) << " KB" << std::endl;

    // Find JPEG start/end markers (0xFF 0xD8 and 0xFF 0xD9)
    uint8_t* data = file_data.data();
    size_t data_size = file_data.size();
    size_t jpeg_start = 0;
    size_t jpeg_end = data_size;

    // Find JPEG start (SOI marker)
    for (size_t i = 0; i < data_size - 1; i++) {
        if (data[i] == 0xFF && data[i + 1] == 0xD8) {
            jpeg_start = i;
            break;
        }
    }

    // Find JPEG end (EOI marker)
    for (size_t i = jpeg_start + 2; i < data_size - 1; i++) {
        if (data[i] == 0xFF && data[i + 1] == 0xD9) {
            jpeg_end = i + 2;
            break;
        }
    }

    size_t jpeg_size = jpeg_end - jpeg_start;
    std::cout << "[DECODE] JPEG data: offset=" << jpeg_start
              << ", size=" << jpeg_size << " bytes" << std::endl;

    // Prepare stream structure
    VDEC_STREAM_S stream;
    memset(&stream, 0, sizeof(stream));
    stream.pu8Addr = data + jpeg_start;
    stream.u32Len = static_cast<CVI_U32>(jpeg_size);
    stream.bEndOfFrame = CVI_TRUE;
    stream.bEndOfStream = CVI_FALSE;
    stream.bDisplay = CVI_TRUE;
    stream.u64PTS = 0;

    // Step 1: SendStream with retry logic
    std::cout << "[DECODE] Sending stream to VDEC..." << std::endl;
    CVI_S32 rc;
    int retry_count = 0;
    const int max_retries = 500;

    do {
        rc = CVI_VDEC_SendStream(vdec_chn, &stream, -1);

        if (rc == CVI_SUCCESS) {
            std::cout << "[DECODE] SendStream succeeded" << std::endl;
            break;
        } else {
            retry_count++;
            if (retry_count >= max_retries) {
                std::cerr << "[ERROR] SendStream timeout after "
                          << max_retries << " retries, error: 0x"
                          << std::hex << rc << std::dec << std::endl;
                return false;
            }
            std::cout << "[DECODE] SendStream failed (0x" << std::hex << rc << std::dec
                      << "), retry " << retry_count << std::endl;
            usleep(10000);  // 10ms delay between retries
        }
    } while (rc != CVI_SUCCESS);

    // Step 2: Wait for decoding and GetFrame
    std::cout << "[DECODE] Waiting for decoded frame..." << std::endl;
    VIDEO_FRAME_INFO_S frame;
    memset(&frame, 0, sizeof(frame));

    int getframe_retries = 0;
    const int max_getframe_retries = 100;  // 100 retries * 10ms = 1 second

    while (getframe_retries < max_getframe_retries) {
        rc = CVI_VDEC_GetFrame(vdec_chn, &frame, 100);  // 100ms timeout

        if (rc == CVI_SUCCESS) {
            std::cout << "[DECODE] GetFrame succeeded!" << std::endl;

            // Invalidate cache for all planes
            for (int i = 0; i < 3; i++) {
                if (frame.stVFrame.pu8VirAddr[i]) {
                    CVI_SYS_IonInvalidateCache(
                        frame.stVFrame.u64PhyAddr[i],
                        frame.stVFrame.pu8VirAddr[i],
                        frame.stVFrame.u32Stride[i] * frame.stVFrame.u32Height);
                }
            }

            std::cout << "[DECODE] Frame info:" << std::endl;
            std::cout << "  - Width: " << frame.stVFrame.u32Width << std::endl;
            std::cout << "  - Height: " << frame.stVFrame.u32Height << std::endl;
            std::cout << "  - PixelFormat: " << (int)frame.stVFrame.enPixelFormat << std::endl;
            std::cout << "  - PhysAddr[0]: 0x" << std::hex
                      << frame.stVFrame.u64PhyAddr[0] << std::dec << std::endl;

            // Save YUV data to file
            if (output_filepath) {
                FILE* fp = fopen(output_filepath, "wb");
                if (fp) {
                    VIDEO_FRAME_S* pstVFrame = &frame.stVFrame;

                    // Calculate total YUV size
                    size_t total_size = 0;
                    for (int i = 0; i < 3; i++) {
                        if (pstVFrame->pu8VirAddr[i]) {
                            size_t plane_size = pstVFrame->u32Stride[i] * pstVFrame->u32Height;
                            if (i > 0) {
                                plane_size = pstVFrame->u32Stride[i] * (pstVFrame->u32Height / 2);
                            }
                            total_size += plane_size;
                        }
                    }

                    std::cout << "[DECODE] Saving YUV to " << output_filepath
                              << " (" << total_size << " bytes)" << std::endl;

                    // Write each plane
                    for (int i = 0; i < 3; i++) {
                        if (pstVFrame->pu8VirAddr[i]) {
                            size_t plane_size = pstVFrame->u32Stride[i] * pstVFrame->u32Height;
                            if (i > 0) {
                                plane_size = pstVFrame->u32Stride[i] * (pstVFrame->u32Height / 2);
                            }
                            fwrite(pstVFrame->pu8VirAddr[i], 1, plane_size, fp);
                        }
                    }

                    fclose(fp);
                    std::cout << "[DECODE] YUV file saved successfully" << std::endl;
                } else {
                    std::cerr << "[ERROR] Failed to open output file: "
                              << output_filepath << std::endl;
                }
            }

            // Release frame
            CVI_VDEC_ReleaseFrame(vdec_chn, &frame);
            std::cout << "[DECODE] Frame released" << std::endl;
            return true;

        } else if (rc == CVI_ERR_VDEC_BUF_EMPTY) {
            // No frame available yet, continue waiting
            getframe_retries++;
            usleep(10000);  // 10ms
            continue;
        } else {
            std::cerr << "[ERROR] CVI_VDEC_GetFrame failed: 0x"
                      << std::hex << rc << std::dec << std::endl;
            return false;
        }
    }

    std::cerr << "[ERROR] GetFrame timeout after " << max_getframe_retries << " retries" << std::endl;
    return false;
}

struct VdecTestConfig {
    std::string jpeg_file;
    ExecutionMode mode = MODE_THREAD;
    std::string output_file;
};

namespace {
VdecTestConfig g_vdec_config;

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " <jpeg_file> [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --mode MODE     Execution mode (default: thread)" << std::endl;
    std::cout << "                   thread   : Use threads (like SDK sample)" << std::endl;
    std::cout << "                   sequential: Sequential execution (simplified)" << std::endl;
    std::cout << "  -o OUTPUT       YUV output file path (optional)" << std::endl;
    std::cout << "  --jpeg FILE     Input JPEG file path" << std::endl;
}

void parse_args(int* argc, char** argv) {
    std::vector<char*> keep;
    keep.reserve(static_cast<size_t>(*argc));
    keep.push_back(argv[0]);

    for (int i = 1; i < *argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        }
        if (arg == "--mode" && i + 1 < *argc) {
            std::string mode_str = argv[++i];
            if (mode_str == "thread" || mode_str == "t") {
                g_vdec_config.mode = MODE_THREAD;
            } else if (mode_str == "sequential" || mode_str == "s") {
                g_vdec_config.mode = MODE_SEQUENTIAL;
            } else {
                std::cerr << "[ERROR] Invalid mode: " << mode_str << std::endl;
                std::exit(1);
            }
            continue;
        }
        if ((arg == "-o" || arg == "--output") && i + 1 < *argc) {
            g_vdec_config.output_file = argv[++i];
            continue;
        }
        if (arg == "--jpeg" && i + 1 < *argc) {
            g_vdec_config.jpeg_file = argv[++i];
            continue;
        }
        if (arg.rfind("--jpeg=", 0) == 0) {
            g_vdec_config.jpeg_file = arg.substr(7);
            continue;
        }
        if (!arg.empty() && arg[0] != '-') {
            if (g_vdec_config.jpeg_file.empty()) {
                g_vdec_config.jpeg_file = arg;
                continue;
            }
        }

        keep.push_back(argv[i]);
    }

    int out_argc = 0;
    for (char* arg : keep) {
        argv[out_argc++] = arg;
    }
    *argc = out_argc;
}
}  // namespace

static bool run_vdec_test(const VdecTestConfig& config) {
    if (config.jpeg_file.empty()) {
        return false;
    }

    std::cout << "[CONFIG] Execution mode: "
              << (config.mode == MODE_THREAD ? "THREAD (parallel)" : "SEQUENTIAL (simplified)")
              << std::endl;
    std::cout << "[CONFIG] Input file: " << config.jpeg_file << std::endl;
    if (!config.output_file.empty()) {
        std::cout << "[CONFIG] Output file: " << config.output_file << std::endl;
    }

    const VDEC_CHN vdec_chn = 0;
    const uint32_t max_width = 1920;
    const uint32_t max_height = 1080;

    if (!init_vb_and_sys_for_vdec(max_width, max_height)) {
        return false;
    }

    if (!init_vdec_channel(vdec_chn, max_width, max_height)) {
        cleanup_vb_and_sys();
        return false;
    }

    bool success = false;
    if (config.mode == MODE_SEQUENTIAL) {
        success = decode_jpeg_file_sequential(vdec_chn, config.jpeg_file,
                                              config.output_file.empty() ? nullptr : config.output_file.c_str());
    } else {
        success = decode_jpeg_file(vdec_chn, config.jpeg_file,
                                   config.output_file.empty() ? nullptr : config.output_file.c_str());
    }

    cleanup_vdec_channel(vdec_chn);
    cleanup_vb_and_sys();
    return success;
}

TEST(VdecStandaloneTest, Decode) {
    if (g_vdec_config.jpeg_file.empty()) {
        GTEST_SKIP() << "No JPEG file provided. Use --jpeg <path> or positional arg.";
    }

    bool ok = run_vdec_test(g_vdec_config);
    EXPECT_TRUE(ok);
}

int main(int argc, char* argv[]) {
    parse_args(&argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

#else

TEST(VdecStandaloneTest, Skipped) {
    GTEST_SKIP() << "USE_CVI_MPI not defined";
}

int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

#endif // USE_CVI_MPI
