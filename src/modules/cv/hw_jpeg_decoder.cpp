#include "hw_jpeg_decoder.h"

#ifdef USE_CVI_MPI
#include <cvi_sys.h>
#include <cvi_vb.h>
#include <cvi_buffer.h>
#include <linux/cvi_errno.h>
#include <iostream>
#include <stdexcept>
#include <cstring>
#include <sstream>
#include <unistd.h>
#include <fstream>
#include <chrono>
#include "mmf_context.h"
#endif

namespace lua_cv {

HwJpegDecoder::HwJpegDecoder() = default;

HwJpegDecoder::~HwJpegDecoder() {
#ifdef USE_CVI_MPI
    cleanup();
#endif
}

#ifdef USE_CVI_MPI
namespace {
void dump_proc_file(const char* path, size_t max_bytes) {
    std::ifstream file(path);
    if (!file) {
        std::cerr << "[WARN] HwJpegDecoder: cannot read " << path << std::endl;
        return;
    }

    std::ostringstream oss;
    std::string line;
    size_t total = 0;
    while (std::getline(file, line)) {
        if (total + line.size() + 1 > max_bytes) {
            line.resize(max_bytes - total);
        }
        oss << line << "\n";
        total += line.size() + 1;
        if (total >= max_bytes) {
            break;
        }
    }
    std::cerr << "[INFO] " << path << "\n" << oss.str() << std::endl;
}

void log_vdec_status(VDEC_CHN chn) {
    VDEC_CHN_STATUS_S status{};
    CVI_S32 rc = CVI_VDEC_QueryStatus(chn, &status);
    if (rc != CVI_SUCCESS) {
        std::cerr << "[WARN] HwJpegDecoder: CVI_VDEC_QueryStatus failed: 0x"
                  << std::hex << rc << std::dec << std::endl;
    } else {
        std::cerr << "[INFO] VDEC status chn=" << chn
                  << " left_bytes=" << status.u32LeftStreamBytes
                  << " left_frames=" << status.u32LeftStreamFrames
                  << " left_pics=" << status.u32LeftPics
                  << " recv=" << status.u32RecvStreamFrames
                  << " decoded=" << status.u32DecodeStreamFrames
                  << " size=" << status.u32Width << "x" << status.u32Height
                  << " err_picbuf=" << status.stVdecDecErr.s32PicBufSizeErrSet
                  << " err_stream=" << status.stVdecDecErr.s32StreamUnsprt
                  << std::endl;
    }
}
}  // namespace

bool HwJpegDecoder::init(uint32_t max_width, uint32_t max_height) {
    if (initialized_) {
        return true;
    }
    if (!MmfContext::instance().is_initialized()) {
        return false;
    }
    if (!MmfContext::is_supported_image_size(max_width, max_height)) {
        std::cerr << "[ERROR] HwJpegDecoder: unsupported image resolution "
                  << max_width << "x" << max_height << std::endl;
        return false;
    }

    if (!configure_channel(max_width, max_height)) {
        return false;
    }

    CVI_S32 rc = CVI_VDEC_StartRecvStream(chn_);
    if (rc != CVI_SUCCESS) {
        std::cerr << "[ERROR] CVI_VDEC_StartRecvStream failed: 0x"
                  << std::hex << rc << std::dec << std::endl;
        cleanup();
        return false;
    }

    initialized_ = true;
    max_width_ = max_width;
    max_height_ = max_height;
    return true;
}

bool HwJpegDecoder::ensure_initialized(uint32_t width, uint32_t height) {
    if (width == 0 || height == 0) {
        return false;
    }
    if (initialized_) {
        if (width <= max_width_ && height <= max_height_) {
            return true;
        }
        cleanup();
    }
    return init(width, height);
}

void HwJpegDecoder::cleanup() {
    if (!initialized_) {
        return;
    }

    std::cout << "[VDEC] Cleanup - chn=" << chn_ << std::endl;
    CVI_VDEC_StopRecvStream(chn_);
    CVI_VDEC_DestroyChn(chn_);
    initialized_ = false;
    max_width_ = 0;
    max_height_ = 0;
    std::cout << "[VDEC] Cleanup DONE" << std::endl;
}

VIDEO_FRAME_INFO_S HwJpegDecoder::decode(const uint8_t* data, size_t size) {
    return decode_sync(data, size);
}

VIDEO_FRAME_INFO_S HwJpegDecoder::decode_sync(const uint8_t* data, size_t size) {
    if (!initialized_) {
        throw std::runtime_error("HwJpegDecoder::decode_sync - decoder not initialized");
    }

    size_t jpeg_start = 0;
    size_t jpeg_end = 0;
    if (!find_jpeg_range(data, size, jpeg_start, jpeg_end)) {
        throw std::runtime_error("HwJpegDecoder::decode_sync - invalid JPEG data");
    }

    VDEC_STREAM_S stream{};
    stream.pu8Addr = const_cast<CVI_U8*>(reinterpret_cast<const CVI_U8*>(data + jpeg_start));
    stream.u32Len = static_cast<CVI_U32>(jpeg_end - jpeg_start);
    stream.u64PTS = 0;
    stream.bEndOfFrame = CVI_TRUE;
    stream.bEndOfStream = CVI_FALSE;
    stream.bDisplay = CVI_TRUE;

    CVI_S32 rc = CVI_SUCCESS;
    const int send_attempts = 3;
    const int send_timeout_ms = 200;
    for (int attempt = 0; attempt < send_attempts; ++attempt) {
        rc = CVI_VDEC_SendStream(chn_, &stream, send_timeout_ms);
        if (rc == CVI_SUCCESS) {
            break;
        }
        if (rc != CVI_ERR_VDEC_BUF_FULL && rc != CVI_ERR_VDEC_BUSY) {
            break;
        }
        usleep(2000);
    }
    if (rc != CVI_SUCCESS) {
        log_vdec_status(chn_);
        dump_proc_file("/proc/cvitek/vdec", 4096);
        dump_proc_file("/proc/cvitek/vb", 4096);
        std::ostringstream oss;
        oss << "HwJpegDecoder::decode_sync - CVI_VDEC_SendStream failed: 0x"
            << std::hex << rc;
        throw std::runtime_error(oss.str());
    }

    VIDEO_FRAME_INFO_S frame{};
    const int get_timeout_ms = 200;
    const int64_t max_wait_ms = 5000;
    auto start_time = std::chrono::steady_clock::now();
    while (true) {
        rc = CVI_VDEC_GetFrame(chn_, &frame, get_timeout_ms);
        if (rc == CVI_SUCCESS) {
            break;
        }
        if (rc != CVI_ERR_VDEC_BUF_EMPTY && rc != CVI_ERR_VDEC_BUSY) {
            log_vdec_status(chn_);
            dump_proc_file("/proc/cvitek/vdec", 4096);
            dump_proc_file("/proc/cvitek/vb", 4096);
            std::ostringstream oss;
            oss << "HwJpegDecoder::decode_sync - CVI_VDEC_GetFrame failed: 0x"
                << std::hex << rc;
            throw std::runtime_error(oss.str());
        }

        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count() >=
            max_wait_ms) {
            log_vdec_status(chn_);
            dump_proc_file("/proc/cvitek/vdec", 4096);
            dump_proc_file("/proc/cvitek/vb", 4096);
            throw std::runtime_error("HwJpegDecoder::decode_sync - timeout waiting for frame");
        }
    }

    std::cout << "[VDEC] GetFrame SUCCESS - chn=" << chn_
              << " phys=0x" << std::hex << frame.stVFrame.u64PhyAddr[0] << std::dec
              << " size=" << frame.stVFrame.u32Width << "x" << frame.stVFrame.u32Height
              << std::endl;
    return frame;
}

void HwJpegDecoder::release_frame(const VIDEO_FRAME_INFO_S& frame) {
    if (!initialized_) {
        std::cout << "[VDEC] ReleaseFrame SKIP - not initialized" << std::endl;
        return;
    }
    std::cout << "[VDEC] ReleaseFrame - chn=" << chn_
              << " phys=0x" << std::hex << frame.stVFrame.u64PhyAddr[0] << std::dec
              << std::endl;
    CVI_VDEC_ReleaseFrame(chn_, &frame);
}

bool HwJpegDecoder::find_jpeg_range(const uint8_t* data, size_t size, size_t& start, size_t& end) const {
    if (!data || size < 4) {
        return false;
    }

    start = size;
    for (size_t i = 0; i + 1 < size; ++i) {
        if (data[i] == 0xFF && data[i + 1] == 0xD8) {
            start = i;
            break;
        }
    }
    if (start == size) {
        return false;
    }

    end = 0;
    for (size_t i = size - 2; i > start; --i) {
        if (data[i] == 0xFF && data[i + 1] == 0xD9) {
            end = i + 2;
            break;
        }
    }

    return end > start;
}

bool HwJpegDecoder::configure_channel(uint32_t max_width, uint32_t max_height) {
    VDEC_CHN_ATTR_S attr{};
    attr.enType = PT_JPEG;
    attr.enMode = VIDEO_MODE_FRAME;
    attr.u32PicWidth = max_width;
    attr.u32PicHeight = max_height;
    attr.u32StreamBufSize = ALIGN(max_width * max_height, 0x4000);
    attr.u32FrameBufSize = VDEC_GetPicBufferSize(PT_JPEG, max_width, max_height,
                                                 PIXEL_FORMAT_NV21, DATA_BITWIDTH_8,
                                                 COMPRESS_MODE_NONE);
    attr.u32FrameBufCnt = 2;

    CVI_S32 rc = CVI_VDEC_CreateChn(chn_, &attr);
    if (rc != CVI_SUCCESS) {
        std::cerr << "[ERROR] CVI_VDEC_CreateChn failed: 0x"
                  << std::hex << rc << std::dec << std::endl;
        return false;
    }

    VDEC_CHN_PARAM_S param{};
    param.enType = PT_JPEG;
    param.enPixelFormat = PIXEL_FORMAT_NV21;
    param.u32DisplayFrameNum = 0;
    param.stVdecPictureParam.u32Alpha = 255;
    rc = CVI_VDEC_SetChnParam(chn_, &param);
    if (rc != CVI_SUCCESS) {
        std::cerr << "[ERROR] CVI_VDEC_SetChnParam failed: 0x"
                  << std::hex << rc << std::dec << std::endl;
        CVI_VDEC_DestroyChn(chn_);
        return false;
    }

    VDEC_MOD_PARAM_S mod_param{};
    rc = CVI_VDEC_GetModParam(&mod_param);
    if (rc == CVI_SUCCESS && mod_param.enVdecVBSource == VB_SOURCE_USER) {
        VB_POOL pool = select_pool(max_width, max_height);
        if (pool != VB_INVALID_POOLID) {
            VDEC_CHN_POOL_S pool_attr{};
            pool_attr.hPicVbPool = pool;
            pool_attr.hTmvVbPool = VB_INVALID_POOLID;
            rc = CVI_VDEC_AttachVbPool(chn_, &pool_attr);
            if (rc != CVI_SUCCESS) {
                std::cerr << "[ERROR] CVI_VDEC_AttachVbPool failed: 0x"
                          << std::hex << rc << std::dec << std::endl;
                CVI_VDEC_DestroyChn(chn_);
                return false;
            }
        }
    }

    return true;
}

VB_POOL HwJpegDecoder::select_pool(uint32_t width, uint32_t height) const {
    if (MmfContext::instance().is_initialized()) {
        return MmfContext::instance().vb_plan().find_pool(width, height, PixelFormat::NV21);
    }
    return VB_INVALID_POOLID;
}
#endif

} // namespace lua_cv
