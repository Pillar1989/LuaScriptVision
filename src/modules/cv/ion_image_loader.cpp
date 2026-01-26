#include "ion_image_loader.h"

#ifdef USE_CVI_MPI
#include <opencv2/opencv.hpp>
#include <fstream>
#include <stdexcept>
#include <cstring>

#include "cvi_vpss_processor.h"
#include "hw_jpeg_decoder.h"
#include "mmf_context.h"
#include <cvi_sys.h>
#endif

namespace lua_cv {

IonImageLoader::IonImageLoader() {
#ifdef USE_CVI_MPI
    hw_decoder_ = new HwJpegDecoder();
#endif
}

IonImageLoader::~IonImageLoader() {
#ifdef USE_CVI_MPI
    release();
    delete hw_decoder_;
    hw_decoder_ = nullptr;
#endif
}

#ifdef USE_CVI_MPI
VIDEO_FRAME_INFO_S IonImageLoader::load(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("IonImageLoader::load - cannot open file: " + filepath);
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> data(size);
    if (!file.read(reinterpret_cast<char*>(data.data()), size)) {
        throw std::runtime_error("IonImageLoader::load - failed to read file: " + filepath);
    }

    return load_from_memory(data.data(), data.size());
}

VIDEO_FRAME_INFO_S IonImageLoader::load_from_memory(const uint8_t* data, size_t size) {
    if (!MmfContext::instance().is_initialized()) {
        throw std::runtime_error("IonImageLoader requires MmfContext initialization");
    }

    release_last_frame();

    if (!is_jpeg_format(data, size)) {
        throw std::runtime_error("IonImageLoader::load_from_memory - only JPEG supported");
    }

    if (!hw_decoder_) {
        throw std::runtime_error("IonImageLoader::load_from_memory - hardware decoder unavailable");
    }

    uint32_t jpeg_width = 0;
    uint32_t jpeg_height = 0;
    if (!get_jpeg_dimensions(data, size, jpeg_width, jpeg_height)) {
        throw std::runtime_error("IonImageLoader::load_from_memory - failed to parse JPEG header");
    }
    if (!hw_decoder_->ensure_initialized(jpeg_width, jpeg_height)) {
        throw std::runtime_error("IonImageLoader::load_from_memory - VDEC init failed");
    }

    last_hw_frame_ = hw_decoder_->decode_sync(data, size);
    has_hw_frame_ = true;
    return last_hw_frame_;
}

VIDEO_FRAME_INFO_S IonImageLoader::load_from_memory_fast(const uint8_t* data, size_t size) {
    if (!MmfContext::instance().is_initialized()) {
        throw std::runtime_error("IonImageLoader requires MmfContext initialization");
    }

    if (prealloc_width_ == 0 || prealloc_height_ == 0 || prealloc_channels_ == 0) {
        throw std::runtime_error("IonImageLoader::load_from_memory_fast - buffer not preallocated");
    }

    release_last_frame();

    if (!is_jpeg_format(data, size)) {
        throw std::runtime_error("IonImageLoader::load_from_memory_fast - only JPEG supported");
    }

    if (!hw_decoder_) {
        throw std::runtime_error("IonImageLoader::load_from_memory_fast - hardware decoder unavailable");
    }

    uint32_t jpeg_width = 0;
    uint32_t jpeg_height = 0;
    if (!get_jpeg_dimensions(data, size, jpeg_width, jpeg_height)) {
        throw std::runtime_error("IonImageLoader::load_from_memory_fast - failed to parse JPEG header");
    }
    if (jpeg_width != prealloc_width_ || jpeg_height != prealloc_height_) {
        throw std::runtime_error("IonImageLoader::load_from_memory_fast - JPEG size mismatch with preallocated buffer");
    }

    if (!hw_decoder_->is_initialized()) {
        if (!hw_decoder_->init(prealloc_width_, prealloc_height_)) {
            throw std::runtime_error("IonImageLoader::load_from_memory_fast - VDEC init failed");
        }
    }

    last_hw_frame_ = hw_decoder_->decode_sync(data, size);
    has_hw_frame_ = true;
    return last_hw_frame_;
}

bool IonImageLoader::preallocate(uint32_t width, uint32_t height, uint32_t channels) {
    if (!MmfContext::instance().is_initialized()) {
        return false;
    }

    release();

    prealloc_width_ = width;
    prealloc_height_ = height;
    prealloc_channels_ = channels;

    PixelFormat fmt = (channels == 1) ? PixelFormat::GRAY : PixelFormat::BGR;
    VB_POOL pool = select_pool(width, height, fmt);
    VB_CAL_CONFIG_S cal{};
    COMMON_GetPicBufferConfig(width, height, to_cvi_pixel_format(fmt),
                              DATA_BITWIDTH_8, COMPRESS_MODE_NONE, 0, &cal);

    vb_block_ = CVI_VB_GetBlock(pool, cal.u32VBSize);
    if (vb_block_ == VB_INVALID_HANDLE) {
        return false;
    }

    has_vb_block_ = true;
    vb_block_size_ = cal.u32VBSize;
    return true;
}

void IonImageLoader::release() {
    release_last_frame();
    if (has_vb_block_ && vb_block_ != VB_INVALID_HANDLE) {
        CVI_VB_ReleaseBlock(vb_block_);
    }
    vb_block_ = VB_INVALID_HANDLE;
    has_vb_block_ = false;
    vb_block_size_ = 0;
    prealloc_width_ = 0;
    prealloc_height_ = 0;
    prealloc_channels_ = 0;
}

bool IonImageLoader::is_jpeg_format(const uint8_t* data, size_t size) const {
    if (!data || size < 4) {
        return false;
    }
    return data[0] == 0xFF && data[1] == 0xD8 && data[size - 2] == 0xFF && data[size - 1] == 0xD9;
}

bool IonImageLoader::get_jpeg_dimensions(const uint8_t* data, size_t size,
                                         uint32_t& width, uint32_t& height) const {
    width = 0;
    height = 0;
    if (!data || size < 4 || data[0] != 0xFF || data[1] != 0xD8) {
        return false;
    }

    size_t i = 2;
    while (i + 1 < size) {
        if (data[i] != 0xFF) {
            ++i;
            continue;
        }

        while (i < size && data[i] == 0xFF) {
            ++i;
        }
        if (i >= size) {
            break;
        }

        uint8_t marker = data[i++];
        if (marker == 0xD9 || marker == 0xDA) {
            break;
        }
        if (marker == 0x01 || (marker >= 0xD0 && marker <= 0xD7)) {
            continue;
        }
        if (i + 1 >= size) {
            break;
        }

        uint16_t segment_len = static_cast<uint16_t>(data[i] << 8 | data[i + 1]);
        if (segment_len < 2 || i + segment_len > size) {
            break;
        }

        bool is_sof = (marker == 0xC0 || marker == 0xC1 || marker == 0xC2 ||
                       marker == 0xC3 || marker == 0xC5 || marker == 0xC6 ||
                       marker == 0xC7 || marker == 0xC9 || marker == 0xCA ||
                       marker == 0xCB || marker == 0xCD || marker == 0xCE ||
                       marker == 0xCF);
        if (is_sof) {
            if (segment_len < 7) {
                return false;
            }
            height = static_cast<uint32_t>(data[i + 3] << 8 | data[i + 4]);
            width = static_cast<uint32_t>(data[i + 5] << 8 | data[i + 6]);
            return width > 0 && height > 0;
        }

        i += segment_len;
    }

    return false;
}

bool IonImageLoader::decode_with_opencv(const uint8_t* data, size_t size) {
    cv::Mat mat = cv::imdecode(cv::_InputArray(data, static_cast<int>(size)), cv::IMREAD_COLOR);
    if (mat.empty()) {
        return false;
    }
    return decode_into_vb(mat);
}

bool IonImageLoader::decode_into_vb(const cv::Mat& mat) {
    PixelFormat fmt = (mat.channels() == 1) ? PixelFormat::GRAY : PixelFormat::BGR;
    VB_POOL pool = select_pool(static_cast<uint32_t>(mat.cols), static_cast<uint32_t>(mat.rows), fmt);

    VB_CAL_CONFIG_S cal{};
    COMMON_GetPicBufferConfig(static_cast<CVI_U32>(mat.cols),
                              static_cast<CVI_U32>(mat.rows),
                              to_cvi_pixel_format(fmt), DATA_BITWIDTH_8,
                              COMPRESS_MODE_NONE, 0, &cal);

    if (has_vb_block_ && vb_block_size_ < cal.u32VBSize) {
        CVI_VB_ReleaseBlock(vb_block_);
        vb_block_ = VB_INVALID_HANDLE;
        has_vb_block_ = false;
        vb_block_size_ = 0;
    }

    if (!has_vb_block_) {
        vb_block_ = CVI_VB_GetBlock(pool, cal.u32VBSize);
        if (vb_block_ == VB_INVALID_HANDLE) {
            return false;
        }
        has_vb_block_ = true;
        vb_block_size_ = cal.u32VBSize;
    }

    CVI_U64 phys = CVI_VB_Handle2PhysAddr(vb_block_);
    void* virt = CVI_SYS_MmapCache(phys, cal.u32VBSize);
    if (!virt) {
        return false;
    }

    int copy_bytes = mat.cols * mat.channels();
    int dst_stride = static_cast<int>(cal.u32MainStride);
    uint8_t* dst = static_cast<uint8_t*>(virt);

    for (int y = 0; y < mat.rows; ++y) {
        std::memcpy(dst + static_cast<size_t>(y) * dst_stride,
                    mat.ptr<uint8_t>(y),
                    static_cast<size_t>(copy_bytes));
    }

    CVI_SYS_IonFlushCache(phys, virt, cal.u32VBSize);
    CVI_SYS_Munmap(virt, cal.u32VBSize);

    last_frame_ = {};
    last_frame_.stVFrame.u32Width = static_cast<CVI_U32>(mat.cols);
    last_frame_.stVFrame.u32Height = static_cast<CVI_U32>(mat.rows);
    last_frame_.stVFrame.enPixelFormat = to_cvi_pixel_format(fmt);
    last_frame_.stVFrame.enVideoFormat = VIDEO_FORMAT_LINEAR;
    last_frame_.stVFrame.enCompressMode = COMPRESS_MODE_NONE;
    last_frame_.stVFrame.enDynamicRange = DYNAMIC_RANGE_SDR8;
    last_frame_.stVFrame.enColorGamut = COLOR_GAMUT_BT709;

    if (cal.plane_num <= 1) {
        last_frame_.stVFrame.u32Stride[0] = cal.u32MainStride;
        last_frame_.stVFrame.u32Length[0] = cal.u32MainSize;
    } else {
        last_frame_.stVFrame.u32Stride[0] = cal.u32MainStride;
        last_frame_.stVFrame.u32Stride[1] = cal.u32CStride;
        last_frame_.stVFrame.u32Length[0] = cal.u32MainYSize;
        last_frame_.stVFrame.u32Length[1] = cal.u32MainCSize;
        last_frame_.stVFrame.u64PhyAddr[1] = phys + cal.u32MainYSize;
    }

    last_frame_.stVFrame.u64PhyAddr[0] = phys;
    last_frame_.u32PoolId = CVI_VB_Handle2PoolId(vb_block_);

    return true;
}

VB_POOL IonImageLoader::select_pool(uint32_t width, uint32_t height, PixelFormat format) const {
    PixelFormat pool_format = format;
    if (format == PixelFormat::RGB || format == PixelFormat::GRAY) {
        pool_format = PixelFormat::BGR;
    }
    if (MmfContext::instance().is_initialized()) {
        VB_POOL pool = MmfContext::instance().vb_plan().find_pool(width, height, pool_format);
        if (pool != VB_INVALID_POOLID) {
            return pool;
        }
    }
    return VB_INVALID_POOLID;
}

void IonImageLoader::release_last_frame() {
    if (has_hw_frame_ && hw_decoder_) {
        hw_decoder_->release_frame(last_hw_frame_);
        has_hw_frame_ = false;
    }
    last_frame_ = {};
}
#endif

} // namespace lua_cv
