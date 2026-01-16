#include "ion_image_loader.h"
#include "hw_jpeg_decoder.h"

#ifdef USE_CVI_MPI

#include <fstream>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <iostream>
#include <sstream>
#include <opencv2/imgcodecs.hpp>

namespace lua_cv {

// DMA alignment requirement
constexpr uint32_t ALIGNMENT = 64;

static inline uint32_t ALIGN(uint32_t x, uint32_t a) {
    return (x + a - 1) & ~(a - 1);
}

IonImageLoader::IonImageLoader()
    : vb_block_(VB_INVALID_HANDLE)
    , vb_phys_addr_(0)
    , vb_virt_addr_(nullptr)
    , vb_size_(0)
    , cached_width_(0)
    , cached_height_(0)
    , cached_channels_(0)
    , jpeg_decoder_(JPEG_DECODER_OPENCV)
    , vdec_decoder_(nullptr)
    , vdec_initialized_(false)
    , vdec_max_width_(0)
    , vdec_max_height_(0)
    , has_vdec_frame_(false) {
    std::memset(&current_vdec_frame_, 0, sizeof(current_vdec_frame_));
}

IonImageLoader::~IonImageLoader() {
    release();
}

IonImageLoader::IonImageLoader(IonImageLoader&& other) noexcept
    : vb_block_(other.vb_block_)
    , vb_phys_addr_(other.vb_phys_addr_)
    , vb_virt_addr_(other.vb_virt_addr_)
    , vb_size_(other.vb_size_)
    , cached_width_(other.cached_width_)
    , cached_height_(other.cached_height_)
    , cached_channels_(other.cached_channels_)
    , jpeg_decoder_(other.jpeg_decoder_)
    , vdec_decoder_(nullptr)  // Do NOT transfer VDEC decoder
    , vdec_initialized_(false)
    , vdec_max_width_(0)
    , vdec_max_height_(0)
    , has_vdec_frame_(false) {
    std::memset(&current_vdec_frame_, 0, sizeof(current_vdec_frame_));

    // Clear source VB resources (but NOT VDEC decoder)
    other.vb_block_ = VB_INVALID_HANDLE;
    other.vb_phys_addr_ = 0;
    other.vb_virt_addr_ = nullptr;
    other.vb_size_ = 0;
}

IonImageLoader& IonImageLoader::operator=(IonImageLoader&& other) noexcept {
    if (this != &other) {
        release();  // Release current resources

        // Move VB resources
        vb_block_ = other.vb_block_;
        vb_phys_addr_ = other.vb_phys_addr_;
        vb_virt_addr_ = other.vb_virt_addr_;
        vb_size_ = other.vb_size_;
        cached_width_ = other.cached_width_;
        cached_height_ = other.cached_height_;
        cached_channels_ = other.cached_channels_;
        jpeg_decoder_ = other.jpeg_decoder_;

        // Do NOT move VDEC resources
        vdec_decoder_ = nullptr;
        vdec_initialized_ = false;
        vdec_max_width_ = 0;
        vdec_max_height_ = 0;
        has_vdec_frame_ = false;
        std::memset(&current_vdec_frame_, 0, sizeof(current_vdec_frame_));

        // Clear source VB resources
        other.vb_block_ = VB_INVALID_HANDLE;
        other.vb_phys_addr_ = 0;
        other.vb_virt_addr_ = nullptr;
        other.vb_size_ = 0;
    }
    return *this;
}

void IonImageLoader::release() {
    // Release VDEC frame (if any)
    if (has_vdec_frame_ && vdec_decoder_) {
        vdec_decoder_->release_frame(current_vdec_frame_);
        has_vdec_frame_ = false;
        std::memset(&current_vdec_frame_, 0, sizeof(current_vdec_frame_));
    }

    // Release VDEC decoder
    if (vdec_decoder_) {
        delete vdec_decoder_;
        vdec_decoder_ = nullptr;
    }
    vdec_initialized_ = false;
    vdec_max_width_ = 0;
    vdec_max_height_ = 0;

    // Unmap and release VB buffer
    if (vb_virt_addr_ != nullptr) {
        CVI_SYS_Munmap(vb_virt_addr_, vb_size_);
        vb_virt_addr_ = nullptr;
    }
    if (vb_block_ != VB_INVALID_HANDLE) {
        CVI_VB_ReleaseBlock(vb_block_);
        vb_block_ = VB_INVALID_HANDLE;
        vb_phys_addr_ = 0;
        vb_size_ = 0;
    }
}

bool IonImageLoader::ensure_buffer(uint32_t width, uint32_t height, uint32_t channels) {
    // Calculate required buffer size with stride alignment
    uint32_t stride = ALIGN(width * channels, ALIGNMENT);
    uint32_t required_size = stride * height;

    // Check if existing buffer can be reused
    if (vb_virt_addr_ != nullptr && vb_size_ >= required_size) {
        // Buffer is large enough, can reuse
        cached_width_ = width;
        cached_height_ = height;
        cached_channels_ = channels;
        return true;
    }

    // Need to allocate new buffer
    release();  // Free existing buffer if any

    // Allocate VB block from public pool (VB_INVALID_POOLID = auto-select)
    VB_BLK vb_block = CVI_VB_GetBlock(VB_INVALID_POOLID, required_size);
    if (vb_block == VB_INVALID_HANDLE) {
        vb_block_ = VB_INVALID_HANDLE;
        vb_phys_addr_ = 0;
        vb_virt_addr_ = nullptr;
        vb_size_ = 0;
        return false;
    }

    // Get physical address
    CVI_U64 phys_addr = CVI_VB_Handle2PhysAddr(vb_block);
    if (phys_addr == 0) {
        CVI_VB_ReleaseBlock(vb_block);
        vb_block_ = VB_INVALID_HANDLE;
        vb_phys_addr_ = 0;
        vb_virt_addr_ = nullptr;
        vb_size_ = 0;
        return false;
    }

    // Map to cached virtual address for fast CPU writes
    void* virt_addr = CVI_SYS_MmapCache(phys_addr, required_size);
    if (virt_addr == nullptr) {
        CVI_VB_ReleaseBlock(vb_block);
        vb_block_ = VB_INVALID_HANDLE;
        vb_phys_addr_ = 0;
        vb_virt_addr_ = nullptr;
        vb_size_ = 0;
        return false;
    }

    vb_block_ = vb_block;
    vb_phys_addr_ = phys_addr;
    vb_virt_addr_ = virt_addr;
    vb_size_ = required_size;
    cached_width_ = width;
    cached_height_ = height;
    cached_channels_ = channels;

    return true;
}

VIDEO_FRAME_INFO_S IonImageLoader::create_video_frame(uint32_t width, uint32_t height,
                                                       PIXEL_FORMAT_E format, uint32_t stride) {
    VIDEO_FRAME_INFO_S frame;
    std::memset(&frame, 0, sizeof(frame));

    VIDEO_FRAME_S& vf = frame.stVFrame;
    vf.u32Width = width;
    vf.u32Height = height;
    vf.enPixelFormat = format;
    vf.enVideoFormat = VIDEO_FORMAT_LINEAR;
    vf.enCompressMode = COMPRESS_MODE_NONE;
    vf.enColorGamut = COLOR_GAMUT_BT709;

    // Physical address and stride
    vf.u64PhyAddr[0] = vb_phys_addr_;
    vf.u32Stride[0] = stride;
    vf.u32Length[0] = stride * height;

    // Virtual address (for debugging, VPSS uses physical)
    vf.pu8VirAddr[0] = static_cast<CVI_U8*>(vb_virt_addr_);

    return frame;
}

VIDEO_FRAME_INFO_S IonImageLoader::load(const std::string& filepath) {
    // Step 1: Read file into memory
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("IonImageLoader::load - cannot open file: " + filepath);
    }

    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> file_data(file_size);
    if (!file.read(reinterpret_cast<char*>(file_data.data()), file_size)) {
        throw std::runtime_error("IonImageLoader::load - failed to read file: " + filepath);
    }
    file.close();

    return load_from_memory(file_data.data(), file_data.size());
}

VIDEO_FRAME_INFO_S IonImageLoader::load_from_memory(const uint8_t* data, size_t size) {
    // Check if VDEC should be used for JPEG
    if (jpeg_decoder_ == JPEG_DECODER_VDEC && is_jpeg_format(data, size)) {
        try {
            return decode_with_vdec(data, size);
        } catch (const std::exception& e) {
            std::cerr << "[WARN] VDEC decode failed, falling back to OpenCV: "
                      << e.what() << std::endl;
            // Fall through to OpenCV
        }
    }

    // Default: OpenCV software decoding
    return decode_with_opencv(data, size);
}

bool IonImageLoader::preallocate(uint32_t width, uint32_t height, uint32_t channels) {
    return ensure_buffer(width, height, channels);
}

VIDEO_FRAME_INFO_S IonImageLoader::load_from_memory_fast(const uint8_t* data, size_t size) {
    // Check if buffer is pre-allocated
    if (vb_virt_addr_ == nullptr || cached_width_ == 0 || cached_height_ == 0) {
        // Fallback to standard path
        return load_from_memory(data, size);
    }

    // Calculate stride (64-byte aligned)
    uint32_t stride = ALIGN(cached_width_ * cached_channels_, ALIGNMENT);

    // Create Mat backed by VB buffer
    // Important: step parameter must match stride for proper memory layout
    cv::Mat vb_mat(cached_height_, cached_width_,
                    (cached_channels_ == 3) ? CV_8UC3 : CV_8UC1,
                    vb_virt_addr_, stride);

    // Decode directly to VB-backed Mat
    // The imdecode function will use our pre-allocated buffer if size matches
    cv::Mat decoded = cv::imdecode(
        cv::_InputArray(data, static_cast<int>(size)),
        cv::IMREAD_COLOR,
        &vb_mat);

    // Check if decode succeeded and used our buffer
    if (decoded.empty()) {
        throw std::runtime_error("IonImageLoader::load_from_memory_fast - failed to decode image");
    }

    // Check if imdecode used our VB buffer or allocated new memory
    if (decoded.data != vb_virt_addr_) {
        // imdecode allocated new memory (dimensions mismatch)
        // Copy to VB buffer with stride alignment
        uint32_t row_bytes = decoded.cols * decoded.channels();
        uint8_t* dst_ptr = static_cast<uint8_t*>(vb_virt_addr_);
        const uint8_t* src_ptr = decoded.data;

        // Update cached dimensions
        cached_width_ = decoded.cols;
        cached_height_ = decoded.rows;
        cached_channels_ = decoded.channels();

        // Check if buffer is large enough
        uint32_t new_stride = ALIGN(cached_width_ * cached_channels_, ALIGNMENT);
        uint32_t required_size = new_stride * cached_height_;
        if (required_size > vb_size_) {
            // Buffer too small, need to reallocate
            return load_from_memory(data, size);
        }

        stride = new_stride;

        if (stride == row_bytes && decoded.isContinuous()) {
            std::memcpy(dst_ptr, src_ptr, row_bytes * cached_height_);
        } else {
            for (uint32_t row = 0; row < cached_height_; ++row) {
                std::memcpy(dst_ptr + row * stride,
                           src_ptr + row * decoded.step[0],
                           row_bytes);
            }
        }
    }

    // Flush CPU cache to DRAM for hardware access
    CVI_SYS_IonFlushCache(vb_phys_addr_, vb_virt_addr_, vb_size_);

    // Create VIDEO_FRAME_INFO_S with physical address
    PIXEL_FORMAT_E format = (cached_channels_ == 3) ? PIXEL_FORMAT_BGR_888 : PIXEL_FORMAT_YUV_400;
    return create_video_frame(cached_width_, cached_height_, format, stride);
}

// ========== New helper functions for VDEC integration ==========

bool IonImageLoader::is_jpeg_format(const uint8_t* data, size_t size) {
    // JPEG magic number: FF D8 FF
    return (size >= 3 && data[0] == 0xFF && data[1] == 0xD8 && data[2] == 0xFF);
}

VIDEO_FRAME_INFO_S IonImageLoader::decode_with_opencv(const uint8_t* data, size_t size) {
    // Decode to temporary Mat to get dimensions
    cv::Mat temp_mat = cv::imdecode(
        cv::_InputArray(data, static_cast<int>(size)),
        cv::IMREAD_COLOR);

    if (temp_mat.empty()) {
        throw std::runtime_error("IonImageLoader::decode_with_opencv - failed to decode image");
    }

    uint32_t width = temp_mat.cols;
    uint32_t height = temp_mat.rows;
    uint32_t channels = temp_mat.channels();

    // Ensure VB buffer is allocated
    if (!ensure_buffer(width, height, channels)) {
        throw std::runtime_error("IonImageLoader::decode_with_opencv - failed to allocate VB buffer");
    }

    // Calculate stride (64-byte aligned)
    uint32_t stride = ALIGN(width * channels, ALIGNMENT);
    uint32_t row_bytes = width * channels;

    // Copy decoded data to VB buffer with stride
    uint8_t* dst_ptr = static_cast<uint8_t*>(vb_virt_addr_);
    const uint8_t* src_ptr = temp_mat.data;

    if (stride == row_bytes && temp_mat.isContinuous()) {
        std::memcpy(dst_ptr, src_ptr, row_bytes * height);
    } else {
        for (uint32_t row = 0; row < height; ++row) {
            std::memcpy(dst_ptr + row * stride,
                       src_ptr + row * temp_mat.step[0],
                       row_bytes);
        }
    }

    // Flush CPU cache to DRAM for hardware access
    CVI_SYS_IonFlushCache(vb_phys_addr_, vb_virt_addr_, vb_size_);

    // Create VIDEO_FRAME_INFO_S with physical address
    PIXEL_FORMAT_E format = (channels == 3) ? PIXEL_FORMAT_BGR_888 : PIXEL_FORMAT_YUV_400;
    return create_video_frame(width, height, format, stride);
}

VIDEO_FRAME_INFO_S IonImageLoader::decode_with_vdec(const uint8_t* data, size_t size) {
    // Create VDEC decoder if not exists
    if (!vdec_decoder_) {
        vdec_decoder_ = new HwJpegDecoder();
    }

    // Parse JPEG header to get dimensions (fast, no full decode)
    int width = 0, height = 0;
    if (!parse_jpeg_dimensions(data, size, width, height)) {
        throw std::runtime_error("IonImageLoader::decode_with_vdec - failed to parse JPEG header");
    }

    // Initialize or reinitialize VDEC if needed
    if (!vdec_initialized_ || width > static_cast<int>(vdec_max_width_) || height > static_cast<int>(vdec_max_height_)) {
        // Release old frame before reinitializing
        if (has_vdec_frame_) {
            vdec_decoder_->release_frame(current_vdec_frame_);
            has_vdec_frame_ = false;
            std::memset(&current_vdec_frame_, 0, sizeof(current_vdec_frame_));
        }

        vdec_decoder_->cleanup();
        if (!vdec_decoder_->init(width, height)) {
            throw std::runtime_error("IonImageLoader::decode_with_vdec - failed to initialize VDEC decoder");
        }
        vdec_initialized_ = true;
        vdec_max_width_ = width;
        vdec_max_height_ = height;
    }

    // Decode with VDEC (using synchronous blocking mode for best performance)
    VIDEO_FRAME_INFO_S frame = vdec_decoder_->decode_sync(data, size);

    // Manage frame lifecycle: release previous frame, save new frame
    if (has_vdec_frame_) {
        vdec_decoder_->release_frame(current_vdec_frame_);
    }

    current_vdec_frame_ = frame;
    has_vdec_frame_ = true;

    return frame;
}

bool IonImageLoader::parse_jpeg_dimensions(const uint8_t* data, size_t size, int& width, int& height) {
    // JPEG format:
    // - Start of Image (SOI): FF D8
    // - Then various segments, each starting with FF <marker> <length_hi> <length_lo>
    // - SOF0 (Start of Frame, baseline DCT): FF C0
    //   - Length: 2 bytes
    //   - Precision: 1 byte (usually 8)
    //   - Height: 2 bytes (big-endian)
    //   - Width: 2 bytes (big-endian)

    if (size < 4) return false;

    // Check JPEG SOI
    if (data[0] != 0xFF || data[1] != 0xD8) return false;

    size_t pos = 2;
    while (pos < size - 1) {
        // Find marker
        if (data[pos] != 0xFF) {
            pos++;
            continue;
        }

        uint8_t marker = data[pos + 1];
        pos += 2;

        // Skip padding bytes (FF FF FF ...)
        if (marker == 0xFF) {
            continue;
        }

        // SOF0 (Start of Frame, baseline DCT) - contains dimensions
        // Note: SOF0 marker is 0xC0, but there are variants (0xC0, 0xC1, 0xC2, etc.)
        // We accept any SOF marker (0xC0-0xCF)
        if ((marker & 0xF0) == 0xC0 && marker != 0xC4 && marker != 0xC8 && marker != 0xCC) {
            // Found SOF marker, parse dimensions
            if (pos + 6 > size) return false;

            // Skip length (2 bytes) and precision (1 byte)
            // Height is at offset 3, Width at offset 5 (big-endian)
            height = (data[pos + 3] << 8) | data[pos + 4];
            width = (data[pos + 5] << 8) | data[pos + 6];

            return (width > 0 && height > 0);
        }

        // For other markers, skip the segment
        if (pos + 1 >= size) return false;

        // Get segment length (big-endian, includes the length bytes themselves)
        uint16_t segment_length = (data[pos] << 8) | data[pos + 1];
        if (segment_length < 2) return false;

        pos += segment_length;
    }

    return false;
}

} // namespace lua_cv

#endif // USE_CVI_MPI
