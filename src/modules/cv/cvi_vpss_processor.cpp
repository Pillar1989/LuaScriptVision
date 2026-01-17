#include "cvi_vpss_processor.h"

#ifdef USE_CVI_MPI

#include <stdexcept>
#include <chrono>
#include <iostream>
#include <cvi_sys.h>
#include <cvi_vb.h>

namespace lua_cv {

CviVpssProcessor::CviVpssProcessor()
    : vpss_grp_(1)  // Use GRP=1 (GRP=0 reserved for camera pipeline)
    , vpss_chn_(0)
    , vb_pool_(VB_INVALID_POOLID)
    , initialized_(false)
    , cached_input_w_(0)
    , cached_input_h_(0)
    , cached_output_w_(0)
    , cached_output_h_(0)
    , cached_input_fmt_(PIXEL_FORMAT_MAX)
    , cached_output_fmt_(PIXEL_FORMAT_MAX) {
}

CviVpssProcessor::~CviVpssProcessor() {
    cleanup_vpss_pipeline();
}

void CviVpssProcessor::resize(Frame& frame, int width, int height) {
    if (frame.empty()) {
        throw std::invalid_argument("CviVpssProcessor::resize() - frame is empty");
    }

    if (width <= 0 || height <= 0) {
        throw std::invalid_argument("CviVpssProcessor::resize() - invalid dimensions");
    }

    VIDEO_FRAME_INFO_S input;
    VB_BLK temp_vb_block = VB_INVALID_HANDLE;
    bool needs_vb_release = false;

    // Handle both zero-copy (VIDEO_FRAME) and file input (cv::Mat) paths
    if (frame.has_physical_addr()) {
        // Path A: Zero-copy (from Camera or previous VPSS)
        input = frame.to_video_frame();
    } else {
        // Path B: File input - convert Mat to VIDEO_FRAME_INFO_S
        input = mat_to_video_frame(frame.to_mat(), temp_vb_block);
        needs_vb_release = true;
    }

    PIXEL_FORMAT_E input_format = input.stVFrame.enPixelFormat;
    uint32_t input_width = input.stVFrame.u32Width;
    uint32_t input_height = input.stVFrame.u32Height;

    // Initialize/reconfigure VPSS pipeline if needed
    CVI_S32 rc = init_vpss_pipeline(
        input_width, input_height, input_format,
        width, height, input_format);  // Keep same format

    if (rc != CVI_SUCCESS) {
        if (needs_vb_release) {
            CVI_VB_ReleaseBlock(temp_vb_block);
        }
        throw std::runtime_error("CviVpssProcessor::resize() - Failed to initialize VPSS pipeline");
    }

    // Process frame through VPSS
    VIDEO_FRAME_INFO_S output;
    rc = vpss_process_frame(input, output);

    // Release temporary VB block (VPSS processing complete)
    if (needs_vb_release) {
        CVI_VB_ReleaseBlock(temp_vb_block);
    }

    if (rc != CVI_SUCCESS) {
        throw std::runtime_error("CviVpssProcessor::resize() - VPSS processing failed");
    }

    // Replace frame with VPSS output
    // Use new constructor with VPSS context for proper memory management
    frame = Frame(output, vpss_grp_, vpss_chn_);
}

void CviVpssProcessor::cvtColor(Frame& frame, ColorConversion code) {
    if (frame.empty()) {
        throw std::invalid_argument("CviVpssProcessor::cvtColor() - frame is empty");
    }

    VIDEO_FRAME_INFO_S input;
    VB_BLK temp_vb_block = VB_INVALID_HANDLE;
    bool needs_vb_release = false;

    // Handle both zero-copy and file input paths
    if (frame.has_physical_addr()) {
        input = frame.to_video_frame();
    } else {
        input = mat_to_video_frame(frame.to_mat(), temp_vb_block);
        needs_vb_release = true;
    }

    PIXEL_FORMAT_E input_fmt = input.stVFrame.enPixelFormat;
    uint32_t width = input.stVFrame.u32Width;
    uint32_t height = input.stVFrame.u32Height;

    // Map ColorConversion to PIXEL_FORMAT_E
    PIXEL_FORMAT_E output_fmt;
    switch (code) {
        case ColorConversion::BGR2RGB:  // BGR2RGB and RGB2BGR have same value (4)
            // RGB/BGR swap - VPSS doesn't directly support, use same format
            // (actual conversion happens at pixel level, VPSS may not distinguish)
            output_fmt = (input_fmt == PIXEL_FORMAT_RGB_888 || input_fmt == PIXEL_FORMAT_BGR_888)
                         ? PIXEL_FORMAT_RGB_888 : input_fmt;
            break;

        case ColorConversion::BGR2NV12:
        case ColorConversion::RGB2NV12:
            output_fmt = PIXEL_FORMAT_YUV_PLANAR_420;  // NV12
            break;

        case ColorConversion::NV12_BGR:
        case ColorConversion::NV12_RGB:
            output_fmt = PIXEL_FORMAT_RGB_888;
            break;

        case ColorConversion::BGR2GRAY:
        case ColorConversion::RGB2GRAY:
            output_fmt = PIXEL_FORMAT_YUV_400;  // Grayscale
            break;

        default:
            if (needs_vb_release) {
                CVI_VB_ReleaseBlock(temp_vb_block);
            }
            throw std::runtime_error(
                "CviVpssProcessor::cvtColor() - unsupported color conversion code");
    }

    // If no format change needed, return early
    if (input_fmt == output_fmt && code != ColorConversion::BGR2RGB && code != ColorConversion::RGB2BGR) {
        if (needs_vb_release) {
            CVI_VB_ReleaseBlock(temp_vb_block);
        }
        return;
    }

    // Initialize VPSS pipeline with format conversion
    CVI_S32 rc = init_vpss_pipeline(
        width, height, input_fmt,
        width, height, output_fmt);  // Same size, different format

    if (rc != CVI_SUCCESS) {
        if (needs_vb_release) {
            CVI_VB_ReleaseBlock(temp_vb_block);
        }
        throw std::runtime_error("CviVpssProcessor::cvtColor() - Failed to initialize VPSS pipeline");
    }

    // Process frame through VPSS
    VIDEO_FRAME_INFO_S output;
    rc = vpss_process_frame(input, output);

    // Release temporary VB block
    if (needs_vb_release) {
        CVI_VB_ReleaseBlock(temp_vb_block);
    }

    if (rc != CVI_SUCCESS) {
        throw std::runtime_error("CviVpssProcessor::cvtColor() - VPSS processing failed");
    }

    // Replace frame with VPSS output (with VPSS context for proper cleanup)
    frame = Frame(output, vpss_grp_, vpss_chn_);
}

void CviVpssProcessor::crop(Frame& frame, int x, int y, int w, int h) {
    if (frame.empty()) {
        throw std::invalid_argument("CviVpssProcessor::crop() - frame is empty");
    }

    // Validate crop parameters
    if (w <= 0 || h <= 0) {
        throw std::invalid_argument("CviVpssProcessor::crop() - invalid crop dimensions");
    }

    if (x < 0 || y < 0) {
        throw std::invalid_argument("CviVpssProcessor::crop() - invalid crop coordinates");
    }

    VIDEO_FRAME_INFO_S input;
    VB_BLK temp_vb_block = VB_INVALID_HANDLE;
    bool needs_vb_release = false;

    // Handle both zero-copy and file input paths
    if (frame.has_physical_addr()) {
        input = frame.to_video_frame();
    } else {
        input = mat_to_video_frame(frame.to_mat(), temp_vb_block);
        needs_vb_release = true;
    }

    int frame_width = static_cast<int>(input.stVFrame.u32Width);
    int frame_height = static_cast<int>(input.stVFrame.u32Height);

    if (x + w > frame_width || y + h > frame_height) {
        if (needs_vb_release) {
            CVI_VB_ReleaseBlock(temp_vb_block);
        }
        throw std::invalid_argument(
            "CviVpssProcessor::crop() - crop region exceeds frame bounds");
    }

    PIXEL_FORMAT_E input_fmt = input.stVFrame.enPixelFormat;

    // Initialize VPSS pipeline for crop + resize to output size
    CVI_S32 rc = init_vpss_pipeline(
        frame_width, frame_height, input_fmt,
        w, h, input_fmt);  // Output is crop size

    if (rc != CVI_SUCCESS) {
        if (needs_vb_release) {
            CVI_VB_ReleaseBlock(temp_vb_block);
        }
        throw std::runtime_error("CviVpssProcessor::crop() - Failed to initialize VPSS pipeline");
    }

    // Configure crop region on VPSS channel
    VPSS_CROP_INFO_S crop_info;
    crop_info.bEnable = CVI_TRUE;
    crop_info.enCropCoordinate = VPSS_CROP_ABS_COOR;  // Absolute coordinates
    crop_info.stCropRect.s32X = x;
    crop_info.stCropRect.s32Y = y;
    crop_info.stCropRect.u32Width = w;
    crop_info.stCropRect.u32Height = h;

    rc = CVI_VPSS_SetChnCrop(vpss_grp_, vpss_chn_, &crop_info);
    if (rc != CVI_SUCCESS) {
        if (needs_vb_release) {
            CVI_VB_ReleaseBlock(temp_vb_block);
        }
        std::cerr << "[ERROR] CVI_VPSS_SetChnCrop failed: " << rc << std::endl;
        throw std::runtime_error("CviVpssProcessor::crop() - Failed to set crop region");
    }

    // Process frame through VPSS
    VIDEO_FRAME_INFO_S output;
    rc = vpss_process_frame(input, output);

    // Release temporary VB block
    if (needs_vb_release) {
        CVI_VB_ReleaseBlock(temp_vb_block);
    }

    // Disable crop for next operation
    crop_info.bEnable = CVI_FALSE;
    CVI_VPSS_SetChnCrop(vpss_grp_, vpss_chn_, &crop_info);

    if (rc != CVI_SUCCESS) {
        throw std::runtime_error("CviVpssProcessor::crop() - VPSS processing failed");
    }

    // Replace frame with cropped output (with VPSS context for proper cleanup)
    frame = Frame(output, vpss_grp_, vpss_chn_);
}

// ========== Private Implementation ==========

CVI_S32 CviVpssProcessor::init_vpss_pipeline(
    uint32_t input_w, uint32_t input_h, PIXEL_FORMAT_E input_fmt,
    uint32_t output_w, uint32_t output_h, PIXEL_FORMAT_E output_fmt) {

    // Check if we can reuse existing pipeline
    if (initialized_ &&
        cached_input_w_ == input_w &&
        cached_input_h_ == input_h &&
        cached_output_w_ == output_w &&
        cached_output_h_ == output_h &&
        cached_input_fmt_ == input_fmt &&
        cached_output_fmt_ == output_fmt) {
        // Pipeline already configured correctly
        return CVI_SUCCESS;
    }

    // Need to reconfigure pipeline - cleanup first
    cleanup_vpss_pipeline();

    CVI_S32 rc;

    // Step 0: Ensure VPSS group doesn't exist (force cleanup from previous crash)
    CVI_VPSS_StopGrp(vpss_grp_);
    CVI_VPSS_DisableChn(vpss_grp_, vpss_chn_);
    CVI_VPSS_DestroyGrp(vpss_grp_);

    // Step 1: Create VPSS group
    VPSS_GRP_ATTR_S grp_attr = {0};
    grp_attr.u32MaxW = input_w;
    grp_attr.u32MaxH = input_h;
    grp_attr.enPixelFormat = input_fmt;
    grp_attr.stFrameRate.s32SrcFrameRate = -1;  // No frame rate control
    grp_attr.stFrameRate.s32DstFrameRate = -1;

    rc = CVI_VPSS_CreateGrp(vpss_grp_, &grp_attr);
    if (rc != CVI_SUCCESS) {
        std::cerr << "[ERROR] CVI_VPSS_CreateGrp failed: " << rc << std::endl;
        return rc;
    }

    // Step 2: Reset VPSS group
    rc = CVI_VPSS_ResetGrp(vpss_grp_);
    if (rc != CVI_SUCCESS) {
        std::cerr << "[ERROR] CVI_VPSS_ResetGrp failed: " << rc << std::endl;
        CVI_VPSS_DestroyGrp(vpss_grp_);
        return rc;
    }

    // Step 3: Configure VPSS channel
    VPSS_CHN_ATTR_S chn_attr = {0};
    chn_attr.u32Width = output_w;
    chn_attr.u32Height = output_h;
    chn_attr.enVideoFormat = VIDEO_FORMAT_LINEAR;
    chn_attr.enPixelFormat = output_fmt;
    chn_attr.stFrameRate.s32SrcFrameRate = -1;
    chn_attr.stFrameRate.s32DstFrameRate = -1;
    chn_attr.bFlip = CVI_FALSE;
    chn_attr.bMirror = CVI_FALSE;
    chn_attr.u32Depth = 1;  // Output queue depth
    chn_attr.stAspectRatio.enMode = ASPECT_RATIO_NONE;
    chn_attr.stNormalize.bEnable = CVI_FALSE;

    rc = CVI_VPSS_SetChnAttr(vpss_grp_, vpss_chn_, &chn_attr);
    if (rc != CVI_SUCCESS) {
        std::cerr << "[ERROR] CVI_VPSS_SetChnAttr failed: " << rc << std::endl;
        CVI_VPSS_DestroyGrp(vpss_grp_);
        return rc;
    }

    // Step 4: Enable VPSS channel
    rc = CVI_VPSS_EnableChn(vpss_grp_, vpss_chn_);
    if (rc != CVI_SUCCESS) {
        std::cerr << "[ERROR] CVI_VPSS_EnableChn failed: " << rc << std::endl;
        CVI_VPSS_DestroyGrp(vpss_grp_);
        return rc;
    }

    // Note: We do NOT create a dynamic VB pool here.
    // VPSS will use the pre-configured public pools (set up by init_cvi_system).
    // The public pools must cover all possible output sizes using COMMON_GetPicBufferSize.
    // This avoids dynamic pool creation issues (pool count limits, cleanup on crash).

    // Step 5: Start VPSS group
    rc = CVI_VPSS_StartGrp(vpss_grp_);
    if (rc != CVI_SUCCESS) {
        std::cerr << "[ERROR] CVI_VPSS_StartGrp failed: " << rc << std::endl;
        CVI_VPSS_DisableChn(vpss_grp_, vpss_chn_);
        CVI_VPSS_DestroyGrp(vpss_grp_);
        return rc;
    }

    // Update cache
    cached_input_w_ = input_w;
    cached_input_h_ = input_h;
    cached_output_w_ = output_w;
    cached_output_h_ = output_h;
    cached_input_fmt_ = input_fmt;
    cached_output_fmt_ = output_fmt;
    initialized_ = true;

    return CVI_SUCCESS;
}

void CviVpssProcessor::cleanup_vpss_pipeline() {
    if (!initialized_) {
        return;
    }

    // Stop VPSS group
    CVI_VPSS_StopGrp(vpss_grp_);

    // Disable channel
    CVI_VPSS_DisableChn(vpss_grp_, vpss_chn_);

    // Destroy group
    CVI_VPSS_DestroyGrp(vpss_grp_);

    initialized_ = false;
}

CVI_S32 CviVpssProcessor::vpss_process_frame(
    const VIDEO_FRAME_INFO_S& input,
    VIDEO_FRAME_INFO_S& output) {

    // Send frame to VPSS group
    CVI_S32 rc = CVI_VPSS_SendFrame(vpss_grp_, &input, 1000);  // 1 second timeout
    if (rc != CVI_SUCCESS) {
        std::cerr << "[ERROR] CVI_VPSS_SendFrame failed: " << rc << std::endl;
        return rc;
    }

    // Get processed frame from VPSS channel
    rc = CVI_VPSS_GetChnFrame(vpss_grp_, vpss_chn_, &output, 1000);  // 1 second timeout
    if (rc != CVI_SUCCESS) {
        std::cerr << "[ERROR] CVI_VPSS_GetChnFrame failed: " << rc << std::endl;
        return rc;
    }

    return CVI_SUCCESS;
}

VIDEO_FRAME_INFO_S CviVpssProcessor::mat_to_video_frame(const cv::Mat& mat, VB_BLK& out_vb_block, VB_POOL vb_pool) {
    if (mat.empty()) {
        throw std::runtime_error("mat_to_video_frame: input Mat is empty");
    }

    if (!mat.isContinuous()) {
        throw std::runtime_error("mat_to_video_frame: input Mat must be continuous");
    }

    // Alignment requirement for DMA
    constexpr uint32_t ALIGNMENT = 64;
    auto ALIGN = [](uint32_t x, uint32_t a) -> uint32_t { return (x + a - 1) & ~(a - 1); };

    uint32_t width = mat.cols;
    uint32_t height = mat.rows;
    int cv_type = mat.type();

    // Determine pixel format and bytes per pixel
    PIXEL_FORMAT_E pixel_format;
    uint32_t bpp;  // bytes per pixel

    switch (cv_type) {
        case CV_8UC3:
            pixel_format = PIXEL_FORMAT_BGR_888;  // OpenCV default
            bpp = 3;
            break;
        case CV_8UC1:
            pixel_format = PIXEL_FORMAT_YUV_400;  // Grayscale
            bpp = 1;
            break;
        default:
            throw std::runtime_error("mat_to_video_frame: unsupported Mat type (only CV_8UC3 and CV_8UC1)");
    }

    // Calculate stride (must be 64-byte aligned)
    uint32_t stride = ALIGN(width * bpp, ALIGNMENT);

    // Calculate total memory size
    uint32_t plane_size = stride * height;
    uint32_t total_size = plane_size;  // Single plane for RGB/BGR/GRAY

    // Allocate VB block from specified pool
    VB_BLK vb_block = CVI_VB_GetBlock(vb_pool, total_size);
    if (vb_block == VB_INVALID_HANDLE) {
        throw std::runtime_error("mat_to_video_frame: CVI_VB_GetBlock failed (size=" +
                                std::to_string(total_size) + ")");
    }

    // Get physical address
    CVI_U64 phys_addr = CVI_VB_Handle2PhysAddr(vb_block);
    if (phys_addr == 0) {
        CVI_VB_ReleaseBlock(vb_block);
        throw std::runtime_error("mat_to_video_frame: CVI_VB_Handle2PhysAddr failed");
    }

    // Map to cached virtual address for fast CPU writes
    // VB pools are configured with VB_REMAP_MODE_CACHED
    void* virt_addr = CVI_SYS_MmapCache(phys_addr, total_size);
    if (virt_addr == nullptr) {
        CVI_VB_ReleaseBlock(vb_block);
        throw std::runtime_error("mat_to_video_frame: CVI_SYS_MmapCache failed");
    }

    // Copy Mat data to VB memory (with stride)
    // Using cached mapping: CPU writes go to cache first (fast),
    // then flush cache to physical memory for DMA access
    try {
        const uint8_t* src_ptr = mat.data;
        uint8_t* dst_ptr = static_cast<uint8_t*>(virt_addr);
        uint32_t row_bytes = width * bpp;

        if (stride == row_bytes) {
            // Optimized: single large memcpy when no padding
            std::memcpy(dst_ptr, src_ptr, row_bytes * height);
        } else {
            // Copy row by row when stride alignment needed
            for (uint32_t row = 0; row < height; ++row) {
                std::memcpy(dst_ptr + row * stride,
                           src_ptr + row * row_bytes,
                           row_bytes);
            }
        }

        // Flush CPU cache to physical memory before VPSS DMA reads
        CVI_SYS_IonFlushCache(phys_addr, virt_addr, total_size);

    } catch (...) {
        CVI_SYS_Munmap(virt_addr, total_size);
        CVI_VB_ReleaseBlock(vb_block);
        throw;
    }

    // Unmap virtual address (VPSS will access via physical address)
    CVI_SYS_Munmap(virt_addr, total_size);

    // Construct VIDEO_FRAME_INFO_S
    VIDEO_FRAME_INFO_S frame_info;
    std::memset(&frame_info, 0, sizeof(frame_info));

    VIDEO_FRAME_S& vf = frame_info.stVFrame;
    vf.u32Width = width;
    vf.u32Height = height;
    vf.enPixelFormat = pixel_format;
    vf.enVideoFormat = VIDEO_FORMAT_LINEAR;
    vf.enCompressMode = COMPRESS_MODE_NONE;
    vf.enColorGamut = COLOR_GAMUT_BT709;

    // Physical address and stride
    vf.u64PhyAddr[0] = phys_addr;
    vf.u32Stride[0] = stride;
    vf.u32Length[0] = plane_size;

    // Virtual address (0 for VPSS usage - it will map internally if needed)
    vf.pu8VirAddr[0] = nullptr;

    // Time stamp (not used for static images)
    vf.u64PTS = 0;

    // Return VB block handle for later release
    out_vb_block = vb_block;

    return frame_info;
}

} // namespace lua_cv

#endif // USE_CVI_MPI
