#include "cvi_camera.h"

#ifdef USE_CVI_MPI
#ifdef USE_CVI_CAMERA

#include <stdexcept>
#include <iostream>
#include <cstring>

namespace lua_cv {

// Camera uses VPSS_GRP 0 (different from CviVpssProcessor which uses GRP 1)
static constexpr VPSS_GRP CAMERA_VPSS_GRP = 0;
static constexpr VPSS_CHN CAMERA_VPSS_CHN = 0;

CviCamera::CviCamera(const Config& config)
    : config_(config)
    , opened_(false)
    , vpss_grp_(CAMERA_VPSS_GRP)
    , vpss_chn_(CAMERA_VPSS_CHN)
    , vb_pool_(VB_INVALID_POOLID) {
}

CviCamera::~CviCamera() {
    release();
}

bool CviCamera::open() {
    if (opened_) {
        std::cerr << "[WARNING] CviCamera::open() - Camera already opened" << std::endl;
        return true;
    }

    try {
        // Step 1: Initialize VB system
        CVI_S32 rc = CVI_VB_Init();
        if (rc != CVI_SUCCESS && rc != CVI_ERR_VB_NOTREADY) {
            throw std::runtime_error("CVI_VB_Init failed: " + std::to_string(rc));
        }

        // Step 2: Initialize system
        rc = CVI_SYS_Init();
        if (rc != CVI_SUCCESS) {
            throw std::runtime_error("CVI_SYS_Init failed: " + std::to_string(rc));
        }

        // Step 3: Initialize sensor with ISP (auto-detect)
        // CviSensor handles: VI_Open, sensor probe, MIPI, VI dev/pipe/chn, ISP, ISP thread
        if (!sensor_.init()) {
            throw std::runtime_error("CviSensor::init failed - no sensor detected");
        }

        std::cout << "[INFO] CviCamera: Detected sensor " << sensor_.get_sensor_name()
                  << " (" << sensor_.get_width() << "x" << sensor_.get_height() << ")" << std::endl;

        // Step 4: Initialize VB pool for VPSS
        init_vb_pool();

        // Step 5: Initialize VPSS
        init_vpss_module();

        // Step 6: Bind VI to VPSS
        bind_vi_vpss();

        opened_ = true;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] CviCamera::open() - " << e.what() << std::endl;
        cleanup();
        return false;
    }
}

bool CviCamera::read(Frame& frame) {
    if (!opened_) {
        std::cerr << "[ERROR] CviCamera::read() - Camera not opened" << std::endl;
        return false;
    }

    // Get frame from VPSS channel
    VIDEO_FRAME_INFO_S vpss_frame;
    CVI_S32 rc = CVI_VPSS_GetChnFrame(vpss_grp_, vpss_chn_, &vpss_frame, 1000);

    if (rc != CVI_SUCCESS) {
        std::cerr << "[ERROR] CviCamera::read() - CVI_VPSS_GetChnFrame failed: 0x"
                  << std::hex << rc << std::dec << std::endl;
        return false;
    }

    // Wrap in Frame object (owns_memory=false, caller must release)
    // Note: Caller should call frame.release() or CVI_VPSS_ReleaseChnFrame()
    frame = Frame(vpss_frame, false);

    return true;
}

void CviCamera::release() {
    if (!opened_) {
        return;
    }

    cleanup();
    opened_ = false;
}

// ========== Private Implementation ==========

void CviCamera::init_vb_pool() {
    // Calculate VB pool size based on sensor resolution
    // Use stride-aligned size for safety
    uint32_t width = sensor_.get_width();
    uint32_t height = sensor_.get_height();

    // NV21 format: Y plane (width*height) + UV plane (width*height/2)
    // Add alignment padding (align to 64 bytes)
    uint32_t y_size = ((width + 63) & ~63) * height;
    uint32_t uv_size = ((width + 63) & ~63) * height / 2;
    uint32_t blk_size = y_size + uv_size;

    // Round up to 1MB boundary for safety
    blk_size = (blk_size + 0xFFFFF) & ~0xFFFFF;

    uint32_t blk_cnt = 3;  // 3 buffers for triple buffering

    VB_POOL_CONFIG_S pool_cfg;
    std::memset(&pool_cfg, 0, sizeof(pool_cfg));
    pool_cfg.u32BlkSize = blk_size;
    pool_cfg.u32BlkCnt = blk_cnt;
    pool_cfg.enRemapMode = VB_REMAP_MODE_NONE;

    vb_pool_ = CVI_VB_CreatePool(&pool_cfg);
    if (vb_pool_ == VB_INVALID_POOLID) {
        throw std::runtime_error("CVI_VB_CreatePool failed");
    }

    std::cout << "[INFO] CviCamera: Created VB pool (blk_size=" << blk_size
              << ", blk_cnt=" << blk_cnt << ")" << std::endl;
}

void CviCamera::init_vpss_module() {
    CVI_S32 rc;
    uint32_t width = sensor_.get_width();
    uint32_t height = sensor_.get_height();

    // 1. Create VPSS group
    VPSS_GRP_ATTR_S grp_attr;
    std::memset(&grp_attr, 0, sizeof(grp_attr));
    grp_attr.u32MaxW = width;
    grp_attr.u32MaxH = height;
    grp_attr.enPixelFormat = PIXEL_FORMAT_NV21;  // Input from VI is NV21
    grp_attr.stFrameRate.s32SrcFrameRate = -1;
    grp_attr.stFrameRate.s32DstFrameRate = -1;
    grp_attr.u8VpssDev = 0;

    rc = CVI_VPSS_CreateGrp(vpss_grp_, &grp_attr);
    if (rc != CVI_SUCCESS) {
        throw std::runtime_error("CVI_VPSS_CreateGrp failed: 0x" + std::to_string(rc));
    }

    // 2. Reset VPSS group
    rc = CVI_VPSS_ResetGrp(vpss_grp_);
    if (rc != CVI_SUCCESS) {
        throw std::runtime_error("CVI_VPSS_ResetGrp failed: 0x" + std::to_string(rc));
    }

    // 3. Set VPSS channel attributes
    VPSS_CHN_ATTR_S chn_attr;
    std::memset(&chn_attr, 0, sizeof(chn_attr));
    chn_attr.u32Width = width;
    chn_attr.u32Height = height;
    chn_attr.enVideoFormat = VIDEO_FORMAT_LINEAR;
    chn_attr.enPixelFormat = to_cvi_pixel_format(config_.format);
    chn_attr.stFrameRate.s32SrcFrameRate = -1;
    chn_attr.stFrameRate.s32DstFrameRate = -1;
    chn_attr.u32Depth = 1;
    chn_attr.bMirror = CVI_FALSE;
    chn_attr.bFlip = CVI_FALSE;
    chn_attr.stAspectRatio.enMode = ASPECT_RATIO_NONE;
    chn_attr.stNormalize.bEnable = CVI_FALSE;

    rc = CVI_VPSS_SetChnAttr(vpss_grp_, vpss_chn_, &chn_attr);
    if (rc != CVI_SUCCESS) {
        throw std::runtime_error("CVI_VPSS_SetChnAttr failed: 0x" + std::to_string(rc));
    }

    // 4. Attach VB pool to VPSS
    rc = CVI_VPSS_AttachVbPool(vpss_grp_, vpss_chn_, vb_pool_);
    if (rc != CVI_SUCCESS) {
        throw std::runtime_error("CVI_VPSS_AttachVbPool failed: 0x" + std::to_string(rc));
    }

    // 5. Enable VPSS channel
    rc = CVI_VPSS_EnableChn(vpss_grp_, vpss_chn_);
    if (rc != CVI_SUCCESS) {
        throw std::runtime_error("CVI_VPSS_EnableChn failed: 0x" + std::to_string(rc));
    }

    // 6. Start VPSS group
    rc = CVI_VPSS_StartGrp(vpss_grp_);
    if (rc != CVI_SUCCESS) {
        throw std::runtime_error("CVI_VPSS_StartGrp failed: 0x" + std::to_string(rc));
    }

    std::cout << "[INFO] CviCamera: VPSS initialized (GRP=" << vpss_grp_
              << ", CHN=" << vpss_chn_ << ")" << std::endl;
}

void CviCamera::bind_vi_vpss() {
    // Bind VI channel (from CviSensor) to VPSS group (zero-copy data path)
    MMF_CHN_S src_chn;
    src_chn.enModId = CVI_ID_VI;
    src_chn.s32DevId = sensor_.get_vi_pipe();
    src_chn.s32ChnId = sensor_.get_vi_chn();

    MMF_CHN_S dst_chn;
    dst_chn.enModId = CVI_ID_VPSS;
    dst_chn.s32DevId = vpss_grp_;
    dst_chn.s32ChnId = 0;  // VPSS group input

    CVI_S32 rc = CVI_SYS_Bind(&src_chn, &dst_chn);
    if (rc != CVI_SUCCESS) {
        throw std::runtime_error("CVI_SYS_Bind (VI->VPSS) failed: 0x" + std::to_string(rc));
    }

    std::cout << "[INFO] CviCamera: Bound VI(" << src_chn.s32DevId << "," << src_chn.s32ChnId
              << ") -> VPSS(" << dst_chn.s32DevId << ")" << std::endl;
}

void CviCamera::cleanup() {
    // Step 1: Unbind VI from VPSS
    if (sensor_.is_initialized()) {
        MMF_CHN_S src_chn;
        src_chn.enModId = CVI_ID_VI;
        src_chn.s32DevId = sensor_.get_vi_pipe();
        src_chn.s32ChnId = sensor_.get_vi_chn();

        MMF_CHN_S dst_chn;
        dst_chn.enModId = CVI_ID_VPSS;
        dst_chn.s32DevId = vpss_grp_;
        dst_chn.s32ChnId = 0;

        CVI_SYS_UnBind(&src_chn, &dst_chn);
    }

    // Step 2: Stop and destroy VPSS
    CVI_VPSS_StopGrp(vpss_grp_);
    CVI_VPSS_DisableChn(vpss_grp_, vpss_chn_);
    CVI_VPSS_DetachVbPool(vpss_grp_, vpss_chn_);
    CVI_VPSS_DestroyGrp(vpss_grp_);

    // Step 3: Cleanup sensor (handles VI/ISP/MIPI)
    sensor_.cleanup();

    // Step 4: Destroy VB pool
    if (vb_pool_ != VB_INVALID_POOLID) {
        CVI_VB_DestroyPool(vb_pool_);
        vb_pool_ = VB_INVALID_POOLID;
    }

    // Step 5: Exit system
    CVI_SYS_Exit();
    CVI_VB_Exit();
}

PIXEL_FORMAT_E CviCamera::to_cvi_pixel_format(PixelFormat format) const {
    switch (format) {
        case PixelFormat::RGB888:
            return PIXEL_FORMAT_RGB_888;
        case PixelFormat::BGR888:
            return PIXEL_FORMAT_BGR_888;
        case PixelFormat::NV12:
            return PIXEL_FORMAT_NV12;
        case PixelFormat::NV21:
            return PIXEL_FORMAT_NV21;
        case PixelFormat::GRAY:
            return PIXEL_FORMAT_YUV_400;
        default:
            return PIXEL_FORMAT_NV21;  // Default
    }
}

} // namespace lua_cv

#endif // USE_CVI_CAMERA
#endif // USE_CVI_MPI
