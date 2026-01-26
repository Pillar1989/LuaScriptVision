/**
 * test_vi_vpss_link.cpp - VI vs VPSS A/B capture test
 *
 * Purpose:
 *   Determine whether frames reach VI but not VPSS by checking VI status
 *   and VPSS output on the same pipeline without calling CVI_VI_GetChnFrame.
 */

#include "test_common.h"

#ifdef USE_CVI_MPI
#ifdef USE_CVI_CAMERA

#include <fstream>
#include <sstream>
#include <unistd.h>

#include <cvi_sys.h>
#include <cvi_vi.h>
#include <cvi_vpss.h>

#include "cv/cvi_camera.h"
#include "cv/cvi_vpss_processor.h"
#include "cv/mmf_context.h"

using namespace lua_cv;

namespace {
constexpr int kVpssTimeoutMs = 200;

void dump_runtime_status(const char* tag) {
    std::cerr << "\n[DIAG] Runtime status: " << (tag ? tag : "") << std::endl;
    std::system("cat /proc/cvitek/sys | grep -A 10 'BIND RELATION'");
    std::system("cat /proc/cvitek/vi | grep -A 6 'VI CHN STATUS'");
    std::system("cat /proc/cvitek/vpss | grep -A 24 'WORK STATUS'");
    std::system("cat /proc/cvitek/vpss | grep -A 12 'VPSS CHN OUTPUT RESOLUTION'");
    std::system("cat /proc/cvitek/vb | grep -A 12 'PoolId'");
}

void dump_sys_modes(const char* tag) {
    std::cerr << "[DIAG] SYS modes: " << (tag ? tag : "") << std::endl;

    VI_VPSS_MODE_S vi_vpss{};
    CVI_S32 rc = CVI_SYS_GetVIVPSSMode(&vi_vpss);
    if (rc == CVI_SUCCESS) {
        std::cerr << "  VI_VPSS mode:";
        for (int i = 0; i < VI_MAX_PIPE_NUM && i < 2; ++i) {
            std::cerr << " pipe" << i << "=" << vi_vpss.aenMode[i];
        }
        std::cerr << std::endl;
    } else {
        std::cerr << "  CVI_SYS_GetVIVPSSMode rc=0x"
                  << std::hex << rc << std::dec << std::endl;
    }

    VPSS_MODE_S vpss_mode{};
    rc = CVI_SYS_GetVPSSModeEx(&vpss_mode);
    if (rc == CVI_SUCCESS) {
        std::cerr << "  VPSS mode=" << vpss_mode.enMode;
        for (int i = 0; i < VPSS_IP_NUM && i < 2; ++i) {
            std::cerr << " dev" << i
                      << " input=" << vpss_mode.aenInput[i]
                      << " viPipe=" << vpss_mode.ViPipe[i];
        }
        std::cerr << std::endl;
    } else {
        std::cerr << "  CVI_SYS_GetVPSSModeEx rc=0x"
                  << std::hex << rc << std::dec << std::endl;
    }
}

void dump_vi_attr(VI_PIPE pipe, VI_CHN chn, const char* tag) {
    VI_CHN_ATTR_S attr{};
    CVI_S32 rc = CVI_VI_GetChnAttr(pipe, chn, &attr);
    if (rc != CVI_SUCCESS) {
        std::cerr << "[DIAG] VI attr (" << (tag ? tag : "") << ")"
                  << " CVI_VI_GetChnAttr rc=0x" << std::hex << rc << std::dec
                  << " pipe=" << pipe << " chn=" << chn << std::endl;
        return;
    }
    std::cerr << "[DIAG] VI attr (" << (tag ? tag : "") << ")"
              << " size=" << attr.stSize.u32Width << "x" << attr.stSize.u32Height
              << " pixfmt=" << attr.enPixelFormat
              << " video=" << attr.enVideoFormat
              << " depth=" << attr.u32Depth
              << " compress=" << attr.enCompressMode
              << std::endl;
}

void dump_vpss_attr(VPSS_GRP grp, VPSS_CHN chn, const char* tag) {
    VPSS_GRP_ATTR_S grp_attr{};
    CVI_S32 rc = CVI_VPSS_GetGrpAttr(grp, &grp_attr);
    if (rc != CVI_SUCCESS) {
        std::cerr << "[DIAG] VPSS grp attr (" << (tag ? tag : "") << ")"
                  << " CVI_VPSS_GetGrpAttr rc=0x" << std::hex << rc << std::dec
                  << " grp=" << grp << std::endl;
    } else {
        std::cerr << "[DIAG] VPSS grp attr (" << (tag ? tag : "") << ")"
                  << " max=" << grp_attr.u32MaxW << "x" << grp_attr.u32MaxH
                  << " pixfmt=" << grp_attr.enPixelFormat
                  << " dev=" << static_cast<int>(grp_attr.u8VpssDev)
                  << std::endl;
    }

    VPSS_CHN_ATTR_S chn_attr{};
    rc = CVI_VPSS_GetChnAttr(grp, chn, &chn_attr);
    if (rc != CVI_SUCCESS) {
        std::cerr << "[DIAG] VPSS chn attr (" << (tag ? tag : "") << ")"
                  << " CVI_VPSS_GetChnAttr rc=0x" << std::hex << rc << std::dec
                  << " grp=" << grp << " chn=" << chn << std::endl;
        return;
    }
    std::cerr << "[DIAG] VPSS chn attr (" << (tag ? tag : "") << ")"
              << " out=" << chn_attr.u32Width << "x" << chn_attr.u32Height
              << " pixfmt=" << chn_attr.enPixelFormat
              << " depth=" << chn_attr.u32Depth
              << " video=" << chn_attr.enVideoFormat
              << std::endl;
}

bool read_vpss_recv_cnt(VPSS_GRP grp, uint64_t* recv_cnt) {
    if (!recv_cnt) {
        return false;
    }
    std::ifstream file("/proc/cvitek/vpss");
    if (!file) {
        return false;
    }
    std::string line;
    bool in_section = false;
    while (std::getline(file, line)) {
        if (line.find("VPSS GRP WORK STATUS") != std::string::npos) {
            in_section = true;
            continue;
        }
        if (!in_section) {
            continue;
        }
        if (line.find("VPSS GRP") != std::string::npos || line.find("----") != std::string::npos) {
            continue;
        }
        std::istringstream iss(line);
        std::string token;
        if (!(iss >> token)) {
            continue;
        }
        if (token != "#") {
            continue;
        }
        int grp_id = -1;
        if (!(iss >> grp_id)) {
            continue;
        }
        uint64_t recv = 0;
        if (!(iss >> recv)) {
            continue;
        }
        if (grp_id == grp) {
            *recv_cnt = recv;
            return true;
        }
    }
    return false;
}

bool get_vpss_frame_once(VPSS_GRP grp, VPSS_CHN chn, VIDEO_FRAME_INFO_S* out, CVI_S32* last_rc) {
    if (!out) {
        return false;
    }
    VIDEO_FRAME_INFO_S frame{};
    CVI_S32 rc = CVI_VPSS_GetChnFrame(grp, chn, &frame, kVpssTimeoutMs);
    if (last_rc) {
        *last_rc = rc;
    }
    if (rc == CVI_SUCCESS) {
        *out = frame;
        std::cerr << "[INFO] VPSS GetChnFrame OK"
                  << " size=" << frame.stVFrame.u32Width
                  << "x" << frame.stVFrame.u32Height
                  << " pixfmt=" << frame.stVFrame.enPixelFormat
                  << " grp=" << grp << " chn=" << chn << std::endl;
        return true;
    }
    std::cerr << "[INFO] VPSS GetChnFrame rc=0x" << std::hex << rc << std::dec
              << " grp=" << grp << " chn=" << chn << std::endl;
    return false;
}

bool run_mem_vpss_send_test(uint32_t width, uint32_t height) {
    const int vpss_dev = MmfContext::vpss_dev_for_mem();
    if (vpss_dev < 0) {
        std::cerr << "[ERROR] MEM VPSS dev invalid" << std::endl;
        return false;
    }

    VPSS_GRP grp = CVI_VPSS_GetAvailableGrp();
    if (grp < 0) {
        std::cerr << "[ERROR] No available VPSS group for MEM test" << std::endl;
        return false;
    }

    int chn_id = MmfContext::vpss_channel_for_mem();
    if (chn_id < 0) {
        chn_id = 0;
    }
    VPSS_CHN chn = static_cast<VPSS_CHN>(chn_id);

    struct VpssGuard {
        VPSS_GRP grp = 0;
        VPSS_CHN chn = 0;
        VB_POOL pool = VB_INVALID_POOLID;
        bool grp_created = false;
        bool chn_enabled = false;
        bool pool_attached = false;
        bool grp_started = false;

        ~VpssGuard() {
            if (pool_attached) {
                CVI_VPSS_DetachVbPool(grp, chn);
            }
            if (chn_enabled) {
                CVI_VPSS_DisableChn(grp, chn);
            }
            if (grp_started) {
                CVI_VPSS_StopGrp(grp);
            }
            if (grp_created) {
                CVI_VPSS_DestroyGrp(grp);
            }
        }
    } guard;

    guard.grp = grp;
    guard.chn = chn;

    VPSS_GRP_ATTR_S grp_attr{};
    grp_attr.u32MaxW = width;
    grp_attr.u32MaxH = height;
    grp_attr.enPixelFormat = to_cvi_pixel_format(PixelFormat::BGR);
    grp_attr.stFrameRate.s32SrcFrameRate = -1;
    grp_attr.stFrameRate.s32DstFrameRate = -1;
    grp_attr.u8VpssDev = static_cast<CVI_U8>(vpss_dev);

    CVI_S32 rc = CVI_VPSS_CreateGrp(grp, &grp_attr);
    if (rc != CVI_SUCCESS) {
        std::cerr << "[ERROR] CVI_VPSS_CreateGrp failed: 0x"
                  << std::hex << rc << std::dec << std::endl;
        return false;
    }
    guard.grp_created = true;

    rc = CVI_VPSS_ResetGrp(grp);
    if (rc != CVI_SUCCESS) {
        std::cerr << "[ERROR] CVI_VPSS_ResetGrp failed: 0x"
                  << std::hex << rc << std::dec << std::endl;
        return false;
    }

    VPSS_CHN_ATTR_S chn_attr{};
    chn_attr.u32Width = width;
    chn_attr.u32Height = height;
    chn_attr.enVideoFormat = VIDEO_FORMAT_LINEAR;
    chn_attr.enPixelFormat = to_cvi_pixel_format(PixelFormat::BGR);
    chn_attr.stFrameRate.s32SrcFrameRate = -1;
    chn_attr.stFrameRate.s32DstFrameRate = -1;
    chn_attr.u32Depth = 2;
    chn_attr.bMirror = CVI_FALSE;
    chn_attr.bFlip = CVI_FALSE;
    chn_attr.stAspectRatio.enMode = ASPECT_RATIO_AUTO;

    rc = CVI_VPSS_SetChnAttr(grp, chn, &chn_attr);
    if (rc != CVI_SUCCESS) {
        std::cerr << "[ERROR] CVI_VPSS_SetChnAttr failed: 0x"
                  << std::hex << rc << std::dec << std::endl;
        return false;
    }

    rc = CVI_VPSS_EnableChn(grp, chn);
    if (rc != CVI_SUCCESS) {
        std::cerr << "[ERROR] CVI_VPSS_EnableChn failed: 0x"
                  << std::hex << rc << std::dec << std::endl;
        return false;
    }
    guard.chn_enabled = true;

    VB_POOL pool = MmfContext::instance().vb_plan().find_pool(width, height, PixelFormat::BGR);
    if (pool == VB_INVALID_POOLID) {
        std::cerr << "[ERROR] No VB pool for MEM VPSS output "
                  << width << "x" << height << " BGR" << std::endl;
        return false;
    }
    guard.pool = pool;

    rc = CVI_VPSS_AttachVbPool(grp, chn, pool);
    if (rc != CVI_SUCCESS) {
        std::cerr << "[ERROR] CVI_VPSS_AttachVbPool failed: 0x"
                  << std::hex << rc << std::dec << " pool=" << pool << std::endl;
        return false;
    }
    guard.pool_attached = true;

    rc = CVI_VPSS_StartGrp(grp);
    if (rc != CVI_SUCCESS) {
        std::cerr << "[ERROR] CVI_VPSS_StartGrp failed: 0x"
                  << std::hex << rc << std::dec << std::endl;
        return false;
    }
    guard.grp_started = true;

    cv::Mat mat = create_test_image(static_cast<int>(width), static_cast<int>(height));
    CviVpssProcessor helper;
    VB_BLK input_block = VB_INVALID_HANDLE;
    VIDEO_FRAME_INFO_S input_frame{};
    try {
        input_frame = helper.mat_to_video_frame(mat, input_block, pool);
    } catch (const std::exception& ex) {
        std::cerr << "[ERROR] mat_to_video_frame failed: " << ex.what() << std::endl;
        return false;
    }
    VbBlockGuard block_guard(input_block);

    rc = CVI_VPSS_SendFrame(grp, &input_frame, kVpssTimeoutMs);
    if (rc != CVI_SUCCESS) {
        std::cerr << "[ERROR] CVI_VPSS_SendFrame failed: 0x"
                  << std::hex << rc << std::dec << std::endl;
        return false;
    }

    VIDEO_FRAME_INFO_S output_frame{};
    rc = CVI_VPSS_GetChnFrame(grp, chn, &output_frame, kVpssTimeoutMs);
    if (rc != CVI_SUCCESS) {
        std::cerr << "[ERROR] CVI_VPSS_GetChnFrame (MEM) failed: 0x"
                  << std::hex << rc << std::dec << std::endl;
        return false;
    }

    CVI_VPSS_ReleaseChnFrame(grp, chn, &output_frame);
    return true;
}
}  // namespace

// NOTE: Camera-based VI-VPSS link tests have been consolidated into
// CameraCaptureTest to avoid multiple camera open/close cycles.
// See: test_camera_capture.cpp
//   - CameraCaptureTest.StreamChannel (VPSS chn0 stream path)
//   - CameraCaptureTest.InferChannel (VPSS chn1 infer path)
//
// MEM VPSS tests are covered by VpssProcessorTest (test_vpss_processor.cpp)

TEST(CameraLink, ConsolidatedIntoCameraCaptureTest) {
    GTEST_SKIP() << "Test consolidated into CameraCaptureTest fixture";
}

#else

TEST(CameraLink, Skipped) {
    GTEST_SKIP() << "USE_CVI_CAMERA not defined";
}

#endif
#else

TEST(CameraLink, Skipped) {
    GTEST_SKIP() << "USE_CVI_MPI not defined";
}

#endif
