#include "cvi_camera.h"

#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <unistd.h>
#include <chrono>

#ifdef USE_CVI_CAMERA
#include <cvi_buffer.h>
#include <cvi_sys.h>
#include <cvi_vb.h>
#include <cvi_vi.h>
#include <cvi_vpss.h>
#include <cvi_bin.h>
#include <cvi_isp.h>
#include <cvi_ae.h>
#include <cvi_awb.h>
#include <linux/cvi_errno.h>
#include <linux/cif_uapi.h>
#include <sys/prctl.h>
#include "mmf_context.h"
#endif

namespace lua_cv {

#ifdef USE_CVI_CAMERA
namespace {
constexpr int kIspWarmupDelayUs = 300000;
constexpr int kGetFrameTimeoutMs = 2000;
constexpr int kReadyPollTimeoutMs = 200;
constexpr int kReadyPollSleepUs = 20000;

VI_DEV_ATTR_S make_vi_dev_attr_base() {
    VI_DEV_ATTR_S attr{};
    attr.enIntfMode = VI_MODE_MIPI;
    attr.enWorkMode = VI_WORK_MODE_1Multiplex;
    attr.enScanMode = VI_SCAN_PROGRESSIVE;
    attr.as32AdChnId[0] = -1;
    attr.as32AdChnId[1] = -1;
    attr.as32AdChnId[2] = -1;
    attr.as32AdChnId[3] = -1;
    attr.enDataSeq = VI_DATA_SEQ_YUYV;
    attr.stSynCfg.enVsync = VI_VSYNC_PULSE;
    attr.stSynCfg.enVsyncNeg = VI_VSYNC_NEG_LOW;
    attr.stSynCfg.enHsync = VI_HSYNC_VALID_SINGNAL;
    attr.stSynCfg.enHsyncNeg = VI_HSYNC_NEG_HIGH;
    attr.stSynCfg.enVsyncValid = VI_VSYNC_VALID_SIGNAL;
    attr.stSynCfg.enVsyncValidNeg = VI_VSYNC_VALID_NEG_HIGH;
    attr.stSynCfg.stTimingBlank.u32HsyncHfb = 0;
    attr.stSynCfg.stTimingBlank.u32HsyncAct = 1920;
    attr.stSynCfg.stTimingBlank.u32HsyncHbb = 0;
    attr.stSynCfg.stTimingBlank.u32VsyncVfb = 0;
    attr.stSynCfg.stTimingBlank.u32VsyncVact = 1080;
    attr.stSynCfg.stTimingBlank.u32VsyncVbb = 0;
    attr.stSynCfg.stTimingBlank.u32VsyncVbfb = 0;
    attr.stSynCfg.stTimingBlank.u32VsyncVbact = 0;
    attr.stSynCfg.stTimingBlank.u32VsyncVbbb = 0;
    attr.enInputDataType = VI_DATA_TYPE_RGB;
    attr.stSize.u32Width = 1920;
    attr.stSize.u32Height = 1080;
    attr.stWDRAttr.enWDRMode = WDR_MODE_NONE;
    attr.stWDRAttr.u32CacheLine = 1080;
    attr.enBayerFormat = BAYER_FORMAT_BG;
    return attr;
}

VI_PIPE_ATTR_S make_vi_pipe_attr_base() {
    VI_PIPE_ATTR_S attr{};
    attr.enPipeBypassMode = VI_PIPE_BYPASS_NONE;
    attr.bYuvSkip = CVI_FALSE;
    attr.bIspBypass = CVI_FALSE;
    attr.u32MaxW = 1920;
    attr.u32MaxH = 1080;
    attr.enPixFmt = PIXEL_FORMAT_RGB_BAYER_12BPP;
    attr.enCompressMode = COMPRESS_MODE_NONE;
    attr.enBitWidth = DATA_BITWIDTH_12;
    attr.bNrEn = CVI_TRUE;
    attr.bSharpenEn = CVI_FALSE;
    attr.stFrameRate.s32SrcFrameRate = -1;
    attr.stFrameRate.s32DstFrameRate = -1;
    attr.bDiscardProPic = CVI_FALSE;
    attr.bYuvBypassPath = CVI_FALSE;
    return attr;
}

VI_CHN_ATTR_S make_vi_chn_attr_base() {
    VI_CHN_ATTR_S attr{};
    attr.stSize.u32Width = 1920;
    attr.stSize.u32Height = 1080;
    attr.enPixelFormat = PIXEL_FORMAT_YUV_PLANAR_420;
    attr.enDynamicRange = DYNAMIC_RANGE_SDR8;
    attr.enVideoFormat = VIDEO_FORMAT_LINEAR;
    attr.enCompressMode = COMPRESS_MODE_NONE;
    attr.bMirror = CVI_FALSE;
    attr.bFlip = CVI_FALSE;
    attr.u32Depth = 0;
    attr.stFrameRate.s32SrcFrameRate = -1;
    attr.stFrameRate.s32DstFrameRate = -1;
    return attr;
}

ISP_BAYER_FORMAT_E to_isp_bayer(BAYER_FORMAT_E bayer) {
    switch (bayer) {
        case BAYER_FORMAT_BG:
            return BAYER_BGGR;
        case BAYER_FORMAT_GB:
            return BAYER_GBRG;
        case BAYER_FORMAT_GR:
            return BAYER_GRBG;
        case BAYER_FORMAT_RG:
            return BAYER_RGGB;
        default:
            return BAYER_BGGR;
    }
}

bool check_rc(CVI_S32 rc, const char* what) {
    if (rc != CVI_SUCCESS) {
        std::cerr << "[ERROR] " << what << " failed: 0x" << std::hex << rc << std::dec << std::endl;
        return false;
    }
    return true;
}

CVI_U8 get_vpss_dev_for_input(VPSS_INPUT_E input_type) {
    VPSS_MODE_S mode{};
    if (CVI_SYS_GetVPSSModeEx(&mode) == CVI_SUCCESS && mode.enMode == VPSS_MODE_DUAL) {
        for (int i = 0; i < VPSS_IP_NUM; ++i) {
            if (mode.aenInput[i] == input_type) {
                return static_cast<CVI_U8>(i);
            }
        }
    }
    return 0;
}

const char* pq_bin_path_for_sensor(const CviSensor::Profile* profile) {
    static char path[64];
    std::snprintf(path, sizeof(path), "/mnt/cfg/param/");
    if (!profile) {
        std::strncat(path, "ov_ov5647_sdr.bin", sizeof(path) - std::strlen(path) - 1);
        return path;
    }
    if (std::strcmp(profile->name, "SC530AI_2L") == 0) {
        std::strncat(path, "sms_sc530ai_2l_sdr.bin", sizeof(path) - std::strlen(path) - 1);
    } else if (std::strcmp(profile->name, "GC2053") == 0) {
        std::strncat(path, "gcore_gc2053_sdr.bin", sizeof(path) - std::strlen(path) - 1);
    } else {
        std::strncat(path, "ov_ov5647_sdr.bin", sizeof(path) - std::strlen(path) - 1);
    }
    return path;
}

bool load_pq_bin(const char* path) {
    FILE* fp = std::fopen(path, "rb");
    if (!fp) {
        std::cerr << "[WARN] PQ bin not found: " << path << std::endl;
        return false;
    }

    std::fseek(fp, 0L, SEEK_END);
    long size = std::ftell(fp);
    std::rewind(fp);
    if (size <= 0) {
        std::fclose(fp);
        std::cerr << "[WARN] PQ bin empty: " << path << std::endl;
        return false;
    }

    std::string buffer;
    buffer.resize(static_cast<size_t>(size));
    size_t read_size = std::fread(buffer.data(), 1, buffer.size(), fp);
    std::fclose(fp);
    if (read_size != buffer.size()) {
        std::cerr << "[WARN] PQ bin read incomplete: " << path << std::endl;
        return false;
    }

    CVI_S32 rc = CVI_BIN_ImportBinData(reinterpret_cast<CVI_U8*>(buffer.data()),
                                       static_cast<CVI_U32>(buffer.size()));
    if (rc != CVI_SUCCESS) {
        std::cerr << "[WARN] CVI_BIN_ImportBinData failed: 0x" << std::hex << rc << std::dec << std::endl;
        return false;
    }
    return true;
}

void* isp_thread_entry(void* arg) {
    VI_PIPE pipe = *static_cast<VI_PIPE*>(arg);
    char name[16];
    std::snprintf(name, sizeof(name), "ISP%d_RUN", pipe);
    prctl(PR_SET_NAME, name, 0, 0, 0);

    CVI_S32 rc = CVI_ISP_Run(pipe);
    if (rc != CVI_SUCCESS) {
        std::cerr << "[ERROR] CVI_ISP_Run(" << pipe << ") failed: 0x"
                  << std::hex << rc << std::dec << std::endl;
    }
    return nullptr;
}
} // namespace

CviCamera::CviCamera() : config_(Config{}) {}

CviCamera::CviCamera(const Config& config) : config_(config) {}

CviCamera::~CviCamera() {
    release();
}

bool CviCamera::open() {
    if (opened_) {
        std::cerr << "[WARN] CviCamera::open - already opened" << std::endl;
        return true;
    }

    if (!sensor_.init()) {
        return false;
    }
    sensor_profile_ = sensor_.profile();

    width_ = (config_.width > 0) ? config_.width : sensor_.get_width();
    height_ = (config_.height > 0) ? config_.height : sensor_.get_height();
    fps_ = (config_.fps > 0.0) ? config_.fps : static_cast<double>(sensor_.get_fps());

    if (!init_system()) {
        cleanup();
        return false;
    }

    if (!start_vi()) {
        cleanup();
        return false;
    }

    if (!init_vpss()) {
        cleanup();
        return false;
    }

    if (!bind_vi_vpss()) {
        cleanup();
        return false;
    }

    std::cout << "[INFO] CviCamera: Sensor " << get_sensor_name()
              << " " << width_ << "x" << height_ << " @ " << fps_ << " fps" << std::endl;
    usleep(kIspWarmupDelayUs);

    opened_ = true;
    return true;
}

bool CviCamera::read(Frame& frame) {
    return read_internal(frame, kGetFrameTimeoutMs, true);
}

bool CviCamera::read(Frame& frame, int timeout_ms) {
    return read_internal(frame, timeout_ms, true);
}

bool CviCamera::wait_for_ready(int timeout_ms) {
    if (!opened_) {
        std::cerr << "[ERROR] CviCamera::wait_for_ready - camera not opened" << std::endl;
        return false;
    }

    auto start = std::chrono::steady_clock::now();
    while (true) {
        Frame frame;
        if (read_internal(frame, kReadyPollTimeoutMs, false)) {
            frame.release();
            return true;
        }

        auto now = std::chrono::steady_clock::now();
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
        if (elapsed_ms >= timeout_ms) {
            std::cerr << "[WARN] CviCamera::wait_for_ready - timeout after "
                      << elapsed_ms << " ms" << std::endl;
            return false;
        }
        usleep(kReadyPollSleepUs);
    }
}

bool CviCamera::read_internal(Frame& frame, int timeout_ms, bool log_error) {
    if (!opened_) {
        if (log_error) {
            std::cerr << "[ERROR] CviCamera::read - camera not opened" << std::endl;
        }
        return false;
    }

    VIDEO_FRAME_INFO_S cvi_frame{};
    CVI_S32 rc = CVI_VPSS_GetChnFrame(vpss_grp_, vpss_chn_, &cvi_frame, timeout_ms);
    if (rc != CVI_SUCCESS) {
        if (log_error) {
            std::cerr << "[ERROR] CviCamera::read - CVI_VPSS_GetChnFrame failed: 0x"
                      << std::hex << rc << std::dec << std::endl;
        }
        return false;
    }

    Frame out(cvi_frame, static_cast<int>(vpss_grp_), static_cast<int>(vpss_chn_));
    out.set_vpss_owner(static_cast<int>(vpss_grp_), static_cast<int>(vpss_chn_));
    frame = std::move(out);
    return true;
}

void CviCamera::release() {
    cleanup();
}

int CviCamera::width() const {
    return width_;
}

int CviCamera::height() const {
    return height_;
}

double CviCamera::fps() const {
    return fps_;
}

bool CviCamera::is_opened() const {
    return opened_;
}

const char* CviCamera::get_sensor_name() const {
    return sensor_profile_ ? sensor_profile_->name : "unknown";
}

bool CviCamera::init_system() {
    if (!MmfContext::instance().is_initialized()) {
        std::cerr << "[ERROR] CviCamera: MmfContext not initialized" << std::endl;
        return false;
    }

    if (CVI_VB_IsInited() != CVI_TRUE || CVI_SYS_IsInited() != CVI_TRUE) {
        std::cerr << "[ERROR] CviCamera: CVI VB/SYS not initialized" << std::endl;
        return false;
    }

    VI_VPSS_MODE_S vi_vpss_mode{};
    if (!check_rc(CVI_SYS_GetVIVPSSMode(&vi_vpss_mode), "CVI_SYS_GetVIVPSSMode")) {
        return false;
    }
    if (vi_vpss_mode.aenMode[vi_pipe_] != VI_OFFLINE_VPSS_ONLINE) {
        std::cerr << "[ERROR] CviCamera: VI-VPSS mode must be VI_OFFLINE_VPSS_ONLINE" << std::endl;
        return false;
    }

    VPSS_MODE_S vpss_mode{};
    if (!check_rc(CVI_SYS_GetVPSSModeEx(&vpss_mode), "CVI_SYS_GetVPSSModeEx")) {
        return false;
    }
    if (vpss_mode.enMode != VPSS_MODE_DUAL) {
        std::cerr << "[ERROR] CviCamera: VPSS mode must be DUAL (ISP + MEM)" << std::endl;
        return false;
    }

    bool has_isp_input = false;
    for (int i = 0; i < VPSS_IP_NUM; ++i) {
        if (vpss_mode.aenInput[i] == VPSS_INPUT_ISP) {
            has_isp_input = true;
            break;
        }
    }
    if (!has_isp_input) {
        std::cerr << "[ERROR] CviCamera: VPSS mode must include ISP input" << std::endl;
        return false;
    }

    return true;
}

bool CviCamera::start_vi() {
    if (!check_rc(CVI_SYS_VI_Open(), "CVI_SYS_VI_Open")) {
        return false;
    }
    vi_opened_ = true;

    if (!start_sensor()) {
        return false;
    }
    if (!start_mipi()) {
        return false;
    }
    if (!start_vi_dev()) {
        return false;
    }
    if (!start_vi_pipe()) {
        return false;
    }
    if (!init_isp()) {
        return false;
    }
    if (!start_isp()) {
        return false;
    }
    if (!start_vi_channel()) {
        return false;
    }
    return true;
}

bool CviCamera::start_sensor() {
    if (!sensor_profile_ || !sensor_profile_->sns_obj) {
        std::cerr << "[ERROR] CviCamera: sensor profile missing" << std::endl;
        return false;
    }

    ISP_SNS_OBJ_S* sns_obj = sensor_profile_->sns_obj;

    RX_INIT_ATTR_S rx_attr{};
    rx_attr.MipiDev = sensor_profile_->mipi_dev;
    if (sensor_profile_->mclk_en) {
        rx_attr.stMclkAttr.bMclkEn = CVI_TRUE;
        rx_attr.stMclkAttr.u8Mclk = sensor_profile_->mclk;
    }
    for (size_t i = 0; i < sizeof(rx_attr.as16LaneId) / sizeof(rx_attr.as16LaneId[0]); ++i) {
        rx_attr.as16LaneId[i] = sensor_profile_->lane_id[i];
    }
    for (size_t i = 0; i < sizeof(rx_attr.as8PNSwap) / sizeof(rx_attr.as8PNSwap[0]); ++i) {
        rx_attr.as8PNSwap[i] = sensor_profile_->pn_swap[i];
    }
    if (sns_obj->pfnPatchRxAttr) {
        if (!check_rc(sns_obj->pfnPatchRxAttr(&rx_attr), "pfnPatchRxAttr")) {
            return false;
        }
    }

    ISP_INIT_ATTR_S init_attr{};
    init_attr.u16UseHwSync = sensor_profile_->hw_sync ? 1 : 0;
    init_attr.enGainMode = SNS_GAIN_MODE_SHARE;
    init_attr.enL2SMode = SNS_L2S_MODE_AUTO;
    init_attr.enSnsBdgMuxMode = SNS_BDG_MUX_NONE;
    if (sns_obj->pfnSetInit) {
        if (!check_rc(sns_obj->pfnSetInit(vi_pipe_, &init_attr), "pfnSetInit")) {
            return false;
        }
    }

    ISP_SNS_COMMBUS_U bus_info{};
    bus_info.s8I2cDev = (sensor_profile_->bus_id >= 0) ?
        static_cast<CVI_S8>(sensor_profile_->bus_id) : 0x3;
    if (sns_obj->pfnSetBusInfo) {
        if (!check_rc(sns_obj->pfnSetBusInfo(vi_pipe_, bus_info), "pfnSetBusInfo")) {
            return false;
        }
    }

    if (sns_obj->pfnPatchI2cAddr) {
        sns_obj->pfnPatchI2cAddr(sensor_profile_->i2c_addr);
    }

    ALG_LIB_S ae_lib{};
    ALG_LIB_S awb_lib{};
    ae_lib.s32Id = vi_pipe_;
    awb_lib.s32Id = vi_pipe_;
    std::snprintf(ae_lib.acLibName, sizeof(ae_lib.acLibName), "%s", CVI_AE_LIB_NAME);
    std::snprintf(awb_lib.acLibName, sizeof(awb_lib.acLibName), "%s", CVI_AWB_LIB_NAME);
    if (!sns_obj->pfnRegisterCallback) {
        std::cerr << "[ERROR] CviCamera: sensor missing pfnRegisterCallback" << std::endl;
        return false;
    }
    if (!check_rc(sns_obj->pfnRegisterCallback(vi_pipe_, &ae_lib, &awb_lib),
                  "pfnRegisterCallback")) {
        return false;
    }
    sensor_callbacks_registered_ = true;

    ISP_SENSOR_EXP_FUNC_S sensor_exp{};
    if (sns_obj->pfnExpSensorCb) {
        if (!check_rc(sns_obj->pfnExpSensorCb(&sensor_exp), "pfnExpSensorCb")) {
            return false;
        }
        if (sensor_exp.pfn_cmos_sensor_global_init) {
            sensor_exp.pfn_cmos_sensor_global_init(vi_pipe_);
        }
        ISP_CMOS_SENSOR_IMAGE_MODE_S sensor_mode{};
        sensor_mode.u16Width = static_cast<CVI_U16>(width_);
        sensor_mode.u16Height = static_cast<CVI_U16>(height_);
        sensor_mode.f32Fps = static_cast<CVI_FLOAT>(fps_);
        if (sensor_exp.pfn_cmos_set_image_mode) {
            if (!check_rc(sensor_exp.pfn_cmos_set_image_mode(vi_pipe_, &sensor_mode),
                          "pfn_cmos_set_image_mode")) {
                return false;
            }
        }
        if (sensor_exp.pfn_cmos_set_wdr_mode) {
            if (!check_rc(sensor_exp.pfn_cmos_set_wdr_mode(vi_pipe_, sensor_profile_->wdr_mode),
                          "pfn_cmos_set_wdr_mode")) {
                return false;
            }
        }
    }

    return true;
}

bool CviCamera::start_mipi() {
    if (!sensor_profile_ || !sensor_profile_->sns_obj) {
        return false;
    }
    ISP_SNS_OBJ_S* sns_obj = sensor_profile_->sns_obj;
    SNS_COMBO_DEV_ATTR_S combo_attr{};

    if (!sns_obj->pfnGetRxAttr) {
        std::cerr << "[ERROR] CviCamera: sensor missing pfnGetRxAttr" << std::endl;
        return false;
    }
    if (!check_rc(sns_obj->pfnGetRxAttr(vi_pipe_, &combo_attr), "pfnGetRxAttr")) {
        return false;
    }

    CVI_S32 devno = static_cast<CVI_S32>(combo_attr.devno);
    if (!check_rc(CVI_MIPI_SetSensorReset(devno, 1), "CVI_MIPI_SetSensorReset(1)")) {
        return false;
    }
    if (!check_rc(CVI_MIPI_SetMipiReset(devno, 1), "CVI_MIPI_SetMipiReset(1)")) {
        return false;
    }
    if (!check_rc(CVI_MIPI_SetMipiAttr(devno, &combo_attr), "CVI_MIPI_SetMipiAttr")) {
        return false;
    }
    if (!check_rc(CVI_MIPI_SetSensorClock(devno, 1), "CVI_MIPI_SetSensorClock")) {
        return false;
    }
    usleep(20);
    if (!check_rc(CVI_MIPI_SetSensorReset(devno, 0), "CVI_MIPI_SetSensorReset(0)")) {
        return false;
    }

    if (sns_obj->pfnSnsProbe) {
        if (!check_rc(sns_obj->pfnSnsProbe(vi_pipe_), "pfnSnsProbe")) {
            return false;
        }
    }
    return true;
}

bool CviCamera::start_vi_dev() {
    VI_DEV_ATTR_S dev_attr = make_vi_dev_attr_base();
    dev_attr.stSize.u32Width = static_cast<CVI_U32>(width_);
    dev_attr.stSize.u32Height = static_cast<CVI_U32>(height_);
    dev_attr.stSynCfg.stTimingBlank.u32HsyncAct = static_cast<CVI_U32>(width_);
    dev_attr.stSynCfg.stTimingBlank.u32VsyncVact = static_cast<CVI_U32>(height_);
    dev_attr.stWDRAttr.enWDRMode = sensor_profile_ ? sensor_profile_->wdr_mode : WDR_MODE_NONE;
    dev_attr.stWDRAttr.u32CacheLine = static_cast<CVI_U32>(height_);
    if (sensor_profile_) {
        dev_attr.enBayerFormat = sensor_profile_->bayer;
    }

    if (!check_rc(CVI_VI_SetDevAttr(vi_dev_, &dev_attr), "CVI_VI_SetDevAttr")) {
        return false;
    }
    if (!check_rc(CVI_VI_EnableDev(vi_dev_), "CVI_VI_EnableDev")) {
        return false;
    }
    vi_dev_enabled_ = true;
    return true;
}

bool CviCamera::start_vi_pipe() {
    VI_PIPE_ATTR_S pipe_attr = make_vi_pipe_attr_base();
    pipe_attr.u32MaxW = static_cast<CVI_U32>(width_);
    pipe_attr.u32MaxH = static_cast<CVI_U32>(height_);
    pipe_attr.enCompressMode = COMPRESS_MODE_NONE;

    if (!check_rc(CVI_VI_CreatePipe(vi_pipe_, &pipe_attr), "CVI_VI_CreatePipe")) {
        return false;
    }
    vi_pipe_created_ = true;

    if (!check_rc(CVI_VI_StartPipe(vi_pipe_), "CVI_VI_StartPipe")) {
        return false;
    }
    vi_pipe_started_ = true;
    return true;
}

bool CviCamera::init_isp() {
    ALG_LIB_S ae_lib{};
    ALG_LIB_S awb_lib{};
    ae_lib.s32Id = vi_pipe_;
    awb_lib.s32Id = vi_pipe_;
    std::snprintf(ae_lib.acLibName, sizeof(ae_lib.acLibName), "%s", CVI_AE_LIB_NAME);
    std::snprintf(awb_lib.acLibName, sizeof(awb_lib.acLibName), "%s", CVI_AWB_LIB_NAME);
    if (!check_rc(CVI_AE_Register(vi_pipe_, &ae_lib), "CVI_AE_Register")) {
        return false;
    }
    ae_registered_ = true;
    if (!check_rc(CVI_AWB_Register(vi_pipe_, &awb_lib), "CVI_AWB_Register")) {
        return false;
    }
    awb_registered_ = true;

    ISP_BIND_ATTR_S bind_attr{};
    bind_attr.sensorId = 0;
    std::snprintf(bind_attr.stAeLib.acLibName, sizeof(bind_attr.stAeLib.acLibName), "%s", CVI_AE_LIB_NAME);
    bind_attr.stAeLib.s32Id = vi_pipe_;
    std::snprintf(bind_attr.stAwbLib.acLibName, sizeof(bind_attr.stAwbLib.acLibName), "%s", CVI_AWB_LIB_NAME);
    bind_attr.stAwbLib.s32Id = vi_pipe_;
    if (!check_rc(CVI_ISP_SetBindAttr(vi_pipe_, &bind_attr), "CVI_ISP_SetBindAttr")) {
        return false;
    }

    if (!check_rc(CVI_ISP_MemInit(vi_pipe_), "CVI_ISP_MemInit")) {
        return false;
    }

    ISP_PUB_ATTR_S pub_attr{};
    pub_attr.stWndRect.s32X = 0;
    pub_attr.stWndRect.s32Y = 0;
    pub_attr.stWndRect.u32Width = static_cast<CVI_U32>(width_);
    pub_attr.stWndRect.u32Height = static_cast<CVI_U32>(height_);
    pub_attr.stSnsSize.u32Width = static_cast<CVI_U32>(width_);
    pub_attr.stSnsSize.u32Height = static_cast<CVI_U32>(height_);
    pub_attr.f32FrameRate = static_cast<CVI_FLOAT>(fps_);
    pub_attr.enWDRMode = sensor_profile_ ? sensor_profile_->wdr_mode : WDR_MODE_NONE;
    pub_attr.enBayer = sensor_profile_ ? to_isp_bayer(sensor_profile_->bayer) : BAYER_BGGR;
    pub_attr.u8SnsMode = 0;
    if (!check_rc(CVI_ISP_SetPubAttr(vi_pipe_, &pub_attr), "CVI_ISP_SetPubAttr")) {
        return false;
    }

    ISP_STATISTICS_CFG_S sts_cfg{};
    if (!check_rc(CVI_ISP_GetStatisticsConfig(vi_pipe_, &sts_cfg), "CVI_ISP_GetStatisticsConfig")) {
        return false;
    }

    sts_cfg.stAECfg.stCrop[0].bEnable = 0;
    sts_cfg.stAECfg.stCrop[0].u16X = 0;
    sts_cfg.stAECfg.stCrop[0].u16Y = 0;
    sts_cfg.stAECfg.stCrop[0].u16W = pub_attr.stWndRect.u32Width;
    sts_cfg.stAECfg.stCrop[0].u16H = pub_attr.stWndRect.u32Height;
    std::memset(sts_cfg.stAECfg.au8Weight, 1,
                AE_WEIGHT_ZONE_ROW * AE_WEIGHT_ZONE_COLUMN * sizeof(CVI_U8));

    sts_cfg.stWBCfg.u16ZoneRow = AWB_ZONE_ORIG_ROW;
    sts_cfg.stWBCfg.u16ZoneCol = AWB_ZONE_ORIG_COLUMN;
    sts_cfg.stWBCfg.stCrop.u16X = 0;
    sts_cfg.stWBCfg.stCrop.u16Y = 0;
    sts_cfg.stWBCfg.stCrop.u16W = pub_attr.stWndRect.u32Width;
    sts_cfg.stWBCfg.stCrop.u16H = pub_attr.stWndRect.u32Height;
    sts_cfg.stWBCfg.u16BlackLevel = 0;
    sts_cfg.stWBCfg.u16WhiteLevel = 4095;

    sts_cfg.stFocusCfg.stConfig.bEnable = 1;
    sts_cfg.stFocusCfg.stConfig.u8HFltShift = 1;
    sts_cfg.stFocusCfg.stConfig.s8HVFltLpCoeff[0] = 1;
    sts_cfg.stFocusCfg.stConfig.s8HVFltLpCoeff[1] = 2;
    sts_cfg.stFocusCfg.stConfig.s8HVFltLpCoeff[2] = 3;
    sts_cfg.stFocusCfg.stConfig.s8HVFltLpCoeff[3] = 5;
    sts_cfg.stFocusCfg.stConfig.s8HVFltLpCoeff[4] = 10;
    sts_cfg.stFocusCfg.stConfig.stRawCfg.PreGammaEn = 0;
    sts_cfg.stFocusCfg.stConfig.stPreFltCfg.PreFltEn = 1;
    sts_cfg.stFocusCfg.stConfig.u16Hwnd = 17;
    sts_cfg.stFocusCfg.stConfig.u16Vwnd = 15;
    sts_cfg.stFocusCfg.stConfig.stCrop.bEnable = 0;
    sts_cfg.stFocusCfg.stConfig.stCrop.u16X = AF_XOFFSET_MIN;
    sts_cfg.stFocusCfg.stConfig.stCrop.u16Y = AF_YOFFSET_MIN;
    sts_cfg.stFocusCfg.stConfig.stCrop.u16W = pub_attr.stWndRect.u32Width - AF_XOFFSET_MIN * 2;
    sts_cfg.stFocusCfg.stConfig.stCrop.u16H = pub_attr.stWndRect.u32Height - AF_YOFFSET_MIN * 2;
    sts_cfg.stFocusCfg.stHParam_FIR0.s8HFltHpCoeff[0] = 0;
    sts_cfg.stFocusCfg.stHParam_FIR0.s8HFltHpCoeff[1] = 0;
    sts_cfg.stFocusCfg.stHParam_FIR0.s8HFltHpCoeff[2] = 13;
    sts_cfg.stFocusCfg.stHParam_FIR0.s8HFltHpCoeff[3] = 24;
    sts_cfg.stFocusCfg.stHParam_FIR0.s8HFltHpCoeff[4] = 0;
    sts_cfg.stFocusCfg.stHParam_FIR1.s8HFltHpCoeff[0] = 1;
    sts_cfg.stFocusCfg.stHParam_FIR1.s8HFltHpCoeff[1] = 2;
    sts_cfg.stFocusCfg.stHParam_FIR1.s8HFltHpCoeff[2] = 4;
    sts_cfg.stFocusCfg.stHParam_FIR1.s8HFltHpCoeff[3] = 8;
    sts_cfg.stFocusCfg.stHParam_FIR1.s8HFltHpCoeff[4] = 0;
    sts_cfg.stFocusCfg.stVParam_FIR.s8VFltHpCoeff[0] = 13;
    sts_cfg.stFocusCfg.stVParam_FIR.s8VFltHpCoeff[1] = 24;
    sts_cfg.stFocusCfg.stVParam_FIR.s8VFltHpCoeff[2] = 0;
    sts_cfg.unKey.bit1FEAeGloStat = 1;
    sts_cfg.unKey.bit1FEAeLocStat = 1;
    sts_cfg.unKey.bit1AwbStat1 = 1;
    sts_cfg.unKey.bit1AwbStat2 = 1;
    sts_cfg.unKey.bit1FEAfStat = 1;
    sts_cfg.stFocusCfg.stConfig.u8ThLow = 0;
    sts_cfg.stFocusCfg.stConfig.u8ThHigh = 255;
    sts_cfg.stFocusCfg.stConfig.u8GainLow = 30;
    sts_cfg.stFocusCfg.stConfig.u8GainHigh = 20;
    sts_cfg.stFocusCfg.stConfig.u8SlopLow = 8;
    sts_cfg.stFocusCfg.stConfig.u8SlopHigh = 15;

    if (!check_rc(CVI_ISP_SetStatisticsConfig(vi_pipe_, &sts_cfg), "CVI_ISP_SetStatisticsConfig")) {
        return false;
    }

    if (!check_rc(CVI_ISP_Init(vi_pipe_), "CVI_ISP_Init")) {
        return false;
    }
    isp_inited_ = true;

    const char* pq_path = pq_bin_path_for_sensor(sensor_profile_);
    load_pq_bin(pq_path);
    return true;
}

bool CviCamera::start_isp() {
    if (pthread_create(&isp_thread_, nullptr, isp_thread_entry, &vi_pipe_) != 0) {
        std::cerr << "[ERROR] CviCamera: failed to create ISP thread" << std::endl;
        return false;
    }
    isp_thread_running_ = true;

    VI_DEV_ATTR_S dev_attr{};
    if (check_rc(CVI_VI_GetDevAttr(vi_dev_, &dev_attr), "CVI_VI_GetDevAttr")) {
        const char* pq_path = pq_bin_path_for_sensor(sensor_profile_);
        CVI_S32 rc = CVI_BIN_SetBinName(dev_attr.stWDRAttr.enWDRMode, pq_path);
        if (rc != CVI_SUCCESS) {
            std::cerr << "[WARN] CVI_BIN_SetBinName failed: 0x" << std::hex << rc << std::dec << std::endl;
        }
    }

    ISP_PUB_ATTR_S pub_attr{};
    if (CVI_ISP_GetPubAttr(vi_pipe_, &pub_attr) == CVI_SUCCESS) {
        pub_attr.f32FrameRate = static_cast<CVI_FLOAT>(fps_);
        CVI_ISP_SetPubAttr(vi_pipe_, &pub_attr);
    }

    ISP_CTRL_PARAM_S ctrl_param{};
    ctrl_param.u32ProcLevel = 0;
    ctrl_param.u32ProcParam = 15;
    ctrl_param.u32AEStatIntvl = 1;
    ctrl_param.u32AWBStatIntvl = 6;
    ctrl_param.u32AFStatIntvl = 1;
    ctrl_param.u32UpdatePos = 0;
    ctrl_param.u32IntTimeOut = 0;
    ctrl_param.u32PwmNumber = 0;
    ctrl_param.u32PortIntDelay = 0;
    CVI_ISP_SetCtrlParam(vi_pipe_, &ctrl_param);

    return true;
}

bool CviCamera::start_vi_channel() {
    VI_CHN_ATTR_S chn_attr = make_vi_chn_attr_base();
    chn_attr.stSize.u32Width = static_cast<CVI_U32>(width_);
    chn_attr.stSize.u32Height = static_cast<CVI_U32>(height_);
    chn_attr.enCompressMode = COMPRESS_MODE_NONE;

    chn_attr.enPixelFormat = PIXEL_FORMAT_NV21;
    chn_attr.u32Depth = 3;

    if (sensor_profile_) {
        chn_attr.bMirror = (sensor_profile_->orientation & ISP_SNS_MIRROR) != 0;
        chn_attr.bFlip = (sensor_profile_->orientation & ISP_SNS_FLIP) != 0;
    }

    if (!check_rc(CVI_VI_SetChnAttr(vi_pipe_, vi_chn_, &chn_attr), "CVI_VI_SetChnAttr")) {
        return false;
    }

    if (sensor_profile_ && sensor_profile_->sns_obj && sensor_profile_->sns_obj->pfnMirrorFlip) {
        CVI_VI_RegChnFlipMirrorCallBack(vi_pipe_, vi_chn_,
                                        reinterpret_cast<void*>(sensor_profile_->sns_obj->pfnMirrorFlip));
    }

    if (!check_rc(CVI_VI_EnableChn(vi_pipe_, vi_chn_), "CVI_VI_EnableChn")) {
        return false;
    }
    vi_chn_enabled_ = true;

    CVI_VI_SetChnFlipMirror(vi_pipe_, vi_chn_, chn_attr.bFlip, chn_attr.bMirror);
    return true;
}

bool CviCamera::init_vpss() {
    VPSS_GRP_ATTR_S grp_attr{};
    grp_attr.u32MaxW = static_cast<CVI_U32>(width_);
    grp_attr.u32MaxH = static_cast<CVI_U32>(height_);
    grp_attr.enPixelFormat = PIXEL_FORMAT_NV21;
    grp_attr.stFrameRate.s32SrcFrameRate = -1;
    grp_attr.stFrameRate.s32DstFrameRate = -1;
    grp_attr.u8VpssDev = get_vpss_dev_for_input(VPSS_INPUT_ISP);

    CVI_S32 rc = CVI_VPSS_CreateGrp(vpss_grp_, &grp_attr);
    if (rc == CVI_ERR_VPSS_EXIST) {
        CVI_VPSS_DisableChn(vpss_grp_, vpss_chn_);
        CVI_VPSS_StopGrp(vpss_grp_);
        CVI_VPSS_DestroyGrp(vpss_grp_);
        rc = CVI_VPSS_CreateGrp(vpss_grp_, &grp_attr);
    }
    if (!check_rc(rc, "CVI_VPSS_CreateGrp")) {
        return false;
    }
    vpss_created_ = true;

    if (!check_rc(CVI_VPSS_ResetGrp(vpss_grp_), "CVI_VPSS_ResetGrp")) {
        return false;
    }

    VPSS_CHN_ATTR_S chn_attr{};
    chn_attr.u32Width = static_cast<CVI_U32>(width_);
    chn_attr.u32Height = static_cast<CVI_U32>(height_);
    chn_attr.enVideoFormat = VIDEO_FORMAT_LINEAR;
    chn_attr.enPixelFormat = to_cvi_pixel_format(config_.format);
    if (chn_attr.enPixelFormat == PIXEL_FORMAT_MAX) {
        std::cerr << "[ERROR] CviCamera: invalid VPSS output format" << std::endl;
        return false;
    }
    chn_attr.stFrameRate.s32SrcFrameRate = -1;
    chn_attr.stFrameRate.s32DstFrameRate = -1;
    chn_attr.u32Depth = 3;
    chn_attr.bMirror = CVI_FALSE;
    chn_attr.bFlip = CVI_FALSE;
    chn_attr.stAspectRatio.enMode = ASPECT_RATIO_AUTO;

    if (!check_rc(CVI_VPSS_SetChnAttr(vpss_grp_, vpss_chn_, &chn_attr), "CVI_VPSS_SetChnAttr")) {
        return false;
    }

    if (!check_rc(CVI_VPSS_EnableChn(vpss_grp_, vpss_chn_), "CVI_VPSS_EnableChn")) {
        return false;
    }
    vpss_chn_enabled_ = true;

    if (!check_rc(CVI_VPSS_StartGrp(vpss_grp_), "CVI_VPSS_StartGrp")) {
        return false;
    }
    vpss_started_ = true;

    return true;
}

bool CviCamera::bind_vi_vpss() {
    MMF_CHN_S src{};
    src.enModId = CVI_ID_VI;
    src.s32DevId = vi_pipe_;
    src.s32ChnId = vi_chn_;

    MMF_CHN_S dst{};
    dst.enModId = CVI_ID_VPSS;
    dst.s32DevId = vpss_grp_;
    dst.s32ChnId = 0;

    if (!check_rc(CVI_SYS_Bind(&src, &dst), "CVI_SYS_Bind")) {
        return false;
    }
    vi_vpss_bound_ = true;
    return true;
}

void CviCamera::cleanup() {
    if (vi_vpss_bound_) {
        MMF_CHN_S src{};
        src.enModId = CVI_ID_VI;
        src.s32DevId = vi_pipe_;
        src.s32ChnId = vi_chn_;
        MMF_CHN_S dst{};
        dst.enModId = CVI_ID_VPSS;
        dst.s32DevId = vpss_grp_;
        dst.s32ChnId = 0;
        CVI_SYS_UnBind(&src, &dst);
        vi_vpss_bound_ = false;
    }

    if (vpss_chn_enabled_) {
        CVI_VPSS_DisableChn(vpss_grp_, vpss_chn_);
        vpss_chn_enabled_ = false;
    }
    if (vpss_started_) {
        CVI_VPSS_StopGrp(vpss_grp_);
        vpss_started_ = false;
    }
    if (vpss_created_) {
        CVI_VPSS_DestroyGrp(vpss_grp_);
        vpss_created_ = false;
    }

    if (vi_chn_enabled_) {
        CVI_VI_DisableChn(vi_pipe_, vi_chn_);
        CVI_VI_UnRegChnFlipMirrorCallBack(vi_pipe_, vi_chn_);
        vi_chn_enabled_ = false;
    }

    if (isp_thread_running_ || isp_inited_) {
        CVI_ISP_Exit(vi_pipe_);
        if (isp_thread_running_) {
            pthread_join(isp_thread_, nullptr);
            isp_thread_running_ = false;
        }
        isp_inited_ = false;
    }

    if (sensor_callbacks_registered_ && sensor_profile_ && sensor_profile_->sns_obj &&
        sensor_profile_->sns_obj->pfnUnRegisterCallback) {
        ALG_LIB_S ae_lib{};
        ALG_LIB_S awb_lib{};
        ae_lib.s32Id = vi_pipe_;
        awb_lib.s32Id = vi_pipe_;
        std::snprintf(ae_lib.acLibName, sizeof(ae_lib.acLibName), "%s", CVI_AE_LIB_NAME);
        std::snprintf(awb_lib.acLibName, sizeof(awb_lib.acLibName), "%s", CVI_AWB_LIB_NAME);
        sensor_profile_->sns_obj->pfnUnRegisterCallback(vi_pipe_, &ae_lib, &awb_lib);
        sensor_callbacks_registered_ = false;
    }

    if (ae_registered_) {
        ALG_LIB_S ae_lib{};
        ae_lib.s32Id = vi_pipe_;
        std::snprintf(ae_lib.acLibName, sizeof(ae_lib.acLibName), "%s", CVI_AE_LIB_NAME);
        CVI_AE_UnRegister(vi_pipe_, &ae_lib);
        ae_registered_ = false;
    }
    if (awb_registered_) {
        ALG_LIB_S awb_lib{};
        awb_lib.s32Id = vi_pipe_;
        std::snprintf(awb_lib.acLibName, sizeof(awb_lib.acLibName), "%s", CVI_AWB_LIB_NAME);
        CVI_AWB_UnRegister(vi_pipe_, &awb_lib);
        awb_registered_ = false;
    }

    if (vi_pipe_started_) {
        CVI_VI_StopPipe(vi_pipe_);
        vi_pipe_started_ = false;
    }
    if (vi_pipe_created_) {
        CVI_VI_DestroyPipe(vi_pipe_);
        vi_pipe_created_ = false;
    }
    if (vi_dev_enabled_) {
        CVI_VI_DisableDev(vi_dev_);
        vi_dev_enabled_ = false;
    }
    if (vi_opened_) {
        CVI_SYS_VI_Close();
        vi_opened_ = false;
    }

    sensor_.cleanup();
    opened_ = false;
    sensor_profile_ = nullptr;
    width_ = 0;
    height_ = 0;
    fps_ = 0.0;
}
#endif

} // namespace lua_cv
