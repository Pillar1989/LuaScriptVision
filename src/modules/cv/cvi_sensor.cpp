#include "cvi_sensor.h"

#ifdef USE_CVI_MPI

#include <cstring>
#include <unistd.h>
#include <sys/prctl.h>

// Sensor driver objects (external symbols from sensor libraries)
extern "C" {
    extern ISP_SNS_OBJ_S stSnsOv5647_Obj;
    extern ISP_SNS_OBJ_S stSnsGc2053_Obj;
}

namespace lua_cv {

// Default I2C addresses for supported sensors
static const int32_t OV5647_I2C_ADDR = 0x36;
static const int32_t GC2053_I2C_ADDR = 0x3f;

// PQ bin paths for ISP tuning
static const char* PQ_BIN_OV5647 = "/mnt/cfg/param/ov_ov5647_sdr.bin";
static const char* PQ_BIN_GC2053 = "/mnt/cfg/param/gcore_gc2053_sdr.bin";

int32_t get_default_i2c_addr(SensorType type) {
    switch (type) {
        case SensorType::OV5647: return OV5647_I2C_ADDR;
        case SensorType::GC2053: return GC2053_I2C_ADDR;
        default: return 0;
    }
}

ISP_SNS_OBJ_S* get_sensor_obj(SensorType type) {
    switch (type) {
        case SensorType::OV5647: return &stSnsOv5647_Obj;
        case SensorType::GC2053: return &stSnsGc2053_Obj;
        default: return nullptr;
    }
}

CviSensor::CviSensor()
    : sns_obj_(nullptr)
    , detected_type_(SensorType::NONE)
    , vi_dev_(0)
    , vi_pipe_(0)
    , vi_chn_(0)
    , isp_thread_(0)
    , isp_running_(false)
    , initialized_(false)
    , actual_width_(0)
    , actual_height_(0) {
}

CviSensor::~CviSensor() {
    cleanup();
}

const char* CviSensor::get_sensor_name() const {
    switch (detected_type_) {
        case SensorType::OV5647: return "OV5647";
        case SensorType::GC2053: return "GC2053";
        default: return "unknown";
    }
}

bool CviSensor::init() {
    if (initialized_) {
        return true;
    }

    // Use default configuration (auto-detect, max resolution)
    config_ = SensorConfig{};

    // Step 1: Open VI system
    CVI_S32 rc = CVI_SYS_VI_Open();
    if (rc != CVI_SUCCESS) {
        return false;
    }

    // Step 2: Probe sensor (auto-detect)
    if (!probe_sensor()) {
        CVI_SYS_VI_Close();
        return false;
    }

    // Step 3: Initialize MIPI
    if (!init_mipi()) {
        CVI_SYS_VI_Close();
        return false;
    }

    // Step 4: Initialize VI device
    if (!init_vi_dev()) {
        cleanup_mipi();
        CVI_SYS_VI_Close();
        return false;
    }

    // Step 5: Initialize VI pipe
    if (!init_vi_pipe()) {
        cleanup_vi_dev();
        cleanup_mipi();
        CVI_SYS_VI_Close();
        return false;
    }

    // Step 6: Initialize ISP
    if (!init_isp()) {
        cleanup_vi_pipe();
        cleanup_vi_dev();
        cleanup_mipi();
        CVI_SYS_VI_Close();
        return false;
    }

    // Step 7: Start ISP thread
    if (!start_isp_thread()) {
        cleanup_vi_pipe();
        cleanup_vi_dev();
        cleanup_mipi();
        CVI_SYS_VI_Close();
        return false;
    }

    // Step 8: Initialize VI channel
    if (!init_vi_chn()) {
        stop_isp_thread();
        cleanup_vi_pipe();
        cleanup_vi_dev();
        cleanup_mipi();
        CVI_SYS_VI_Close();
        return false;
    }

    initialized_ = true;
    return true;
}

void CviSensor::cleanup() {
    if (!initialized_) {
        return;
    }

    stop_isp_thread();
    cleanup_vi_chn();
    cleanup_vi_pipe();
    cleanup_vi_dev();
    cleanup_mipi();
    CVI_SYS_VI_Close();

    initialized_ = false;
    detected_type_ = SensorType::NONE;
    sns_obj_ = nullptr;
}

bool CviSensor::probe_sensor() {
    // List of sensors to try (in priority order)
    SensorType sensors_to_try[2];
    int num_sensors = 0;

    if (config_.type == SensorType::AUTO) {
        // Auto-detect: try OV5647 first, then GC2053
        sensors_to_try[num_sensors++] = SensorType::OV5647;
        sensors_to_try[num_sensors++] = SensorType::GC2053;
    } else {
        sensors_to_try[num_sensors++] = config_.type;
    }

    for (int i = 0; i < num_sensors; i++) {
        SensorType type = sensors_to_try[i];
        ISP_SNS_OBJ_S* obj = get_sensor_obj(type);

        if (obj == nullptr) {
            continue;
        }

        // Configure I2C bus
        ISP_SNS_COMMBUS_U bus_info;
        std::memset(&bus_info, 0, sizeof(bus_info));
        bus_info.s8I2cDev = static_cast<CVI_S8>(config_.bus_id);

        if (obj->pfnSetBusInfo) {
            CVI_S32 rc = obj->pfnSetBusInfo(vi_pipe_, bus_info);
            if (rc != CVI_SUCCESS) {
                continue;
            }
        }

        // Patch I2C address
        int32_t addr = config_.i2c_addr ? config_.i2c_addr : get_default_i2c_addr(type);
        if (obj->pfnPatchI2cAddr) {
            obj->pfnPatchI2cAddr(addr);
        }

        // Probe sensor
        if (obj->pfnSnsProbe) {
            CVI_S32 rc = obj->pfnSnsProbe(vi_pipe_);
            if (rc == CVI_SUCCESS) {
                detected_type_ = type;
                sns_obj_ = obj;

                // Initialize sensor with RX attributes
                RX_INIT_ATTR_S rx_init;
                std::memset(&rx_init, 0, sizeof(rx_init));
                rx_init.MipiDev = config_.mipi_dev;
                rx_init.stMclkAttr.bMclkEn = config_.mclk_enable ? CVI_TRUE : CVI_FALSE;
                rx_init.stMclkAttr.u8Mclk = config_.mclk_id;

                for (int j = 0; j < 5; j++) {
                    rx_init.as16LaneId[j] = config_.lane_id[j];
                    rx_init.as8PNSwap[j] = config_.pn_swap[j];
                }

                if (obj->pfnPatchRxAttr) {
                    obj->pfnPatchRxAttr(&rx_init);
                }

                // Set ISP init attributes
                ISP_INIT_ATTR_S isp_init;
                std::memset(&isp_init, 0, sizeof(isp_init));
                isp_init.u16UseHwSync = CVI_FALSE;

                if (obj->pfnSetInit) {
                    obj->pfnSetInit(vi_pipe_, &isp_init);
                }

                // Register AE/AWB callbacks
                ALG_LIB_S ae_lib, awb_lib;
                ae_lib.s32Id = vi_pipe_;
                awb_lib.s32Id = vi_pipe_;
                std::strcpy(ae_lib.acLibName, CVI_AE_LIB_NAME);
                std::strcpy(awb_lib.acLibName, CVI_AWB_LIB_NAME);

                if (obj->pfnRegisterCallback) {
                    obj->pfnRegisterCallback(vi_pipe_, &ae_lib, &awb_lib);
                }

                // Set sensor image mode
                if (obj->pfnExpSensorCb) {
                    ISP_SENSOR_EXP_FUNC_S exp_func;
                    obj->pfnExpSensorCb(&exp_func);

                    exp_func.pfn_cmos_sensor_global_init(vi_pipe_);

                    ISP_CMOS_SENSOR_IMAGE_MODE_S image_mode;
                    std::memset(&image_mode, 0, sizeof(image_mode));
                    image_mode.u16Width = config_.width;
                    image_mode.u16Height = config_.height;
                    image_mode.f32Fps = static_cast<CVI_FLOAT>(config_.framerate);

                    exp_func.pfn_cmos_set_image_mode(vi_pipe_, &image_mode);
                    exp_func.pfn_cmos_set_wdr_mode(vi_pipe_, WDR_MODE_NONE);
                }

                actual_width_ = config_.width;
                actual_height_ = config_.height;

                return true;
            }
        }
    }

    return false;
}

bool CviSensor::init_mipi() {
    if (sns_obj_ == nullptr) {
        return false;
    }

    SNS_COMBO_DEV_ATTR_S combo_attr;
    std::memset(&combo_attr, 0, sizeof(combo_attr));

    if (sns_obj_->pfnGetRxAttr) {
        CVI_S32 rc = sns_obj_->pfnGetRxAttr(vi_pipe_, &combo_attr);
        if (rc != CVI_SUCCESS) {
            return false;
        }
    }

    CVI_S32 rc;

    // Reset sensor and MIPI
    rc = CVI_MIPI_SetSensorReset(config_.mipi_dev, 1);
    if (rc != CVI_SUCCESS) return false;

    rc = CVI_MIPI_SetMipiReset(config_.mipi_dev, 1);
    if (rc != CVI_SUCCESS) return false;

    // Set MIPI attributes
    rc = CVI_MIPI_SetMipiAttr(config_.mipi_dev, &combo_attr);
    if (rc != CVI_SUCCESS) return false;

    // Enable sensor clock
    rc = CVI_MIPI_SetSensorClock(config_.mipi_dev, 1);
    if (rc != CVI_SUCCESS) return false;

    // Release sensor reset
    usleep(20);
    rc = CVI_MIPI_SetSensorReset(config_.mipi_dev, 0);
    if (rc != CVI_SUCCESS) return false;

    return true;
}

bool CviSensor::init_vi_dev() {
    VI_DEV_ATTR_S dev_attr;
    std::memset(&dev_attr, 0, sizeof(dev_attr));

    dev_attr.enIntfMode = VI_MODE_MIPI;
    dev_attr.enWorkMode = VI_WORK_MODE_1Multiplex;
    dev_attr.enScanMode = VI_SCAN_PROGRESSIVE;
    dev_attr.as32AdChnId[0] = -1;
    dev_attr.stSize.u32Width = config_.width;
    dev_attr.stSize.u32Height = config_.height;
    dev_attr.stWDRAttr.enWDRMode = WDR_MODE_NONE;

    CVI_S32 rc = CVI_VI_SetDevAttr(vi_dev_, &dev_attr);
    if (rc != CVI_SUCCESS) {
        return false;
    }

    rc = CVI_VI_EnableDev(vi_dev_);
    if (rc != CVI_SUCCESS) {
        return false;
    }

    return true;
}

bool CviSensor::init_vi_pipe() {
    VI_PIPE_ATTR_S pipe_attr;
    std::memset(&pipe_attr, 0, sizeof(pipe_attr));

    pipe_attr.enPipeBypassMode = VI_PIPE_BYPASS_NONE;
    pipe_attr.bYuvSkip = CVI_FALSE;
    pipe_attr.bIspBypass = CVI_FALSE;
    pipe_attr.u32MaxW = config_.width;
    pipe_attr.u32MaxH = config_.height;
    pipe_attr.enPixFmt = PIXEL_FORMAT_NV21;
    pipe_attr.enBitWidth = DATA_BITWIDTH_8;
    pipe_attr.stFrameRate.s32SrcFrameRate = -1;
    pipe_attr.stFrameRate.s32DstFrameRate = -1;
    pipe_attr.bNrEn = CVI_TRUE;
    pipe_attr.enCompressMode = COMPRESS_MODE_TILE;

    CVI_S32 rc = CVI_VI_CreatePipe(vi_pipe_, &pipe_attr);
    if (rc != CVI_SUCCESS) {
        return false;
    }

    rc = CVI_VI_StartPipe(vi_pipe_);
    if (rc != CVI_SUCCESS) {
        CVI_VI_DestroyPipe(vi_pipe_);
        return false;
    }

    return true;
}

bool CviSensor::init_isp() {
    CVI_S32 rc;

    // Register AE algorithm
    ALG_LIB_S ae_lib;
    ae_lib.s32Id = vi_pipe_;
    std::strcpy(ae_lib.acLibName, CVI_AE_LIB_NAME);
    rc = CVI_AE_Register(vi_pipe_, &ae_lib);
    if (rc != CVI_SUCCESS) {
        return false;
    }

    // Register AWB algorithm
    ALG_LIB_S awb_lib;
    awb_lib.s32Id = vi_pipe_;
    std::strcpy(awb_lib.acLibName, CVI_AWB_LIB_NAME);
    rc = CVI_AWB_Register(vi_pipe_, &awb_lib);
    if (rc != CVI_SUCCESS) {
        CVI_AE_UnRegister(vi_pipe_, &ae_lib);
        return false;
    }

    // Set ISP bind attributes
    ISP_BIND_ATTR_S bind_attr;
    std::memset(&bind_attr, 0, sizeof(bind_attr));
    bind_attr.sensorId = 0;
    std::snprintf(bind_attr.stAeLib.acLibName, sizeof(bind_attr.stAeLib.acLibName), "%s", CVI_AE_LIB_NAME);
    bind_attr.stAeLib.s32Id = vi_pipe_;
    std::snprintf(bind_attr.stAwbLib.acLibName, sizeof(bind_attr.stAwbLib.acLibName), "%s", CVI_AWB_LIB_NAME);
    bind_attr.stAwbLib.s32Id = vi_pipe_;

    rc = CVI_ISP_SetBindAttr(vi_pipe_, &bind_attr);
    if (rc != CVI_SUCCESS) {
        CVI_AWB_UnRegister(vi_pipe_, &awb_lib);
        CVI_AE_UnRegister(vi_pipe_, &ae_lib);
        return false;
    }

    // Initialize ISP memory
    rc = CVI_ISP_MemInit(vi_pipe_);
    if (rc != CVI_SUCCESS) {
        return false;
    }

    // Set ISP public attributes
    ISP_PUB_ATTR_S pub_attr;
    std::memset(&pub_attr, 0, sizeof(pub_attr));
    pub_attr.stWndRect.s32X = 0;
    pub_attr.stWndRect.s32Y = 0;
    pub_attr.stWndRect.u32Width = config_.width;
    pub_attr.stWndRect.u32Height = config_.height;
    pub_attr.stSnsSize.u32Width = config_.width;
    pub_attr.stSnsSize.u32Height = config_.height;
    pub_attr.f32FrameRate = static_cast<CVI_FLOAT>(config_.framerate);
    pub_attr.enWDRMode = WDR_MODE_NONE;
    pub_attr.enBayer = BAYER_RGGB;  // Default, sensor driver may override

    rc = CVI_ISP_SetPubAttr(vi_pipe_, &pub_attr);
    if (rc != CVI_SUCCESS) {
        return false;
    }

    // Initialize ISP
    rc = CVI_ISP_Init(vi_pipe_);
    if (rc != CVI_SUCCESS) {
        return false;
    }

    return true;
}

bool CviSensor::start_isp_thread() {
    isp_running_ = true;

    pthread_attr_t attr;
    struct sched_param param;
    param.sched_priority = 80;

    pthread_attr_init(&attr);
    pthread_attr_setschedpolicy(&attr, SCHED_RR);
    pthread_attr_setschedparam(&attr, &param);
    pthread_attr_setinheritsched(&attr, PTHREAD_EXPLICIT_SCHED);

    int ret = pthread_create(&isp_thread_, &attr, isp_thread_func, this);
    pthread_attr_destroy(&attr);

    if (ret != 0) {
        isp_running_ = false;
        return false;
    }

    return true;
}

void CviSensor::stop_isp_thread() {
    if (!isp_running_) {
        return;
    }

    CVI_ISP_Exit(vi_pipe_);
    isp_running_ = false;

    if (isp_thread_) {
        pthread_join(isp_thread_, nullptr);
        isp_thread_ = 0;
    }

    // Unregister AE/AWB
    ALG_LIB_S ae_lib, awb_lib;
    ae_lib.s32Id = vi_pipe_;
    awb_lib.s32Id = vi_pipe_;
    std::strcpy(ae_lib.acLibName, CVI_AE_LIB_NAME);
    std::strcpy(awb_lib.acLibName, CVI_AWB_LIB_NAME);

    if (sns_obj_ && sns_obj_->pfnUnRegisterCallback) {
        sns_obj_->pfnUnRegisterCallback(vi_pipe_, &ae_lib, &awb_lib);
    }

    CVI_AE_UnRegister(vi_pipe_, &ae_lib);
    CVI_AWB_UnRegister(vi_pipe_, &awb_lib);
}

void* CviSensor::isp_thread_func(void* arg) {
    CviSensor* self = static_cast<CviSensor*>(arg);
    self->isp_run_loop();
    return nullptr;
}

void CviSensor::isp_run_loop() {
    prctl(PR_SET_NAME, "ISP_RUN", 0, 0, 0);
    CVI_ISP_Run(vi_pipe_);
}

bool CviSensor::init_vi_chn() {
    VI_CHN_ATTR_S chn_attr;
    std::memset(&chn_attr, 0, sizeof(chn_attr));

    chn_attr.stSize.u32Width = config_.width;
    chn_attr.stSize.u32Height = config_.height;
    chn_attr.enPixelFormat = PIXEL_FORMAT_NV21;
    chn_attr.enDynamicRange = DYNAMIC_RANGE_SDR8;
    chn_attr.enVideoFormat = VIDEO_FORMAT_LINEAR;
    chn_attr.enCompressMode = COMPRESS_MODE_TILE;
    chn_attr.bMirror = config_.mirror ? CVI_TRUE : CVI_FALSE;
    chn_attr.bFlip = config_.flip ? CVI_TRUE : CVI_FALSE;
    chn_attr.u32Depth = 3;  // Buffer queue depth
    chn_attr.stFrameRate.s32SrcFrameRate = -1;
    chn_attr.stFrameRate.s32DstFrameRate = -1;

    CVI_S32 rc = CVI_VI_SetChnAttr(vi_pipe_, vi_chn_, &chn_attr);
    if (rc != CVI_SUCCESS) {
        return false;
    }

    // Register mirror/flip callback if sensor supports it
    if (sns_obj_ && sns_obj_->pfnMirrorFlip) {
        CVI_VI_RegChnFlipMirrorCallBack(vi_pipe_, vi_chn_,
            reinterpret_cast<void*>(sns_obj_->pfnMirrorFlip));
    }

    rc = CVI_VI_EnableChn(vi_pipe_, vi_chn_);
    if (rc != CVI_SUCCESS) {
        return false;
    }

    return true;
}

void CviSensor::cleanup_vi_chn() {
    CVI_VI_UnRegChnFlipMirrorCallBack(vi_pipe_, vi_chn_);
    CVI_VI_DisableChn(vi_pipe_, vi_chn_);
}

void CviSensor::cleanup_vi_pipe() {
    CVI_VI_StopPipe(vi_pipe_);
    CVI_VI_DestroyPipe(vi_pipe_);
}

void CviSensor::cleanup_vi_dev() {
    CVI_VI_DisableDev(vi_dev_);
}

void CviSensor::cleanup_mipi() {
    // MIPI cleanup is handled by VI_Close
}

} // namespace lua_cv

#endif // USE_CVI_MPI
