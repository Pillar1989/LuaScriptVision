#include "cvi_sensor.h"

#include <iostream>

#ifdef USE_CVI_CAMERA
#include <linux/cvi_errno.h>
#endif

namespace lua_cv {

#ifdef USE_CVI_CAMERA
} // namespace lua_cv

#if defined(__GNUC__)
extern "C" ISP_SNS_OBJ_S stSnsSC530AI_2L_Obj __attribute__((weak));
#else
extern "C" ISP_SNS_OBJ_S stSnsSC530AI_2L_Obj;
#endif

namespace lua_cv {
namespace {

ISP_SNS_OBJ_S* get_sc530ai_obj() {
    return &stSnsSC530AI_2L_Obj;
}

const CviSensor::Profile kSensorProfiles[] = {
    {
        "OV5647",
        &stSnsOv5647_Obj,
        1920,
        1080,
        30,
        2,
        0x36,
        0,
        true,
        0,
        ISP_SNS_MIRROR,
        false,
        {2, 0, 3, -1, -1},
        {0, 0, 0, 0, 0},
        WDR_MODE_NONE,
        BAYER_FORMAT_BG,
    },
    {
        "SC530AI_2L",
        get_sc530ai_obj(),
        2880,
        1620,
        30,
        2,
        0x30,
        0,
        true,
        0,
        ISP_SNS_MIRROR_FLIP,
        false,
        {2, 0, 3, -1, -1},
        {0, 0, 0, 0, 0},
        WDR_MODE_NONE,
        BAYER_FORMAT_BG,
    },
    {
        "GC2053",
        &stSnsGc2053_Obj,
        1920,
        1080,
        30,
        2,
        0x3f,
        0,
        true,
        0,
        ISP_SNS_NORMAL,
        false,
        {2, 0, 3, -1, -1},
        {0, 0, 0, 0, 0},
        WDR_MODE_NONE,
        BAYER_FORMAT_RG,
    },
};

size_t sensor_profile_count() {
    return sizeof(kSensorProfiles) / sizeof(kSensorProfiles[0]);
}
} // namespace

bool CviSensor::init() {
    profile_ = nullptr;
    initialized_ = false;

    for (size_t i = 0; i < sensor_profile_count(); ++i) {
        const Profile& candidate = kSensorProfiles[i];
        if (!candidate.sns_obj) {
            continue;
        }

        ISP_SNS_OBJ_S* sns_obj = candidate.sns_obj;
        ISP_SNS_COMMBUS_U bus_info{};
        bus_info.s8I2cDev = (candidate.bus_id >= 0) ? static_cast<CVI_S8>(candidate.bus_id) : 0x3;

        if (sns_obj->pfnSetBusInfo) {
            CVI_S32 rc = sns_obj->pfnSetBusInfo(0, bus_info);
            if (rc != CVI_SUCCESS) {
                continue;
            }
        }

        if (sns_obj->pfnPatchI2cAddr) {
            sns_obj->pfnPatchI2cAddr(candidate.i2c_addr);
        }

        if (sns_obj->pfnSnsProbe) {
            CVI_S32 rc = sns_obj->pfnSnsProbe(0);
            if (rc != CVI_SUCCESS) {
                continue;
            }
        } else {
            continue;
        }

        profile_ = &candidate;
        initialized_ = true;
        break;
    }

    if (!initialized_) {
        std::cerr << "[ERROR] CviSensor: no supported sensor detected" << std::endl;
    }

    return initialized_;
}

void CviSensor::cleanup() {
    profile_ = nullptr;
    initialized_ = false;
}

const char* CviSensor::get_sensor_name() const {
    return profile_ ? profile_->name : "unknown";
}

int CviSensor::get_width() const {
    return profile_ ? profile_->width : 0;
}

int CviSensor::get_height() const {
    return profile_ ? profile_->height : 0;
}

int CviSensor::get_fps() const {
    return profile_ ? profile_->fps : 0;
}
#endif

} // namespace lua_cv
