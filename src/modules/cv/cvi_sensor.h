#pragma once

#include <cstdint>

#include "cv_types.h"

#ifdef USE_CVI_CAMERA
#include <cvi_sns_ctrl.h>
#include <cvi_comm_isp.h>
#include <linux/cvi_comm_video.h>
#endif

namespace lua_cv {

#ifdef USE_CVI_CAMERA
class CviSensor {
public:
    struct Profile {
        const char* name = nullptr;
        ISP_SNS_OBJ_S* sns_obj = nullptr;
        int width = 0;
        int height = 0;
        int fps = 0;
        int bus_id = 0;
        int i2c_addr = 0;
        int mipi_dev = 0;
        bool mclk_en = false;
        uint8_t mclk = 0;
        ISP_SNS_MIRRORFLIP_TYPE_E orientation = ISP_SNS_NORMAL;
        bool hw_sync = false;
        int16_t lane_id[5]{};
        int8_t pn_swap[5]{};
        WDR_MODE_E wdr_mode = WDR_MODE_NONE;
        BAYER_FORMAT_E bayer = BAYER_FORMAT_BG;
    };

    CviSensor() = default;

    bool init();
    void cleanup();

    const char* get_sensor_name() const;
    int get_width() const;
    int get_height() const;
    int get_fps() const;

    const Profile* profile() const { return profile_; }

private:
    const Profile* profile_ = nullptr;
    bool initialized_ = false;
};
#endif

} // namespace lua_cv
