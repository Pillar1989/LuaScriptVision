#pragma once

#include <cstdint>

#include "cv_camera.h"
#include "cvi_sensor.h"

#ifdef USE_CVI_CAMERA
#include <pthread.h>
#include <linux/cvi_comm_vpss.h>
#include <linux/cvi_comm_vi.h>
#include <linux/cvi_comm_video.h>
#endif

namespace lua_cv {

#ifdef USE_CVI_CAMERA
class CviCamera : public CvCamera {
public:
    struct Config {
        int width = 0;
        int height = 0;
        double fps = 30.0;
        PixelFormat format = PixelFormat::NV21;
    };

    CviCamera();
    explicit CviCamera(const Config& config);
    ~CviCamera() override;

    CviCamera(const CviCamera&) = delete;
    CviCamera& operator=(const CviCamera&) = delete;
    CviCamera(CviCamera&&) = delete;
    CviCamera& operator=(CviCamera&&) = delete;

    bool open() override;
    bool read(Frame& frame) override;
    bool read(Frame& frame, int timeout_ms);
    bool wait_for_ready(int timeout_ms);
    void release() override;

    int width() const override;
    int height() const override;
    double fps() const override;
    bool is_opened() const override;

    const char* get_sensor_name() const;

private:
    bool read_internal(Frame& frame, int timeout_ms, bool log_error);
    bool init_system();
    bool start_vi();
    bool start_sensor();
    bool start_mipi();
    bool start_vi_dev();
    bool start_vi_pipe();
    bool init_isp();
    bool start_isp();
    bool start_vi_channel();
    bool init_vpss();
    bool bind_vi_vpss();
    void cleanup();

    Config config_;
    CviSensor sensor_;
    const CviSensor::Profile* sensor_profile_ = nullptr;

    int width_ = 0;
    int height_ = 0;
    double fps_ = 0.0;

    bool opened_ = false;
    bool vi_opened_ = false;
    bool vi_dev_enabled_ = false;
    bool vi_pipe_started_ = false;
    bool vi_pipe_created_ = false;
    bool vi_chn_enabled_ = false;
    bool vpss_created_ = false;
    bool vpss_started_ = false;
    bool vpss_chn_enabled_ = false;
    bool vi_vpss_bound_ = false;
    bool isp_inited_ = false;
    bool isp_thread_running_ = false;
    bool sensor_callbacks_registered_ = false;
    bool ae_registered_ = false;
    bool awb_registered_ = false;

    VPSS_GRP vpss_grp_ = 0;
    VPSS_CHN vpss_chn_ = 0;
    VI_DEV vi_dev_ = 0;
    VI_PIPE vi_pipe_ = 0;
    VI_CHN vi_chn_ = 0;

    pthread_t isp_thread_{};
};
#endif

} // namespace lua_cv
