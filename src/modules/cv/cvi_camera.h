#pragma once

#include "cv_camera.h"
#include "cv_types.h"

#ifdef USE_CVI_MPI
#ifdef USE_CVI_CAMERA

#include "cvi_sensor.h"
#include <linux/cvi_comm_video.h>
#include <cvi_vi.h>
#include <cvi_vpss.h>
#include <cvi_vb.h>
#include <cvi_sys.h>

namespace lua_cv {

/**
 * CviCamera - Hardware camera using SG200X VI/VPSS pipeline
 *
 * This camera implementation uses CviSensor for sensor/VI/ISP initialization,
 * then binds to VPSS for frame output with zero-copy support.
 *
 * Resource allocation:
 * - VI_DEV: 0, VI_PIPE: 0, VI_CHN: 0 (managed by CviSensor)
 * - VPSS_GRP: 0, VPSS_CHN: 0 (managed by CviCamera)
 * - VB Pool: 8MB × 3 blocks
 *
 * Pipeline:
 *   Camera Sensor → VI → ISP → VPSS → Frame (VIDEO_FRAME_INFO_S with physical address)
 *
 * Supported sensors: OV5647, GC2053 (auto-detected by CviSensor)
 */
class CviCamera : public CvCamera {
public:
    /**
     * Camera configuration
     * Note: Sensor detection and resolution are handled by CviSensor internally
     */
    struct Config {
        PixelFormat format; // Output pixel format

        Config() : format(PixelFormat::NV21) {}
    };

    explicit CviCamera(const Config& config = Config());
    ~CviCamera() override;

    // Disable copy and move
    CviCamera(const CviCamera&) = delete;
    CviCamera& operator=(const CviCamera&) = delete;
    CviCamera(CviCamera&&) = delete;
    CviCamera& operator=(CviCamera&&) = delete;

    // CvCamera interface implementation
    bool open() override;
    bool read(Frame& frame) override;
    void release() override;

    int width() const override { return sensor_.get_width(); }
    int height() const override { return sensor_.get_height(); }
    double fps() const override { return 30.0; }  // Default from CviSensor
    bool is_opened() const override { return opened_; }

    /**
     * Get detected sensor name
     */
    const char* get_sensor_name() const { return sensor_.get_sensor_name(); }

private:
    // ========== Initialization Steps ==========

    /**
     * Initialize VB (Video Buffer) pool
     * Allocates memory pool for VPSS zero-copy transfer
     */
    void init_vb_pool();

    /**
     * Initialize VPSS (Video Post-Processing) module
     * Configures VPSS group and channel for frame output
     */
    void init_vpss_module();

    /**
     * Bind VI to VPSS
     * Creates zero-copy data path from VI (CviSensor) to VPSS
     */
    void bind_vi_vpss();

    /**
     * Cleanup all resources
     * Releases VPSS/VB resources in correct order
     */
    void cleanup();

    /**
     * Convert PixelFormat to CVI PIXEL_FORMAT_E
     */
    PIXEL_FORMAT_E to_cvi_pixel_format(PixelFormat format) const;

    // ========== Configuration and State ==========

    Config config_;
    bool opened_;

    // Sensor driver (handles VI/ISP)
    CviSensor sensor_;

    // VPSS resources
    VPSS_GRP vpss_grp_;
    VPSS_CHN vpss_chn_;

    // VB pool
    VB_POOL vb_pool_;
};

} // namespace lua_cv

#endif // USE_CVI_CAMERA
#endif // USE_CVI_MPI
