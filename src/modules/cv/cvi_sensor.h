#pragma once

/**
 * cvi_sensor.h - SG200X Sensor Driver Abstraction Layer
 *
 * Provides unified interface for sensor initialization with auto-detection.
 * Supports: OV5647, GC2053 (auto-detect based on I2C probe)
 *
 * Initialization sequence:
 * 1. CVI_SYS_VI_Open()
 * 2. Sensor_Start (sensor driver + MIPI)
 * 3. Dev_Start (VI device)
 * 4. Pipe_Start (VI pipe)
 * 5. ISP_Init (AE/AWB registration)
 * 6. ISP_Start (ISP thread)
 * 7. Chn_Start (VI channel)
 */

#ifdef USE_CVI_MPI

#include <cstdint>
#include <string>

// CVI SDK headers
#include <cvi_sys.h>
#include <cvi_vi.h>
#include <cvi_isp.h>
#include <cvi_mipi.h>
#include <cvi_ae.h>
#include <cvi_awb.h>
#include <pthread.h>

namespace lua_cv {

/**
 * Supported sensor types
 */
enum class SensorType {
    NONE = 0,
    OV5647,    // OmniVision 5MP sensor
    GC2053,    // Galaxycore 2MP sensor
    AUTO       // Auto-detect (default)
};

/**
 * Internal sensor configuration (not exposed to Lua)
 * Uses default values for auto-detection and max resolution
 */
struct SensorConfig {
    SensorType type = SensorType::AUTO;    // Auto-detect sensor
    int32_t bus_id = 2;                     // I2C bus ID (fixed)
    int32_t i2c_addr = 0;                   // 0 = use default for sensor type
    int32_t mipi_dev = 0;                   // MIPI device ID (fixed)
    int16_t lane_id[5] = {2, 0, 3, -1, -1}; // MIPI lane mapping (fixed)
    int8_t pn_swap[5] = {0, 0, 0, 0, 0};    // PN swap (fixed)
    bool mclk_enable = true;                // Clock enable (fixed)
    uint8_t mclk_id = 0;                    // Clock ID (fixed)
    uint32_t width = 1920;                  // Default max resolution
    uint32_t height = 1080;                 // Default max resolution
    int32_t framerate = 30;                 // Default framerate
    bool mirror = false;                    // No mirror by default
    bool flip = false;                      // No flip by default
};

/**
 * CviSensor - Sensor driver management
 *
 * Handles sensor auto-detection, initialization, and ISP setup.
 * Thread-safe ISP loop runs in background after Start().
 */
class CviSensor {
public:
    CviSensor();
    ~CviSensor();

    // Disable copy
    CviSensor(const CviSensor&) = delete;
    CviSensor& operator=(const CviSensor&) = delete;

    /**
     * Probe and initialize sensor with auto-detection
     *
     * Uses default configuration:
     * - Auto-detect sensor type (OV5647 → GC2053 priority)
     * - Max resolution: 1920x1080
     * - Framerate: 30fps
     *
     * @return true if sensor initialized successfully
     *
     * Performs:
     * 1. Sensor probe (auto-detect)
     * 2. MIPI initialization
     * 3. VI device/pipe/channel setup
     * 4. ISP initialization (AE/AWB)
     * 5. ISP thread start
     */
    bool init();

    /**
     * Stop sensor and cleanup resources
     */
    void cleanup();

    /**
     * Get detected sensor type
     */
    SensorType get_sensor_type() const { return detected_type_; }

    /**
     * Get sensor name string
     */
    const char* get_sensor_name() const;

    /**
     * Get VI pipe ID
     */
    VI_PIPE get_vi_pipe() const { return vi_pipe_; }

    /**
     * Get VI channel ID
     */
    VI_CHN get_vi_chn() const { return vi_chn_; }

    /**
     * Check if sensor is initialized
     */
    bool is_initialized() const { return initialized_; }

    /**
     * Get actual sensor resolution
     */
    uint32_t get_width() const { return actual_width_; }
    uint32_t get_height() const { return actual_height_; }

private:
    // Initialization steps
    bool probe_sensor();
    bool init_mipi();
    bool init_vi_dev();
    bool init_vi_pipe();
    bool init_isp();
    bool start_isp_thread();
    bool init_vi_chn();

    // Cleanup steps
    void stop_isp_thread();
    void cleanup_vi_chn();
    void cleanup_vi_pipe();
    void cleanup_vi_dev();
    void cleanup_mipi();

    // ISP thread function
    static void* isp_thread_func(void* arg);
    void isp_run_loop();

    // Sensor object (from sensor driver library)
    ISP_SNS_OBJ_S* sns_obj_;

    // Configuration
    SensorConfig config_;
    SensorType detected_type_;

    // VI resources
    VI_DEV vi_dev_;
    VI_PIPE vi_pipe_;
    VI_CHN vi_chn_;

    // ISP thread
    pthread_t isp_thread_;
    bool isp_running_;

    // State
    bool initialized_;
    uint32_t actual_width_;
    uint32_t actual_height_;
};

/**
 * Get default I2C address for sensor type
 */
int32_t get_default_i2c_addr(SensorType type);

/**
 * Get sensor object for sensor type
 * (requires linking with sensor driver library)
 */
ISP_SNS_OBJ_S* get_sensor_obj(SensorType type);

} // namespace lua_cv

#else // !USE_CVI_MPI

// Stub when SDK not available
namespace lua_cv {

enum class SensorType { NONE, OV5647, GC2053, AUTO };

class CviSensor {
public:
    bool init() { return false; }
    void cleanup() {}
    SensorType get_sensor_type() const { return SensorType::NONE; }
    const char* get_sensor_name() const { return "none"; }
    bool is_initialized() const { return false; }
    uint32_t get_width() const { return 0; }
    uint32_t get_height() const { return 0; }
};

} // namespace lua_cv

#endif // USE_CVI_MPI
