/**
 * preprocess_config.h - High-level preprocessing configuration
 *
 * Provides declarative preprocessing configuration for Model Node (Phase 3)
 * and future Lua script bindings. Designed as POD-like structure for easy
 * Lua interaction via LuaIntf.
 */

#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <linux/cvi_comm_video.h>

namespace lua_cv {

struct PreprocessConfig {
    uint32_t input_width = 0;
    uint32_t input_height = 0;
    uint32_t output_width = 640;
    uint32_t output_height = 640;
    PIXEL_FORMAT_E output_format = PIXEL_FORMAT_RGB_888_PLANAR;
    bool letterbox = true;
    uint8_t pad_value = 114;
    bool normalized = false;
    std::array<float, 3> mean{0.0f, 0.0f, 0.0f};
    std::array<float, 3> scale{1.0f, 1.0f, 1.0f};

    /**
     * Validate configuration parameters.
     * @param error Optional error message output
     * @return true if configuration is valid, false otherwise
     */
    bool is_valid(std::string* error = nullptr) const;
};

} // namespace lua_cv
