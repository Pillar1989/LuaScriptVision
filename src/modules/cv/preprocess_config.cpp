/**
 * preprocess_config.cpp - PreprocessConfig validation implementation
 */

#include "cv/preprocess_config.h"

namespace lua_cv {

bool PreprocessConfig::is_valid(std::string* error) const {
    if (input_width == 0 || input_height == 0) {
        if (error) {
            *error = "Input dimensions must be non-zero";
        }
        return false;
    }

    if (output_width == 0 || output_height == 0) {
        if (error) {
            *error = "Output dimensions must be non-zero";
        }
        return false;
    }

    constexpr uint32_t MIN_DIM = 16;
    constexpr uint32_t MAX_DIM = 4096;

    if (input_width < MIN_DIM || input_width > MAX_DIM ||
        input_height < MIN_DIM || input_height > MAX_DIM) {
        if (error) {
            *error = "Input dimensions must be in range [16, 4096]";
        }
        return false;
    }

    if (output_width < MIN_DIM || output_width > MAX_DIM ||
        output_height < MIN_DIM || output_height > MAX_DIM) {
        if (error) {
            *error = "Output dimensions must be in range [16, 4096]";
        }
        return false;
    }

    if (output_format != PIXEL_FORMAT_RGB_888_PLANAR &&
        output_format != PIXEL_FORMAT_BGR_888_PLANAR &&
        output_format != PIXEL_FORMAT_RGB_888 &&
        output_format != PIXEL_FORMAT_BGR_888) {
        if (error) {
            *error = "Output format must be RGB/BGR 888 (planar or packed)";
        }
        return false;
    }

    return true;
}

} // namespace lua_cv
