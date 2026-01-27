/**
 * vpss_capability.cpp - VPSS constraint validation implementation
 */

#include "vpss_capability.h"

bool VpssCapability::validate_config(const lua_cv::PreprocessConfig& config,
                                      std::string* error) {
    if (!config.is_valid(error)) {
        return false;
    }

    if (!check_dimension_limits(config.input_width, config.input_height)) {
        if (error) {
            *error = "Input dimensions exceed VPSS limits [16, 4096]";
        }
        return false;
    }

    if (!check_dimension_limits(config.output_width, config.output_height)) {
        if (error) {
            *error = "Output dimensions exceed VPSS limits [16, 4096]";
        }
        return false;
    }

    if (!check_color_conversion(config.output_format)) {
        if (error) {
            *error = "Unsupported color format for VPSS";
        }
        return false;
    }

    if (config.normalized) {
        if (error) {
            *error = "VPSS does not support normalize operation (handled by TPU quantization)";
        }
        return false;
    }

    return true;
}

bool VpssCapability::validate_ops(const std::vector<lua_cv::PreprocessOp>& ops,
                                   std::string* error) {
    if (!check_crop_before_resize(ops)) {
        if (error) {
            *error = "VPSS requires CROP operations to occur before RESIZE";
        }
        return false;
    }

    if (!check_single_resize(ops)) {
        if (error) {
            *error = "VPSS allows only one RESIZE operation per pipeline";
        }
        return false;
    }

    for (const auto& op : ops) {
        if (op.type == lua_cv::PreprocessOp::COLOR) {
            if (!check_color_conversion(op.params.color.format)) {
                if (error) {
                    *error = "Unsupported color format for VPSS";
                }
                return false;
            }
        } else if (op.type == lua_cv::PreprocessOp::RESIZE) {
            if (!check_dimension_limits(op.params.resize.width, op.params.resize.height)) {
                if (error) {
                    *error = "Resize dimensions exceed VPSS limits [16, 4096]";
                }
                return false;
            }
        } else if (op.type == lua_cv::PreprocessOp::CROP) {
            if (!check_dimension_limits(op.params.crop.width, op.params.crop.height)) {
                if (error) {
                    *error = "Crop dimensions exceed VPSS limits [16, 4096]";
                }
                return false;
            }
        }
    }

    return true;
}

std::vector<lua_cv::PreprocessOp> VpssCapability::config_to_ops(
    const lua_cv::PreprocessConfig& config) {
    std::vector<lua_cv::PreprocessOp> ops;

    if (config.letterbox) {
        float scale = std::min(
            static_cast<float>(config.output_width) / config.input_width,
            static_cast<float>(config.output_height) / config.input_height
        );
        uint32_t scaled_w = static_cast<uint32_t>(config.input_width * scale);
        uint32_t scaled_h = static_cast<uint32_t>(config.input_height * scale);

        ops.push_back(lua_cv::PreprocessOp::resize(scaled_w, scaled_h, true));

        uint32_t pad_x = (config.output_width - scaled_w) / 2;
        uint32_t pad_y = (config.output_height - scaled_h) / 2;
        uint32_t pad_right = config.output_width - scaled_w - pad_x;
        uint32_t pad_bottom = config.output_height - scaled_h - pad_y;

        if (pad_x > 0 || pad_y > 0 || pad_right > 0 || pad_bottom > 0) {
            ops.push_back(lua_cv::PreprocessOp::pad(
                pad_y, pad_bottom, pad_x, pad_right, config.pad_value));
        }
    } else {
        ops.push_back(lua_cv::PreprocessOp::resize(
            config.output_width, config.output_height, false));
    }

    ops.push_back(lua_cv::PreprocessOp::color(config.output_format));

    return ops;
}

bool VpssCapability::check_crop_before_resize(const std::vector<lua_cv::PreprocessOp>& ops) {
    bool seen_resize = false;
    for (const auto& op : ops) {
        if (op.type == lua_cv::PreprocessOp::RESIZE) {
            seen_resize = true;
        } else if (op.type == lua_cv::PreprocessOp::CROP && seen_resize) {
            return false;
        }
    }
    return true;
}

bool VpssCapability::check_single_resize(const std::vector<lua_cv::PreprocessOp>& ops) {
    int resize_count = 0;
    for (const auto& op : ops) {
        if (op.type == lua_cv::PreprocessOp::RESIZE) {
            ++resize_count;
        }
    }
    return resize_count <= 1;
}

bool VpssCapability::check_color_conversion(PIXEL_FORMAT_E format) {
    switch (format) {
        case PIXEL_FORMAT_RGB_888:
        case PIXEL_FORMAT_BGR_888:
        case PIXEL_FORMAT_RGB_888_PLANAR:
        case PIXEL_FORMAT_BGR_888_PLANAR:
        case PIXEL_FORMAT_NV12:
        case PIXEL_FORMAT_NV21:
        case PIXEL_FORMAT_YUV_400:
            return true;
        default:
            return false;
    }
}

bool VpssCapability::check_dimension_limits(uint32_t width, uint32_t height) {
    constexpr uint32_t MIN_DIM = 16;
    constexpr uint32_t MAX_DIM = 4096;
    return (width >= MIN_DIM && width <= MAX_DIM &&
            height >= MIN_DIM && height <= MAX_DIM);
}
