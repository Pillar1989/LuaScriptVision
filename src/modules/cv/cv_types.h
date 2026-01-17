#pragma once

namespace lua_cv {

/**
 * BackendType - CV backend type enum
 */
enum class BackendType {
    CPU,      // CPU OpenCV backend
    HARDWARE  // SG200X hardware backend (VI/VPSS)
};

/**
 * PixelFormat - Common pixel format enum
 */
enum class PixelFormat {
    RGB888,
    BGR888,
    NV12,
    NV21,
    GRAY,
    UNKNOWN
};

/**
 * ColorConversion - Color space conversion codes
 * Compatible with OpenCV cv::ColorConversionCodes
 */
enum class ColorConversion {
    BGR2RGB = 4,
    RGB2BGR = 4,
    BGR2GRAY = 6,
    RGB2GRAY = 7,
    GRAY2BGR = 8,
    GRAY2RGB = 9,
    BGR2NV12 = 143,
    RGB2NV12 = 144,
    NV12_BGR = 145,
    NV12_RGB = 146
};

} // namespace lua_cv
