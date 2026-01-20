#pragma once

#include <cstdint>
#include <string>

#ifdef USE_CVI_MPI
#include <linux/cvi_comm_video.h>
#endif

namespace lua_cv {

enum class DeviceType {
    CPU = 0,
    VB,
    ION,
    UNKNOWN,
};

enum class PixelFormat {
    BGR = 0,
    RGB,
    NV12,
    NV21,
    GRAY,
    UNKNOWN,
};

enum class ColorConversion {
    BGR2RGB = 0,
    RGB2BGR,
    BGR2GRAY,
    RGB2GRAY,
    GRAY2BGR,
    GRAY2RGB,
};

struct Size {
    int width = 0;
    int height = 0;
};

int channels_for_format(PixelFormat format);
const char* pixel_format_name(PixelFormat format);

#ifdef USE_CVI_MPI
PIXEL_FORMAT_E to_cvi_pixel_format(PixelFormat format);
PixelFormat from_cvi_pixel_format(PIXEL_FORMAT_E format);
#endif

} // namespace lua_cv
