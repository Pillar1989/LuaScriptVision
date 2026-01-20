#include "cv_types.h"

namespace lua_cv {

int channels_for_format(PixelFormat format) {
    switch (format) {
        case PixelFormat::BGR:
        case PixelFormat::RGB:
            return 3;
        case PixelFormat::GRAY:
            return 1;
        case PixelFormat::NV12:
        case PixelFormat::NV21:
            return 1;
        default:
            return 0;
    }
}

const char* pixel_format_name(PixelFormat format) {
    switch (format) {
        case PixelFormat::BGR:
            return "BGR";
        case PixelFormat::RGB:
            return "RGB";
        case PixelFormat::NV12:
            return "NV12";
        case PixelFormat::NV21:
            return "NV21";
        case PixelFormat::GRAY:
            return "GRAY";
        default:
            return "UNKNOWN";
    }
}

#ifdef USE_CVI_MPI
PIXEL_FORMAT_E to_cvi_pixel_format(PixelFormat format) {
    switch (format) {
        case PixelFormat::BGR:
            return PIXEL_FORMAT_BGR_888;
        case PixelFormat::RGB:
            return PIXEL_FORMAT_RGB_888;
        case PixelFormat::NV12:
            return PIXEL_FORMAT_NV12;
        case PixelFormat::NV21:
            return PIXEL_FORMAT_NV21;
        case PixelFormat::GRAY:
            return PIXEL_FORMAT_YUV_400;
        default:
            return PIXEL_FORMAT_MAX;
    }
}

PixelFormat from_cvi_pixel_format(PIXEL_FORMAT_E format) {
    switch (format) {
        case PIXEL_FORMAT_BGR_888:
            return PixelFormat::BGR;
        case PIXEL_FORMAT_RGB_888:
            return PixelFormat::RGB;
        case PIXEL_FORMAT_NV12:
            return PixelFormat::NV12;
        case PIXEL_FORMAT_NV21:
            return PixelFormat::NV21;
        case PIXEL_FORMAT_YUV_400:
            return PixelFormat::GRAY;
        default:
            return PixelFormat::UNKNOWN;
    }
}
#endif

} // namespace lua_cv
