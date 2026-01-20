#include "image.h"

#include <stdexcept>

namespace lua_cv {

CvImage::CvImage(int width, int height, PixelFormat format,
                 std::shared_ptr<DeviceBuffer> buffer,
                 int stride_bytes, size_t offset_bytes)
    : width_(width),
      height_(height),
      stride_bytes_(stride_bytes),
      offset_bytes_(offset_bytes),
      format_(format),
      device_(buffer ? buffer->device() : DeviceType::UNKNOWN),
      buffer_(std::move(buffer)) {}

CvImage CvImage::roi_view(int x, int y, int w, int h) const {
    if (empty()) {
        throw std::invalid_argument("CvImage::roi_view - empty image");
    }
    if (x < 0 || y < 0 || w <= 0 || h <= 0 ||
        x + w > width_ || y + h > height_) {
        throw std::invalid_argument("CvImage::roi_view - invalid ROI bounds");
    }

    int bytes_per_pixel = 0;
    switch (format_) {
        case PixelFormat::BGR:
        case PixelFormat::RGB:
            bytes_per_pixel = 3;
            break;
        case PixelFormat::GRAY:
            bytes_per_pixel = 1;
            break;
        default:
            throw std::invalid_argument("CvImage::roi_view - unsupported format for ROI view");
    }

    size_t new_offset = offset_bytes_ + static_cast<size_t>(y) * stride_bytes_
                        + static_cast<size_t>(x) * bytes_per_pixel;

    return CvImage(w, h, format_, buffer_, stride_bytes_, new_offset);
}

} // namespace lua_cv
