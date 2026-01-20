#pragma once

#include <cstddef>
#include <memory>

#include "cv_types.h"
#include "device_buffer.h"

namespace lua_cv {

class CvImage {
public:
    CvImage() = default;
    CvImage(int width, int height, PixelFormat format,
            std::shared_ptr<DeviceBuffer> buffer,
            int stride_bytes, size_t offset_bytes = 0);

    bool empty() const { return width_ <= 0 || height_ <= 0 || !buffer_; }
    int width() const { return width_; }
    int height() const { return height_; }
    int stride() const { return stride_bytes_; }
    size_t offset() const { return offset_bytes_; }
    PixelFormat format() const { return format_; }
    DeviceType device() const { return device_; }

    std::shared_ptr<DeviceBuffer> buffer() const { return buffer_; }

    CvImage roi_view(int x, int y, int w, int h) const;

private:
    int width_ = 0;
    int height_ = 0;
    int stride_bytes_ = 0;
    size_t offset_bytes_ = 0;
    PixelFormat format_ = PixelFormat::UNKNOWN;
    DeviceType device_ = DeviceType::UNKNOWN;
    std::shared_ptr<DeviceBuffer> buffer_;
};

} // namespace lua_cv
