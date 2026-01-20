#include "opencv_processor.h"

#include <stdexcept>

namespace lua_cv {

void OpenCvProcessor::resize(Frame& frame, int width, int height) {
    if (frame.empty()) {
        throw std::invalid_argument("OpenCvProcessor::resize - frame is empty");
    }
    if (width <= 0 || height <= 0) {
        throw std::invalid_argument("OpenCvProcessor::resize - invalid dimensions");
    }

    const cv::Mat& src = frame.to_mat();
    cv::Mat dst;
    cv::resize(src, dst, cv::Size(width, height));

    PixelFormat out_format = frame.pixel_format();
    if (out_format == PixelFormat::UNKNOWN || frame.storage_type() != Frame::StorageType::OPENCV) {
        out_format = (dst.channels() == 1) ? PixelFormat::GRAY : PixelFormat::BGR;
    }
    frame = Frame(std::move(dst), out_format);
}

void OpenCvProcessor::cvtColor(Frame& frame, ColorConversion code) {
    if (frame.empty()) {
        throw std::invalid_argument("OpenCvProcessor::cvtColor - frame is empty");
    }

    const cv::Mat& src = frame.to_mat();
    cv::Mat dst;
    int cv_code = -1;

    switch (code) {
        case ColorConversion::BGR2RGB:
            cv_code = cv::COLOR_BGR2RGB;
            break;
        case ColorConversion::RGB2BGR:
            cv_code = cv::COLOR_RGB2BGR;
            break;
        case ColorConversion::BGR2GRAY:
            cv_code = cv::COLOR_BGR2GRAY;
            break;
        case ColorConversion::RGB2GRAY:
            cv_code = cv::COLOR_RGB2GRAY;
            break;
        case ColorConversion::GRAY2BGR:
            cv_code = cv::COLOR_GRAY2BGR;
            break;
        case ColorConversion::GRAY2RGB:
            cv_code = cv::COLOR_GRAY2RGB;
            break;
        default:
            throw std::invalid_argument("OpenCvProcessor::cvtColor - unsupported conversion");
    }

    cv::cvtColor(src, dst, cv_code);

    PixelFormat out_format = frame.pixel_format();
    switch (code) {
        case ColorConversion::BGR2RGB:
        case ColorConversion::GRAY2RGB:
            out_format = PixelFormat::RGB;
            break;
        case ColorConversion::RGB2BGR:
        case ColorConversion::GRAY2BGR:
            out_format = PixelFormat::BGR;
            break;
        case ColorConversion::BGR2GRAY:
        case ColorConversion::RGB2GRAY:
            out_format = PixelFormat::GRAY;
            break;
        default:
            break;
    }
    frame = Frame(std::move(dst), out_format);
}

void OpenCvProcessor::crop(Frame& frame, int x, int y, int w, int h) {
    if (frame.empty()) {
        throw std::invalid_argument("OpenCvProcessor::crop - frame is empty");
    }
    if (x < 0 || y < 0 || w <= 0 || h <= 0 ||
        x + w > frame.width() || y + h > frame.height()) {
        throw std::invalid_argument("OpenCvProcessor::crop - invalid crop region");
    }

    const cv::Mat& src = frame.to_mat();
    cv::Rect roi(x, y, w, h);
    cv::Mat cropped = src(roi).clone();

    PixelFormat out_format = frame.pixel_format();
    if (out_format == PixelFormat::UNKNOWN || frame.storage_type() != Frame::StorageType::OPENCV) {
        out_format = (cropped.channels() == 1) ? PixelFormat::GRAY : PixelFormat::BGR;
    }
    frame = Frame(std::move(cropped), out_format);
}

} // namespace lua_cv
