#include "opencv_processor.h"
#include <stdexcept>
#include <chrono>
#include <iostream>

namespace lua_cv {

void OpenCvProcessor::resize(Frame& frame, int width, int height) {
    if (frame.empty()) {
        throw std::invalid_argument("OpenCvProcessor::resize() - frame is empty");
    }

    if (width <= 0 || height <= 0) {
        throw std::invalid_argument("OpenCvProcessor::resize() - invalid dimensions");
    }

    // Convert to cv::Mat (no-op if already cv::Mat)
    const cv::Mat& input = frame.to_mat();

    // Resize using OpenCV
    cv::Mat resized;
    cv::resize(input, resized, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);

    // Replace frame with resized result
    frame = Frame(resized);
}

void OpenCvProcessor::cvtColor(Frame& frame, ColorConversion code) {
    if (frame.empty()) {
        throw std::invalid_argument("OpenCvProcessor::cvtColor() - frame is empty");
    }

    // Convert to cv::Mat
    const cv::Mat& input = frame.to_mat();

    // Map ColorConversion to OpenCV code
    int cv_code = static_cast<int>(code);

    // Perform conversion
    cv::Mat converted;
    cv::cvtColor(input, converted, cv_code);

    // Replace frame
    frame = Frame(converted);
}

void OpenCvProcessor::crop(Frame& frame, int x, int y, int w, int h) {
    if (frame.empty()) {
        throw std::invalid_argument("OpenCvProcessor::crop() - frame is empty");
    }

    // Convert to cv::Mat
    const cv::Mat& input = frame.to_mat();

    // Validate crop region
    if (x < 0 || y < 0 || w <= 0 || h <= 0 ||
        x + w > input.cols || y + h > input.rows) {
        throw std::invalid_argument("OpenCvProcessor::crop() - invalid crop region");
    }

    // Crop using ROI (zero-copy, creates view)
    cv::Mat cropped = input(cv::Rect(x, y, w, h));

    // Clone to ensure we own the data (ROI is just a view)
    frame = Frame(cropped.clone());
}

} // namespace lua_cv
