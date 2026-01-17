#include "cv_helpers.h"
#include "opencv_processor.h"
#include "tensor/tensor.h"

#ifdef USE_CVI_MPI
#include "cvi_vpss_processor.h"
#endif

#include <iostream>
#include <opencv2/opencv.hpp>

namespace lua_cv {
namespace cv_helpers {

// Singleton processor instances
static OpenCvProcessor& get_opencv_processor() {
    static OpenCvProcessor instance;
    return instance;
}

#ifdef USE_CVI_MPI
static CviVpssProcessor& get_vpss_processor() {
    static CviVpssProcessor instance;
    return instance;
}
#endif

void resize(Frame& frame, int width, int height) {
    if (frame.empty()) {
        throw std::invalid_argument("cv_helpers::resize() - frame is empty");
    }

#ifdef USE_CVI_MPI
    // Smart backend selection: VPSS for zero-copy, OpenCV for CPU
    if (frame.has_physical_addr()) {
        try {
            get_vpss_processor().resize(frame, width, height);
            return;
        } catch (const std::exception& e) {
            std::cerr << "[WARNING] VPSS resize failed: " << e.what()
                      << ", falling back to OpenCV" << std::endl;
            // Fall through to OpenCV fallback
        }
    }
#endif

    // CPU fallback (always available)
    get_opencv_processor().resize(frame, width, height);
}

void cvt_color(Frame& frame, ColorConversion code) {
    if (frame.empty()) {
        throw std::invalid_argument("cv_helpers::cvt_color() - frame is empty");
    }

#ifdef USE_CVI_MPI
    // Try VPSS first for zero-copy frames
    if (frame.has_physical_addr()) {
        try {
            get_vpss_processor().cvtColor(frame, code);
            return;
        } catch (const std::exception& e) {
            // VPSS may not support this conversion, fall back to OpenCV
            std::cerr << "[WARNING] VPSS cvtColor failed: " << e.what()
                      << ", falling back to OpenCV" << std::endl;
            // Fall through to OpenCV fallback
        }
    }
#endif

    // CPU fallback
    get_opencv_processor().cvtColor(frame, code);
}

void crop(Frame& frame, int x, int y, int w, int h) {
    if (frame.empty()) {
        throw std::invalid_argument("cv_helpers::crop() - frame is empty");
    }

#ifdef USE_CVI_MPI
    // Try VPSS first for zero-copy frames
    if (frame.has_physical_addr()) {
        try {
            get_vpss_processor().crop(frame, x, y, w, h);
            return;
        } catch (const std::exception& e) {
            std::cerr << "[WARNING] VPSS crop failed: " << e.what()
                      << ", falling back to OpenCV" << std::endl;
            // Fall through to OpenCV fallback
        }
    }
#endif

    // CPU fallback
    get_opencv_processor().crop(frame, x, y, w, h);
}

const char* get_backend_name(const Frame& frame) {
#ifdef USE_CVI_MPI
    if (frame.has_physical_addr()) {
        return "vpss";
    }
#endif
    return "opencv";
}

tensor::Tensor frame_to_tensor(
    const Frame& frame,
    double scale,
    const std::vector<double>& mean,
    const std::vector<double>& std) {

    if (frame.empty()) {
        throw std::invalid_argument("frame_to_tensor() - frame is empty");
    }

    // Convert Frame to cv::Mat (lazy conversion, may be zero-copy)
    const cv::Mat& mat = frame.to_mat();

    // Convert to float
    cv::Mat float_mat;
    mat.convertTo(float_mat, CV_32F);

    // HWC → CHW conversion with cv::split optimization
    int H = float_mat.rows;
    int W = float_mat.cols;
    int C = float_mat.channels();

    // Ensure mean/std have enough elements
    std::vector<double> mean_vec = mean;
    std::vector<double> std_vec = std;
    if (mean_vec.size() < static_cast<size_t>(C)) {
        mean_vec.resize(C, 0.0);
    }
    if (std_vec.size() < static_cast<size_t>(C)) {
        std_vec.resize(C, 1.0);
    }

    // Split channels
    std::vector<cv::Mat> channels(C);
    cv::split(float_mat, channels);

    // Per-channel normalization and CHW data assembly
    std::vector<float> chw_data(C * H * W);
    size_t idx = 0;

    for (int c = 0; c < C; ++c) {
        const float* channel_ptr = channels[c].ptr<float>();
        double m = mean_vec[c];
        double s = std_vec[c];
        for (int i = 0; i < H * W; ++i) {
            chw_data[idx++] = (channel_ptr[i] * scale - m) / s;
        }
    }

    // Create Tensor object (NCHW format)
    std::vector<int64_t> shape = {1, static_cast<int64_t>(C),
                                   static_cast<int64_t>(H),
                                   static_cast<int64_t>(W)};
    return tensor::Tensor(std::move(chw_data), shape);
}

} // namespace cv_helpers
} // namespace lua_cv
