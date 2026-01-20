#include "cv_helpers.h"

#include <stdexcept>

#include "opencv_processor.h"
#include "tensor/tensor.h"

#ifdef USE_CVI_MPI
#include "cvi_vpss_processor.h"
#endif

namespace lua_cv::cv_helpers {

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
        throw std::invalid_argument("cv_helpers::resize - frame is empty");
    }
#ifdef USE_CVI_MPI
    if (frame.storage_type() == Frame::StorageType::CVI) {
        try {
            get_vpss_processor().resize(frame, width, height);
            return;
        } catch (const std::exception&) {
            // Fall back to CPU below.
        }
    }
#endif
    get_opencv_processor().resize(frame, width, height);
}

void cvt_color(Frame& frame, ColorConversion code) {
    if (frame.empty()) {
        throw std::invalid_argument("cv_helpers::cvt_color - frame is empty");
    }
#ifdef USE_CVI_MPI
    if (frame.storage_type() == Frame::StorageType::CVI) {
        try {
            get_vpss_processor().cvtColor(frame, code);
            return;
        } catch (const std::exception&) {
            // Fall back to CPU below.
        }
    }
#endif
    get_opencv_processor().cvtColor(frame, code);
}

void crop(Frame& frame, int x, int y, int w, int h) {
    if (frame.empty()) {
        throw std::invalid_argument("cv_helpers::crop - frame is empty");
    }
#ifdef USE_CVI_MPI
    if (frame.storage_type() == Frame::StorageType::CVI) {
        try {
            get_vpss_processor().crop(frame, x, y, w, h);
            return;
        } catch (const std::exception&) {
            // Fall back to CPU below.
        }
    }
#endif
    get_opencv_processor().crop(frame, x, y, w, h);
}

const char* get_backend_name(const Frame& frame) {
#ifdef USE_CVI_MPI
    if (frame.storage_type() == Frame::StorageType::CVI) {
        return "vpss";
    }
#else
    (void)frame;
#endif
    return "opencv";
}

static void validate_norm_params(int channels,
                                 const std::vector<double>& mean,
                                 const std::vector<double>& stddev,
                                 std::vector<double>& mean_out,
                                 std::vector<double>& std_out) {
    auto expand = [](int channels, const std::vector<double>& input, double default_val) {
        if (input.empty()) {
            return std::vector<double>(channels, default_val);
        }
        if (input.size() == 1) {
            return std::vector<double>(channels, input[0]);
        }
        if (static_cast<int>(input.size()) == channels) {
            return input;
        }
        throw std::invalid_argument("frame_to_tensor - mean/std size mismatch");
    };

    mean_out = expand(channels, mean, 0.0);
    std_out = expand(channels, stddev, 1.0);
    for (double val : std_out) {
        if (val == 0.0) {
            throw std::invalid_argument("frame_to_tensor - std contains zero");
        }
    }
}

tensor::Tensor frame_to_tensor(const Frame& frame,
                               double scale,
                               const std::vector<double>& mean,
                               const std::vector<double>& stddev) {
    if (frame.empty()) {
        throw std::invalid_argument("frame_to_tensor - frame is empty");
    }

    const cv::Mat& mat = frame.to_mat();
    if (mat.empty()) {
        throw std::invalid_argument("frame_to_tensor - mat is empty");
    }

    int channels = mat.channels();
    if (channels != 1 && channels != 3) {
        throw std::invalid_argument("frame_to_tensor - unsupported channel count");
    }

    std::vector<double> mean_vec;
    std::vector<double> std_vec;
    validate_norm_params(channels, mean, stddev, mean_vec, std_vec);

    int height = mat.rows;
    int width = mat.cols;
    std::vector<float> data(static_cast<size_t>(channels) * height * width);

    if (channels == 3) {
        for (int y = 0; y < height; ++y) {
            const cv::Vec3b* row = mat.ptr<cv::Vec3b>(y);
            for (int x = 0; x < width; ++x) {
                const cv::Vec3b& pix = row[x];
                for (int c = 0; c < 3; ++c) {
                    size_t idx = static_cast<size_t>(c) * height * width +
                                 static_cast<size_t>(y) * width +
                                 static_cast<size_t>(x);
                    double v = static_cast<double>(pix[c]);
                    data[idx] = static_cast<float>((v * scale - mean_vec[c]) / std_vec[c]);
                }
            }
        }
    } else {
        for (int y = 0; y < height; ++y) {
            const uint8_t* row = mat.ptr<uint8_t>(y);
            for (int x = 0; x < width; ++x) {
                size_t idx = static_cast<size_t>(y) * width + static_cast<size_t>(x);
                double v = static_cast<double>(row[x]);
                data[idx] = static_cast<float>((v * scale - mean_vec[0]) / std_vec[0]);
            }
        }
    }

    std::vector<int64_t> shape = {1, channels, height, width};
    return tensor::Tensor(std::move(data), shape);
}

} // namespace lua_cv::cv_helpers
