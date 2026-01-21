#include "frame_tensor.h"

#include <stdexcept>

#include "inference/layout.h"
#include <opencv2/opencv.hpp>

namespace lua_cv {

tensor::Tensor frame_to_tensor(const Frame& frame,
                               const inference::TensorDescriptor& desc) {
    if (frame.empty()) {
        throw std::invalid_argument("frame_to_tensor - frame is empty");
    }

    if (desc.dtype != inference::DType::Unknown &&
        desc.dtype != inference::DType::Float32) {
        throw std::runtime_error("frame_to_tensor - only float32 tensors are supported");
    }

    cv::Mat mat = frame.to_mat();
    if (mat.empty()) {
        throw std::runtime_error("frame_to_tensor - failed to convert frame to mat");
    }

    if (mat.channels() != 1 && mat.channels() != 3) {
        throw std::runtime_error("frame_to_tensor - unsupported channel count");
    }

    int height = mat.rows;
    int width = mat.cols;
    int channels = mat.channels();

    inference::Layout layout = desc.layout;
    if (layout == inference::Layout::Unknown) {
        layout = inference::Layout::NCHW;
    }

    std::vector<int64_t> shape;
    if (!desc.shape.empty()) {
        shape = desc.shape;
    } else if (layout == inference::Layout::NHWC) {
        shape = {1, height, width, channels};
    } else {
        shape = {1, channels, height, width};
    }

    if (layout == inference::Layout::NCHW) {
        if (shape.size() != 4 || shape[0] != 1 || shape[1] != channels ||
            shape[2] != height || shape[3] != width) {
            throw std::runtime_error("frame_to_tensor - shape mismatch for NCHW");
        }

        std::vector<float> data(static_cast<size_t>(channels) * height * width);
        for (int y = 0; y < height; ++y) {
            if (channels == 3) {
                const cv::Vec3b* row = mat.ptr<cv::Vec3b>(y);
                for (int x = 0; x < width; ++x) {
                    const cv::Vec3b& pix = row[x];
                    for (int c = 0; c < 3; ++c) {
                        size_t idx = static_cast<size_t>(c) * height * width +
                                     static_cast<size_t>(y) * width +
                                     static_cast<size_t>(x);
                        data[idx] = static_cast<float>(pix[c]);
                    }
                }
            } else {
                const uint8_t* row = mat.ptr<uint8_t>(y);
                for (int x = 0; x < width; ++x) {
                    size_t idx = static_cast<size_t>(y) * width + static_cast<size_t>(x);
                    data[idx] = static_cast<float>(row[x]);
                }
            }
        }
        return tensor::Tensor(std::move(data), shape);
    }

    if (layout == inference::Layout::NHWC) {
        if (shape.size() != 4 || shape[0] != 1 || shape[1] != height ||
            shape[2] != width || shape[3] != channels) {
            throw std::runtime_error("frame_to_tensor - shape mismatch for NHWC");
        }

        size_t total = static_cast<size_t>(height) * width * channels;
        std::vector<float> data(total);
        size_t idx = 0;
        for (int y = 0; y < height; ++y) {
            if (channels == 3) {
                const cv::Vec3b* row = mat.ptr<cv::Vec3b>(y);
                for (int x = 0; x < width; ++x) {
                    const cv::Vec3b& pix = row[x];
                    data[idx++] = static_cast<float>(pix[0]);
                    data[idx++] = static_cast<float>(pix[1]);
                    data[idx++] = static_cast<float>(pix[2]);
                }
            } else {
                const uint8_t* row = mat.ptr<uint8_t>(y);
                for (int x = 0; x < width; ++x) {
                    data[idx++] = static_cast<float>(row[x]);
                }
            }
        }
        return tensor::Tensor(std::move(data), shape);
    }

    throw std::runtime_error("frame_to_tensor - unsupported layout");
}

Frame tensor_to_frame(const tensor::Tensor& tensor) {
    if (tensor.device() != tensor::DeviceType::CPU) {
        throw std::runtime_error("tensor_to_frame - tensor must be on CPU");
    }

    tensor::Tensor cont = tensor.contiguous();
    auto shape = cont.shape();
    if (shape.size() != 3 && shape.size() != 4) {
        throw std::runtime_error("tensor_to_frame - unsupported tensor shape");
    }

    inference::Layout layout = inference::infer_layout(shape);
    if (layout == inference::Layout::Unknown) {
        throw std::runtime_error("tensor_to_frame - unable to infer layout");
    }

    int64_t batch = 1;
    int64_t channels = 0;
    int64_t height = 0;
    int64_t width = 0;

    if (layout == inference::Layout::NCHW) {
        if (shape.size() == 3) {
            channels = shape[0];
            height = shape[1];
            width = shape[2];
        } else {
            batch = shape[0];
            channels = shape[1];
            height = shape[2];
            width = shape[3];
        }
    } else if (layout == inference::Layout::NHWC) {
        if (shape.size() == 3) {
            height = shape[0];
            width = shape[1];
            channels = shape[2];
        } else {
            batch = shape[0];
            height = shape[1];
            width = shape[2];
            channels = shape[3];
        }
    }

    if (batch != 1 || (channels != 1 && channels != 3)) {
        throw std::runtime_error("tensor_to_frame - unsupported batch or channel count");
    }

    const float* src = cont.data();
    if (!src) {
        throw std::runtime_error("tensor_to_frame - tensor data is null");
    }

    int type = (channels == 3) ? CV_32FC3 : CV_32FC1;
    cv::Mat mat(static_cast<int>(height), static_cast<int>(width), type);

    if (layout == inference::Layout::NCHW) {
        for (int y = 0; y < height; ++y) {
            if (channels == 3) {
                cv::Vec3f* row = mat.ptr<cv::Vec3f>(y);
                for (int x = 0; x < width; ++x) {
                    cv::Vec3f pix;
                    for (int c = 0; c < 3; ++c) {
                        size_t idx = static_cast<size_t>(c) * height * width +
                                     static_cast<size_t>(y) * width +
                                     static_cast<size_t>(x);
                        pix[c] = src[idx];
                    }
                    row[x] = pix;
                }
            } else {
                float* row = mat.ptr<float>(y);
                for (int x = 0; x < width; ++x) {
                    size_t idx = static_cast<size_t>(y) * width + static_cast<size_t>(x);
                    row[x] = src[idx];
                }
            }
        }
    } else {
        size_t idx = 0;
        for (int y = 0; y < height; ++y) {
            if (channels == 3) {
                cv::Vec3f* row = mat.ptr<cv::Vec3f>(y);
                for (int x = 0; x < width; ++x) {
                    row[x] = cv::Vec3f(src[idx], src[idx + 1], src[idx + 2]);
                    idx += 3;
                }
            } else {
                float* row = mat.ptr<float>(y);
                for (int x = 0; x < width; ++x) {
                    row[x] = src[idx++];
                }
            }
        }
    }

    if (channels == 1) {
        return Frame(mat, PixelFormat::GRAY);
    }
    return Frame(mat, PixelFormat::BGR);
}

uint64_t get_physical_address(const Frame& frame) {
    return frame.has_physical_addr() ? frame.physical_addr() : 0;
}

} // namespace lua_cv
