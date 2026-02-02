#include "cv_helpers.h"

#include <stdexcept>

#include "opencv_processor.h"
#include "tensor/tensor.h"

#ifdef USE_CVI_MPI
#include "cvi_vpss_processor.h"
#include <cvi_sys.h>
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
    // For CVI frames, check against VPSS MEM capacity
    if (frame.storage_type() == Frame::StorageType::CVI) {
        const uint32_t max_w = MmfContext::vpss_max_width_for_mem();
        const uint32_t max_h = MmfContext::vpss_max_height_for_mem();
        if (static_cast<uint32_t>(width) > max_w || static_cast<uint32_t>(height) > max_h) {
            throw std::invalid_argument("cv_helpers::resize - exceeds VPSS capacity");
        }
    } else
#endif
    {
        // For CPU frames, check against current size (upscale requires new allocation)
        if (width > frame.width() || height > frame.height()) {
            throw std::invalid_argument("cv_helpers::resize - upscale not allowed for CPU frames");
        }
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

#ifdef USE_CVI_MPI
bool can_zero_copy(const Frame& frame,
                   PIXEL_FORMAT_E required_format,
                   uint32_t required_width,
                   uint32_t required_height,
                   std::string* reason) {
    if (frame.storage_type() != Frame::StorageType::CVI) {
        if (reason) {
            *reason = "frame is not CVI storage type";
        }
        return false;
    }
    if (!frame.has_physical_addr()) {
        if (reason) {
            *reason = "frame has no physical address";
        }
        return false;
    }
    uint64_t paddr = frame.physical_addr();
    if ((paddr & 0x3F) != 0) {
        if (reason) {
            *reason = "physical address is not 64-byte aligned";
        }
        return false;
    }
    PIXEL_FORMAT_E frame_format = to_cvi_pixel_format(frame.pixel_format());
    if (frame_format != required_format) {
        if (reason) {
            *reason = "pixel format mismatch";
        }
        return false;
    }
    if (static_cast<uint32_t>(frame.width()) != required_width ||
        static_cast<uint32_t>(frame.height()) != required_height) {
        if (reason) {
            *reason = "dimension mismatch";
        }
        return false;
    }
    return true;
}
#endif

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

// Fast path for CVI frames: process directly from mmap without clone
#ifdef USE_CVI_MPI
static tensor::Tensor frame_to_tensor_cvi(const Frame& frame,
                                          const float* scale_factor,
                                          const float* offset,
                                          int channels) {
    const VIDEO_FRAME_INFO_S* vf = frame.video_frame();
    if (!vf || vf->stVFrame.u64PhyAddr[0] == 0) {
        throw std::runtime_error("frame_to_tensor_cvi - invalid CVI frame");
    }

    int width = static_cast<int>(vf->stVFrame.u32Width);
    int height = static_cast<int>(vf->stVFrame.u32Height);
    int stride = static_cast<int>(vf->stVFrame.u32Stride[0]);
    size_t plane_size = static_cast<size_t>(height) * width;

    size_t total_size = static_cast<size_t>(vf->stVFrame.u32Length[0]);
    if (total_size == 0) {
        total_size = static_cast<size_t>(stride) * height;
    }

    void* mapped = CVI_SYS_MmapCache(vf->stVFrame.u64PhyAddr[0],
                                     static_cast<CVI_U32>(total_size));
    if (!mapped) {
        throw std::runtime_error("frame_to_tensor_cvi - mmap failed");
    }
    CVI_SYS_IonInvalidateCache(vf->stVFrame.u64PhyAddr[0], mapped,
                               static_cast<CVI_U32>(total_size));

    std::vector<float> data(static_cast<size_t>(channels) * plane_size);
    const uint8_t* base = static_cast<const uint8_t*>(mapped);

    if (channels == 3) {
        float* dst0 = data.data();
        float* dst1 = data.data() + plane_size;
        float* dst2 = data.data() + plane_size * 2;

        for (int y = 0; y < height; ++y) {
            const uint8_t* src = base + y * stride;
            size_t row_offset = static_cast<size_t>(y) * width;

            for (int x = 0; x < width; ++x) {
                dst0[row_offset + x] = src[0] * scale_factor[0] - offset[0];
                dst1[row_offset + x] = src[1] * scale_factor[1] - offset[1];
                dst2[row_offset + x] = src[2] * scale_factor[2] - offset[2];
                src += 3;
            }
        }
    } else {
        float* dst = data.data();
        float sf = scale_factor[0];
        float off = offset[0];

        for (int y = 0; y < height; ++y) {
            const uint8_t* src = base + y * stride;
            size_t row_offset = static_cast<size_t>(y) * width;

            for (int x = 0; x < width; ++x) {
                dst[row_offset + x] = src[x] * sf - off;
            }
        }
    }

    CVI_SYS_Munmap(mapped, static_cast<CVI_U32>(total_size));

    std::vector<int64_t> shape = {1, channels, height, width};
    return tensor::Tensor(std::move(data), shape);
}
#endif

tensor::Tensor frame_to_tensor(const Frame& frame,
                               double scale,
                               const std::vector<double>& mean,
                               const std::vector<double>& stddev) {
    if (frame.empty()) {
        throw std::invalid_argument("frame_to_tensor - frame is empty");
    }

    // Determine channels from pixel format
    int channels = 3;
    PixelFormat fmt = frame.pixel_format();
    if (fmt == PixelFormat::GRAY) {
        channels = 1;
    } else if (fmt != PixelFormat::BGR && fmt != PixelFormat::RGB) {
        // For other formats, fall through to cv::Mat path
        channels = -1;
    }

    std::vector<double> mean_vec;
    std::vector<double> std_vec;
    if (channels > 0) {
        validate_norm_params(channels, mean, stddev, mean_vec, std_vec);
    }

    // Precompute: (v * scale - mean) / std = v * (scale/std) - (mean/std)
    float scale_factor[3];
    float offset[3];
    if (channels > 0) {
        for (int c = 0; c < channels; ++c) {
            scale_factor[c] = static_cast<float>(scale / std_vec[c]);
            offset[c] = static_cast<float>(mean_vec[c] / std_vec[c]);
        }
    }

#ifdef USE_CVI_MPI
    // Fast path: process CVI frame directly without clone
    if (channels > 0 && frame.has_physical_addr() &&
        frame.storage_type() == Frame::StorageType::CVI) {
        return frame_to_tensor_cvi(frame, scale_factor, offset, channels);
    }
#endif

    // Fallback: use cv::Mat (may involve clone for CVI frames)
    const cv::Mat& mat = frame.to_mat();
    if (mat.empty()) {
        throw std::invalid_argument("frame_to_tensor - mat is empty");
    }

    channels = mat.channels();
    if (channels != 1 && channels != 3) {
        throw std::invalid_argument("frame_to_tensor - unsupported channel count");
    }

    // Re-validate if we didn't know channels before
    if (mean_vec.empty()) {
        validate_norm_params(channels, mean, stddev, mean_vec, std_vec);
        for (int c = 0; c < channels; ++c) {
            scale_factor[c] = static_cast<float>(scale / std_vec[c]);
            offset[c] = static_cast<float>(mean_vec[c] / std_vec[c]);
        }
    }

    int height = mat.rows;
    int width = mat.cols;
    size_t plane_size = static_cast<size_t>(height) * width;
    std::vector<float> data(static_cast<size_t>(channels) * plane_size);

    if (channels == 3) {
        float* dst0 = data.data();
        float* dst1 = data.data() + plane_size;
        float* dst2 = data.data() + plane_size * 2;

        for (int y = 0; y < height; ++y) {
            const uint8_t* src = mat.ptr<uint8_t>(y);
            size_t row_offset = static_cast<size_t>(y) * width;

            for (int x = 0; x < width; ++x) {
                // BGR interleaved -> CHW planar
                dst0[row_offset + x] = src[0] * scale_factor[0] - offset[0];
                dst1[row_offset + x] = src[1] * scale_factor[1] - offset[1];
                dst2[row_offset + x] = src[2] * scale_factor[2] - offset[2];
                src += 3;
            }
        }
    } else {
        float* dst = data.data();
        float sf = scale_factor[0];
        float off = offset[0];

        for (int y = 0; y < height; ++y) {
            const uint8_t* src = mat.ptr<uint8_t>(y);
            size_t row_offset = static_cast<size_t>(y) * width;

            for (int x = 0; x < width; ++x) {
                dst[row_offset + x] = src[x] * sf - off;
            }
        }
    }

    std::vector<int64_t> shape = {1, channels, height, width};
    return tensor::Tensor(std::move(data), shape);
}

} // namespace lua_cv::cv_helpers
