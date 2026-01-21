#include "cvi_session.h"

#ifdef USE_CVI_TPU

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include "modules/tensor/cvi_tpu_memory.h"

namespace inference {
namespace {
void check_cvi_rc(CVI_RC rc, const char* what) {
    if (rc != CVI_RC_SUCCESS) {
        std::string msg = std::string(what) + " failed: " + std::to_string(rc);
        throw std::runtime_error(msg);
    }
}

std::vector<int64_t> shape_from_cvi(const CVI_SHAPE& shape) {
    std::vector<int64_t> out;
    out.reserve(shape.dim_size);
    for (size_t i = 0; i < shape.dim_size; ++i) {
        out.push_back(static_cast<int64_t>(shape.dim[i]));
    }
    return out;
}

int64_t element_count(const std::vector<int64_t>& shape) {
    return std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>());
}

const char* cvi_fmt_name(CVI_FMT fmt) {
    switch (fmt) {
        case CVI_FMT_FP32:
            return "fp32";
        case CVI_FMT_INT32:
            return "int32";
        case CVI_FMT_UINT32:
            return "uint32";
        case CVI_FMT_BF16:
            return "bf16";
        case CVI_FMT_INT16:
            return "int16";
        case CVI_FMT_UINT16:
            return "uint16";
        case CVI_FMT_INT8:
            return "int8";
        case CVI_FMT_UINT8:
            return "uint8";
        default:
            return "unknown";
    }
}

const char* cvi_pixel_format_name(CVI_NN_PIXEL_FORMAT_E fmt) {
    switch (fmt) {
        case CVI_NN_PIXEL_RGB_PACKED:
            return "rgb_packed";
        case CVI_NN_PIXEL_BGR_PACKED:
            return "bgr_packed";
        case CVI_NN_PIXEL_RGB_PLANAR:
            return "rgb_planar";
        case CVI_NN_PIXEL_BGR_PLANAR:
            return "bgr_planar";
        case CVI_NN_PIXEL_YUV_NV12:
            return "nv12";
        case CVI_NN_PIXEL_YUV_NV21:
            return "nv21";
        case CVI_NN_PIXEL_YUV_420_PLANAR:
            return "yuv420p";
        case CVI_NN_PIXEL_GRAYSCALE:
            return "grayscale";
        case CVI_NN_PIXEL_TENSOR:
            return "tensor";
        case CVI_NN_PIXEL_RGBA_PLANAR:
            return "rgba_planar";
        default:
            return "unknown";
    }
}

std::string shape_to_string(const std::vector<int64_t>& shape) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            oss << ", ";
        }
        oss << shape[i];
    }
    oss << "]";
    return oss.str();
}

bool env_enabled(const char* name) {
    const char* value = std::getenv(name);
    return value && value[0] != '\0' && value[0] != '0';
}

bool shape_compatible(const std::vector<int64_t>& model_shape,
                      const std::vector<int64_t>& input_shape) {
    if (model_shape.size() != input_shape.size()) {
        return false;
    }
    for (size_t i = 0; i < model_shape.size(); ++i) {
        int64_t expected = model_shape[i];
        if (expected > 0 && expected != input_shape[i]) {
            return false;
        }
    }
    return true;
}

uint16_t float_to_bf16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<uint16_t>(bits >> 16);
}

float bf16_to_float(uint16_t value) {
    uint32_t bits = static_cast<uint32_t>(value) << 16;
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

template <typename T>
T quantize_value(float value, float scale, int zero_point) {
    if (scale == 0.0f) {
        scale = 1.0f;
    }
    double scaled = static_cast<double>(value) * static_cast<double>(scale) +
                    static_cast<double>(zero_point);
    long long q = std::llround(scaled);
    long long min_val = static_cast<long long>(std::numeric_limits<T>::min());
    long long max_val = static_cast<long long>(std::numeric_limits<T>::max());
    if (q < min_val) {
        q = min_val;
    } else if (q > max_val) {
        q = max_val;
    }
    return static_cast<T>(q);
}

template <typename T>
float dequantize_value(T value, float scale, int zero_point) {
    if (scale == 0.0f) {
        scale = 1.0f;
    }
    return (static_cast<double>(value) - static_cast<double>(zero_point)) *
           static_cast<double>(scale);
}
} // namespace

CviSession::CviSession(const std::string& model_path) {
    check_cvi_rc(CVI_RT_Init(&rt_handle_), "CVI_RT_Init");
    check_cvi_rc(CVI_NN_RegisterModel(model_path.c_str(), &model_), "CVI_NN_RegisterModel");
    check_cvi_rc(CVI_NN_GetInputOutputTensors(model_, &input_tensors_, &input_num_,
                                             &output_tensors_, &output_num_),
                 "CVI_NN_GetInputOutputTensors");
    if (!input_tensors_ || input_num_ <= 0) {
        throw std::runtime_error("CviSession: no input tensors");
    }
    if (!output_tensors_ || output_num_ <= 0) {
        throw std::runtime_error("CviSession: no output tensors");
    }

    input_names_.reserve(static_cast<size_t>(input_num_));
    for (int32_t i = 0; i < input_num_; ++i) {
        char* name = CVI_NN_TensorName(&input_tensors_[i]);
        input_names_.push_back(name ? name : "");
    }

    output_names_.reserve(static_cast<size_t>(output_num_));
    for (int32_t i = 0; i < output_num_; ++i) {
        char* name = CVI_NN_TensorName(&output_tensors_[i]);
        output_names_.push_back(name ? name : "");
    }

    init_device_buffers();
    primary_output_index_ = select_primary_output_index();

    if (env_enabled("CVI_SESSION_LOG_IO")) {
        for (int32_t i = 0; i < input_num_; ++i) {
            auto info = get_input_info(static_cast<size_t>(i));
            std::cerr << "[CVI] input[" << i << "] name=" << info.name
                      << " shape=" << shape_to_string(info.shape)
                      << " fmt=" << cvi_fmt_name(info.fmt)
                      << " qscale=" << info.qscale
                      << " zero_point=" << info.zero_point
                      << " pixel_format=" << cvi_pixel_format_name(info.pixel_format)
                      << " mean=(" << info.mean[0] << ", " << info.mean[1] << ", " << info.mean[2] << ")"
                      << " scale=(" << info.scale[0] << ", " << info.scale[1] << ", " << info.scale[2] << ")"
                      << "\n";
        }
        for (int32_t i = 0; i < output_num_; ++i) {
            auto info = get_output_info(static_cast<size_t>(i));
            std::cerr << "[CVI] output[" << i << "] name=" << info.name
                      << " shape=" << shape_to_string(info.shape)
                      << " fmt=" << cvi_fmt_name(info.fmt)
                      << " qscale=" << info.qscale
                      << " zero_point=" << info.zero_point
                      << " pixel_format=" << cvi_pixel_format_name(info.pixel_format)
                      << "\n";
        }
        std::cerr << "[CVI] primary_output_index=" << primary_output_index_ << "\n";
    }
}

CviSession::~CviSession() {
    if (model_) {
        CVI_NN_CleanupModel(model_);
        model_ = nullptr;
    }
    output_buffers_.clear();
    input_buffers_.clear();
    if (rt_handle_) {
        CVI_RT_DeInit(rt_handle_);
        rt_handle_ = nullptr;
    }
}

std::vector<int64_t> CviSession::get_input_shape(size_t index) const {
    if (index >= static_cast<size_t>(input_num_)) {
        return {};
    }
    return shape_from_cvi(CVI_NN_TensorShape(&input_tensors_[index]));
}

std::vector<int64_t> CviSession::get_output_shape(size_t index) const {
    if (index >= static_cast<size_t>(output_num_)) {
        return {};
    }
    return shape_from_cvi(CVI_NN_TensorShape(&output_tensors_[index]));
}

int32_t CviSession::select_primary_output_index() const {
    const char* env_value = std::getenv("CVI_SESSION_OUTPUT_INDEX");
    if (env_value && env_value[0] != '\0') {
        char* endptr = nullptr;
        long parsed = std::strtol(env_value, &endptr, 10);
        if (endptr != env_value && parsed >= 0 && parsed < output_num_) {
            return static_cast<int32_t>(parsed);
        }
    }

    int32_t best = 0;
    size_t best_count = 0;
    for (int32_t i = 0; i < output_num_; ++i) {
        size_t count = CVI_NN_TensorCount(&output_tensors_[i]);
        if (count > best_count) {
            best = i;
            best_count = count;
        }
    }
    return best;
}

void CviSession::init_device_buffers() {
    input_buffers_.resize(static_cast<size_t>(input_num_));
    for (int32_t i = 0; i < input_num_; ++i) {
        size_t bytes = CVI_NN_TensorSize(&input_tensors_[i]);
        if (bytes == 0) {
            throw std::runtime_error("CviSession: input tensor size is zero");
        }
        auto buffer = tensor::CviTpuMemory::allocate(rt_handle_, bytes);
        check_cvi_rc(CVI_NN_SetTensorPhysicalAddr(&input_tensors_[i], buffer->physical_addr()),
                     "CVI_NN_SetTensorPhysicalAddr(input)");
        input_buffers_[static_cast<size_t>(i)] = std::move(buffer);
    }

    output_buffers_.resize(static_cast<size_t>(output_num_));
    for (int32_t i = 0; i < output_num_; ++i) {
        size_t bytes = CVI_NN_TensorSize(&output_tensors_[i]);
        if (bytes == 0) {
            throw std::runtime_error("CviSession: output tensor size is zero");
        }
        auto buffer = tensor::CviTpuMemory::allocate(rt_handle_, bytes);
        check_cvi_rc(CVI_NN_SetTensorPhysicalAddr(&output_tensors_[i], buffer->physical_addr()),
                     "CVI_NN_SetTensorPhysicalAddr(output)");
        output_buffers_[static_cast<size_t>(i)] = std::move(buffer);
    }
}

CviSession::TensorInfo CviSession::get_input_info(size_t index) const {
    if (index >= static_cast<size_t>(input_num_)) {
        throw std::out_of_range("CviSession::get_input_info - index out of range");
    }
    CVI_TENSOR* tensor = &input_tensors_[index];
    TensorInfo info;
    char* name = CVI_NN_TensorName(tensor);
    info.name = name ? name : "";
    info.shape = shape_from_cvi(CVI_NN_TensorShape(tensor));
    info.fmt = tensor->fmt;
    info.qscale = CVI_NN_TensorQuantScale(tensor);
    info.zero_point = CVI_NN_TensorQuantZeroPoint(tensor);
    info.pixel_format = tensor->pixel_format;
    info.mean = {tensor->mean[0], tensor->mean[1], tensor->mean[2]};
    info.scale = {tensor->scale[0], tensor->scale[1], tensor->scale[2]};
    return info;
}

CviSession::TensorInfo CviSession::get_output_info(size_t index) const {
    if (index >= static_cast<size_t>(output_num_)) {
        throw std::out_of_range("CviSession::get_output_info - index out of range");
    }
    CVI_TENSOR* tensor = &output_tensors_[index];
    TensorInfo info;
    char* name = CVI_NN_TensorName(tensor);
    info.name = name ? name : "";
    info.shape = shape_from_cvi(CVI_NN_TensorShape(tensor));
    info.fmt = tensor->fmt;
    info.qscale = CVI_NN_TensorQuantScale(tensor);
    info.zero_point = CVI_NN_TensorQuantZeroPoint(tensor);
    info.pixel_format = tensor->pixel_format;
    info.mean = {tensor->mean[0], tensor->mean[1], tensor->mean[2]};
    info.scale = {tensor->scale[0], tensor->scale[1], tensor->scale[2]};
    return info;
}

std::pair<std::vector<float>, std::vector<int64_t>>
CviSession::run(const float* input_data, const std::vector<int64_t>& input_shape) {
    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<int64_t> actual_input_shape;
    std::vector<float> padded_input;
    const float* input_ptr = nullptr;

    prepare_input(input_data, input_shape, &actual_input_shape, &padded_input, &input_ptr);

    CVI_TENSOR* input_tensor = &input_tensors_[0];
    size_t input_count = CVI_NN_TensorCount(input_tensor);
    int64_t input_elements = element_count(actual_input_shape);
    if (input_elements <= 0) {
        throw std::runtime_error("CviSession::run - input element count invalid");
    }
    if (input_count != static_cast<size_t>(input_elements)) {
        throw std::runtime_error("CviSession::run - input tensor size mismatch");
    }

    copy_input_to_device(input_ptr, input_elements);
    auto t1 = std::chrono::high_resolution_clock::now();

    check_cvi_rc(CVI_NN_Forward(model_, input_tensors_, input_num_,
                                output_tensors_, output_num_),
                 "CVI_NN_Forward");
    auto t2 = std::chrono::high_resolution_clock::now();

    if (output_buffers_.empty() || !output_buffers_[static_cast<size_t>(primary_output_index_)]) {
        throw std::runtime_error("CviSession::run - output buffer not initialized");
    }
    output_buffers_[static_cast<size_t>(primary_output_index_)]->invalidate_cache();

    auto output_data = read_output_tensor(primary_output_index_);
    auto t3 = std::chrono::high_resolution_clock::now();
    auto output_shape = get_output_shape(static_cast<size_t>(primary_output_index_));

    last_run_stats_.input_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    last_run_stats_.forward_ms =
        std::chrono::duration<double, std::milli>(t2 - t1).count();
    last_run_stats_.output_ms =
        std::chrono::duration<double, std::milli>(t3 - t2).count();

    return {std::move(output_data), output_shape};
}

void CviSession::run_all(const float* input_data,
                         const std::vector<int64_t>& input_shape,
                         std::vector<std::vector<float>>* outputs,
                         std::vector<std::vector<int64_t>>* output_shapes) {
    auto t0 = std::chrono::high_resolution_clock::now();
    if (!outputs || !output_shapes) {
        throw std::invalid_argument("CviSession::run_all - outputs and output_shapes cannot be null");
    }

    std::vector<int64_t> actual_input_shape;
    std::vector<float> padded_input;
    const float* input_ptr = nullptr;

    prepare_input(input_data, input_shape, &actual_input_shape, &padded_input, &input_ptr);

    CVI_TENSOR* input_tensor = &input_tensors_[0];
    size_t input_count = CVI_NN_TensorCount(input_tensor);
    int64_t input_elements = element_count(actual_input_shape);
    if (input_elements <= 0) {
        throw std::runtime_error("CviSession::run_all - input element count invalid");
    }
    if (input_count != static_cast<size_t>(input_elements)) {
        throw std::runtime_error("CviSession::run_all - input tensor size mismatch");
    }

    copy_input_to_device(input_ptr, input_elements);
    auto t1 = std::chrono::high_resolution_clock::now();

    check_cvi_rc(CVI_NN_Forward(model_, input_tensors_, input_num_,
                                output_tensors_, output_num_),
                 "CVI_NN_Forward");
    auto t2 = std::chrono::high_resolution_clock::now();

    outputs->clear();
    output_shapes->clear();
    outputs->reserve(static_cast<size_t>(output_num_));
    output_shapes->reserve(static_cast<size_t>(output_num_));

    for (int32_t i = 0; i < output_num_; ++i) {
        if (output_buffers_[static_cast<size_t>(i)]) {
            output_buffers_[static_cast<size_t>(i)]->invalidate_cache();
        }
        outputs->push_back(read_output_tensor(i));
        output_shapes->push_back(get_output_shape(static_cast<size_t>(i)));
    }

    auto t3 = std::chrono::high_resolution_clock::now();
    last_run_stats_.input_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    last_run_stats_.forward_ms =
        std::chrono::duration<double, std::milli>(t2 - t1).count();
    last_run_stats_.output_ms =
        std::chrono::duration<double, std::milli>(t3 - t2).count();
}

void CviSession::prepare_input(const float* input_data,
                               const std::vector<int64_t>& input_shape,
                               std::vector<int64_t>* actual_shape,
                               std::vector<float>* padded_input,
                               const float** input_ptr) const {
    if (!input_data) {
        throw std::invalid_argument("CviSession::prepare_input - input_data is null");
    }
    if (!actual_shape || !padded_input || !input_ptr) {
        throw std::invalid_argument("CviSession::prepare_input - output pointers are null");
    }
    if (input_shape.empty()) {
        throw std::invalid_argument("CviSession::prepare_input - input_shape is empty");
    }
    if (input_num_ != 1) {
        throw std::runtime_error("CviSession::prepare_input - only single-input models are supported");
    }
    if (output_num_ < 1) {
        throw std::runtime_error("CviSession::prepare_input - no outputs available");
    }

    std::vector<int64_t> model_input_shape = get_input_shape(0);
    *actual_shape = input_shape;
    *input_ptr = input_data;
    padded_input->clear();

    if (!model_input_shape.empty() && !shape_compatible(model_input_shape, *actual_shape)) {
        throw std::runtime_error("CviSession::prepare_input - input shape mismatch");
    }
}

void CviSession::copy_input_to_device(const float* input_ptr, int64_t input_elements) {
    if (input_buffers_.empty() || !input_buffers_[0]) {
        throw std::runtime_error("CviSession::copy_input_to_device - input buffer not initialized");
    }

    CVI_TENSOR* input_tensor = &input_tensors_[0];
    void* input_sys = input_buffers_[0]->data();
    if (!input_sys) {
        throw std::runtime_error("CviSession::copy_input_to_device - input buffer has no virtual address");
    }

    float qscale = CVI_NN_TensorQuantScale(input_tensor);
    int zero_point = CVI_NN_TensorQuantZeroPoint(input_tensor);

    switch (input_tensor->fmt) {
        case CVI_FMT_FP32: {
            std::memcpy(input_sys, input_ptr,
                        static_cast<size_t>(input_elements) * sizeof(float));
            break;
        }
        case CVI_FMT_BF16: {
            uint16_t* dst = static_cast<uint16_t*>(input_sys);
            for (int64_t i = 0; i < input_elements; ++i) {
                dst[i] = float_to_bf16(input_ptr[i]);
            }
            break;
        }
        case CVI_FMT_INT8: {
            int8_t* dst = static_cast<int8_t*>(input_sys);
            for (int64_t i = 0; i < input_elements; ++i) {
                dst[i] = quantize_value<int8_t>(input_ptr[i], qscale, zero_point);
            }
            break;
        }
        case CVI_FMT_UINT8: {
            uint8_t* dst = static_cast<uint8_t*>(input_sys);
            for (int64_t i = 0; i < input_elements; ++i) {
                dst[i] = quantize_value<uint8_t>(input_ptr[i], qscale, zero_point);
            }
            break;
        }
        case CVI_FMT_INT16: {
            int16_t* dst = static_cast<int16_t*>(input_sys);
            for (int64_t i = 0; i < input_elements; ++i) {
                dst[i] = quantize_value<int16_t>(input_ptr[i], qscale, zero_point);
            }
            break;
        }
        case CVI_FMT_UINT16: {
            uint16_t* dst = static_cast<uint16_t*>(input_sys);
            for (int64_t i = 0; i < input_elements; ++i) {
                dst[i] = quantize_value<uint16_t>(input_ptr[i], qscale, zero_point);
            }
            break;
        }
        case CVI_FMT_INT32: {
            int32_t* dst = static_cast<int32_t*>(input_sys);
            for (int64_t i = 0; i < input_elements; ++i) {
                dst[i] = quantize_value<int32_t>(input_ptr[i], qscale, zero_point);
            }
            break;
        }
        case CVI_FMT_UINT32: {
            uint32_t* dst = static_cast<uint32_t*>(input_sys);
            for (int64_t i = 0; i < input_elements; ++i) {
                dst[i] = quantize_value<uint32_t>(input_ptr[i], qscale, zero_point);
            }
            break;
        }
        default:
            throw std::runtime_error("CviSession::copy_input_to_device - unsupported input tensor format");
    }

    input_buffers_[0]->flush_cache();
}

std::vector<float> CviSession::read_output_tensor(int32_t output_index) const {
    if (output_index < 0 || output_index >= output_num_) {
        throw std::out_of_range("CviSession::read_output_tensor - output index out of range");
    }

    if (output_buffers_.empty() || !output_buffers_[static_cast<size_t>(output_index)]) {
        throw std::runtime_error("CviSession::read_output_tensor - output buffer not initialized");
    }

    CVI_TENSOR* output_tensor = &output_tensors_[output_index];
    size_t output_count = CVI_NN_TensorCount(output_tensor);
    void* output_sys = output_buffers_[static_cast<size_t>(output_index)]->data();
    if (!output_sys) {
        throw std::runtime_error("CviSession::read_output_tensor - output buffer has no virtual address");
    }

    std::vector<float> output_data(output_count);

    float out_qscale = CVI_NN_TensorQuantScale(output_tensor);
    int out_zero_point = CVI_NN_TensorQuantZeroPoint(output_tensor);

    switch (output_tensor->fmt) {
        case CVI_FMT_FP32: {
            std::memcpy(output_data.data(), output_sys, output_count * sizeof(float));
            break;
        }
        case CVI_FMT_BF16: {
            const uint16_t* src = static_cast<const uint16_t*>(output_sys);
            for (size_t i = 0; i < output_count; ++i) {
                output_data[i] = bf16_to_float(src[i]);
            }
            break;
        }
        case CVI_FMT_INT8: {
            const int8_t* src = static_cast<const int8_t*>(output_sys);
            for (size_t i = 0; i < output_count; ++i) {
                output_data[i] = dequantize_value<int8_t>(src[i], out_qscale, out_zero_point);
            }
            break;
        }
        case CVI_FMT_UINT8: {
            const uint8_t* src = static_cast<const uint8_t*>(output_sys);
            for (size_t i = 0; i < output_count; ++i) {
                output_data[i] = dequantize_value<uint8_t>(src[i], out_qscale, out_zero_point);
            }
            break;
        }
        case CVI_FMT_INT16: {
            const int16_t* src = static_cast<const int16_t*>(output_sys);
            for (size_t i = 0; i < output_count; ++i) {
                output_data[i] = dequantize_value<int16_t>(src[i], out_qscale, out_zero_point);
            }
            break;
        }
        case CVI_FMT_UINT16: {
            const uint16_t* src = static_cast<const uint16_t*>(output_sys);
            for (size_t i = 0; i < output_count; ++i) {
                output_data[i] = dequantize_value<uint16_t>(src[i], out_qscale, out_zero_point);
            }
            break;
        }
        case CVI_FMT_INT32: {
            const int32_t* src = static_cast<const int32_t*>(output_sys);
            for (size_t i = 0; i < output_count; ++i) {
                output_data[i] = dequantize_value<int32_t>(src[i], out_qscale, out_zero_point);
            }
            break;
        }
        case CVI_FMT_UINT32: {
            const uint32_t* src = static_cast<const uint32_t*>(output_sys);
            for (size_t i = 0; i < output_count; ++i) {
                output_data[i] = dequantize_value<uint32_t>(src[i], out_qscale, out_zero_point);
            }
            break;
        }
        default:
            throw std::runtime_error("CviSession::read_output_tensor - unsupported output tensor format");
    }

    return output_data;
}

} // namespace inference

#endif  // USE_CVI_TPU
