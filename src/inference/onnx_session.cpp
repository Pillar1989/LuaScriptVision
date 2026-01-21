#include "onnx_session.h"

#ifdef USE_ONNX_RUNTIME

#include <algorithm>
#include <numeric>
#include <stdexcept>

#if __has_include("onnxruntime_float16.h")
#include "onnxruntime_float16.h"
#endif

namespace inference {

OnnxSession::OnnxSession(const std::string& model_path, int num_threads) {
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "inference");

    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(num_threads);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_options);

    size_t num_input_nodes = session_->GetInputCount();
    for (size_t i = 0; i < num_input_nodes; i++) {
        auto input_name = session_->GetInputNameAllocated(i, allocator_);
        input_names_.push_back(input_name.get());
        input_names_cstr_.push_back(input_names_.back().c_str());
    }

    size_t num_output_nodes = session_->GetOutputCount();
    for (size_t i = 0; i < num_output_nodes; i++) {
        auto output_name = session_->GetOutputNameAllocated(i, allocator_);
        output_names_.push_back(output_name.get());
        output_names_cstr_.push_back(output_names_.back().c_str());
    }
}

std::pair<std::vector<float>, std::vector<int64_t>>
OnnxSession::run(const float* input_data, const std::vector<int64_t>& input_shape) {
    auto model_input_shape = get_input_shape(0);
    std::vector<int64_t> actual_input_shape = input_shape;
    std::vector<float> padded_input;
    const float* input_ptr = input_data;

    if (model_input_shape.size() >= 4 && actual_input_shape.size() >= 4) {
        int64_t model_h = model_input_shape[2];
        int64_t model_w = model_input_shape[3];
        int64_t input_h = actual_input_shape[2];
        int64_t input_w = actual_input_shape[3];

        if (model_h > 0 && model_w > 0) {
            if (input_h < model_h || input_w < model_w) {
                size_t padded_size = 1 * 3 * model_h * model_w;
                padded_input.resize(padded_size, 114.0f / 255.0f);

                for (int c = 0; c < 3; ++c) {
                    for (int h = 0; h < input_h; ++h) {
                        const float* src = input_data + c * input_h * input_w + h * input_w;
                        float* dst = padded_input.data() + c * model_h * model_w + h * model_w;
                        std::copy(src, src + input_w, dst);
                    }
                }

                input_ptr = padded_input.data();
                actual_input_shape[2] = model_h;
                actual_input_shape[3] = model_w;
            }
        }
    }

    auto input_type_info = session_->GetInputTypeInfo(0);
    auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType input_type = input_tensor_info.GetElementType();

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        const_cast<float*>(input_ptr),
        std::accumulate(actual_input_shape.begin(), actual_input_shape.end(), 1LL, std::multiplies<int64_t>()),
        actual_input_shape.data(),
        actual_input_shape.size()
    );

    std::vector<Ort::Float16_t> fp16_input_values;
    if (input_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        size_t input_size = std::accumulate(actual_input_shape.begin(), actual_input_shape.end(), 1LL,
                                            std::multiplies<int64_t>());
        fp16_input_values.reserve(input_size);
        for (size_t i = 0; i < input_size; ++i) {
            fp16_input_values.emplace_back(input_ptr[i]);
        }
        input_tensor = Ort::Value::CreateTensor<Ort::Float16_t>(
            memory_info,
            fp16_input_values.data(),
            fp16_input_values.size(),
            actual_input_shape.data(),
            actual_input_shape.size()
        );
    }

    auto output_tensors = session_->Run(
        Ort::RunOptions{nullptr},
        input_names_cstr_.data(),
        &input_tensor,
        1,
        output_names_cstr_.data(),
        output_names_cstr_.size()
    );

    auto output_info = output_tensors[0].GetTensorTypeAndShapeInfo();
    auto output_shape = output_info.GetShape();
    size_t output_size = output_info.GetElementCount();

    std::vector<float> output_data(output_size);

    if (output_info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        const Ort::Float16_t* fp16_out = output_tensors[0].GetTensorData<Ort::Float16_t>();
        for (size_t i = 0; i < output_size; ++i) {
            output_data[i] = fp16_out[i].ToFloat();
        }
    } else {
        const float* float_out = output_tensors[0].GetTensorData<float>();
        std::copy(float_out, float_out + output_size, output_data.begin());
    }

    return {std::move(output_data), output_shape};
}

std::vector<int64_t> OnnxSession::get_input_shape(size_t index) const {
    auto type_info = session_->GetInputTypeInfo(index);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    return tensor_info.GetShape();
}

std::vector<int64_t> OnnxSession::get_output_shape(size_t index) const {
    auto type_info = session_->GetOutputTypeInfo(index);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    return tensor_info.GetShape();
}

} // namespace inference

#endif  // USE_ONNX_RUNTIME
