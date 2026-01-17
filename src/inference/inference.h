#pragma once

#include <vector>
#include <string>
#include <memory>

// Generic inference engine abstraction
// Provides ONNX Runtime session management for any model
namespace inference {

#ifdef USE_ONNX_RUNTIME

#include "onnxruntime_cxx_api.h"

// ============ ONNX Session ============

class OnnxSession {
public:
    explicit OnnxSession(const std::string& model_path, int num_threads = 4);
    ~OnnxSession() = default;

    // Run inference on input data
    // Returns: (output_data, output_shape)
    std::pair<std::vector<float>, std::vector<int64_t>>
    run(const float* input_data, const std::vector<int64_t>& input_shape);

    // Get model input/output tensor info
    std::vector<int64_t> get_input_shape(size_t index = 0) const;
    std::vector<int64_t> get_output_shape(size_t index = 0) const;

    // Get model input/output names
    const std::vector<std::string>& get_input_names() const { return input_names_; }
    const std::vector<std::string>& get_output_names() const { return output_names_; }

private:
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    Ort::AllocatorWithDefaultOptions allocator_;

    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<const char*> input_names_cstr_;
    std::vector<const char*> output_names_cstr_;
};

#endif  // USE_ONNX_RUNTIME

} // namespace inference
