#pragma once

#ifdef USE_ONNX_RUNTIME

#include <memory>
#include <string>
#include <vector>

#include "inference_session.h"
#include "onnxruntime_cxx_api.h"

namespace inference {

class OnnxSession : public InferenceSession {
public:
    explicit OnnxSession(const std::string& model_path, int num_threads = 4);
    ~OnnxSession() override = default;

    std::pair<std::vector<float>, std::vector<int64_t>>
    run(const float* input_data, const std::vector<int64_t>& input_shape) override;

    std::vector<int64_t> get_input_shape(size_t index = 0) const override;
    std::vector<int64_t> get_output_shape(size_t index = 0) const override;

    const std::vector<std::string>& get_input_names() const override { return input_names_; }
    const std::vector<std::string>& get_output_names() const override { return output_names_; }

    const char* backend_name() const override { return "onnxruntime"; }

private:
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    Ort::AllocatorWithDefaultOptions allocator_;

    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<const char*> input_names_cstr_;
    std::vector<const char*> output_names_cstr_;
};

} // namespace inference

#endif  // USE_ONNX_RUNTIME
