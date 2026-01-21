#pragma once

#include <string>
#include <utility>
#include <vector>

namespace inference {

class InferenceSession {
public:
    virtual ~InferenceSession() = default;

    virtual std::pair<std::vector<float>, std::vector<int64_t>>
    run(const float* input_data, const std::vector<int64_t>& input_shape) = 0;

    virtual std::vector<int64_t> get_input_shape(size_t index = 0) const = 0;
    virtual std::vector<int64_t> get_output_shape(size_t index = 0) const = 0;

    virtual const std::vector<std::string>& get_input_names() const = 0;
    virtual const std::vector<std::string>& get_output_names() const = 0;

    virtual const char* backend_name() const = 0;
};

} // namespace inference
