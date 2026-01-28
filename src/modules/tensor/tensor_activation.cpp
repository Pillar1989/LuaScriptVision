#include "tensor.h"
#include "cpu_memory.h"
#include <cmath>
#include <stdexcept>
#include <vector>

namespace tensor {

// ========== Activation 函数 ==========

Tensor Tensor::sigmoid() const {
    check_cpu();
    Tensor a = contiguous();

    std::vector<float> result_data(compute_size());
    const float* src = static_cast<const float*>(a.buffer_->data()) + a.offset_;

    for (int64_t i = 0; i < compute_size(); ++i) {
        result_data[i] = 1.0f / (1.0f + std::exp(-src[i]));
    }

    return Tensor(std::move(result_data), shape_);
}

Tensor Tensor::softmax(int axis) const {
    check_cpu();
    int ax = axis;
    if (ax < 0) ax += static_cast<int>(shape_.size());
    if (ax < 0 || ax >= static_cast<int>(shape_.size())) {
        throw std::runtime_error("Axis out of range");
    }

    Tensor a = contiguous();
    const float* src = static_cast<const float*>(a.buffer_->data()) + a.offset_;

    // Calculate outer_size (product of dims before axis)
    // Calculate axis_size (size of the axis dim)
    // Calculate inner_size (product of dims after axis)
    int64_t outer_size = 1;
    for (int i = 0; i < ax; ++i) {
        outer_size *= shape_[i];
    }
    int64_t axis_size = shape_[ax];
    int64_t inner_size = 1;
    for (int i = ax + 1; i < static_cast<int>(shape_.size()); ++i) {
        inner_size *= shape_[i];
    }

    std::vector<float> result_data(compute_size());

    // For each combination of outer and inner indices
    for (int64_t outer = 0; outer < outer_size; ++outer) {
        for (int64_t inner = 0; inner < inner_size; ++inner) {
            // Find max along axis for numerical stability
            float max_val = -1e30f;
            for (int64_t k = 0; k < axis_size; ++k) {
                int64_t idx = (outer * axis_size + k) * inner_size + inner;
                if (src[idx] > max_val) max_val = src[idx];
            }

            // Compute exp and sum
            float sum = 0.0f;
            for (int64_t k = 0; k < axis_size; ++k) {
                int64_t idx = (outer * axis_size + k) * inner_size + inner;
                float e = std::exp(src[idx] - max_val);
                result_data[idx] = e;
                sum += e;
            }

            // Normalize
            float inv_sum = 1.0f / sum;
            for (int64_t k = 0; k < axis_size; ++k) {
                int64_t idx = (outer * axis_size + k) * inner_size + inner;
                result_data[idx] *= inv_sum;
            }
        }
    }

    return Tensor(std::move(result_data), shape_);
}

Tensor Tensor::exp_() const {
    check_cpu();
    Tensor a = contiguous();

    std::vector<float> result_data(compute_size());
    const float* src = static_cast<const float*>(a.buffer_->data()) + a.offset_;

    for (int64_t i = 0; i < compute_size(); ++i) {
        result_data[i] = std::exp(src[i]);
    }

    return Tensor(std::move(result_data), shape_);
}

Tensor Tensor::log_() const {
    check_cpu();
    Tensor a = contiguous();

    std::vector<float> result_data(compute_size());
    const float* src = static_cast<const float*>(a.buffer_->data()) + a.offset_;

    for (int64_t i = 0; i < compute_size(); ++i) {
        result_data[i] = std::log(src[i]);
    }

    return Tensor(std::move(result_data), shape_);
}

} // namespace tensor
