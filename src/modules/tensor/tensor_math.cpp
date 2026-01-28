#include "tensor.h"
#include "cpu_memory.h"
#include <cmath>
#include <stdexcept>

namespace tensor {

// ========== Level 2: 数学运算 ==========

Tensor Tensor::add(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::runtime_error("Shape mismatch for element-wise operation");
    }

    check_cpu();
    other.check_cpu();

    int64_t total = compute_size();
    auto new_storage = CpuMemory::allocate(total * sizeof(float));
    float* dst = static_cast<float*>(new_storage->data());

    // 快速路径：两者都是 contiguous
    if (contiguous_ && other.contiguous_) {
        const float* a = static_cast<const float*>(buffer_->data()) + offset_;
        const float* b = static_cast<const float*>(other.buffer_->data()) + other.offset_;

        for (int64_t i = 0; i < total; ++i) {
            dst[i] = a[i] + b[i];
        }
    } else {
        // 慢速路径：使用 stride-based 访问
        Tensor a_cont = contiguous();
        Tensor b_cont = other.contiguous();
        const float* a = static_cast<const float*>(a_cont.buffer_->data()) + a_cont.offset_;
        const float* b = static_cast<const float*>(b_cont.buffer_->data()) + b_cont.offset_;

        for (int64_t i = 0; i < total; ++i) {
            dst[i] = a[i] + b[i];
        }
    }

    return Tensor(new_storage, shape_, compute_strides(shape_), 0, true);
}

Tensor Tensor::add(float scalar) const {
    check_cpu();

    int64_t total = compute_size();
    auto new_storage = CpuMemory::allocate(total * sizeof(float));
    float* dst = static_cast<float*>(new_storage->data());

    // 快速路径：contiguous
    if (contiguous_) {
        const float* src = static_cast<const float*>(buffer_->data()) + offset_;
        for (int64_t i = 0; i < total; ++i) {
            dst[i] = src[i] + scalar;
        }
    } else {
        Tensor a = contiguous();
        const float* src = static_cast<const float*>(a.buffer_->data()) + a.offset_;
        for (int64_t i = 0; i < total; ++i) {
            dst[i] = src[i] + scalar;
        }
    }

    return Tensor(new_storage, shape_, compute_strides(shape_), 0, true);
}

Tensor Tensor::sub(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::runtime_error("Shape mismatch for element-wise operation");
    }

    check_cpu();
    other.check_cpu();

    int64_t total = compute_size();
    auto new_storage = CpuMemory::allocate(total * sizeof(float));
    float* dst = static_cast<float*>(new_storage->data());

    if (contiguous_ && other.contiguous_) {
        const float* a = static_cast<const float*>(buffer_->data()) + offset_;
        const float* b = static_cast<const float*>(other.buffer_->data()) + other.offset_;
        for (int64_t i = 0; i < total; ++i) {
            dst[i] = a[i] - b[i];
        }
    } else {
        Tensor a_cont = contiguous();
        Tensor b_cont = other.contiguous();
        const float* a = static_cast<const float*>(a_cont.buffer_->data()) + a_cont.offset_;
        const float* b = static_cast<const float*>(b_cont.buffer_->data()) + b_cont.offset_;
        for (int64_t i = 0; i < total; ++i) {
            dst[i] = a[i] - b[i];
        }
    }

    return Tensor(new_storage, shape_, compute_strides(shape_), 0, true);
}

Tensor Tensor::sub(float scalar) const {
    check_cpu();

    int64_t total = compute_size();
    auto new_storage = CpuMemory::allocate(total * sizeof(float));
    float* dst = static_cast<float*>(new_storage->data());

    if (contiguous_) {
        const float* src = static_cast<const float*>(buffer_->data()) + offset_;
        for (int64_t i = 0; i < total; ++i) {
            dst[i] = src[i] - scalar;
        }
    } else {
        Tensor a = contiguous();
        const float* src = static_cast<const float*>(a.buffer_->data()) + a.offset_;
        for (int64_t i = 0; i < total; ++i) {
            dst[i] = src[i] - scalar;
        }
    }

    return Tensor(new_storage, shape_, compute_strides(shape_), 0, true);
}

Tensor Tensor::mul(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::runtime_error("Shape mismatch for element-wise operation");
    }

    check_cpu();
    other.check_cpu();

    int64_t total = compute_size();
    auto new_storage = CpuMemory::allocate(total * sizeof(float));
    float* dst = static_cast<float*>(new_storage->data());

    if (contiguous_ && other.contiguous_) {
        const float* a = static_cast<const float*>(buffer_->data()) + offset_;
        const float* b = static_cast<const float*>(other.buffer_->data()) + other.offset_;
        for (int64_t i = 0; i < total; ++i) {
            dst[i] = a[i] * b[i];
        }
    } else {
        Tensor a_cont = contiguous();
        Tensor b_cont = other.contiguous();
        const float* a = static_cast<const float*>(a_cont.buffer_->data()) + a_cont.offset_;
        const float* b = static_cast<const float*>(b_cont.buffer_->data()) + b_cont.offset_;
        for (int64_t i = 0; i < total; ++i) {
            dst[i] = a[i] * b[i];
        }
    }

    return Tensor(new_storage, shape_, compute_strides(shape_), 0, true);
}

Tensor Tensor::mul(float scalar) const {
    check_cpu();

    int64_t total = compute_size();
    auto new_storage = CpuMemory::allocate(total * sizeof(float));
    float* dst = static_cast<float*>(new_storage->data());

    if (contiguous_) {
        const float* src = static_cast<const float*>(buffer_->data()) + offset_;
        for (int64_t i = 0; i < total; ++i) {
            dst[i] = src[i] * scalar;
        }
    } else {
        Tensor a = contiguous();
        const float* src = static_cast<const float*>(a.buffer_->data()) + a.offset_;
        for (int64_t i = 0; i < total; ++i) {
            dst[i] = src[i] * scalar;
        }
    }

    return Tensor(new_storage, shape_, compute_strides(shape_), 0, true);
}

Tensor Tensor::div(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::runtime_error("Shape mismatch for element-wise operation");
    }

    check_cpu();
    other.check_cpu();

    int64_t total = compute_size();
    auto new_storage = CpuMemory::allocate(total * sizeof(float));
    float* dst = static_cast<float*>(new_storage->data());

    if (contiguous_ && other.contiguous_) {
        const float* a = static_cast<const float*>(buffer_->data()) + offset_;
        const float* b = static_cast<const float*>(other.buffer_->data()) + other.offset_;
        for (int64_t i = 0; i < total; ++i) {
            if (std::abs(b[i]) < 1e-7f) {
                throw std::runtime_error("Division by zero");
            }
            dst[i] = a[i] / b[i];
        }
    } else {
        Tensor a_cont = contiguous();
        Tensor b_cont = other.contiguous();
        const float* a = static_cast<const float*>(a_cont.buffer_->data()) + a_cont.offset_;
        const float* b = static_cast<const float*>(b_cont.buffer_->data()) + b_cont.offset_;
        for (int64_t i = 0; i < total; ++i) {
            if (std::abs(b[i]) < 1e-7f) {
                throw std::runtime_error("Division by zero");
            }
            dst[i] = a[i] / b[i];
        }
    }

    return Tensor(new_storage, shape_, compute_strides(shape_), 0, true);
}

Tensor Tensor::div(float scalar) const {
    if (std::abs(scalar) < 1e-7f) {
        throw std::runtime_error("Division by zero");
    }

    check_cpu();

    int64_t total = compute_size();
    auto new_storage = CpuMemory::allocate(total * sizeof(float));
    float* dst = static_cast<float*>(new_storage->data());
    float inv_scalar = 1.0f / scalar;

    if (contiguous_) {
        const float* src = static_cast<const float*>(buffer_->data()) + offset_;
        for (int64_t i = 0; i < total; ++i) {
            dst[i] = src[i] * inv_scalar;
        }
    } else {
        Tensor a_cont = contiguous();
        const float* src = static_cast<const float*>(a_cont.buffer_->data()) + a_cont.offset_;
        for (int64_t i = 0; i < total; ++i) {
            dst[i] = src[i] * inv_scalar;
        }
    }

    return Tensor(new_storage, shape_, compute_strides(shape_), 0, true);
}

// ========== In-place 操作 ==========

Tensor& Tensor::add_(float scalar) {
    check_cpu();
    if (!contiguous_) {
        throw std::runtime_error("In-place ops require contiguous tensor");
    }

    float* ptr = static_cast<float*>(buffer_->data()) + offset_;
    int64_t total = compute_size();

    for (int64_t i = 0; i < total; ++i) {
        ptr[i] += scalar;
    }

    return *this;
}

Tensor& Tensor::sub_(float scalar) {
    check_cpu();
    if (!contiguous_) {
        throw std::runtime_error("In-place ops require contiguous tensor");
    }

    float* ptr = static_cast<float*>(buffer_->data()) + offset_;
    int64_t total = compute_size();

    for (int64_t i = 0; i < total; ++i) {
        ptr[i] -= scalar;
    }

    return *this;
}

Tensor& Tensor::mul_(float scalar) {
    check_cpu();
    if (!contiguous_) {
        throw std::runtime_error("In-place ops require contiguous tensor");
    }

    float* ptr = static_cast<float*>(buffer_->data()) + offset_;
    int64_t total = compute_size();

    // 循环展开
    int64_t i = 0;
    for (; i + 4 <= total; i += 4) {
        ptr[i] *= scalar;
        ptr[i+1] *= scalar;
        ptr[i+2] *= scalar;
        ptr[i+3] *= scalar;
    }
    for (; i < total; ++i) {
        ptr[i] *= scalar;
    }

    return *this;
}

Tensor& Tensor::div_(float scalar) {
    if (std::abs(scalar) < 1e-7f) {
        throw std::runtime_error("Division by zero");
    }

    check_cpu();
    if (!contiguous_) {
        throw std::runtime_error("In-place ops require contiguous tensor");
    }

    float* ptr = static_cast<float*>(buffer_->data()) + offset_;
    int64_t total = compute_size();
    float inv_scalar = 1.0f / scalar;

    for (int64_t i = 0; i < total; ++i) {
        ptr[i] *= inv_scalar;
    }

    return *this;
}

// ========== Clamp ==========

Tensor Tensor::clamp(float min_val, float max_val) const {
    check_cpu();

    int64_t total = compute_size();
    auto new_storage = CpuMemory::allocate(total * sizeof(float));
    float* dst = static_cast<float*>(new_storage->data());

    if (contiguous_) {
        const float* src = static_cast<const float*>(buffer_->data()) + offset_;
        for (int64_t i = 0; i < total; ++i) {
            float v = src[i];
            if (v < min_val) v = min_val;
            if (v > max_val) v = max_val;
            dst[i] = v;
        }
    } else {
        Tensor a = contiguous();
        const float* src = static_cast<const float*>(a.buffer_->data()) + a.offset_;
        for (int64_t i = 0; i < total; ++i) {
            float v = src[i];
            if (v < min_val) v = min_val;
            if (v > max_val) v = max_val;
            dst[i] = v;
        }
    }

    return Tensor(new_storage, shape_, compute_strides(shape_), 0, true);
}

} // namespace tensor
