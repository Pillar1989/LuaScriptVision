#include "tensor.h"
#include "cpu_memory.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cfloat>
#include <stdexcept>
#include <sstream>
#include <vector>

#include "LuaIntf.h"

namespace tensor {

// ========== 辅助方法 ==========

float Tensor::get_item(const std::vector<int64_t>& indices) const {
    return at(indices);
}

void Tensor::set_item(const std::vector<int64_t>& indices, float value) {
    check_cpu();
    int64_t offset = compute_offset(indices);
    float* ptr = static_cast<float*>(buffer_->data());
    ptr[offset] = value;
}

std::string Tensor::to_string(int max_elements) const {
    std::ostringstream oss;
    oss << "Tensor(shape=[";
    for (size_t i = 0; i < shape_.size(); ++i) {
        oss << shape_[i];
        if (i < shape_.size() - 1) oss << ", ";
    }
    oss << "]";

    if (buffer_->device() == DeviceType::CPU) {
        oss << ", data=[";
        Tensor cont = contiguous();
        const float* ptr = static_cast<const float*>(cont.buffer_->data()) + cont.offset_;
        int64_t total = compute_size();
        int64_t show = std::min(static_cast<int64_t>(max_elements), total);

        for (int64_t i = 0; i < show; ++i) {
            oss << ptr[i];
            if (i < show - 1) oss << ", ";
        }
        if (total > show) {
            oss << ", ...";
        }
        oss << "]";
    }

    oss << ", device=" << device_type_to_string(buffer_->device()) << ")";
    return oss.str();
}

// ========== Reduction 操作 ==========

Tensor Tensor::sum(int axis, bool keepdims) const {
    check_cpu();
    Tensor a = contiguous();

    if (axis == -1) {
        float total = 0.0f;
        const float* src = static_cast<const float*>(a.buffer_->data()) + a.offset_;
        for (int64_t i = 0; i < compute_size(); ++i) {
            total += src[i];
        }

        if (keepdims) {
            std::vector<int64_t> new_shape(shape_.size(), 1);
            return Tensor(std::vector<float>{total}, new_shape);
        } else {
            return Tensor(std::vector<float>{total}, {1});
        }
    }

    int ax = axis;
    if (ax < 0) ax += shape_.size();
    if (ax < 0 || ax >= static_cast<int>(shape_.size())) {
        throw std::runtime_error("Axis out of range");
    }

    std::vector<int64_t> new_shape;
    for (int i = 0; i < static_cast<int>(shape_.size()); ++i) {
        if (i != ax) {
            new_shape.push_back(shape_[i]);
        } else if (keepdims) {
            new_shape.push_back(1);
        }
    }
    if (new_shape.empty()) new_shape.push_back(1);

    int64_t outer_size = 1;
    for (int i = 0; i < ax; ++i) {
        outer_size *= shape_[i];
    }
    int64_t axis_size = shape_[ax];
    int64_t inner_size = 1;
    for (int i = ax + 1; i < static_cast<int>(shape_.size()); ++i) {
        inner_size *= shape_[i];
    }

    std::vector<float> result_data(outer_size * inner_size, 0.0f);
    const float* src = static_cast<const float*>(a.buffer_->data()) + a.offset_;

    for (int64_t i = 0; i < outer_size; ++i) {
        for (int64_t j = 0; j < axis_size; ++j) {
            for (int64_t k = 0; k < inner_size; ++k) {
                int64_t src_idx = (i * axis_size + j) * inner_size + k;
                int64_t dst_idx = i * inner_size + k;
                result_data[dst_idx] += src[src_idx];
            }
        }
    }

    return Tensor(std::move(result_data), new_shape);
}

Tensor Tensor::mean(int axis, bool keepdims) const {
    Tensor sum_result = sum(axis, keepdims);

    int64_t count;
    if (axis == -1) {
        count = compute_size();
    } else {
        int ax = axis;
        if (ax < 0) ax += shape_.size();
        count = shape_[ax];
    }

    return sum_result.div(static_cast<float>(count));
}

Tensor Tensor::max(int axis, bool keepdims) const {
    check_cpu();
    Tensor a = contiguous();

    if (axis == -1) {
        const float* src = static_cast<const float*>(a.buffer_->data()) + a.offset_;
        float max_val = src[0];
        for (int64_t i = 1; i < compute_size(); ++i) {
            max_val = std::max(max_val, src[i]);
        }

        if (keepdims) {
            std::vector<int64_t> new_shape(shape_.size(), 1);
            return Tensor(std::vector<float>{max_val}, new_shape);
        } else {
            return Tensor(std::vector<float>{max_val}, {1});
        }
    }

    int ax = axis;
    if (ax < 0) ax += shape_.size();
    if (ax < 0 || ax >= static_cast<int>(shape_.size())) {
        throw std::runtime_error("Axis out of range");
    }

    std::vector<int64_t> new_shape;
    for (int i = 0; i < static_cast<int>(shape_.size()); ++i) {
        if (i != ax) {
            new_shape.push_back(shape_[i]);
        } else if (keepdims) {
            new_shape.push_back(1);
        }
    }
    if (new_shape.empty()) new_shape.push_back(1);

    int64_t outer_size = 1;
    for (int i = 0; i < ax; ++i) {
        outer_size *= shape_[i];
    }
    int64_t axis_size = shape_[ax];
    int64_t inner_size = 1;
    for (int i = ax + 1; i < static_cast<int>(shape_.size()); ++i) {
        inner_size *= shape_[i];
    }

    std::vector<float> result_data(outer_size * inner_size);
    const float* src = static_cast<const float*>(a.buffer_->data()) + a.offset_;

    for (int64_t i = 0; i < outer_size; ++i) {
        for (int64_t k = 0; k < inner_size; ++k) {
            int64_t src_idx = (i * axis_size + 0) * inner_size + k;
            float max_val = src[src_idx];

            for (int64_t j = 1; j < axis_size; ++j) {
                src_idx = (i * axis_size + j) * inner_size + k;
                max_val = std::max(max_val, src[src_idx]);
            }

            int64_t dst_idx = i * inner_size + k;
            result_data[dst_idx] = max_val;
        }
    }

    return Tensor(std::move(result_data), new_shape);
}

Tensor Tensor::min(int axis, bool keepdims) const {
    check_cpu();
    Tensor a = contiguous();

    if (axis == -1) {
        const float* src = static_cast<const float*>(a.buffer_->data()) + a.offset_;
        float min_val = src[0];
        for (int64_t i = 1; i < compute_size(); ++i) {
            min_val = std::min(min_val, src[i]);
        }

        if (keepdims) {
            std::vector<int64_t> new_shape(shape_.size(), 1);
            return Tensor(std::vector<float>{min_val}, new_shape);
        } else {
            return Tensor(std::vector<float>{min_val}, {1});
        }
    }

    int ax = axis;
    if (ax < 0) ax += shape_.size();
    if (ax < 0 || ax >= static_cast<int>(shape_.size())) {
        throw std::runtime_error("Axis out of range");
    }

    std::vector<int64_t> new_shape;
    for (int i = 0; i < static_cast<int>(shape_.size()); ++i) {
        if (i != ax) {
            new_shape.push_back(shape_[i]);
        } else if (keepdims) {
            new_shape.push_back(1);
        }
    }
    if (new_shape.empty()) new_shape.push_back(1);

    int64_t outer_size = 1;
    for (int i = 0; i < ax; ++i) {
        outer_size *= shape_[i];
    }
    int64_t axis_size = shape_[ax];
    int64_t inner_size = 1;
    for (int i = ax + 1; i < static_cast<int>(shape_.size()); ++i) {
        inner_size *= shape_[i];
    }

    std::vector<float> result_data(outer_size * inner_size);
    const float* src = static_cast<const float*>(a.buffer_->data()) + a.offset_;

    for (int64_t i = 0; i < outer_size; ++i) {
        for (int64_t k = 0; k < inner_size; ++k) {
            int64_t src_idx = (i * axis_size + 0) * inner_size + k;
            float min_val = src[src_idx];

            for (int64_t j = 1; j < axis_size; ++j) {
                src_idx = (i * axis_size + j) * inner_size + k;
                min_val = std::min(min_val, src[src_idx]);
            }

            int64_t dst_idx = i * inner_size + k;
            result_data[dst_idx] = min_val;
        }
    }

    return Tensor(std::move(result_data), new_shape);
}

// ========== 向量化过滤操作 ==========

// ========== Argmax/Argmin ==========

LuaIntf::LuaRef Tensor::argmax_lua(lua_State* L, int axis) const {
    check_cpu();
    Tensor a = contiguous();

    if (axis == -1) {
        const float* src = static_cast<const float*>(a.buffer_->data()) + a.offset_;
        int64_t max_idx = 0;
        float max_val = src[0];

        for (int64_t i = 1; i < compute_size(); ++i) {
            if (src[i] > max_val) {
                max_val = src[i];
                max_idx = i;
            }
        }

        return LuaIntf::LuaRef::fromValue(L, max_idx);
    }

    int ax = axis;
    if (ax < 0) ax += shape_.size();
    if (ax < 0 || ax >= static_cast<int>(shape_.size())) {
        throw std::runtime_error("Axis out of range");
    }

    int64_t outer_size = 1;
    for (int i = 0; i < ax; ++i) {
        outer_size *= shape_[i];
    }
    int64_t axis_size = shape_[ax];
    int64_t inner_size = 1;
    for (int i = ax + 1; i < static_cast<int>(shape_.size()); ++i) {
        inner_size *= shape_[i];
    }

    const float* src = static_cast<const float*>(a.buffer_->data()) + a.offset_;
    LuaIntf::LuaRef result = LuaIntf::LuaRef::createTable(L);
    int lua_idx = 1;

    for (int64_t i = 0; i < outer_size; ++i) {
        for (int64_t k = 0; k < inner_size; ++k) {
            int64_t src_idx = (i * axis_size + 0) * inner_size + k;
            float max_val = src[src_idx];
            int64_t max_pos = 0;

            for (int64_t j = 1; j < axis_size; ++j) {
                src_idx = (i * axis_size + j) * inner_size + k;
                if (src[src_idx] > max_val) {
                    max_val = src[src_idx];
                    max_pos = j;
                }
            }

            result[lua_idx++] = max_pos;
        }
    }

    return result;
}

LuaIntf::LuaRef Tensor::argmin_lua(lua_State* L, int axis) const {
    check_cpu();
    Tensor a = contiguous();

    if (axis == -1) {
        const float* src = static_cast<const float*>(a.buffer_->data()) + a.offset_;
        int64_t min_idx = 0;
        float min_val = src[0];

        for (int64_t i = 1; i < compute_size(); ++i) {
            if (src[i] < min_val) {
                min_val = src[i];
                min_idx = i;
            }
        }

        return LuaIntf::LuaRef::fromValue(L, min_idx);
    }

    int ax = axis;
    if (ax < 0) ax += shape_.size();

    int64_t outer_size = 1;
    for (int i = 0; i < ax; ++i) {
        outer_size *= shape_[i];
    }
    int64_t axis_size = shape_[ax];
    int64_t inner_size = 1;
    for (int i = ax + 1; i < static_cast<int>(shape_.size()); ++i) {
        inner_size *= shape_[i];
    }

    const float* src = static_cast<const float*>(a.buffer_->data()) + a.offset_;
    LuaIntf::LuaRef result = LuaIntf::LuaRef::createTable(L);
    int lua_idx = 1;

    for (int64_t i = 0; i < outer_size; ++i) {
        for (int64_t k = 0; k < inner_size; ++k) {
            int64_t src_idx = (i * axis_size + 0) * inner_size + k;
            float min_val = src[src_idx];
            int64_t min_pos = 0;

            for (int64_t j = 1; j < axis_size; ++j) {
                src_idx = (i * axis_size + j) * inner_size + k;
                if (src[src_idx] < min_val) {
                    min_val = src[src_idx];
                    min_pos = j;
                }
            }

            result[lua_idx++] = min_pos;
        }
    }

    return result;
}

// ========== Max with Argmax (Fused) ==========

LuaIntf::LuaRef Tensor::max_with_argmax(lua_State* L, int axis) const {
#ifdef ENABLE_RV_THEAD
    // Special case: 2D tensor with axis=0 -> use RVV optimized version
    if (ndim() == 2 && axis == 0) {
        return max_with_argmax_thead(L, axis);
    }
#endif

    // Generic CPU implementation for other cases
    check_cpu();
    Tensor a = contiguous();

    int ax = axis;
    if (ax < 0) ax += shape_.size();
    if (ax < 0 || ax >= static_cast<int>(shape_.size())) {
        throw std::runtime_error("Axis out of range");
    }

    // 计算维度
    std::vector<int64_t> new_shape;
    for (int i = 0; i < static_cast<int>(shape_.size()); ++i) {
        if (i != ax) {
            new_shape.push_back(shape_[i]);
        }
    }
    if (new_shape.empty()) new_shape.push_back(1);

    int64_t outer_size = 1;
    for (int i = 0; i < ax; ++i) {
        outer_size *= shape_[i];
    }
    int64_t axis_size = shape_[ax];
    int64_t inner_size = 1;
    for (int i = ax + 1; i < static_cast<int>(shape_.size()); ++i) {
        inner_size *= shape_[i];
    }

    int64_t result_size = outer_size * inner_size;

    // 预分配结果
    std::vector<float> max_values(result_size);
    const float* src = static_cast<const float*>(a.buffer_->data()) + a.offset_;

    // 预分配 Lua table
    lua_createtable(L, static_cast<int>(result_size), 0);
    LuaIntf::LuaRef indices = LuaIntf::LuaRef::popFromStack(L);

    int lua_idx = 1;

    // 单次遍历，同时计算 max 和 argmax
    for (int64_t i = 0; i < outer_size; ++i) {
        for (int64_t k = 0; k < inner_size; ++k) {
            int64_t result_idx = i * inner_size + k;
            int64_t src_idx = (i * axis_size + 0) * inner_size + k;

            float max_val = src[src_idx];
            int64_t max_pos = 0;

            for (int64_t j = 1; j < axis_size; ++j) {
                src_idx = (i * axis_size + j) * inner_size + k;
                if (src[src_idx] > max_val) {
                    max_val = src[src_idx];
                    max_pos = j;
                }
            }

            max_values[result_idx] = max_val;
            indices[lua_idx++] = max_pos;
        }
    }

    // 创建结果 Tensor
    Tensor max_tensor(std::move(max_values), new_shape);

    // 返回 {values = Tensor, indices = table}
    LuaIntf::LuaRef result = LuaIntf::LuaRef::createTable(L);
    result["values"] = max_tensor;
    result["indices"] = indices;

    return result;
}

// ========== 融合 Softmax + Weighted Sum ==========

Tensor Tensor::weighted_sum(int axis, const Tensor& weights) const {
#ifdef ENABLE_RV_THEAD
    // 使用玄铁优化版本
    return weighted_sum_thead(axis, weights);
#else
    // CPU fallback
    return weighted_sum_cpu(axis, weights);
#endif
}

Tensor Tensor::weighted_sum_cpu(int axis, const Tensor& weights) const {
    check_cpu();

    // 确保连续内存
    if (!contiguous_) {
        return contiguous().weighted_sum_cpu(axis, weights);
    }

    // 验证输入
    if (ndim() != 3 || axis != 1) {
        throw std::runtime_error(
            "weighted_sum_cpu: requires 3D tensor with axis=1");
    }

    int64_t outer = shape_[0];
    int64_t axis_size = shape_[1];
    int64_t inner = shape_[2];

    if (weights.size() != axis_size) {
        throw std::runtime_error(
            "weighted_sum_cpu: weights size must match axis dimension");
    }

    const float* input_data = data();
    const float* weight_data = weights.data();

    // 分配输出 [outer, inner]
    auto out_storage = CpuMemory::allocate(outer * inner * sizeof(float));
    float* output_data = static_cast<float*>(out_storage->data());

    // 处理每个 (outer, inner) 组合
    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t i = 0; i < inner; ++i) {
            const float* ptr = input_data + o * axis_size * inner + i;
            const int64_t stride = inner;

            // Step 1: 找 max（数值稳定性）
            float max_val = -FLT_MAX;
            for (int64_t a = 0; a < axis_size; ++a) {
                max_val = std::max(max_val, ptr[a * stride]);
            }

            // Step 2 & 3: exp + weighted sum
            float sum_exp = 0.0f;
            float weighted_sum = 0.0f;

            for (int64_t a = 0; a < axis_size; ++a) {
                float exp_val = std::exp(ptr[a * stride] - max_val);
                sum_exp += exp_val;
                weighted_sum += weight_data[a] * exp_val;
            }

            // 归一化
            output_data[o * inner + i] = weighted_sum / sum_exp;
        }
    }

    return Tensor(out_storage, {outer, inner},
                  compute_strides({outer, inner}), 0, true);
}

// ========== 融合 Sigmoid + Max + Argmax ==========

LuaIntf::LuaRef Tensor::sigmoid_max_with_argmax(lua_State* L, int axis) const {
#ifdef ENABLE_RV_THEAD
    // 使用玄铁优化版本
    return sigmoid_max_with_argmax_thead(L, axis);
#else
    // CPU fallback
    return sigmoid_max_with_argmax_cpu(L, axis);
#endif
}

LuaIntf::LuaRef Tensor::sigmoid_max_with_argmax_cpu(lua_State* L, int axis) const {
    // CPU fallback: 调用现有的 sigmoid() + max_with_argmax()
    Tensor scores = this->sigmoid();
    return scores.max_with_argmax(L, axis);
}

}  // namespace tensor
