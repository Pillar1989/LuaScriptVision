#include "tensor.h"
#include "cpu_memory.h"
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <cstring>

#include "LuaIntf.h"

namespace tensor {

// ========== 构造函数 ==========

Tensor::Tensor(const std::vector<float>& data, const std::vector<int64_t>& shape)
    : shape_(shape)
    , strides_(compute_strides(shape))
    , offset_(0)
    , contiguous_(true)
    , device_cache_(DeviceType::CPU) {
    // 分配 CPU Storage 并复制数据
    buffer_ = CpuMemory::allocate(data.size() * sizeof(float));
    std::memcpy(buffer_->data(), data.data(), data.size() * sizeof(float));
}

Tensor::Tensor(std::vector<float>&& data, const std::vector<int64_t>& shape)
    : shape_(shape)
    , strides_(compute_strides(shape))
    , offset_(0)
    , contiguous_(true)
    , device_cache_(DeviceType::CPU) {
    // 分配 CPU Storage 并移动数据
    buffer_ = CpuMemory::allocate(data.size() * sizeof(float));
    std::memcpy(buffer_->data(), data.data(), data.size() * sizeof(float));
}

Tensor::Tensor(const float* data, const std::vector<int64_t>& shape,
               std::shared_ptr<DeviceBuffer> owner)
    : shape_(shape)
    , strides_(compute_strides(shape))
    , offset_(0)
    , contiguous_(true)
    , device_cache_(DeviceType::CPU) {
    if (owner) {
        buffer_ = owner;
        device_cache_ = buffer_->device();  // 更新为实际设备类型
    } else {
        int64_t total_size = compute_size();
        buffer_ = CpuMemory::allocate(total_size * sizeof(float));
        std::memcpy(buffer_->data(), data, total_size * sizeof(float));
    }
}

Tensor::Tensor(std::shared_ptr<DeviceBuffer> storage,
               const std::vector<int64_t>& shape,
               const std::vector<int64_t>& strides,
               int64_t offset,
               bool contiguous)
    : buffer_(storage)
    , shape_(shape)
    , strides_(strides)
    , offset_(offset)
    , contiguous_(contiguous)
    , device_cache_(storage->device()) {}

// Static factory method for Lua binding
Tensor Tensor::create(const std::vector<float>& data, const std::vector<int64_t>& shape) {
    return Tensor(data, shape);
}

// Static factory method - create arithmetic sequence tensor
Tensor Tensor::arange(float start, float end, float step) {
    if (std::abs(step) < 1e-7f) {
        throw std::runtime_error("arange: step cannot be zero");
    }
    if ((end > start && step < 0) || (end < start && step > 0)) {
        throw std::runtime_error("arange: invalid step direction");
    }

    // Calculate number of elements
    int64_t count = static_cast<int64_t>(std::ceil((end - start) / step));
    if (count <= 0) count = 0;

    std::vector<float> data;
    data.reserve(static_cast<size_t>(count));

    float val = start;
    for (int64_t i = 0; i < count; ++i) {
        data.push_back(val);
        val += step;
    }

    std::vector<int64_t> shape = {count};
    return Tensor(std::move(data), shape);
}

// Lua factory method - manually parse LuaRef tables
Tensor Tensor::from_lua(lua_State* L, const LuaIntf::LuaRef& data_table, const LuaIntf::LuaRef& shape_table) {
    // Parse data table
    std::vector<float> data;
    if (data_table.isTable()) {
        int len = data_table.len();
        data.reserve(len);
        for (int i = 1; i <= len; ++i) {
            data.push_back(static_cast<float>(data_table[i].value<double>()));
        }
    } else {
        throw std::runtime_error("data must be a table");
    }

    // Parse shape table
    std::vector<int64_t> shape;
    if (shape_table.isTable()) {
        int len = shape_table.len();
        shape.reserve(len);
        for (int i = 1; i <= len; ++i) {
            shape.push_back(static_cast<int64_t>(shape_table[i].value<int>()));
        }
    } else {
        throw std::runtime_error("shape must be a table");
    }

    return Tensor(data, shape);
}

// Lua wrapper - get single index
float Tensor::get_lua(int idx) const {
    check_cpu();
    if (idx < 0 || idx >= size()) {
        throw std::runtime_error("Index out of bounds");
    }
    const float* data = raw_data();
    if (contiguous_) {
        return data[offset_ + idx];
    } else {
        // Non-contiguous: need to compute actual index
        std::vector<int64_t> indices(shape_.size());
        int64_t remaining = idx;
        for (int i = shape_.size() - 1; i >= 0; --i) {
            indices[i] = remaining % shape_[i];
            remaining /= shape_[i];
        }
        int64_t actual_idx = offset_;
        for (size_t i = 0; i < shape_.size(); ++i) {
            actual_idx += indices[i] * strides_[i];
        }
        return data[actual_idx];
    }
}

// Lua wrapper - set single index
void Tensor::set_lua(int idx, float value) {
    check_cpu();
    if (idx < 0 || idx >= size()) {
        throw std::runtime_error("Index out of bounds");
    }
    float* data = raw_data();
    if (contiguous_) {
        data[offset_ + idx] = value;
    } else {
        std::vector<int64_t> indices(shape_.size());
        int64_t remaining = idx;
        for (int i = shape_.size() - 1; i >= 0; --i) {
            indices[i] = remaining % shape_[i];
            remaining /= shape_[i];
        }
        int64_t actual_idx = offset_;
        for (size_t i = 0; i < shape_.size(); ++i) {
            actual_idx += indices[i] * strides_[i];
        }
        data[actual_idx] = value;
    }
}


// 直接返回 Lua table 的版本 (避免 transpose + to_table 开销)
// 返回行格式: {{row1_col1, row1_col2, ...}, {row2_col1, row2_col2, ...}, ...}
// 对于 boxes[4, 8400], extract_columns({idx1, idx2}) 返回 {{cx1,cy1,w1,h1}, {cx2,cy2,w2,h2}}
LuaIntf::LuaRef Tensor::extract_columns_lua(lua_State* L, const std::vector<int64_t>& cols) const {
    check_cpu();
    if (shape_.size() != 2) {
        throw std::runtime_error("extract_columns requires 2D tensor");
    }

    int64_t rows = shape_[0];
    int64_t src_cols = shape_[1];

    // 使用 stride-based 访问，避免 contiguous() 调用
    const float* base = static_cast<const float*>(buffer_->data()) + offset_;
    int64_t s0 = strides_[0];
    int64_t s1 = strides_[1];

    // 返回行格式（已转置）：外层是每个选中的列（box），内层是该列的所有行（cx,cy,w,h）
    LuaIntf::LuaRef result = LuaIntf::LuaRef::createTable(L);

    for (size_t i = 0; i < cols.size(); ++i) {
        int64_t col = cols[i];
        if (col < 0) col += src_cols;

        // 每个 box 的数据：{cx, cy, w, h}
        LuaIntf::LuaRef row_data = LuaIntf::LuaRef::createTable(L);
        for (int64_t r = 0; r < rows; ++r) {
            row_data[r + 1] = base[r * s0 + col * s1];
        }
        result[i + 1] = row_data;
    }

    return result;
}

// ========== 辅助方法 ==========

int64_t Tensor::size() const {
    return compute_size();
}

int64_t Tensor::size(int dim) const {
    if (dim < 0) dim += shape_.size();
    if (dim < 0 || dim >= static_cast<int>(shape_.size())) {
        throw std::runtime_error("Dimension out of range");
    }
    return shape_[dim];
}

int64_t Tensor::compute_size() const {
    if (shape_.empty()) return 0;
    return std::accumulate(shape_.begin(), shape_.end(), 1LL, std::multiplies<int64_t>());
}

std::vector<int64_t> Tensor::compute_strides(const std::vector<int64_t>& shape) const {
    std::vector<int64_t> strides(shape.size());
    if (shape.empty()) return strides;

    strides[shape.size() - 1] = 1;
    for (int i = shape.size() - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

void Tensor::normalize_axis(int& axis) const {
    if (axis < 0) axis += shape_.size();
    if (axis < 0 || axis >= static_cast<int>(shape_.size())) {
        throw std::runtime_error("Axis out of range");
    }
}

int64_t Tensor::compute_offset(const std::vector<int64_t>& indices) const {
    if (indices.size() != shape_.size()) {
        throw std::runtime_error("Indices size mismatch");
    }

    int64_t offset = offset_;
    for (size_t i = 0; i < indices.size(); ++i) {
        int64_t idx = indices[i];
        if (idx < 0) idx += shape_[i];
        if (idx < 0 || idx >= shape_[i]) {
            throw std::runtime_error("Index out of range");
        }
        offset += idx * strides_[i];
    }
    return offset;
}

// check_cpu() 已内联到 tensor.h 中

// ========== 数据访问 ==========

const float* Tensor::raw_data() const {
    check_cpu();
    return static_cast<const float*>(buffer_->data());
}

float* Tensor::raw_data() {
    check_cpu();
    return static_cast<float*>(buffer_->data());
}

const float* Tensor::data() const {
    check_cpu();
    return static_cast<const float*>(buffer_->data()) + offset_;
}

float* Tensor::data() {
    check_cpu();
    return static_cast<float*>(buffer_->data()) + offset_;
}

// ========== 统一元素访问 at() ==========

float Tensor::at(int64_t i) const {
    check_cpu();
    if (shape_.size() != 1) {
        throw std::runtime_error("at(i) requires 1D tensor");
    }
    if (i < 0) i += shape_[0];
    if (i < 0 || i >= shape_[0]) {
        throw std::runtime_error("Index out of range");
    }
    const float* ptr = static_cast<const float*>(buffer_->data());
    return ptr[offset_ + i * strides_[0]];
}

float Tensor::at(int64_t i, int64_t j) const {
    check_cpu();
    if (shape_.size() != 2) {
        throw std::runtime_error("at(i,j) requires 2D tensor");
    }
    if (i < 0) i += shape_[0];
    if (j < 0) j += shape_[1];
    if (i < 0 || i >= shape_[0] || j < 0 || j >= shape_[1]) {
        throw std::runtime_error("Index out of range");
    }
    const float* ptr = static_cast<const float*>(buffer_->data());
    return ptr[offset_ + i * strides_[0] + j * strides_[1]];
}

float Tensor::at(const std::vector<int64_t>& indices) const {
    check_cpu();
    int64_t offset = compute_offset(indices);
    const float* ptr = static_cast<const float*>(buffer_->data());
    return ptr[offset];
}

float& Tensor::at(int64_t i) {
    check_cpu();
    if (shape_.size() != 1) {
        throw std::runtime_error("at(i) requires 1D tensor");
    }
    if (i < 0) i += shape_[0];
    if (i < 0 || i >= shape_[0]) {
        throw std::runtime_error("Index out of range");
    }
    float* ptr = static_cast<float*>(buffer_->data());
    return ptr[offset_ + i * strides_[0]];
}

float& Tensor::at(int64_t i, int64_t j) {
    check_cpu();
    if (shape_.size() != 2) {
        throw std::runtime_error("at(i,j) requires 2D tensor");
    }
    if (i < 0) i += shape_[0];
    if (j < 0) j += shape_[1];
    if (i < 0 || i >= shape_[0] || j < 0 || j >= shape_[1]) {
        throw std::runtime_error("Index out of range");
    }
    float* ptr = static_cast<float*>(buffer_->data());
    return ptr[offset_ + i * strides_[0] + j * strides_[1]];
}

} // namespace tensor
