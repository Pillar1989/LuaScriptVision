#pragma once

#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <vector>

#include "dtype.h"
#include "layout.h"
#include "quant_params.h"
#include "tensor/tensor.h"

namespace inference {

struct TensorDescriptor {
    std::string name;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    DType dtype = DType::Unknown;
    Layout layout = Layout::Unknown;
    QuantParams quant_params;
    size_t alignment = 64;
    std::set<tensor::DeviceType> supported_devices;
    size_t plane_count = 1;
    std::vector<size_t> plane_offsets;

    int64_t element_count() const {
        if (shape.empty()) {
            return 0;
        }
        int64_t count = 1;
        for (int64_t dim : shape) {
            if (dim <= 0) {
                return 0;
            }
            count *= dim;
        }
        return count;
    }

    size_t size_bytes() const {
        int64_t count = element_count();
        if (count <= 0) {
            return 0;
        }
        return static_cast<size_t>(count) * dtype_size(dtype);
    }

    bool is_quantized() const {
        return inference::is_quantized(dtype);
    }

    bool is_compatible(const tensor::Tensor& tensor) const {
        if (!shape.empty()) {
            auto actual = tensor.shape();
            if (actual.size() != shape.size()) {
                return false;
            }
            for (size_t i = 0; i < shape.size(); ++i) {
                int64_t expected = shape[i];
                if (expected > 0 && expected != actual[i]) {
                    return false;
                }
            }
        }
        if (dtype != DType::Unknown && dtype != DType::Float32) {
            return false;
        }
        if (!supported_devices.empty() &&
            supported_devices.count(tensor.device()) == 0) {
            return false;
        }
        if (alignment > 0) {
            auto buf = tensor.buffer();
            if (buf && buf->alignment() < alignment) {
                return false;
            }
        }
        return true;
    }

    bool is_compatible(const tensor::DeviceBuffer& buffer) const {
        if (!supported_devices.empty() &&
            supported_devices.count(buffer.device()) == 0) {
            return false;
        }
        size_t required = size_bytes();
        if (required > 0 && buffer.size_bytes() < required) {
            return false;
        }
        if (alignment > 0 && buffer.alignment() < alignment) {
            return false;
        }
        return true;
    }
};

} // namespace inference
