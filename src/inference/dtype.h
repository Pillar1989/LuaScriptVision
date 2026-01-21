#pragma once

#include <cstddef>

namespace inference {

enum class DType {
    Float32,
    Float16,
    BFloat16,
    Int8,
    UInt8,
    Int32,
    Unknown
};

inline size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::Float32:
            return 4;
        case DType::Float16:
        case DType::BFloat16:
            return 2;
        case DType::Int8:
        case DType::UInt8:
            return 1;
        case DType::Int32:
            return 4;
        default:
            return 0;
    }
}

inline const char* dtype_name(DType dtype) {
    switch (dtype) {
        case DType::Float32:
            return "float32";
        case DType::Float16:
            return "float16";
        case DType::BFloat16:
            return "bfloat16";
        case DType::Int8:
            return "int8";
        case DType::UInt8:
            return "uint8";
        case DType::Int32:
            return "int32";
        default:
            return "unknown";
    }
}

inline bool is_quantized(DType dtype) {
    return dtype == DType::Int8 || dtype == DType::UInt8;
}

} // namespace inference
