#pragma once

namespace memory {

enum class DeviceType {
    CPU,      // CPU heap memory (malloc/new)
    VB,       // CVI Video Buffer (VPSS/VENC/TPU)
    TPU,      // CVI TPU runtime allocation
    UNKNOWN
};

inline const char* device_type_to_string(DeviceType device) {
    switch (device) {
        case DeviceType::CPU: return "CPU";
        case DeviceType::VB: return "VB";
        case DeviceType::TPU: return "TPU";
        case DeviceType::UNKNOWN: return "UNKNOWN";
        default: return "Unknown";
    }
}

} // namespace memory

// Backward compatibility: tensor::DeviceType maps to memory::DeviceType
namespace tensor {
    using DeviceType = memory::DeviceType;
    using memory::device_type_to_string;
} // namespace tensor
