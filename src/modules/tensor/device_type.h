#pragma once

namespace tensor {

/**
 * 设备类型枚举
 * 支持的计算设备类型
 */
enum class DeviceType {
    CPU,    // CPU 内存
    VB,     // CVI VB pool memory
    NPU,    // Rockchip NPU (RKNN)
    TPU,    // Sophgo TPU (BMRuntime)
};

/**
 * 设备类型转字符串
 */
inline const char* device_type_to_string(DeviceType device) {
    switch (device) {
        case DeviceType::CPU: return "CPU";
        case DeviceType::VB: return "VB";
        case DeviceType::NPU: return "NPU";
        case DeviceType::TPU: return "TPU";
        default: return "Unknown";
    }
}

} // namespace tensor
