#include "device_buffer.h"
#include "cpu_memory.h"

// 条件编译：仅在启用 Sophgo TPU 支持时包含 CVI TPU Memory
#if defined(USE_CVI_TPU)
#include "cvi_tpu_memory.h"
#endif

#include <stdexcept>

namespace tensor {

std::shared_ptr<DeviceBuffer> DeviceBuffer::allocate(
    size_t size_bytes,
    DeviceType device,
    size_t alignment
) {
    switch (device) {
        case DeviceType::CPU:
            return CpuMemory::allocate(size_bytes, alignment);

        case DeviceType::NPU:
            // TODO: 实现 Rockchip NPU Memory
            throw std::runtime_error("DeviceBuffer::allocate: NPU buffer not implemented");

        case DeviceType::TPU:
#if defined(USE_CVI_TPU)
            // Sophgo TPU 内存分配需要 CVI_RT_HANDLE，请直接使用 CviTpuMemory::allocate()
            throw std::runtime_error(
                "DeviceBuffer::allocate: TPU buffer requires CVI_RT_HANDLE. "
                "Please use CviTpuMemory::allocate(rt_handle, size_bytes) directly.");
#else
            throw std::runtime_error("DeviceBuffer::allocate: TPU support not enabled (USE_CVI_TPU not defined)");
#endif

        default:
            throw std::invalid_argument("DeviceBuffer::allocate: unknown device type");
    }
}

std::shared_ptr<DeviceBuffer> DeviceBuffer::from_external(
    void* ptr,
    size_t size_bytes,
    DeviceType device,
    bool take_ownership
) {
    switch (device) {
        case DeviceType::CPU:
            return CpuMemory::from_external(ptr, size_bytes, take_ownership);

        case DeviceType::NPU:
            // TODO: 实现 Rockchip NPU Memory
            throw std::runtime_error("DeviceBuffer::from_external: NPU buffer not implemented");

        case DeviceType::TPU:
#if defined(USE_CVI_TPU)
            // Sophgo TPU 外部内存需要物理地址，请使用 CviTpuMemory::from_physical()
            throw std::runtime_error(
                "DeviceBuffer::from_external: TPU buffer requires physical address (uint64_t). "
                "Please use CviTpuMemory::from_physical(paddr, size_bytes, take_ownership) instead.");
#else
            throw std::runtime_error("DeviceBuffer::from_external: TPU support not enabled (USE_CVI_TPU not defined)");
#endif

        default:
            throw std::invalid_argument("DeviceBuffer::from_external: unknown device type");
    }
}

} // namespace tensor
