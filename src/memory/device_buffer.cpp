#include "device_buffer.h"
#include "cpu_memory.h"

#if defined(USE_CVI_TPU)
#include "tpu_memory.h"
#endif
#if defined(USE_CVI_MPI)
#include "vb_memory.h"
#endif

#include <stdexcept>

namespace memory {

std::shared_ptr<DeviceBuffer> DeviceBuffer::allocate(
    size_t size_bytes,
    DeviceType device,
    size_t alignment
) {
    switch (device) {
        case DeviceType::CPU:
            return CpuMemory::allocate(size_bytes, alignment);

        case DeviceType::VB:
#if defined(USE_CVI_MPI)
            throw std::runtime_error(
                "DeviceBuffer::allocate: VB buffer requires VbMemory::allocate(size_bytes, cached) "
                "or VbMemory::allocate_frame(width, height, format).");
#else
            throw std::runtime_error("DeviceBuffer::allocate: VB support not enabled (USE_CVI_MPI not defined)");
#endif

        case DeviceType::TPU:
#if defined(USE_CVI_TPU)
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

        case DeviceType::VB:
#if defined(USE_CVI_MPI)
            throw std::runtime_error(
                "DeviceBuffer::from_external: VB buffer requires VB_BLK handle. "
                "Use VbMemory::from_block(block, size_bytes, cached, take_ownership).");
#else
            throw std::runtime_error("DeviceBuffer::from_external: VB support not enabled (USE_CVI_MPI not defined)");
#endif

        case DeviceType::TPU:
#if defined(USE_CVI_TPU)
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

} // namespace memory
