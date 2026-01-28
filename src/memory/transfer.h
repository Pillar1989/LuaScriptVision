#pragma once

#include "device_buffer.h"
#include <cstdint>

namespace memory {

class MemoryTransfer {
public:
    // High-level copy (auto handles cache coherence)
    static void copy(DeviceBuffer* dst, const DeviceBuffer* src);

    // Zero-copy support
    static uint64_t get_paddr(const DeviceBuffer* buf);
    static bool supports_zero_copy(const DeviceBuffer* src, const DeviceBuffer* dst);
};

} // namespace memory
