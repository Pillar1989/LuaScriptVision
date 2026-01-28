#include "transfer.h"
#include <stdexcept>
#include <cstring>

namespace memory {

void MemoryTransfer::copy(DeviceBuffer* dst, const DeviceBuffer* src) {
    if (!dst || !src) {
        throw std::invalid_argument("MemoryTransfer::copy: null buffer");
    }
    if (dst->size_bytes() < src->size_bytes()) {
        throw std::runtime_error("MemoryTransfer::copy: destination too small");
    }

    // Same device type: use direct copy
    if (src->device() == dst->device()) {
        src->copy_to(dst);
        return;
    }

    // Cross-device: delegate to source's copy_to (handles cache coherence)
    src->copy_to(dst);
}

uint64_t MemoryTransfer::get_paddr(const DeviceBuffer* buf) {
    if (!buf) {
        return 0;
    }
    return buf->physical_addr();
}

bool MemoryTransfer::supports_zero_copy(const DeviceBuffer* src, const DeviceBuffer* dst) {
    if (!src || !dst) {
        return false;
    }

    // Zero-copy requires physical address from source
    if (!src->has_physical_addr()) {
        return false;
    }

    // Physical address must be 64-byte aligned
    uint64_t paddr = src->physical_addr();
    if ((paddr & 0x3F) != 0) {
        return false;
    }

    // VB → TPU zero-copy is the primary path
    if (src->device() == DeviceType::VB && dst->device() == DeviceType::TPU) {
        return true;
    }

    return false;
}

} // namespace memory
