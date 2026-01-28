#pragma once

#include <cstddef>
#include <memory>
#include "device_type.h"

namespace memory {

// Forward declaration
class SyncHandle;

class DeviceBuffer {
public:
    virtual ~DeviceBuffer() = default;

    // Data access
    virtual void* data() = 0;
    virtual const void* data() const = 0;
    virtual size_t size_bytes() const = 0;

    // Device info
    virtual DeviceType device() const = 0;
    virtual size_t alignment() const = 0;
    virtual bool owns_memory() const = 0;

    // Hardware acceleration support
    virtual bool has_physical_addr() const { return false; }
    virtual uint64_t physical_addr() const { return 0; }
    virtual void flush_cache() {}
    virtual void invalidate_cache() {}

    // Data transfer
    virtual void copy_to(DeviceBuffer* dst) const = 0;
    virtual void copy_from(const DeviceBuffer* src) = 0;

    // Async data transfer
    virtual void copy_to_async(DeviceBuffer* dst, SyncHandle* handle = nullptr) const = 0;
    virtual void copy_from_async(const DeviceBuffer* src, SyncHandle* handle = nullptr) = 0;
    virtual void sync(SyncHandle* handle = nullptr) const = 0;
    virtual bool supports_async() const = 0;

    // Factory methods
    static std::shared_ptr<DeviceBuffer> allocate(
        size_t size_bytes,
        DeviceType device = DeviceType::CPU,
        size_t alignment = 64
    );

    static std::shared_ptr<DeviceBuffer> from_external(
        void* ptr,
        size_t size_bytes,
        DeviceType device = DeviceType::CPU,
        bool take_ownership = false
    );
};

} // namespace memory

// Backward compatibility
namespace tensor {
    using DeviceBuffer = memory::DeviceBuffer;
} // namespace tensor
