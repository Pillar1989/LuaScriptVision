#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include "device_buffer.h"

namespace memory {

class CpuMemory : public DeviceBuffer {
public:
    static std::shared_ptr<CpuMemory> allocate(size_t size_bytes, size_t alignment = 64, bool zero_init = false);
    static std::shared_ptr<CpuMemory> from_external(void* ptr, size_t size_bytes, bool take_ownership = false);

    ~CpuMemory() override;

    CpuMemory(const CpuMemory&) = delete;
    CpuMemory& operator=(const CpuMemory&) = delete;

    CpuMemory(CpuMemory&& other) noexcept;
    CpuMemory& operator=(CpuMemory&& other) noexcept;

    // DeviceBuffer interface
    void* data() override { return data_; }
    const void* data() const override { return data_; }
    size_t size_bytes() const override { return size_bytes_; }
    DeviceType device() const override { return DeviceType::CPU; }
    size_t alignment() const override { return alignment_; }
    bool owns_memory() const override { return owns_memory_; }

    void copy_to(DeviceBuffer* dst) const override;
    void copy_from(const DeviceBuffer* src) override;

    // Async interface (CPU: sync fallback)
    void copy_to_async(DeviceBuffer* dst, SyncHandle* handle = nullptr) const override;
    void copy_from_async(const DeviceBuffer* src, SyncHandle* handle = nullptr) override;
    void sync(SyncHandle* handle = nullptr) const override;
    bool supports_async() const override;

private:
    CpuMemory() = default;

    void* data_ = nullptr;
    size_t size_bytes_ = 0;
    size_t alignment_ = 64;
    bool owns_memory_ = false;
};

} // namespace memory

// Backward compatibility
namespace tensor {
    using CpuMemory = memory::CpuMemory;
} // namespace tensor
