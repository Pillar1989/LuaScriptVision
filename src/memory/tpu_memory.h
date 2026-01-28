#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include "device_buffer.h"
#include "sync_handle.h"

#if defined(USE_CVI_TPU)
#include <cviruntime.h>
#include <cviruntime_context.h>
#endif

namespace memory {

#if defined(USE_CVI_TPU)

class CviTpuMemory : public DeviceBuffer {
public:
    static std::shared_ptr<CviTpuMemory> allocate(
        CVI_RT_HANDLE rt_handle,
        size_t size_bytes
    );

    static std::shared_ptr<CviTpuMemory> from_physical(
        uint64_t paddr,
        size_t size_bytes,
        bool take_ownership = false
    );

    ~CviTpuMemory() override;

    CviTpuMemory(const CviTpuMemory&) = delete;
    CviTpuMemory& operator=(const CviTpuMemory&) = delete;

    CviTpuMemory(CviTpuMemory&& other) noexcept;
    CviTpuMemory& operator=(CviTpuMemory&& other) noexcept;

    // DeviceBuffer interface
    void* data() override { return vaddr_; }
    const void* data() const override { return vaddr_; }
    size_t size_bytes() const override { return size_bytes_; }
    DeviceType device() const override { return DeviceType::TPU; }
    size_t alignment() const override { return 64; }
    bool owns_memory() const override { return owns_memory_; }

    void copy_to(DeviceBuffer* dst) const override;
    void copy_from(const DeviceBuffer* src) override;

    // Async interface
    void copy_to_async(DeviceBuffer* dst, SyncHandle* handle = nullptr) const override;
    void copy_from_async(const DeviceBuffer* src, SyncHandle* handle = nullptr) override;
    void sync(SyncHandle* handle = nullptr) const override;
    bool supports_async() const override { return false; }

    // Hardware acceleration
    bool has_physical_addr() const override { return paddr_ != 0; }
    uint64_t physical_addr() const override { return paddr_; }
    void flush_cache() override;
    void invalidate_cache() override;

    // CVI Runtime specific
    CVI_RT_HANDLE runtime_handle() const { return rt_handle_; }
    CVI_RT_MEM runtime_mem() const { return rt_mem_; }

private:
    CviTpuMemory() = default;

    CVI_RT_HANDLE rt_handle_ = nullptr;
    CVI_RT_MEM rt_mem_ = nullptr;
    uint8_t* vaddr_ = nullptr;
    uint64_t paddr_ = 0;
    size_t size_bytes_ = 0;
    bool owns_memory_ = false;
};

#endif // USE_CVI_TPU

} // namespace memory

// Backward compatibility
namespace tensor {
#if defined(USE_CVI_TPU)
    using CviTpuMemory = memory::CviTpuMemory;
    using TpuMemory = memory::CviTpuMemory;
#endif
} // namespace tensor
