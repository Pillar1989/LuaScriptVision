#pragma once

#ifdef USE_CVI_MPI

#include <cstddef>
#include <cstdint>
#include <memory>

#include "device_buffer.h"
#include "sync_handle.h"
#include <cvi_vb.h>
#include <cvi_sys.h>

namespace memory {

class VbMemory : public DeviceBuffer {
public:
    static std::shared_ptr<VbMemory> from_block(VB_BLK block,
                                                size_t size_bytes,
                                                bool cached,
                                                bool take_ownership);
    static std::shared_ptr<VbMemory> allocate(size_t size_bytes, bool cached = true);
    static std::shared_ptr<VbMemory> allocate_frame(uint32_t width,
                                                    uint32_t height,
                                                    PIXEL_FORMAT_E format,
                                                    bool cached = true);

    ~VbMemory() override;

    VbMemory(const VbMemory&) = delete;
    VbMemory& operator=(const VbMemory&) = delete;

    void* data() override;
    const void* data() const override;
    size_t size_bytes() const override { return size_bytes_; }
    DeviceType device() const override { return DeviceType::VB; }
    size_t alignment() const override { return alignment_; }
    bool owns_memory() const override { return owns_block_; }

    void copy_to(DeviceBuffer* dst) const override;
    void copy_from(const DeviceBuffer* src) override;

    void copy_to_async(DeviceBuffer* dst, SyncHandle* handle = nullptr) const override;
    void copy_from_async(const DeviceBuffer* src, SyncHandle* handle = nullptr) override;
    void sync(SyncHandle* handle = nullptr) const override;
    bool supports_async() const override { return false; }

    // Hardware acceleration
    bool has_physical_addr() const override { return phys_addr_ != 0; }
    uint64_t physical_addr() const override { return phys_addr_; }
    uint64_t physical_address() const { return phys_addr_; }  // Backward compatible alias
    void flush_cache() override;
    void invalidate_cache() override;

    VB_BLK block() const { return block_; }
    bool is_cached() const { return cached_; }

private:
    VbMemory(VB_BLK block, uint64_t phys_addr, size_t size_bytes, bool cached, bool owns_block);

    VB_BLK block_ = VB_INVALID_HANDLE;
    uint64_t phys_addr_ = 0;
    size_t size_bytes_ = 0;
    size_t alignment_ = 64;
    bool cached_ = false;
    bool owns_block_ = false;

    void* data_ = nullptr;
    size_t mapped_size_ = 0;
};

} // namespace memory

// Backward compatibility
namespace tensor {
    using VbMemory = memory::VbMemory;
} // namespace tensor

#endif  // USE_CVI_MPI
