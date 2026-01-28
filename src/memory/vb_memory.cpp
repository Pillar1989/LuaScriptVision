#include "vb_memory.h"

#ifdef USE_CVI_MPI

#include <cstring>
#include <stdexcept>

#include "cpu_memory.h"
#include "sync_handle.h"
#include "vb_pool_manager.h"
#include <cvi_buffer.h>
#if defined(USE_CVI_TPU)
#include "tpu_memory.h"
#include <cviruntime_context.h>
#endif

namespace memory {
namespace {
void invalidate_if_cached(uint64_t phys_addr, void* data, size_t size, bool cached) {
    if (cached && phys_addr != 0 && data != nullptr && size > 0) {
        CVI_SYS_IonInvalidateCache(phys_addr, data, static_cast<CVI_U32>(size));
    }
}

void flush_if_cached(uint64_t phys_addr, void* data, size_t size, bool cached) {
    if (cached && phys_addr != 0 && data != nullptr && size > 0) {
        CVI_SYS_IonFlushCache(phys_addr, data, static_cast<CVI_U32>(size));
    }
}

size_t get_pool_block_size(VB_BLK block) {
    VB_CONFIG_S vb_config{};
    CVI_S32 rc = CVI_VB_GetConfig(&vb_config);
    if (rc != CVI_SUCCESS) {
        return 0;
    }
    VB_POOL pool = CVI_VB_Handle2PoolId(block);
    if (pool >= vb_config.u32MaxPoolCnt) {
        return 0;
    }
    return vb_config.astCommPool[pool].u32BlkSize;
}
} // namespace

VbMemory::VbMemory(VB_BLK block, uint64_t phys_addr, size_t size_bytes, bool cached, bool owns_block)
    : block_(block),
      phys_addr_(phys_addr),
      size_bytes_(size_bytes),
      cached_(cached),
      owns_block_(owns_block) {}

std::shared_ptr<VbMemory> VbMemory::from_block(VB_BLK block,
                                               size_t size_bytes,
                                               bool cached,
                                               bool take_ownership) {
    if (block == VB_INVALID_HANDLE) {
        throw std::invalid_argument("VbMemory::from_block - invalid VB block");
    }
    if (size_bytes == 0) {
        throw std::invalid_argument("VbMemory::from_block - size_bytes must be > 0");
    }
    uint64_t phys = CVI_VB_Handle2PhysAddr(block);
    if (phys == 0) {
        throw std::runtime_error("VbMemory::from_block - physical address is zero");
    }
    size_t pool_size = get_pool_block_size(block);
    if (pool_size > 0 && size_bytes > pool_size) {
        throw std::runtime_error("VbMemory::from_block - size_bytes exceeds pool block size");
    }
    return std::shared_ptr<VbMemory>(new VbMemory(block, phys, size_bytes, cached, take_ownership));
}

std::shared_ptr<VbMemory> VbMemory::allocate(size_t size_bytes, bool cached) {
    if (size_bytes == 0) {
        throw std::invalid_argument("VbMemory::allocate - size_bytes must be > 0");
    }
    VB_BLK block = VbPoolManager::instance().get_block(size_bytes);
    if (block == VB_INVALID_HANDLE) {
        throw std::runtime_error("VbMemory::allocate - no available VB block");
    }
    uint64_t phys = CVI_VB_Handle2PhysAddr(block);
    if (phys == 0) {
        CVI_VB_ReleaseBlock(block);
        throw std::runtime_error("VbMemory::allocate - physical address is zero");
    }
    size_t pool_size = get_pool_block_size(block);
    if (pool_size > 0 && size_bytes > pool_size) {
        CVI_VB_ReleaseBlock(block);
        throw std::runtime_error("VbMemory::allocate - size_bytes exceeds pool block size");
    }
    return std::shared_ptr<VbMemory>(new VbMemory(block, phys, size_bytes, cached, true));
}

std::shared_ptr<VbMemory> VbMemory::allocate_frame(uint32_t width,
                                                   uint32_t height,
                                                   PIXEL_FORMAT_E format,
                                                   bool cached) {
    CVI_U32 size = COMMON_GetPicBufferSize(width, height, format,
                                           DATA_BITWIDTH_8, COMPRESS_MODE_NONE, 0);
    if (size == 0) {
        throw std::runtime_error("VbMemory::allocate_frame - invalid frame size");
    }
    return allocate(size, cached);
}

VbMemory::~VbMemory() {
    if (data_ && mapped_size_ > 0) {
        CVI_SYS_Munmap(data_, static_cast<CVI_U32>(mapped_size_));
        data_ = nullptr;
        mapped_size_ = 0;
    }
    if (owns_block_ && block_ != VB_INVALID_HANDLE) {
        CVI_VB_ReleaseBlock(block_);
        block_ = VB_INVALID_HANDLE;
    }
}

void* VbMemory::data() {
    if (!data_ && phys_addr_ != 0) {
        if (cached_) {
            data_ = CVI_SYS_MmapCache(phys_addr_, static_cast<CVI_U32>(size_bytes_));
        } else {
            data_ = CVI_SYS_Mmap(phys_addr_, static_cast<CVI_U32>(size_bytes_));
        }
        mapped_size_ = size_bytes_;
    }
    return data_;
}

const void* VbMemory::data() const {
    return const_cast<VbMemory*>(this)->data();
}

void VbMemory::flush_cache() {
    void* ptr = data();
    if (!ptr) {
        throw std::runtime_error("VbMemory::flush_cache - map failed");
    }
    flush_if_cached(phys_addr_, ptr, size_bytes_, cached_);
}

void VbMemory::invalidate_cache() {
    void* ptr = data();
    if (!ptr) {
        throw std::runtime_error("VbMemory::invalidate_cache - map failed");
    }
    invalidate_if_cached(phys_addr_, ptr, size_bytes_, cached_);
}

void VbMemory::copy_to(DeviceBuffer* dst) const {
    if (!dst) {
        throw std::invalid_argument("VbMemory::copy_to - dst is null");
    }
    if (dst->size_bytes() < size_bytes_) {
        throw std::runtime_error("VbMemory::copy_to - dst buffer too small");
    }

    void* src_ptr = const_cast<VbMemory*>(this)->data();
    if (!src_ptr) {
        throw std::runtime_error("VbMemory::copy_to - source pointer is null");
    }

    if (cached_) {
        invalidate_if_cached(phys_addr_, src_ptr, size_bytes_, cached_);
    }

    if (dst->device() == DeviceType::CPU) {
        std::memcpy(dst->data(), src_ptr, size_bytes_);
        return;
    }

#if defined(USE_CVI_TPU)
    if (dst->device() == DeviceType::TPU) {
        auto tpu = dynamic_cast<CviTpuMemory*>(dst);
        if (!tpu) {
            throw std::runtime_error("VbMemory::copy_to - unsupported TPU buffer");
        }
        CVI_RT_HANDLE rt = tpu->runtime_handle();
        CVI_RT_MEM mem = tpu->runtime_mem();
        CVI_RC rc = CVI_RT_MemCopyS2D(rt, mem, static_cast<uint8_t*>(src_ptr));
        if (rc != CVI_RC_SUCCESS) {
            throw std::runtime_error("VbMemory::copy_to - CVI_RT_MemCopyS2D failed");
        }
        return;
    }
#endif

    throw std::runtime_error("VbMemory::copy_to - unsupported device");
}

void VbMemory::copy_from(const DeviceBuffer* src) {
    if (!src) {
        throw std::invalid_argument("VbMemory::copy_from - src is null");
    }
    if (src->size_bytes() < size_bytes_) {
        throw std::runtime_error("VbMemory::copy_from - src buffer too small");
    }

    void* dst_ptr = data();
    if (!dst_ptr) {
        throw std::runtime_error("VbMemory::copy_from - destination pointer is null");
    }

    if (src->device() == DeviceType::CPU) {
        std::memcpy(dst_ptr, src->data(), size_bytes_);
        flush_if_cached(phys_addr_, dst_ptr, size_bytes_, cached_);
        return;
    }

#if defined(USE_CVI_TPU)
    if (src->device() == DeviceType::TPU) {
        auto tpu = dynamic_cast<const CviTpuMemory*>(src);
        if (!tpu) {
            throw std::runtime_error("VbMemory::copy_from - unsupported TPU buffer");
        }
        CVI_RT_HANDLE rt = tpu->runtime_handle();
        CVI_RT_MEM mem = tpu->runtime_mem();
        CVI_RC rc = CVI_RT_MemCopyD2S(rt, static_cast<uint8_t*>(dst_ptr), mem);
        if (rc != CVI_RC_SUCCESS) {
            throw std::runtime_error("VbMemory::copy_from - CVI_RT_MemCopyD2S failed");
        }
        flush_if_cached(phys_addr_, dst_ptr, size_bytes_, cached_);
        return;
    }
#endif

    throw std::runtime_error("VbMemory::copy_from - unsupported device");
}

void VbMemory::copy_to_async(DeviceBuffer* dst, SyncHandle* handle) const {
    copy_to(dst);
    if (handle) {
        handle->synchronize();
    }
}

void VbMemory::copy_from_async(const DeviceBuffer* src, SyncHandle* handle) {
    copy_from(src);
    if (handle) {
        handle->synchronize();
    }
}

void VbMemory::sync(SyncHandle* handle) const {
    if (handle) {
        handle->synchronize();
    }
}

} // namespace memory

#endif  // USE_CVI_MPI
