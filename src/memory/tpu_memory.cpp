#include "tpu_memory.h"
#include <cstring>
#include <stdexcept>
#include <sstream>

namespace memory {

#if defined(USE_CVI_TPU)

std::shared_ptr<CviTpuMemory> CviTpuMemory::allocate(
    CVI_RT_HANDLE rt_handle,
    size_t size_bytes) {

    if (!rt_handle) {
        throw std::invalid_argument("CviTpuMemory::allocate: rt_handle cannot be null");
    }
    if (size_bytes == 0) {
        throw std::invalid_argument("CviTpuMemory::allocate: size_bytes cannot be 0");
    }

    auto mem = std::shared_ptr<CviTpuMemory>(new CviTpuMemory());
    mem->rt_handle_ = rt_handle;
    mem->size_bytes_ = size_bytes;

    mem->rt_mem_ = CVI_RT_MemAlloc(rt_handle, size_bytes);
    if (!mem->rt_mem_) {
        throw std::runtime_error("CviTpuMemory::allocate: CVI_RT_MemAlloc failed");
    }

    mem->vaddr_ = CVI_RT_MemGetVAddr(mem->rt_mem_);
    if (!mem->vaddr_) {
        CVI_RT_MemFree(rt_handle, mem->rt_mem_);
        throw std::runtime_error("CviTpuMemory::allocate: CVI_RT_MemGetVAddr failed");
    }

    mem->paddr_ = CVI_RT_MemGetPAddr(mem->rt_mem_);
    if (mem->paddr_ == 0) {
        CVI_RT_MemFree(rt_handle, mem->rt_mem_);
        throw std::runtime_error("CviTpuMemory::allocate: CVI_RT_MemGetPAddr failed");
    }

    if (mem->paddr_ & 0x3F) {
        CVI_RT_MemFree(rt_handle, mem->rt_mem_);
        std::ostringstream oss;
        oss << "CviTpuMemory::allocate: Physical address 0x" << std::hex << mem->paddr_
            << " is not 64-byte aligned";
        throw std::runtime_error(oss.str());
    }

    mem->owns_memory_ = true;
    return mem;
}

std::shared_ptr<CviTpuMemory> CviTpuMemory::from_physical(
    uint64_t paddr,
    size_t size_bytes,
    bool take_ownership) {

    if (paddr == 0) {
        throw std::invalid_argument("CviTpuMemory::from_physical: paddr cannot be 0");
    }
    if (size_bytes == 0) {
        throw std::invalid_argument("CviTpuMemory::from_physical: size_bytes cannot be 0");
    }

    if (paddr & 0x3F) {
        std::ostringstream oss;
        oss << "CviTpuMemory::from_physical: Physical address 0x" << std::hex << paddr
            << " is not 64-byte aligned (required for TPU DMA)";
        throw std::invalid_argument(oss.str());
    }

    auto mem = std::shared_ptr<CviTpuMemory>(new CviTpuMemory());
    mem->paddr_ = paddr;
    mem->vaddr_ = nullptr;
    mem->size_bytes_ = size_bytes;
    mem->owns_memory_ = take_ownership;

    return mem;
}

CviTpuMemory::~CviTpuMemory() {
    if (owns_memory_ && rt_mem_ && rt_handle_) {
        CVI_RT_MemFree(rt_handle_, rt_mem_);
        rt_mem_ = nullptr;
        vaddr_ = nullptr;
        paddr_ = 0;
    }
}

CviTpuMemory::CviTpuMemory(CviTpuMemory&& other) noexcept
    : rt_handle_(other.rt_handle_),
      rt_mem_(other.rt_mem_),
      vaddr_(other.vaddr_),
      paddr_(other.paddr_),
      size_bytes_(other.size_bytes_),
      owns_memory_(other.owns_memory_) {

    other.rt_handle_ = nullptr;
    other.rt_mem_ = nullptr;
    other.vaddr_ = nullptr;
    other.paddr_ = 0;
    other.size_bytes_ = 0;
    other.owns_memory_ = false;
}

CviTpuMemory& CviTpuMemory::operator=(CviTpuMemory&& other) noexcept {
    if (this != &other) {
        if (owns_memory_ && rt_mem_ && rt_handle_) {
            CVI_RT_MemFree(rt_handle_, rt_mem_);
        }

        rt_handle_ = other.rt_handle_;
        rt_mem_ = other.rt_mem_;
        vaddr_ = other.vaddr_;
        paddr_ = other.paddr_;
        size_bytes_ = other.size_bytes_;
        owns_memory_ = other.owns_memory_;

        other.rt_handle_ = nullptr;
        other.rt_mem_ = nullptr;
        other.vaddr_ = nullptr;
        other.paddr_ = 0;
        other.size_bytes_ = 0;
        other.owns_memory_ = false;
    }
    return *this;
}

void CviTpuMemory::copy_to(DeviceBuffer* dst) const {
    if (!dst) {
        throw std::invalid_argument("CviTpuMemory::copy_to: dst cannot be null");
    }
    if (dst->size_bytes() < size_bytes_) {
        throw std::invalid_argument("CviTpuMemory::copy_to: dst too small");
    }

    if (dst->device() == DeviceType::CPU) {
        if (!rt_handle_ || !rt_mem_) {
            throw std::runtime_error(
                "CviTpuMemory::copy_to: Cannot copy from external physical memory without runtime handle");
        }

        CVI_RC rc = CVI_RT_MemCopyD2S(rt_handle_, static_cast<uint8_t*>(dst->data()), rt_mem_);
        if (rc != CVI_SUCCESS) {
            throw std::runtime_error("CviTpuMemory::copy_to: CVI_RT_MemCopyD2S failed");
        }
    } else if (dst->device() == DeviceType::TPU) {
        if (!vaddr_) {
            throw std::runtime_error(
                "CviTpuMemory::copy_to: TPU-to-TPU copy requires virtual address mapping");
        }
        if (!dst->data()) {
            throw std::runtime_error(
                "CviTpuMemory::copy_to: Destination TPU memory has no virtual address mapping");
        }
        std::memcpy(dst->data(), vaddr_, size_bytes_);
    } else {
        throw std::runtime_error("CviTpuMemory::copy_to: Unsupported target device type");
    }
}

void CviTpuMemory::copy_from(const DeviceBuffer* src) {
    if (!src) {
        throw std::invalid_argument("CviTpuMemory::copy_from: src cannot be null");
    }
    if (size_bytes_ < src->size_bytes()) {
        throw std::invalid_argument("CviTpuMemory::copy_from: this buffer too small");
    }

    if (src->device() == DeviceType::CPU) {
        if (!rt_handle_ || !rt_mem_) {
            throw std::runtime_error(
                "CviTpuMemory::copy_from: Cannot copy to external physical memory without runtime handle");
        }

        CVI_RC rc = CVI_RT_MemCopyS2D(rt_handle_, rt_mem_, const_cast<uint8_t*>(
            static_cast<const uint8_t*>(src->data())));
        if (rc != CVI_SUCCESS) {
            throw std::runtime_error("CviTpuMemory::copy_from: CVI_RT_MemCopyS2D failed");
        }
    } else if (src->device() == DeviceType::TPU) {
        if (!vaddr_) {
            throw std::runtime_error(
                "CviTpuMemory::copy_from: TPU-to-TPU copy requires virtual address mapping");
        }
        if (!src->data()) {
            throw std::runtime_error(
                "CviTpuMemory::copy_from: Source TPU memory has no virtual address mapping");
        }
        std::memcpy(vaddr_, src->data(), src->size_bytes());
    } else {
        throw std::runtime_error("CviTpuMemory::copy_from: Unsupported source device type");
    }
}

void CviTpuMemory::copy_to_async(DeviceBuffer* dst, SyncHandle* handle) const {
    (void)handle;
    copy_to(dst);
}

void CviTpuMemory::copy_from_async(const DeviceBuffer* src, SyncHandle* handle) {
    (void)handle;
    copy_from(src);
}

void CviTpuMemory::sync(SyncHandle* handle) const {
    (void)handle;
}

void CviTpuMemory::flush_cache() {
    if (!rt_handle_ || !rt_mem_) {
        return;
    }

    CVI_RC rc = CVI_RT_MemFlush(rt_handle_, rt_mem_);
    if (rc != CVI_SUCCESS) {
        throw std::runtime_error("CviTpuMemory::flush_cache: CVI_RT_MemFlush failed");
    }
}

void CviTpuMemory::invalidate_cache() {
    if (!rt_handle_ || !rt_mem_) {
        return;
    }

    CVI_RC rc = CVI_RT_MemInvld(rt_handle_, rt_mem_);
    if (rc != CVI_SUCCESS) {
        throw std::runtime_error("CviTpuMemory::invalidate_cache: CVI_RT_MemInvld failed");
    }
}

#endif // USE_CVI_TPU

} // namespace memory
