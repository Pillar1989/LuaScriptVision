#include "cvi_tpu_memory.h"
#include <cstring>
#include <stdexcept>
#include <sstream>

namespace tensor {

#if defined(USE_CVI_TPU)

// ========== 工厂方法 ==========

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

    // 使用 CVI Runtime 官方 API 分配内存
    mem->rt_mem_ = CVI_RT_MemAlloc(rt_handle, size_bytes);
    if (!mem->rt_mem_) {
        throw std::runtime_error("CviTpuMemory::allocate: CVI_RT_MemAlloc failed");
    }

    // 获取虚拟地址（CPU 端可以 memcpy）
    mem->vaddr_ = CVI_RT_MemGetVAddr(mem->rt_mem_);
    if (!mem->vaddr_) {
        // 释放已分配的内存
        CVI_RT_MemFree(rt_handle, mem->rt_mem_);
        throw std::runtime_error("CviTpuMemory::allocate: CVI_RT_MemGetVAddr failed");
    }

    // 获取物理地址（Sophgo TPU DMA 使用，零拷贝）
    mem->paddr_ = CVI_RT_MemGetPAddr(mem->rt_mem_);
    if (mem->paddr_ == 0) {
        // 释放已分配的内存
        CVI_RT_MemFree(rt_handle, mem->rt_mem_);
        throw std::runtime_error("CviTpuMemory::allocate: CVI_RT_MemGetPAddr failed");
    }

    // 验证对齐（CVI Runtime 应该保证 64 字节对齐）
    if (mem->paddr_ & 0x3F) {
        // 释放已分配的内存
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

    // 验证 64 字节对齐（关键！）
    if (paddr & 0x3F) {  // 检查低 6 位（64 字节 = 2^6）
        std::ostringstream oss;
        oss << "CviTpuMemory::from_physical: Physical address 0x" << std::hex << paddr
            << " is not 64-byte aligned (required for TPU DMA)";
        throw std::invalid_argument(oss.str());
    }

    auto mem = std::shared_ptr<CviTpuMemory>(new CviTpuMemory());
    mem->paddr_ = paddr;
    mem->vaddr_ = nullptr;  // 外部物理内存可能无虚拟地址映射
    mem->size_bytes_ = size_bytes;
    mem->owns_memory_ = take_ownership;

    // 注意：rt_handle_ 和 rt_mem_ 保持为 nullptr
    // 外部物理内存不由 CVI Runtime 管理

    return mem;
}

// ========== 析构函数 ==========

CviTpuMemory::~CviTpuMemory() {
    if (owns_memory_ && rt_mem_ && rt_handle_) {
        // 使用官方 API 释放内存
        CVI_RT_MemFree(rt_handle_, rt_mem_);
        rt_mem_ = nullptr;
        vaddr_ = nullptr;
        paddr_ = 0;
    }
}

// ========== 移动语义 ==========

CviTpuMemory::CviTpuMemory(CviTpuMemory&& other) noexcept
    : rt_handle_(other.rt_handle_),
      rt_mem_(other.rt_mem_),
      vaddr_(other.vaddr_),
      paddr_(other.paddr_),
      size_bytes_(other.size_bytes_),
      owns_memory_(other.owns_memory_) {

    // 清空源对象
    other.rt_handle_ = nullptr;
    other.rt_mem_ = nullptr;
    other.vaddr_ = nullptr;
    other.paddr_ = 0;
    other.size_bytes_ = 0;
    other.owns_memory_ = false;
}

CviTpuMemory& CviTpuMemory::operator=(CviTpuMemory&& other) noexcept {
    if (this != &other) {
        // 释放当前资源
        if (owns_memory_ && rt_mem_ && rt_handle_) {
            CVI_RT_MemFree(rt_handle_, rt_mem_);
        }

        // 移动资源
        rt_handle_ = other.rt_handle_;
        rt_mem_ = other.rt_mem_;
        vaddr_ = other.vaddr_;
        paddr_ = other.paddr_;
        size_bytes_ = other.size_bytes_;
        owns_memory_ = other.owns_memory_;

        // 清空源对象
        other.rt_handle_ = nullptr;
        other.rt_mem_ = nullptr;
        other.vaddr_ = nullptr;
        other.paddr_ = 0;
        other.size_bytes_ = 0;
        other.owns_memory_ = false;
    }
    return *this;
}

// ========== 数据传输 ==========

void CviTpuMemory::copy_to(DeviceBuffer* dst) const {
    if (!dst) {
        throw std::invalid_argument("CviTpuMemory::copy_to: dst cannot be null");
    }
    if (dst->size_bytes() < size_bytes_) {
        throw std::invalid_argument("CviTpuMemory::copy_to: dst too small");
    }

    if (dst->device() == DeviceType::CPU) {
        // Sophgo TPU -> CPU: 使用 CVI_RT_MemCopyD2S
        if (!rt_handle_ || !rt_mem_) {
            throw std::runtime_error(
                "CviTpuMemory::copy_to: Cannot copy from external physical memory without runtime handle");
        }

        CVI_RC rc = CVI_RT_MemCopyD2S(rt_handle_, static_cast<uint8_t*>(dst->data()), rt_mem_);
        if (rc != CVI_SUCCESS) {
            throw std::runtime_error("CviTpuMemory::copy_to: CVI_RT_MemCopyD2S failed");
        }
    } else if (dst->device() == DeviceType::TPU) {
        // Sophgo TPU -> TPU: CVI Runtime 不提供 D2D API，需要通过虚拟地址
        if (!vaddr_) {
            throw std::runtime_error(
                "CviTpuMemory::copy_to: TPU-to-TPU copy requires virtual address mapping. "
                "External physical memory (from from_physical) cannot be copied directly. "
                "Consider using CPU as intermediate: tpu->cpu->tpu");
        }
        if (!dst->data()) {
            throw std::runtime_error(
                "CviTpuMemory::copy_to: Destination TPU memory has no virtual address mapping");
        }
        // 使用虚拟地址进行内存拷贝（CPU 侧操作）
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
        // CPU -> Sophgo TPU: 使用 CVI_RT_MemCopyS2D
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
        // Sophgo TPU -> TPU: CVI Runtime 不提供 D2D API，需要通过虚拟地址
        if (!vaddr_) {
            throw std::runtime_error(
                "CviTpuMemory::copy_from: TPU-to-TPU copy requires virtual address mapping. "
                "External physical memory (from from_physical) cannot be copied directly. "
                "Consider using CPU as intermediate: tpu->cpu->tpu");
        }
        if (!src->data()) {
            throw std::runtime_error(
                "CviTpuMemory::copy_from: Source TPU memory has no virtual address mapping");
        }
        // 使用虚拟地址进行内存拷贝（CPU 侧操作）
        std::memcpy(vaddr_, src->data(), src->size_bytes());
    } else {
        throw std::runtime_error("CviTpuMemory::copy_from: Unsupported source device type");
    }
}

// ========== 异步接口实现 ==========

void CviTpuMemory::copy_to_async(DeviceBuffer* dst, SyncHandle* handle) const {
    // CVI Runtime 不提供异步内存拷贝 API（CVI_RT_MemCopyS2D/D2S 均为同步）
    // 异步操作仅适用于推理 (CVI_NN_ForwardAsync)，不适用于内存传输
    // 因此此处实现为同步执行，与 CpuMemory 行为一致
    (void)handle;  // 忽略未使用的同步句柄
    copy_to(dst);
}

void CviTpuMemory::copy_from_async(const DeviceBuffer* src, SyncHandle* handle) {
    // CVI Runtime 不提供异步内存拷贝 API
    // 同步执行内存拷贝
    (void)handle;  // 忽略未使用的同步句柄
    copy_from(src);
}

void CviTpuMemory::sync(SyncHandle* handle) const {
    // CVI Runtime 内存拷贝为同步操作，无需等待
    // 此方法为空实现，保持接口一致性
    (void)handle;  // 忽略未使用的同步句柄
}

// ========== Cache 一致性管理 ==========

void CviTpuMemory::flush_cache() {
    if (!rt_handle_ || !rt_mem_) {
        // 外部物理内存无需刷新 cache（假设调用者已处理）
        return;
    }

    // 刷新 cache（CPU 写入后，确保 TPU 看到最新数据）
    CVI_RC rc = CVI_RT_MemFlush(rt_handle_, rt_mem_);
    if (rc != CVI_SUCCESS) {
        throw std::runtime_error("CviTpuMemory::flush_cache: CVI_RT_MemFlush failed");
    }
}

void CviTpuMemory::invalidate_cache() {
    if (!rt_handle_ || !rt_mem_) {
        // 外部物理内存无需失效 cache
        return;
    }

    // 失效 cache（TPU 写入后，确保 CPU 读取最新数据）
    CVI_RC rc = CVI_RT_MemInvld(rt_handle_, rt_mem_);
    if (rc != CVI_SUCCESS) {
        throw std::runtime_error("CviTpuMemory::invalidate_cache: CVI_RT_MemInvld failed");
    }
}

#endif // USE_CVI_TPU

} // namespace tensor
