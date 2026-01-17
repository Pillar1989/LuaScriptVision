#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include "device_buffer.h"

// 条件编译：仅在启用 Sophgo TPU 支持时包含 CVI Runtime 头文件
#if defined(USE_CVI_TPU)
#include <cviruntime.h>
#include <cviruntime_context.h>
#endif

namespace tensor {

#if defined(USE_CVI_TPU)

/**
 * CviTpuMemory - Sophgo TPU 内存管理实现
 *
 * 特点：
 * - 使用 CVI Runtime 官方 API (CVI_RT_MemAlloc/Free)
 * - 双重寻址：vaddr (CPU 虚拟地址) + paddr (TPU 物理地址)
 * - 64 字节对齐（TPU DMA 要求，由 CVI Runtime 自动保证）
 * - 支持零拷贝优化（物理地址模式）
 * - Cache 一致性管理 (CVI_RT_MemFlush/Invld)
 *
 * 使用场景：
 * 1. 通用推理：allocate() 分配新内存
 * 2. 相机零拷贝：from_physical() 包装外部物理内存（高级用法）
 */
class CviTpuMemory : public DeviceBuffer {
public:
    /**
     * 分配 Sophgo TPU 内存（推荐的通用方式）
     * @param rt_handle CVI Runtime 上下文句柄
     * @param size_bytes 字节数
     * @return shared_ptr 指向 CviTpuMemory
     *
     * 注意：
     * - 使用 CVI_RT_MemAlloc 官方 API
     * - 自动保证 64 字节对齐
     * - 同时获取 vaddr (CPU) 和 paddr (TPU)
     */
    static std::shared_ptr<CviTpuMemory> allocate(
        CVI_RT_HANDLE rt_handle,
        size_t size_bytes
    );

    /**
     * 包装外部物理内存（仅用于特殊优化，如相机零拷贝）
     * @param paddr 物理地址（必须 64 字节对齐）
     * @param size_bytes 字节数
     * @param take_ownership 是否拥有所有权（默认 false）
     * @return shared_ptr 指向 CviTpuMemory
     *
     * 警告：
     * - 物理地址必须 64 字节对齐，否则抛出异常
     * - 外部物理内存的生命周期由调用者管理
     * - 仅用于高级优化场景（如相机驱动提供的物理地址）
     */
    static std::shared_ptr<CviTpuMemory> from_physical(
        uint64_t paddr,
        size_t size_bytes,
        bool take_ownership = false
    );

    ~CviTpuMemory() override;

    // 禁用拷贝
    CviTpuMemory(const CviTpuMemory&) = delete;
    CviTpuMemory& operator=(const CviTpuMemory&) = delete;

    // 允许移动
    CviTpuMemory(CviTpuMemory&& other) noexcept;
    CviTpuMemory& operator=(CviTpuMemory&& other) noexcept;

    // ========== DeviceBuffer 接口实现 ==========

    /**
     * 获取虚拟地址（CPU 可访问）
     * 注意：from_physical() 创建的对象可能返回 nullptr
     */
    void* data() override { return vaddr_; }
    const void* data() const override { return vaddr_; }

    size_t size_bytes() const override { return size_bytes_; }
    DeviceType device() const override { return DeviceType::TPU; }

    /**
     * 返回 64 字节对齐（CVI Runtime 保证）
     */
    size_t alignment() const override { return 64; }

    bool owns_memory() const override { return owns_memory_; }

    /**
     * 复制数据到目标缓冲区
     * - TPU -> TPU: 使用 CVI_RT_MemCopy (TODO)
     * - TPU -> CPU: 使用 CVI_RT_MemCopyD2S
     */
    void copy_to(DeviceBuffer* dst) const override;

    /**
     * 从源缓冲区复制数据
     * - CPU -> TPU: 使用 CVI_RT_MemCopyS2D
     * - TPU -> TPU: 使用 CVI_RT_MemCopy (TODO)
     */
    void copy_from(const DeviceBuffer* src) override;

    // ========== 异步接口 ==========

    void copy_to_async(DeviceBuffer* dst, SyncHandle* handle = nullptr) const override;
    void copy_from_async(const DeviceBuffer* src, SyncHandle* handle = nullptr) override;
    void sync(SyncHandle* handle = nullptr) const override;

    /**
     * Sophgo TPU 内存拷贝不支持异步操作
     * CVI Runtime 仅在推理层面支持异步 (CVI_NN_ForwardAsync)
     * 内存传输 API (CVI_RT_MemCopyS2D/D2S) 为同步操作
     */
    bool supports_async() const override { return false; }

    // ========== Sophgo TPU 特有接口 ==========

    /**
     * 获取物理地址（用于 CVI_NN_SetTensorPhysicalAddr）
     */
    uint64_t physical_addr() const { return paddr_; }

    /**
     * 获取 CVI Runtime 句柄
     */
    CVI_RT_HANDLE runtime_handle() const { return rt_handle_; }

    /**
     * 获取 CVI Runtime 内存句柄（内部使用）
     */
    CVI_RT_MEM runtime_mem() const { return rt_mem_; }

    /**
     * 刷新 cache（CPU 写入后，确保 TPU 看到最新数据）
     */
    void flush_cache();

    /**
     * 失效 cache（TPU 写入后，确保 CPU 读取最新数据）
     */
    void invalidate_cache();

private:
    CviTpuMemory() = default;

    CVI_RT_HANDLE rt_handle_ = nullptr;  // CVI Runtime 上下文
    CVI_RT_MEM rt_mem_ = nullptr;        // CVI Runtime 内存句柄
    uint8_t* vaddr_ = nullptr;           // 虚拟地址（CPU 访问）
    uint64_t paddr_ = 0;                 // 物理地址（TPU DMA）
    size_t size_bytes_ = 0;              // 字节数
    bool owns_memory_ = false;           // 所有权标志
};

#endif // USE_CVI_TPU

} // namespace tensor
