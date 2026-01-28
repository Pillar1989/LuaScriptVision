#pragma once

#include <memory>
#include "device_type.h"

namespace memory {

class SyncHandle {
public:
    virtual ~SyncHandle() = default;
    virtual void synchronize() = 0;
    virtual bool is_idle() const = 0;
    virtual DeviceType device() const = 0;

    static std::shared_ptr<SyncHandle> create(DeviceType device);
};

class CpuSyncHandle : public SyncHandle {
public:
    CpuSyncHandle() = default;
    ~CpuSyncHandle() override = default;

    void synchronize() override {}
    bool is_idle() const override { return true; }
    DeviceType device() const override { return DeviceType::CPU; }
};

} // namespace memory

// Backward compatibility
namespace tensor {
    using SyncHandle = memory::SyncHandle;
    using CpuSyncHandle = memory::CpuSyncHandle;
} // namespace tensor
