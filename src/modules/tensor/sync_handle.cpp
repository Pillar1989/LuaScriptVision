#include "sync_handle.h"
#include <stdexcept>

namespace tensor {

std::shared_ptr<SyncHandle> SyncHandle::create(DeviceType device) {
    switch (device) {
        case DeviceType::CPU:
            return std::make_shared<CpuSyncHandle>();

        case DeviceType::VB:
            return std::make_shared<CpuSyncHandle>();

        case DeviceType::NPU:
            throw std::runtime_error("SyncHandle::create: NPU handle not implemented");

        case DeviceType::TPU:
            throw std::runtime_error("SyncHandle::create: TPU handle not implemented");

        default:
            throw std::invalid_argument("SyncHandle::create: unknown device type");
    }
}

} // namespace tensor
