#include "sync_handle.h"
#include <stdexcept>

namespace memory {

std::shared_ptr<SyncHandle> SyncHandle::create(DeviceType device) {
    switch (device) {
        case DeviceType::CPU:
        case DeviceType::VB:
            return std::make_shared<CpuSyncHandle>();

        case DeviceType::TPU:
            throw std::runtime_error("SyncHandle::create: TPU handle not implemented");

        default:
            throw std::invalid_argument("SyncHandle::create: unknown device type");
    }
}

} // namespace memory
