#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "cv_types.h"

#ifdef USE_CVI_MPI
#include <cvi_sys.h>
#endif

namespace lua_cv {

class DeviceBuffer {
public:
    static std::shared_ptr<DeviceBuffer> CreateCpu(size_t size);
    static std::shared_ptr<DeviceBuffer> WrapCpu(void* data, size_t size,
                                                 std::shared_ptr<void> owner);

#ifdef USE_CVI_MPI
    static std::shared_ptr<DeviceBuffer> FromPhysical(uint64_t phys_addr, size_t size, bool cached);
#endif

    DeviceType device() const { return device_; }
    size_t size() const { return size_; }
    uint64_t physical_addr() const { return phys_addr_; }
    bool is_cached() const { return cached_; }

    void* data();
    const void* data() const;

#ifdef USE_CVI_MPI
    void flush_cache();
    void invalidate_cache();
#endif

    ~DeviceBuffer();

private:
    DeviceBuffer(DeviceType device, size_t size);

    DeviceType device_ = DeviceType::UNKNOWN;
    size_t size_ = 0;
    uint64_t phys_addr_ = 0;
    bool cached_ = false;

    void* data_ = nullptr;
    size_t mapped_size_ = 0;
    std::shared_ptr<void> owner_;
};

} // namespace lua_cv
