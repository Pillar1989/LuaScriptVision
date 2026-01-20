#include "device_buffer.h"

#include <cstring>

namespace lua_cv {

DeviceBuffer::DeviceBuffer(DeviceType device, size_t size)
    : device_(device), size_(size) {}

DeviceBuffer::~DeviceBuffer() {
#ifdef USE_CVI_MPI
    if (device_ != DeviceType::CPU && data_ && mapped_size_ > 0) {
        CVI_SYS_Munmap(data_, static_cast<CVI_U32>(mapped_size_));
    }
#endif
}

std::shared_ptr<DeviceBuffer> DeviceBuffer::CreateCpu(size_t size) {
    auto buffer = std::shared_ptr<DeviceBuffer>(new DeviceBuffer(DeviceType::CPU, size));
    buffer->owner_.reset(new uint8_t[size], std::default_delete<uint8_t[]>());
    buffer->data_ = buffer->owner_.get();
    buffer->mapped_size_ = size;
    return buffer;
}

std::shared_ptr<DeviceBuffer> DeviceBuffer::WrapCpu(void* data, size_t size,
                                                    std::shared_ptr<void> owner) {
    auto buffer = std::shared_ptr<DeviceBuffer>(new DeviceBuffer(DeviceType::CPU, size));
    buffer->owner_ = std::move(owner);
    buffer->data_ = data;
    buffer->mapped_size_ = size;
    return buffer;
}

#ifdef USE_CVI_MPI
std::shared_ptr<DeviceBuffer> DeviceBuffer::FromPhysical(uint64_t phys_addr, size_t size, bool cached) {
    auto buffer = std::shared_ptr<DeviceBuffer>(new DeviceBuffer(DeviceType::VB, size));
    buffer->phys_addr_ = phys_addr;
    buffer->cached_ = cached;
    return buffer;
}
#endif

void* DeviceBuffer::data() {
    if (device_ == DeviceType::CPU) {
        return data_;
    }
#ifdef USE_CVI_MPI
    if (!data_ && phys_addr_ != 0) {
        if (cached_) {
            data_ = CVI_SYS_MmapCache(phys_addr_, static_cast<CVI_U32>(size_));
        } else {
            data_ = CVI_SYS_Mmap(phys_addr_, static_cast<CVI_U32>(size_));
        }
        mapped_size_ = size_;
    }
#endif
    return data_;
}

const void* DeviceBuffer::data() const {
    return const_cast<DeviceBuffer*>(this)->data();
}

#ifdef USE_CVI_MPI
void DeviceBuffer::flush_cache() {
    if (phys_addr_ == 0 || data_ == nullptr || !cached_) {
        return;
    }
    CVI_SYS_IonFlushCache(phys_addr_, data_, static_cast<CVI_U32>(size_));
}

void DeviceBuffer::invalidate_cache() {
    if (phys_addr_ == 0 || data_ == nullptr || !cached_) {
        return;
    }
    CVI_SYS_IonInvalidateCache(phys_addr_, data_, static_cast<CVI_U32>(size_));
}
#endif

} // namespace lua_cv
