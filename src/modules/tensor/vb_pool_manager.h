#pragma once

#ifdef USE_CVI_MPI

#include <cstdint>
#include <vector>

#include <cvi_vb.h>
#include <cvi_sys.h>
#include <cvi_buffer.h>

namespace tensor {

struct VbPoolConfig {
    uint32_t width = 0;
    uint32_t height = 0;
    PIXEL_FORMAT_E pixel_format = PIXEL_FORMAT_MAX;
    uint32_t block_count = 0;
    bool cached = true;

    uint32_t block_size() const;
    VB_REMAP_MODE_E remap_mode() const;
};

class VbPoolManager {
public:
    struct PoolStatus {
        uint32_t pool_id = 0;
        uint32_t total_blocks = 0;
        uint32_t free_blocks = 0;
        uint32_t block_size = 0;
    };

    static VbPoolManager& instance();

    void init(const std::vector<VbPoolConfig>& configs);
    void shutdown();
    bool is_initialized() const { return initialized_; }

    VB_BLK get_block(size_t min_size);
    void release_block(VB_BLK block);

    std::vector<PoolStatus> get_status() const;

private:
    VbPoolManager() = default;

    bool initialized_ = false;
    bool owns_vb_ = false;
    bool owns_sys_ = false;
    std::vector<VbPoolConfig> configs_;
    VB_CONFIG_S vb_config_{};

    VB_POOL select_pool(size_t min_size) const;
    bool config_matches(const VB_CONFIG_S& other) const;
};

} // namespace tensor

#endif  // USE_CVI_MPI
