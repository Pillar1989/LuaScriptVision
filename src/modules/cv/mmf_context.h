#pragma once

#include <cstdint>
#include <vector>

#include "cv_types.h"

#ifdef USE_CVI_MPI
#include <cvi_sys.h>
#include <cvi_vb.h>
#include <cvi_buffer.h>
#include <cvi_comm_vb.h>
#endif

namespace lua_cv {

#ifdef USE_CVI_MPI
class VbPoolPlan {
public:
    struct Pool {
        uint32_t width = 0;
        uint32_t height = 0;
        PixelFormat format = PixelFormat::UNKNOWN;
        uint32_t block_count = 0;
        VB_REMAP_MODE_E remap = VB_REMAP_MODE_CACHED;
        uint32_t block_size = 0;
    };

    void add_pool(uint32_t width, uint32_t height, PixelFormat format,
                  uint32_t block_count, VB_REMAP_MODE_E remap);

    bool build();
    const VB_CONFIG_S& vb_config() const { return vb_config_; }
    const std::vector<Pool>& pools() const { return pools_; }

    VB_POOL find_pool(uint32_t width, uint32_t height, PixelFormat format) const;
    uint32_t find_block_size(uint32_t width, uint32_t height, PixelFormat format) const;

private:
    std::vector<Pool> pools_;
    VB_CONFIG_S vb_config_{};
    bool built_ = false;
};

class MmfContext {
public:
    struct ModePlan {
        VI_VPSS_MODE_S vi_vpss_mode{};
        VPSS_MODE_S vpss_mode{};
    };

    struct Config {
        VbPoolPlan vb_plan;
        ModePlan mode_plan;
        bool force_reset = false;
    };

    static MmfContext& instance();

    bool init(const Config& config);
    void shutdown();
    bool is_initialized() const { return initialized_; }

    const VbPoolPlan& vb_plan() const { return vb_plan_; }
    const ModePlan& mode_plan() const { return mode_plan_; }

    void dump_status(const char* tag) const;

private:
    MmfContext() = default;

    bool initialized_ = false;
    VbPoolPlan vb_plan_;
    ModePlan mode_plan_;
};
#endif

} // namespace lua_cv
