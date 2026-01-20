#include "mmf_context.h"

#include <iostream>
#include <cstring>

namespace lua_cv {

#ifdef USE_CVI_MPI
void VbPoolPlan::add_pool(uint32_t width, uint32_t height, PixelFormat format,
                          uint32_t block_count, VB_REMAP_MODE_E remap) {
    Pool pool;
    pool.width = width;
    pool.height = height;
    pool.format = format;
    pool.block_count = block_count;
    pool.remap = remap;
    pools_.push_back(pool);
}

bool VbPoolPlan::build() {
    std::memset(&vb_config_, 0, sizeof(vb_config_));
    vb_config_.u32MaxPoolCnt = static_cast<CVI_U32>(pools_.size());

    for (size_t i = 0; i < pools_.size(); ++i) {
        Pool& pool = pools_[i];
        PIXEL_FORMAT_E fmt = to_cvi_pixel_format(pool.format);
        if (fmt == PIXEL_FORMAT_MAX) {
            std::cerr << "[ERROR] VbPoolPlan: invalid pixel format" << std::endl;
            return false;
        }

        pool.block_size = COMMON_GetPicBufferSize(pool.width, pool.height, fmt,
                                                  DATA_BITWIDTH_8, COMPRESS_MODE_NONE, 0);
        vb_config_.astCommPool[i].u32BlkSize = pool.block_size;
        vb_config_.astCommPool[i].u32BlkCnt = pool.block_count;
        vb_config_.astCommPool[i].enRemapMode = pool.remap;
        std::snprintf(vb_config_.astCommPool[i].acName,
                      sizeof(vb_config_.astCommPool[i].acName),
                      "pool_%zu", i);
    }

    built_ = true;
    return true;
}

VB_POOL VbPoolPlan::find_pool(uint32_t width, uint32_t height, PixelFormat format) const {
    if (!built_ || pools_.empty()) {
        return VB_INVALID_POOLID;
    }

    uint32_t needed = find_block_size(width, height, format);
    if (needed == 0) {
        return VB_INVALID_POOLID;
    }

    VB_POOL best_pool = VB_INVALID_POOLID;
    uint32_t best_size = 0;

    for (size_t i = 0; i < pools_.size(); ++i) {
        const Pool& pool = pools_[i];
        if (pool.format != format || pool.block_size == 0) {
            continue;
        }
        if (pool.block_size >= needed &&
            (best_pool == VB_INVALID_POOLID || pool.block_size < best_size)) {
            best_pool = static_cast<VB_POOL>(i);
            best_size = pool.block_size;
        }
    }

    return best_pool;
}

uint32_t VbPoolPlan::find_block_size(uint32_t width, uint32_t height, PixelFormat format) const {
    PIXEL_FORMAT_E fmt = to_cvi_pixel_format(format);
    if (fmt == PIXEL_FORMAT_MAX) {
        return 0;
    }
    return COMMON_GetPicBufferSize(width, height, fmt, DATA_BITWIDTH_8,
                                   COMPRESS_MODE_NONE, 0);
}

MmfContext& MmfContext::instance() {
    static MmfContext ctx;
    return ctx;
}

bool MmfContext::init(const Config& config) {
    if (initialized_) {
        return true;
    }

    if (config.force_reset) {
        CVI_SYS_Exit();
        CVI_VB_Exit();
    }

    vb_plan_ = config.vb_plan;
    if (!vb_plan_.build()) {
        return false;
    }

    CVI_S32 rc = CVI_VB_SetConfig(&vb_plan_.vb_config());
    if (rc != CVI_SUCCESS) {
        std::cerr << "[ERROR] CVI_VB_SetConfig failed: " << rc << std::endl;
        return false;
    }

    rc = CVI_VB_Init();
    if (rc != CVI_SUCCESS) {
        std::cerr << "[ERROR] CVI_VB_Init failed: " << rc << std::endl;
        return false;
    }

    rc = CVI_SYS_Init();
    if (rc != CVI_SUCCESS) {
        std::cerr << "[ERROR] CVI_SYS_Init failed: " << rc << std::endl;
        CVI_VB_Exit();
        return false;
    }

    mode_plan_ = config.mode_plan;

    rc = CVI_SYS_SetVIVPSSMode(&mode_plan_.vi_vpss_mode);
    if (rc != CVI_SUCCESS) {
        std::cerr << "[ERROR] CVI_SYS_SetVIVPSSMode failed: " << rc << std::endl;
        CVI_SYS_Exit();
        CVI_VB_Exit();
        return false;
    }

    rc = CVI_SYS_SetVPSSModeEx(&mode_plan_.vpss_mode);
    if (rc != CVI_SUCCESS) {
        std::cerr << "[ERROR] CVI_SYS_SetVPSSModeEx failed: " << rc << std::endl;
        CVI_SYS_Exit();
        CVI_VB_Exit();
        return false;
    }

    std::cout << "[MMF] VB pools configured:" << std::endl;
    for (size_t i = 0; i < vb_plan_.pools().size(); ++i) {
        const auto& pool = vb_plan_.pools()[i];
        std::cout << "  Pool " << i << ": " << pool.width << "x" << pool.height
                  << " " << pixel_format_name(pool.format)
                  << " size=" << pool.block_size
                  << " count=" << pool.block_count << std::endl;
    }

    initialized_ = true;
    return true;
}

void MmfContext::shutdown() {
    if (!initialized_) {
        return;
    }
    CVI_SYS_Exit();
    CVI_VB_Exit();
    initialized_ = false;
}

void MmfContext::dump_status(const char* tag) const {
    std::cout << "\n[MMF] Status dump: " << (tag ? tag : "") << std::endl;
    std::system("cat /proc/cvitek/sys | grep -A 10 'BIND RELATION'");
    std::system("cat /proc/cvitek/vi | grep -A 5 'VI CHN STATUS'");
    std::system("cat /proc/cvitek/vpss | grep -A 20 'WORK STATUS'");
    std::system("cat /proc/cvitek/vb | grep -A 20 'PoolId.*0'");
}
#endif

} // namespace lua_cv
