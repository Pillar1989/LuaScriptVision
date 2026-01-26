#include "vb_pool_manager.h"

#ifdef USE_CVI_MPI

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace tensor {

uint32_t VbPoolConfig::block_size() const {
    if (width == 0 || height == 0 || block_count == 0 || pixel_format == PIXEL_FORMAT_MAX) {
        return 0;
    }
    return COMMON_GetPicBufferSize(width, height, pixel_format,
                                   DATA_BITWIDTH_8, COMPRESS_MODE_NONE, 0);
}

VB_REMAP_MODE_E VbPoolConfig::remap_mode() const {
    return cached ? VB_REMAP_MODE_CACHED : VB_REMAP_MODE_NONE;
}

VbPoolManager& VbPoolManager::instance() {
    static VbPoolManager mgr;
    return mgr;
}

void VbPoolManager::init(const std::vector<VbPoolConfig>& configs) {
    if (initialized_) {
        return;
    }
    if (configs.empty()) {
        throw std::invalid_argument("VbPoolManager::init - configs empty");
    }
    if (configs.size() > VB_MAX_COMM_POOLS) {
        throw std::invalid_argument("VbPoolManager::init - too many pools");
    }

    std::memset(&vb_config_, 0, sizeof(vb_config_));
    vb_config_.u32MaxPoolCnt = static_cast<CVI_U32>(configs.size());
    for (size_t i = 0; i < configs.size(); ++i) {
        uint32_t blk_size = configs[i].block_size();
        if (blk_size == 0) {
            throw std::invalid_argument("VbPoolManager::init - invalid pool config");
        }
        vb_config_.astCommPool[i].u32BlkSize = blk_size;
        vb_config_.astCommPool[i].u32BlkCnt = configs[i].block_count;
        vb_config_.astCommPool[i].enRemapMode = configs[i].remap_mode();
        std::snprintf(vb_config_.astCommPool[i].acName,
                      sizeof(vb_config_.astCommPool[i].acName),
                      "pool_%zu", i);
    }

    if (CVI_VB_IsInited() == CVI_TRUE) {
        VB_CONFIG_S current{};
        CVI_S32 rc = CVI_VB_GetConfig(&current);
        if (rc != CVI_SUCCESS) {
            throw std::runtime_error("VbPoolManager::init - CVI_VB_GetConfig failed");
        }
        if (!config_matches(current)) {
            throw std::runtime_error("VbPoolManager::init - VB already initialized with different config");
        }
        configs_ = configs;
        initialized_ = true;
        owns_vb_ = false;
        if (CVI_SYS_IsInited() != CVI_TRUE) {
            rc = CVI_SYS_Init();
            if (rc != CVI_SUCCESS) {
                throw std::runtime_error("VbPoolManager::init - CVI_SYS_Init failed");
            }
            owns_sys_ = true;
        } else {
            owns_sys_ = false;
        }
        return;
    }

    CVI_S32 rc = CVI_VB_SetConfig(&vb_config_);
    if (rc != CVI_SUCCESS) {
        throw std::runtime_error("VbPoolManager::init - CVI_VB_SetConfig failed");
    }

    rc = CVI_VB_Init();
    if (rc != CVI_SUCCESS) {
        throw std::runtime_error("VbPoolManager::init - CVI_VB_Init failed");
    }
    owns_vb_ = true;

    if (CVI_SYS_IsInited() != CVI_TRUE) {
        rc = CVI_SYS_Init();
        if (rc != CVI_SUCCESS) {
            CVI_VB_Exit();
            owns_vb_ = false;
            throw std::runtime_error("VbPoolManager::init - CVI_SYS_Init failed");
        }
        owns_sys_ = true;
    }

    configs_ = configs;
    initialized_ = true;
}

void VbPoolManager::shutdown() {
    if (!initialized_) {
        return;
    }
    if (owns_sys_) {
        CVI_SYS_Exit();
        owns_sys_ = false;
    }
    if (owns_vb_) {
        CVI_VB_Exit();
        owns_vb_ = false;
    }
    configs_.clear();
    std::memset(&vb_config_, 0, sizeof(vb_config_));
    initialized_ = false;
}

VB_POOL VbPoolManager::select_pool(size_t min_size) const {
    VB_CONFIG_S current{};
    CVI_S32 rc = CVI_VB_GetConfig(&current);
    if (rc != CVI_SUCCESS || current.u32MaxPoolCnt == 0) {
        return VB_INVALID_POOLID;
    }

    VB_POOL best_pool = VB_INVALID_POOLID;
    uint32_t best_size = 0;

    for (uint32_t i = 0; i < current.u32MaxPoolCnt; ++i) {
        uint32_t blk_size = current.astCommPool[i].u32BlkSize;
        uint32_t blk_cnt = current.astCommPool[i].u32BlkCnt;
        if (blk_size == 0 || blk_cnt == 0) {
            continue;
        }
        if (blk_size >= min_size && (best_pool == VB_INVALID_POOLID || blk_size < best_size)) {
            best_pool = static_cast<VB_POOL>(i);
            best_size = blk_size;
        }
    }

    return best_pool;
}

VB_BLK VbPoolManager::get_block(size_t min_size) {
    if (!initialized_) {
        throw std::runtime_error("VbPoolManager::get_block - not initialized");
    }
    if (min_size == 0) {
        throw std::invalid_argument("VbPoolManager::get_block - min_size must be > 0");
    }

    VB_POOL pool = select_pool(min_size);
    VB_BLK block = CVI_VB_GetBlock(pool, static_cast<CVI_U32>(min_size));
    if (block != VB_INVALID_HANDLE) {
        return block;
    }

    if (pool != VB_INVALID_POOLID) {
        block = CVI_VB_GetBlock(VB_INVALID_POOLID, static_cast<CVI_U32>(min_size));
    }
    return block;
}

void VbPoolManager::release_block(VB_BLK block) {
    if (block == VB_INVALID_HANDLE) {
        throw std::invalid_argument("VbPoolManager::release_block - invalid VB block");
    }
    CVI_S32 rc = CVI_VB_ReleaseBlock(block);
    if (rc != CVI_SUCCESS) {
        throw std::runtime_error("VbPoolManager::release_block - CVI_VB_ReleaseBlock failed");
    }
}

std::vector<VbPoolManager::PoolStatus> VbPoolManager::get_status() const {
    std::ifstream proc("/proc/cvitek/vb");
    if (!proc.is_open()) {
        throw std::runtime_error("VbPoolManager::get_status - failed to open /proc/cvitek/vb");
    }

    std::vector<PoolStatus> status_list;
    std::string line;
    PoolStatus current{};
    bool in_pool = false;

    auto parse_value = [](const std::string& text) -> uint32_t {
        std::string value = text;
        value.erase(0, value.find_first_not_of(" \t:"));
        return static_cast<uint32_t>(std::stoul(value));
    };

    while (std::getline(proc, line)) {
        if (line.rfind("PoolId", 0) == 0 && line.find(':') != std::string::npos) {
            if (in_pool) {
                status_list.push_back(current);
            }
            current = PoolStatus{};
            current.pool_id = parse_value(line.substr(line.find(':') + 1));
            in_pool = true;
            continue;
        }
        if (!in_pool) {
            continue;
        }
        if (line.rfind("BlkCnt", 0) == 0 && line.find(':') != std::string::npos) {
            current.total_blocks = parse_value(line.substr(line.find(':') + 1));
            continue;
        }
        if (line.rfind("BlkSz", 0) == 0 && line.find(':') != std::string::npos) {
            current.block_size = parse_value(line.substr(line.find(':') + 1));
            continue;
        }
        if (line.rfind("Free", 0) == 0 && line.find(':') != std::string::npos) {
            current.free_blocks = parse_value(line.substr(line.find(':') + 1));
            continue;
        }
    }

    if (in_pool) {
        status_list.push_back(current);
    }

    std::sort(status_list.begin(), status_list.end(),
              [](const PoolStatus& a, const PoolStatus& b) {
                  return a.pool_id < b.pool_id;
              });

    return status_list;
}

bool VbPoolManager::config_matches(const VB_CONFIG_S& other) const {
    if (vb_config_.u32MaxPoolCnt != other.u32MaxPoolCnt) {
        return false;
    }

    for (uint32_t i = 0; i < vb_config_.u32MaxPoolCnt; ++i) {
        const VB_POOL_CONFIG_S& expected = vb_config_.astCommPool[i];
        const VB_POOL_CONFIG_S& actual = other.astCommPool[i];
        if (expected.u32BlkSize != actual.u32BlkSize) {
            return false;
        }
        if (expected.u32BlkCnt != actual.u32BlkCnt) {
            return false;
        }
        if (expected.enRemapMode != actual.enRemapMode) {
            return false;
        }
    }
    return true;
}

} // namespace tensor

#endif  // USE_CVI_MPI
