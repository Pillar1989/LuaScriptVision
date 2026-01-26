#include <gtest/gtest.h>

#ifdef USE_CVI_MPI

#include <set>
#include <cstring>
#include <unistd.h>
#include "modules/tensor/vb_pool_manager.h"
#include "modules/tensor/vb_memory.h"

namespace tensor {
namespace {
std::vector<VbPoolConfig> default_configs() {
    std::vector<VbPoolConfig> configs;
    VbPoolConfig pool;
    pool.width = 640;
    pool.height = 640;
    pool.pixel_format = PIXEL_FORMAT_RGB_888;
    pool.block_count = 4;
    pool.cached = true;
    configs.push_back(pool);
    return configs;
}

VbPoolManager::PoolStatus find_pool(const std::vector<VbPoolManager::PoolStatus>& status,
                                    uint32_t pool_id) {
    for (const auto& entry : status) {
        if (entry.pool_id == pool_id) {
            return entry;
        }
    }
    return VbPoolManager::PoolStatus{};
}

class VbPoolGuard {
public:
    explicit VbPoolGuard(const std::vector<VbPoolConfig>& configs) {
        VbPoolManager::instance().init(configs);
    }

    ~VbPoolGuard() {
        VbPoolManager::instance().shutdown();
    }
};
}  // namespace

TEST(VbPoolManagerTest, InitAndShutdown) {
    {
        VbPoolGuard guard(default_configs());
        EXPECT_TRUE(VbPoolManager::instance().is_initialized());
        auto status = VbPoolManager::instance().get_status();
        EXPECT_FALSE(status.empty());
    }
    EXPECT_FALSE(VbPoolManager::instance().is_initialized());
}

TEST(VbPoolManagerTest, MultiplePoolsConfig) {
    std::vector<VbPoolConfig> configs;
    configs.push_back({1920, 1080, PIXEL_FORMAT_NV21, 4, true});
    configs.push_back({640, 640, PIXEL_FORMAT_RGB_888, 4, true});
    configs.push_back({1280, 720, PIXEL_FORMAT_NV21, 3, false});

    VbPoolGuard guard(configs);
    auto status = VbPoolManager::instance().get_status();
    EXPECT_GE(status.size(), configs.size());

    auto pool0 = find_pool(status, 0);
    auto pool1 = find_pool(status, 1);
    auto pool2 = find_pool(status, 2);

    EXPECT_GE(pool0.block_size, 1920u * 1080u * 3u / 2u);
    EXPECT_GE(pool1.block_size, 640u * 640u * 3u);
    EXPECT_GE(pool2.block_size, 1280u * 720u * 3u / 2u);
}

TEST(VbPoolManagerTest, SingleBlockAllocation) {
    VbPoolGuard guard(default_configs());

    auto before = VbPoolManager::instance().get_status();
    ASSERT_FALSE(before.empty());
    auto before_pool = find_pool(before, 0);

    VB_BLK block = VbPoolManager::instance().get_block(640 * 640 * 3);
    ASSERT_NE(block, VB_INVALID_HANDLE);
    uint64_t phys = CVI_VB_Handle2PhysAddr(block);
    EXPECT_NE(phys, 0u);

    auto during = VbPoolManager::instance().get_status();
    auto during_pool = find_pool(during, 0);
    EXPECT_EQ(during_pool.free_blocks, before_pool.free_blocks - 1);

    VbPoolManager::instance().release_block(block);
    usleep(1000);

    auto after = VbPoolManager::instance().get_status();
    auto after_pool = find_pool(after, 0);
    EXPECT_EQ(after_pool.free_blocks, before_pool.free_blocks);
}

TEST(VbPoolManagerTest, MultipleBlockAllocation) {
    VbPoolGuard guard(default_configs());

    std::vector<VB_BLK> blocks;
    std::set<uint64_t> addresses;

    for (int i = 0; i < 4; ++i) {
        VB_BLK blk = VbPoolManager::instance().get_block(640 * 640 * 3);
        ASSERT_NE(blk, VB_INVALID_HANDLE);
        blocks.push_back(blk);
        addresses.insert(CVI_VB_Handle2PhysAddr(blk));
    }

    EXPECT_EQ(addresses.size(), 4u);

    VB_BLK overflow = VbPoolManager::instance().get_block(640 * 640 * 3);
    EXPECT_EQ(overflow, VB_INVALID_HANDLE);

    VbPoolManager::instance().release_block(blocks[0]);
    blocks.erase(blocks.begin());

    VB_BLK reused = VbPoolManager::instance().get_block(640 * 640 * 3);
    EXPECT_NE(reused, VB_INVALID_HANDLE);
    blocks.push_back(reused);

    for (auto blk : blocks) {
        VbPoolManager::instance().release_block(blk);
    }
}

TEST(VbMemoryTest, AllocationAndAccess) {
    VbPoolGuard guard(default_configs());

    {
        auto mem = VbMemory::allocate(1024, true);
        ASSERT_NE(mem, nullptr);
        EXPECT_NE(mem->physical_address(), 0u);

        void* ptr = mem->data();
        ASSERT_NE(ptr, nullptr);
        std::memset(ptr, 0xAB, 1024);

        uint8_t* bytes = static_cast<uint8_t*>(ptr);
        EXPECT_EQ(bytes[0], 0xAB);
        EXPECT_EQ(bytes[1023], 0xAB);
    }

    auto status = VbPoolManager::instance().get_status();
    auto pool0 = find_pool(status, 0);
    EXPECT_EQ(pool0.free_blocks, 4u);
}

TEST(VbMemoryTest, CacheOperations) {
    VbPoolGuard guard(default_configs());

    auto mem = VbMemory::allocate(4096, true);
    ASSERT_NE(mem, nullptr);

    uint32_t* data = static_cast<uint32_t*>(mem->data());
    for (int i = 0; i < 1024; ++i) {
        data[i] = static_cast<uint32_t>(i);
    }

    EXPECT_NO_THROW(mem->flush_cache());
    EXPECT_NO_THROW(mem->invalidate_cache());

    for (int i = 0; i < 1024; ++i) {
        EXPECT_EQ(data[i], static_cast<uint32_t>(i));
    }
}

TEST(VbMemoryTest, FromExternalBlock) {
    VbPoolGuard guard(default_configs());

    VB_BLK ext_block = CVI_VB_GetBlock(VB_INVALID_POOLID, 1024);
    ASSERT_NE(ext_block, VB_INVALID_HANDLE);

    {
        auto mem = VbMemory::from_block(ext_block, 1024, true, false);
        ASSERT_NE(mem, nullptr);
        EXPECT_NE(mem->data(), nullptr);
        EXPECT_NE(mem->physical_address(), 0u);
    }

    uint64_t phys = CVI_VB_Handle2PhysAddr(ext_block);
    EXPECT_NE(phys, 0u);

    CVI_VB_ReleaseBlock(ext_block);
}

TEST(VbPoolManagerTest, ProcFileStatus) {
    VbPoolGuard guard(default_configs());

    auto status = VbPoolManager::instance().get_status();
    ASSERT_FALSE(status.empty());

    auto pool0 = find_pool(status, 0);
    EXPECT_GT(pool0.total_blocks, 0u);
    EXPECT_GT(pool0.block_size, 0u);
}

} // namespace tensor

#else

TEST(VbPoolManagerTest, Skipped) {
    GTEST_SKIP() << "USE_CVI_MPI not defined";
}

#endif
