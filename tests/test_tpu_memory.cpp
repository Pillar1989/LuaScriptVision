// TPU memory tests using gtest

#include <iostream>
#include <cstdlib>
#include <chrono>
#include <iomanip>
#include "modules/tensor/cvi_tpu_memory.h"
#include "modules/tensor/cpu_memory.h"

#if defined(USE_CVI_TPU)
#include <cviruntime.h>
#include <cviruntime_context.h>
#endif

#include <gtest/gtest.h>

#if defined(USE_CVI_TPU)

class TpuMemoryTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        CVI_RC rc = CVI_RT_Init(&rt_handle_);
        if (rc != CVI_SUCCESS) {
            rt_ready_ = false;
            return;
        }
        rt_ready_ = true;
    }

    static void TearDownTestSuite() {
        if (rt_ready_) {
            CVI_RT_DeInit(rt_handle_);
            rt_ready_ = false;
        }
    }

    static void RequireRtReady() {
        if (!rt_ready_) {
            GTEST_SKIP() << "CVI_RT_Init failed";
        }
    }

    static CVI_RT_HANDLE rt_handle_;
    static bool rt_ready_;
};

CVI_RT_HANDLE TpuMemoryTest::rt_handle_ = nullptr;
bool TpuMemoryTest::rt_ready_ = false;

TEST_F(TpuMemoryTest, AllocateAndVerify) {
    RequireRtReady();

    const size_t test_size = 1024 * 1024;
    auto tpu_mem = tensor::CviTpuMemory::allocate(rt_handle_, test_size);
    ASSERT_NE(tpu_mem, nullptr);

    uint8_t* vaddr = static_cast<uint8_t*>(tpu_mem->data());
    ASSERT_NE(vaddr, nullptr);

    for (size_t i = 0; i < 16; ++i) {
        vaddr[i] = static_cast<uint8_t>(i);
    }

    tpu_mem->flush_cache();
    tpu_mem->invalidate_cache();

    for (size_t i = 0; i < 16; ++i) {
        EXPECT_EQ(vaddr[i], static_cast<uint8_t>(i));
    }
}

TEST_F(TpuMemoryTest, Alignment) {
    RequireRtReady();

    const size_t test_size = 1024 * 1024;
    auto tpu_mem = tensor::CviTpuMemory::allocate(rt_handle_, test_size);
    ASSERT_NE(tpu_mem, nullptr);

    EXPECT_EQ(tpu_mem->physical_addr() & 0x3F, 0u);
}

TEST_F(TpuMemoryTest, TransferPerformance) {
    RequireRtReady();

    const size_t test_sizes[] = {
        1024,
        64 * 1024,
        1024 * 1024,
        4 * 1024 * 1024
    };

    for (size_t test_size : test_sizes) {
        auto cpu_mem = tensor::CpuMemory::allocate(test_size, 64);
        auto tpu_mem = tensor::CviTpuMemory::allocate(rt_handle_, test_size);
        ASSERT_NE(cpu_mem, nullptr);
        ASSERT_NE(tpu_mem, nullptr);

        uint8_t* cpu_data = static_cast<uint8_t*>(cpu_mem->data());
        for (size_t i = 0; i < test_size; ++i) {
            cpu_data[i] = static_cast<uint8_t>(i & 0xFF);
        }

        const int iterations = (test_size <= 64 * 1024) ? 100 : 10;

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            tpu_mem->copy_from(cpu_mem.get());
        }
        auto end = std::chrono::high_resolution_clock::now();

        double cpu_to_tpu_us = std::chrono::duration<double, std::micro>(end - start).count() / iterations;
        double cpu_to_tpu_bandwidth = (test_size / (1024.0 * 1024.0)) / (cpu_to_tpu_us / 1e6);

        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            tpu_mem->copy_to(cpu_mem.get());
        }
        end = std::chrono::high_resolution_clock::now();

        double tpu_to_cpu_us = std::chrono::duration<double, std::micro>(end - start).count() / iterations;
        double tpu_to_cpu_bandwidth = (test_size / (1024.0 * 1024.0)) / (tpu_to_cpu_us / 1e6);

        bool data_match = true;
        for (size_t i = 0; i < std::min(test_size, size_t(256)); ++i) {
            if (cpu_data[i] != static_cast<uint8_t>(i & 0xFF)) {
                data_match = false;
                break;
            }
        }

        std::cout << "\n[TPU Mem] size=" << test_size
                  << " cpu->tpu=" << std::fixed << std::setprecision(2)
                  << cpu_to_tpu_us << " us, " << cpu_to_tpu_bandwidth << " MB/s"
                  << " tpu->cpu=" << tpu_to_cpu_us << " us, "
                  << tpu_to_cpu_bandwidth << " MB/s" << std::endl;

        EXPECT_TRUE(data_match);
        EXPECT_GT(cpu_to_tpu_us, 0.0);
        EXPECT_GT(tpu_to_cpu_us, 0.0);
    }
}

#else

TEST(TpuMemoryTest, Skipped) {
    GTEST_SKIP() << "USE_CVI_TPU not defined";
}

#endif
