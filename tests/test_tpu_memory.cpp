// Minimal test program for Phase 1: Sophgo TPU Memory Management
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

int main() {
    std::cout << "========== Phase 1: Sophgo TPU Memory Test ==========" << std::endl;

#if defined(USE_CVI_TPU)
    std::cout << "USE_CVI_TPU is defined - TPU support enabled" << std::endl;

    // Test 1: Initialize CVI Runtime
    CVI_RT_HANDLE rt_handle = nullptr;
    CVI_RC rc = CVI_RT_Init(&rt_handle);
    if (rc != CVI_SUCCESS) {
        std::cerr << "ERROR: CVI_RT_Init failed" << std::endl;
        return 1;
    }
    std::cout << "✓ CVI Runtime initialized" << std::endl;

    // Test 2: Allocate Sophgo TPU memory
    const size_t test_size = 1024 * 1024;  // 1MB
    try {
        auto tpu_mem = tensor::CviTpuMemory::allocate(rt_handle, test_size);
        std::cout << "✓ Allocated " << test_size << " bytes Sophgo TPU memory" << std::endl;
        std::cout << "  Physical address: 0x" << std::hex << tpu_mem->physical_addr() << std::dec << std::endl;
        std::cout << "  Size: " << tpu_mem->size_bytes() << " bytes" << std::endl;
        std::cout << "  Alignment: " << tpu_mem->alignment() << " bytes" << std::endl;
        std::cout << "  Owns memory: " << (tpu_mem->owns_memory() ? "yes" : "no") << std::endl;

        // Test 3: Write and read data
        uint8_t* vaddr = static_cast<uint8_t*>(tpu_mem->data());
        if (vaddr) {
            // Write pattern
            for (size_t i = 0; i < 16; ++i) {
                vaddr[i] = static_cast<uint8_t>(i);
            }

            // Flush cache to ensure TPU sees the data
            tpu_mem->flush_cache();
            std::cout << "✓ Written test pattern and flushed cache" << std::endl;

            // Invalidate cache before reading
            tpu_mem->invalidate_cache();

            // Verify pattern
            bool match = true;
            for (size_t i = 0; i < 16; ++i) {
                if (vaddr[i] != static_cast<uint8_t>(i)) {
                    match = false;
                    break;
                }
            }

            if (match) {
                std::cout << "✓ Data verification passed" << std::endl;
            } else {
                std::cerr << "ERROR: Data verification failed" << std::endl;
                return 1;
            }
        } else {
            std::cerr << "ERROR: Virtual address is null" << std::endl;
            return 1;
        }

        // Test 4: Alignment verification
        if ((tpu_mem->physical_addr() & 0x3F) == 0) {
            std::cout << "✓ Physical address is 64-byte aligned" << std::endl;
        } else {
            std::cerr << "ERROR: Physical address not 64-byte aligned" << std::endl;
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "ERROR: Exception during memory operations: " << e.what() << std::endl;
        CVI_RT_DeInit(rt_handle);
        return 1;
    }

    // Test 5: Memory transfer performance
    std::cout << "\n========== Memory Transfer Performance ==========" << std::endl;

    const size_t test_sizes[] = {
        1024,           // 1 KB
        64 * 1024,      // 64 KB
        1024 * 1024,    // 1 MB
        4 * 1024 * 1024 // 4 MB
    };

    for (size_t test_size : test_sizes) {
        try {
            // Allocate buffers
            auto cpu_mem = tensor::CpuMemory::allocate(test_size, 64);
            auto tpu_mem = tensor::CviTpuMemory::allocate(rt_handle, test_size);

            // Initialize CPU buffer with test data
            uint8_t* cpu_data = static_cast<uint8_t*>(cpu_mem->data());
            for (size_t i = 0; i < test_size; ++i) {
                cpu_data[i] = static_cast<uint8_t>(i & 0xFF);
            }

            // Test CPU -> TPU transfer
            const int iterations = (test_size <= 64 * 1024) ? 100 : 10;

            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < iterations; ++i) {
                tpu_mem->copy_from(cpu_mem.get());
            }
            auto end = std::chrono::high_resolution_clock::now();

            double cpu_to_tpu_us = std::chrono::duration<double, std::micro>(end - start).count() / iterations;
            double cpu_to_tpu_bandwidth = (test_size / (1024.0 * 1024.0)) / (cpu_to_tpu_us / 1e6); // MB/s

            // Test TPU -> CPU transfer
            start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < iterations; ++i) {
                tpu_mem->copy_to(cpu_mem.get());
            }
            end = std::chrono::high_resolution_clock::now();

            double tpu_to_cpu_us = std::chrono::duration<double, std::micro>(end - start).count() / iterations;
            double tpu_to_cpu_bandwidth = (test_size / (1024.0 * 1024.0)) / (tpu_to_cpu_us / 1e6); // MB/s

            // Verify data integrity
            bool data_match = true;
            for (size_t i = 0; i < std::min(test_size, size_t(256)); ++i) {
                if (cpu_data[i] != static_cast<uint8_t>(i & 0xFF)) {
                    data_match = false;
                    break;
                }
            }

            std::cout << "\nTest size: ";
            if (test_size >= 1024 * 1024) {
                std::cout << (test_size / (1024 * 1024)) << " MB";
            } else if (test_size >= 1024) {
                std::cout << (test_size / 1024) << " KB";
            } else {
                std::cout << test_size << " B";
            }
            std::cout << " (" << iterations << " iterations)" << std::endl;

            std::cout << "  CPU -> TPU: " << std::fixed << std::setprecision(2)
                      << cpu_to_tpu_us << " us/transfer, "
                      << cpu_to_tpu_bandwidth << " MB/s" << std::endl;

            std::cout << "  TPU -> CPU: " << std::fixed << std::setprecision(2)
                      << tpu_to_cpu_us << " us/transfer, "
                      << tpu_to_cpu_bandwidth << " MB/s" << std::endl;

            std::cout << "  Data integrity: " << (data_match ? "✓ PASS" : "✗ FAIL") << std::endl;

            if (!data_match) {
                std::cerr << "ERROR: Data verification failed for size " << test_size << std::endl;
                CVI_RT_DeInit(rt_handle);
                return 1;
            }

        } catch (const std::exception& e) {
            std::cerr << "ERROR: Performance test failed for size " << test_size
                      << ": " << e.what() << std::endl;
            CVI_RT_DeInit(rt_handle);
            return 1;
        }
    }

    // Cleanup
    CVI_RT_DeInit(rt_handle);
    std::cout << "✓ CVI Runtime deinitialized" << std::endl;

    std::cout << "\n========== All Phase 1 Tests PASSED ==========" << std::endl;
    return 0;

#else
    std::cerr << "ERROR: USE_CVI_TPU not defined - TPU support disabled" << std::endl;
    std::cerr << "Please build with SG200X_SDK_PATH environment variable set" << std::endl;
    return 1;
#endif
}
