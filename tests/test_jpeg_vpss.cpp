/**
 * test_jpeg_vpss.cpp - JPEG → VDEC → VPSS Pipeline Test
 *
 * Tests the image processing pipeline:
 *   JPEG file → HwJpegDecoder (VDEC) → CviVpssProcessor (VPSS) → Output frame
 *
 * Resource allocation:
 *   - HwJpegDecoder: VDEC_CHN=0
 *   - CviVpssProcessor: VPSS_GRP=1, VPSS_CHN=0
 *
 * Build:
 *   cmake -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-sg200x.cmake -B build
 *   cmake --build build -j16
 *
 * Deploy & Run:
 *   sshpass -p '11' scp build/test_jpeg_vpss recamera@192.168.42.1:/tmp/
 *   sshpass -p '11' ssh recamera@192.168.42.1 'echo "11" | sudo -S /tmp/test_jpeg_vpss /tmp/test.jpg'
 */

#include <iostream>
#include <cstring>
#include <chrono>

#include "modules/cv/hw_jpeg_decoder.h"
#include "modules/cv/cvi_vpss_processor.h"

#ifdef USE_CVI_MPI

#include <cvi_sys.h>
#include <cvi_vb.h>

using namespace lua_cv;

// Test configuration
static constexpr uint32_t TEST_WIDTH = 640;
static constexpr uint32_t TEST_HEIGHT = 640;
static constexpr uint32_t MAX_JPEG_WIDTH = 1920;
static constexpr uint32_t MAX_JPEG_HEIGHT = 1080;

bool test_jpeg_decode(HwJpegDecoder& decoder, const std::string& filepath) {
    std::cout << "\n=== Test: JPEG Decode ===" << std::endl;

    try {
        auto start = std::chrono::high_resolution_clock::now();
        VIDEO_FRAME_INFO_S frame = decoder.decode_file_sync(filepath);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "[PASS] JPEG decode successful" << std::endl;
        std::cout << "  - Decode time: " << duration.count() << " ms" << std::endl;
        std::cout << "  - Width: " << frame.stVFrame.u32Width << std::endl;
        std::cout << "  - Height: " << frame.stVFrame.u32Height << std::endl;
        std::cout << "  - Pixel format: " << frame.stVFrame.enPixelFormat << std::endl;
        std::cout << "  - Stride[0]: " << frame.stVFrame.u32Stride[0] << std::endl;
        std::cout << "  - PhyAddr[0]: 0x" << std::hex << frame.stVFrame.u64PhyAddr[0]
                  << std::dec << std::endl;

        // Release frame
        decoder.release_frame(frame);
        return true;

    } catch (const std::exception& e) {
        std::cerr << "[FAIL] JPEG decode failed: " << e.what() << std::endl;
        return false;
    }
}

bool test_vpss_resize(CviVpssProcessor& vpss, HwJpegDecoder& decoder,
                       const std::string& filepath) {
    std::cout << "\n=== Test: VPSS Resize ===" << std::endl;

    try {
        // Decode JPEG
        VIDEO_FRAME_INFO_S decoded = decoder.decode_file_sync(filepath);
        Frame frame(decoded, false);

        std::cout << "  - Input: " << frame.width() << "x" << frame.height() << std::endl;

        // Resize to TEST_WIDTH x TEST_HEIGHT (in-place)
        auto start = std::chrono::high_resolution_clock::now();
        vpss.resize(frame, TEST_WIDTH, TEST_HEIGHT);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "[PASS] VPSS resize successful" << std::endl;
        std::cout << "  - Resize time: " << duration.count() << " ms" << std::endl;
        std::cout << "  - Output: " << frame.width() << "x" << frame.height() << std::endl;

        // Verify output dimensions
        if (frame.width() != TEST_WIDTH || frame.height() != TEST_HEIGHT) {
            std::cerr << "[FAIL] Output size mismatch! Expected "
                      << TEST_WIDTH << "x" << TEST_HEIGHT << std::endl;
            frame.release();
            decoder.release_frame(decoded);
            return false;
        }

        // Cleanup
        frame.release();
        decoder.release_frame(decoded);
        return true;

    } catch (const std::exception& e) {
        std::cerr << "[FAIL] VPSS resize failed: " << e.what() << std::endl;
        return false;
    }
}

bool test_vpss_cvtcolor(CviVpssProcessor& vpss, HwJpegDecoder& decoder,
                         const std::string& filepath) {
    std::cout << "\n=== Test: VPSS Color Conversion ===" << std::endl;

    try {
        // Decode JPEG (output is YUV_PLANAR_444)
        VIDEO_FRAME_INFO_S decoded = decoder.decode_file_sync(filepath);
        Frame frame(decoded, false);

        std::cout << "  - Input format: " << decoded.stVFrame.enPixelFormat << std::endl;
        std::cout << "  - Input size: " << frame.width() << "x" << frame.height() << std::endl;

        // First resize the frame (VPSS can do color conversion during resize)
        vpss.resize(frame, frame.width(), frame.height());

        std::cout << "[PASS] VPSS resize+format completed (no explicit cvtColor)" << std::endl;

        // Cleanup
        frame.release();
        decoder.release_frame(decoded);
        return true;

    } catch (const std::exception& e) {
        std::cerr << "[FAIL] VPSS processing failed: " << e.what() << std::endl;
        return false;
    }
}

bool test_pipeline_benchmark(CviVpssProcessor& vpss, HwJpegDecoder& decoder,
                              const std::string& filepath, int iterations = 10) {
    std::cout << "\n=== Test: Pipeline Benchmark (" << iterations << " iterations) ===" << std::endl;

    try {
        uint64_t total_decode_us = 0;
        uint64_t total_vpss_us = 0;

        for (int i = 0; i < iterations; i++) {
            // Decode
            auto t1 = std::chrono::high_resolution_clock::now();
            VIDEO_FRAME_INFO_S decoded = decoder.decode_file_sync(filepath);
            auto t2 = std::chrono::high_resolution_clock::now();

            Frame frame(decoded, false);

            // VPSS resize (in-place)
            vpss.resize(frame, TEST_WIDTH, TEST_HEIGHT);
            auto t3 = std::chrono::high_resolution_clock::now();

            total_decode_us += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
            total_vpss_us += std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();

            // Cleanup
            frame.release();
            decoder.release_frame(decoded);
        }

        double avg_decode_ms = total_decode_us / (1000.0 * iterations);
        double avg_vpss_ms = total_vpss_us / (1000.0 * iterations);
        double avg_total_ms = avg_decode_ms + avg_vpss_ms;

        std::cout << "[PASS] Pipeline benchmark completed" << std::endl;
        std::cout << "  - Avg decode time: " << avg_decode_ms << " ms" << std::endl;
        std::cout << "  - Avg VPSS time: " << avg_vpss_ms << " ms" << std::endl;
        std::cout << "  - Avg total time: " << avg_total_ms << " ms" << std::endl;
        std::cout << "  - Throughput: " << (1000.0 / avg_total_ms) << " fps" << std::endl;

        return true;

    } catch (const std::exception& e) {
        std::cerr << "[FAIL] Pipeline benchmark failed: " << e.what() << std::endl;
        return false;
    }
}

int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "  JPEG → VDEC → VPSS Pipeline Test" << std::endl;
    std::cout << "========================================" << std::endl;

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <jpeg_file>" << std::endl;
        return 1;
    }

    std::string filepath = argv[1];
    std::cout << "Input file: " << filepath << std::endl;

    // Initialize system
    CVI_S32 rc = CVI_SYS_Init();
    if (rc != CVI_SUCCESS) {
        std::cerr << "[ERROR] CVI_SYS_Init failed: 0x" << std::hex << rc << std::dec << std::endl;
        return 1;
    }

    rc = CVI_VB_Init();
    if (rc != CVI_SUCCESS && rc != CVI_ERR_VB_NOTREADY) {
        std::cerr << "[ERROR] CVI_VB_Init failed: 0x" << std::hex << rc << std::dec << std::endl;
        CVI_SYS_Exit();
        return 1;
    }

    int passed = 0;
    int failed = 0;

    try {
        // Initialize JPEG decoder
        HwJpegDecoder decoder;
        if (!decoder.init(MAX_JPEG_WIDTH, MAX_JPEG_HEIGHT)) {
            std::cerr << "[ERROR] HwJpegDecoder init failed" << std::endl;
            CVI_VB_Exit();
            CVI_SYS_Exit();
            return 1;
        }
        std::cout << "[OK] HwJpegDecoder initialized" << std::endl;

        // Initialize VPSS processor
        CviVpssProcessor vpss;
        std::cout << "[OK] CviVpssProcessor initialized" << std::endl;

        // Run tests
        if (test_jpeg_decode(decoder, filepath)) passed++; else failed++;
        if (test_vpss_resize(vpss, decoder, filepath)) passed++; else failed++;
        if (test_vpss_cvtcolor(vpss, decoder, filepath)) passed++; else failed++;
        if (test_pipeline_benchmark(vpss, decoder, filepath, 10)) passed++; else failed++;

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Unexpected exception: " << e.what() << std::endl;
        failed++;
    }

    // Cleanup
    CVI_VB_Exit();
    CVI_SYS_Exit();

    std::cout << "\n========================================" << std::endl;
    std::cout << "  Test Results: " << passed << " passed, " << failed << " failed" << std::endl;
    std::cout << "========================================" << std::endl;

    return (failed > 0) ? 1 : 0;
}

#else // !USE_CVI_MPI

int main() {
    std::cerr << "[SKIP] This test requires CVI MPI (SG200X platform)" << std::endl;
    return 0;
}

#endif // USE_CVI_MPI
