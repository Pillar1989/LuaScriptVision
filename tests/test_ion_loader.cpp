/**
 * test_ion_loader.cpp - IonImageLoader performance test
 *
 * Detailed timing breakdown to identify the real bottleneck:
 * 1. File I/O time
 * 2. JPEG decode time
 * 3. VB pool allocation time
 * 4. Memcpy time (to cached VB)
 * 5. Cache flush time
 */

#include "test_common.h"
#include <numeric>
#include <algorithm>

#ifdef USE_CVI_MPI

#include "cv/ion_image_loader.h"
#include "cv/hw_jpeg_decoder.h"
#include <fstream>
#include <vector>

void run_ion_loader_tests(TestSuite& suite, const std::string& image_path) {
    std::cout << "\n[Test] IonImageLoader Performance - Detailed Breakdown" << std::endl;

    if (image_path.empty()) {
        std::cout << "  ⊘  Skipped (no image provided)" << std::endl;
        return;
    }

    Timer timer;
    CviVpssProcessor processor;

    // ========== Step 0: Load test image and read file to memory ==========
    cv::Mat test_mat = cv::imread(image_path);
    if (test_mat.empty()) {
        std::cout << "  ✗  Failed to load test image: " << image_path << std::endl;
        return;
    }

    std::ifstream file(image_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cout << "  ✗  Failed to open test file" << std::endl;
        return;
    }
    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> file_data(file_size);
    file.read(reinterpret_cast<char*>(file_data.data()), file_size);
    file.close();

    uint32_t width = test_mat.cols;
    uint32_t height = test_mat.rows;
    uint32_t channels = test_mat.channels();
    size_t raw_size = test_mat.total() * test_mat.elemSize();

    std::cout << "  Test image: " << width << "x" << height
              << " (" << (raw_size / 1024.0 / 1024.0) << " MB raw, "
              << (file_size / 1024.0) << " KB compressed)" << std::endl;

    const int iterations = 10;
    std::vector<double> times;

    // ========== Test 1: File read only ==========
    times.clear();
    for (int i = 0; i < iterations; ++i) {
        std::ifstream f(image_path, std::ios::binary | std::ios::ate);
        std::streamsize sz = f.tellg();
        f.seekg(0, std::ios::beg);
        std::vector<uint8_t> buf(sz);

        timer.start();
        f.read(reinterpret_cast<char*>(buf.data()), sz);
        times.push_back(timer.elapsed_ms());
        f.close();
    }
    double file_read_time = *std::min_element(times.begin(), times.end());
    suite.add_result("Breakdown: File read", true, file_read_time,
                    std::to_string(file_size/1024) + " KB");

    // ========== Test 2: JPEG decode only (cv::imdecode) ==========
    times.clear();
    for (int i = 0; i < iterations; ++i) {
        timer.start();
        cv::Mat decoded = cv::imdecode(
            cv::_InputArray(file_data.data(), static_cast<int>(file_data.size())),
            cv::IMREAD_COLOR);
        times.push_back(timer.elapsed_ms());
    }
    double decode_time = *std::min_element(times.begin(), times.end());
    suite.add_result("Breakdown: JPEG decode (imdecode)", true, decode_time);

    // ========== Test 3: cv::imread (file read + decode combined) ==========
    times.clear();
    for (int i = 0; i < iterations; ++i) {
        timer.start();
        cv::Mat img = cv::imread(image_path);
        times.push_back(timer.elapsed_ms());
    }
    double imread_time = *std::min_element(times.begin(), times.end());
    suite.add_result("Breakdown: cv::imread (file+decode)", true, imread_time);

    // ========== Test 4: VB block allocation only ==========
    times.clear();
    for (int i = 0; i < iterations; ++i) {
        timer.start();
        VB_BLK block = CVI_VB_GetBlock(VB_INVALID_POOLID, raw_size);
        times.push_back(timer.elapsed_ms());
        CVI_VB_ReleaseBlock(block);
    }
    double vb_alloc_time = *std::min_element(times.begin(), times.end());
    suite.add_result("Breakdown: VB block alloc", true, vb_alloc_time);

    // ========== Test 5: VB + Mmap (cached) ==========
    double vb_map_time;
    {
        timer.start();
        VB_BLK block = CVI_VB_GetBlock(VB_INVALID_POOLID, raw_size);
        CVI_U64 phys = CVI_VB_Handle2PhysAddr(block);
        void* virt = CVI_SYS_MmapCache(phys, raw_size);
        vb_map_time = timer.elapsed_ms();

        CVI_SYS_Munmap(virt, raw_size);
        CVI_VB_ReleaseBlock(block);
    }
    suite.add_result("Breakdown: VB alloc + MmapCache", true, vb_map_time);

    // ========== Test 6: Memcpy to VB (cached) ==========
    double vb_memcpy_time;
    {
        VB_BLK block = CVI_VB_GetBlock(VB_INVALID_POOLID, raw_size);
        CVI_U64 phys = CVI_VB_Handle2PhysAddr(block);
        void* virt = CVI_SYS_MmapCache(phys, raw_size);

        times.clear();
        for (int i = 0; i < iterations; ++i) {
            timer.start();
            std::memcpy(virt, test_mat.data, raw_size);
            times.push_back(timer.elapsed_ms());
        }
        vb_memcpy_time = *std::min_element(times.begin(), times.end());
        suite.add_result("Breakdown: memcpy to VB (cached)", true, vb_memcpy_time,
                        std::to_string(raw_size/1024/1024) + " MB");

        CVI_SYS_Munmap(virt, raw_size);
        CVI_VB_ReleaseBlock(block);
    }

    // ========== Test 7: Cache flush only (VB) ==========
    double flush_time;
    {
        VB_BLK block = CVI_VB_GetBlock(VB_INVALID_POOLID, raw_size);
        CVI_U64 phys = CVI_VB_Handle2PhysAddr(block);
        void* virt = CVI_SYS_MmapCache(phys, raw_size);
        std::memcpy(virt, test_mat.data, raw_size);

        times.clear();
        for (int i = 0; i < iterations; ++i) {
            timer.start();
            CVI_SYS_IonFlushCache(phys, virt, raw_size);
            times.push_back(timer.elapsed_ms());
        }
        flush_time = *std::min_element(times.begin(), times.end());
        suite.add_result("Breakdown: Cache flush (VB)", true, flush_time);

        CVI_SYS_Munmap(virt, raw_size);
        CVI_VB_ReleaseBlock(block);
    }

    // ========== Test 8: Full pipeline comparison ==========
    std::cout << "\n  Full Pipeline Comparison:" << std::endl;

    // Current flow: imread + mat_to_video_frame
    times.clear();
    for (int i = 0; i < iterations; ++i) {
        timer.start();
        cv::Mat mat = cv::imread(image_path);
        VB_BLK vb_block;
        VIDEO_FRAME_INFO_S video_frame = processor.mat_to_video_frame(mat, vb_block);
        times.push_back(timer.elapsed_ms());
        CVI_VB_ReleaseBlock(vb_block);
    }
    double current_flow = *std::min_element(times.begin(), times.end());
    suite.add_result("Pipeline: Current (imread + VB)", true, current_flow);

    // Ion flow: IonImageLoader.load()
    times.clear();
    for (int i = 0; i < iterations; ++i) {
        IonImageLoader loader;
        timer.start();
        VIDEO_FRAME_INFO_S video_frame = loader.load(image_path);
        times.push_back(timer.elapsed_ms());
    }
    double ion_flow = *std::min_element(times.begin(), times.end());
    suite.add_result("Pipeline: VB (IonImageLoader.load)", true, ion_flow);

    // Optimized flow: file already in memory + preallocated VB
    IonImageLoader prealloc_loader;
    prealloc_loader.preallocate(width, height, channels);

    times.clear();
    for (int i = 0; i < iterations; ++i) {
        timer.start();
        VIDEO_FRAME_INFO_S video_frame = prealloc_loader.load_from_memory_fast(
            file_data.data(), file_data.size());
        times.push_back(timer.elapsed_ms());
    }
    double fast_flow = *std::min_element(times.begin(), times.end());
    suite.add_result("Pipeline: VB fast (preallocated)", true, fast_flow);

    // ========== Test 9: Hardware JPEG Decoder ==========
    double hw_decode_time = 0;
    double hw_decode_sync_time = 0;
    bool hw_decode_success = false;
    {
        HwJpegDecoder hw_decoder;
        if (hw_decoder.init(width, height)) {
            // Warm up with async version
            try {
                VIDEO_FRAME_INFO_S frame = hw_decoder.decode(file_data.data(), file_data.size());
                hw_decoder.release_frame(frame);

                // Measure async performance
                times.clear();
                for (int i = 0; i < iterations; ++i) {
                    timer.start();
                    VIDEO_FRAME_INFO_S frame = hw_decoder.decode(file_data.data(), file_data.size());
                    times.push_back(timer.elapsed_ms());
                    hw_decoder.release_frame(frame);
                }
                hw_decode_time = *std::min_element(times.begin(), times.end());
                suite.add_result("Pipeline: HW JPEG decode (VDEC async)", true, hw_decode_time,
                                "Output: NV21");

                // Measure sync performance (blocking mode, no 50ms delay)
                times.clear();
                for (int i = 0; i < iterations; ++i) {
                    timer.start();
                    VIDEO_FRAME_INFO_S frame = hw_decoder.decode_sync(file_data.data(), file_data.size());
                    times.push_back(timer.elapsed_ms());
                    hw_decoder.release_frame(frame);
                }
                hw_decode_sync_time = *std::min_element(times.begin(), times.end());
                hw_decode_success = true;
                suite.add_result("Pipeline: HW JPEG decode (VDEC sync)", true, hw_decode_sync_time,
                                "Output: NV21, blocking mode");
            } catch (const std::exception& e) {
                suite.add_result("Pipeline: HW JPEG decode (VDEC)", false, 0,
                                std::string("Error: ") + e.what());
            }
            hw_decoder.cleanup();
        } else {
            suite.add_result("Pipeline: HW JPEG decode (VDEC)", false, 0,
                            "Init failed");
        }
    }

    // ========== Summary ==========
    std::cout << "\n  ========== Timing Breakdown Summary ==========" << std::endl;
    std::cout << "    File read:        " << std::fixed << std::setprecision(2)
              << file_read_time << " ms" << std::endl;
    std::cout << "    JPEG decode (SW): " << decode_time << " ms (cv::imdecode)" << std::endl;
    if (hw_decode_success) {
        std::cout << "    JPEG decode (HW async): " << hw_decode_time << " ms (VDEC with threads)" << std::endl;
        std::cout << "    JPEG decode (HW sync):  " << hw_decode_sync_time << " ms (VDEC blocking) - "
                  << std::setprecision(1) << (hw_decode_time / hw_decode_sync_time) << "x faster than async!" << std::endl;
        std::cout << "    SW vs HW sync speedup:  " << (decode_time / hw_decode_sync_time) << "x" << std::endl;
    }
    std::cout << "    VB alloc:         " << vb_alloc_time << " ms" << std::endl;
    std::cout << "    VB + MmapCache:   " << vb_map_time << " ms" << std::endl;
    std::cout << "    VB memcpy:        " << vb_memcpy_time << " ms (cached)" << std::endl;
    std::cout << "    Cache flush:      " << flush_time << " ms" << std::endl;
    std::cout << std::endl;
    std::cout << "    -------- Full Pipeline Comparison --------" << std::endl;
    std::cout << "    Current flow:     " << current_flow << " ms (imread + VB memcpy)" << std::endl;
    std::cout << "    VB flow:          " << ion_flow << " ms ("
              << std::setprecision(1) << ((current_flow - ion_flow) / current_flow * 100)
              << "% faster)" << std::endl;
    std::cout << "    VB fast flow:     " << fast_flow << " ms ("
              << ((current_flow - fast_flow) / current_flow * 100)
              << "% faster)" << std::endl;
    if (hw_decode_success) {
        std::cout << "    HW JPEG flow (sync): " << (file_read_time + hw_decode_sync_time) << " ms ("
                  << ((current_flow - (file_read_time + hw_decode_sync_time)) / current_flow * 100)
                  << "% faster) - BEST!" << std::endl;
    }
    std::cout << "  ===============================================" << std::endl;
}

#else

void run_ion_loader_tests(TestSuite& suite, const std::string& image_path) {
    std::cout << "\n[Test] IonImageLoader Performance" << std::endl;
    std::cout << "  ⊘  Skipped (USE_CVI_MPI not defined)" << std::endl;
}

#endif
