/**
 * test_ion_loader.cpp - IonImageLoader performance test
 */

#include "test_common.h"
#include <numeric>
#include <algorithm>

#ifdef USE_CVI_MPI

#include "cv/ion_image_loader.h"
#include "cv/hw_jpeg_decoder.h"
#include <fstream>
#include <vector>

TEST(IonLoaderTest, Breakdown) {
    if (!is_cvi_ready()) {
        GTEST_SKIP() << "CVI system not initialized";
    }

    const std::string& image_path = test_image_path();
    if (image_path.empty()) {
        GTEST_SKIP() << "No test image path provided";
    }

    Timer timer;
    CviVpssProcessor processor;

    cv::Mat test_mat = cv::imread(image_path);
    ASSERT_FALSE(test_mat.empty()) << "Failed to load test image: " << image_path;

    std::ifstream file(image_path, std::ios::binary | std::ios::ate);
    ASSERT_TRUE(file.is_open()) << "Failed to open test file";

    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> file_data(static_cast<size_t>(file_size));
    file.read(reinterpret_cast<char*>(file_data.data()), file_size);
    file.close();

    uint32_t width = test_mat.cols;
    uint32_t height = test_mat.rows;
    uint32_t channels = test_mat.channels();
    size_t raw_size = test_mat.total() * test_mat.elemSize();

    std::cout << "[IonLoader] Test image: " << width << "x" << height
              << " (" << (raw_size / 1024.0 / 1024.0) << " MB raw, "
              << (file_size / 1024.0) << " KB compressed)" << std::endl;

    const int iterations = 10;
    std::vector<double> times;

    times.clear();
    for (int i = 0; i < iterations; ++i) {
        std::ifstream f(image_path, std::ios::binary | std::ios::ate);
        std::streamsize sz = f.tellg();
        f.seekg(0, std::ios::beg);
        std::vector<uint8_t> buf(static_cast<size_t>(sz));

        timer.start();
        f.read(reinterpret_cast<char*>(buf.data()), sz);
        times.push_back(timer.elapsed_ms());
        f.close();
    }
    double file_read_time = *std::min_element(times.begin(), times.end());

    times.clear();
    for (int i = 0; i < iterations; ++i) {
        timer.start();
        cv::Mat decoded = cv::imdecode(
            cv::_InputArray(file_data.data(), static_cast<int>(file_data.size())),
            cv::IMREAD_COLOR);
        (void)decoded;
        times.push_back(timer.elapsed_ms());
    }
    double decode_time = *std::min_element(times.begin(), times.end());

    times.clear();
    for (int i = 0; i < iterations; ++i) {
        timer.start();
        cv::Mat img = cv::imread(image_path);
        (void)img;
        times.push_back(timer.elapsed_ms());
    }
    double imread_time = *std::min_element(times.begin(), times.end());

    times.clear();
    for (int i = 0; i < iterations; ++i) {
        timer.start();
        VB_BLK block = CVI_VB_GetBlock(VB_INVALID_POOLID, raw_size);
        times.push_back(timer.elapsed_ms());
        CVI_VB_ReleaseBlock(block);
    }
    double vb_alloc_time = *std::min_element(times.begin(), times.end());

    double vb_map_time = 0.0;
    {
        timer.start();
        VB_BLK block = CVI_VB_GetBlock(VB_INVALID_POOLID, raw_size);
        CVI_U64 phys = CVI_VB_Handle2PhysAddr(block);
        void* virt = CVI_SYS_MmapCache(phys, raw_size);
        vb_map_time = timer.elapsed_ms();

        CVI_SYS_Munmap(virt, raw_size);
        CVI_VB_ReleaseBlock(block);
    }

    double vb_memcpy_time = 0.0;
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

        CVI_SYS_Munmap(virt, raw_size);
        CVI_VB_ReleaseBlock(block);
    }

    double flush_time = 0.0;
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

        CVI_SYS_Munmap(virt, raw_size);
        CVI_VB_ReleaseBlock(block);
    }

    times.clear();
    for (int i = 0; i < iterations; ++i) {
        timer.start();
        cv::Mat mat = cv::imread(image_path);
        VB_BLK vb_block;
        VIDEO_FRAME_INFO_S video_frame = processor.mat_to_video_frame(mat, vb_block);
        (void)video_frame;
        times.push_back(timer.elapsed_ms());
        CVI_VB_ReleaseBlock(vb_block);
    }
    double current_flow = *std::min_element(times.begin(), times.end());

    times.clear();
    for (int i = 0; i < iterations; ++i) {
        IonImageLoader loader;
        timer.start();
        VIDEO_FRAME_INFO_S video_frame = loader.load(image_path);
        (void)video_frame;
        times.push_back(timer.elapsed_ms());
    }
    double ion_flow = *std::min_element(times.begin(), times.end());

    IonImageLoader prealloc_loader;
    prealloc_loader.preallocate(width, height, channels);

    times.clear();
    for (int i = 0; i < iterations; ++i) {
        timer.start();
        VIDEO_FRAME_INFO_S video_frame = prealloc_loader.load_from_memory_fast(
            file_data.data(), file_data.size());
        (void)video_frame;
        times.push_back(timer.elapsed_ms());
    }
    double fast_flow = *std::min_element(times.begin(), times.end());

    double hw_decode_time = 0.0;
    double hw_decode_sync_time = 0.0;
    bool hw_decode_success = false;
    {
        HwJpegDecoder hw_decoder;
        if (hw_decoder.init(width, height)) {
            try {
                VIDEO_FRAME_INFO_S frame = hw_decoder.decode(file_data.data(), file_data.size());
                hw_decoder.release_frame(frame);

                times.clear();
                for (int i = 0; i < iterations; ++i) {
                    timer.start();
                    VIDEO_FRAME_INFO_S frame = hw_decoder.decode(file_data.data(), file_data.size());
                    times.push_back(timer.elapsed_ms());
                    hw_decoder.release_frame(frame);
                }
                hw_decode_time = *std::min_element(times.begin(), times.end());

                times.clear();
                for (int i = 0; i < iterations; ++i) {
                    timer.start();
                    VIDEO_FRAME_INFO_S frame = hw_decoder.decode_sync(file_data.data(), file_data.size());
                    times.push_back(timer.elapsed_ms());
                    hw_decoder.release_frame(frame);
                }
                hw_decode_sync_time = *std::min_element(times.begin(), times.end());
                hw_decode_success = true;
            } catch (const std::exception& e) {
                std::cerr << "[IonLoader] HW JPEG decode failed: " << e.what() << std::endl;
            }
            hw_decoder.cleanup();
        }
    }

    std::cout << "\n[IonLoader] Timing breakdown" << std::endl;
    std::cout << "  File read:        " << std::fixed << std::setprecision(2)
              << file_read_time << " ms" << std::endl;
    std::cout << "  JPEG decode (SW): " << decode_time << " ms" << std::endl;
    std::cout << "  cv::imread:       " << imread_time << " ms" << std::endl;
    std::cout << "  VB alloc:         " << vb_alloc_time << " ms" << std::endl;
    std::cout << "  VB + MmapCache:   " << vb_map_time << " ms" << std::endl;
    std::cout << "  VB memcpy:        " << vb_memcpy_time << " ms" << std::endl;
    std::cout << "  Cache flush:      " << flush_time << " ms" << std::endl;
    std::cout << "\n[IonLoader] Full pipeline" << std::endl;
    std::cout << "  Current flow:     " << current_flow << " ms" << std::endl;
    std::cout << "  VB flow:          " << ion_flow << " ms" << std::endl;
    std::cout << "  VB fast flow:     " << fast_flow << " ms" << std::endl;
    if (hw_decode_success) {
        std::cout << "  HW JPEG sync:     " << hw_decode_sync_time << " ms" << std::endl;
    }

    EXPECT_GT(file_read_time, 0.0);
    EXPECT_GT(decode_time, 0.0);
    EXPECT_GT(imread_time, 0.0);
    EXPECT_GT(vb_alloc_time, 0.0);
    EXPECT_GT(vb_map_time, 0.0);
    EXPECT_GT(vb_memcpy_time, 0.0);
    EXPECT_GT(flush_time, 0.0);
    EXPECT_GT(current_flow, 0.0);
    EXPECT_GT(ion_flow, 0.0);
    EXPECT_GT(fast_flow, 0.0);
}

#else

TEST(IonLoaderTest, Skipped) {
    GTEST_SKIP() << "USE_CVI_MPI not defined";
}

#endif
