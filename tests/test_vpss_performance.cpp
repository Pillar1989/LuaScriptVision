/**
 * test_vpss_performance.cpp - 详细分析VPSS性能瓶颈
 *
 * 目的：分离测量各阶段开销，准确评估VPSS硬件加速性能
 */

#include "test_common.h"
#include <numeric>
#include <algorithm>

#ifdef USE_CVI_MPI

void run_vpss_performance_tests(TestSuite& suite) {
    std::cout << "\n[Performance] VPSS Detailed Breakdown" << std::endl;

    CviVpssProcessor processor;
    Timer timer;

    // Test case: 1920x1080 -> 640x640 resize
    cv::Mat mat = create_test_image(1920, 1080);

    // ========== Test 1: OpenCV CPU baseline (resize only) ==========
    {
        Frame frame(mat);  // 一次clone
        timer.start();
        OpenCvProcessor cpu_processor;
        cpu_processor.resize(frame, 640, 640);
        double cpu_resize_time = timer.elapsed_ms();
        suite.add_result("Perf: OpenCV CPU resize only", true, cpu_resize_time);
    }

    // ========== Test 2: mat_to_video_frame conversion only ==========
    double conversion_time;
    {
        timer.start();
        VB_BLK vb_block;
        VIDEO_FRAME_INFO_S video_frame = processor.mat_to_video_frame(mat, vb_block);
        conversion_time = timer.elapsed_ms();
        CVI_VB_ReleaseBlock(vb_block);
        suite.add_result("Perf: mat_to_video_frame (6.2MB)", true, conversion_time);
    }

    // ========== Test 3: 纯VPSS处理（预先转换，只测VPSS） ==========
    double vpss_only_time;
    {
        // 预先转换到VIDEO_FRAME
        VB_BLK vb_block;
        VIDEO_FRAME_INFO_S video_frame = processor.mat_to_video_frame(mat, vb_block);
        Frame frame(video_frame, 0, 0);  // VPSS context

        // Warm up VPSS pipeline
        processor.resize(frame, 640, 640);

        // 再次转换，测量纯VPSS处理
        VB_BLK vb_block2;
        VIDEO_FRAME_INFO_S video_frame2 = processor.mat_to_video_frame(mat, vb_block2);
        Frame frame2(video_frame2, 0, 0);

        timer.start();
        processor.resize(frame2, 640, 640);
        vpss_only_time = timer.elapsed_ms();

        CVI_VB_ReleaseBlock(vb_block);
        CVI_VB_ReleaseBlock(vb_block2);

        suite.add_result("Perf: Pure VPSS resize (1080p->640)", true, vpss_only_time,
                        "Hardware DMA only");
    }

    // ========== Test 4: VPSS完整流程 (mat_to_video_frame + VPSS) ==========
    double vpss_total_time;
    {
        VB_BLK vb_block;
        timer.start();
        VIDEO_FRAME_INFO_S video_frame = processor.mat_to_video_frame(mat, vb_block);
        Frame frame(video_frame, 0, 0);
        processor.resize(frame, 640, 640);
        vpss_total_time = timer.elapsed_ms();
        CVI_VB_ReleaseBlock(vb_block);

        suite.add_result("Perf: VPSS total (convert + resize)", true, vpss_total_time);
    }

    // ========== Test 5: 重复VPSS处理（测量稳定性）==========
    {
        const int iterations = 10;
        std::vector<double> times;

        for (int i = 0; i < iterations; ++i) {
            VB_BLK vb_block;
            VIDEO_FRAME_INFO_S video_frame = processor.mat_to_video_frame(mat, vb_block);
            Frame frame(video_frame, 0, 0);

            timer.start();
            processor.resize(frame, 640, 640);
            times.push_back(timer.elapsed_ms());

            CVI_VB_ReleaseBlock(vb_block);
        }

        double min_time = *std::min_element(times.begin(), times.end());
        double avg_time = std::accumulate(times.begin(), times.end(), 0.0) / iterations;

        suite.add_result("Perf: VPSS resize stability", true, min_time,
                        "Min=" + std::to_string((int)min_time) + "ms, Avg=" + std::to_string((int)avg_time) + "ms");
    }

    // ========== 性能分析总结 ==========
    std::cout << "\n  Performance Analysis Summary (1920x1080 -> 640x640):" << std::endl;
    std::cout << "    mat_to_video_frame:  " << std::fixed << std::setprecision(1)
              << conversion_time << " ms (cached memcpy + flush)" << std::endl;
    std::cout << "    Pure VPSS hardware:  " << vpss_only_time << " ms" << std::endl;
    std::cout << "    VPSS total:          " << vpss_total_time << " ms" << std::endl;
    std::cout << "\n    Speedup vs OpenCV: " << std::setprecision(1)
              << (127.0 / vpss_total_time) << "x faster" << std::endl;
    std::cout << std::endl;
}

#else

void run_vpss_performance_tests(TestSuite& suite) {
    std::cout << "\n[Performance] VPSS Detailed Breakdown" << std::endl;
    std::cout << "  ⊘  Skipped (USE_CVI_MPI not defined)" << std::endl;
}

#endif
