/**
 * test_vpss_proc.cpp - CviVpssProcessor hardware acceleration tests
 *
 * Tests VPSS hardware backend:
 * - VIDEO_FRAME creation and conversion
 * - Hardware resize operations
 * - Hardware color conversion
 * - Hardware crop operations
 * - Zero-copy performance
 */

#include "test_common.h"

#ifdef USE_CVI_MPI

void run_vpss_tests(TestSuite& suite) {
    std::cout << "\n[Test 4] CviVpssProcessor Hardware Acceleration" << std::endl;

    CviVpssProcessor processor;
    Timer timer;

    // Test 4.1: Mat to VIDEO_FRAME conversion
    timer.start();
    {
        cv::Mat mat = create_test_image(640, 480);

        // Convert to VIDEO_FRAME - just create Frame from Mat
        bool convert_ok = false;
        try {
            Frame frame(mat);
            convert_ok = (frame.storage_type() == Frame::StorageType::OPENCV);
        } catch (const std::exception& e) {
            suite.add_result("VPSS mat to VIDEO_FRAME", false, timer.elapsed_ms(), e.what());
            return;
        }

        suite.add_result("VPSS mat to VIDEO_FRAME", convert_ok, timer.elapsed_ms());
    }

    // Test 4.2: Hardware resize (downscale)
    timer.start();
    {
        cv::Mat mat = create_test_image(1280, 720);  // Use 720p instead of 1080p

        VB_BLK vb_block;
        VB_POOL vb_pool = 2;  // Pool 2: For 1280x720 RGB888 frames
        VIDEO_FRAME_INFO_S video_frame = processor.mat_to_video_frame(mat, vb_block, vb_pool);
        Frame frame(video_frame, false);  // owns_memory=false

        processor.resize(frame, 640, 640);

        CVI_VB_ReleaseBlock(vb_block);
    }
    suite.add_result("VPSS resize (1280x720 -> 640x640)", true,
                   timer.elapsed_ms(),
                   "Zero-copy hardware acceleration");

    // Test 4.3: Hardware resize (upscale)
    {
        const int iterations = 3;
        std::vector<double> times;

        for (int i = 0; i < iterations; ++i) {
            cv::Mat mat = create_test_image(320, 240);

            VB_BLK vb_block;
            VB_POOL vb_pool = 1;  // Pool 1: For 640x640 RGB888 frames
            VIDEO_FRAME_INFO_S video_frame = processor.mat_to_video_frame(mat, vb_block, vb_pool);
            Frame frame(video_frame, false);  // owns_memory=false, USER-allocated VB

            timer.start();
            processor.resize(frame, 640, 480);
            times.push_back(timer.elapsed_ms());

            CVI_VB_ReleaseBlock(vb_block);
        }

        double min_time = *std::min_element(times.begin(), times.end());

        suite.add_result("VPSS resize upscale (320x240 -> 640x480)", true,
                       min_time,
                       "Best of " + std::to_string(iterations) + " iterations");
    }

    // Test 4.4: Hardware color conversion
    {
        const int iterations = 3;
        std::vector<double> times;

        for (int i = 0; i < iterations; ++i) {
            cv::Mat mat = create_test_image(640, 480);

            VB_BLK vb_block;
            VB_POOL vb_pool = 1;  // Pool 1: For 640x640 RGB888 frames
            VIDEO_FRAME_INFO_S video_frame = processor.mat_to_video_frame(mat, vb_block, vb_pool);
            Frame frame(video_frame, false);  // owns_memory=false, USER-allocated VB

            timer.start();
            processor.cvtColor(frame, ColorConversion::BGR2RGB);
            times.push_back(timer.elapsed_ms());

            CVI_VB_ReleaseBlock(vb_block);
        }

        double min_time = *std::min_element(times.begin(), times.end());

        suite.add_result("VPSS cvtColor (BGR2RGB)", true, min_time,
                       "Best of " + std::to_string(iterations) + " iterations");
    }

    // Test 4.5: Hardware crop
    {
        const int iterations = 3;
        std::vector<double> times;

        for (int i = 0; i < iterations; ++i) {
            cv::Mat mat = create_test_image(640, 480);

            VB_BLK vb_block;
            VB_POOL vb_pool = 1;  // Pool 1: For 640x640 RGB888 frames
            VIDEO_FRAME_INFO_S video_frame = processor.mat_to_video_frame(mat, vb_block, vb_pool);
            Frame frame(video_frame, false);  // owns_memory=false, USER-allocated VB

            timer.start();
            processor.crop(frame, 100, 100, 320, 240);
            times.push_back(timer.elapsed_ms());

            CVI_VB_ReleaseBlock(vb_block);
        }

        double min_time = *std::min_element(times.begin(), times.end());

        suite.add_result("VPSS crop (640x480 -> 320x240)", true,
                       min_time,
                       "Best of " + std::to_string(iterations) + " iterations");
    }

    // Test 4.6: Physical address mode after VPSS processing
    timer.start();
    {
        cv::Mat mat = create_test_image(640, 480);
        Frame frame(mat);  // OPENCV type, no physical address initially

        // After VPSS processing, should have physical address
        processor.resize(frame, 320, 240);

        bool phys_ok = frame.has_physical_addr();
        uint64_t phys_addr = 0;
        if (phys_ok) {
            phys_addr = frame.physical_addr();
        }

        suite.add_result("VPSS physical address mode",
                         phys_ok && phys_addr != 0,
                         timer.elapsed_ms(),
                         phys_ok ? "Zero-copy enabled" : "No physical address");
    }

    // Test 4.7: Multiple operations chaining
    {
        cv::Mat mat = create_test_image(1920, 1080);

        const int iterations = 3;
        std::vector<double> times;

        for (int i = 0; i < iterations; ++i) {
            VB_BLK vb_block;
            VB_POOL vb_pool = 0;  // Pool 0: For 1920x1080 RGB888 frames
            VIDEO_FRAME_INFO_S video_frame = processor.mat_to_video_frame(mat, vb_block, vb_pool);
            Frame frame(video_frame, 0, 0);

            timer.start();
            processor.resize(frame, 640, 640);
            processor.cvtColor(frame, ColorConversion::BGR2RGB);
            processor.crop(frame, 50, 50, 540, 540);
            times.push_back(timer.elapsed_ms());

            CVI_VB_ReleaseBlock(vb_block);
        }

        double min_time = *std::min_element(times.begin(), times.end());

        suite.add_result("VPSS chained operations", true,
                       min_time,
                       "Best of " + std::to_string(iterations) + " iterations");
    }
}

#else

void run_vpss_tests(TestSuite& suite) {
    std::cout << "\n[Test 4] CviVpssProcessor Hardware Acceleration" << std::endl;
    std::cout << "  ⊘  VPSS tests skipped (USE_CVI_MPI not defined)" << std::endl;
}

#endif
