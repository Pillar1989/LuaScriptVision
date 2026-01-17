/**
 * test_frame.cpp - Frame class tests
 *
 * Tests Frame basic functionality:
 * - Construction from cv::Mat
 * - Property access (width, height, channels)
 * - Conversion (to_mat)
 * - Storage type detection
 * - Physical address detection
 * - Move semantics
 */

#include "test_common.h"

void run_frame_tests(TestSuite& suite) {
    std::cout << "\n[Test 1] Frame Basic Functionality" << std::endl;

    Timer timer;

    // Test 1.1: Empty frame
    timer.start();
    {
        Frame frame;
        bool empty_ok = frame.empty();
        suite.add_result("Frame empty constructor", empty_ok, timer.elapsed_ms());
    }

    // Test 1.2: Frame from cv::Mat
    timer.start();
    {
        cv::Mat mat = create_test_image(640, 480);
        Frame frame(mat);

        bool props_ok = (frame.width() == 640 &&
                        frame.height() == 480 &&
                        frame.channels() == 3 &&
                        !frame.empty());

        suite.add_result("Frame from cv::Mat", props_ok, timer.elapsed_ms(),
                        std::to_string(frame.width()) + "x" +
                        std::to_string(frame.height()));
    }

    // Test 1.3: Frame to_mat conversion
    timer.start();
    {
        cv::Mat mat = create_test_image(320, 240);
        Frame frame(mat);

        const cv::Mat& result = frame.to_mat();
        bool convert_ok = (result.cols == 320 && result.rows == 240);

        suite.add_result("Frame to_mat conversion", convert_ok, timer.elapsed_ms());
    }

    // Test 1.4: Frame storage type
    timer.start();
    {
        cv::Mat mat = create_test_image(640, 480);
        Frame frame(mat);

        bool type_ok = (frame.storage_type() == Frame::StorageType::OPENCV);
        suite.add_result("Frame storage type check", type_ok, timer.elapsed_ms());
    }

    // Test 1.5: Frame physical address (should be false for cv::Mat)
    timer.start();
    {
        cv::Mat mat = create_test_image(640, 480);
        Frame frame(mat);

        bool phys_ok = !frame.has_physical_addr();
        suite.add_result("Frame physical address (cv::Mat)", phys_ok, timer.elapsed_ms(),
                        "Should be false for OpenCV Mat");
    }

    // Test 1.6: Frame move semantics
    timer.start();
    {
        cv::Mat mat = create_test_image(640, 480);
        Frame frame1(mat);
        Frame frame2(std::move(frame1));

        bool move_ok = (frame1.empty() && !frame2.empty() &&
                       frame2.width() == 640 && frame2.height() == 480);

        suite.add_result("Frame move constructor", move_ok, timer.elapsed_ms());
    }

    // Test 1.7: Frame clone
    timer.start();
    {
        cv::Mat mat = create_test_image(640, 480);
        Frame frame1(mat);
        Frame frame2 = frame1.clone();

        bool clone_ok = (!frame1.empty() && !frame2.empty() &&
                        frame2.width() == 640 && frame2.height() == 480);

        suite.add_result("Frame clone method", clone_ok, timer.elapsed_ms());
    }
}
