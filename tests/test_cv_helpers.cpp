/**
 * test_cv_helpers.cpp - cv_helpers smart backend selection tests
 *
 * Tests automatic backend selection logic:
 * - Backend detection for cv::Mat (should use OpenCV)
 * - Backend detection for VIDEO_FRAME (should use VPSS if available)
 * - Fallback mechanism when hardware is unavailable
 * - Smart resize, cvtColor, crop operations
 */

#include "test_common.h"

void run_cv_helpers_tests(TestSuite& suite) {
    std::cout << "\n[Test 3] cv_helpers Smart Backend Selection" << std::endl;

    Timer timer;

    // Test 3.1: Backend detection for cv::Mat
    timer.start();
    {
        cv::Mat mat = create_test_image(640, 480);
        Frame frame(mat);

        const char* backend = cv_helpers::get_backend_name(frame);
        bool backend_ok = (std::string(backend) == "opencv");

        suite.add_result("cv_helpers backend detection (cv::Mat)", backend_ok,
                        timer.elapsed_ms(), backend);
    }

    // Test 3.2: Smart resize on cv::Mat (should use OpenCV)
    timer.start();
    {
        cv::Mat mat = create_test_image(1920, 1080);
        Frame frame(mat);

        const char* backend = cv_helpers::get_backend_name(frame);
        bool backend_ok = (std::string(backend) == "opencv");

        cv_helpers::resize(frame, 640, 640);
        bool resize_ok = (frame.width() == 640 && frame.height() == 640);

        suite.add_result("cv_helpers::resize (CPU path)", resize_ok && backend_ok,
                        timer.elapsed_ms(), backend);
    }

    // Test 3.3: Smart cvt_color
    timer.start();
    {
        cv::Mat mat = create_test_image(640, 480);
        Frame frame(mat);

        try {
            cv_helpers::cvt_color(frame, ColorConversion::BGR2RGB);
            suite.add_result("cv_helpers::cvt_color", true, timer.elapsed_ms());
        } catch (const std::exception& e) {
            suite.add_result("cv_helpers::cvt_color", false, timer.elapsed_ms(), e.what());
        }
    }

    // Test 3.4: Smart crop
    timer.start();
    {
        cv::Mat mat = create_test_image(640, 480);
        Frame frame(mat);

        try {
            cv_helpers::crop(frame, 100, 100, 320, 240);
            bool crop_ok = (frame.width() == 320 && frame.height() == 240);
            suite.add_result("cv_helpers::crop", crop_ok, timer.elapsed_ms());
        } catch (const std::exception& e) {
            suite.add_result("cv_helpers::crop", false, timer.elapsed_ms(), e.what());
        }
    }

    // Test 3.5: Error handling - empty frame
    timer.start();
    {
        Frame frame;

        bool error_ok = false;
        try {
            cv_helpers::resize(frame, 640, 640);
        } catch (const std::invalid_argument&) {
            error_ok = true;
        }

        suite.add_result("cv_helpers error handling (empty frame)", error_ok,
                        timer.elapsed_ms());
    }

    // Test 3.6: Multiple operations chaining
    timer.start();
    {
        cv::Mat mat = create_test_image(1920, 1080);
        Frame frame(mat);

        try {
            cv_helpers::resize(frame, 640, 640);
            cv_helpers::cvt_color(frame, ColorConversion::BGR2RGB);
            cv_helpers::crop(frame, 50, 50, 540, 540);

            bool chain_ok = (frame.width() == 540 && frame.height() == 540);
            suite.add_result("cv_helpers chained operations", chain_ok,
                           timer.elapsed_ms());
        } catch (const std::exception& e) {
            suite.add_result("cv_helpers chained operations", false, timer.elapsed_ms(), e.what());
        }
    }
}
