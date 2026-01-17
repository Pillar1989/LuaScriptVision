/**
 * test_opencv_proc.cpp - OpenCvProcessor tests
 *
 * Tests OpenCV CPU-based processing:
 * - Resize (downscale and upscale)
 * - Color conversion (BGR2RGB, etc.)
 * - Crop operations
 * - Error handling
 */

#include "test_common.h"

void run_opencv_tests(TestSuite& suite) {
    std::cout << "\n[Test 2] OpenCvProcessor Functionality" << std::endl;

    OpenCvProcessor processor;
    Timer timer;

    // Test 2.1: Resize downscale
    timer.start();
    {
        cv::Mat mat = create_test_image(1920, 1080);
        Frame frame(mat);

        processor.resize(frame, 640, 640);

        bool resize_ok = (frame.width() == 640 && frame.height() == 640);
        suite.add_result("OpenCV resize (1920x1080 -> 640x640)", resize_ok,
                        timer.elapsed_ms());
    }

    // Test 2.2: Resize upscale
    timer.start();
    {
        cv::Mat mat = create_test_image(320, 240);
        Frame frame(mat);

        processor.resize(frame, 640, 480);

        bool resize_ok = (frame.width() == 640 && frame.height() == 480);
        suite.add_result("OpenCV resize upscale (320x240 -> 640x480)", resize_ok,
                        timer.elapsed_ms());
    }

    // Test 2.3: Color conversion BGR2RGB
    timer.start();
    {
        cv::Mat mat = create_test_image(640, 480);
        Frame frame(mat);

        try {
            processor.cvtColor(frame, ColorConversion::BGR2RGB);
            suite.add_result("OpenCV cvtColor (BGR2RGB)", true, timer.elapsed_ms());
        } catch (const std::exception& e) {
            suite.add_result("OpenCV cvtColor (BGR2RGB)", false, timer.elapsed_ms(), e.what());
        }
    }

    // Test 2.4: Crop
    timer.start();
    {
        cv::Mat mat = create_test_image(640, 480);
        Frame frame(mat);

        try {
            processor.crop(frame, 100, 100, 320, 240);
            bool crop_ok = (frame.width() == 320 && frame.height() == 240);
            suite.add_result("OpenCV crop (640x480 -> 320x240)", crop_ok,
                           timer.elapsed_ms());
        } catch (const std::exception& e) {
            suite.add_result("OpenCV crop", false, timer.elapsed_ms(), e.what());
        }
    }

    // Test 2.5: Error handling - invalid dimensions
    timer.start();
    {
        cv::Mat mat = create_test_image(640, 480);
        Frame frame(mat);

        bool error_ok = false;
        try {
            processor.resize(frame, -100, -100);
        } catch (const std::invalid_argument&) {
            error_ok = true;
        }

        suite.add_result("OpenCV error handling (invalid dims)", error_ok,
                        timer.elapsed_ms());
    }

    // Test 2.6: Error handling - invalid crop region
    timer.start();
    {
        cv::Mat mat = create_test_image(640, 480);
        Frame frame(mat);

        bool error_ok = false;
        try {
            processor.crop(frame, 500, 500, 500, 500);  // Exceeds bounds
        } catch (const std::invalid_argument&) {
            error_ok = true;
        }

        suite.add_result("OpenCV error handling (invalid crop)", error_ok,
                        timer.elapsed_ms());
    }
}
