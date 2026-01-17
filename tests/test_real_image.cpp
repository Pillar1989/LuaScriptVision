/**
 * test_real_image.cpp - Real image processing tests
 *
 * Tests CV module with real-world images:
 * - Load image from file
 * - Apply preprocessing pipeline
 * - Convert to tensor
 * - Verify output validity
 */

#include "test_common.h"

void run_real_image_tests(TestSuite& suite, const std::string& image_path) {
    std::cout << "\n[Test 5] Real Image Processing" << std::endl;

    Timer timer;

    // Test 5.1: Image loading
    timer.start();
    cv::Mat img;
    {
        img = cv::imread(image_path, cv::IMREAD_COLOR);
        bool load_ok = (!img.empty());

        if (!load_ok) {
            suite.add_result("Real image load", false, timer.elapsed_ms(),
                           "Image not found or invalid: " + image_path);
            std::cout << "  ⊘  Skipping real image tests (image not found)" << std::endl;
            return;
        }

        suite.add_result("Real image load", true, timer.elapsed_ms(),
                        std::to_string(img.cols) + "x" + std::to_string(img.rows));
    }

    // Test 5.2: Real image resize
    timer.start();
    {
        Frame frame(img.clone());
        cv_helpers::resize(frame, 640, 640);

        bool resize_ok = (frame.width() == 640 && frame.height() == 640);
        suite.add_result("Real image resize", resize_ok, timer.elapsed_ms());
    }

    // Test 5.3: Real image color conversion
    timer.start();
    {
        Frame frame(img.clone());
        try {
            cv_helpers::cvt_color(frame, ColorConversion::BGR2RGB);
            suite.add_result("Real image cvtColor", true, timer.elapsed_ms());
        } catch (const std::exception& e) {
            suite.add_result("Real image cvtColor", false, timer.elapsed_ms(), e.what());
        }
    }

    // Test 5.4: Real image preprocessing pipeline
    timer.start();
    {
        Frame frame(img.clone());
        try {
            // Typical YOLO preprocessing
            cv_helpers::resize(frame, 640, 640);
            cv_helpers::cvt_color(frame, ColorConversion::BGR2RGB);

            bool pipeline_ok = (frame.width() == 640 && frame.height() == 640);
            suite.add_result("Real image preprocessing pipeline", pipeline_ok,
                           timer.elapsed_ms());
        } catch (const std::exception& e) {
            suite.add_result("Real image preprocessing pipeline", false,
                           timer.elapsed_ms(), e.what());
        }
    }

    // Test 5.5: Real image to tensor conversion
    timer.start();
    {
        Frame frame(img.clone());
        try {
            cv_helpers::resize(frame, 640, 640);
            cv_helpers::cvt_color(frame, ColorConversion::BGR2RGB);

            std::vector<double> mean = {0.0, 0.0, 0.0};
            std::vector<double> std = {1.0, 1.0, 1.0};
            auto tensor = cv_helpers::frame_to_tensor(frame, 1.0/255.0, mean, std);

            bool tensor_ok = (tensor.ndim() == 4 &&
                            tensor.size(0) == 1 &&
                            tensor.size(1) == 3 &&
                            tensor.size(2) == 640 &&
                            tensor.size(3) == 640);

            suite.add_result("Real image to tensor", tensor_ok, timer.elapsed_ms(),
                           tensor_ok ? "Shape: [1,3,640,640]" : "Invalid shape");
        } catch (const std::exception& e) {
            suite.add_result("Real image to tensor", false, timer.elapsed_ms(), e.what());
        }
    }

    // Test 5.6: Real image crop region
    timer.start();
    {
        Frame frame(img.clone());
        try {
            // Crop center region
            int crop_w = std::min(img.cols, 320);
            int crop_h = std::min(img.rows, 240);
            int x = (img.cols - crop_w) / 2;
            int y = (img.rows - crop_h) / 2;

            cv_helpers::crop(frame, x, y, crop_w, crop_h);

            bool crop_ok = (frame.width() == crop_w && frame.height() == crop_h);
            suite.add_result("Real image crop", crop_ok, timer.elapsed_ms());
        } catch (const std::exception& e) {
            suite.add_result("Real image crop", false, timer.elapsed_ms(), e.what());
        }
    }

    // Test 5.7: Real image full inference preparation
    timer.start();
    {
        Frame frame(img.clone());
        try {
            // Complete preprocessing for model inference
            cv_helpers::resize(frame, 640, 640);
            cv_helpers::cvt_color(frame, ColorConversion::BGR2RGB);

            std::vector<double> mean = {0.485, 0.456, 0.406};
            std::vector<double> std = {0.229, 0.224, 0.225};
            auto tensor = cv_helpers::frame_to_tensor(frame, 1.0/255.0, mean, std);

            // frame_to_tensor already returns [1,3,640,640], no need to unsqueeze
            bool inference_ready = (tensor.ndim() == 4 &&
                                  tensor.size(0) == 1 &&
                                  tensor.size(1) == 3 &&
                                  tensor.size(2) == 640 &&
                                  tensor.size(3) == 640);

            suite.add_result("Real image inference preparation", inference_ready,
                           timer.elapsed_ms(),
                           inference_ready ? "Shape: [1,3,640,640]" : "Invalid shape");
        } catch (const std::exception& e) {
            suite.add_result("Real image inference preparation", false,
                           timer.elapsed_ms(), e.what());
        }
    }
}
