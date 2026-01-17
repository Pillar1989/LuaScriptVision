/**
 * test_benchmark.cpp - Performance benchmarks for CV operations
 *
 * Compares CPU vs VPSS performance:
 * - Resize throughput
 * - Color conversion throughput
 * - Crop throughput
 * - End-to-end pipeline performance
 */

#include "test_common.h"

void run_benchmark_tests(TestSuite& suite) {
    std::cout << "\n[Benchmark] Performance Comparison" << std::endl;

    const int iterations = 10;

    // Benchmark 1: CPU resize (1920x1080 -> 640x640)
    {
        auto result = run_benchmark("CPU Resize 1920x1080->640x640", [&]() {
            cv::Mat mat = create_test_image(1920, 1080);
            Frame frame(mat);
            OpenCvProcessor processor;
            processor.resize(frame, 640, 640);
        }, iterations);

        print_benchmark_result(result);
        suite.add_result("Benchmark: CPU Resize", true, result.avg_ms);
    }

#ifdef USE_CVI_MPI
    // Benchmark 2: VPSS resize (1920x1080 -> 640x640)
    {
        auto result = run_benchmark("VPSS Resize 1920x1080->640x640", [&]() {
            cv::Mat mat = create_test_image(1920, 1080);
            Frame frame(mat);
            CviVpssProcessor processor;
            processor.resize(frame, 640, 640);
        }, iterations);

        print_benchmark_result(result);
        suite.add_result("Benchmark: VPSS Resize", true, result.avg_ms);
    }
#endif

    // Benchmark 3: CPU color conversion
    {
        auto result = run_benchmark("CPU cvtColor BGR2RGB", [&]() {
            cv::Mat mat = create_test_image(640, 480);
            Frame frame(mat);
            OpenCvProcessor processor;
            processor.cvtColor(frame, ColorConversion::BGR2RGB);
        }, iterations);

        print_benchmark_result(result);
        suite.add_result("Benchmark: CPU cvtColor", true, result.avg_ms);
    }

#ifdef USE_CVI_MPI
    // Benchmark 4: VPSS color conversion
    {
        auto result = run_benchmark("VPSS cvtColor BGR2RGB", [&]() {
            cv::Mat mat = create_test_image(640, 480);
            Frame frame(mat);
            CviVpssProcessor processor;
            processor.cvtColor(frame, ColorConversion::BGR2RGB);
        }, iterations);

        print_benchmark_result(result);
        suite.add_result("Benchmark: VPSS cvtColor", true, result.avg_ms);
    }
#endif

    // Benchmark 5: CPU full pipeline (resize + cvtColor + crop)
    {
        auto result = run_benchmark("CPU Full Pipeline", [&]() {
            cv::Mat mat = create_test_image(1920, 1080);
            Frame frame(mat);
            OpenCvProcessor processor;
            processor.resize(frame, 640, 640);
            processor.cvtColor(frame, ColorConversion::BGR2RGB);
            processor.crop(frame, 50, 50, 540, 540);
        }, iterations);

        print_benchmark_result(result);
        suite.add_result("Benchmark: CPU Pipeline", true, result.avg_ms);
    }

#ifdef USE_CVI_MPI
    // Benchmark 6: VPSS full pipeline (resize + cvtColor + crop)
    {
        auto result = run_benchmark("VPSS Full Pipeline", [&]() {
            cv::Mat mat = create_test_image(1920, 1080);
            Frame frame(mat);
            CviVpssProcessor processor;
            processor.resize(frame, 640, 640);
            processor.cvtColor(frame, ColorConversion::BGR2RGB);
            processor.crop(frame, 50, 50, 540, 540);
        }, iterations);

        print_benchmark_result(result);
        suite.add_result("Benchmark: VPSS Pipeline", true, result.avg_ms);
    }
#endif

    // Benchmark 7: cv_helpers smart backend (should auto-select)
    {
        auto result = run_benchmark("cv_helpers Smart Backend Pipeline", [&]() {
            cv::Mat mat = create_test_image(1920, 1080);
            Frame frame(mat);
            cv_helpers::resize(frame, 640, 640);
            cv_helpers::cvt_color(frame, ColorConversion::BGR2RGB);
            cv_helpers::crop(frame, 50, 50, 540, 540);
        }, iterations);

        print_benchmark_result(result);
        suite.add_result("Benchmark: cv_helpers Pipeline", true, result.avg_ms);
    }

    // Benchmark 8: frame_to_tensor conversion
    {
        auto result = run_benchmark("frame_to_tensor (640x640)", [&]() {
            cv::Mat mat = create_test_image(640, 640);
            Frame frame(mat);
            std::vector<double> mean = {0.485, 0.456, 0.406};
            std::vector<double> std = {0.229, 0.224, 0.225};
            auto tensor = cv_helpers::frame_to_tensor(frame, 1.0/255.0, mean, std);
        }, iterations);

        print_benchmark_result(result);
        suite.add_result("Benchmark: frame_to_tensor", true, result.avg_ms);
    }
}
