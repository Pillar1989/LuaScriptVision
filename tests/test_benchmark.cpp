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
    uint32_t bench_width = 1920;
    uint32_t bench_height = 1080;

#ifdef USE_CVI_MPI
    if (find_suitable_vb_pool(bench_width, bench_height, PIXEL_FORMAT_BGR_888) == VB_INVALID_POOLID) {
        bench_width = 1280;
        bench_height = 720;
    }
#endif

    const std::string resize_label = "CPU Resize " + std::to_string(bench_width) + "x" +
                                     std::to_string(bench_height) + "->640x640";
    const std::string vpss_resize_label = "VPSS Resize " + std::to_string(bench_width) + "x" +
                                          std::to_string(bench_height) + "->640x640";
    const std::string cpu_pipeline_label = "CPU Full Pipeline (" + std::to_string(bench_width) + "x" +
                                           std::to_string(bench_height) + ")";
    const std::string vpss_pipeline_label = "VPSS Full Pipeline (" + std::to_string(bench_width) + "x" +
                                            std::to_string(bench_height) + ")";
    const std::string helpers_pipeline_label = "cv_helpers Pipeline (" + std::to_string(bench_width) + "x" +
                                               std::to_string(bench_height) + ")";

    // Benchmark 1: CPU resize
    {
        auto result = run_benchmark(resize_label, [&]() {
            cv::Mat mat = create_test_image(bench_width, bench_height);
            Frame frame(mat);
            OpenCvProcessor processor;
            processor.resize(frame, 640, 640);
        }, iterations);

        print_benchmark_result(result);
        suite.add_result("Benchmark: CPU Resize", true, result.avg_ms);
    }

#ifdef USE_CVI_MPI
    // Benchmark 2: VPSS resize
    {
        auto result = run_benchmark(vpss_resize_label, [&]() {
            cv::Mat mat = create_test_image(bench_width, bench_height);
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
        auto result = run_benchmark(cpu_pipeline_label, [&]() {
            cv::Mat mat = create_test_image(bench_width, bench_height);
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
        auto result = run_benchmark(vpss_pipeline_label, [&]() {
            cv::Mat mat = create_test_image(bench_width, bench_height);
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
        auto result = run_benchmark(helpers_pipeline_label, [&]() {
            cv::Mat mat = create_test_image(bench_width, bench_height);
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
