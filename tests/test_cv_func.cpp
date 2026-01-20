/**
 * test_cv_func.cpp - CV Module Comprehensive Test Suite
 *
 * Integrates all modular CV tests:
 * - Frame basic functionality
 * - OpenCV processor
 * - cv_helpers smart backend
 * - VPSS hardware acceleration (when available)
 * - Real image processing
 * - Performance benchmarks
 *
 * Usage:
 *   test_cv_func [options] [image_path]
 *
 * Options:
 *   --list              List available tests
 *   --test=<name>       Run specific test(s), can be used multiple times
 *   --help              Show this help
 *
 * Examples:
 *   test_cv_func                           # Run all tests
 *   test_cv_func /tmp/image.jpg            # Run all tests with image
 *   test_cv_func --test=ion /tmp/image.jpg # Run only ion tests
 *   test_cv_func --test=vpss --test=perf   # Run vpss and perf tests
 */

#include "test_common.h"
#include <cstdlib>
#include <cstring>
#include <set>

// Forward declarations of test functions
void run_frame_tests(TestSuite& suite);
void run_opencv_tests(TestSuite& suite);
void run_cv_helpers_tests(TestSuite& suite);
void run_vpss_tests(TestSuite& suite);
void run_real_image_tests(TestSuite& suite, const std::string& image_path);
void run_benchmark_tests(TestSuite& suite);
void run_vpss_performance_tests(TestSuite& suite);
void run_ion_loader_tests(TestSuite& suite, const std::string& image_path);
void run_camera_capture_tests(TestSuite& suite);

// Test registry
struct TestInfo {
    const char* name;
    const char* description;
    bool needs_image;
};

static const TestInfo available_tests[] = {
    {"frame",     "Frame basic functionality",           false},
    {"opencv",    "OpenCvProcessor functionality",       false},
    {"helpers",   "cv_helpers smart backend selection",  false},
    {"vpss",      "CviVpssProcessor hardware accel",     false},
    {"real",      "Real image processing",               true},
    {"bench",     "Performance benchmarks",              false},
    {"perf",      "VPSS detailed performance analysis",  false},
    {"ion",       "IonImageLoader performance",          true},
    {"camera",    "Camera capture (VI→ISP→VPSS)",        false},
    {"all",       "Run all tests",                       false},
};

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options] [image_path]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --list              List available tests\n";
    std::cout << "  --test=<name>       Run specific test(s), can be used multiple times\n";
    std::cout << "  --help              Show this help\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << prog << "                           # Run all tests\n";
    std::cout << "  " << prog << " /tmp/image.jpg            # Run all tests with image\n";
    std::cout << "  " << prog << " --test=ion /tmp/image.jpg # Run only ion tests\n";
    std::cout << "  " << prog << " --test=vpss --test=perf   # Run vpss and perf tests\n";
}

static void print_test_list() {
    std::cout << "Available tests:\n\n";
    for (const auto& test : available_tests) {
        std::cout << "  " << std::left << std::setw(10) << test.name
                  << " - " << test.description;
        if (test.needs_image) {
            std::cout << " (requires image)";
        }
        std::cout << "\n";
    }
}

int main(int argc, char* argv[]) {
    std::string image_path;
    std::set<std::string> selected_tests;
    bool run_all = false;

    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
        else if (arg == "--list") {
            print_test_list();
            return 0;
        }
        else if (arg.rfind("--test=", 0) == 0) {
            std::string test_name = arg.substr(7);
            if (test_name == "all") {
                run_all = true;
            } else {
                selected_tests.insert(test_name);
            }
        }
        else if (arg[0] != '-') {
            image_path = arg;
        }
        else {
            std::cerr << "Unknown option: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    // If no tests specified, run all
    if (selected_tests.empty()) {
        run_all = true;
    }

    std::cout << "========================================" << std::endl;
    std::cout << "  CV Module Comprehensive Test Suite" << std::endl;
    std::cout << "  Built: " << __DATE__ << " " << __TIME__ << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    if (!image_path.empty()) {
        std::cout << "Using test image: " << image_path << std::endl;
    } else {
        std::cout << "No test image provided (some tests will be skipped)" << std::endl;
    }

    if (!run_all) {
        std::cout << "Running selected tests:";
        for (const auto& t : selected_tests) {
            std::cout << " " << t;
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    bool cvi_initialized = false;

#ifdef USE_CVI_MPI
    std::cout << "========================================" << std::endl;
    if (!init_cvi_system()) {
        std::cerr << "[ERROR] Failed to initialize CVI system" << std::endl;
        std::cerr << "VPSS hardware tests will fail" << std::endl;
        std::cout << std::endl;
    } else {
        cvi_initialized = true;
    }
    std::cout << "========================================" << std::endl;
#else
    std::cout << "[INFO] USE_CVI_MPI not defined, VPSS tests will be skipped" << std::endl;
    std::cout << std::endl;
#endif

    // Helper to check if test should run
    auto should_run = [&](const std::string& name) {
        return run_all || selected_tests.count(name) > 0;
    };

    // Create test suite
    TestSuite suite;

    // Run selected test modules
    if (should_run("frame")) {
        run_frame_tests(suite);
    }

    if (should_run("opencv")) {
        run_opencv_tests(suite);
    }

    if (should_run("helpers")) {
        run_cv_helpers_tests(suite);
    }

    if (should_run("vpss")) {
        run_vpss_tests(suite);
    }

    if (should_run("real")) {
        if (!image_path.empty()) {
            run_real_image_tests(suite, image_path);
        } else {
            std::cout << "\n[Test] Real Image Processing" << std::endl;
            std::cout << "  ⊘  Skipped (no image provided)" << std::endl;
        }
    }

    if (should_run("bench")) {
        run_benchmark_tests(suite);
    }

    if (should_run("perf")) {
        run_vpss_performance_tests(suite);
    }

    if (should_run("ion")) {
        if (!image_path.empty()) {
            run_ion_loader_tests(suite, image_path);
        } else {
            std::cout << "\n[Test] IonImageLoader Performance" << std::endl;
            std::cout << "  ⊘  Skipped (no image provided)" << std::endl;
        }
    }

    if (should_run("camera")) {
        run_camera_capture_tests(suite);
    }

    // Print summary
    suite.print_summary();

#ifdef USE_CVI_MPI
    // Cleanup CVI system (only if we initialized it)
    if (cvi_initialized) {
        cleanup_cvi_system();
    }
#endif

    return suite.all_passed() ? 0 : 1;
}
