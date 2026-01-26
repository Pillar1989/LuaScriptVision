/**
 * test_cv_func.cpp - CV module tests using gtest
 *
 * Custom arguments:
 *   --image <path> or --image=<path>  Path to test image
 *
 * Any positional argument is treated as the image path if --image is not set.
 */

#include "test_common.h"
#include <cstdlib>
#include <vector>

namespace {
void parse_custom_args(int* argc, char** argv) {
    std::vector<char*> keep;
    keep.reserve(static_cast<size_t>(*argc));
    keep.push_back(argv[0]);

    for (int i = 1; i < *argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--image" && i + 1 < *argc) {
            set_test_image_path(argv[++i]);
            continue;
        }
        if (arg.rfind("--image=", 0) == 0) {
            set_test_image_path(arg.substr(8));
            continue;
        }
        if (!arg.empty() && arg[0] != '-') {
            if (test_image_path().empty()) {
                set_test_image_path(arg);
                continue;
            }
        }

        keep.push_back(argv[i]);
    }

    int out_argc = 0;
    for (char* arg : keep) {
        argv[out_argc++] = arg;
    }
    *argc = out_argc;
}
}  // namespace

int main(int argc, char* argv[]) {
    parse_custom_args(&argc, argv);

    if (test_image_path().empty()) {
        const char* env_image = std::getenv("TEST_IMAGE_PATH");
        if (env_image && env_image[0] != '\0') {
            set_test_image_path(env_image);
        }
    }

    ::testing::InitGoogleTest(&argc, argv);
#ifdef USE_CVI_MPI
    register_cvi_environment();
#endif

    return RUN_ALL_TESTS();
}
