#include "test_common.h"

#if defined(USE_CVI_TPU) && defined(USE_CVI_MPI)

#include "inference/cvi_session.h"
#include "tensor/vb_memory.h"
#include <cstdlib>
#include <fstream>

namespace {
std::string resolve_model_path() {
    const char* env = std::getenv("TEST_CVI_MODEL");
    if (env && env[0] != '\0') {
        return std::string(env);
    }
    return "/userdata/Models/model.cvimodel";
}

bool file_exists(const std::string& path) {
    std::ifstream file(path);
    return file.good();
}

bool prepare_environment(std::string* model_path, std::string* reason) {
    if (!is_cvi_ready()) {
        if (reason) {
            *reason = "CVI system not ready";
        }
        return false;
    }
    std::string resolved = resolve_model_path();
    if (!file_exists(resolved)) {
        if (reason) {
            *reason = "Model not found: " + resolved;
        }
        return false;
    }
    if (model_path) {
        *model_path = resolved;
    }
    return true;
}

int channels_for_pixel_format(PIXEL_FORMAT_E fmt) {
    switch (fmt) {
        case PIXEL_FORMAT_RGB_888:
        case PIXEL_FORMAT_BGR_888:
        case PIXEL_FORMAT_RGB_888_PLANAR:
        case PIXEL_FORMAT_BGR_888_PLANAR:
            return 3;
        case PIXEL_FORMAT_YUV_400:
            return 1;
        default:
            return 0;
    }
}

size_t element_count(const std::vector<int64_t>& shape) {
    size_t count = 1;
    for (int64_t dim : shape) {
        if (dim <= 0) {
            return 0;
        }
        count *= static_cast<size_t>(dim);
    }
    return count;
}

void fill_input_pattern(void* data, size_t size_bytes) {
    std::memset(data, 0x7f, size_bytes);
}

std::shared_ptr<tensor::VbMemory> allocate_vb_block(uint32_t width,
                                                    uint32_t height,
                                                    PIXEL_FORMAT_E fmt,
                                                    size_t size_bytes,
                                                    bool cached) {
    VB_POOL pool = find_suitable_vb_pool(width, height, fmt);
    if (pool == VB_INVALID_POOLID) {
        return nullptr;
    }
    VB_BLK block = CVI_VB_GetBlock(pool, static_cast<CVI_U32>(size_bytes));
    if (block == VB_INVALID_HANDLE) {
        return nullptr;
    }
    try {
        return tensor::VbMemory::from_block(block, size_bytes, cached, true);
    } catch (...) {
        CVI_VB_ReleaseBlock(block);
        throw;
    }
}
}  // namespace

TEST(CviSessionVb, SupportsVbInputQuery) {
    std::string model_path;
    std::string skip_reason;
    if (!prepare_environment(&model_path, &skip_reason)) {
        GTEST_SKIP() << skip_reason;
    }
    inference::CviSession session(model_path);

    auto info = session.get_input_info(0);
    bool supported = session.supports_vb_input();

    if (info.fmt == CVI_FMT_FP32) {
        EXPECT_FALSE(supported);
    } else {
        EXPECT_TRUE(supported);
    }
    EXPECT_EQ(info.shape.size(), 4u);
}

TEST(CviSessionVb, GetVbInputSpec) {
    std::string model_path;
    std::string skip_reason;
    if (!prepare_environment(&model_path, &skip_reason)) {
        GTEST_SKIP() << skip_reason;
    }
    inference::CviSession session(model_path);

    if (!session.supports_vb_input()) {
        GTEST_SKIP() << "Model does not support VB input";
    }

    auto spec = session.get_vb_input_spec();
    EXPECT_GT(spec.width, 0u);
    EXPECT_GT(spec.height, 0u);
    EXPECT_NE(spec.pixel_format, PIXEL_FORMAT_MAX);
}

TEST(CviSessionVb, RunVbPhysicalAddress) {
    std::string model_path;
    std::string skip_reason;
    if (!prepare_environment(&model_path, &skip_reason)) {
        GTEST_SKIP() << skip_reason;
    }
    inference::CviSession session(model_path);

    if (!session.supports_vb_input()) {
        GTEST_SKIP() << "Model does not support VB input";
    }

    auto spec = session.get_vb_input_spec();
    int channels = channels_for_pixel_format(spec.pixel_format);
    ASSERT_GT(channels, 0);

    size_t expected_size =
        static_cast<size_t>(spec.width) * static_cast<size_t>(spec.height) * static_cast<size_t>(channels);
    auto input = allocate_vb_block(spec.width, spec.height, spec.pixel_format, expected_size, true);
    if (!input) {
        GTEST_SKIP() << "No VB block available for input";
    }
    ASSERT_NE(input, nullptr);
    ASSERT_NE(input->physical_address(), 0u);
    ASSERT_NE(input->data(), nullptr);

    fill_input_pattern(input->data(), input->size_bytes());
    input->flush_cache();

    std::vector<std::vector<float>> outputs;
    std::vector<std::vector<int64_t>> shapes;
    EXPECT_NO_THROW(session.run_vb(input->physical_address(),
                                   input->size_bytes(),
                                   &outputs, &shapes));

    ASSERT_EQ(outputs.size(), shapes.size());
    EXPECT_GT(outputs.size(), 0u);
    for (size_t i = 0; i < outputs.size(); ++i) {
        size_t expected_count = element_count(shapes[i]);
        EXPECT_GT(expected_count, 0u);
        EXPECT_EQ(outputs[i].size(), expected_count);
    }
}

TEST(CviSessionVb, RunVbMemoryWrapper) {
    std::string model_path;
    std::string skip_reason;
    if (!prepare_environment(&model_path, &skip_reason)) {
        GTEST_SKIP() << skip_reason;
    }
    inference::CviSession session(model_path);

    if (!session.supports_vb_input()) {
        GTEST_SKIP() << "Model does not support VB input";
    }

    auto spec = session.get_vb_input_spec();
    int channels = channels_for_pixel_format(spec.pixel_format);
    ASSERT_GT(channels, 0);

    size_t expected_size =
        static_cast<size_t>(spec.width) * static_cast<size_t>(spec.height) * static_cast<size_t>(channels);
    auto input = allocate_vb_block(spec.width, spec.height, spec.pixel_format, expected_size, true);
    if (!input) {
        GTEST_SKIP() << "No VB block available for input";
    }
    ASSERT_NE(input, nullptr);
    ASSERT_NE(input->physical_address(), 0u);
    ASSERT_NE(input->data(), nullptr);

    fill_input_pattern(input->data(), input->size_bytes());
    input->flush_cache();

    std::vector<std::vector<float>> outputs;
    std::vector<std::vector<int64_t>> shapes;
    EXPECT_NO_THROW(session.run_vb(input, &outputs, &shapes));
    EXPECT_EQ(outputs.size(), shapes.size());
    EXPECT_GT(outputs.size(), 0u);
}

TEST(CviSessionVb, RunVbSelectedOutput) {
    std::string model_path;
    std::string skip_reason;
    if (!prepare_environment(&model_path, &skip_reason)) {
        GTEST_SKIP() << skip_reason;
    }
    inference::CviSession session(model_path);

    if (!session.supports_vb_input()) {
        GTEST_SKIP() << "Model does not support VB input";
    }

    auto spec = session.get_vb_input_spec();
    int channels = channels_for_pixel_format(spec.pixel_format);
    ASSERT_GT(channels, 0);

    size_t expected_size =
        static_cast<size_t>(spec.width) * static_cast<size_t>(spec.height) * static_cast<size_t>(channels);
    auto input = allocate_vb_block(spec.width, spec.height, spec.pixel_format, expected_size, true);
    if (!input) {
        GTEST_SKIP() << "No VB block available for input";
    }
    ASSERT_NE(input, nullptr);
    ASSERT_NE(input->physical_address(), 0u);
    ASSERT_NE(input->data(), nullptr);

    fill_input_pattern(input->data(), input->size_bytes());
    input->flush_cache();

    std::vector<int32_t> output_indices = {session.primary_output_index()};
    std::vector<std::vector<float>> outputs;
    std::vector<std::vector<int64_t>> shapes;
    EXPECT_NO_THROW(session.run_vb_selected(input, output_indices, &outputs, &shapes));

    ASSERT_EQ(outputs.size(), output_indices.size());
    ASSERT_EQ(outputs.size(), shapes.size());
    size_t expected_count = element_count(shapes[0]);
    EXPECT_GT(expected_count, 0u);
    EXPECT_EQ(outputs[0].size(), expected_count);
}

TEST(CviSessionVb, VbSelectedOutputBenchmark) {
    std::string model_path;
    std::string skip_reason;
    if (!prepare_environment(&model_path, &skip_reason)) {
        GTEST_SKIP() << skip_reason;
    }
    inference::CviSession session(model_path);

    if (!session.supports_vb_input()) {
        GTEST_SKIP() << "Model does not support VB input";
    }

    auto spec = session.get_vb_input_spec();
    int channels = channels_for_pixel_format(spec.pixel_format);
    ASSERT_GT(channels, 0);

    size_t expected_size =
        static_cast<size_t>(spec.width) * static_cast<size_t>(spec.height) * static_cast<size_t>(channels);
    auto input = allocate_vb_block(spec.width, spec.height, spec.pixel_format, expected_size, true);
    if (!input) {
        GTEST_SKIP() << "No VB block available for input";
    }

    fill_input_pattern(input->data(), input->size_bytes());
    input->flush_cache();

    std::vector<int32_t> output_indices = {session.primary_output_index()};
    std::vector<std::vector<float>> outputs;
    std::vector<std::vector<int64_t>> shapes;

    const int iterations = 20;
    int call_count = 0;
    double input_sum = 0.0;
    double forward_sum = 0.0;
    double output_sum = 0.0;
    int stats_count = 0;
    BenchmarkResult result = run_benchmark(
        "CviSessionVB selected-output steady-state",
        [&]() {
            session.run_vb_selected(input, output_indices, &outputs, &shapes);
            ++call_count;
            if (call_count > 2) {
                const auto& stats = session.last_run_stats();
                input_sum += stats.input_ms;
                forward_sum += stats.forward_ms;
                output_sum += stats.output_ms;
                ++stats_count;
            }
        },
        iterations);

    print_benchmark_result(result);
    if (stats_count > 0) {
        std::cout << "  Breakdown avg: input=" << std::fixed << std::setprecision(3)
                  << (input_sum / stats_count) << " ms, forward="
                  << (forward_sum / stats_count) << " ms, output="
                  << (output_sum / stats_count) << " ms" << std::endl;
    }

    ASSERT_EQ(outputs.size(), output_indices.size());
    EXPECT_GT(outputs[0].size(), 0u);
}

TEST(CviSessionVb, InvalidPhysAddr) {
    std::string model_path;
    std::string skip_reason;
    if (!prepare_environment(&model_path, &skip_reason)) {
        GTEST_SKIP() << skip_reason;
    }
    inference::CviSession session(model_path);

    if (!session.supports_vb_input()) {
        GTEST_SKIP() << "Model does not support VB input";
    }

    std::vector<std::vector<float>> outputs;
    std::vector<std::vector<int64_t>> shapes;
    EXPECT_THROW(session.run_vb(0, 1, &outputs, &shapes), std::invalid_argument);
}

TEST(CviSessionVb, InputSizeMismatch) {
    std::string model_path;
    std::string skip_reason;
    if (!prepare_environment(&model_path, &skip_reason)) {
        GTEST_SKIP() << skip_reason;
    }
    inference::CviSession session(model_path);

    if (!session.supports_vb_input()) {
        GTEST_SKIP() << "Model does not support VB input";
    }

    auto spec = session.get_vb_input_spec();
    int channels = channels_for_pixel_format(spec.pixel_format);
    ASSERT_GT(channels, 0);

    size_t expected_size =
        static_cast<size_t>(spec.width) * static_cast<size_t>(spec.height) * static_cast<size_t>(channels);
    size_t small_size = expected_size > 1 ? expected_size / 2 : 1;
    auto input = allocate_vb_block(spec.width, spec.height, spec.pixel_format, expected_size, true);
    if (!input) {
        GTEST_SKIP() << "No VB block available for input";
    }
    auto small_view = tensor::VbMemory::from_block(input->block(), small_size, true, false);

    std::vector<std::vector<float>> outputs;
    std::vector<std::vector<int64_t>> shapes;
    EXPECT_THROW(session.run_vb(small_view, &outputs, &shapes), std::runtime_error);
}

TEST(CviSessionVb, VbSteadyStateBenchmark) {
    std::string model_path;
    std::string skip_reason;
    if (!prepare_environment(&model_path, &skip_reason)) {
        GTEST_SKIP() << skip_reason;
    }
    inference::CviSession session(model_path);

    if (!session.supports_vb_input()) {
        GTEST_SKIP() << "Model does not support VB input";
    }

    auto spec = session.get_vb_input_spec();
    int channels = channels_for_pixel_format(spec.pixel_format);
    ASSERT_GT(channels, 0);

    size_t expected_size =
        static_cast<size_t>(spec.width) * static_cast<size_t>(spec.height) * static_cast<size_t>(channels);
    auto input = allocate_vb_block(spec.width, spec.height, spec.pixel_format, expected_size, true);
    if (!input) {
        GTEST_SKIP() << "No VB block available for input";
    }

    fill_input_pattern(input->data(), input->size_bytes());
    input->flush_cache();

    std::vector<std::vector<float>> outputs;
    std::vector<std::vector<int64_t>> shapes;

    const int iterations = 20;
    int call_count = 0;
    double input_sum = 0.0;
    double forward_sum = 0.0;
    double output_sum = 0.0;
    int stats_count = 0;
    BenchmarkResult result = run_benchmark(
        "CviSessionVB steady-state",
        [&]() {
            session.run_vb(input, &outputs, &shapes);
            ++call_count;
            if (call_count > 2) {
                const auto& stats = session.last_run_stats();
                input_sum += stats.input_ms;
                forward_sum += stats.forward_ms;
                output_sum += stats.output_ms;
                ++stats_count;
            }
        },
        iterations);

    print_benchmark_result(result);
    if (stats_count > 0) {
        std::cout << "  Breakdown avg: input=" << std::fixed << std::setprecision(3)
                  << (input_sum / stats_count) << " ms, forward="
                  << (forward_sum / stats_count) << " ms, output="
                  << (output_sum / stats_count) << " ms" << std::endl;
    }

    ASSERT_FALSE(outputs.empty());
    EXPECT_GT(outputs[0].size(), 0u);
}

#endif  // USE_CVI_TPU && USE_CVI_MPI

int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
#if defined(USE_CVI_MPI)
    register_cvi_environment();
#endif
    return RUN_ALL_TESTS();
}
