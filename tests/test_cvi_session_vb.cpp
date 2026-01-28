#include "test_common.h"

#if defined(USE_CVI_TPU) && defined(USE_CVI_MPI)

#include "inference/cvi_session.h"
#include "memory/vb_memory.h"
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

void fill_normalized_pattern(float* data, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        data[i] = static_cast<float>((i % 256)) / 255.0f;
    }
}

void convert_float_to_uint8(const float* input, uint8_t* output, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        float scaled = input[i] * 255.0f;
        scaled = std::max(0.0f, std::min(255.0f, scaled));
        output[i] = static_cast<uint8_t>(scaled);
    }
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

TEST(CviSessionVb, CompareVbVsCpuPerformance) {
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
    auto input_shape = session.get_input_shape(0);
    int channels = channels_for_pixel_format(spec.pixel_format);
    ASSERT_GT(channels, 0);
    ASSERT_EQ(input_shape.size(), 4u);

    size_t input_size = element_count(input_shape);
    ASSERT_GT(input_size, 0u);

    std::vector<float> cpu_input(input_size);
    fill_normalized_pattern(cpu_input.data(), input_size);

    size_t vb_size_bytes =
        static_cast<size_t>(spec.width) * static_cast<size_t>(spec.height) * static_cast<size_t>(channels);
    auto vb_input = allocate_vb_block(spec.width, spec.height, spec.pixel_format, vb_size_bytes, true);
    if (!vb_input) {
        GTEST_SKIP() << "No VB block available for input";
    }

    convert_float_to_uint8(cpu_input.data(), static_cast<uint8_t*>(vb_input->data()), input_size);
    vb_input->flush_cache();

    std::vector<std::vector<float>> outputs;
    std::vector<std::vector<int64_t>> shapes;

    const int iterations = 50;
    const int warmup_count = 2;

    double cpu_input_sum = 0.0;
    double cpu_forward_sum = 0.0;
    double cpu_output_sum = 0.0;
    double cpu_total_sum = 0.0;
    int cpu_stats_count = 0;

    for (int i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        session.run_all(cpu_input.data(), input_shape, &outputs, &shapes);
        auto end = std::chrono::high_resolution_clock::now();

        if (i >= warmup_count) {
            const auto& stats = session.last_run_stats();
            cpu_input_sum += stats.input_ms;
            cpu_forward_sum += stats.forward_ms;
            cpu_output_sum += stats.output_ms;
            cpu_total_sum += std::chrono::duration<double, std::milli>(end - start).count();
            ++cpu_stats_count;
        }
    }

    double vb_input_sum = 0.0;
    double vb_forward_sum = 0.0;
    double vb_output_sum = 0.0;
    double vb_total_sum = 0.0;
    int vb_stats_count = 0;

    for (int i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        session.run_vb(vb_input, &outputs, &shapes);
        auto end = std::chrono::high_resolution_clock::now();

        if (i >= warmup_count) {
            const auto& stats = session.last_run_stats();
            vb_input_sum += stats.input_ms;
            vb_forward_sum += stats.forward_ms;
            vb_output_sum += stats.output_ms;
            vb_total_sum += std::chrono::duration<double, std::milli>(end - start).count();
            ++vb_stats_count;
        }
    }

    ASSERT_GT(cpu_stats_count, 0);
    ASSERT_GT(vb_stats_count, 0);

    double cpu_avg_ms = cpu_total_sum / cpu_stats_count;
    double cpu_input_avg = cpu_input_sum / cpu_stats_count;
    double cpu_forward_avg = cpu_forward_sum / cpu_stats_count;
    double cpu_output_avg = cpu_output_sum / cpu_stats_count;

    double vb_avg_ms = vb_total_sum / vb_stats_count;
    double vb_input_avg = vb_input_sum / vb_stats_count;
    double vb_forward_avg = vb_forward_sum / vb_stats_count;
    double vb_output_avg = vb_output_sum / vb_stats_count;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n========== VB vs CPU Performance Comparison ==========\n";
    std::cout << "CPU Path:\n";
    std::cout << "  Total:   " << cpu_avg_ms << " ms\n";
    std::cout << "  Input:   " << cpu_input_avg << " ms\n";
    std::cout << "  Forward: " << cpu_forward_avg << " ms\n";
    std::cout << "  Output:  " << cpu_output_avg << " ms\n\n";

    std::cout << "VB Path (Zero-Copy):\n";
    std::cout << "  Total:   " << vb_avg_ms << " ms\n";
    std::cout << "  Input:   " << vb_input_avg << " ms\n";
    std::cout << "  Forward: " << vb_forward_avg << " ms\n";
    std::cout << "  Output:  " << vb_output_avg << " ms\n\n";

    double speedup = cpu_avg_ms / vb_avg_ms;
    double time_saved = cpu_avg_ms - vb_avg_ms;
    std::cout << "Speedup: " << speedup << "x\n";
    std::cout << "Time saved: " << time_saved << " ms\n";
    std::cout << "====================================================\n\n";

    EXPECT_LT(vb_avg_ms, cpu_avg_ms) << "VB path should be faster than CPU path";
    EXPECT_LT(vb_avg_ms, 60.0) << "VB path total should be <60ms (Forward ~36ms + Output ~17ms + overhead)";
    EXPECT_LT(vb_input_avg, 1.0) << "VB input should achieve near-zero copy (<1ms)";
    EXPECT_LT(vb_forward_avg, 40.0) << "TPU forward should be <40ms (hardware baseline ~36ms)";
    EXPECT_LT(vb_output_avg, 20.0) << "Output processing should be <20ms (quantization conversion)";
    EXPECT_GT(speedup, 2.0) << "VB path should be at least 2x faster than CPU path";
}

#endif  // USE_CVI_TPU && USE_CVI_MPI

int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
#if defined(USE_CVI_MPI)
    register_cvi_environment();
#endif
    return RUN_ALL_TESTS();
}
