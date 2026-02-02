/**
 * test_benchmark.cpp - Performance benchmarks for CV operations
 */

#include "test_common.h"

#ifdef USE_CVI_MPI
#include "cv/hw_jpeg_decoder.h"
#include <cvi_buffer.h>
#include <fstream>
#endif

namespace {
#ifdef USE_CVI_MPI
struct VbInputFrame {
    VIDEO_FRAME_INFO_S frame{};
    VB_BLK block = VB_INVALID_HANDLE;
};

bool read_file(const std::string& path, std::vector<uint8_t>& data) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return false;
    }
    file.seekg(0, std::ios::end);
    std::streamoff size = file.tellg();
    if (size <= 0) {
        return false;
    }
    data.resize(static_cast<size_t>(size));
    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char*>(data.data()), size);
    return file.good();
}

bool parse_jpeg_size(const uint8_t* data, size_t size, uint32_t& width, uint32_t& height) {
    if (!data || size < 4 || data[0] != 0xFF || data[1] != 0xD8) {
        return false;
    }
    size_t i = 2;
    while (i + 1 < size) {
        if (data[i] != 0xFF) {
            ++i;
            continue;
        }
        while (i < size && data[i] == 0xFF) {
            ++i;
        }
        if (i >= size) {
            break;
        }
        uint8_t marker = data[i++];
        if (marker == 0xD9 || marker == 0xDA) {
            break;
        }
        if (i + 1 >= size) {
            break;
        }
        uint16_t len = static_cast<uint16_t>(data[i] << 8 | data[i + 1]);
        if (len < 2 || i + len > size) {
            break;
        }
        if (marker == 0xC0 || marker == 0xC2) {
            if (len < 7) {
                return false;
            }
            uint16_t h = static_cast<uint16_t>(data[i + 3] << 8 | data[i + 4]);
            uint16_t w = static_cast<uint16_t>(data[i + 5] << 8 | data[i + 6]);
            width = w;
            height = h;
            return true;
        }
        i += len;
    }
    return false;
}

VbInputFrame create_bgr_vb_input(uint32_t width, uint32_t height, VB_POOL pool) {
    VbInputFrame input{};

    VB_CAL_CONFIG_S cal{};
    COMMON_GetPicBufferConfig(width, height, PIXEL_FORMAT_BGR_888,
                              DATA_BITWIDTH_8, COMPRESS_MODE_NONE, 0, &cal);
    if (cal.u32VBSize == 0) {
        throw std::runtime_error("create_bgr_vb_input - invalid VB size");
    }

    VB_BLK block = CVI_VB_GetBlock(pool, cal.u32VBSize);
    if (block == VB_INVALID_HANDLE && pool != VB_INVALID_POOLID) {
        block = CVI_VB_GetBlock(VB_INVALID_POOLID, cal.u32VBSize);
    }
    if (block == VB_INVALID_HANDLE) {
        throw std::runtime_error("create_bgr_vb_input - CVI_VB_GetBlock failed");
    }

    CVI_U64 phys = CVI_VB_Handle2PhysAddr(block);
    void* mapped = CVI_SYS_MmapCache(phys, cal.u32VBSize);
    if (!mapped) {
        CVI_VB_ReleaseBlock(block);
        throw std::runtime_error("create_bgr_vb_input - CVI_SYS_MmapCache failed");
    }

    uint8_t* dst = static_cast<uint8_t*>(mapped);
    const uint32_t stride = cal.u32MainStride;
    for (uint32_t y = 0; y < height; ++y) {
        uint8_t* row = dst + static_cast<size_t>(y) * stride;
        for (uint32_t x = 0; x < width; ++x) {
            row[x * 3 + 0] = static_cast<uint8_t>((y * 255) / height);
            row[x * 3 + 1] = static_cast<uint8_t>((x * 255) / width);
            row[x * 3 + 2] = static_cast<uint8_t>(((x + y) * 255) / (width + height));
        }
    }

    CVI_SYS_IonFlushCache(phys, mapped, cal.u32VBSize);
    CVI_SYS_Munmap(mapped, cal.u32VBSize);

    VIDEO_FRAME_INFO_S frame{};
    frame.stVFrame.u32Width = width;
    frame.stVFrame.u32Height = height;
    frame.stVFrame.enPixelFormat = PIXEL_FORMAT_BGR_888;
    frame.stVFrame.enVideoFormat = VIDEO_FORMAT_LINEAR;
    frame.stVFrame.enCompressMode = COMPRESS_MODE_NONE;
    frame.stVFrame.enDynamicRange = DYNAMIC_RANGE_SDR8;
    frame.stVFrame.enColorGamut = COLOR_GAMUT_BT709;
    frame.stVFrame.u32Stride[0] = cal.u32MainStride;
    frame.stVFrame.u32Length[0] = cal.u32MainSize;
    frame.stVFrame.u64PhyAddr[0] = phys;
    frame.u32PoolId = CVI_VB_Handle2PoolId(block);

    input.frame = frame;
    input.block = block;
    return input;
}
#endif
}  // namespace

TEST(BenchmarkTest, CpuBenchmarks) {
    const int iterations = 10;
    uint32_t bench_width = 1920;
    uint32_t bench_height = 1080;

    const std::string resize_label = "CPU Resize " + std::to_string(bench_width) + "x" +
                                     std::to_string(bench_height) + "->640x640";
    const std::string cpu_pipeline_label = "CPU Full Pipeline (" + std::to_string(bench_width) + "x" +
                                           std::to_string(bench_height) + ")";
    const std::string helpers_pipeline_label = "cv_helpers Pipeline (" + std::to_string(bench_width) + "x" +
                                               std::to_string(bench_height) + ")";

    auto resize_result = run_benchmark(resize_label, [&]() {
        cv::Mat mat = create_test_image(bench_width, bench_height);
        Frame frame(mat);
        OpenCvProcessor processor;
        processor.resize(frame, 640, 640);
    }, iterations);
    print_benchmark_result(resize_result);
    EXPECT_GT(resize_result.avg_ms, 0.0);

    auto cvt_result = run_benchmark("CPU cvtColor BGR2RGB", [&]() {
        cv::Mat mat = create_test_image(640, 480);
        Frame frame(mat);
        OpenCvProcessor processor;
        processor.cvtColor(frame, ColorConversion::BGR2RGB);
    }, iterations);
    print_benchmark_result(cvt_result);
    EXPECT_GT(cvt_result.avg_ms, 0.0);

    auto pipeline_result = run_benchmark(cpu_pipeline_label, [&]() {
        cv::Mat mat = create_test_image(bench_width, bench_height);
        Frame frame(mat);
        OpenCvProcessor processor;
        processor.resize(frame, 640, 640);
        processor.cvtColor(frame, ColorConversion::BGR2RGB);
        processor.crop(frame, 50, 50, 540, 540);
    }, iterations);
    print_benchmark_result(pipeline_result);
    EXPECT_GT(pipeline_result.avg_ms, 0.0);

    auto helpers_result = run_benchmark(helpers_pipeline_label, [&]() {
        cv::Mat mat = create_test_image(bench_width, bench_height);
        Frame frame(mat);
        cv_helpers::resize(frame, 640, 640);
        cv_helpers::cvt_color(frame, ColorConversion::BGR2RGB);
        cv_helpers::crop(frame, 50, 50, 540, 540);
    }, iterations);
    print_benchmark_result(helpers_result);
    EXPECT_GT(helpers_result.avg_ms, 0.0);

    auto tensor_result = run_benchmark("frame_to_tensor (640x640)", [&]() {
        cv::Mat mat = create_test_image(640, 640);
        Frame frame(mat);
        std::vector<double> mean = {0.485, 0.456, 0.406};
        std::vector<double> std = {0.229, 0.224, 0.225};
        auto tensor = cv_helpers::frame_to_tensor(frame, 1.0 / 255.0, mean, std);
        (void)tensor;
    }, iterations);
    print_benchmark_result(tensor_result);
    EXPECT_GT(tensor_result.avg_ms, 0.0);
}

#ifdef USE_CVI_MPI
TEST(BenchmarkTest, VpssBenchmarks) {
    if (!is_cvi_ready()) {
        GTEST_SKIP() << "CVI system not initialized";
    }

    const int iterations = 10;
    uint32_t bench_width = 1280;
    uint32_t bench_height = 720;
    VB_POOL vb_pool = find_suitable_vb_pool(bench_width, bench_height, PIXEL_FORMAT_BGR_888);
    if (vb_pool == VB_INVALID_POOLID) {
        GTEST_SKIP() << "No suitable VB pool for VPSS benchmarks";
    }

    CviVpssProcessor processor;
    VbInputFrame input = create_bgr_vb_input(bench_width, bench_height, vb_pool);
    VbBlockGuard input_guard(input.block);

    const std::string vpss_steady_label = "VPSS Letterbox (VB steady-state) " +
                                          std::to_string(bench_width) + "x" +
                                          std::to_string(bench_height) + "->640x640";
    auto vpss_steady_result = run_benchmark(vpss_steady_label, [&]() {
        Frame frame(input.frame, false);
        processor.letterbox(frame, 640, 640, 114, nullptr);
    }, iterations);
    print_benchmark_result(vpss_steady_result);
    EXPECT_GT(vpss_steady_result.avg_ms, 0.0);

    if (!test_image_path().empty()) {
#ifdef USE_VDEC_DECODE
        // Hardware VDEC decode
        std::vector<uint8_t> jpeg_data;
        if (read_file(test_image_path(), jpeg_data)) {
            uint32_t jpeg_w = 0;
            uint32_t jpeg_h = 0;
            if (!parse_jpeg_size(jpeg_data.data(), jpeg_data.size(), jpeg_w, jpeg_h)) {
                std::cout << "[WARN] JPEG size parse failed; skipping full pipeline benchmark" << std::endl;
            } else {
                HwJpegDecoder decoder;
                if (decoder.init(jpeg_w, jpeg_h)) {
                    const std::string vpss_full_label = "VDEC->VPSS Letterbox (full pipeline)";
                    auto vpss_full_result = run_benchmark(vpss_full_label, [&]() {
                        VIDEO_FRAME_INFO_S decoded = decoder.decode_sync(jpeg_data.data(), jpeg_data.size());
                        VdecFrameGuard frame_guard(&decoder, decoded);  // RAII: 自动释放
                        Frame frame(decoded, false);
                        processor.letterbox(frame, 640, 640, 114, nullptr);
                        // frame_guard 析构时自动调用 decoder.release_frame(decoded)
                    }, iterations);
                    print_benchmark_result(vpss_full_result);
                    EXPECT_GT(vpss_full_result.avg_ms, 0.0);
                } else {
                    std::cout << "[WARN] HwJpegDecoder init failed; skipping full pipeline benchmark" << std::endl;
                }
            }
        } else {
            std::cout << "[WARN] Failed to read test image; skipping full pipeline benchmark" << std::endl;
        }
#else
        // Software JPEG decode (OpenCV) - no VB pool required
        cv::Mat jpeg_img = decode_jpeg_software(test_image_path());
        if (!jpeg_img.empty()) {
            const std::string vpss_full_label = "SW Decode->VPSS Letterbox (full pipeline)";
            auto vpss_full_result = run_benchmark(vpss_full_label, [&]() {
                Frame frame(jpeg_img);  // cv::Mat -> Frame
                processor.letterbox(frame, 640, 640, 114, nullptr);
            }, iterations);
            print_benchmark_result(vpss_full_result);
            EXPECT_GT(vpss_full_result.avg_ms, 0.0);
        } else {
            std::cout << "[WARN] Failed to decode test image; skipping full pipeline benchmark" << std::endl;
        }
#endif
    } else {
        std::cout << "[WARN] No test image path; skipping full pipeline benchmark" << std::endl;
    }

}
#endif
