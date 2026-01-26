/**
 * test_vpss_performance.cpp - VPSS performance breakdown
 */

#include "test_common.h"
#include <numeric>
#include <algorithm>

#ifdef USE_CVI_MPI
#include "cv/hw_jpeg_decoder.h"
#include <cvi_buffer.h>
#include <fstream>
#include <vector>

namespace {
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
}  // namespace

TEST(VpssPerformanceTest, Breakdown) {
    if (!is_cvi_ready()) {
        GTEST_SKIP() << "CVI system not initialized";
    }

    const int iterations = 10;
    CviVpssProcessor processor;
    Timer timer;

    uint32_t input_width = 1280;
    uint32_t input_height = 720;
    VB_POOL vb_pool = find_suitable_vb_pool(input_width, input_height, PIXEL_FORMAT_BGR_888);
    if (vb_pool == VB_INVALID_POOLID) {
        GTEST_SKIP() << "No suitable VB pool for VPSS tests";
    }

    VbInputFrame input = create_bgr_vb_input(input_width, input_height, vb_pool);
    VbBlockGuard input_guard(input.block);

    double cpu_resize_time = 0.0;
    {
        cv::Mat mat = create_test_image(input_width, input_height);
        Frame frame(mat);
        timer.start();
        OpenCvProcessor cpu_processor;
        cpu_processor.resize(frame, 640, 640);
        cpu_resize_time = timer.elapsed_ms();
    }

    {
        Frame warm(input.frame, false);
        processor.letterbox(warm, 640, 640, 114, nullptr);
    }

    std::vector<double> vpss_times;
    vpss_times.reserve(iterations);
    for (int i = 0; i < iterations; ++i) {
        Frame frame(input.frame, false);
        timer.start();
        processor.letterbox(frame, 640, 640, 114, nullptr);
        vpss_times.push_back(timer.elapsed_ms());
    }

    double vpss_min = *std::min_element(vpss_times.begin(), vpss_times.end());
    double vpss_avg = std::accumulate(vpss_times.begin(), vpss_times.end(), 0.0) / iterations;

    bool full_pipeline_ok = false;
    double full_min = 0.0;
    double full_avg = 0.0;
    if (!test_image_path().empty()) {
        std::vector<uint8_t> jpeg_data;
        if (read_file(test_image_path(), jpeg_data)) {
            uint32_t jpeg_w = 0;
            uint32_t jpeg_h = 0;
            if (!parse_jpeg_size(jpeg_data.data(), jpeg_data.size(), jpeg_w, jpeg_h)) {
                std::cout << "[WARN] JPEG size parse failed; skipping full pipeline stats" << std::endl;
            } else {
                HwJpegDecoder decoder;
                if (decoder.init(jpeg_w, jpeg_h)) {
                    std::vector<double> pipeline_times;
                    pipeline_times.reserve(iterations);
                    for (int i = 0; i < iterations; ++i) {
                        timer.start();
                        VIDEO_FRAME_INFO_S decoded = decoder.decode_sync(jpeg_data.data(), jpeg_data.size());
                        Frame frame(decoded, false);
                        processor.letterbox(frame, 640, 640, 114, nullptr);
                        decoder.release_frame(decoded);
                        pipeline_times.push_back(timer.elapsed_ms());
                    }
                    full_min = *std::min_element(pipeline_times.begin(), pipeline_times.end());
                    full_avg = std::accumulate(pipeline_times.begin(), pipeline_times.end(), 0.0) / iterations;
                    full_pipeline_ok = true;
                } else {
                    std::cout << "[WARN] HwJpegDecoder init failed; skipping full pipeline stats" << std::endl;
                }
            }
        } else {
            std::cout << "[WARN] Failed to read test image; skipping full pipeline stats" << std::endl;
        }
    } else {
        std::cout << "[WARN] No test image path; skipping full pipeline stats" << std::endl;
    }

    std::cout << "\n[VPSS Perf Summary] " << input_width << "x" << input_height
              << " -> 640x640 (letterbox)" << std::endl;
    std::cout << "  VPSS steady (min/avg): " << std::fixed << std::setprecision(1)
              << vpss_min << " / " << vpss_avg << " ms" << std::endl;
    if (full_pipeline_ok) {
        std::cout << "  VDEC+VPSS (min/avg):   " << full_min << " / " << full_avg << " ms" << std::endl;
    }
    if (cpu_resize_time > 0.0) {
        std::cout << "  Speedup vs CPU:        " << std::setprecision(1)
                  << (cpu_resize_time / vpss_avg) << "x" << std::endl;
    }

    EXPECT_GT(vpss_avg, 0.0);
}

#else

TEST(VpssPerformanceTest, Skipped) {
    GTEST_SKIP() << "USE_CVI_MPI not defined";
}

#endif
