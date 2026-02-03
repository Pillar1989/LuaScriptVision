/**
 * parallel_infer_stream - Parallel inference + RTSP streaming application
 *
 * Runs TPU inference and RTSP streaming concurrently on the reCamera SG2002.
 * Uses ParallelPipeline for dual-thread architecture:
 *   - Stream thread: VPSS Chn0 → VENC → RTSP
 *   - Inference thread: VPSS Chn1 → preprocess → TPU → Lua postprocess
 */

#include <chrono>
#include <csignal>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>
#include "LuaIntf.h"

#include "pipeline/parallel_pipeline.h"
#include "pipeline/cvi_pipeline.h"
#include "inference/cvi_session.h"
#include "modules/lua_cv.h"
#include "modules/lua_nn.h"
#include "modules/lua_utils.h"
#include "modules/cv/mmf_context.h"
#include "modules/cv/cvi_vpss_processor.h"
#include "modules/cv/cv_helpers.h"

namespace {

volatile std::sig_atomic_t g_stop_requested = 0;

void handle_signal(int) {
    g_stop_requested = 1;
}

bool stop_requested() {
    return g_stop_requested != 0;
}

double elapsed_ms(std::chrono::steady_clock::time_point start) {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(now - start).count();
}

struct AppConfig {
    std::string script_path;
    std::string model_path;
    std::string input = "camera";  // "camera" or image path
    int rtsp_port = 554;
    lua_cv::VencEncoder::CodecType codec = lua_cv::VencEncoder::CodecType::H264;
    int duration_sec = 0;  // 0 = run until Ctrl+C
    int stream_fps = 30;
    int stream_bitrate_kbps = 4000;  // Increased from 2000 for better quality
};

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog
              << " <script.lua> <model.cvimodel> [options]\n"
              << "\nOptions:\n"
              << "  --input SOURCE       Input source: 'camera' or image path (default: camera)\n"
              << "  --rtsp-port PORT     RTSP server port (default: 554)\n"
              << "  --codec h264|h265    Video codec (default: h264)\n"
              << "  --duration SECONDS   Run duration, 0=infinite (default: 0)\n"
              << "  --fps FPS            Stream FPS (default: 30)\n"
              << "  --bitrate KBPS       Stream bitrate in kbps (default: 4000)\n"
              << "\nExamples:\n"
              << "  " << prog << " scripts/yolo11_detector.lua "
              << "/userdata/Models/yolo11n.cvimodel --rtsp-port 8554\n"
              << "  " << prog << " scripts/yolo11_detector.lua "
              << "/userdata/Models/yolo11n.cvimodel --input /tmp/image.jpg\n"
              << "  " << prog << " scripts/yolo11_detector.lua "
              << "/userdata/Models/yolo11n.cvimodel --bitrate 8000\n";
}

bool parse_args(int argc, char* argv[], AppConfig* config) {
    if (argc < 3) {
        print_usage(argv[0]);
        return false;
    }

    config->script_path = argv[1];
    config->model_path = argv[2];

    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--input" && i + 1 < argc) {
            config->input = argv[++i];
        } else if (arg == "--rtsp-port" && i + 1 < argc) {
            config->rtsp_port = std::atoi(argv[++i]);
        } else if (arg == "--codec" && i + 1 < argc) {
            std::string codec_str = argv[++i];
            if (codec_str == "h265" || codec_str == "H265") {
                config->codec = lua_cv::VencEncoder::CodecType::H265;
            } else if (codec_str == "h264" || codec_str == "H264") {
                config->codec = lua_cv::VencEncoder::CodecType::H264;
            } else {
                std::cerr << "Unknown codec: " << codec_str << "\n";
                return false;
            }
        } else if (arg == "--duration" && i + 1 < argc) {
            config->duration_sec = std::atoi(argv[++i]);
        } else if (arg == "--fps" && i + 1 < argc) {
            config->stream_fps = std::atoi(argv[++i]);
        } else if (arg == "--bitrate" && i + 1 < argc) {
            config->stream_bitrate_kbps = std::atoi(argv[++i]);
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            print_usage(argv[0]);
            return false;
        }
    }

    return true;
}

// Inference context shared between main thread setup and inference callback.
// Only accessed from the inference thread after setup is complete.
struct InferContext {
    lua_State* L = nullptr;
    LuaIntf::LuaRef postprocess;
    LuaIntf::LuaRef preprocess_config;

    std::unique_ptr<inference::CviSession> session;
    lua_cv::CviVpssProcessor vpss_processor;

    // Model input spec
    uint32_t model_input_w = 0;
    uint32_t model_input_h = 0;
    PIXEL_FORMAT_E model_input_format = PIXEL_FORMAT_MAX;
    bool letterbox_enabled = true;
    uint8_t pad_value = 114;

    // Stats
    uint64_t infer_count = 0;
    uint64_t detect_count = 0;
    double total_preprocess_ms = 0.0;
    double total_infer_ms = 0.0;
    double total_postprocess_ms = 0.0;
};

}  // namespace

int main(int argc, char* argv[]) {
    AppConfig config;
    if (!parse_args(argc, argv, &config)) {
        return 1;
    }

    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);

    // 1. Initialize MMF context
    std::cout << "[App] Initializing MMF context...\n";
    lua_cv::MmfContext::Config mmf_config;
    if (!lua_cv::MmfContext::build_default_config(&mmf_config)) {
        std::cerr << "[App] Failed to build MMF config\n";
        return 1;
    }
    mmf_config.force_reset = true;
    if (!lua_cv::MmfContext::instance().init(mmf_config)) {
        std::cerr << "[App] Failed to init MMF context\n";
        return 1;
    }

    // 2. Initialize Lua state
    std::cout << "[App] Initializing Lua state...\n";
    lua_State* L = luaL_newstate();
    if (!L) {
        std::cerr << "[App] Failed to create Lua state\n";
        return 1;
    }
    luaL_openlibs(L);
    lua_cv::register_module(L);
    lua_nn::register_module(L);
    lua_utils::register_module(L);

    // 3. Load Lua script
    std::cout << "[App] Loading script: " << config.script_path << "\n";
    if (luaL_dofile(L, config.script_path.c_str()) != LUA_OK) {
        std::cerr << "[App] Failed to load script: "
                  << lua_tostring(L, -1) << "\n";
        lua_close(L);
        return 1;
    }

    LuaIntf::LuaRef model_table = LuaIntf::LuaRef::popFromStack(L);
    if (!model_table.isTable()) {
        std::cerr << "[App] Script must return a Model table\n";
        lua_close(L);
        return 1;
    }

    LuaIntf::LuaRef postprocess = model_table["postprocess"];
    if (!postprocess.isFunction()) {
        std::cerr << "[App] Model must have postprocess function\n";
        lua_close(L);
        return 1;
    }

    LuaIntf::LuaRef preprocess_config_lua = model_table["preprocess_config"];

    // 4. Load TPU model
    std::cout << "[App] Loading model: " << config.model_path << "\n";
    auto session = std::make_unique<inference::CviSession>(config.model_path);
    std::cout << "[App] Backend: " << session->backend_name()
              << " Inputs: " << session->input_count()
              << " Outputs: " << session->output_count() << "\n";

    // 5. Build preprocess plan from model input spec
    InferContext ctx;
    ctx.L = L;
    ctx.postprocess = postprocess;
    ctx.preprocess_config = preprocess_config_lua;
    ctx.session = std::move(session);

    if (ctx.session->supports_vb_input()) {
        auto vb_spec = ctx.session->get_vb_input_spec();
        ctx.model_input_w = vb_spec.width;
        ctx.model_input_h = vb_spec.height;
        ctx.model_input_format = vb_spec.pixel_format;
    } else {
        auto input_shape = ctx.session->get_input_shape(0);
        if (input_shape.size() < 4) {
            std::cerr << "[App] Model input shape must be 4D\n";
            lua_close(L);
            return 1;
        }
        ctx.model_input_h = static_cast<uint32_t>(input_shape[2]);
        ctx.model_input_w = static_cast<uint32_t>(input_shape[3]);
        ctx.model_input_format = PIXEL_FORMAT_RGB_888_PLANAR;
    }

    if (preprocess_config_lua.isValid() && preprocess_config_lua.isTable()) {
        if (preprocess_config_lua.has("fill_value")) {
            ctx.pad_value = static_cast<uint8_t>(
                preprocess_config_lua.get<int>("fill_value"));
        }
    }

    std::cout << "[App] Preprocess: " << ctx.model_input_w << "x"
              << ctx.model_input_h << " format=" << ctx.model_input_format
              << " letterbox=" << (ctx.letterbox_enabled ? "yes" : "no")
              << " pad=" << static_cast<int>(ctx.pad_value) << "\n";

    // 6. Check input type and choose pipeline
    bool is_camera_mode = (config.input == "camera");

    if (!is_camera_mode) {
        // Image mode: use CviPipeline
        std::cout << "[App] Image mode: " << config.input << "\n";

        pipeline::CviPipeline pipeline;
        pipeline.init(config.model_path, L, postprocess, preprocess_config_lua);

        auto result = pipeline.run_image(config.input);

        std::cout << "\n=== Timings ===\n"
                  << "  Input: " << result.timings.input_ms << " ms\n"
                  << "  Preprocess: " << result.timings.preprocess_ms << " ms\n"
                  << "  Inference: " << result.timings.inference_ms << " ms\n"
                  << "  Postprocess: " << result.timings.postprocess_ms << " ms\n"
                  << "  Total: " << result.timings.total_ms << " ms\n";

        // Cleanup
        ctx.postprocess = LuaIntf::LuaRef();
        ctx.preprocess_config = LuaIntf::LuaRef();
        ctx.session.reset();
        ctx.L = nullptr;
        lua_close(L);
        lua_cv::MmfContext::instance().shutdown();
        ctx.vpss_processor.destroy_group();

        std::cout << "[App] Done" << std::endl;
        std::_Exit(0);
        return 0;
    }

    // Camera mode: use ParallelPipeline
    std::cout << "[App] Camera mode with streaming\n";

    // 7. Configure and start ParallelPipeline
    lua_cv::ParallelPipeline::Config pipe_config;
    pipe_config.stream_fps = config.stream_fps;
    pipe_config.stream_bitrate_kbps = config.stream_bitrate_kbps;
    pipe_config.stream_codec = config.codec;
    pipe_config.rtsp_port = config.rtsp_port;
    pipe_config.rtsp_session = "live";
    pipe_config.camera_ready_timeout_ms = 10000;

    {
        lua_cv::ParallelPipeline pipeline(pipe_config);

        // Inference callback: preprocess → TPU → Lua postprocess
    auto callback = [&ctx](const lua_cv::Frame& frame,
                           lua_cv::ParallelPipeline::InferenceResult* result) -> bool {
        if (frame.width() == 0 || frame.height() == 0) {
            return false;
        }

        lua_cv::PixelFormat output_pf =
            lua_cv::from_cvi_pixel_format(ctx.model_input_format);

        // Check if frame already matches model input (size + format).
        // When the camera outputs the exact format the model needs,
        // we skip all VPSS preprocessing and go directly to TPU zero-copy.
        bool frame_matches = (static_cast<uint32_t>(frame.width()) == ctx.model_input_w &&
                              static_cast<uint32_t>(frame.height()) == ctx.model_input_h &&
                              frame.pixel_format() == output_pf &&
                              frame.has_physical_addr());

        auto t_pre = std::chrono::steady_clock::now();
        lua_cv::CviVpssProcessor::LetterboxMeta lb_meta{};
        lua_cv::Frame input_copy;

        if (frame_matches) {
            // Direct zero-copy path: no VPSS preprocessing needed.
            lb_meta.scale = 1.0f;
            lb_meta.pad_x = 0;
            lb_meta.pad_y = 0;
            lb_meta.ori_w = frame.width();
            lb_meta.ori_h = frame.height();
        } else {
            // VPSS preprocessing path: letterbox + format conversion
            input_copy = frame.clone();
            if (ctx.letterbox_enabled) {
                ctx.vpss_processor.letterbox(
                    input_copy,
                    static_cast<int>(ctx.model_input_w),
                    static_cast<int>(ctx.model_input_h),
                    ctx.pad_value,
                    &lb_meta,
                    output_pf);
            } else {
                ctx.vpss_processor.resize(
                    input_copy,
                    static_cast<int>(ctx.model_input_w),
                    static_cast<int>(ctx.model_input_h));
                lb_meta.scale = static_cast<float>(ctx.model_input_w) / frame.width();
                lb_meta.pad_x = 0;
                lb_meta.pad_y = 0;
                lb_meta.ori_w = frame.width();
                lb_meta.ori_h = frame.height();
            }
            if (input_copy.pixel_format() != output_pf) {
                ctx.vpss_processor.convert_format(input_copy, output_pf);
            }
        }
        double pre_ms = elapsed_ms(t_pre);

        // TPU inference via zero-copy
        auto t_infer = std::chrono::steady_clock::now();
        const lua_cv::Frame& infer_frame = frame_matches ? frame : input_copy;

        std::string reason;
        if (!lua_cv::cv_helpers::can_zero_copy(
                infer_frame, ctx.model_input_format,
                ctx.model_input_w, ctx.model_input_h, &reason)) {
            std::cerr << "[Infer] Zero-copy not available: " << reason << "\n";
            if (!frame_matches) input_copy.release();
            return false;
        }

        auto vb_mem = infer_frame.as_vb_memory();
        if (!vb_mem) {
            std::cerr << "[Infer] Failed to get VB memory from frame\n";
            if (!frame_matches) input_copy.release();
            return false;
        }

        std::vector<std::vector<float>> outputs;
        std::vector<std::vector<int64_t>> output_shapes;
        ctx.session->run_vb(vb_mem, &outputs, &output_shapes);
        double infer_ms = elapsed_ms(t_infer);

        // Package outputs as Lua table
        auto t_post = std::chrono::steady_clock::now();
        LuaIntf::LuaRef output_table = LuaIntf::LuaRef::createTable(ctx.L);
        const auto& output_names = ctx.session->get_output_names();
        for (size_t i = 0; i < outputs.size(); ++i) {
            lua_nn::Tensor tensor(std::move(outputs[i]), output_shapes[i]);
            std::string generic_name = "output" + std::to_string(i);
            output_table[generic_name] = tensor;
            if (i < output_names.size() && output_names[i] != generic_name) {
                output_table[output_names[i]] = tensor;
            }
        }

        // Build meta table
        LuaIntf::LuaRef meta = LuaIntf::LuaRef::createTable(ctx.L);
        meta["scale"] = lb_meta.scale;
        meta["pad_x"] = lb_meta.pad_x;
        meta["pad_y"] = lb_meta.pad_y;
        meta["ori_w"] = lb_meta.ori_w;
        meta["ori_h"] = lb_meta.ori_h;
        meta["input_w"] = static_cast<int>(ctx.model_input_w);
        meta["input_h"] = static_cast<int>(ctx.model_input_h);
        meta["output_count"] = ctx.session->output_count();

        // Call Lua postprocess
        LuaIntf::LuaRef detections =
            ctx.postprocess.call<LuaIntf::LuaRef>(output_table, meta);
        double post_ms = elapsed_ms(t_post);

        int num_dets = detections.isTable() ? detections.len() : 0;

        // Update stats
        ctx.infer_count++;
        ctx.detect_count += num_dets;
        ctx.total_preprocess_ms += pre_ms;
        ctx.total_infer_ms += infer_ms;
        ctx.total_postprocess_ms += post_ms;

        if (!frame_matches) input_copy.release();
        return true;
    };

    std::cout << "[App] Starting pipeline...\n";
    if (!pipeline.start(callback)) {
        std::cerr << "[App] Failed to start pipeline\n";
        ctx.postprocess = LuaIntf::LuaRef();
        ctx.preprocess_config = LuaIntf::LuaRef();
        lua_close(L);
        lua_cv::MmfContext::instance().shutdown();
        return 1;
    }

    std::string codec_name = (config.codec == lua_cv::VencEncoder::CodecType::H265)
                             ? "H.265" : "H.264";
    std::cout << "[App] Pipeline running\n"
              << "  RTSP: rtsp://192.168.42.1:" << config.rtsp_port << "/live\n"
              << "  Codec: " << codec_name << "\n"
              << "  Stream: " << config.stream_fps << " FPS, "
              << config.stream_bitrate_kbps << " kbps\n"
              << "  Duration: " << (config.duration_sec == 0 ? "infinite" :
                 std::to_string(config.duration_sec) + "s") << "\n"
              << "  Press Ctrl+C to stop\n\n";

    // 7. Main loop: print stats periodically
    auto t_start = std::chrono::steady_clock::now();
    auto t_last_stats = t_start;
    uint64_t last_infer_count = 0;

    while (!stop_requested()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // Check duration
        if (config.duration_sec > 0) {
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - t_start).count();
            if (elapsed >= config.duration_sec) {
                std::cout << "\n[App] Duration reached (" << config.duration_sec
                          << "s), stopping...\n";
                break;
            }
        }

        // Print stats every 2 seconds
        auto now = std::chrono::steady_clock::now();
        auto stats_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - t_last_stats).count();
        if (stats_elapsed >= 2000) {
            auto stats = pipeline.get_stats();
            uint64_t new_infers = ctx.infer_count - last_infer_count;
            double infer_fps = (stats_elapsed > 0)
                ? (new_infers * 1000.0 / stats_elapsed) : 0.0;

            double avg_pre = (ctx.infer_count > 0)
                ? (ctx.total_preprocess_ms / ctx.infer_count) : 0.0;
            double avg_infer = (ctx.infer_count > 0)
                ? (ctx.total_infer_ms / ctx.infer_count) : 0.0;
            double avg_post = (ctx.infer_count > 0)
                ? (ctx.total_postprocess_ms / ctx.infer_count) : 0.0;

            std::cout << "\r[Stats] infer=" << std::fixed << std::setprecision(1)
                      << infer_fps << " fps"
                      << " stream=" << stats.stream_fps << " fps"
                      << " | pre=" << std::setprecision(1) << avg_pre << "ms"
                      << " tpu=" << avg_infer << "ms"
                      << " post=" << avg_post << "ms"
                      << " | dets=" << ctx.detect_count
                      << " drops=" << stats.dropped_results
                      << "      " << std::flush;

            last_infer_count = ctx.infer_count;
            t_last_stats = now;
        }
    }

    // 8. Shutdown - get stats before stopping pipeline
    std::cout << "\n[App] Stopping pipeline...\n";

    // Get final stats BEFORE stopping, as stop() will reset member variables
    auto final_stats = pipeline.get_stats();

    pipeline.stop();

    // Print final stats
    auto total_sec = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_start).count();
    std::cout << "\n========== Final Stats ==========\n"
              << "  Runtime: " << std::fixed << std::setprecision(1)
              << total_sec << " s\n"
              << "  Inferred frames: " << ctx.infer_count << "\n"
              << "  Total detections: " << ctx.detect_count << "\n";
    if (ctx.infer_count > 0) {
        std::cout << "  Avg preprocess: " << std::setprecision(2)
                  << (ctx.total_preprocess_ms / ctx.infer_count) << " ms\n"
                  << "  Avg inference: "
                  << (ctx.total_infer_ms / ctx.infer_count) << " ms\n"
                  << "  Avg postprocess: "
                  << (ctx.total_postprocess_ms / ctx.infer_count) << " ms\n"
                  << "  Avg total/frame: "
                  << ((ctx.total_preprocess_ms + ctx.total_infer_ms +
                       ctx.total_postprocess_ms) / ctx.infer_count) << " ms\n"
                  << "  Effective FPS: " << std::setprecision(1)
                  << (ctx.infer_count / total_sec) << "\n";
    }
    std::cout << "  Stream frames: " << final_stats.stream_frames << "\n"
              << "  Stream FPS: " << std::setprecision(1)
              << final_stats.stream_fps << "\n"
              << "  Dropped results: " << final_stats.dropped_results << "\n"
              << "=================================\n";

    // End of pipeline scope - pipeline will be destroyed here
    }

    // Release Lua references before closing state
    ctx.postprocess = LuaIntf::LuaRef();
    ctx.preprocess_config = LuaIntf::LuaRef();

    ctx.session.reset();  // Release TPU session before MMF shutdown
    ctx.L = nullptr;
    lua_close(L);

    // Shutdown MMF and explicitly destroy VPSS resources
    lua_cv::MmfContext::instance().shutdown();

    // Explicitly destroy vpss_processor AFTER MMF shutdown
    // This prevents its destructor from accessing destroyed VPSS resources
    ctx.vpss_processor.destroy_group();

    std::cout << "[App] Clean shutdown" << std::endl;

    // Use std::_Exit(0) to bypass destructors and avoid potential segfault
    // during C++ runtime cleanup. All resources have been explicitly cleaned up.
    std::_Exit(0);
    return 0;  // Never reached
}
