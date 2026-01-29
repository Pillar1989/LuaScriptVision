#include <iostream>
#include <string>
#include <iomanip>
#include <fstream>
#include <csignal>
#include <opencv2/opencv.hpp>
#include "LuaIntf.h"

#include "modules/lua_cv.h"
#include "modules/lua_nn.h"
#include "modules/lua_utils.h"

#if defined(USE_CVI_TPU) && defined(USE_CVI_MPI)
#include "pipeline.h"
#include "modules/cv/mmf_context.h"
#endif

struct InferenceContext {
    lua_State* L = nullptr;
#ifdef USE_ONNX_RUNTIME
    std::unique_ptr<lua_nn::Session> session;
#endif
    LuaIntf::LuaRef preprocess;
    LuaIntf::LuaRef postprocess;
    LuaIntf::LuaRef preprocess_config;

    ~InferenceContext() {
        preprocess = LuaIntf::LuaRef();
        postprocess = LuaIntf::LuaRef();
        preprocess_config = LuaIntf::LuaRef();

        if (L) {
            lua_close(L);
            L = nullptr;
        }
    }
};

static bool ends_with(const std::string& str, const std::string& suffix) {
    if (suffix.size() > str.size()) return false;
    return str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

namespace {
volatile std::sig_atomic_t g_stop_requested = 0;

void handle_signal(int signal) {
    (void)signal;
    g_stop_requested = 1;
}

bool stop_requested() {
    return g_stop_requested != 0;
}
}  // namespace

void print_usage(const char* prog_name) {
    std::cout << "Usage:\n";
    std::cout << "  1. Inference mode: " << prog_name << " <script.lua> <model> <input> [options]\n";
    std::cout << "  2. Test mode:      " << prog_name << " <test_script.lua>\n";
    std::cout << "\n=== Inference Mode ===\n";
    std::cout << "Model: .onnx (CPU) or .cvimodel (TPU)\n";
    std::cout << "Input: image file (.jpg, .png) or 'camera'\n";
    std::cout << "\nOptions:\n";
    std::cout << "  show          - Display window during processing\n";
    std::cout << "  save=OUTPUT   - Save output image\n";
    std::cout << "  frames=N      - Run N frames (camera mode only, 0=infinite)\n";
    std::cout << "  fps           - Display FPS counter (camera mode only)\n";
    std::cout << "\nExamples:\n";
    std::cout << "  " << prog_name << " scripts/yolo11_detector.lua model.cvimodel zidane.jpg\n";
    std::cout << "  " << prog_name << " scripts/yolo11_detector.lua model.cvimodel camera frames=100 fps\n";
    std::cout << "  " << prog_name << " scripts/yolo11_detector.lua model.cvimodel camera frames=0 fps\n";
    std::cout << "\n=== Test Mode ===\n";
    std::cout << "  " << prog_name << " tests/run_all_tests.lua\n";
}

#ifdef USE_ONNX_RUNTIME
std::unique_ptr<InferenceContext> init_inference(const std::string& script_path,
                                                  const std::string& model_path) {
    auto ctx = std::make_unique<InferenceContext>();

    ctx->L = luaL_newstate();
    if (!ctx->L) throw std::runtime_error("Failed to create Lua state");
    luaL_openlibs(ctx->L);

    std::cout << "Registering modules...\n";
    lua_cv::register_module(ctx->L);
    lua_nn::register_module(ctx->L);
    lua_utils::register_module(ctx->L);

    std::cout << "Loading model: " << model_path << "\n";
    ctx->session = std::make_unique<lua_nn::Session>(model_path);

    std::cout << "Loading script: " << script_path << "\n";
    if (luaL_dofile(ctx->L, script_path.c_str()) != LUA_OK) {
        throw std::runtime_error("Failed to load script: " + std::string(lua_tostring(ctx->L, -1)));
    }

    LuaIntf::LuaRef model = LuaIntf::LuaRef::popFromStack(ctx->L);
    if (!model.isTable()) {
        throw std::runtime_error("Script must return a Model table");
    }

    ctx->preprocess = model["preprocess"];
    ctx->postprocess = model["postprocess"];
    ctx->preprocess_config = model["preprocess_config"];

    if (!ctx->postprocess.isFunction()) {
        throw std::runtime_error("Model must have postprocess function");
    }

    bool has_lua_preprocess = ctx->preprocess.isFunction();
    bool has_cpp_preprocess = ctx->preprocess_config.isValid() && ctx->preprocess_config.isTable();

    if (!has_lua_preprocess && !has_cpp_preprocess) {
        throw std::runtime_error("Model must have either preprocess function or preprocess_config table");
    }

    return ctx;
}

LuaIntf::LuaRef run_inference(InferenceContext* ctx, lua_cv::Image& img) {
    if (ctx->preprocess_config.isValid() && ctx->preprocess_config.isTable()) {
        std::string type = ctx->preprocess_config.get<std::string>("type");

        auto& registry = lua_cv::PreprocessRegistry::instance();

        if (registry.has(type)) {
            auto result = registry.run(type, img, ctx->L, ctx->preprocess_config);
            LuaIntf::LuaRef outputs = ctx->session->run(ctx->L, result.tensor);
            return ctx->postprocess.call<LuaIntf::LuaRef>(outputs, result.meta);
        } else {
            std::cerr << "Warning: Unknown preprocess type '" << type
                      << "', falling back to Lua\n";
        }
    }

    ctx->preprocess.pushToStack();
    LuaIntf::LuaRef::fromValue(ctx->L, img).pushToStack();

    if (lua_pcall(ctx->L, 1, 2, 0) != LUA_OK) {
        throw std::runtime_error("Preprocess error: " + std::string(lua_tostring(ctx->L, -1)));
    }

    LuaIntf::LuaRef meta = LuaIntf::LuaRef::popFromStack(ctx->L);
    LuaIntf::LuaRef input_tensor_ref = LuaIntf::LuaRef::popFromStack(ctx->L);

    lua_nn::Tensor input_tensor = input_tensor_ref.toValue<lua_nn::Tensor>();
    LuaIntf::LuaRef outputs = ctx->session->run(ctx->L, input_tensor);

    return ctx->postprocess.call<LuaIntf::LuaRef>(outputs, meta);
}
#endif  // USE_ONNX_RUNTIME

void print_results(LuaIntf::LuaRef& results) {
    int len = results.len();
    if (len == 0) {
        std::cout << "No results.\n";
        return;
    }

    LuaIntf::LuaRef first = results[1];
    if (first.has("x") && first.has("y")) {
        std::cout << "\n=== Detection Results ===\n";
        for (int i = 1; i <= len; ++i) {
            LuaIntf::LuaRef det = results[i];
            float x = det.get<float>("x");
            float y = det.get<float>("y");
            float w = det.get<float>("w");
            float h = det.get<float>("h");
            float score = det.get<float>("score");
            std::string label = det.has("label") ? det.get<std::string>("label") : "unknown";

            std::cout << "Box " << i << ": " << label << " "
                      << "(" << x << ", " << y << ", " << w << ", " << h << ") "
                      << "conf=" << score;
            if (det.has("keypoints")) std::cout << " +kpts";
            std::cout << "\n";
        }
        std::cout << "Total: " << len << " detections\n";
    } else if (first.has("class_id")) {
        std::cout << "\n=== Classification Results ===\n";
        for (int i = 1; i <= len; ++i) {
            LuaIntf::LuaRef item = results[i];
            int class_id = item.get<int>("class_id");
            float conf = item.get<float>("confidence");
            std::string label = item.has("label") ? " (" + item.get<std::string>("label") + ")" : "";
            std::cout << "Top " << i << ": Class " << class_id << label << " Conf=" << conf << "\n";
        }
    }
}

void draw_detections(cv::Mat& frame, LuaIntf::LuaRef& detections) {
    int len = detections.len();
    if (len == 0) return;

    LuaIntf::LuaRef first = detections[1];
    if (!first.has("x") || !first.has("y")) return;

    static std::vector<cv::Scalar> colors;
    if (colors.empty()) {
        for (int i = 0; i < 80; ++i) {
            int hue = (i * 40) % 180;
            colors.push_back(cv::Scalar(
                (hue < 60 ? 255 : (hue < 120 ? 255 - (hue-60)*4 : 0)),
                (hue < 60 ? hue*4 : (hue < 120 ? 255 : 255 - (hue-120)*4)),
                (hue < 60 ? 0 : (hue < 120 ? (hue-60)*4 : 255))
            ));
        }
    }

    for (int i = 1; i <= len; ++i) {
        LuaIntf::LuaRef det = detections[i];
        if (!det.isTable()) continue;

        float x = det.get<float>("x");
        float y = det.get<float>("y");
        float w = det.get<float>("w");
        float h = det.get<float>("h");
        float score = det.get<float>("score");
        int class_id = det.has("class_id") ? det.get<int>("class_id") : 0;
        std::string label = det.has("label") ? det.get<std::string>("label") : "unknown";

        if (det.has("mask")) {
            try {
                lua_nn::Tensor mask_tensor = det.get<lua_nn::Tensor>("mask");
                lua_nn::Tensor mask_2d = mask_tensor.squeeze(0);
                int mh = mask_2d.shape()[0];
                int mw = mask_2d.shape()[1];

                cv::Mat mask(mh, mw, CV_32F);
                std::memcpy(mask.data, mask_2d.data(), mh * mw * sizeof(float));

                cv::Mat mask_u8;
                mask.convertTo(mask_u8, CV_8U, 255);

                cv::Mat color_mask = cv::Mat::zeros(mh, mw, CV_8UC3);
                color_mask.setTo(colors[class_id % colors.size()], mask_u8);

                cv::addWeighted(frame, 0.7, color_mask, 0.3, 0, frame);
            } catch (...) {}
        }

        cv::rectangle(frame, cv::Rect(x, y, w, h), cv::Scalar(0, 255, 0), 2);

        std::string text = label + " " + std::to_string(score).substr(0, 4);
        int baseline = 0;
        cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        cv::rectangle(frame, cv::Point(x, y - textSize.height - 5),
                     cv::Point(x + textSize.width, y), cv::Scalar(0, 255, 0), -1);
        cv::putText(frame, text, cv::Point(x, y - 5),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

        if (det.has("keypoints")) {
            LuaIntf::LuaRef kpts = det.get<LuaIntf::LuaRef>("keypoints");
            int num_kpts = kpts.len();
            std::vector<cv::Point> points;
            for(int k=1; k<=num_kpts; ++k) {
                LuaIntf::LuaRef kp = kpts[k];
                float kx = kp.get<float>("x");
                float ky = kp.get<float>("y");
                float kv = kp.get<float>("v");
                if (kv > 0.5) {
                    cv::circle(frame, cv::Point(kx, ky), 3, cv::Scalar(0, 0, 255), -1);
                    points.push_back(cv::Point(kx, ky));
                } else {
                    points.push_back(cv::Point(-1, -1));
                }
            }

            const std::vector<std::pair<int, int>> skeleton = {
                {0,1}, {0,2}, {1,3}, {2,4}, {5,6}, {5,7}, {7,9},
                {6,8}, {8,10}, {11,12}, {11,13}, {13,15}, {12,14},
                {14,16}, {5,11}, {6,12}
            };

            for (const auto& limb : skeleton) {
                if (limb.first < points.size() && limb.second < points.size() &&
                    points[limb.first].x != -1 && points[limb.second].x != -1) {
                    cv::line(frame, points[limb.first], points[limb.second],
                            cv::Scalar(255, 0, 0), 2);
                }
            }
        }
    }
}

int run_test_script(const std::string& script_path) {
    std::cout << "\n========== Running Lua Test Script ==========\n";
    std::cout << "Script: " << script_path << "\n";

    std::ifstream file(script_path);
    if (!file.good()) {
        std::cerr << "Error: Script not found: " << script_path << "\n";
        return 1;
    }
    file.close();

    lua_State* L = luaL_newstate();
    if (!L) {
        std::cerr << "Error: Failed to create Lua state\n";
        return 1;
    }
    luaL_openlibs(L);

    std::cout << "Registering modules...\n";
    lua_cv::register_module(L);
    lua_nn::register_module(L);
    lua_utils::register_module(L);

    std::cout << "Executing...\n\n";
    int result = luaL_dofile(L, script_path.c_str());

    if (result != LUA_OK) {
        std::cerr << "\nError: " << lua_tostring(L, -1) << "\n";
        lua_pop(L, 1);
        lua_close(L);
        return 1;
    }

    lua_close(L);
    return 0;
}

#if defined(USE_CVI_TPU) && defined(USE_CVI_MPI)
int run_cvi_inference(const std::string& script_path,
                      const std::string& model_path,
                      const std::string& input_path,
                      const std::string& save_path,
                      int max_frames,
                      bool show_fps) {
    // Initialize MMF context (VB pools, VPSS mode)
    lua_cv::MmfContext::Config mmf_config;
    if (!lua_cv::MmfContext::build_default_config(&mmf_config)) {
        throw std::runtime_error("Failed to build MMF config");
    }
    mmf_config.force_reset = true;
    if (!lua_cv::MmfContext::instance().init(mmf_config)) {
        throw std::runtime_error("Failed to init MMF context");
    }

    // Create Lua state and register modules
    lua_State* L = luaL_newstate();
    if (!L) throw std::runtime_error("Failed to create Lua state");
    luaL_openlibs(L);

    lua_cv::register_module(L);
    lua_nn::register_module(L);
    lua_utils::register_module(L);

    // Load Lua script
    std::cout << "Loading script: " << script_path << "\n";
    if (luaL_dofile(L, script_path.c_str()) != LUA_OK) {
        std::string err = lua_tostring(L, -1);
        lua_close(L);
        throw std::runtime_error("Failed to load script: " + err);
    }

    LuaIntf::LuaRef model_table = LuaIntf::LuaRef::popFromStack(L);
    if (!model_table.isTable()) {
        lua_close(L);
        throw std::runtime_error("Script must return a Model table");
    }

    LuaIntf::LuaRef postprocess = model_table["postprocess"];
    if (!postprocess.isFunction()) {
        lua_close(L);
        throw std::runtime_error("Model must have postprocess function");
    }

    LuaIntf::LuaRef preprocess_config = model_table["preprocess_config"];

    // Use block scope to ensure pipeline is destroyed before lua_close.
    // CviPipeline holds LuaRef members that must be released while Lua state is alive.
    {
        pipeline::CviPipeline pipe;
        pipe.init(model_path, L, postprocess, preprocess_config);

        bool is_camera = (input_path == "camera");

        if (is_camera && max_frames != 1) {
            // Continuous camera inference mode
            std::cout << "\n========== Continuous Camera Inference ==========\n";
            std::cout << "Max frames: " << (max_frames == 0 ? "infinite" : std::to_string(max_frames)) << "\n";
            std::cout << "Press Ctrl+C to stop...\n\n";

            g_stop_requested = 0;
            std::signal(SIGINT, handle_signal);
            std::signal(SIGTERM, handle_signal);

            // Open camera once
            if (!pipe.open_camera()) {
                throw std::runtime_error("Failed to open camera");
            }

            int frame_count = 0;
            int skipped_count = 0;
            auto t_start = std::chrono::steady_clock::now();
            auto t_last_fps = t_start;
            int frames_since_fps = 0;
            int skipped_since_fps = 0;

            try {
                while (!stop_requested() && (max_frames == 0 || frame_count < max_frames)) {
                    // Read frame, infer when due (adaptive skip).
                    pipeline::InferenceResult result;
                    bool inferred = pipe.run_camera_frame_adaptive(&result);
                    if (inferred) {
                        frame_count++;
                        frames_since_fps++;
                    } else {
                        skipped_count++;
                        skipped_since_fps++;
                    }

                    // Calculate and display FPS
                    auto now = std::chrono::steady_clock::now();
                    auto fps_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                        now - t_last_fps).count();

                    if (show_fps && fps_elapsed >= 1000) {  // Update FPS every second
                        double fps = frames_since_fps * 1000.0 / fps_elapsed;
                        int total_window = frames_since_fps + skipped_since_fps;
                        double skip_ratio = (total_window > 0)
                            ? (skipped_since_fps * 100.0 / total_window)
                            : 0.0;
                        auto total_elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                            now - t_start).count();

                        std::cout << "\r[Frame " << std::setw(6) << frame_count
                                  << "] FPS: " << std::fixed << std::setprecision(1) << fps
                                  << " | Skip: " << std::setprecision(1) << skip_ratio << "%"
                                  << " | Time: " << total_elapsed << "s"
                                  << std::flush;

                        t_last_fps = now;
                        frames_since_fps = 0;
                        skipped_since_fps = 0;
                    }

                    // Print detections for inferred frames.
                    if (inferred && result.detections.isTable()) {
                        int num_dets = result.detections.len();
                        if (num_dets > 0) {
                            if (show_fps) {
                                std::cout << "\n";
                            }
                            std::cout << "[Frame " << frame_count << "]\n";
                            print_results(result.detections);
                        }
                    }

                    // Clear result LuaRef
                    result.detections = LuaIntf::LuaRef();
                }
            } catch (...) {
                pipe.close_camera();
                throw;
            }

            // Close camera
            pipe.close_camera();

            if (show_fps) {
                std::cout << "\n";  // Newline after FPS display
            }

            auto t_end = std::chrono::steady_clock::now();
            auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                t_end - t_start).count();

            std::cout << "\n========== Summary ==========\n";
            int total_frames = frame_count + skipped_count;
            double skip_ratio = (total_frames > 0)
                ? (skipped_count * 100.0 / total_frames)
                : 0.0;
            std::cout << "Total frames: " << frame_count << "\n";
            std::cout << "Skipped frames: " << skipped_count
                      << " (" << std::fixed << std::setprecision(1)
                      << skip_ratio << "%)\n";
            std::cout << "Total time: " << total_ms / 1000.0 << "s\n";
            std::cout << "Average FPS: " << std::fixed << std::setprecision(2)
                      << (frame_count * 1000.0 / total_ms) << "\n";
        } else {
            // Single-shot inference (image or single camera frame)
            pipeline::InferenceResult result;
            if (is_camera) {
                result = pipe.run_camera();
            } else {
                result = pipe.run_image(input_path);
            }

            // Print results
            print_results(result.detections);

            // Save output if requested
            if (!save_path.empty() && !is_camera) {
                cv::Mat draw_img = cv::imread(input_path);
                if (!draw_img.empty()) {
                    draw_detections(draw_img, result.detections);
                    cv::imwrite(save_path, draw_img);
                    std::cout << "\nResult saved to: " << save_path << "\n";
                }
            }

            // Clear InferenceResult's LuaRef
            result.detections = LuaIntf::LuaRef();
        }
    }
    // pipe destroyed here → LuaRefs released while L is still alive

    // Cleanup: clear remaining Lua refs before closing state
    postprocess = LuaIntf::LuaRef();
    preprocess_config = LuaIntf::LuaRef();
    model_table = LuaIntf::LuaRef();

    lua_close(L);

    lua_cv::MmfContext::instance().shutdown();

    return 0;
}
#endif  // USE_CVI_TPU && USE_CVI_MPI

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string first_arg = argv[1];

    // Test mode: single .lua argument
    if (argc == 2 && ends_with(first_arg, ".lua")) {
        return run_test_script(first_arg);
    }

    // Inference mode: script, model, input
    if (argc < 4) {
        print_usage(argv[0]);
        return 1;
    }

    std::string script_path = argv[1];
    std::string model_path = argv[2];
    std::string input_path = argv[3];

    std::string save_path;
    bool show_result = false;
    int max_frames = 1;  // Default: single frame
    bool show_fps = false;

    for (int i = 4; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "show") {
            show_result = true;
        } else if (arg.find("save=") == 0) {
            save_path = arg.substr(5);
        } else if (arg.find("frames=") == 0) {
            max_frames = std::stoi(arg.substr(7));
        } else if (arg == "fps") {
            show_fps = true;
        }
    }

    try {
#if defined(USE_CVI_TPU) && defined(USE_CVI_MPI)
        if (ends_with(model_path, ".cvimodel")) {
            return run_cvi_inference(script_path, model_path, input_path, save_path,
                                    max_frames, show_fps);
        }
#endif

#ifdef USE_ONNX_RUNTIME
        auto ctx = init_inference(script_path, model_path);

        std::cout << "Loading image: " << input_path << "\n";
        auto img = lua_cv::imread(input_path);
        std::cout << "Image size: " << img.width() << "x" << img.height() << "\n\n";

        LuaIntf::LuaRef detections = run_inference(ctx.get(), img);
        print_results(detections);

        if (show_result || !save_path.empty()) {
            cv::Mat draw_img = cv::imread(input_path);
            draw_detections(draw_img, detections);

            if (!save_path.empty()) {
                cv::imwrite(save_path, draw_img);
                std::cout << "\nResult saved to: " << save_path << "\n";
            }

#ifdef HAVE_OPENCV_VIDEOIO
            if (show_result) {
                cv::imshow("Result", draw_img);
                std::cout << "Press any key to exit...\n";
                cv::waitKey(0);
                cv::destroyAllWindows();
            }
#else
            if (show_result) {
                std::cerr << "Warning: Display mode not supported\n";
            }
#endif
        }
#else
        std::cerr << "Error: No inference backend available for " << model_path << "\n";
        return 1;
#endif  // USE_ONNX_RUNTIME

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
#if defined(USE_CVI_MPI)
        lua_cv::MmfContext::instance().shutdown();
#endif
        return 1;
    }

    return 0;
}
