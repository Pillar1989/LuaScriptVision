#include <iostream>
#include <string>
#include <iomanip>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "LuaIntf.h"

// 模块头文件
#include "modules/lua_cv.h"
#include "modules/lua_nn.h"
#include "modules/lua_utils.h"

// 推理上下文
struct InferenceContext {
    lua_State* L = nullptr;
#ifdef USE_ONNX_RUNTIME
    std::unique_ptr<lua_nn::Session> session;
#endif
    LuaIntf::LuaRef preprocess;
    LuaIntf::LuaRef postprocess;
    LuaIntf::LuaRef preprocess_config;  // 新增：C++预处理配置

    ~InferenceContext() {
        // 重要：先清空 LuaRef 对象，再关闭 Lua 状态
        // 否则 LuaRef 析构时会访问已关闭的 Lua 状态
        preprocess = LuaIntf::LuaRef();
        postprocess = LuaIntf::LuaRef();
        preprocess_config = LuaIntf::LuaRef();

        if (L) {
            lua_close(L);
            L = nullptr;
        }
    }
};

void print_usage(const char* prog_name) {
    std::cout << "Usage:\n";
    std::cout << "  1. Inference mode: " << prog_name << " <script.lua> <model.onnx> <input> [options]\n";
    std::cout << "  2. Test mode:      " << prog_name << " <test_script.lua>\n";
    std::cout << "\n=== Inference Mode ===\n";
    std::cout << "Input: image file (.jpg, .png)\n";
    std::cout << "\nOptions:\n";
    std::cout << "  show          - Display window during processing\n";
    std::cout << "  save=OUTPUT   - Save output image\n";
    std::cout << "\nInference Examples:\n";
    std::cout << "  " << prog_name << " scripts/yolo11_detector.lua models/yolo11n.onnx images/zidane.jpg show\n";
    std::cout << "\n=== Test Mode ===\n";
    std::cout << "Test Examples:\n";
    std::cout << "  " << prog_name << " tests/run_all_tests.lua\n";
    std::cout << "  " << prog_name << " tests/test_basic.lua\n";
}

#ifdef USE_ONNX_RUNTIME
// 初始化推理上下文
std::unique_ptr<InferenceContext> init_inference(const std::string& script_path,
                                                  const std::string& model_path) {
    auto ctx = std::make_unique<InferenceContext>();

    // 初始化Lua
    ctx->L = luaL_newstate();
    if (!ctx->L) throw std::runtime_error("Failed to create Lua state");
    luaL_openlibs(ctx->L);

    // 注册C++模块
    std::cout << "Registering modules...\n";
    lua_cv::register_module(ctx->L);
    lua_nn::register_module(ctx->L);
    lua_utils::register_module(ctx->L);

    // 加载ONNX模型
    std::cout << "Loading model: " << model_path << "\n";
    ctx->session = std::make_unique<lua_nn::Session>(model_path);

    // 加载Lua脚本
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
    ctx->preprocess_config = model["preprocess_config"];  // 新增：读取C++预处理配置

    // postprocess必须存在
    if (!ctx->postprocess.isFunction()) {
        throw std::runtime_error("Model must have postprocess function");
    }

    // preprocess可以是函数(Lua模式)或通过preprocess_config(C++模式)提供
    bool has_lua_preprocess = ctx->preprocess.isFunction();
    bool has_cpp_preprocess = ctx->preprocess_config.isValid() && ctx->preprocess_config.isTable();

    if (!has_lua_preprocess && !has_cpp_preprocess) {
        throw std::runtime_error("Model must have either preprocess function or preprocess_config table");
    }

    return ctx;
}

// 执行推理
LuaIntf::LuaRef run_inference(InferenceContext* ctx, lua_cv::Image& img) {
    // 尝试使用C++预处理
    if (ctx->preprocess_config.isValid() && ctx->preprocess_config.isTable()) {
        std::string type = ctx->preprocess_config.get<std::string>("type");

        auto& registry = lua_cv::PreprocessRegistry::instance();

        if (registry.has(type)) {
            // 执行C++预处理
            auto result = registry.run(type, img, ctx->L, ctx->preprocess_config);

            // 推理
            LuaIntf::LuaRef outputs = ctx->session->run(ctx->L, result.tensor);

            // 后处理
            return ctx->postprocess.call<LuaIntf::LuaRef>(outputs, result.meta);
        } else {
            std::cerr << "Warning: Unknown preprocess type '" << type
                      << "', falling back to Lua\n";
        }
    }

    // 回退到Lua预处理
    ctx->preprocess.pushToStack();
    LuaIntf::LuaRef::fromValue(ctx->L, img).pushToStack();

    if (lua_pcall(ctx->L, 1, 2, 0) != LUA_OK) {
        throw std::runtime_error("Preprocess error: " + std::string(lua_tostring(ctx->L, -1)));
    }

    LuaIntf::LuaRef meta = LuaIntf::LuaRef::popFromStack(ctx->L);
    LuaIntf::LuaRef input_tensor_ref = LuaIntf::LuaRef::popFromStack(ctx->L);

    // 推理
    lua_nn::Tensor input_tensor = input_tensor_ref.toValue<lua_nn::Tensor>();
    LuaIntf::LuaRef outputs = ctx->session->run(ctx->L, input_tensor);

    // 后处理
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

    // 每个类别生成确定性颜色
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

        // 绘制 mask
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

        // 绘制边界框
        cv::rectangle(frame, cv::Rect(x, y, w, h), cv::Scalar(0, 255, 0), 2);

        // 绘制标签
        std::string text = label + " " + std::to_string(score).substr(0, 4);
        int baseline = 0;
        cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        cv::rectangle(frame, cv::Point(x, y - textSize.height - 5),
                     cv::Point(x + textSize.width, y), cv::Scalar(0, 255, 0), -1);
        cv::putText(frame, text, cv::Point(x, y - 5),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

        // 绘制关键点
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

// 运行测试脚本
int run_test_script(const std::string& script_path) {
    std::cout << "\n========== 运行Lua测试脚本 ==========\n";
    std::cout << "加载脚本: " << script_path << "\n";

    // 检查文件是否存在
    std::ifstream file(script_path);
    if (!file.good()) {
        std::cerr << "错误: 找不到脚本文件: " << script_path << "\n";
        return 1;
    }
    file.close();

    // 创建Lua状态机
    lua_State* L = luaL_newstate();
    if (!L) {
        std::cerr << "错误: 无法创建Lua状态机\n";
        return 1;
    }
    luaL_openlibs(L);

    // 注册所有模块
    std::cout << "注册模块...\n";
    lua_cv::register_module(L);
    lua_nn::register_module(L);
    lua_utils::register_module(L);

    // 加载并运行脚本
    std::cout << "执行测试脚本...\n\n";
    int result = luaL_dofile(L, script_path.c_str());

    if (result != LUA_OK) {
        std::cerr << "\n错误: 脚本执行失败: " << lua_tostring(L, -1) << "\n";
        lua_pop(L, 1);
        lua_close(L);
        return 1;
    }

    lua_close(L);
    std::cout << "\n✓ Lua脚本测试完成!\n";
    return 0;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string first_arg = argv[1];

    // 判断是测试模式还是推理模式
    // 测试模式：只有一个参数，且是 .lua 文件
    if (argc == 2 && first_arg.size() > 4 &&
        first_arg.substr(first_arg.size() - 4) == ".lua") {
        return run_test_script(first_arg);
    }

#ifdef USE_ONNX_RUNTIME
    // 推理模式：需要至少 3 个参数（script, model, input）
    if (argc < 4) {
        print_usage(argv[0]);
        return 1;
    }

    std::string script_path = argv[1];
    std::string model_path = argv[2];
    std::string input_path = argv[3];

    bool show_result = false;
    std::string save_path = "";

    // 解析可选参数
    for (int i = 4; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "show") {
            show_result = true;
        } else if (arg.find("save=") == 0) {
            save_path = arg.substr(5);
        }
    }

    try {
        // 初始化推理上下文（共用）
        auto ctx = init_inference(script_path, model_path);

        // ========== 图片推理模式 ==========
        std::cout << "Loading image: " << input_path << "\n";
        auto img = lua_cv::imread(input_path);
        std::cout << "Image size: " << img.width() << "x" << img.height() << "\n\n";

        std::cout << "Preprocessing...\n";
        std::cout << "Running inference...\n";
        std::cout << "Postprocessing...\n";

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
                std::cerr << "Warning: Display mode not supported (OpenCV built without videoio/highgui)\n";
            }
#endif
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
#else
    // Inference mode requires ONNX Runtime
    std::cerr << "Error: Inference mode is not available.\n";
    std::cerr << "This build was compiled without ONNX Runtime support.\n";
    std::cerr << "Only test mode is available: " << argv[0] << " <test_script.lua>\n";
    return 1;
#endif  // USE_ONNX_RUNTIME

    return 0;
}
