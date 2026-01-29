#include "pipeline.h"

#if defined(USE_CVI_TPU) && defined(USE_CVI_MPI)

#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>

#include "modules/cv/image_source.h"
#include "modules/cv/cv_helpers.h"
#include "modules/cv/mmf_context.h"

#ifdef USE_CVI_CAMERA
#include "modules/cv/camera_source.h"
#endif

namespace pipeline {

namespace {

double elapsed_ms(std::chrono::steady_clock::time_point start) {
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

constexpr double kSkipEmaAlpha = 0.2;
constexpr int kSkipBudgetMax = 16;
constexpr double kDefaultFrameIntervalMs = 33.333;

int compute_skip_budget(double infer_ms, double camera_fps) {
    if (infer_ms <= 0.0 || camera_fps <= 0.0) {
        return 0;
    }
    double frame_interval_ms = 1000.0 / camera_fps;
    if (frame_interval_ms <= 0.0) {
        return 0;
    }
    int budget = static_cast<int>(std::ceil(infer_ms / frame_interval_ms)) - 1;
    if (budget < 0) {
        budget = 0;
    } else if (budget > kSkipBudgetMax) {
        budget = kSkipBudgetMax;
    }
    return budget;
}

double clamp_value(double value, double min_value, double max_value) {
    if (value < min_value) {
        return min_value;
    }
    if (value > max_value) {
        return max_value;
    }
    return value;
}

int compute_nobuf_threshold(double infer_ms, double frame_interval_ms) {
    if (infer_ms <= 0.0 || frame_interval_ms <= 0.0) {
        return 3;
    }
    double ratio = infer_ms / frame_interval_ms;
    if (ratio >= 2.0) {
        return 1;
    }
    if (ratio >= 1.3) {
        return 2;
    }
    return 3;
}

double compute_nobuf_cooldown_ms(int streak, int threshold,
                                 double infer_ms, double frame_interval_ms) {
    double base_ms = (infer_ms > 0.0) ? infer_ms * 1.2 : frame_interval_ms * 2.0;
    base_ms = clamp_value(base_ms, frame_interval_ms, 250.0);
    int exponent = streak - threshold;
    if (exponent < 0) {
        exponent = 0;
    } else if (exponent > 4) {
        exponent = 4;
    }
    double cooldown_ms = base_ms * std::pow(2.0, exponent);
    return clamp_value(cooldown_ms, frame_interval_ms, 1000.0);
}

}  // namespace

CviPipeline::CviPipeline() = default;
CviPipeline::~CviPipeline() = default;

void CviPipeline::init(const std::string& model_path,
                       lua_State* L,
                       const LuaIntf::LuaRef& postprocess,
                       const LuaIntf::LuaRef& preprocess_config) {
    L_ = L;
    postprocess_ = postprocess;
    preprocess_config_lua_ = preprocess_config;

    session_ = std::make_unique<inference::CviSession>(model_path);
    std::cout << "[Pipeline] Model loaded: " << model_path << "\n";
    std::cout << "[Pipeline] Backend: " << session_->backend_name() << "\n";
    std::cout << "[Pipeline] Inputs: " << session_->input_count()
              << " Outputs: " << session_->output_count() << "\n";

    // Log output tensor details
    const auto& out_names = session_->get_output_names();
    for (int32_t i = 0; i < session_->output_count(); ++i) {
        auto shape = session_->get_output_shape(static_cast<size_t>(i));
        std::string shape_str;
        for (size_t j = 0; j < shape.size(); ++j) {
            if (j > 0) shape_str += "x";
            shape_str += std::to_string(shape[j]);
        }
        std::string name = (static_cast<size_t>(i) < out_names.size())
                           ? out_names[static_cast<size_t>(i)] : "?";
        std::cout << "[Pipeline]   output" << i << " \"" << name
                  << "\" shape=" << shape_str << "\n";
    }
    std::cout << "[Pipeline] Primary output index: "
              << session_->primary_output_index() << "\n";

    build_preprocess_plan();
}

void CviPipeline::build_preprocess_plan() {
    // Get model input dimensions and pixel format from VB spec if available.
    // VB spec correctly handles both NCHW and NHWC layouts.
    if (session_->supports_vb_input()) {
        auto vb_spec = session_->get_vb_input_spec();
        model_input_w_ = vb_spec.width;
        model_input_h_ = vb_spec.height;
        model_input_format_ = vb_spec.pixel_format;
    } else {
        // Fallback: assume NCHW layout
        auto input_shape = session_->get_input_shape(0);
        if (input_shape.size() < 4) {
            throw std::runtime_error("[Pipeline] Model input shape must be 4D");
        }
        model_input_h_ = static_cast<uint32_t>(input_shape[2]);
        model_input_w_ = static_cast<uint32_t>(input_shape[3]);
        model_input_format_ = PIXEL_FORMAT_RGB_888_PLANAR;
    }

    // Parse preprocess_config from Lua if available
    if (preprocess_config_lua_.isValid() && preprocess_config_lua_.isTable()) {
        if (preprocess_config_lua_.has("fill_value")) {
            pad_value_ = static_cast<uint8_t>(
                preprocess_config_lua_.get<int>("fill_value"));
        }
        // "letterbox" is the default and only supported type for VPSS
        if (preprocess_config_lua_.has("type")) {
            std::string type = preprocess_config_lua_.get<std::string>("type");
            if (type != "letterbox") {
                throw std::runtime_error(
                    "[Pipeline] Unsupported preprocess type: " + type +
                    " (only 'letterbox' is supported on VPSS)");
            }
        }
    }

    std::cout << "[Pipeline] Preprocess plan: "
              << model_input_w_ << "x" << model_input_h_
              << " format=" << model_input_format_
              << " letterbox=" << (letterbox_enabled_ ? "yes" : "no")
              << " pad=" << static_cast<int>(pad_value_) << "\n";
    std::cout << "[Pipeline] Preprocess mode: VPSS hardware (letterbox, no CPU fallback)\n";
}

lua_cv::Frame CviPipeline::preprocess_frame(lua_cv::Frame& input_frame,
                                            LuaIntf::LuaRef* meta_out) {
    int ori_w = input_frame.width();
    int ori_h = input_frame.height();

    // Determine target pixel format from model input spec
    lua_cv::PixelFormat output_pf = lua_cv::from_cvi_pixel_format(model_input_format_);

    // Use VPSS letterbox which handles resize + pad + format conversion in one pass
    lua_cv::CviVpssProcessor::LetterboxMeta lb_meta{};
    lua_cv::Frame result;

    if (letterbox_enabled_) {
        // Move the input frame to avoid expensive CVI→Mat copy in Frame::clone().
        // The CVI frame is consumed by VPSS (released after processing).
        result = std::move(input_frame);
        vpss_processor_.letterbox(result,
                                  static_cast<int>(model_input_w_),
                                  static_cast<int>(model_input_h_),
                                  pad_value_,
                                  &lb_meta,
                                  output_pf);
    } else {
        result = std::move(input_frame);
        vpss_processor_.resize(result,
                               static_cast<int>(model_input_w_),
                               static_cast<int>(model_input_h_));
        lb_meta.scale = static_cast<float>(model_input_w_) / ori_w;
        lb_meta.pad_x = 0;
        lb_meta.pad_y = 0;
        lb_meta.ori_w = ori_w;
        lb_meta.ori_h = ori_h;
    }

    // Format conversion needed only for resize path (letterbox already outputs correct format)
    if (result.pixel_format() != output_pf) {
        vpss_processor_.convert_format(result, output_pf);
    }

    // Build meta table for Lua postprocess
    if (meta_out) {
        *meta_out = LuaIntf::LuaRef::createTable(L_);
        (*meta_out)["scale"] = lb_meta.scale;
        (*meta_out)["pad_x"] = lb_meta.pad_x;
        (*meta_out)["pad_y"] = lb_meta.pad_y;
        (*meta_out)["ori_w"] = lb_meta.ori_w;
        (*meta_out)["ori_h"] = lb_meta.ori_h;
        (*meta_out)["input_w"] = static_cast<int>(model_input_w_);
        (*meta_out)["input_h"] = static_cast<int>(model_input_h_);
    }

    return result;
}

LuaIntf::LuaRef CviPipeline::run_tpu(const lua_cv::Frame& preprocessed_frame) {
    // Verify zero-copy conditions
    std::string reason;
    bool zero_copy = lua_cv::cv_helpers::can_zero_copy(
        preprocessed_frame,
        model_input_format_,
        model_input_w_,
        model_input_h_,
        &reason);

    if (!zero_copy) {
        throw std::runtime_error(
            "[Pipeline] Zero-copy not available: " + reason);
    }

    uint64_t paddr = preprocessed_frame.physical_addr();
    auto vb_mem = preprocessed_frame.as_vb_memory();
    if (!vb_mem) {
        throw std::runtime_error("[Pipeline] Failed to get VB memory from frame");
    }

    // Run TPU inference via physical address (zero-copy)
    std::vector<std::vector<float>> outputs;
    std::vector<std::vector<int64_t>> output_shapes;
    session_->run_vb(vb_mem, &outputs, &output_shapes);

    // Package outputs as Lua table of tensors
    // Generic output format: output0, output1, ... + model original names
    LuaIntf::LuaRef output_table = LuaIntf::LuaRef::createTable(L_);

    const auto& output_names = session_->get_output_names();
    for (size_t i = 0; i < outputs.size(); ++i) {
        lua_nn::Tensor tensor(std::move(outputs[i]), output_shapes[i]);

        std::string generic_name = "output" + std::to_string(i);
        output_table[generic_name] = tensor;

        if (i < output_names.size() && output_names[i] != generic_name) {
            output_table[output_names[i]] = tensor;
        }
    }

    return output_table;
}

InferenceResult CviPipeline::run_image(const std::string& image_path) {
    InferenceResult result;
    auto t_total = std::chrono::steady_clock::now();

    // 1. Input: Load image via ImageSource (VDEC → VB frame)
    auto t_input = std::chrono::steady_clock::now();
    lua_cv::ImageSource source;
    if (!source.open(image_path)) {
        throw std::runtime_error("[Pipeline] Failed to open image: " + image_path);
    }
    lua_cv::Frame input_frame;
    if (!source.read(input_frame)) {
        throw std::runtime_error("[Pipeline] Failed to read image: " + image_path);
    }
    result.timings.input_ms = elapsed_ms(t_input);

    std::cout << "[Pipeline] Image loaded: " << input_frame.width() << "x"
              << input_frame.height()
              << " format=" << lua_cv::pixel_format_name(input_frame.pixel_format())
              << " backend=" << lua_cv::cv_helpers::get_backend_name(input_frame) << "\n";

    // 2. Preprocess via VPSS
    auto t_preprocess = std::chrono::steady_clock::now();
    LuaIntf::LuaRef meta;
    lua_cv::Frame preprocessed = preprocess_frame(input_frame, &meta);
    result.timings.preprocess_ms = elapsed_ms(t_preprocess);

    std::cout << "[Pipeline] Preprocessed: " << preprocessed.width() << "x"
              << preprocessed.height()
              << " format=" << lua_cv::pixel_format_name(preprocessed.pixel_format())
              << " paddr=0x" << std::hex << preprocessed.physical_addr() << std::dec << "\n";

    // 3. TPU inference
    auto t_infer = std::chrono::steady_clock::now();
    LuaIntf::LuaRef outputs = run_tpu(preprocessed);
    result.timings.inference_ms = elapsed_ms(t_infer);

    // Add output_count to meta for Lua postprocess
    meta["output_count"] = session_->output_count();

    const auto& stats = session_->last_run_stats();
    std::cout << "[Pipeline] TPU: input=" << stats.input_ms
              << "ms forward=" << stats.forward_ms
              << "ms output=" << stats.output_ms << "ms\n";

    // 4. Lua postprocess
    auto t_post = std::chrono::steady_clock::now();
    result.detections = postprocess_.call<LuaIntf::LuaRef>(outputs, meta);
    result.timings.postprocess_ms = elapsed_ms(t_post);

    // Release frames
    source.release(input_frame);
    preprocessed.release();

    result.timings.total_ms = elapsed_ms(t_total);
    last_timings_ = result.timings;

    std::cout << "[Pipeline] Timings: input=" << result.timings.input_ms
              << "ms preprocess=" << result.timings.preprocess_ms
              << "ms inference=" << result.timings.inference_ms
              << "ms postprocess=" << result.timings.postprocess_ms
              << "ms total=" << result.timings.total_ms << "ms\n";

    return result;
}

InferenceResult CviPipeline::run_camera() {
#ifndef USE_CVI_CAMERA
    throw std::runtime_error("[Pipeline] Camera support not enabled (USE_CVI_CAMERA)");
#else
    // Single-shot: open camera, capture one frame, close camera
    if (!open_camera()) {
        throw std::runtime_error("[Pipeline] Failed to open camera");
    }

    InferenceResult result = run_camera_frame();
    close_camera();

    return result;
#endif
}

bool CviPipeline::open_camera() {
#ifndef USE_CVI_CAMERA
    return false;
#else
    if (camera_ && camera_->is_opened()) {
        return true;  // Already open
    }

    camera_ = std::make_unique<lua_cv::CameraSource>();
    if (!camera_->open("")) {
        camera_.reset();
        return false;
    }

    if (!camera_->wait_for_ready(5000)) {
        camera_->close();
        camera_.reset();
        return false;
    }

    camera_fps_ = camera_->fps();
    if (camera_fps_ <= 0.0) {
        camera_fps_ = 30.0;
    }
    infer_ema_ms_ = 0.0;
    next_infer_time_ = std::chrono::steady_clock::time_point{};
    skip_state_ready_ = false;
    vpss_nobuf_streak_ = 0;

    return true;
#endif
}

InferenceResult CviPipeline::run_camera_frame() {
#ifndef USE_CVI_CAMERA
    throw std::runtime_error("[Pipeline] Camera support not enabled (USE_CVI_CAMERA)");
#else
    if (!camera_ || !camera_->is_opened()) {
        throw std::runtime_error("[Pipeline] Camera not open - call open_camera() first");
    }

    auto t_total = std::chrono::steady_clock::now();

    // 1. Camera input
    auto t_input = std::chrono::steady_clock::now();
    lua_cv::Frame input_frame;
    if (!camera_->read(input_frame)) {
        throw std::runtime_error("[Pipeline] Failed to read camera frame");
    }
    double input_ms = elapsed_ms(t_input);

    return run_camera_frame_from_input(input_frame, input_ms, t_total);
#endif
}

#ifdef USE_CVI_CAMERA
InferenceResult CviPipeline::run_camera_frame_from_input(lua_cv::Frame& input_frame,
                                                         double input_ms,
                                                         std::chrono::steady_clock::time_point total_start) {
    InferenceResult result;
    result.timings.input_ms = input_ms;

    // 2. Preprocess via VPSS
    auto t_preprocess = std::chrono::steady_clock::now();
    LuaIntf::LuaRef meta;
    lua_cv::Frame preprocessed = preprocess_frame(input_frame, &meta);
    result.timings.preprocess_ms = elapsed_ms(t_preprocess);

    // 3. TPU inference
    auto t_infer = std::chrono::steady_clock::now();
    LuaIntf::LuaRef outputs = run_tpu(preprocessed);
    result.timings.inference_ms = elapsed_ms(t_infer);

    // Add output_count to meta for Lua postprocess
    meta["output_count"] = session_->output_count();

    // 4. Lua postprocess
    auto t_post = std::chrono::steady_clock::now();
    result.detections = postprocess_.call<LuaIntf::LuaRef>(outputs, meta);
    result.timings.postprocess_ms = elapsed_ms(t_post);

    // Release frames
    camera_->release(input_frame);
    preprocessed.release();

    result.timings.total_ms = elapsed_ms(total_start);
    last_timings_ = result.timings;

    return result;
}
#endif

bool CviPipeline::run_camera_frame_adaptive(InferenceResult* result) {
#ifndef USE_CVI_CAMERA
    (void)result;
    throw std::runtime_error("[Pipeline] Camera support not enabled (USE_CVI_CAMERA)");
#else
    if (!result) {
        throw std::invalid_argument("[Pipeline] run_camera_frame_adaptive - result is null");
    }
    if (!camera_ || !camera_->is_opened()) {
        throw std::runtime_error("[Pipeline] Camera not open - call open_camera() first");
    }

    auto t_read_start = std::chrono::steady_clock::now();
    lua_cv::Frame input_frame;
    if (!camera_->read(input_frame)) {
        throw std::runtime_error("[Pipeline] Failed to read camera frame");
    }
    auto t_read_end = std::chrono::steady_clock::now();

    if (!skip_state_ready_) {
        next_infer_time_ = t_read_end;
        skip_state_ready_ = true;
    }

    if (t_read_end < next_infer_time_) {
        camera_->release(input_frame);
        *result = InferenceResult{};
        return false;
    }

    double input_ms =
        std::chrono::duration<double, std::milli>(t_read_end - t_read_start).count();

    InferenceResult local;
    try {
        local = run_camera_frame_from_input(input_frame, input_ms, t_read_start);
    } catch (const std::exception&) {
        int last_error = vpss_processor_.last_error();
        if (last_error == CVI_ERR_VPSS_NOBUF || last_error == CVI_ERR_VPSS_BUF_EMPTY) {
            vpss_nobuf_streak_++;

            double frame_interval_ms = kDefaultFrameIntervalMs;
            if (camera_fps_ > 0.0) {
                frame_interval_ms = 1000.0 / camera_fps_;
            }
            int threshold = compute_nobuf_threshold(infer_ema_ms_, frame_interval_ms);
            if (vpss_nobuf_streak_ >= threshold) {
                double cooldown_ms = compute_nobuf_cooldown_ms(
                    vpss_nobuf_streak_, threshold, infer_ema_ms_, frame_interval_ms);
                auto interval = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                    std::chrono::duration<double, std::milli>(cooldown_ms));
                next_infer_time_ = t_read_end + interval;
            }

            *result = InferenceResult{};
            return false;
        }
        throw;
    }
    vpss_nobuf_streak_ = 0;

    // Update EWMA based on processing time (preprocess + inference + postprocess).
    const double current_proc_ms =
        local.timings.preprocess_ms + local.timings.inference_ms + local.timings.postprocess_ms;
    const double kEmaAlpha = 0.2;
    const double kSafetyFactor = 1.2;
    if (infer_ema_ms_ <= 0.0) {
        infer_ema_ms_ = current_proc_ms;
    } else {
        infer_ema_ms_ = infer_ema_ms_ * (1.0 - kEmaAlpha) + current_proc_ms * kEmaAlpha;
    }

    double frame_interval_ms = kDefaultFrameIntervalMs;
    if (camera_fps_ > 0.0) {
        frame_interval_ms = 1000.0 / camera_fps_;
    }
    double target_interval_ms = infer_ema_ms_ * kSafetyFactor;
    if (target_interval_ms < frame_interval_ms) {
        target_interval_ms = frame_interval_ms;
    }
    auto interval = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<double, std::milli>(target_interval_ms));
    next_infer_time_ = t_read_end + interval;

    *result = std::move(local);
    return true;
#endif
}

void CviPipeline::close_camera() {
#ifdef USE_CVI_CAMERA
    if (camera_) {
        camera_->close();
        camera_.reset();
    }
    camera_fps_ = 0.0;
    infer_ema_ms_ = 0.0;
    next_infer_time_ = std::chrono::steady_clock::time_point{};
    skip_state_ready_ = false;
    vpss_nobuf_streak_ = 0;
#endif
}

bool CviPipeline::is_camera_open() const {
#ifdef USE_CVI_CAMERA
    return camera_ && camera_->is_opened();
#else
    return false;
#endif
}

}  // namespace pipeline

#endif  // USE_CVI_TPU && USE_CVI_MPI
