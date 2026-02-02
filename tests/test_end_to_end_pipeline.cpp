/**
 * test_end_to_end_pipeline.cpp - End-to-end zero-copy pipeline tests
 */

#include "test_common.h"

#if defined(USE_CVI_TPU) && defined(USE_CVI_MPI)

#include "inference/cvi_session.h"
#include "cv/image_source.h"
#include "cv/cvi_vpss_processor.h"
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#ifdef USE_CVI_CAMERA
#include "cv/camera_source.h"
#endif

namespace {

std::string get_test_image_path() {
    const char* env = std::getenv("TEST_IMAGE");
    if (env && env[0] != '\0') {
        return std::string(env);
    }
    return "/home/recamera/zidane.jpg";
}

std::string get_test_model_path() {
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

std::string get_env_path(const char* name) {
    const char* env = std::getenv(name);
    if (!env || env[0] == '\0') {
        return {};
    }
    return std::string(env);
}

uint8_t get_letterbox_pad_value() {
    std::string value = get_env_path("TEST_LETTERBOX_PAD");
    if (value.empty()) {
        return 114;
    }
    char* endptr = nullptr;
    long parsed = std::strtol(value.c_str(), &endptr, 10);
    if (endptr == value.c_str() || parsed < 0 || parsed > 255) {
        std::cout << "  Warning: Invalid TEST_LETTERBOX_PAD=" << value << " (using 114)\n";
        return 114;
    }
    return static_cast<uint8_t>(parsed);
}

struct Detection {
    float x, y, w, h;  // Center coords + dimensions (normalized [0,1])
    float confidence;
    int class_id;
};

// Sigmoid activation
inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// Inverse sigmoid (for pre-threshold optimization)
inline float inverse_sigmoid(float x) {
    return std::log(x / (1.0f - x));
}

// DFL (Distribution Focal Loss) decoding
void compute_dfl(const float* tensor, int dfl_len, float* box) {
    std::vector<float> exp_t(dfl_len);

    for (int b = 0; b < 4; b++) {
        float exp_sum = 0.0f;
        for (int i = 0; i < dfl_len; i++) {
            exp_t[i] = std::exp(tensor[i + b * dfl_len]);
            exp_sum += exp_t[i];
        }

        float acc_sum = 0.0f;
        for (int i = 0; i < dfl_len; i++) {
            acc_sum += (exp_t[i] / exp_sum) * i;
        }
        box[b] = acc_sum;
    }
}

// IoU calculation for NMS
float compute_iou(const Detection& a, const Detection& b) {
    float a_x1 = a.x - a.w / 2, a_y1 = a.y - a.h / 2;
    float a_x2 = a.x + a.w / 2, a_y2 = a.y + a.h / 2;
    float b_x1 = b.x - b.w / 2, b_y1 = b.y - b.h / 2;
    float b_x2 = b.x + b.w / 2, b_y2 = b.y + b.h / 2;

    float inter_x1 = std::max(a_x1, b_x1);
    float inter_y1 = std::max(a_y1, b_y1);
    float inter_x2 = std::min(a_x2, b_x2);
    float inter_y2 = std::min(a_y2, b_y2);

    float inter_w = std::max(0.0f, inter_x2 - inter_x1);
    float inter_h = std::max(0.0f, inter_y2 - inter_y1);
    float inter_area = inter_w * inter_h;

    float a_area = a.w * a.h;
    float b_area = b.w * b.h;
    float union_area = a_area + b_area - inter_area;

    return union_area > 0 ? inter_area / union_area : 0.0f;
}

// NMS (Non-Maximum Suppression)
void apply_nms(std::vector<Detection>& detections, float iou_threshold) {
    std::sort(detections.begin(), detections.end(),
              [](const Detection& a, const Detection& b) {
                  return a.confidence > b.confidence;
              });

    for (size_t i = 0; i < detections.size(); i++) {
        if (detections[i].confidence == 0.0f) continue;

        for (size_t j = i + 1; j < detections.size(); j++) {
            if (detections[j].confidence == 0.0f) continue;

            float iou = compute_iou(detections[i], detections[j]);
            if (iou > iou_threshold) {
                detections[j].confidence = 0.0f;
            }
        }
    }

    detections.erase(
        std::remove_if(detections.begin(), detections.end(),
                       [](const Detection& d) { return d.confidence == 0.0f; }),
        detections.end());
}

// YOLO11n output parsing (3 scales × 2 tensors)
std::vector<Detection> parse_yolo_output(
    const std::vector<std::vector<float>>& outputs,
    const std::vector<std::vector<int64_t>>& shapes,
    int img_width = 640,
    int img_height = 640,
    float conf_threshold = 0.25f,
    float iou_threshold = 0.45f,
    const lua_cv::CviVpssProcessor::LetterboxMeta* letterbox_meta = nullptr) {

    std::vector<Detection> detections;

    if (outputs.size() != 6 || shapes.size() != 6) {
        std::cout << "Warning: Expected 6 outputs (3 scales), got " << outputs.size() << "\n";
        return detections;
    }

    lua_cv::CviVpssProcessor::LetterboxMeta meta{};
    if (letterbox_meta) {
        meta = *letterbox_meta;
    } else {
        meta.scale = 1.0f;
        meta.pad_x = 0;
        meta.pad_y = 0;
        meta.ori_w = img_width;
        meta.ori_h = img_height;
    }

    std::cout << "  Letterbox params: scale=" << meta.scale
              << ", pad=" << meta.pad_x << "," << meta.pad_y
              << ", orig=" << meta.ori_w << "x" << meta.ori_h << "\n";

    const int num_classes = 80;
    const int dfl_len = 16;
    const float score_threshold_non_sigmoid = inverse_sigmoid(conf_threshold);

    // Group tensors by scale (identify by shape, not by index order)
    // bbox tensor: [1, 64, H, W], score tensor: [1, 80, H, W]
    struct ScaleTensors {
        int bbox_idx;
        int score_idx;
        int grid_h;
        int grid_w;
    };
    std::vector<ScaleTensors> scale_groups;

    // Find all bbox tensors (channel=64) and match with score tensors (channel=80)
    for (size_t i = 0; i < shapes.size(); i++) {
        if (shapes[i].size() == 4 && shapes[i][1] == 64) {
            int grid_h = static_cast<int>(shapes[i][2]);
            int grid_w = static_cast<int>(shapes[i][3]);

            // Find matching score tensor with same spatial size
            for (size_t j = 0; j < shapes.size(); j++) {
                if (shapes[j].size() == 4 && shapes[j][1] == 80 &&
                    shapes[j][2] == grid_h && shapes[j][3] == grid_w) {
                    scale_groups.push_back({static_cast<int>(i), static_cast<int>(j), grid_h, grid_w});
                    break;
                }
            }
        }
    }

    if (scale_groups.size() != 3) {
        std::cout << "Warning: Expected 3 scale groups, found " << scale_groups.size() << "\n";
        return detections;
    }

    // Sort by grid size (largest first: 80x80, 40x40, 20x20)
    std::sort(scale_groups.begin(), scale_groups.end(),
              [](const ScaleTensors& a, const ScaleTensors& b) {
                  return a.grid_h > b.grid_h;
              });

    std::cout << "  Tensor grouping:\n";
    for (size_t i = 0; i < scale_groups.size(); i++) {
        std::cout << "    Scale " << i << ": bbox=tensor[" << scale_groups[i].bbox_idx
                  << "], score=tensor[" << scale_groups[i].score_idx
                  << "], grid=" << scale_groups[i].grid_h << "x" << scale_groups[i].grid_w << "\n";
    }

    // Process 3 scales
    for (size_t scale_idx = 0; scale_idx < scale_groups.size(); scale_idx++) {
        const auto& scale = scale_groups[scale_idx];
        const auto& bbox_output = outputs[scale.bbox_idx];
        const auto& score_output = outputs[scale.score_idx];

        int grid_h = scale.grid_h;
        int grid_w = scale.grid_w;
        int stride = img_height / grid_h;

        for (int j = 0; j < grid_h; j++) {
            for (int k = 0; k < grid_w; k++) {
                int score_offset = j * grid_w + k;
                int grid_l = grid_h * grid_w;

                float max_score = -1000.0f;
                int target_class = -1;

                for (int c = 0; c < num_classes; c++) {
                    float score = score_output[score_offset + c * grid_l];
                    if (score > max_score) {
                        max_score = score;
                        target_class = c;
                    }
                }

                if (max_score <= score_threshold_non_sigmoid) {
                    continue;
                }

                float confidence = sigmoid(max_score);
                if (confidence < conf_threshold) {
                    continue;
                }

                // DFL decode bbox (64 channels → 4 coordinates)
                // Extract 64 channels for this grid position (NCHW format)
                std::vector<float> bbox_data(64);
                for (int ch = 0; ch < 64; ch++) {
                    bbox_data[ch] = bbox_output[ch * grid_l + j * grid_w + k];
                }
                float dfl_box[4];
                compute_dfl(bbox_data.data(), dfl_len, dfl_box);

                float x1 = (-dfl_box[0] + k + 0.5f) * stride;
                float y1 = (-dfl_box[1] + j + 0.5f) * stride;
                float x2 = (dfl_box[2] + k + 0.5f) * stride;
                float y2 = (dfl_box[3] + j + 0.5f) * stride;

                // Normalize to [0,1] relative to original image
                Detection det;
                float w = x2 - x1;
                float h = y2 - y1;
                det.x = (x1 + w / 2.0f) / img_width;   // Center X
                det.y = (y1 + h / 2.0f) / img_height;  // Center Y
                det.w = w / img_width;                // Width
                det.h = h / img_height;               // Height
                det.confidence = confidence;
                det.class_id = target_class;

                detections.push_back(det);
            }
        }
    }

    apply_nms(detections, iou_threshold);

    std::vector<Detection> mapped;
    mapped.reserve(detections.size());
    for (const auto& det : detections) {
        float x1 = (det.x - det.w / 2) * img_width;
        float y1 = (det.y - det.h / 2) * img_height;
        float x2 = (det.x + det.w / 2) * img_width;
        float y2 = (det.y + det.h / 2) * img_height;

        float x1_orig = (x1 - meta.pad_x) / meta.scale;
        float y1_orig = (y1 - meta.pad_y) / meta.scale;
        float x2_orig = (x2 - meta.pad_x) / meta.scale;
        float y2_orig = (y2 - meta.pad_y) / meta.scale;

        if (x2_orig <= 0 || y2_orig <= 0 ||
            x1_orig >= meta.ori_w || y1_orig >= meta.ori_h) {
            continue;
        }

        x1_orig = std::max(0.0f, std::min(x1_orig, static_cast<float>(meta.ori_w - 1)));
        y1_orig = std::max(0.0f, std::min(y1_orig, static_cast<float>(meta.ori_h - 1)));
        x2_orig = std::max(0.0f, std::min(x2_orig, static_cast<float>(meta.ori_w - 1)));
        y2_orig = std::max(0.0f, std::min(y2_orig, static_cast<float>(meta.ori_h - 1)));

        float w_orig = x2_orig - x1_orig;
        float h_orig = y2_orig - y1_orig;
        if (w_orig <= 0.0f || h_orig <= 0.0f) {
            continue;
        }

        Detection mapped_det;
        mapped_det.x = (x1_orig + w_orig / 2.0f) / meta.ori_w;
        mapped_det.y = (y1_orig + h_orig / 2.0f) / meta.ori_h;
        mapped_det.w = w_orig / meta.ori_w;
        mapped_det.h = h_orig / meta.ori_h;
        mapped_det.confidence = det.confidence;
        mapped_det.class_id = det.class_id;
        mapped.push_back(mapped_det);
    }

    detections.swap(mapped);

    // Print detections in original image coordinates
    std::cout << "Detections: " << detections.size() << " objects found\n";
    for (const auto& det : detections) {
        float x1 = (det.x - det.w/2) * meta.ori_w;
        float y1 = (det.y - det.h/2) * meta.ori_h;
        float x2 = (det.x + det.w/2) * meta.ori_w;
        float y2 = (det.y + det.h/2) * meta.ori_h;
        printf("- Class %d: %.2f confidence at [%.0f,%.0f,%.0f,%.0f]\n",
               det.class_id, det.confidence, x1, y1, x2, y2);
    }

    return detections;
}

const char* vb_pixel_format_name(PIXEL_FORMAT_E fmt) {
    switch (fmt) {
        case PIXEL_FORMAT_RGB_888:
            return "rgb_packed";
        case PIXEL_FORMAT_BGR_888:
            return "bgr_packed";
        case PIXEL_FORMAT_RGB_888_PLANAR:
            return "rgb_planar";
        case PIXEL_FORMAT_BGR_888_PLANAR:
            return "bgr_planar";
        case PIXEL_FORMAT_NV12:
            return "nv12";
        case PIXEL_FORMAT_NV21:
            return "nv21";
        case PIXEL_FORMAT_YUV_400:
            return "grayscale";
        default:
            return "unknown";
    }
}

bool parse_force_pixel_format(lua_cv::PixelFormat* output_format) {
    if (!output_format) {
        return false;
    }
    std::string value = get_env_path("TEST_FORCE_PIXEL_FORMAT");
    if (value.empty()) {
        return false;
    }
    for (char& ch : value) {
        ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    }
    if (value == "BGR") {
        *output_format = lua_cv::PixelFormat::BGR;
        return true;
    }
    if (value == "RGB") {
        *output_format = lua_cv::PixelFormat::RGB;
        return true;
    }
    if (value == "RGB_PLANAR") {
        *output_format = lua_cv::PixelFormat::RGB_PLANAR;
        return true;
    }
    if (value == "GRAY") {
        *output_format = lua_cv::PixelFormat::GRAY;
        return true;
    }
    std::cout << "  Warning: Unknown TEST_FORCE_PIXEL_FORMAT=" << value << "\n";
    return false;
}

bool choose_output_format(const inference::CviSession::VbInputSpec& spec,
                          lua_cv::PixelFormat* output_format) {
    if (!output_format) {
        return false;
    }
    switch (spec.pixel_format) {
        case PIXEL_FORMAT_RGB_888:
            *output_format = lua_cv::PixelFormat::RGB;
            return true;
        case PIXEL_FORMAT_BGR_888:
            *output_format = lua_cv::PixelFormat::BGR;
            return true;
        case PIXEL_FORMAT_RGB_888_PLANAR:
            *output_format = lua_cv::PixelFormat::RGB_PLANAR;
            return true;
        case PIXEL_FORMAT_YUV_400:
            *output_format = lua_cv::PixelFormat::GRAY;
            return true;
        case PIXEL_FORMAT_BGR_888_PLANAR:
            std::cout << "  Warning: Model input pixel_format=bgr_planar not supported by PixelFormat\n";
            return false;
        default:
            std::cout << "  Warning: Model input pixel_format unsupported: "
                      << vb_pixel_format_name(spec.pixel_format) << "\n";
            return false;
    }
}

struct ModelInputPlan {
    lua_cv::PixelFormat desired_format = lua_cv::PixelFormat::UNKNOWN;
    bool has_desired_format = false;
    uint8_t pad_value = 0;
};

ModelInputPlan build_model_input_plan(const inference::CviSession::VbInputSpec& spec) {
    ModelInputPlan plan;
    plan.has_desired_format = choose_output_format(spec, &plan.desired_format);
    lua_cv::PixelFormat forced_format = lua_cv::PixelFormat::UNKNOWN;
    if (parse_force_pixel_format(&forced_format)) {
        plan.desired_format = forced_format;
        plan.has_desired_format = true;
        std::cout << "  Forcing output format: "
                  << lua_cv::pixel_format_name(plan.desired_format) << "\n";
    }
    plan.pad_value = get_letterbox_pad_value();
    std::cout << "  Letterbox pad value: " << static_cast<int>(plan.pad_value) << "\n";
    return plan;
}

void log_frame_info(const char* label, const lua_cv::Frame& frame) {
    std::cout << "  " << label << ": " << frame.width() << "x" << frame.height()
              << ", format=" << lua_cv::pixel_format_name(frame.pixel_format()) << "\n";
}

void prepare_frame_for_model(lua_cv::Frame& frame,
                             lua_cv::CviVpssProcessor& vpss,
                             const inference::CviSession::VbInputSpec& spec,
                             const ModelInputPlan& plan,
                             lua_cv::CviVpssProcessor::LetterboxMeta* meta) {
    if (!meta) {
        throw std::invalid_argument("prepare_frame_for_model - meta is null");
    }

    log_frame_info("Before VPSS", frame);

    if (frame.width() == static_cast<int>(spec.width) &&
        frame.height() == static_cast<int>(spec.height)) {
        meta->scale = 1.0f;
        meta->pad_x = 0;
        meta->pad_y = 0;
        meta->ori_w = frame.width();
        meta->ori_h = frame.height();
        std::cout << "  VPSS skip: frame already " << spec.width << "x"
                  << spec.height << "\n";
    } else {
        // Pass desired output format to letterbox to do resize+pad+format in one VPSS pass
        lua_cv::PixelFormat lb_format = lua_cv::PixelFormat::RGB_PLANAR;
        if (plan.has_desired_format) {
            lb_format = plan.desired_format;
        }
        vpss.letterbox(frame, spec.width, spec.height, plan.pad_value, meta, lb_format);
    }

    if (plan.has_desired_format && frame.pixel_format() != plan.desired_format) {
        std::cout << "  Converting format: "
                  << lua_cv::pixel_format_name(frame.pixel_format())
                  << " -> " << lua_cv::pixel_format_name(plan.desired_format) << "\n";
        vpss.convert_format(frame, plan.desired_format);
    }

    log_frame_info("After VPSS", frame);
}

void log_input_spec(const inference::CviSession::VbInputSpec& spec) {
    std::cout << "  Model input spec: format=" << vb_pixel_format_name(spec.pixel_format)
              << ", normalized=" << (spec.normalized ? "true" : "false")
              << ", mean=(" << spec.mean[0] << ", " << spec.mean[1] << ", " << spec.mean[2] << ")"
              << ", scale=(" << spec.scale[0] << ", " << spec.scale[1] << ", " << spec.scale[2] << ")\n";
}

void log_output_tensors(const std::vector<std::vector<float>>& outputs,
                        const std::vector<std::vector<int64_t>>& shapes,
                        size_t max_tensors,
                        bool include_range) {
    std::cout << "  Model outputs: " << outputs.size() << " tensors\n";
    size_t count = std::min(outputs.size(), shapes.size());
    for (size_t i = 0; i < count && i < max_tensors; i++) {
        std::cout << "    Tensor " << i << ": shape=[";
        for (size_t j = 0; j < shapes[i].size(); j++) {
            if (j > 0) std::cout << ",";
            std::cout << shapes[i][j];
        }
        std::cout << "], size=" << outputs[i].size() << "\n";
        std::cout << "      First 5 values: ";
        for (size_t j = 0; j < std::min(outputs[i].size(), size_t(5)); j++) {
            std::cout << outputs[i][j] << " ";
        }
        std::cout << "\n";
        if (include_range && !outputs[i].empty()) {
            float min_val = *std::min_element(outputs[i].begin(), outputs[i].end());
            float max_val = *std::max_element(outputs[i].begin(), outputs[i].end());
            std::cout << "      Range: [" << min_val << ", " << max_val << "]\n";
        }
    }
}

void log_tpu_stats(const inference::CviSession::RunStats& stats, double tpu_ms) {
    std::cout << "  TPU time: " << tpu_ms << " ms\n";
    std::cout << "    Input:   " << stats.input_ms << " ms\n";
    std::cout << "    Forward: " << stats.forward_ms << " ms\n";
    std::cout << "    Output:  " << stats.output_ms << " ms\n\n";
}

bool draw_detections_and_save(const std::vector<Detection>& detections,
                              const lua_cv::Frame& frame,
                              int orig_width,
                              int orig_height,
                              const std::string& output_path) {
    if (frame.empty()) {
        std::cout << "  Warning: Visualization frame is empty\n\n";
        return false;
    }

    std::cout << "  Visualization frame: " << frame.width() << "x" << frame.height() << "\n";

    cv::Mat bgr_mat = frame.to_mat_copy();
    if (bgr_mat.empty()) {
        std::cout << "  Warning: Failed to convert frame to BGR\n\n";
        return false;
    }

    std::cout << "  BGR Mat size: " << bgr_mat.cols << "x" << bgr_mat.rows << "\n";

    int width = orig_width > 0 ? orig_width : bgr_mat.cols;
    int height = orig_height > 0 ? orig_height : bgr_mat.rows;

    for (const auto& det : detections) {
        float x1_orig = (det.x - det.w / 2) * width;
        float y1_orig = (det.y - det.h / 2) * height;
        float x2_orig = (det.x + det.w / 2) * width;
        float y2_orig = (det.y + det.h / 2) * height;

        x1_orig = std::max(0.0f, std::min(x1_orig, static_cast<float>(width - 1)));
        y1_orig = std::max(0.0f, std::min(y1_orig, static_cast<float>(height - 1)));
        x2_orig = std::max(0.0f, std::min(x2_orig, static_cast<float>(width - 1)));
        y2_orig = std::max(0.0f, std::min(y2_orig, static_cast<float>(height - 1)));

        cv::rectangle(bgr_mat,
                      cv::Point(static_cast<int>(x1_orig), static_cast<int>(y1_orig)),
                      cv::Point(static_cast<int>(x2_orig), static_cast<int>(y2_orig)),
                      cv::Scalar(0, 255, 0), 2);

        std::string label = "Class " + std::to_string(det.class_id) +
                            ": " + std::to_string(static_cast<int>(det.confidence * 100)) + "%";
        cv::putText(bgr_mat, label,
                    cv::Point(static_cast<int>(x1_orig), static_cast<int>(y1_orig) - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }

    if (cv::imwrite(output_path, bgr_mat)) {
        std::cout << "  Saved detection result to: " << output_path << "\n\n";
        return true;
    }

    std::cout << "  Warning: Failed to save image to " << output_path << "\n\n";
    return false;
}

void dump_preprocess_frame(const std::string& output_path,
                           lua_cv::CviVpssProcessor& vpss,
                           lua_cv::Frame& frame) {
    if (output_path.empty()) {
        return;
    }

    if (frame.empty()) {
        std::cout << "  Warning: Preprocess dump skipped (frame empty)\n";
        return;
    }

    if (frame.pixel_format() == lua_cv::PixelFormat::RGB_PLANAR) {
        std::cout << "  Converting RGB_PLANAR to BGR for preprocess dump\n";
        vpss.convert_format(frame, lua_cv::PixelFormat::BGR);
    }

    cv::Mat dump = frame.to_mat_copy();
    if (dump.empty()) {
        std::cout << "  Warning: Preprocess dump skipped (failed to map frame)\n";
        return;
    }

    if (frame.pixel_format() == lua_cv::PixelFormat::RGB) {
        cv::cvtColor(dump, dump, cv::COLOR_RGB2BGR);
    }

    if (cv::imwrite(output_path, dump)) {
        std::cout << "  Saved preprocess image to: " << output_path << "\n";
    } else {
        std::cout << "  Warning: Failed to save preprocess image to " << output_path << "\n";
    }
}

} // namespace

TEST(EndToEndPipeline, ImageFileToDetection) {
    std::string image_path = get_test_image_path();
    std::string model_path = get_test_model_path();

    if (!is_cvi_ready()) {
        GTEST_SKIP() << "CVI system not ready";
    }
    if (!file_exists(image_path)) {
        GTEST_SKIP() << "Test image not found: " << image_path;
    }
    if (!file_exists(model_path)) {
        GTEST_SKIP() << "Test model not found: " << model_path;
    }

    std::cout << "\nTesting file input pipeline with:\n";
    std::cout << "  Image: " << image_path << "\n";
    std::cout << "  Model: " << model_path << "\n\n";

    auto start_total = std::chrono::high_resolution_clock::now();

    auto start_stage = std::chrono::high_resolution_clock::now();

#ifdef USE_VDEC_DECODE
    std::cout << "[Stage 1] Loading image (hardware VDEC decode)...\n";

    lua_cv::ImageSource image_source;
    ASSERT_TRUE(image_source.open(image_path)) << "Failed to open image";

    lua_cv::Frame frame;
    ASSERT_TRUE(image_source.read(frame)) << "Failed to read image";
    ASSERT_FALSE(frame.empty()) << "Frame is empty";

    // Save original frame for visualization
    lua_cv::Frame orig_frame = frame.clone();
    int orig_width = frame.width();
    int orig_height = frame.height();
#else
    std::cout << "[Stage 1] Loading image (software decode)...\n";

    // Use software JPEG decode (OpenCV) - no VB pool/VDEC required
    cv::Mat decoded_img = cv::imread(image_path, cv::IMREAD_COLOR);
    ASSERT_FALSE(decoded_img.empty()) << "Failed to load image: " << image_path;

    lua_cv::Frame frame(decoded_img);
    ASSERT_FALSE(frame.empty()) << "Frame is empty";

    // Save original frame for visualization
    lua_cv::Frame orig_frame = frame.clone();
    int orig_width = frame.width();
    int orig_height = frame.height();
#endif

    auto end_stage = std::chrono::high_resolution_clock::now();
    double stage1_ms = std::chrono::duration<double, std::milli>(end_stage - start_stage).count();
    std::cout << "  Image loaded: " << frame.width() << "x" << frame.height()
              << " (" << stage1_ms << " ms)\n\n";

    std::cout << "[Stage 2] Loading TPU model...\n";
    start_stage = std::chrono::high_resolution_clock::now();

    inference::CviSession session(model_path);
    ASSERT_TRUE(session.supports_vb_input()) << "Model does not support VB input";
    auto vb_spec = session.get_vb_input_spec();
    log_input_spec(vb_spec);
    ModelInputPlan input_plan = build_model_input_plan(vb_spec);

    end_stage = std::chrono::high_resolution_clock::now();
    double stage2_ms = std::chrono::duration<double, std::milli>(end_stage - start_stage).count();
    std::cout << "  Model loaded: input " << vb_spec.width << "x" << vb_spec.height
              << " (" << stage2_ms << " ms)\n\n";

    std::cout << "[Stage 3] VPSS preprocessing...\n";
    start_stage = std::chrono::high_resolution_clock::now();

    lua_cv::CviVpssProcessor vpss;
    lua_cv::CviVpssProcessor::LetterboxMeta letterbox_meta;
    ASSERT_NO_THROW(prepare_frame_for_model(frame, vpss, vb_spec, input_plan, &letterbox_meta));
    ASSERT_FALSE(frame.empty()) << "VPSS preprocessing failed";

    end_stage = std::chrono::high_resolution_clock::now();
    double vpss_ms = std::chrono::duration<double, std::milli>(end_stage - start_stage).count();
    std::cout << "  VPSS time: " << vpss_ms << " ms\n\n";

    std::cout << "[Stage 4] TPU inference...\n";
    start_stage = std::chrono::high_resolution_clock::now();

    auto vb_input = frame.as_vb_memory();
    ASSERT_NE(vb_input, nullptr) << "Failed to export VB memory";

    std::vector<std::vector<float>> outputs;
    std::vector<std::vector<int64_t>> shapes;
    ASSERT_NO_THROW(session.run_vb(vb_input, &outputs, &shapes));

    const auto& stats = session.last_run_stats();
    end_stage = std::chrono::high_resolution_clock::now();
    double tpu_ms = std::chrono::duration<double, std::milli>(end_stage - start_stage).count();
    log_tpu_stats(stats, tpu_ms);

    dump_preprocess_frame(get_env_path("TEST_DUMP_PREPROCESS_FILE"), vpss, frame);

    std::cout << "[Stage 5] Postprocessing...\n";
    log_output_tensors(outputs, shapes, outputs.size(), true);
    std::cout << "\n";

    auto detections = parse_yolo_output(outputs, shapes,
                                       vb_spec.width, vb_spec.height, 0.5f, 0.45f,
                                       &letterbox_meta);
    std::cout << "  Detections: " << detections.size() << " boxes\n\n";

    // Draw detections and save
    std::cout << "[Stage 6] Drawing detections and saving...\n";
    draw_detections_and_save(detections, orig_frame, orig_width, orig_height,
                             "/tmp/file_detection.jpg");

    auto end_total = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(end_total - start_total).count();

    std::cout << "[Performance Summary]\n";
    std::cout << "  Image load:      " << std::fixed << std::setprecision(3) << stage1_ms << " ms\n";
    std::cout << "  Model load:      " << stage2_ms << " ms (one-time)\n";
    std::cout << "  VPSS preprocess: " << vpss_ms << " ms\n";
    std::cout << "  TPU inference:   " << tpu_ms << " ms\n";
    std::cout << "  Total pipeline:  " << total_ms << " ms\n";
    std::cout << "  Target (<60ms):  " << (total_ms < 60.0 ? "PASS" : "FAIL") << "\n\n";

#ifdef USE_VDEC_DECODE
    // Hardware decode - cleanup ImageSource
    image_source.release(frame);
    image_source.close();
#else
    // Software decode - no cleanup needed
#endif

    EXPECT_LT(stats.input_ms, 1.0) << "Zero-copy input should be <1ms";
    EXPECT_LT(stats.forward_ms, 45.0) << "TPU forward should be <45ms (hardware baseline ~37ms)";
    EXPECT_LT(vpss_ms, 10.0) << "VPSS should be <10ms";
    EXPECT_GT(detections.size(), 0u) << "Should detect at least one object in test image";
}

#ifdef USE_CVI_CAMERA
TEST(EndToEndPipeline, CameraToDetection) {
    std::string model_path = get_test_model_path();

    if (!is_cvi_ready()) {
        GTEST_SKIP() << "CVI system not ready";
    }
    if (!file_exists(model_path)) {
        GTEST_SKIP() << "Test model not found: " << model_path;
    }

    std::cout << "\nTesting camera input pipeline with:\n";
    std::cout << "  Camera: 1920x1080 NV21\n";
    std::cout << "  Model: " << model_path << "\n\n";

    auto start_total = std::chrono::high_resolution_clock::now();

    std::cout << "[Stage 1] Initializing camera...\n";
    auto start_stage = std::chrono::high_resolution_clock::now();

    lua_cv::CameraSource camera;
    ASSERT_TRUE(camera.open("")) << "Failed to open camera";

    ASSERT_TRUE(camera.wait_for_ready(5000)) << "Camera not ready";

    auto end_stage = std::chrono::high_resolution_clock::now();
    double stage1_ms = std::chrono::duration<double, std::milli>(end_stage - start_stage).count();
    std::cout << "  Camera initialized (" << stage1_ms << " ms)\n\n";

    std::cout << "[Stage 2] Capturing frame...\n";
    start_stage = std::chrono::high_resolution_clock::now();

    lua_cv::Frame frame;
    ASSERT_TRUE(camera.read(frame)) << "Failed to capture frame";
    ASSERT_FALSE(frame.empty()) << "Frame is empty";
    ASSERT_NE(frame.physical_addr(), 0u) << "Frame should have physical address";

    end_stage = std::chrono::high_resolution_clock::now();
    double stage2_ms = std::chrono::duration<double, std::milli>(end_stage - start_stage).count();
    std::cout << "  Frame captured: " << frame.width() << "x" << frame.height()
              << ", phys_addr=0x" << std::hex << frame.physical_addr() << std::dec
              << ", format=" << lua_cv::pixel_format_name(frame.pixel_format())
              << " (" << stage2_ms << " ms)\n";

    // Clone frame for visualization (before VPSS preprocessing)
    std::cout << "  Cloning frame for visualization...\n";
    lua_cv::Frame vis_frame = frame.clone();
    int orig_width = vis_frame.width();
    int orig_height = vis_frame.height();
    std::cout << "  Cloned frame: " << orig_width << "x" << orig_height
              << ", format=" << lua_cv::pixel_format_name(vis_frame.pixel_format()) << "\n\n";

    std::cout << "[Stage 3] Loading TPU model...\n";
    start_stage = std::chrono::high_resolution_clock::now();

    inference::CviSession session(model_path);
    ASSERT_TRUE(session.supports_vb_input()) << "Model does not support VB input";
    auto vb_spec = session.get_vb_input_spec();
    log_input_spec(vb_spec);
    ModelInputPlan input_plan = build_model_input_plan(vb_spec);

    end_stage = std::chrono::high_resolution_clock::now();
    double stage3_ms = std::chrono::duration<double, std::milli>(end_stage - start_stage).count();
    std::cout << "  Model loaded: input " << vb_spec.width << "x" << vb_spec.height
              << " (" << stage3_ms << " ms)\n\n";

    std::cout << "[Stage 4] VPSS preprocessing (letterbox + format convert)...\n";
    start_stage = std::chrono::high_resolution_clock::now();

    lua_cv::CviVpssProcessor vpss;
    lua_cv::CviVpssProcessor::LetterboxMeta letterbox_meta;
    ASSERT_NO_THROW(prepare_frame_for_model(frame, vpss, vb_spec, input_plan, &letterbox_meta));
    ASSERT_FALSE(frame.empty()) << "VPSS preprocessing failed";

    end_stage = std::chrono::high_resolution_clock::now();
    double vpss_ms = std::chrono::duration<double, std::milli>(end_stage - start_stage).count();
    std::cout << "  VPSS time: " << vpss_ms << " ms\n\n";

    std::cout << "[Stage 5] TPU inference...\n";
    start_stage = std::chrono::high_resolution_clock::now();

    auto vb_input = frame.as_vb_memory();
    ASSERT_NE(vb_input, nullptr) << "Failed to export VB memory";
    std::cout << "  VB input: paddr=0x"
              << std::hex << vb_input->physical_address() << std::dec << "\n";

    std::vector<std::vector<float>> outputs;
    std::vector<std::vector<int64_t>> shapes;
    ASSERT_NO_THROW(session.run_vb(vb_input, &outputs, &shapes));

    log_output_tensors(outputs, shapes, 6, false);

    const auto& stats = session.last_run_stats();
    end_stage = std::chrono::high_resolution_clock::now();
    double tpu_ms = std::chrono::duration<double, std::milli>(end_stage - start_stage).count();
    log_tpu_stats(stats, tpu_ms);

    dump_preprocess_frame(get_env_path("TEST_DUMP_PREPROCESS_CAMERA"), vpss, frame);

    std::cout << "[Stage 6] Postprocessing...\n";
    auto detections = parse_yolo_output(outputs, shapes,
                                       vb_spec.width, vb_spec.height, 0.5f, 0.45f,
                                       &letterbox_meta);
    std::cout << "  Detections: " << detections.size() << " boxes\n\n";

    // Draw detections on image and save
    std::cout << "[Stage 7] Drawing detections and saving...\n";

    draw_detections_and_save(detections, vis_frame, orig_width, orig_height,
                             "/tmp/camera_detection.jpg");

    auto end_total = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(end_total - start_total).count();

    std::cout << "[Performance Summary]\n";
    std::cout << "  Camera init:     " << std::fixed << std::setprecision(3) << stage1_ms << " ms (one-time)\n";
    std::cout << "  Frame capture:   " << stage2_ms << " ms\n";
    std::cout << "  Model load:      " << stage3_ms << " ms (one-time)\n";
    std::cout << "  VPSS preprocess: " << vpss_ms << " ms\n";
    std::cout << "  TPU inference:   " << tpu_ms << " ms\n";
    std::cout << "  Total pipeline:  " << total_ms << " ms\n";
    std::cout << "  Target (<65ms):  " << (total_ms < 65.0 ? "PASS" : "FAIL") << "\n\n";

    camera.release(frame);
    camera.close();

    EXPECT_LT(stats.input_ms, 1.0) << "Zero-copy input should be <1ms";
    EXPECT_LT(stats.forward_ms, 45.0) << "TPU forward should be <45ms (hardware baseline ~37ms)";
    EXPECT_LT(vpss_ms, 15.0) << "VPSS should be <15ms (includes color conversion)";

    std::cout << "\nNote: Camera test detected " << detections.size() << " objects.\n";
    std::cout << "Detection count depends on camera view (may be 0 if pointing at wall).\n";
}
#endif // USE_CVI_CAMERA

#endif // USE_CVI_TPU && USE_CVI_MPI

int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
#if defined(USE_CVI_MPI)
    register_cvi_environment();
#endif
    return RUN_ALL_TESTS();
}
