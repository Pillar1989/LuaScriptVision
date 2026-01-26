#include <array>
#include <iomanip>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <opencv2/opencv.hpp>
#include "inference/session_factory.h"
#include "inference/layout.h"
#ifdef USE_CVI_TPU
#include "inference/cvi_session.h"
#endif
#include "main_util.h"

// Configuration
const int INPUT_W = 640;
const int INPUT_H = 640;
const float CONF_THRES = 0.25f;
const float IOU_THRES = 0.45f;
const int STRIDE = 32;

// ============ Data Structures ============

struct Detection {
    float x, y, w, h;
    float score;
    int class_id;
};

struct PreprocessMeta {
    float scale;
    int pad_x, pad_y;
    int ori_w, ori_h;
};

struct PreprocessParams {
    std::array<float, 3> mean;
    std::array<float, 3> scale;
    bool swap_rb;
    int pad_value;
};

struct TimingBreakdown {
    double preprocess_ms = 0.0;
    double inference_ms = 0.0;
    double postprocess_ms = 0.0;
    double resize_ms = 0.0;
    double blob_ms = 0.0;
    double cvi_input_ms = 0.0;
    double cvi_forward_ms = 0.0;
    double cvi_output_ms = 0.0;
};

std::string shape_to_string(const std::vector<int64_t>& shape) {
    std::string out = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            out += ", ";
        }
        out += std::to_string(shape[i]);
    }
    out += "]";
    return out;
}

PreprocessParams default_preprocess_params() {
    PreprocessParams params;
    params.mean = {0.0f, 0.0f, 0.0f};
    params.scale = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
    params.swap_rb = true;
    params.pad_value = 114;
    return params;
}

PreprocessParams tdl_yolov8_preprocess_params() {
    PreprocessParams params;
    constexpr float kStd = 254.97195f;
    params.mean = {0.0f, 0.0f, 0.0f};
    params.scale = {1.0f / kStd, 1.0f / kStd, 1.0f / kStd};
    params.swap_rb = true;
    params.pad_value = 0;
    return params;
}

bool ends_with(const std::string& value, const std::string& suffix) {
    if (suffix.size() > value.size()) {
        return false;
    }
    return std::equal(suffix.rbegin(), suffix.rend(), value.rbegin());
}

enum class ModelKind {
    YoloV8Head,
    YoloV8Fused,
};

ModelKind model_kind_from_path(const std::string& model_path) {
    if (ends_with(model_path, ".cvimodel")) {
        return ModelKind::YoloV8Head;
    }
    if (ends_with(model_path, ".onnx")) {
        return ModelKind::YoloV8Fused;
    }
    throw std::runtime_error("Unsupported model extension: " + model_path);
}

// ============ Preprocessing (Optimized) ============

PreprocessMeta letterbox_resize(const cv::Mat& img, cv::Mat& output,
                                 int target_w, int target_h, int stride, uint8_t fill_value) {
    int w = img.cols;
    int h = img.rows;

    float r = std::min((float)target_h / h, (float)target_w / w);
    int new_w = std::floor(w * r);
    int new_h = std::floor(h * r);

    cv::Mat resized;
    if (new_w != w || new_h != h) {
        cv::resize(img, resized, cv::Size(new_w, new_h));
    } else {
        resized = img;  // No copy needed
    }

    int dw = target_w - new_w;
    int dh = target_h - new_h;

    // Note: No stride alignment needed when target size is fixed
    // The target_w and target_h are already multiples of stride

    int top = dh / 2;
    int bottom = dh - top;
    int left = dw / 2;
    int right = dw - left;

    cv::copyMakeBorder(resized, output, top, bottom, left, right,
                      cv::BORDER_CONSTANT, cv::Scalar(fill_value, fill_value, fill_value));

    return {r, left, top, w, h};
}

void hwc_to_blob(const cv::Mat& padded, std::vector<float>& blob,
                 inference::Layout layout, const PreprocessParams& params,
                 bool pre_allocated = false) {
    const int H = padded.rows;
    const int W = padded.cols;
    const int HW = H * W;

    size_t required = static_cast<size_t>(HW) * 3;
    if (!pre_allocated || blob.size() != required) {
        blob.resize(required);
    }

    const uint8_t* src = padded.data;

    for (int i = 0; i < H; ++i) {
        const uint8_t* row = src + i * W * 3;
        for (int j = 0; j < W; ++j) {
            const int idx = i * W + j;
            float b = row[j * 3 + 0];
            float g = row[j * 3 + 1];
            float r = row[j * 3 + 2];

            float c0 = params.swap_rb ? r : b;
            float c1 = g;
            float c2 = params.swap_rb ? b : r;

            if (layout == inference::Layout::NHWC) {
                size_t base = static_cast<size_t>(idx) * 3;
                blob[base + 0] = (c0 - params.mean[0]) * params.scale[0];
                blob[base + 1] = (c1 - params.mean[1]) * params.scale[1];
                blob[base + 2] = (c2 - params.mean[2]) * params.scale[2];
            } else {
                blob[0 * HW + idx] = (c0 - params.mean[0]) * params.scale[0];
                blob[1 * HW + idx] = (c1 - params.mean[1]) * params.scale[1];
                blob[2 * HW + idx] = (c2 - params.mean[2]) * params.scale[2];
            }
        }
    }
}

// ============ Postprocessing ============

float compute_iou(const Detection& a, const Detection& b) {
    float a_x1 = a.x, a_y1 = a.y, a_x2 = a.x + a.w, a_y2 = a.y + a.h;
    float b_x1 = b.x, b_y1 = b.y, b_x2 = b.x + b.w, b_y2 = b.y + b.h;

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

std::vector<Detection> nms(std::vector<Detection>& boxes, float iou_thres) {
    std::sort(boxes.begin(), boxes.end(), [](const Detection& a, const Detection& b) {
        return a.score > b.score;
    });

    std::vector<bool> suppressed(boxes.size(), false);
    std::vector<Detection> result;

    for (size_t i = 0; i < boxes.size(); ++i) {
        if (suppressed[i]) continue;
        result.push_back(boxes[i]);

        for (size_t j = i + 1; j < boxes.size(); ++j) {
            if (suppressed[j]) continue;
            if (compute_iou(boxes[i], boxes[j]) > iou_thres) {
                suppressed[j] = true;
            }
        }
    }
    return result;
}

float clamp_value(float v, float lo, float hi) {
    return std::max(lo, std::min(v, hi));
}

void restore_coords(Detection& det, const PreprocessMeta& meta) {
    float x1 = (det.x - meta.pad_x) / meta.scale;
    float y1 = (det.y - meta.pad_y) / meta.scale;
    float x2 = x1 + det.w / meta.scale;
    float y2 = y1 + det.h / meta.scale;

    float max_w = static_cast<float>(meta.ori_w);
    float max_h = static_cast<float>(meta.ori_h);

    x1 = clamp_value(x1, 0.0f, max_w);
    y1 = clamp_value(y1, 0.0f, max_h);
    x2 = clamp_value(x2, 0.0f, max_w);
    y2 = clamp_value(y2, 0.0f, max_h);

    det.x = x1;
    det.y = y1;
    det.w = std::max(0.0f, x2 - x1);
    det.h = std::max(0.0f, y2 - y1);
}

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

struct YoloV8Head {
    int h = 0;
    int w = 0;
    int stride = 0;
    const float* reg = nullptr;
    const float* cls = nullptr;
};

struct YoloV8Params {
    int num_classes = 80;
    int reg_max = 16;
    std::array<int, 3> strides = {8, 16, 32};

    int reg_channels() const { return reg_max * 4; }
};

float dfl_distance(const float* reg_ptr, int side, int reg_max, int idx, int hw) {
    if (reg_max <= 1) {
        return 0.0f;
    }
    const float* base = reg_ptr + side * reg_max * hw;
    float max_val = -std::numeric_limits<float>::infinity();
    for (int k = 0; k < reg_max; ++k) {
        float v = base[k * hw + idx];
        if (v > max_val) {
            max_val = v;
        }
    }
    float sum = 0.0f;
    float acc = 0.0f;
    for (int k = 0; k < reg_max; ++k) {
        float v = base[k * hw + idx];
        float exp_v = std::exp(v - max_val);
        sum += exp_v;
        acc += exp_v * static_cast<float>(k);
    }
    if (sum <= 0.0f) {
        return 0.0f;
    }
    return acc / sum;
}

std::vector<YoloV8Head> collect_yolov8_heads(
    const std::vector<std::vector<float>>& outputs,
    const std::vector<std::vector<int64_t>>& shapes,
    int input_w,
    int input_h,
    const YoloV8Params& params) {
    if (outputs.size() != shapes.size()) {
        throw std::runtime_error("YOLOv8 head output count mismatch");
    }
    if (outputs.size() < params.strides.size() * 2) {
        throw std::runtime_error("YOLOv8 head outputs missing");
    }

    struct HeadSlot {
        int stride = 0;
        int h = 0;
        int w = 0;
        const float* reg = nullptr;
        const float* cls = nullptr;
    };

    std::vector<HeadSlot> slots;
    slots.reserve(params.strides.size());
    for (int stride : params.strides) {
        if (stride <= 0 || input_w % stride != 0 || input_h % stride != 0) {
            throw std::runtime_error("YOLOv8 stride not aligned with input size");
        }
        slots.push_back({stride, input_h / stride, input_w / stride, nullptr, nullptr});
    }

    int reg_channels = params.reg_channels();
    for (size_t i = 0; i < shapes.size(); ++i) {
        const auto& shape = shapes[i];
        if (shape.size() != 4 || shape[0] != 1) {
            throw std::runtime_error("YOLOv8 head output rank mismatch");
        }
        int c = static_cast<int>(shape[1]);
        int h = static_cast<int>(shape[2]);
        int w = static_cast<int>(shape[3]);

        HeadSlot* slot = nullptr;
        for (auto& s : slots) {
            if (s.h == h && s.w == w) {
                slot = &s;
                break;
            }
        }
        if (!slot) {
            throw std::runtime_error("YOLOv8 head output shape mismatch");
        }
        if (c == reg_channels) {
            if (slot->reg) {
                throw std::runtime_error("YOLOv8 head regression duplicate");
            }
            slot->reg = outputs[i].data();
        } else if (c == params.num_classes) {
            if (slot->cls) {
                throw std::runtime_error("YOLOv8 head class duplicate");
            }
            slot->cls = outputs[i].data();
        } else {
            throw std::runtime_error("YOLOv8 head channel mismatch");
        }
    }

    std::vector<YoloV8Head> heads;
    heads.reserve(slots.size());
    for (const auto& slot : slots) {
        if (!slot.reg || !slot.cls) {
            throw std::runtime_error("YOLOv8 head missing tensors");
        }
        heads.push_back({slot.h, slot.w, slot.stride, slot.reg, slot.cls});
    }

    return heads;
}

std::vector<Detection> postprocess_yolov8_heads(
    const std::vector<YoloV8Head>& heads,
    const YoloV8Params& params,
    int input_w,
    int input_h,
    const PreprocessMeta& meta,
    float conf_thres,
    float iou_thres) {
    std::vector<std::vector<Detection>> per_class(
        static_cast<size_t>(params.num_classes));
    float inverse_th = std::log(conf_thres / (1.0f - conf_thres));
    float input_w_f = static_cast<float>(input_w);
    float input_h_f = static_cast<float>(input_h);

    for (const auto& head : heads) {
        int hw = head.h * head.w;
        for (int idx = 0; idx < hw; ++idx) {
            float max_logit = -std::numeric_limits<float>::infinity();
            int best_class = -1;
            for (int c = 0; c < params.num_classes; ++c) {
                float logit = head.cls[c * hw + idx];
                if (logit > max_logit) {
                    max_logit = logit;
                    best_class = c;
                }
            }
            if (max_logit < inverse_th || best_class < 0) {
                continue;
            }
            float best_score = sigmoid(max_logit);

            float left = dfl_distance(head.reg, 0, params.reg_max, idx, hw) * head.stride;
            float top = dfl_distance(head.reg, 1, params.reg_max, idx, hw) * head.stride;
            float right = dfl_distance(head.reg, 2, params.reg_max, idx, hw) * head.stride;
            float bottom = dfl_distance(head.reg, 3, params.reg_max, idx, hw) * head.stride;

            int x = idx % head.w;
            int y = idx / head.w;
            float cx = (static_cast<float>(x) + 0.5f) * head.stride;
            float cy = (static_cast<float>(y) + 0.5f) * head.stride;

            float x1 = std::max(0.0f, std::min(cx - left, input_w_f));
            float y1 = std::max(0.0f, std::min(cy - top, input_h_f));
            float x2 = std::max(0.0f, std::min(cx + right, input_w_f));
            float y2 = std::max(0.0f, std::min(cy + bottom, input_h_f));

            if (x2 <= x1 || y2 <= y1) {
                continue;
            }

            Detection det{
                x1,
                y1,
                x2 - x1,
                y2 - y1,
                best_score,
                best_class
            };
            per_class[static_cast<size_t>(best_class)].push_back(det);
        }
    }

    std::vector<Detection> final_boxes;
    for (auto& class_boxes : per_class) {
        if (class_boxes.empty()) {
            continue;
        }
        auto class_nms = nms(class_boxes, iou_thres);
        final_boxes.insert(final_boxes.end(), class_nms.begin(), class_nms.end());
    }

    for (auto& det : final_boxes) {
        restore_coords(det, meta);
    }
    return final_boxes;
}

std::vector<Detection> postprocess_yolov8_fused(
    const std::vector<float>& output,
    const std::vector<int64_t>& shape,
    const PreprocessMeta& meta,
    float conf_thres,
    float iou_thres) {
    if (shape.size() != 3 || shape[0] != 1 || shape[1] != 84) {
        throw std::runtime_error("YOLOv8 fused output shape mismatch");
    }
    int num_boxes = static_cast<int>(shape[2]);
    int box_dim = static_cast<int>(shape[1]);
    if (static_cast<int64_t>(output.size()) != shape[1] * shape[2]) {
        throw std::runtime_error("YOLOv8 fused output size mismatch");
    }

    const float* cx_ptr = output.data() + 0 * num_boxes;
    const float* cy_ptr = output.data() + 1 * num_boxes;
    const float* w_ptr = output.data() + 2 * num_boxes;
    const float* h_ptr = output.data() + 3 * num_boxes;

    std::vector<Detection> proposals;
    proposals.reserve(static_cast<size_t>(num_boxes / 10));

    int num_classes = box_dim - 4;
    for (int i = 0; i < num_boxes; ++i) {
        float max_cls_conf = 0.0f;
        int cls_id = 0;
        for (int c = 0; c < num_classes; ++c) {
            float conf = output[(4 + c) * num_boxes + i];
            if (conf > max_cls_conf) {
                max_cls_conf = conf;
                cls_id = c;
            }
        }
        if (max_cls_conf < conf_thres) {
            continue;
        }

        Detection det{
            cx_ptr[i] - w_ptr[i] * 0.5f,
            cy_ptr[i] - h_ptr[i] * 0.5f,
            w_ptr[i],
            h_ptr[i],
            max_cls_conf,
            cls_id
        };
        proposals.push_back(det);
    }

    auto final_boxes = nms(proposals, iou_thres);
    for (auto& det : final_boxes) {
        restore_coords(det, meta);
    }
    return final_boxes;
}

// ============ Visualization ============

void print_results(const std::vector<Detection>& results) {
    std::cout << "\n=== Detection Results ===\n";
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& det = results[i];
        std::string label = (det.class_id >= 0 && det.class_id < COCO_LABELS.size())
                          ? COCO_LABELS[det.class_id] : "unknown";
        std::cout << "Box " << (i+1) << ": " << label << " "
                  << "(" << det.x << ", " << det.y << ", " << det.w << ", " << det.h << ") "
                  << "conf=" << det.score << "\n";
    }
    std::cout << "Total: " << results.size() << " detections\n";
}

void draw_detections(cv::Mat& frame, const std::vector<Detection>& detections) {
    for (const auto& det : detections) {
        cv::rectangle(frame, cv::Rect(det.x, det.y, det.w, det.h),
                     cv::Scalar(0, 255, 0), 2);
        std::string label = (det.class_id >= 0 && det.class_id < COCO_LABELS.size())
                          ? COCO_LABELS[det.class_id] : "unknown";
        std::string text = label + " " + std::to_string(det.score).substr(0, 4);
        cv::putText(frame, text, cv::Point(det.x, det.y - 5),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }
}

// ============ Utilities ============

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " <model.onnx|model.cvimodel> <input> [show] [save=output.png]\n";
    std::cout << "\nInput: image (.jpg, .png)\n";
    std::cout << "Options:\n";
    std::cout << "  show         - Display results\n";
    std::cout << "  save=FILE    - Save output image\n";
    std::cout << "  profile      - Print detailed timing breakdown\n";
    std::cout << "\nModel expectations:\n";
    std::cout << "  .cvimodel: YOLOv8 head outputs (6 tensors: 3 reg + 3 cls)\n";
    std::cout << "  .onnx: YOLOv8 fused output with shape [1,84,N]\n";
}

// ============ Main ============

int main(int argc, char* argv[]) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    std::string model_path = argv[1];
    std::string input_path = argv[2];

    bool show_result = false;
    std::string save_path = "";
    bool profile = false;
    float conf_thres = CONF_THRES;
    float iou_thres = IOU_THRES;

    // Parse options
    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "show") {
            show_result = true;
        } else if (arg == "profile") {
            profile = true;
        } else if (arg.find("save=") == 0) {
            save_path = arg.substr(5);
        }
    }

#if defined(USE_CVI_TPU)
    if (show_result) {
        std::cout << "[WARN] Display not supported on SG200X SDK, disabling show.\n";
        show_result = false;
    }
#endif

    try {
        // Load model
        std::cout << "Loading model: " << model_path << "\n";
        auto session = inference::create_session(model_path, 4);
        std::cout << "Backend: " << session->backend_name() << "\n";

        auto model_input_shape = session->get_input_shape(0);
        auto input_layout = inference::infer_layout(model_input_shape);
        if (input_layout == inference::Layout::Unknown) {
            input_layout = inference::Layout::NCHW;
        }

        int input_h = INPUT_H;
        int input_w = INPUT_W;
        if (model_input_shape.size() == 4) {
            if (input_layout == inference::Layout::NCHW) {
                if (model_input_shape[2] > 0) input_h = static_cast<int>(model_input_shape[2]);
                if (model_input_shape[3] > 0) input_w = static_cast<int>(model_input_shape[3]);
            } else if (input_layout == inference::Layout::NHWC) {
                if (model_input_shape[1] > 0) input_h = static_cast<int>(model_input_shape[1]);
                if (model_input_shape[2] > 0) input_w = static_cast<int>(model_input_shape[2]);
            }
        }

        std::vector<int64_t> run_shape = (input_layout == inference::Layout::NHWC)
            ? std::vector<int64_t>{1, input_h, input_w, 3}
            : std::vector<int64_t>{1, 3, input_h, input_w};

        ModelKind model_kind = model_kind_from_path(model_path);
        YoloV8Params yolo_params;

#ifdef USE_CVI_TPU
        auto* cvi_session = dynamic_cast<inference::CviSession*>(session.get());
        if (model_kind == ModelKind::YoloV8Head && !cvi_session) {
            throw std::runtime_error("CVI model requires TPU backend");
        }
        if (model_kind == ModelKind::YoloV8Head &&
            cvi_session->output_count() != static_cast<int>(yolo_params.strides.size() * 2)) {
            throw std::runtime_error("YOLOv8 head model expects 6 output tensors");
        }
#else
        if (model_kind == ModelKind::YoloV8Head) {
            throw std::runtime_error("CVI model requires TPU backend");
        }
#endif

        PreprocessParams preprocess_params =
            (model_kind == ModelKind::YoloV8Head) ? tdl_yolov8_preprocess_params()
                                                  : default_preprocess_params();

        std::cout << "Model input shape: " << shape_to_string(model_input_shape)
                  << " layout=" << inference::layout_name(input_layout) << "\n";
        std::cout << "Preprocess: swap_rb=" << (preprocess_params.swap_rb ? "true" : "false")
                  << " mean=(" << preprocess_params.mean[0] << ", "
                  << preprocess_params.mean[1] << ", " << preprocess_params.mean[2] << ")"
                  << " scale=(" << preprocess_params.scale[0] << ", "
                  << preprocess_params.scale[1] << ", " << preprocess_params.scale[2] << ")"
                  << " pad=" << preprocess_params.pad_value
                  << ((model_kind == ModelKind::YoloV8Head) ? " (tdl_yolov8)" : "") << "\n";
        std::cout << "Postprocess: conf=" << std::fixed << std::setprecision(2) << conf_thres
                  << " iou=" << iou_thres << "\n";

        // Lambda for inference (image mode)
        auto infer_func = [&](const cv::Mat& frame, TimingBreakdown* timing) -> std::vector<Detection> {
            auto t0 = std::chrono::high_resolution_clock::now();
            // 1. Preprocess
            cv::Mat padded;
            auto meta = letterbox_resize(frame, padded, input_w, input_h, STRIDE,
                                         static_cast<uint8_t>(preprocess_params.pad_value));
            auto t1 = std::chrono::high_resolution_clock::now();
            std::vector<float> blob;
            hwc_to_blob(padded, blob, input_layout, preprocess_params);
            auto t2 = std::chrono::high_resolution_clock::now();

            std::vector<Detection> results;
            std::chrono::high_resolution_clock::time_point t3;
            if (model_kind == ModelKind::YoloV8Head) {
#ifdef USE_CVI_TPU
                std::vector<std::vector<float>> outputs;
                std::vector<std::vector<int64_t>> shapes;
                cvi_session->run_all(blob.data(), run_shape, &outputs, &shapes);
                t3 = std::chrono::high_resolution_clock::now();
                auto heads = collect_yolov8_heads(outputs, shapes, input_w, input_h, yolo_params);
                results = postprocess_yolov8_heads(
                    heads, yolo_params, input_w, input_h, meta, conf_thres, iou_thres);
#else
                throw std::runtime_error("CVI model requires TPU backend");
#endif
            } else {
                auto out_pair = session->run(blob.data(), run_shape);
                t3 = std::chrono::high_resolution_clock::now();
                results = postprocess_yolov8_fused(
                    out_pair.first, out_pair.second, meta, conf_thres, iou_thres);
            }
            auto t4 = std::chrono::high_resolution_clock::now();
            if (timing) {
                timing->preprocess_ms =
                    std::chrono::duration<double, std::milli>(t2 - t0).count();
                timing->inference_ms =
                    std::chrono::duration<double, std::milli>(t3 - t2).count();
                timing->postprocess_ms =
                    std::chrono::duration<double, std::milli>(t4 - t3).count();
                timing->resize_ms =
                    std::chrono::duration<double, std::milli>(t1 - t0).count();
                timing->blob_ms =
                    std::chrono::duration<double, std::milli>(t2 - t1).count();
#ifdef USE_CVI_TPU
                if (model_kind == ModelKind::YoloV8Head) {
                    const auto& stats = cvi_session->last_run_stats();
                    timing->cvi_input_ms = stats.input_ms;
                    timing->cvi_forward_ms = stats.forward_ms;
                    timing->cvi_output_ms = stats.output_ms;
                }
#endif
            }
            return results;
        };

        // ========== Image inference ==========
        std::cout << "Loading image: " << input_path << "\n";
        cv::Mat img = cv::imread(input_path);
        if (img.empty()) {
            throw std::runtime_error("Failed to load image");
        }
        std::cout << "Image size: " << img.cols << "x" << img.rows << "\n\n";

        TimingBreakdown timing;
        auto results = infer_func(img, &timing);
        double total_ms = timing.preprocess_ms + timing.inference_ms + timing.postprocess_ms;
        std::cout << "Timing (ms): preprocess=" << std::fixed << std::setprecision(2)
                  << timing.preprocess_ms << ", inference=" << timing.inference_ms
                  << ", postprocess=" << timing.postprocess_ms
                  << ", total=" << total_ms << "\n";
        if (profile) {
            std::cout << "  preprocess detail (ms): resize=" << timing.resize_ms
                      << ", blob=" << timing.blob_ms << "\n";
#ifdef USE_CVI_TPU
            if (model_kind == ModelKind::YoloV8Head) {
                std::cout << "  cvi detail (ms): input=" << timing.cvi_input_ms
                          << ", forward=" << timing.cvi_forward_ms
                          << ", output=" << timing.cvi_output_ms << "\n";
            }
#endif
        }

        print_results(results);

        if (show_result || !save_path.empty()) {
            draw_detections(img, results);

            if (!save_path.empty()) {
                cv::imwrite(save_path, img);
                std::cout << "\nResult saved: " << save_path << "\n";
            }

#if !defined(USE_CVI_TPU)
            if (show_result) {
                cv::imshow("Result", img);
                std::cout << "Press any key to exit...\n";
                cv::waitKey(0);
            }
#endif
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
