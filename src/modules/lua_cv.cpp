#include "lua_cv.h"
#include "lua_nn.h"
#include "cv/cv_helpers.h"
#include <cmath>
#include <stdexcept>

namespace lua_cv {

Image::Image(lua_cv::Frame&& frame) : frame_(std::move(frame)) {}
Image::Image() {}

Image::Image(const Image& other) : frame_(other.frame_.clone()) {}

Image& Image::operator=(const Image& other) {
    if (this != &other) {
        frame_ = other.frame_.clone();
    }
    return *this;
}

int Image::width() const {
    return frame_.width();
}

int Image::height() const {
    return frame_.height();
}

int Image::channels() const {
    return frame_.channels();
}

bool Image::empty() const {
    return frame_.empty();
}

void Image::resize(int new_w, int new_h) {
    cv_helpers::resize(frame_, new_w, new_h);
}

void Image::pad(int top, int bottom, int left, int right, int fill_value) {
    // VPSS doesn't support pad - must use OpenCV CPU
    // Convert to Mat, apply padding, wrap back to Frame
    cv::Mat mat = frame_.to_mat().clone();
    cv::copyMakeBorder(mat, mat, top, bottom, left, right,
                       cv::BORDER_CONSTANT,
                       cv::Scalar(fill_value, fill_value, fill_value));
    frame_ = lua_cv::Frame(mat);
}

lua_nn::Tensor Image::to_tensor(double scale,
                                 const LuaIntf::LuaRef& mean,
                                 const LuaIntf::LuaRef& std) const {
    // Extract mean/std vectors from Lua tables
    std::vector<double> mean_vec, std_vec;

    if (mean.isTable()) {
        int len = mean.len();
        for (int i = 1; i <= len; ++i) {
            mean_vec.push_back(mean.get<double>(i));
        }
    }

    if (std.isTable()) {
        int len = std.len();
        for (int i = 1; i <= len; ++i) {
            std_vec.push_back(std.get<double>(i));
        }
    }

    // Call cv_helpers::frame_to_tensor
    return cv_helpers::frame_to_tensor(frame_, scale, mean_vec, std_vec);
}

Image Image::clone() const {
    return Image(frame_.clone());
}

Image imread(const std::string& path) {
    cv::Mat mat = cv::imread(path, cv::IMREAD_COLOR);
    if (mat.empty()) {
        throw std::runtime_error("Failed to load image: " + path);
    }
    // Convert BGR to RGB (OpenCV loads BGR, models expect RGB)
    cv::cvtColor(mat, mat, cv::COLOR_BGR2RGB);

    // Create Frame from Mat, then move into Image
    lua_cv::Frame frame(mat);
    return Image(std::move(frame));
}

// ========== PreprocessRegistry ==========

PreprocessRegistry& PreprocessRegistry::instance() {
    static PreprocessRegistry inst;
    return inst;
}

PreprocessRegistry::PreprocessRegistry() {
    register_func("letterbox", preprocess_letterbox);
    register_func("resize_center_crop", preprocess_resize_center_crop);
}

void PreprocessRegistry::register_func(const std::string& type, PreprocessFunc func) {
    registry_[type] = std::move(func);
}

PreprocessResult PreprocessRegistry::run(
    const std::string& type,
    Image& img,
    lua_State* L,
    const LuaIntf::LuaRef& config)
{
    auto it = registry_.find(type);
    if (it == registry_.end()) {
        throw std::runtime_error("Unknown preprocess type: " + type);
    }
    return it->second(img, L, config);
}

bool PreprocessRegistry::has(const std::string& type) const {
    return registry_.count(type) > 0;
}

// ========== preprocess_letterbox ==========
// C++ implementation matching scripts/lib/preprocess.lua:letterbox()

PreprocessResult preprocess_letterbox(
    Image& img,
    lua_State* L,
    const LuaIntf::LuaRef& config)
{
    // Parse config
    LuaIntf::LuaRef input_size = config.get<LuaIntf::LuaRef>("input_size");
    int target_h = input_size.get<int>(1);
    int target_w = input_size.get<int>(2);

    int stride = config.has("stride") ? config.get<int>("stride") : 32;
    int fill_value = config.has("fill_value") ? config.get<int>("fill_value") : 114;

    int ori_w = img.width();
    int ori_h = img.height();

    // Calculate scale ratio (maintain aspect ratio)
    float r = std::min(static_cast<float>(target_h) / ori_h,
                       static_cast<float>(target_w) / ori_w);
    int new_w = static_cast<int>(std::floor(ori_w * r));
    int new_h = static_cast<int>(std::floor(ori_h * r));

    // Resize
    if (new_w != ori_w || new_h != ori_h) {
        img.resize(new_w, new_h);
    }

    // Calculate padding (align to stride)
    int dw = (target_w - new_w) % stride;
    int dh = (target_h - new_h) % stride;

    int top = dh / 2;
    int bottom = dh - top;
    int left = dw / 2;
    int right = dw - left;

    // Add padding
    img.pad(top, bottom, left, right, fill_value);

    // Convert to tensor (scale=1/255, mean={0,0,0}, std={1,1,1})
    double scale = 1.0 / 255.0;
    std::vector<double> mean_vec = {0.0, 0.0, 0.0};
    std::vector<double> std_vec = {1.0, 1.0, 1.0};
    lua_nn::Tensor tensor = cv_helpers::frame_to_tensor(img.data(), scale, mean_vec, std_vec);

    // Build meta table
    LuaIntf::LuaRef meta = LuaIntf::LuaRef::createTable(L);
    meta["scale"] = r;
    meta["pad_x"] = left;
    meta["pad_y"] = top;
    meta["ori_w"] = ori_w;
    meta["ori_h"] = ori_h;

    return PreprocessResult(std::move(tensor), meta);
}

// ========== preprocess_resize_center_crop ==========
// Resize to fill target then center-crop (used for classification models)

PreprocessResult preprocess_resize_center_crop(
    Image& img,
    lua_State* L,
    const LuaIntf::LuaRef& config)
{
    LuaIntf::LuaRef input_size = config.get<LuaIntf::LuaRef>("input_size");
    int target_h = input_size.get<int>(1);
    int target_w = input_size.get<int>(2);

    int ori_w = img.width();
    int ori_h = img.height();

    // Scale to fill target (larger dimension matches target)
    float r = std::max(static_cast<float>(target_h) / ori_h,
                       static_cast<float>(target_w) / ori_w);
    int new_w = static_cast<int>(std::ceil(ori_w * r));
    int new_h = static_cast<int>(std::ceil(ori_h * r));

    img.resize(new_w, new_h);

    // Center crop to target size
    int crop_x = (new_w - target_w) / 2;
    int crop_y = (new_h - target_h) / 2;

    cv::Mat mat = img.data().to_mat();
    cv::Rect roi(crop_x, crop_y, target_w, target_h);
    cv::Mat cropped = mat(roi).clone();
    img = Image(lua_cv::Frame(cropped));

    // Convert to tensor (scale=1/255, mean={0,0,0}, std={1,1,1})
    double scale_val = 1.0 / 255.0;
    std::vector<double> mean_vec = {0.0, 0.0, 0.0};
    std::vector<double> std_vec = {1.0, 1.0, 1.0};

    // Override with config if provided
    if (config.has("scale")) {
        scale_val = config.get<double>("scale");
    }
    if (config.has("mean")) {
        LuaIntf::LuaRef m = config.get<LuaIntf::LuaRef>("mean");
        mean_vec.clear();
        for (int i = 1; i <= m.len(); ++i) mean_vec.push_back(m.get<double>(i));
    }
    if (config.has("std")) {
        LuaIntf::LuaRef s = config.get<LuaIntf::LuaRef>("std");
        std_vec.clear();
        for (int i = 1; i <= s.len(); ++i) std_vec.push_back(s.get<double>(i));
    }

    lua_nn::Tensor tensor = cv_helpers::frame_to_tensor(img.data(), scale_val, mean_vec, std_vec);

    LuaIntf::LuaRef meta = LuaIntf::LuaRef::createTable(L);
    meta["scale"] = r;
    meta["crop_x"] = crop_x;
    meta["crop_y"] = crop_y;
    meta["ori_w"] = ori_w;
    meta["ori_h"] = ori_h;

    return PreprocessResult(std::move(tensor), meta);
}

void register_module(lua_State* L) {
    using namespace LuaIntf;
    
    LuaBinding(L)
        .beginModule("lua_cv")
            .addFactory(imread)  // 全局函数
            .beginClass<Image>("Image")
                .addConstructor(LUA_ARGS())  // 默认构造
                // 使用addProperty封装属性（非addFunction）
                .addProperty("width", &Image::width)
                .addProperty("height", &Image::height)
                .addProperty("channels", &Image::channels)
                .addFunction("empty", &Image::empty)
                .addFunction("resize", &Image::resize)
                .addFunction("pad", &Image::pad)
                .addFunction("clone", &Image::clone)
                .addFunction("to_tensor", &Image::to_tensor)
            .endClass()
        .endModule();
}

} // namespace lua_cv
