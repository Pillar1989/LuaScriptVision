#include "lua_cv.h"
#include "lua_nn.h"
#include "cv/cv_helpers.h"
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
