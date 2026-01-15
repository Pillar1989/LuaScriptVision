#include "lua_cv.h"
#include "lua_nn.h"
#include <stdexcept>

namespace lua_cv {

Image::Image(const cv::Mat& mat) : mat_(mat) {}
Image::Image() {}

void Image::resize(int new_w, int new_h) {
    // 必须使用 cv::resize
    // 插值方法：cv::INTER_LINEAR（默认）
    cv::resize(mat_, mat_, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
}

void Image::pad(int top, int bottom, int left, int right, int fill_value) {
    // 必须使用 cv::copyMakeBorder
    // 边界类型：cv::BORDER_CONSTANT
    cv::copyMakeBorder(mat_, mat_, top, bottom, left, right, 
                       cv::BORDER_CONSTANT, cv::Scalar(fill_value, fill_value, fill_value));
}

lua_nn::Tensor Image::to_tensor(double scale,
                                 const LuaIntf::LuaRef& mean,
                                 const LuaIntf::LuaRef& std) const {
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

    // 1. 转换为浮点型
    cv::Mat float_mat;
    mat_.convertTo(float_mat, CV_32F);
    
    // 2. HWC -> CHW 转换（使用cv::split优化，比三重循环快10倍）
    int H = float_mat.rows;
    int W = float_mat.cols;
    int C = float_mat.channels();
    
    // Ensure mean/std have enough elements
    if (mean_vec.size() < C) mean_vec.resize(C, 0.0);
    if (std_vec.size() < C) std_vec.resize(C, 1.0);

    std::vector<cv::Mat> channels(C);
    cv::split(float_mat, channels);
    
    // 3. 分通道归一化并组装CHW数据
    std::vector<float> chw_data(C * H * W);
    size_t idx = 0;
    
    for (int c = 0; c < C; ++c) {
        const float* channel_ptr = channels[c].ptr<float>();
        double m = mean_vec[c];
        double s = std_vec[c];
        for (int i = 0; i < H * W; ++i) {
            chw_data[idx++] = (channel_ptr[i] * scale - m) / s;
        }
    }
    
    // 4. 创建Tensor对象（NCHW格式）
    std::vector<int64_t> shape = {1, static_cast<int64_t>(C), 
                                   static_cast<int64_t>(H), 
                                   static_cast<int64_t>(W)};
    return lua_nn::Tensor(std::move(chw_data), shape);
}

Image Image::clone() const {
    return Image(mat_.clone());
}

Image imread(const std::string& path) {
    cv::Mat mat = cv::imread(path, cv::IMREAD_COLOR);
    if (mat.empty()) {
        throw std::runtime_error("Failed to load image: " + path);
    }
    // Convert BGR to RGB (OpenCV loads BGR, models expect RGB)
    cv::cvtColor(mat, mat, cv::COLOR_BGR2RGB);
    return Image(mat);
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
