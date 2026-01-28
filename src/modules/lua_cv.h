#ifndef MODEL_INFER_LUA_CV_H_
#define MODEL_INFER_LUA_CV_H_

#include <opencv2/opencv.hpp>
#include <functional>
#include <map>
#include <string>
#include "LuaIntf.h"

// Use Tensor from lua_nn.h
#include "lua_nn.h"

// Use Frame from cv module
#include "cv/frame.h"

namespace lua_cv {

class Image {
public:
    explicit Image(lua_cv::Frame&& frame);
    Image();

    // Copy operations (uses Frame::clone())
    Image(const Image& other);
    Image& operator=(const Image& other);

    // Property access (via getters, not direct member exposure)
    int width() const;
    int height() const;
    int channels() const;
    bool empty() const;

    // Image operations (modify in-place)
    void resize(int new_w, int new_h);
    void pad(int top, int bottom, int left, int right, int fill_value);

    // Return Tensor object
    lua_nn::Tensor to_tensor(double scale,
                             const LuaIntf::LuaRef& mean,
                             const LuaIntf::LuaRef& std) const;

    // Utility methods
    Image clone() const;

    // Internal access (C++ only)
    const lua_cv::Frame& data() const { return frame_; }
    lua_cv::Frame& data() { return frame_; }

private:
    lua_cv::Frame frame_;
};

// 全局函数
Image imread(const std::string& path);

// ========== 预处理注册表 ==========

// 预处理结果：Tensor + Meta信息
struct PreprocessResult {
    lua_nn::Tensor tensor;
    LuaIntf::LuaRef meta;

    PreprocessResult(lua_nn::Tensor t, LuaIntf::LuaRef m)
        : tensor(std::move(t)), meta(m) {}
};

// 预处理函数签名
using PreprocessFunc = std::function<PreprocessResult(
    Image& img,
    lua_State* L,
    const LuaIntf::LuaRef& config
)>;

// 预处理函数注册表
class PreprocessRegistry {
public:
    static PreprocessRegistry& instance();

    // 注册预处理函数
    void register_func(const std::string& type, PreprocessFunc func);

    // 执行预处理
    PreprocessResult run(
        const std::string& type,
        Image& img,
        lua_State* L,
        const LuaIntf::LuaRef& config
    );

    // 检查是否支持某类型
    bool has(const std::string& type) const;

private:
    PreprocessRegistry();
    std::map<std::string, PreprocessFunc> registry_;
};

// 预定义的预处理函数
PreprocessResult preprocess_letterbox(
    Image& img,
    lua_State* L,
    const LuaIntf::LuaRef& config
);

PreprocessResult preprocess_resize_center_crop(
    Image& img,
    lua_State* L,
    const LuaIntf::LuaRef& config
);

// 注册到Lua
void register_module(lua_State* L);

} // namespace lua_cv

#endif
