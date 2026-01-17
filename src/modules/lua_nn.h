#ifndef MODEL_INFER_LUA_NN_H_
#define MODEL_INFER_LUA_NN_H_

#include <vector>
#include <memory>
#include <string>

#include "LuaIntf.h"
#include "inference/inference.h"

// 使用 tensor 模块的 Tensor 类
#include "tensor/tensor.h"

namespace lua_nn {

// 使用 tensor::Tensor 作为 lua_nn::Tensor
using Tensor = tensor::Tensor;

#ifdef USE_ONNX_RUNTIME

class Session {
public:
    explicit Session(const std::string& model_path, int num_threads = 4);

    // 推理方法（接受Tensor对象）
    LuaIntf::LuaRef run(lua_State* L, const Tensor& input_tensor);

    // 属性访问
    std::vector<std::string> input_names() const;
    std::vector<std::string> output_names() const;
    std::vector<int64_t> input_shape(size_t index = 0) const;
    std::vector<int64_t> output_shape(size_t index = 0) const;

private:
    std::unique_ptr<inference::OnnxSession> session_;
};

#endif  // USE_ONNX_RUNTIME

// 注册到Lua
void register_module(lua_State* L);

} // namespace lua_nn

#endif
