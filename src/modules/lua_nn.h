#ifndef MODEL_INFER_LUA_NN_H_
#define MODEL_INFER_LUA_NN_H_

#include <vector>
#include <memory>
#include <string>

#include "LuaIntf.h"
#ifdef USE_ONNX_RUNTIME
#include "inference/onnx_session.h"
#endif
#ifdef USE_CVI_TPU
#include "inference/cvi_session.h"
#endif

// 使用 tensor 模块的 Tensor 类
#include "tensor/tensor.h"

namespace lua_nn {

// 使用 tensor::Tensor 作为 lua_nn::Tensor
using Tensor = tensor::Tensor;

#ifdef USE_ONNX_RUNTIME

class Session {
public:
    explicit Session(const std::string& model_path, int num_threads = 4);

    LuaIntf::LuaRef run(lua_State* L, const Tensor& input_tensor);

    std::vector<std::string> input_names() const;
    std::vector<std::string> output_names() const;
    std::vector<int64_t> input_shape(size_t index = 0) const;
    std::vector<int64_t> output_shape(size_t index = 0) const;

private:
    std::unique_ptr<inference::OnnxSession> session_;
};

#endif  // USE_ONNX_RUNTIME

#ifdef USE_CVI_TPU

class TpuSession {
public:
    explicit TpuSession(const std::string& model_path);

    // Run inference with float tensor input (CPU path)
    LuaIntf::LuaRef run(lua_State* L, const Tensor& input_tensor);

    // Run inference with all outputs
    LuaIntf::LuaRef run_all(lua_State* L, const Tensor& input_tensor);

    std::vector<std::string> input_names() const;
    std::vector<std::string> output_names() const;
    std::vector<int64_t> input_shape(size_t index = 0) const;
    std::vector<int64_t> output_shape(size_t index = 0) const;
    int32_t output_count() const;
    std::string backend_name() const;

    inference::CviSession* raw_session() { return session_.get(); }

private:
    std::unique_ptr<inference::CviSession> session_;
};

#endif  // USE_CVI_TPU

void register_module(lua_State* L);

} // namespace lua_nn

#endif
