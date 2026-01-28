/**
 * lua_nn.cpp - Inference Session bindings and Lua module registration
 *
 * Supports ONNX Runtime (CPU) and CVI Runtime (TPU) backends.
 */

#include "lua_nn.h"

namespace lua_nn {

#ifdef USE_ONNX_RUNTIME

// ========== Session (ONNX) ==========

Session::Session(const std::string& model_path, int num_threads)
    : session_(std::make_unique<inference::OnnxSession>(model_path, num_threads)) {
}

LuaIntf::LuaRef Session::run(lua_State* L, const Tensor& input_tensor) {
    auto shape = input_tensor.shape();
    const float* input_data = input_tensor.raw_data();

    auto [output_data, output_shape] = session_->run(input_data, shape);

    Tensor output_tensor(std::move(output_data), output_shape);

    LuaIntf::LuaRef outputs = LuaIntf::LuaRef::createTable(L);
    const auto& output_names = session_->get_output_names();

    // Add generic "output0" name for consistency with TPU path
    outputs["output0"] = output_tensor;

    // Also add model's original output name if different
    if (!output_names.empty() && output_names[0] != "output0") {
        outputs[output_names[0]] = output_tensor;
    }

    return outputs;
}

std::vector<std::string> Session::input_names() const {
    return session_->get_input_names();
}

std::vector<std::string> Session::output_names() const {
    return session_->get_output_names();
}

std::vector<int64_t> Session::input_shape(size_t index) const {
    return session_->get_input_shape(index);
}

std::vector<int64_t> Session::output_shape(size_t index) const {
    return session_->get_output_shape(index);
}

#endif  // USE_ONNX_RUNTIME

#ifdef USE_CVI_TPU

// ========== TpuSession (CVI Runtime) ==========

TpuSession::TpuSession(const std::string& model_path)
    : session_(std::make_unique<inference::CviSession>(model_path)) {
}

LuaIntf::LuaRef TpuSession::run(lua_State* L, const Tensor& input_tensor) {
    auto shape = input_tensor.shape();
    const float* input_data = input_tensor.raw_data();

    // Use run() for single primary output (compatible with ONNX Session API)
    auto [output_data, output_shape] = session_->run(input_data, shape);

    Tensor output_tensor(std::move(output_data), output_shape);

    LuaIntf::LuaRef outputs = LuaIntf::LuaRef::createTable(L);
    const auto& names = session_->get_output_names();

    int32_t primary_idx = session_->primary_output_index();
    std::string primary_name = (primary_idx >= 0 &&
                                static_cast<size_t>(primary_idx) < names.size())
                               ? names[primary_idx]
                               : "output0";
    outputs[primary_name] = output_tensor;

    return outputs;
}

LuaIntf::LuaRef TpuSession::run_all(lua_State* L, const Tensor& input_tensor) {
    auto shape = input_tensor.shape();
    const float* input_data = input_tensor.raw_data();

    std::vector<std::vector<float>> output_data;
    std::vector<std::vector<int64_t>> output_shapes;
    session_->run_all(input_data, shape, &output_data, &output_shapes);

    LuaIntf::LuaRef outputs = LuaIntf::LuaRef::createTable(L);
    const auto& names = session_->get_output_names();

    for (size_t i = 0; i < output_data.size(); ++i) {
        Tensor tensor(std::move(output_data[i]), output_shapes[i]);
        std::string name = (i < names.size()) ? names[i]
                           : "output" + std::to_string(i);
        outputs[name] = tensor;
    }

    return outputs;
}

std::vector<std::string> TpuSession::input_names() const {
    return session_->get_input_names();
}

std::vector<std::string> TpuSession::output_names() const {
    return session_->get_output_names();
}

std::vector<int64_t> TpuSession::input_shape(size_t index) const {
    return session_->get_input_shape(index);
}

std::vector<int64_t> TpuSession::output_shape(size_t index) const {
    return session_->get_output_shape(index);
}

int32_t TpuSession::output_count() const {
    return session_->output_count();
}

std::string TpuSession::backend_name() const {
    return session_->backend_name();
}

#endif  // USE_CVI_TPU

// ========== Lua Module Registration ==========

void register_module(lua_State* L) {
    using namespace LuaIntf;

    LuaBinding(L)
        .beginModule("lua_nn")
            // Tensor class binding
            .beginClass<tensor::Tensor>("Tensor")
                .addStaticFunction("new", &tensor::Tensor::from_lua)
                .addStaticFunction("arange", &tensor::Tensor::arange)

                .addProperty("ndim", &Tensor::ndim)
                .addFunction("shape", &tensor::Tensor::shape)
                .addFunction("strides", &tensor::Tensor::strides)
                .addFunction("size", static_cast<int64_t(Tensor::*)() const>(&Tensor::size))
                .addFunction("is_contiguous", &Tensor::is_contiguous)
                .addFunction("contiguous", &Tensor::contiguous)
                .addFunction("view", &Tensor::view)

                // Shape operations
                .addFunction("slice", &tensor::Tensor::slice)
                .addFunction("select_dim", &tensor::Tensor::select_dim)
                .addFunction("get_column", &tensor::Tensor::get_column)
                .addFunction("slice_columns", &tensor::Tensor::slice_columns)
                .addFunction("reshape", &tensor::Tensor::reshape)
                .addFunction("transpose",
                    static_cast<tensor::Tensor(tensor::Tensor::*)() const>(&tensor::Tensor::transpose))
                .addFunction("transpose_dims",
                    static_cast<tensor::Tensor(tensor::Tensor::*)(const std::vector<int>&) const>(&tensor::Tensor::transpose))
                .addFunction("squeeze", &tensor::Tensor::squeeze)
                .addFunction("unsqueeze", &tensor::Tensor::unsqueeze)

                // Math operations
                .addFunction("add", static_cast<tensor::Tensor(tensor::Tensor::*)(float) const>(&tensor::Tensor::add))
                .addFunction("add_tensor", static_cast<tensor::Tensor(tensor::Tensor::*)(const tensor::Tensor&) const>(&tensor::Tensor::add))
                .addFunction("sub", static_cast<tensor::Tensor(tensor::Tensor::*)(float) const>(&tensor::Tensor::sub))
                .addFunction("sub_tensor", static_cast<tensor::Tensor(tensor::Tensor::*)(const tensor::Tensor&) const>(&tensor::Tensor::sub))
                .addFunction("mul", static_cast<tensor::Tensor(tensor::Tensor::*)(float) const>(&tensor::Tensor::mul))
                .addFunction("mul_tensor", static_cast<tensor::Tensor(tensor::Tensor::*)(const tensor::Tensor&) const>(&tensor::Tensor::mul))
                .addFunction("div", static_cast<tensor::Tensor(tensor::Tensor::*)(float) const>(&tensor::Tensor::div))
                .addFunction("div_tensor", static_cast<tensor::Tensor(tensor::Tensor::*)(const tensor::Tensor&) const>(&tensor::Tensor::div))

                // In-place operations
                .addFunction("add_", &tensor::Tensor::add_)
                .addFunction("sub_", &tensor::Tensor::sub_)
                .addFunction("mul_", &tensor::Tensor::mul_)
                .addFunction("div_", &tensor::Tensor::div_)

                // Clamp operation
                .addFunction("clamp", &tensor::Tensor::clamp)

                .addFunction("sum", &tensor::Tensor::sum)
                .addFunction("mean", &tensor::Tensor::mean)
                .addFunction("max", &tensor::Tensor::max)
                .addFunction("min", &tensor::Tensor::min)
                .addFunction("argmax", &tensor::Tensor::argmax_lua)
                .addFunction("argmin", &tensor::Tensor::argmin_lua)
                .addFunction("max_with_argmax", &tensor::Tensor::max_with_argmax)

                .addFunction("sigmoid", &tensor::Tensor::sigmoid)
                .addFunction("softmax", &tensor::Tensor::softmax)
                .addFunction("exp", &tensor::Tensor::exp_)
                .addFunction("log", &tensor::Tensor::log_)

                .addFunction("gt", &tensor::Tensor::gt)
                .addFunction("lt", &tensor::Tensor::lt)
                .addFunction("ge", &tensor::Tensor::ge)
                .addFunction("le", &tensor::Tensor::le)
                .addFunction("eq", &tensor::Tensor::eq)

                // Advanced operations
                .addFunction("topk_new", &tensor::Tensor::topk_lua)
                .addFunction("to_table", &tensor::Tensor::to_table)
                .addFunction("to_string", &tensor::Tensor::to_string)
                .addFunction("get", &tensor::Tensor::get_lua)
                .addFunction("set", &tensor::Tensor::set_lua)
                .addFunction("at", &tensor::Tensor::at2d)

                // Vectorized filtering
                .addFunction("nonzero", &tensor::Tensor::nonzero)
                .addFunction("where_indices", &tensor::Tensor::where_indices)
                .addFunction("index_select", &tensor::Tensor::index_select)
                .addFunction("extract_columns", &tensor::Tensor::extract_columns_lua)

                // Gather/Concat/Split
                .addFunction("gather", &tensor::Tensor::gather)
                .addStaticFunction("concat", &tensor::Tensor::concat)
                .addFunction("split", &tensor::Tensor::split)

                // Legacy methods (backward compatible)
                .addFunction("filter_yolo", &tensor::Tensor::filter_yolo)
                .addFunction("filter_yolo_pose", &tensor::Tensor::filter_yolo_pose)
                .addFunction("filter_yolo_seg", &tensor::Tensor::filter_yolo_seg)
                .addFunction("process_mask", &tensor::Tensor::process_mask)
                .addFunction("argmax_old", &tensor::Tensor::argmax)
                .addFunction("topk", &tensor::Tensor::topk)

                // Metamethods
                .addMetaFunction("__len", [](const tensor::Tensor* t) { return t->size(); })
                .addMetaFunction("__tostring", [](const tensor::Tensor* t) {
                    return t->to_string(10);
                })
                .addMetaFunction("__add", [](const tensor::Tensor& t, float scalar) { return t.add(scalar); })
                .addMetaFunction("__sub", [](const tensor::Tensor& t, float scalar) { return t.sub(scalar); })
                .addMetaFunction("__mul", [](const tensor::Tensor& t, float scalar) { return t.mul(scalar); })
                .addMetaFunction("__div", [](const tensor::Tensor& t, float scalar) { return t.div(scalar); })
            .endClass()

            // TensorView binding
            .beginClass<TensorView<float>>("FloatView")
                .addFunction("get", &TensorView<float>::get)
                .addFunction("set", &TensorView<float>::set)
                .addMetaFunction("__len", [](const TensorView<float>* t) { return t->length(); })
            .endClass()

#ifdef USE_ONNX_RUNTIME
            // ONNX Session binding
            .beginClass<Session>("Session")
                .addConstructor(
                    LUA_SP(std::shared_ptr<Session>),
                    LUA_ARGS(const std::string&)
                )
                .addFunction("run", &Session::run)
                .addProperty("input_names", &Session::input_names)
                .addProperty("output_names", &Session::output_names)
            .endClass()
#endif  // USE_ONNX_RUNTIME

#ifdef USE_CVI_TPU
            // TPU Session binding
            .beginClass<TpuSession>("TpuSession")
                .addConstructor(
                    LUA_SP(std::shared_ptr<TpuSession>),
                    LUA_ARGS(const std::string&)
                )
                .addFunction("run", &TpuSession::run)
                .addFunction("run_all", &TpuSession::run_all)
                .addProperty("input_names", &TpuSession::input_names)
                .addProperty("output_names", &TpuSession::output_names)
                .addProperty("output_count", &TpuSession::output_count)
                .addProperty("backend", &TpuSession::backend_name)
            .endClass()
#endif  // USE_CVI_TPU
        .endModule();
}

} // namespace lua_nn
