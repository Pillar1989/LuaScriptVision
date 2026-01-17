/**
 * lua_nn.cpp - ONNX Runtime Session 绑定和 Lua 模块注册
 *
 * 现在使用 inference::OnnxSession 作为底层推理引擎
 */

#include "lua_nn.h"

namespace lua_nn {

#ifdef USE_ONNX_RUNTIME

// ========== Session 实现 ==========

Session::Session(const std::string& model_path, int num_threads)
    : session_(std::make_unique<inference::OnnxSession>(model_path, num_threads)) {
}

LuaIntf::LuaRef Session::run(lua_State* L, const Tensor& input_tensor) {
    // 获取输入数据和形状
    auto shape = input_tensor.shape();
    const float* input_data = input_tensor.raw_data();

    // 调用底层推理引擎
    auto [output_data, output_shape] = session_->run(input_data, shape);

    // 创建输出 Tensor
    Tensor output_tensor(std::move(output_data), output_shape);

    // 返回 Lua table（为了向后兼容，仍然用 table 包装）
    LuaIntf::LuaRef outputs = LuaIntf::LuaRef::createTable(L);
    const auto& output_names = session_->get_output_names();

    if (!output_names.empty()) {
        outputs[output_names[0]] = output_tensor;
    } else {
        outputs["output"] = output_tensor;
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

// ========== Lua 模块注册 ==========

void register_module(lua_State* L) {
    using namespace LuaIntf;

    LuaBinding(L)
        .beginModule("lua_nn")
            // Tensor类绑定 (显式使用 tensor::Tensor)
            .beginClass<tensor::Tensor>("Tensor")
                .addStaticFunction("new", &tensor::Tensor::from_lua)

                // 属性
                .addProperty("ndim", &Tensor::ndim)
                .addFunction("shape", &tensor::Tensor::shape)
                .addFunction("strides", &tensor::Tensor::strides)
                .addFunction("size", static_cast<int64_t(Tensor::*)() const>(&Tensor::size))
                .addFunction("is_contiguous", &Tensor::is_contiguous)
                .addFunction("contiguous", &Tensor::contiguous)
                .addFunction("view", &Tensor::view)

                // Level 1: 基础形状操作
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

                // Level 2: 数学运算
                .addFunction("add", static_cast<tensor::Tensor(tensor::Tensor::*)(float) const>(&tensor::Tensor::add))
                .addFunction("add_tensor", static_cast<tensor::Tensor(tensor::Tensor::*)(const tensor::Tensor&) const>(&tensor::Tensor::add))
                .addFunction("sub", static_cast<tensor::Tensor(tensor::Tensor::*)(float) const>(&tensor::Tensor::sub))
                .addFunction("sub_tensor", static_cast<tensor::Tensor(tensor::Tensor::*)(const tensor::Tensor&) const>(&tensor::Tensor::sub))
                .addFunction("mul", static_cast<tensor::Tensor(tensor::Tensor::*)(float) const>(&tensor::Tensor::mul))
                .addFunction("mul_tensor", static_cast<tensor::Tensor(tensor::Tensor::*)(const tensor::Tensor&) const>(&tensor::Tensor::mul))
                .addFunction("div", static_cast<tensor::Tensor(tensor::Tensor::*)(float) const>(&tensor::Tensor::div))
                .addFunction("div_tensor", static_cast<tensor::Tensor(tensor::Tensor::*)(const tensor::Tensor&) const>(&tensor::Tensor::div))

                // In-place 操作（避免内存分配）
                .addFunction("add_", &tensor::Tensor::add_)
                .addFunction("sub_", &tensor::Tensor::sub_)
                .addFunction("mul_", &tensor::Tensor::mul_)
                .addFunction("div_", &tensor::Tensor::div_)

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

                // Level 3: 高级操作
                .addFunction("topk_new", &tensor::Tensor::topk_lua)
                .addFunction("to_table", &tensor::Tensor::to_table)
                .addFunction("to_string", &tensor::Tensor::to_string)
                .addFunction("get", &tensor::Tensor::get_lua)
                .addFunction("set", &tensor::Tensor::set_lua)
                .addFunction("at", &tensor::Tensor::at2d)

                // 向量化过滤操作
                .addFunction("nonzero", &tensor::Tensor::nonzero)
                .addFunction("where_indices", &tensor::Tensor::where_indices)
                .addFunction("index_select", &tensor::Tensor::index_select)
                .addFunction("extract_columns", &tensor::Tensor::extract_columns_lua)

                // Gather/Concat/Split
                .addFunction("gather", &tensor::Tensor::gather)
                .addStaticFunction("concat", &tensor::Tensor::concat)
                .addFunction("split", &tensor::Tensor::split)

                // Legacy方法（向后兼容）
                .addFunction("filter_yolo", &tensor::Tensor::filter_yolo)
                .addFunction("filter_yolo_pose", &tensor::Tensor::filter_yolo_pose)
                .addFunction("filter_yolo_seg", &tensor::Tensor::filter_yolo_seg)
                .addFunction("process_mask", &tensor::Tensor::process_mask)
                .addFunction("argmax_old", &tensor::Tensor::argmax)
                .addFunction("topk", &tensor::Tensor::topk)

                // Metamethods (使用 const& 避免不必要的 shared_ptr 原子操作)
                .addMetaFunction("__len", [](const tensor::Tensor* t) { return t->size(); })
                .addMetaFunction("__tostring", [](const tensor::Tensor* t) {
                    return t->to_string(10);
                })
                .addMetaFunction("__add", [](const tensor::Tensor& t, float scalar) { return t.add(scalar); })
                .addMetaFunction("__sub", [](const tensor::Tensor& t, float scalar) { return t.sub(scalar); })
                .addMetaFunction("__mul", [](const tensor::Tensor& t, float scalar) { return t.mul(scalar); })
                .addMetaFunction("__div", [](const tensor::Tensor& t, float scalar) { return t.div(scalar); })
            .endClass()

            // TensorView绑定
            .beginClass<TensorView<float>>("FloatView")
                .addFunction("get", &TensorView<float>::get)
                .addFunction("set", &TensorView<float>::set)
                .addMetaFunction("__len", [](const TensorView<float>* t) { return t->length(); })
            .endClass()

#ifdef USE_ONNX_RUNTIME
            // Session绑定 (仅在启用ONNX Runtime时可用)
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
        .endModule();
}

} // namespace lua_nn
