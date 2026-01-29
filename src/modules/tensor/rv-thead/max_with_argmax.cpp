/*
 * T-Head RISC-V optimized max + argmax fusion (row-major scan)
 * Uses RVV (RISC-V Vector Extension) for C906/C908 processors
 */

#include "../tensor.h"
#include "LuaIntf.h"

#ifdef ENABLE_RV_THEAD

#include <vector>
#include "max_argmax_common.h"

namespace tensor {

LuaIntf::LuaRef Tensor::max_with_argmax_thead(lua_State* L, int axis) const {
    check_cpu();

    if (ndim() != 2 || axis != 0) {
        throw std::runtime_error(
            "max_with_argmax_thead: requires 2D tensor with axis=0");
    }

    int64_t num_classes = shape_[0];  // 80
    int64_t n_anchors = shape_[1];    // 8400
    const float* input_data = data();

    std::vector<float> max_values(n_anchors);
    std::vector<int64_t> max_indices(n_anchors);

    // Use shared RVV-optimized max/argmax computation
    rv_thead::compute_max_and_argmax(
        input_data, num_classes, n_anchors,
        max_values.data(), max_indices.data());

    // 构造 Lua 返回值（预分配table容量）
    LuaIntf::LuaRef result = LuaIntf::LuaRef::createTable(L);
    result["values"] = Tensor(std::move(max_values), {n_anchors});

    lua_createtable(L, static_cast<int>(n_anchors), 0);
    for (size_t i = 0; i < max_indices.size(); ++i) {
        lua_pushinteger(L, static_cast<lua_Integer>(max_indices[i]));
        lua_rawseti(L, -2, static_cast<lua_Integer>(i + 1));
    }
    LuaIntf::LuaRef indices_table = LuaIntf::LuaRef::popFromStack(L);
    result["indices"] = indices_table;

    return result;
}

}  // namespace tensor

#endif  // ENABLE_RV_THEAD
