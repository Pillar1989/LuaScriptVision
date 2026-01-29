/*
 * Common utilities for T-Head RISC-V max/argmax operations
 * Shared code between max_with_argmax and sigmoid_max_with_argmax
 */

#pragma once

#ifdef ENABLE_RV_THEAD

#include <riscv_vector.h>
#include <cfloat>
#include <vector>
#include <cstdint>
#include <cmath>
#include "mathfun.h"  // For ::rv_thead::exp_ps

namespace tensor {
namespace rv_thead {

/**
 * Compute max values and argmax indices for 2D tensor along axis=0
 * Uses row-major traversal + RVV vectorization
 *
 * @param input_data  Pointer to input data [num_classes, n_anchors]
 * @param num_classes Number of classes (outer dimension, 80)
 * @param n_anchors   Number of anchors (inner dimension, 8400)
 * @param max_values  Output buffer for max values [n_anchors]
 * @param max_indices Output buffer for argmax indices [n_anchors]
 */
static inline void compute_max_and_argmax(
    const float* input_data,
    int64_t num_classes,
    int64_t n_anchors,
    float* max_values,
    int64_t* max_indices)
{
    // Initialize max_values to -FLT_MAX
    for (int64_t i = 0; i < n_anchors; ++i) {
        max_values[i] = -FLT_MAX;
        max_indices[i] = 0;
    }

    // 第一次遍历：只查找max值（向量化，无分支）
    for (int64_t cls = 0; cls < num_classes; ++cls) {
        const float* row_ptr = input_data + cls * n_anchors;

        // 向量化处理连续的anchors（m4配置，每次16个元素）
        size_t vl = vsetvl_e32m4(16);

        int64_t anchor = 0;
        for (; anchor + 16 <= n_anchors; anchor += 16) {
            // 顺序加载当前类别的anchor数据（cache友好）
            vfloat32m4_t v_data = vle32_v_f32m4(row_ptr + anchor, vl);
            vfloat32m4_t v_old_max = vle32_v_f32m4(&max_values[anchor], vl);

            // 比较并更新max值（无分支）
            vfloat32m4_t v_new_max = vfmax_vv_f32m4(v_data, v_old_max, vl);
            vse32_v_f32m4(&max_values[anchor], v_new_max, vl);
        }

        // 处理剩余元素
        for (; anchor < n_anchors; ++anchor) {
            float val = row_ptr[anchor];
            if (val > max_values[anchor]) {
                max_values[anchor] = val;
            }
        }
    }

    // 第二次遍历：查找argmax索引（标量，展开循环优化分支）
    for (int64_t cls = 0; cls < num_classes; ++cls) {
        const float* row_ptr = input_data + cls * n_anchors;

        // 展开循环减少分支开销
        int64_t anchor = 0;
        for (; anchor + 4 <= n_anchors; anchor += 4) {
            if (row_ptr[anchor] == max_values[anchor]) max_indices[anchor] = cls;
            if (row_ptr[anchor + 1] == max_values[anchor + 1]) max_indices[anchor + 1] = cls;
            if (row_ptr[anchor + 2] == max_values[anchor + 2]) max_indices[anchor + 2] = cls;
            if (row_ptr[anchor + 3] == max_values[anchor + 3]) max_indices[anchor + 3] = cls;
        }

        // 处理剩余元素
        for (; anchor < n_anchors; ++anchor) {
            if (row_ptr[anchor] == max_values[anchor]) {
                max_indices[anchor] = cls;
            }
        }
    }
}

/**
 * Apply vectorized sigmoid in-place to max values
 * sigmoid(x) = 1 / (1 + exp(-x))
 * Uses rv_thead::exp_ps() for fast RVV exp computation
 *
 * @param max_values  Buffer to apply sigmoid [n_anchors]
 * @param n_anchors   Number of elements
 */
static inline void apply_vectorized_sigmoid(float* max_values, int64_t n_anchors) {
    // 使用 m2 配置以匹配 exp_ps() 接口
    size_t vl_sigmoid = vsetvl_e32m2(8);  // m2: 每次8个元素

    int64_t anchor = 0;
    for (; anchor + 8 <= n_anchors; anchor += 8) {
        // 加载 max 值
        vfloat32m2_t v_max = vle32_v_f32m2(&max_values[anchor], vl_sigmoid);

        // 计算 -x
        vfloat32m2_t v_neg = vfneg_v_f32m2(v_max, vl_sigmoid);

        // 向量化 exp(-x) - 使用快速RVV exp实现
        vfloat32m2_t v_exp = ::rv_thead::exp_ps(v_neg, vl_sigmoid);

        // 计算 1 + exp(-x)
        vfloat32m2_t v_denom = vfadd_vf_f32m2(v_exp, 1.0f, vl_sigmoid);

        // 计算 1 / (1 + exp(-x))
        vfloat32m2_t v_one = vfmv_v_f_f32m2(1.0f, vl_sigmoid);
        vfloat32m2_t v_sigmoid = vfdiv_vv_f32m2(v_one, v_denom, vl_sigmoid);

        // 存储 sigmoid 结果
        vse32_v_f32m2(&max_values[anchor], v_sigmoid, vl_sigmoid);
    }

    // 处理 sigmoid 剩余元素（标量）
    for (; anchor < n_anchors; ++anchor) {
        max_values[anchor] = 1.0f / (1.0f + std::exp(-max_values[anchor]));
    }
}

}  // namespace rv_thead
}  // namespace tensor

#endif  // ENABLE_RV_THEAD
