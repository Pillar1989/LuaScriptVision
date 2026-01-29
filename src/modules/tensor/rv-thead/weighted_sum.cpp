/*
 * T-Head RISC-V optimized weighted_sum implementation
 * Uses RVV (RISC-V Vector Extension) for C906/C908 processors
 */

#include "../tensor.h"
#include "memory/cpu_memory.h"

#ifdef ENABLE_RV_THEAD

#include <riscv_vector.h>
#include <cfloat>
#include <cmath>
#include "mathfun.h"

namespace tensor {

/**
 * 融合 softmax + weighted sum (玄铁 RVV 优化版本)
 *
 * 用于 YOLO DFL decode：
 *   输入: [outer, axis_size, inner]  (例如 [4, 16, n_valid])
 *   权重: [axis_size]                (例如 [16])
 *   输出: [outer, inner]              (例如 [4, n_valid])
 *
 * 计算: output[o,i] = sum(weights[a] * softmax(input[o,a,i]))
 *
 * @param axis 沿此轴做 softmax 和加权求和（必须为 1）
 * @param weights 权重张量 [axis_size]
 * @return 结果张量 [outer, inner]
 */
Tensor Tensor::weighted_sum_thead(int axis, const Tensor& weights) const {
    check_cpu();

    // 确保连续内存（RVV 需要规则访问模式）
    if (!contiguous_) {
        return contiguous().weighted_sum_thead(axis, weights);
    }

    // 验证输入
    if (ndim() != 3 || axis != 1) {
        throw std::runtime_error(
            "weighted_sum_thead: requires 3D tensor with axis=1");
    }

    int64_t outer = shape_[0];
    int64_t axis_size = shape_[1];
    int64_t inner = shape_[2];

    if (weights.size() != axis_size) {
        throw std::runtime_error(
            "weighted_sum_thead: weights size must match axis dimension");
    }

    const float* input_data = data();
    const float* weight_data = weights.data();

    // 分配输出 [outer, inner]
    auto out_storage = CpuMemory::allocate(outer * inner * sizeof(float));
    float* output_data = static_cast<float*>(out_storage->data());

    // 处理每个 (outer, inner) 组合
    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t i = 0; i < inner; ++i) {
            // 当前元素的跨步访问起点
            const float* ptr = input_data + o * axis_size * inner + i;
            const int64_t stride = inner;  // 元素间距

            // ===== Step 1: 向量化找 max（数值稳定性） =====
            float max_val = -FLT_MAX;

            // 使用 m2 配置（可处理 8 个 float32）
            size_t vl = vsetvl_e32m2(8);
            vfloat32m2_t v_max = vfmv_v_f_f32m2(max_val, vl);

            int64_t a = 0;
            for (; a + 8 <= axis_size; a += 8) {
                // 跨步加载（stride load）
                vfloat32m2_t v_data = vlse32_v_f32m2(
                    ptr + a * stride,
                    stride * sizeof(float),
                    vl);
                v_max = vfmax_vv_f32m2(v_data, v_max, vl);
            }

            // Reduction: 向量 → 标量
            vfloat32m1_t v_init = vfmv_v_f_f32m1(max_val, 4);
            vfloat32m1_t v_result = vfredmax_vs_f32m2_f32m1(
                vundefined_f32m1(), v_max, v_init, vl);
            max_val = vfmv_f_s_f32m1_f32(v_result);

            // 处理剩余元素
            for (; a < axis_size; ++a) {
                max_val = std::max(max_val, ptr[a * stride]);
            }

            // ===== Step 2 & 3: 向量化 exp + weighted sum =====
            float sum_exp = 0.0f;
            float weighted_sum = 0.0f;

            vl = vsetvl_e32m2(8);
            vfloat32m2_t v_sum_exp = vfmv_v_f_f32m2(0.0f, vl);
            vfloat32m2_t v_weighted = vfmv_v_f_f32m2(0.0f, vl);
            vfloat32m2_t v_max_broadcast = vfmv_v_f_f32m2(max_val, vl);

            a = 0;
            for (; a + 8 <= axis_size; a += 8) {
                // 加载数据和权重
                vfloat32m2_t v_data = vlse32_v_f32m2(
                    ptr + a * stride,
                    stride * sizeof(float),
                    vl);
                vfloat32m2_t v_weights = vle32_v_f32m2(weight_data + a, vl);

                // exp(data - max)
                v_data = vfsub_vv_f32m2(v_data, v_max_broadcast, vl);
                v_data = rv_thead::exp_ps(v_data, vl);  // 向量化 exp

                // 累加 sum_exp 和 weighted_sum
                v_sum_exp = vfadd_vv_f32m2(v_sum_exp, v_data, vl);
                vfloat32m2_t v_prod = vfmul_vv_f32m2(v_weights, v_data, vl);
                v_weighted = vfadd_vv_f32m2(v_weighted, v_prod, vl);
            }

            // Reduction: 向量求和 → 标量
            vfloat32m1_t v_zero = vfmv_v_f_f32m1(0.0f, 4);
            vfloat32m1_t v_sum;

            v_sum = vfredosum_vs_f32m2_f32m1(
                vundefined_f32m1(), v_sum_exp, v_zero, vl);
            sum_exp = vfmv_f_s_f32m1_f32(v_sum);

            v_sum = vfredosum_vs_f32m2_f32m1(
                vundefined_f32m1(), v_weighted, v_zero, vl);
            weighted_sum = vfmv_f_s_f32m1_f32(v_sum);

            // 处理剩余元素
            for (; a < axis_size; ++a) {
                float exp_val = std::exp(ptr[a * stride] - max_val);
                sum_exp += exp_val;
                weighted_sum += weight_data[a] * exp_val;
            }

            // 归一化
            output_data[o * inner + i] = weighted_sum / sum_exp;
        }
    }

    return Tensor(out_storage, {outer, inner},
                  compute_strides({outer, inner}), 0, true);
}

}  // namespace tensor

#endif  // ENABLE_RV_THEAD
