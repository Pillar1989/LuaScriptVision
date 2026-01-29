/*
 * T-Head RISC-V vector math functions
 * Adapted from CSI-NN2 rvv_mathfun_fp32.h
 */

#pragma once

#ifdef ENABLE_RV_THEAD

#include <riscv_vector.h>
#include <cstdint>

namespace rv_thead {

/**
 * 向量化 exp 函数
 * 使用快速近似算法：exp(x) ≈ 2^(x * log2(e))
 *
 * 参考: CSI-NN2 source/thead_rvv/rvv_mathfun_fp32.h
 *
 * @param x 输入向量
 * @param vl 向量长度
 * @return exp(x) 的向量结果
 */
static inline vfloat32m2_t exp_ps(vfloat32m2_t x, size_t vl) {
    // 常数定义
    vfloat32m2_t v_log2e = vfmv_v_f_f32m2(1.44269504088896341f, vl);  // 1/ln(2)
    vfloat32m2_t v_0p5 = vfmv_v_f_f32m2(0.5f, vl);

    // Clamp 输入范围避免溢出
    // exp(-88) ≈ 2e-39, exp(88) ≈ 1.6e38
    vfloat32m2_t v_min = vfmv_v_f_f32m2(-88.0f, vl);
    vfloat32m2_t v_max = vfmv_v_f_f32m2(88.0f, vl);
    x = vfmax_vv_f32m2(x, v_min, vl);
    x = vfmin_vv_f32m2(x, v_max, vl);

    // 计算 z = x * log2(e)
    vfloat32m2_t z = vfmul_vv_f32m2(x, v_log2e, vl);

    // 分离整数和小数部分
    // n = floor(z + 0.5) (四舍五入)
    vfloat32m2_t z_round = vfadd_vv_f32m2(z, v_0p5, vl);
    vint32m2_t n = vfcvt_x_f_v_i32m2(z_round, vl);
    vfloat32m2_t fn = vfcvt_f_x_v_f32m2(n, vl);

    // 小数部分: r = z - n
    vfloat32m2_t r = vfsub_vv_f32m2(z, fn, vl);

    // 泰勒展开: 2^r ≈ 1 + r*ln(2) + (r*ln(2))^2/2 + ...
    // 使用简化版本: 2^r ≈ 1 + r * (c1 + r * (c2 + r * c3))
    vfloat32m2_t c1 = vfmv_v_f_f32m2(0.693147180559945f, vl);   // ln(2)
    vfloat32m2_t c2 = vfmv_v_f_f32m2(0.240226506959101f, vl);   // ln(2)^2/2
    vfloat32m2_t c3 = vfmv_v_f_f32m2(0.055504108664822f, vl);   // ln(2)^3/6

    vfloat32m2_t poly = vfmul_vv_f32m2(r, c3, vl);
    poly = vfadd_vv_f32m2(poly, c2, vl);
    poly = vfmul_vv_f32m2(poly, r, vl);
    poly = vfadd_vv_f32m2(poly, c1, vl);
    poly = vfmul_vv_f32m2(poly, r, vl);
    poly = vfadd_vf_f32m2(poly, 1.0f, vl);

    // 计算 2^n（通过位操作）
    // float(2^n) = *(float*)&((n + 127) << 23)
    vint32m2_t exp_bits = vadd_vx_i32m2(n, 127, vl);
    exp_bits = vsll_vx_i32m2(exp_bits, 23, vl);
    vfloat32m2_t pow2n = vreinterpret_v_i32m2_f32m2(exp_bits);

    // 结果: 2^z = 2^n * 2^r
    return vfmul_vv_f32m2(pow2n, poly, vl);
}

}  // namespace rv_thead

#endif  // ENABLE_RV_THEAD
