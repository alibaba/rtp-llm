#pragma once

#include <deep_gemm/common/math.cuh>

namespace deep_gemm::math {

__device__ __forceinline__ float v41_mul_ftz(float a, float b) {
    float out;
    asm("mul.ftz.f32 %0, %1, %2;" : "=f"(out) : "f"(a), "f"(b));
    return out;
}

struct V41QuantPolicy {
    float legacy;

    __device__ explicit V41QuantPolicy(float value): legacy(value) {}

    template<typename DType>
    __device__ __forceinline__ unsigned scale_exp(nv_bfloat16 value) const {
        const float amax = fmaxf(__bfloat162float(value), 0x1p-126f);
        if (legacy != 0.0f) {
            const float raw = fmaxf(amax * (1.0f / 448.0f), 1.0e-10f);
            return static_cast<int>(ceilf(log2f(raw))) + 127;
        }
        const unsigned bits = __float_as_uint(v41_mul_ftz(amax, 1.0f / 448.0f));
        return ((bits >> 23) & 255) + ((bits & 0x7fffff) != 0);
    }
};

__device__ __forceinline__ unsigned v41_quant4(nv_bfloat162 a, nv_bfloat162 b, nv_bfloat162 inv_a, nv_bfloat162 inv_b) {
    const float2 x = __bfloat1622float2(a), y = __bfloat1622float2(b);
    const float2 sx = __bfloat1622float2(inv_a), sy = __bfloat1622float2(inv_b);
    return __nv_fp8x4_e4m3(
               make_float4(
                   v41_mul_ftz(x.x, sx.x), v41_mul_ftz(x.y, sx.y), v41_mul_ftz(y.x, sy.x), v41_mul_ftz(y.y, sy.y)))
        .__x;
}

}  // namespace deep_gemm::math

// Override only these two epilogue operations in this translation unit.
// The vendor mainloop, BF16 rounding, reductions and TMA stores stay intact.
#define get_ue8m0_sf_exp V41QuantPolicy(epilogue_op.alpha).template scale_exp
#define scale_bf16x2_into_fp8x4 v41_quant4
#include <deep_gemm/epilogue/sm100_store_cd.cuh>
#include <deep_gemm/epilogue/sm100_store_cd_swap_ab.cuh>
#undef scale_bf16x2_into_fp8x4
#undef get_ue8m0_sf_exp
