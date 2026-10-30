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

#include <deep_gemm/epilogue/operators.cuh>

namespace deep_gemm::epilogue::operators {

// Keep the V4.1 standalone quantizer's BF16 rounding and scale policy while
// using DeepGEMM's operator ABI; the GEMM mainloop and store schedules are intact.
struct V41QuantizeToFP8: CastToFP8WithClamp {
    using CastToFP8WithClamp::CastToFP8WithClamp;

    template<bool kSwapAB, uint32_t kNumValues>
    CUTLASS_DEVICE void prepare_fp8(const uint32_t (&values)[kNumValues],
                                    nv_bfloat162 (&rounded)[kNumValues / 2],
                                    uint2& sf_exp,
                                    const uint32_t&,
                                    const uint32_t&) {
#pragma unroll
        for (uint32_t i = 0; i < kNumValues / 2; ++i)
            rounded[i] = math::cast_into_bf16x2(values[i * 2], values[i * 2 + 1]);
        nv_bfloat162 amax;
        if constexpr (kSwapAB) {
            amax = math::warp_reduce_max<4, true>(__hmax2(__hmax2(__habs2(rounded[0]), __habs2(rounded[1])),
                                                          __hmax2(__habs2(rounded[2]), __habs2(rounded[3]))));
        } else {
            amax = __bfloat162bfloat162(math::get_packed_bf16_amax(rounded));
        }
        const math::V41QuantPolicy policy(args.alpha);
        sf_exp.x = policy.scale_exp<cutlass::float_e4m3_t>(amax.x);
        sf_exp.y = kSwapAB ? policy.scale_exp<cutlass::float_e4m3_t>(amax.y) : sf_exp.x;
    }

    template<uint32_t kNumPairs>
    CUTLASS_DEVICE void
    cast_fp8(const nv_bfloat162 (&rounded)[kNumPairs], uint32_t (&packed)[kNumPairs / 2], const uint2& sf_exp) const {
        const auto inv =
            nv_bfloat162(math::get_ue8m0_sf_inv<nv_bfloat16>(sf_exp.x), math::get_ue8m0_sf_inv<nv_bfloat16>(sf_exp.y));
#pragma unroll
        for (uint32_t i = 0; i < kNumPairs / 2; ++i)
            packed[i] = math::v41_quant4(rounded[i * 2], rounded[i * 2 + 1], inv, inv);
    }
};

}  // namespace deep_gemm::epilogue::operators
