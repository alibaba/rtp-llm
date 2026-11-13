#pragma once

// DeepGEMM 2.8.2 port. Quantization moved out of the store headers' free-function
// hooks (`math::get_ue8m0_sf_exp` / `math::scale_bf16x2_into_fp8x4`, which this
// translation unit used to macro-override before including the vendor stores) into
// the `CastToFP8<QuantPolicy>` epilogue operator. Stock policies are compile-time
// only and cannot see the runtime legacy/FTZ switch carried by
// `EpilogueOperatorArgs.alpha`, so the 2.8.1 arithmetic is kept verbatim below and
// re-exposed through an operator deriving the 2.8.2 interface, clamping disabled.

#include <deep_gemm/epilogue/operators.cuh>

namespace deep_gemm::math {

__device__ __forceinline__ float v41_mul_ftz(float a, float b) {
    float out;
    asm("mul.ftz.f32 %0, %1, %2;" : "=f"(out) : "f"(a), "f"(b));
    return out;
}

struct V41QuantPolicy {
    float legacy;

    __device__ explicit V41QuantPolicy(float value): legacy(value) {}

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

namespace deep_gemm::v41 {

// Re-exposes both 2.8.1 modes on the 2.8.2 epilogue interface: `args.alpha != 0`
// selects the legacy 1e-10-floored scale, `alpha == 0` the FTZ scale. Clamping
// stays disabled, matching the retired `operators::QuantizeToFP8` behaviour.
struct WoAQuantOperator final : epilogue::operators::CastToFP8<epilogue::operators::Group32FP8QuantPolicy, false> {
    using Base = epilogue::operators::CastToFP8<epilogue::operators::Group32FP8QuantPolicy, false>;
    using Base::Base;

    CUTLASS_DEVICE uint32_t scale_exp(const nv_bfloat16& value) const {
        return math::V41QuantPolicy(args.alpha).scale_exp(value);
    }

    // Stock `prepare_fp8` specialized for `kAllowClamp == false`; only the scale
    // exponent call differs (runtime legacy/FTZ switch instead of the policy).
    template <bool kSwapAB, uint32_t kNumValues>
    CUTLASS_DEVICE void prepare_fp8(const uint32_t (&values)[kNumValues],
                                    nv_bfloat162 (&values_bf16x2)[kNumValues / 2], uint2& sf_exp,
                                    const uint32_t& row_idx, const uint32_t& col_idx) {
        (void) row_idx; (void) col_idx;  // Only consumed by the clamp counter.
        #pragma unroll
        for (uint32_t pair_idx = 0; pair_idx < kNumValues / 2; ++ pair_idx)
            values_bf16x2[pair_idx] = math::cast_into_bf16x2(values[pair_idx * 2], values[pair_idx * 2 + 1]);
        nv_bfloat162 amax;
        if constexpr (kSwapAB) {
            amax = math::warp_reduce_max<4, true>(
                __hmax2(__hmax2(__habs2(values_bf16x2[0]), __habs2(values_bf16x2[1])),
                        __hmax2(__habs2(values_bf16x2[2]), __habs2(values_bf16x2[3]))));
        } else {
            amax = __bfloat162bfloat162(math::get_packed_bf16_amax(values_bf16x2));
        }
        sf_exp.x = scale_exp(amax.x);
        sf_exp.y = kSwapAB ? scale_exp(amax.y) : sf_exp.x;
    }

    // Stock `cast_fp8` specialized for `kAllowClamp == false`, packed by the 2.8.1
    // FTZ helper instead of the policy.
    template <uint32_t kNumPairs>
    CUTLASS_DEVICE void cast_fp8(const nv_bfloat162 (&values_bf16x2)[kNumPairs],
                                 uint32_t (&packed)[kNumPairs / 2], const uint2& sf_exp) const {
        const auto sf_inv = nv_bfloat162(math::get_ue8m0_sf_inv<nv_bfloat16>(sf_exp.x),
                                         math::get_ue8m0_sf_inv<nv_bfloat16>(sf_exp.y));
        #pragma unroll
        for (uint32_t packed_idx = 0; packed_idx < kNumPairs / 2; ++ packed_idx)
            packed[packed_idx] = math::v41_quant4(values_bf16x2[packed_idx * 2],
                                                  values_bf16x2[packed_idx * 2 + 1], sf_inv, sf_inv);
    }
};

}  // namespace deep_gemm::v41
