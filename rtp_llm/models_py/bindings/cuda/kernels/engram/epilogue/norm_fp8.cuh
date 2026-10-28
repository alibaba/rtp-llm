#pragma once
#include <rtp_llm/models_py/bindings/cuda/kernels/engram/common/math.cuh>
#include <rtp_llm/models_py/bindings/cuda/kernels/engram/layout/pre.cuh>
namespace rtp_llm::engram::epilogue::mega_mhc {
using namespace layout::mega_mhc;
// Every warp-wide pack spans 256 hidden elements.
constexpr uint32_t kHiddenPerNormPack     = 256;
constexpr uint32_t kHiddenPerNormLane     = kHiddenPerNormPack / 32;
constexpr uint32_t kNumBF16PairsPerLane   = sizeof(uint4) / sizeof(nv_bfloat162);
constexpr uint32_t kNumLanesPerSF         = kHiddenPerSF / kHiddenPerNormLane;
constexpr uint32_t kNumSFWordsPerNormPack = kHiddenPerNormPack / kHiddenPerSFWord;
constexpr uint32_t kNumLanesPerSFWord     = kHiddenPerSFWord / kHiddenPerNormLane;

// Shared normalization and FP8 output helpers.
CUTLASS_DEVICE nv_bfloat162 normalize_bf16_pair(const nv_bfloat162& input,
                                                const nv_bfloat162& weight,
                                                const float&        scale) {
    return __float22bfloat162_rn(
        __fmul2_rn(__fmul2_rn(__bfloat1622float2(input), {scale, scale}), __bfloat1622float2(weight)));
}

template<uint32_t SF_BLOCK_M>
CUTLASS_DEVICE uint32_t get_shared_sf_row_idx(const uint32_t& token_idx) {
    // Mega MoE SF pages interleave four 32-row groups within each 128-row page.
    const uint32_t idx = token_idx % SF_BLOCK_M;
    return token_idx / SF_BLOCK_M * ((SF_BLOCK_M + 127u) & ~127u) + (idx & ~127u) + (idx & 31u) * 4 + ((idx >> 5) & 3u);
}

CUTLASS_DEVICE uint32_t store_fp8_and_pack_sf(const NormArgs& norm_args,
                                              const uint64_t& output_offset,
                                              const uint4&    values,
                                              const uint32_t& sf_exp) {
    const auto bf16x2 = reinterpret_cast<const nv_bfloat162*>(&values);
    // Splat the BF16 sf_inv into both packed halves
    const auto  sf_inv     = __bfloat162bfloat162(math::get_ue8m0_sf_inv<nv_bfloat16>(sf_exp));
    const uint2 fp8_values = {math::scale_bf16x2_into_fp8x4(bf16x2[0], bf16x2[1], sf_inv, sf_inv),
                              math::scale_bf16x2_into_fp8x4(bf16x2[2], bf16x2[3], sf_inv, sf_inv)};
    *reinterpret_cast<uint2*>(norm_args.y_fp8 + output_offset) = fp8_values;

    // Pack four 32-element SF groups from each 16-lane group into one int32.
    const uint32_t paired_sf = __byte_perm(sf_exp, __shfl_xor_sync(0xffffffff, sf_exp, kNumLanesPerSF), 0x1140);
    return __byte_perm(paired_sf, __shfl_xor_sync(0xffffffff, paired_sf, 2 * kNumLanesPerSF), 0x5410);
}

}  // namespace rtp_llm::engram::epilogue::mega_mhc
