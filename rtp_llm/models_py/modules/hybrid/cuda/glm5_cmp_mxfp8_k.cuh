#pragma once

#include <rtp_kernel/glm5/deep_gemm/fp8_gemm.cuh>
#include <rtp_kernel/glm5/indexer_k_cache.cuh>

namespace rtp_llm {

__device__ __forceinline__ void store_mxfp8_key(const rtp_kernel::glm5::detail::Bf16x8&            input,
                                                const rtp_kernel::glm5::detail::IndexerKCacheArgs& args,
                                                int                                                row,
                                                int                                                sublane,
                                                bool                                               valid) {
    using namespace rtp_kernel::glm5::detail;
    float values[8];
    unpack_bf16x8(input, values);
    float absmax = 0.0f;
#pragma unroll
    for (float value : values) {
        absmax = fmaxf(absmax, fabsf(value));
    }
#pragma unroll
    for (int mask = 1; mask < 4; mask <<= 1) {
        absmax = fmaxf(absmax, __shfl_xor_sync(0xffffffffu, absmax, mask, 4));
    }
    const auto  exponent      = ue8m0_exponent(absmax);
    const float inverse_scale = ue8m0_inv_scale(exponent);
    if (!valid) {
        return;
    }
    const std::int64_t slot = args.slot_mapping[row];
    if (slot < 0 || slot >= static_cast<std::int64_t>(args.num_cache_blocks) * 64) {
        return;
    }
    const auto block      = slot / 64;
    const auto offset     = slot % 64;
    auto*      block_base = args.cache + block * 64 * 132;
    Fp8x8      quantized;
#pragma unroll
    for (int i = 0; i < 8; i += 2) {
        quantized.pairs[i / 2] = fp32x2_to_e4m3x2(values[i] * inverse_scale, values[i + 1] * inverse_scale);
    }
    *reinterpret_cast<Fp8x8*>(block_base + offset * 128 + sublane * 8) = quantized;
    if ((sublane & 3) == 0) {
        block_base[64 * 128 + offset * 4 + sublane / 4] = exponent;
    }
}

struct IndexerKMxfp8Epilogue {
    static constexpr bool kUsesTensorMapCd = false;

    template<std::uint32_t       BLOCK_M,
             std::uint32_t       BLOCK_N,
             std::uint32_t       STORE_BLOCK_M,
             std::uint32_t       STORE_BLOCK_N,
             std::uint32_t       kSwizzleCDMode,
             std::uint32_t       kNumTMAStoreStages,
             std::uint32_t       kNumUMMAStoreThreads,
             deep_gemm::GemmType kGemmType,
             bool                kWithAccumulation,
             typename cd_dtype_t,
             typename epilogue_type_t,
             typename pattern_cd_t>
    CUTLASS_DEVICE static void store_swap_ab(const deep_gemm::utils::PatternVisitor<pattern_cd_t>& smem_cd,
                                             std::uint32_t&                                        stage,
                                             std::uint32_t                                         tmem_base_addr,
                                             std::uint32_t                                         base_m,
                                             std::uint32_t,
                                             std::uint32_t,
                                             std::uint32_t,
                                             std::uint32_t                                   epilogue_warp,
                                             std::uint32_t                                   lane,
                                             const cutlass::arch::ClusterTransactionBarrier* empty_barrier,
                                             const cute::TmaDescriptor&,
                                             const void* opaque_args) {
        using namespace rtp_kernel::glm5::detail;
        static_assert(BLOCK_M == 16 && BLOCK_N == 128 && STORE_BLOCK_M == 16 && STORE_BLOCK_N == 128
                      && kSwizzleCDMode == 128 && kNumTMAStoreStages == 2 && kNumUMMAStoreThreads == 256
                      && kGemmType == deep_gemm::GemmType::Normal && !kWithAccumulation
                      && cute::is_same_v<cd_dtype_t, cutlass::bfloat16_t>);
        stage_indexer_k_bf16(smem_cd, stage, tmem_base_addr, epilogue_warp, lane, empty_barrier);
        const auto& args      = *static_cast<const IndexerKCacheArgs*>(opaque_args);
        const int   sublane   = lane & 15;
        const int   local_row = epilogue_warp * 2 + (lane >> 4);
        const int   row       = base_m + local_row;
        const bool  valid     = row < args.rows;
        Bf16x8      input{};
        if (valid) {
            input = load_indexer_k_chunk(smem_cd, stage, local_row, sublane);
        }
        const auto normalized  = indexer_k_layer_norm(input, args, sublane * 8, valid);
        const auto transformed = indexer_k_rope_hadamard(normalized, args, row, sublane, sublane * 8, valid);
        store_mxfp8_key(transformed, args, row, sublane, valid);
        stage = (stage + 1) % kNumTMAStoreStages;
    }
};

}  // namespace rtp_llm
