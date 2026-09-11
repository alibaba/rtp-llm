// Reuse the pinned RTP GEMM/TMEM staging, with a local precision-preserving
// epilogue. The installed q_b epilogue is interleaved-only and has a different
// FMA order; do not substitute it for the TP8 split_q_rope contract.
#pragma once
#include <rtp_kernel/glm5/q_b_proj.cuh>

namespace rtp_llm::glm5_h8 {
namespace qb = rtp_kernel::glm5::q_b;

struct Args {
    const float*   cos_sin;
    const void*    positions;
    __nv_bfloat16* q_nope;
    __nv_bfloat16* query;
    int            rows;
    bool           positions64;
    bool           neox;
};

template<typename SmemCd>
__device__ __forceinline__ void
finalize(const SmemCd& smem, uint32_t stage, const Args& args, int base_m, int base_n, int warp, int lane) {
    const int sublane   = lane & 15;
    const int local_row = warp * 2 + (lane >> 4);
    const int row       = base_m + local_row;
    if (row >= args.rows)
        return;
    const int  head       = base_n / 256;
    const int  token_head = row * 8 + head;
    const int  half       = (base_n / 128) & 1;
    const auto packed     = qb::load_chunk(smem, stage, local_row, sublane);
    if (half == 0 || sublane < 8) {
        const int offset                                                        = half * 128 + sublane * 8;
        *reinterpret_cast<qb::Bf16x8*>(args.q_nope + token_head * 192 + offset) = packed;
        return;
    }
    const int     pe       = (sublane - 8) * 8;
    const int64_t position = args.positions64 ? static_cast<const int64_t*>(args.positions)[row] :
                                                static_cast<const int32_t*>(args.positions)[row];
    const float*  cos      = args.cos_sin + position * 64;
    const float*  sin      = cos + 32;
    float         values[8];
    qb::unpack_bf16x8(packed, values);
    if (args.neox) {
        float partner[8];
        qb::unpack_bf16x8(qb::load_chunk(smem, stage, local_row, sublane < 12 ? sublane + 4 : sublane - 4), partner);
        const int pair = pe % 32;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float other = sublane < 12 ? -partner[i] : partner[i];
            values[i]         = __fmaf_rn(values[i], cos[pair + i], __fmul_rn(other, sin[pair + i]));
        }
    } else {
#pragma unroll
        for (int i = 0; i < 8; i += 2) {
            const float x0 = values[i], x1 = values[i + 1];
            values[i]     = __fmaf_rn(x0, cos[(pe + i) / 2], __fmul_rn(-x1, sin[(pe + i) / 2]));
            values[i + 1] = __fmaf_rn(x1, cos[(pe + i) / 2], __fmul_rn(x0, sin[(pe + i) / 2]));
        }
    }
    *reinterpret_cast<qb::Bf16x8*>(args.query + token_head * 576 + 512 + pe) = qb::pack_bf16x8(values);
}

struct Epilogue {
    template<uint32_t            BLOCK_M,
             uint32_t            BLOCK_N,
             uint32_t            STORE_M,
             uint32_t            STORE_N,
             uint32_t            SWIZZLE,
             uint32_t            STAGES,
             uint32_t            THREADS,
             deep_gemm::GemmType TYPE,
             bool                ACCUM,
             typename Dtype,
             typename Transform,
             typename Pattern>
    CUTLASS_DEVICE static void store_swap_ab(const deep_gemm::utils::PatternVisitor<Pattern>& smem,
                                             uint32_t&                                        stage,
                                             uint32_t                                         tmem,
                                             uint32_t                                         base_m,
                                             uint32_t                                         base_n,
                                             uint32_t,
                                             uint32_t                                        effective_m,
                                             uint32_t                                        warp,
                                             uint32_t                                        lane,
                                             const cutlass::arch::ClusterTransactionBarrier* empty,
                                             const cute::TmaDescriptor&,
                                             const void* opaque) {
        static_assert(BLOCK_M % 16 == 0 && BLOCK_M <= 128 && BLOCK_N == 128);
        static_assert(STORE_M == 16 && STORE_N == 128 && SWIZZLE == 128 && STAGES == 2);
        static_assert((THREADS == 256 || THREADS == 512) && !ACCUM);
        static_assert(TYPE == deep_gemm::GemmType::Normal && cute::is_same_v<Dtype, cutlass::bfloat16_t>);
        const auto&        args       = *static_cast<const Args*>(opaque);
        constexpr uint32_t concurrent = THREADS / 256;
        const uint32_t     stores     = effective_m / 16;
        if constexpr (concurrent == 1) {
            for (uint32_t store = 0; store < stores; ++store) {
                qb::stage_bf16<256>(smem, stage, tmem + store * 16, warp, lane, 0, empty, store + 1 == stores);
                finalize(smem, stage, args, base_m + store * 16, base_n, warp, lane);
                stage = (stage + 1) % 2;
            }
        } else {
            const uint32_t group = warp / 8, local_warp = warp % 8;
            stage         = group;
            bool released = false;
            for (uint32_t store = group; store < stores; store += concurrent) {
                const bool last = store + concurrent >= stores;
                qb::stage_bf16<256>(smem, stage, tmem + store * 16, local_warp, lane, group, empty, last);
                finalize(smem, stage, args, base_m + store * 16, base_n, local_warp, lane);
                released = last;
            }
            if (!released) {
                deep_gemm::ptx::tcgen05_before_thread_sync();
                empty->arrive(0u);
            }
        }
    }
};

template<int M, int STAGES, int EWG>
CUTLASS_GLOBAL void __launch_bounds__(128 * (1 + EWG), 1) q_b_h8_kernel(uint32_t                rows,
                                                                        const __grid_constant__ cute::TmaDescriptor a,
                                                                        const __grid_constant__ cute::TmaDescriptor b,
                                                                        const __grid_constant__ cute::TmaDescriptor sfa,
                                                                        const __grid_constant__ cute::TmaDescriptor sfb,
                                                                        const __grid_constant__ cute::TmaDescriptor cd,
                                                                        Args args) {
    using Config = deep_gemm::Fp8GemmConfig<cute::UMMA::Major::K,
                                            cute::UMMA::Major::K,
                                            128,
                                            128,
                                            128,
                                            0,
                                            2048,
                                            2048,
                                            M,
                                            128,
                                            128,
                                            1,
                                            128,
                                            128,
                                            128,
                                            STAGES,
                                            128,
                                            128 * EWG,
                                            2,
                                            true,
                                            Epilogue,
                                            true,
                                            true,
                                            148,
                                            true,
                                            true,
                                            deep_gemm::GemmType::Normal,
                                            false,
                                            cutlass::float_e4m3_t,
                                            cutlass::float_e4m3_t,
                                            cutlass::bfloat16_t,
                                            deep_gemm::epilogue::transform::EpilogueIdentity>;
    Config::template run<deep_gemm::NoAuxiliaryRole>(
        nullptr, &args, deep_gemm::NoAuxiliaryRole::Arguments{}, rows, 2048, 2048, a, b, sfa, sfb, cd);
}
}  // namespace rtp_llm::glm5_h8
