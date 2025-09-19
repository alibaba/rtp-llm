#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include "rtp_llm/cpp/cuda/deep_gemm/utils.h"
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <exception>

#ifdef ENABLE_FP8
#include "rtp_llm/cpp/cuda/deep_gemm/include/fp8_gemm.cuh"
#endif

namespace rtp_llm {

#ifdef ENABLE_FP8

template<uint32_t            N,
         uint32_t            K,
         uint32_t            BM,
         uint32_t            BN,
         uint32_t            BK,
         uint32_t            GROUP_NUM,
         uint32_t            NUM_STAGES,
         uint32_t            NUM_TMA_MULTICAST,
         deep_gemm::GemmType GEMM_TYPE>
static void dispatchGemm(__nv_bfloat16* output,
                         __nv_fp8_e4m3* lhs,
                         float*         lhs_scale,
                         __nv_fp8_e4m3* rhs,
                         float*         rhs_scale,
                         int*           grouped_layout,
                         uint32_t       m,
                         cudaStream_t   stream,
                         uint32_t       num_sms,
                         uint32_t       smem_size,
                         bool           swap_ab) {
    if (swap_ab) {
        if constexpr (GEMM_TYPE == deep_gemm::GemmType::Normal || GEMM_TYPE == deep_gemm::GemmType::GroupedMasked) {
            using gemm_runner =
                deep_gemm::GemmSwapAB<N, K, BM, BN, BK, GROUP_NUM, NUM_STAGES, NUM_TMA_MULTICAST, GEMM_TYPE>;
            auto tma_a_desc        = gemm_runner::template make_2d_tma_a_desc<__nv_fp8_e4m3>(rhs);
            auto tma_b_desc        = gemm_runner::template make_2d_tma_b_desc<__nv_fp8_e4m3>(lhs, m);
            auto tma_scales_b_desc = gemm_runner::template make_2d_tma_scales_b_desc<float>(lhs_scale, m);
            auto tma_d_desc        = gemm_runner::template make_2d_tma_d_desc<__nv_bfloat16>(output, m);
            gemm_runner::run(output,
                             rhs_scale,
                             grouped_layout,
                             m,
                             tma_a_desc,
                             tma_b_desc,
                             tma_scales_b_desc,
                             tma_d_desc,
                             stream,
                             num_sms,
                             smem_size);
        } else {
            throw std::runtime_error(
                "deep_gemm::GemmSwapAB does not support gemm type other than Normal and GroupedMasked yet.");
        }
    } else {
        using gemm_runner      = deep_gemm::Gemm<N, K, BM, BN, BK, GROUP_NUM, NUM_STAGES, NUM_TMA_MULTICAST, GEMM_TYPE>;
        auto tma_a_desc        = gemm_runner::template make_2d_tma_a_desc<__nv_fp8_e4m3>(lhs, m);
        auto tma_b_desc        = gemm_runner::template make_2d_tma_b_desc<__nv_fp8_e4m3>(rhs);
        auto tma_scales_a_desc = gemm_runner::template make_2d_tma_scales_a_desc<float>(lhs_scale, m);
        auto tma_d_desc        = gemm_runner::template make_2d_tma_d_desc<__nv_bfloat16>(output, m);
        gemm_runner::run(output,
                         rhs_scale,
                         grouped_layout,
                         m,
                         tma_a_desc,
                         tma_b_desc,
                         tma_scales_a_desc,
                         tma_d_desc,
                         stream,
                         num_sms,
                         smem_size);
    }
}

#define DISPATCH_NUM_STAGES_AND_TMA(NUM_STAGES, NUM_TMA_MULTICAST)                                                     \
    if (num_stages == NUM_STAGES && num_tma_multicast == NUM_TMA_MULTICAST) {                                          \
        dispatchGemm<N, K, BM, BN, BK, GROUP_NUM, NUM_STAGES, NUM_TMA_MULTICAST, (deep_gemm::GemmType)GEMM_TYPE>(      \
            output, lhs, lhs_scale, rhs, rhs_scale, grouped_layout, m, stream, num_sms, smem_size, swap_ab);           \
    }

template<uint32_t            N,
         uint32_t            K,
         uint32_t            BM,
         uint32_t            BN,
         uint32_t            BK,
         uint32_t            GROUP_NUM,
         deep_gemm::GemmType GEMM_TYPE>
void dispatchNumStagesAndTma(__nv_bfloat16* output,
                             __nv_fp8_e4m3* lhs,
                             float*         lhs_scale,
                             __nv_fp8_e4m3* rhs,
                             float*         rhs_scale,
                             int*           grouped_layout,
                             uint32_t       m,
                             uint32_t       num_stages,
                             uint32_t       num_tma_multicast,
                             cudaStream_t   stream,
                             uint32_t       num_sms,
                             uint32_t       smem_size,
                             bool           swap_ab) {
    DISPATCH_NUM_STAGES_AND_TMA(8, 1)
    DISPATCH_NUM_STAGES_AND_TMA(7, 1)
    DISPATCH_NUM_STAGES_AND_TMA(6, 1)
    DISPATCH_NUM_STAGES_AND_TMA(5, 1)
    DISPATCH_NUM_STAGES_AND_TMA(4, 1)

    DISPATCH_NUM_STAGES_AND_TMA(8, 2)
    DISPATCH_NUM_STAGES_AND_TMA(7, 2)
    DISPATCH_NUM_STAGES_AND_TMA(6, 2)
    DISPATCH_NUM_STAGES_AND_TMA(5, 2)
    DISPATCH_NUM_STAGES_AND_TMA(4, 2)
}

#define DISPATCH_BLOCK_N(BM, BN, BK)                                                                                   \
    if (bm == BM && bn == BN && bk == BK) {                                                                            \
        dispatchNumStagesAndTma<N, K, BM, BN, BK, GROUP_NUM, (deep_gemm::GemmType)GEMM_TYPE>(output,                   \
                                                                                             lhs,                      \
                                                                                             lhs_scale,                \
                                                                                             rhs,                      \
                                                                                             rhs_scale,                \
                                                                                             grouped_layout,           \
                                                                                             m,                        \
                                                                                             num_stages,               \
                                                                                             num_tma_multicast,        \
                                                                                             stream,                   \
                                                                                             num_sms,                  \
                                                                                             smem_size,                \
                                                                                             swap_ab);                 \
        return;                                                                                                        \
    }

#define DISPATCH_BLOCK_MK(BM, BK)                                                                                      \
    DISPATCH_BLOCK_N(BM, 16, BK)                                                                                       \
    DISPATCH_BLOCK_N(BM, 24, BK)                                                                                       \
    DISPATCH_BLOCK_N(BM, 32, BK)                                                                                       \
    DISPATCH_BLOCK_N(BM, 40, BK)                                                                                       \
    DISPATCH_BLOCK_N(BM, 48, BK)                                                                                       \
    DISPATCH_BLOCK_N(BM, 56, BK)                                                                                       \
    DISPATCH_BLOCK_N(BM, 64, BK)                                                                                       \
    DISPATCH_BLOCK_N(BM, 72, BK)                                                                                       \
    DISPATCH_BLOCK_N(BM, 80, BK)                                                                                       \
    DISPATCH_BLOCK_N(BM, 88, BK)                                                                                       \
    DISPATCH_BLOCK_N(BM, 96, BK)                                                                                       \
    DISPATCH_BLOCK_N(BM, 104, BK)                                                                                      \
    DISPATCH_BLOCK_N(BM, 112, BK)                                                                                      \
    DISPATCH_BLOCK_N(BM, 120, BK)                                                                                      \
    DISPATCH_BLOCK_N(BM, 128, BK)

#endif

template<uint32_t N, uint32_t K, uint32_t GROUP_NUM, DeepGemmType GEMM_TYPE>
void dispatchBlockNK(__nv_bfloat16* output,
                     __nv_fp8_e4m3* lhs,
                     float*         lhs_scale,
                     __nv_fp8_e4m3* rhs,
                     float*         rhs_scale,
                     int*           grouped_layout,
                     uint32_t       m,
                     uint32_t       bm,
                     uint32_t       bn,
                     uint32_t       bk,
                     uint32_t       num_stages,
                     uint32_t       num_tma_multicast,
                     cudaStream_t   stream,
                     uint32_t       num_sms,
                     uint32_t       smem_size,
                     bool           swap_ab) {
#ifdef ENABLE_FP8
    DISPATCH_BLOCK_MK(64, 128)
    DISPATCH_BLOCK_MK(128, 128)
    throw std::runtime_error("DISPATCH_DEEP_GEMM(BLOCK_M=%u, BLOCK_N=%u, BLOCK_K=%u) no template found");
#endif
}
}  // namespace rtp_llm
