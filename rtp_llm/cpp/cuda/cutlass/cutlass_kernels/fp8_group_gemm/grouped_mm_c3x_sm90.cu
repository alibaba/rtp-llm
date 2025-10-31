#include <cudaTypedefs.h>

#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>
#include <type_traits>

#include "cutlass/cutlass.h"
#include "rtp_llm/cpp/cuda/cutlass/cutlass_kernels/fp8_group_gemm/include/heuristics_search.hpp"
#include "rtp_llm/cpp/cuda/cutlass/cutlass_kernels/fp8_group_gemm/grouped_mm_c3x.cuh"
#include "rtp_llm/cpp/cuda/cutlass/cutlass_kernels/fp8_group_gemm/fp8_group_gemm.h"

namespace tk = tensorrt_llm::kernels;
namespace tc = tensorrt_llm::cutlass_extensions;

namespace {

// template<typename InType, typename OutType, template<typename, typename, typename> typename Epilogue>
// struct sm90_fp8_config_default {
//     // M in (64, inf)
//     static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
//     using KernelSchedule   = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpongFP8FastAccum;
//     using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
//     using TileShape        = cute::Shape<cute::_64, cute::_256, cute::_128>;
//     using ClusterShape     = cute::Shape<cute::_1, cute::_1, cute::_1>;
//     using ArchTag          = cutlass::arch::Sm90;

//     using Cutlass3xGemm = cutlass_3x_group_gemm<InType,
//                                                 OutType,
//                                                 ArchTag,
//                                                 Epilogue,
//                                                 TileShape,
//                                                 ClusterShape,
//                                                 KernelSchedule,
//                                                 EpilogueSchedule>;
// };

// template<typename InType, typename OutType, template<typename, typename, typename> typename Epilogue>
// struct sm90_fp8_config_M4 {
//     // M in [1, 4]
//     static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
//     using KernelSchedule   = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpongFP8FastAccum;
//     using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
//     using TileShape        = cute::Shape<cute::_128, cute::_16, cute::_128>;
//     using ClusterShape     = cute::Shape<cute::_1, cute::_1, cute::_1>;
//     using ArchTag          = cutlass::arch::Sm90;

//     using Cutlass3xGemm = cutlass_3x_group_gemm<InType,
//                                                 OutType,
//                                                 ArchTag,
//                                                 Epilogue,
//                                                 TileShape,
//                                                 ClusterShape,
//                                                 KernelSchedule,
//                                                 EpilogueSchedule,
//                                                 true>;
// };

// template<typename InType, typename OutType, template<typename, typename, typename> typename Epilogue>
// struct sm90_fp8_config_M64 {
//     // M in (4, 64]
//     static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
//     using KernelSchedule   = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpongFP8FastAccum;
//     using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
//     using TileShape        = cute::Shape<cute::_128, cute::_16, cute::_256>;
//     using ClusterShape     = cute::Shape<cute::_1, cute::_1, cute::_1>;
//     using ArchTag          = cutlass::arch::Sm90;

//     using Cutlass3xGemm = cutlass_3x_group_gemm<InType,
//                                                 OutType,
//                                                 ArchTag,
//                                                 Epilogue,
//                                                 TileShape,
//                                                 ClusterShape,
//                                                 KernelSchedule,
//                                                 EpilogueSchedule,
//                                                 true>;
// };

// template<typename InType, typename OutType, template<typename, typename, typename> typename Epilogue>
// struct sm90_fp8_config_K8192 {
//     // K in [8192, inf)
//     static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
//     using KernelSchedule   = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpongFP8FastAccum;
//     using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
//     using TileShape        = cute::Shape<cute::_128, cute::_128, cute::_128>;
//     using ClusterShape     = cute::Shape<cute::_1, cute::_1, cute::_1>;
//     using ArchTag          = cutlass::arch::Sm90;

//     using Cutlass3xGemm = cutlass_3x_group_gemm<InType,
//                                                 OutType,
//                                                 ArchTag,
//                                                 Epilogue,
//                                                 TileShape,
//                                                 ClusterShape,
//                                                 KernelSchedule,
//                                                 EpilogueSchedule>;
// };

// template<typename InType, typename OutType, template<typename, typename, typename> typename Epilogue>
// struct sm90_fp8_config_N8192 {
//     // N in [8192, inf)
//     static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
//     using KernelSchedule   = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpongFP8FastAccum;
//     using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
//     using TileShape        = cute::Shape<cute::_64, cute::_128, cute::_256>;
//     using ClusterShape     = cute::Shape<cute::_1, cute::_1, cute::_1>;
//     using ArchTag          = cutlass::arch::Sm90;

//     using Cutlass3xGemm = cutlass_3x_group_gemm<InType,
//                                                 OutType,
//                                                 ArchTag,
//                                                 Epilogue,
//                                                 TileShape,
//                                                 ClusterShape,
//                                                 KernelSchedule,
//                                                 EpilogueSchedule>;
// };

template<typename InType, typename OutType, bool SWAP_AB, typename TileShape>
void dispatchMoeGemmSM90SelectClusterShape(torch::Tensor&        out_tensors,
                                           torch::Tensor const&  a_tensors,
                                           torch::Tensor const&  b_tensors,
                                           torch::Tensor const&  a_scales,
                                           torch::Tensor const&  b_scales,
                                           torch::Tensor const&  expert_offsets,
                                           torch::Tensor const&  problem_sizes,
                                           torch::Tensor const&  a_strides,
                                           torch::Tensor const&  b_strides,
                                           torch::Tensor const&  c_strides,
                                           bool                  per_act_token,
                                           bool                  per_out_ch,
                                           tc::CutlassGemmConfig gemm_config) {

    using KernelSchedule   = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpongFP8FastAccum;
    using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;

#define CLUSTER_SHAPE_CASE(M, N, K)                                                                                    \
    case tc::ClusterShape::ClusterShape_##M##x##N##x##K: {                                                             \
        using ClusterShape  = cute::Shape<cute::_##M, cute::_##N, cute::_##K>;                                         \
        using Cutlass3xGemm = cutlass_3x_group_gemm<InType,                                                            \
                                                    OutType,                                                           \
                                                    cutlass::arch::Sm90 /*ArchTag*/,                                   \
                                                    rtp_llm::c3x::ScaledEpilogueArray /*Epilogue*/,                    \
                                                    TileShape,                                                         \
                                                    ClusterShape,                                                      \
                                                    KernelSchedule,                                                    \
                                                    EpilogueSchedule,                                                  \
                                                    SWAP_AB>;                                                          \
        cutlass_group_gemm_caller<Cutlass3xGemm>(out_tensors,                                                          \
                                                 a_tensors,                                                            \
                                                 b_tensors,                                                            \
                                                 a_scales,                                                             \
                                                 b_scales,                                                             \
                                                 expert_offsets,                                                       \
                                                 problem_sizes,                                                        \
                                                 a_strides,                                                            \
                                                 b_strides,                                                            \
                                                 c_strides,                                                            \
                                                 per_act_token,                                                        \
                                                 per_out_ch);                                                          \
        break;                                                                                                         \
    }

    switch (gemm_config.cluster_shape) {
        CLUSTER_SHAPE_CASE(1, 1, 1)
        CLUSTER_SHAPE_CASE(2, 1, 1)
        CLUSTER_SHAPE_CASE(1, 2, 1)
        CLUSTER_SHAPE_CASE(2, 2, 1)
        // CLUSTER_SHAPE_CASE(1, 8, 1)
        // CLUSTER_SHAPE_CASE(8, 1, 1)
    }
#undef CLUSTER_SHAPE_CASE
}

template<typename InType, typename OutType, bool SWAP_AB>
void dispatchMoeGemmSM90SelectTileShape(torch::Tensor&        out_tensors,
                                        torch::Tensor const&  a_tensors,
                                        torch::Tensor const&  b_tensors,
                                        torch::Tensor const&  a_scales,
                                        torch::Tensor const&  b_scales,
                                        torch::Tensor const&  expert_offsets,
                                        torch::Tensor const&  problem_sizes,
                                        torch::Tensor const&  a_strides,
                                        torch::Tensor const&  b_strides,
                                        torch::Tensor const&  c_strides,
                                        bool                  per_act_token,
                                        bool                  per_out_ch,
                                        tc::CutlassGemmConfig gemm_config) {

#define TILE_SHAPE_CASE(M, N, K, SWAP_AB)                                                                              \
    case tc::CutlassTileConfigSM90::CtaShape##M##x##N##x##K##B: {                                                      \
        using TileShape = cute::Shape<cute::_##M, cute::_##N, cute::_##K>;                                             \
        dispatchMoeGemmSM90SelectClusterShape<InType, OutType, SWAP_AB, TileShape>(out_tensors,                        \
                                                                                   a_tensors,                          \
                                                                                   b_tensors,                          \
                                                                                   a_scales,                           \
                                                                                   b_scales,                           \
                                                                                   expert_offsets,                     \
                                                                                   problem_sizes,                      \
                                                                                   a_strides,                          \
                                                                                   b_strides,                          \
                                                                                   c_strides,                          \
                                                                                   per_act_token,                      \
                                                                                   per_out_ch,                         \
                                                                                   gemm_config);                       \
        break;                                                                                                         \
    }

    switch (gemm_config.tile_config_sm90) {
        // TILE_SHAPE_CASE(64, 16, 128, SWAP_AB)
        // TILE_SHAPE_CASE(64, 32, 128, SWAP_AB)
        // TILE_SHAPE_CASE(64, 64, 128, SWAP_AB)
        // TILE_SHAPE_CASE(64, 128, 128, SWAP_AB)
        TILE_SHAPE_CASE(64, 256, 128, SWAP_AB)
        TILE_SHAPE_CASE(128, 16, 128, SWAP_AB)
        TILE_SHAPE_CASE(128, 32, 128, SWAP_AB)
        TILE_SHAPE_CASE(128, 64, 128, SWAP_AB)
        TILE_SHAPE_CASE(128, 128, 128, SWAP_AB)
        TILE_SHAPE_CASE(128, 256, 128, SWAP_AB)
        TILE_SHAPE_CASE(256, 128, 128, SWAP_AB)
        TILE_SHAPE_CASE(128, 16, 256, SWAP_AB)
    }

#undef TILE_SHAPE_CASE
}

template<typename InType, typename OutType>
void run_cutlass_moe_mm_sm90(torch::Tensor&       out_tensors,
                             torch::Tensor const& a_tensors,
                             torch::Tensor const& b_tensors,
                             torch::Tensor const& a_scales,
                             torch::Tensor const& b_scales,
                             torch::Tensor const& expert_offsets,
                             torch::Tensor const& problem_sizes,
                             torch::Tensor const& a_strides,
                             torch::Tensor const& b_strides,
                             torch::Tensor const& c_strides,
                             bool                 per_act_token,
                             bool                 per_out_ch,
                             int32_t              num_sms) {
    TORCH_CHECK(a_tensors.size(0) > 0, "No input A tensors provided.");
    TORCH_CHECK(b_tensors.size(0) > 0, "No input B tensors provided.");
    TORCH_CHECK(out_tensors.size(0) > 0, "No output tensors provided.");

    TORCH_CHECK(a_tensors.dtype() == torch::kFloat8_e4m3fn, "A tensors must be of type float8_e4m3fn.");
    TORCH_CHECK(b_tensors.dtype() == torch::kFloat8_e4m3fn, "B tensors must be of type float8_e4m3fn.");

    // using Cutlass3xGemmN8192 =
    //     typename sm90_fp8_config_N8192<InType, OutType, rtp_llm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;
    // using Cutlass3xGemmK8192 =
    //     typename sm90_fp8_config_K8192<InType, OutType, rtp_llm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;
    // using Cutlass3xGemmM4 =
    //     typename sm90_fp8_config_M4<InType, OutType, rtp_llm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;
    // using Cutlass3xGemmM64 =
    //     typename sm90_fp8_config_M64<InType, OutType, rtp_llm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;
    // using Cutlass3xGemmDefault =
    //     typename sm90_fp8_config_default<InType, OutType, rtp_llm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;

    int const m           = a_tensors.size(0);
    int const n           = out_tensors.size(1);
    int const k           = a_tensors.size(1);
    int const num_experts = b_tensors.size(0);
    // heurisitc search a best group gemm config
    tc::CutlassGemmConfig estimate_best_config =
        rtp_llm::estimate_best_config_customized_sm90<InType, OutType>(m, n, k, 1, num_sms);

    bool swap_ab = (m <= 64);
    static_assert(std::is_same<InType, cutlass::float_e4m3_t>());

    if (swap_ab) {
        dispatchMoeGemmSM90SelectTileShape<InType, OutType, true>(out_tensors,
                                                                  a_tensors,
                                                                  b_tensors,
                                                                  a_scales,
                                                                  b_scales,
                                                                  expert_offsets,
                                                                  problem_sizes,
                                                                  a_strides,
                                                                  b_strides,
                                                                  c_strides,
                                                                  per_act_token,
                                                                  per_out_ch,
                                                                  estimate_best_config);

    } else {
        dispatchMoeGemmSM90SelectTileShape<InType, OutType, false>(out_tensors,
                                                                   a_tensors,
                                                                   b_tensors,
                                                                   a_scales,
                                                                   b_scales,
                                                                   expert_offsets,
                                                                   problem_sizes,
                                                                   a_strides,
                                                                   b_strides,
                                                                   c_strides,
                                                                   per_act_token,
                                                                   per_out_ch,
                                                                   estimate_best_config);
    }

    // // Use swap_ab for M <= 64 by default to reduce padding
    // if (m <= 4) {
    //     cutlass_group_gemm_caller<Cutlass3xGemmM4>(out_tensors,
    //                                                a_tensors,
    //                                                b_tensors,
    //                                                a_scales,
    //                                                b_scales,
    //                                                expert_offsets,
    //                                                problem_sizes,
    //                                                a_strides,
    //                                                b_strides,
    //                                                c_strides,
    //                                                per_act_token,
    //                                                per_out_ch);
    // } else if (m <= 64) {
    //     cutlass_group_gemm_caller<Cutlass3xGemmM64>(out_tensors,
    //                                                 a_tensors,
    //                                                 b_tensors,
    //                                                 a_scales,
    //                                                 b_scales,
    //                                                 expert_offsets,
    //                                                 problem_sizes,
    //                                                 a_strides,
    //                                                 b_strides,
    //                                                 c_strides,
    //                                                 per_act_token,
    //                                                 per_out_ch);
    // } else if (n >= 8192) {
    //     cutlass_group_gemm_caller<Cutlass3xGemmN8192>(out_tensors,
    //                                                   a_tensors,
    //                                                   b_tensors,
    //                                                   a_scales,
    //                                                   b_scales,
    //                                                   expert_offsets,
    //                                                   problem_sizes,
    //                                                   a_strides,
    //                                                   b_strides,
    //                                                   c_strides,
    //                                                   per_act_token,
    //                                                   per_out_ch);
    // } else if (k >= 8192) {
    //     cutlass_group_gemm_caller<Cutlass3xGemmK8192>(out_tensors,
    //                                                   a_tensors,
    //                                                   b_tensors,
    //                                                   a_scales,
    //                                                   b_scales,
    //                                                   expert_offsets,
    //                                                   problem_sizes,
    //                                                   a_strides,
    //                                                   b_strides,
    //                                                   c_strides,
    //                                                   per_act_token,
    //                                                   per_out_ch);
    // } else {
    //     cutlass_group_gemm_caller<Cutlass3xGemmDefault>(out_tensors,
    //                                                     a_tensors,
    //                                                     b_tensors,
    //                                                     a_scales,
    //                                                     b_scales,
    //                                                     expert_offsets,
    //                                                     problem_sizes,
    //                                                     a_strides,
    //                                                     b_strides,
    //                                                     c_strides,
    //                                                     per_act_token,
    //                                                     per_out_ch);
    // }
}

}  // namespace

void rtp_llm::cutlass_moe_mm(torch::Tensor&       out_tensors,
                             torch::Tensor const& a_tensors,
                             torch::Tensor const& b_tensors,
                             torch::Tensor const& a_scales,
                             torch::Tensor const& b_scales,
                             torch::Tensor const& expert_offsets,
                             torch::Tensor const& problem_sizes,
                             torch::Tensor const& a_strides,
                             torch::Tensor const& b_strides,
                             torch::Tensor const& c_strides,
                             bool                 per_act_token,
                             bool                 per_out_ch) {
    int32_t version_num = rtp_llm::get_sm_version_num();
    int32_t num_sms     = rtp_llm::getMultiProcessorCount();
    // #if defined ENABLE_CUTLASS_MOE_SM90 && ENABLE_CUTLASS_MOE_SM90
    if (version_num >= 90) {
        if (out_tensors.dtype() == torch::kBFloat16) {
            run_cutlass_moe_mm_sm90<cutlass::float_e4m3_t, cutlass::bfloat16_t>(out_tensors,
                                                                                a_tensors,
                                                                                b_tensors,
                                                                                a_scales,
                                                                                b_scales,
                                                                                expert_offsets,
                                                                                problem_sizes,
                                                                                a_strides,
                                                                                b_strides,
                                                                                c_strides,
                                                                                per_act_token,
                                                                                per_out_ch,
                                                                                num_sms);
        } else {
            run_cutlass_moe_mm_sm90<cutlass::float_e4m3_t, cutlass::half_t>(out_tensors,
                                                                            a_tensors,
                                                                            b_tensors,
                                                                            a_scales,
                                                                            b_scales,
                                                                            expert_offsets,
                                                                            problem_sizes,
                                                                            a_strides,
                                                                            b_strides,
                                                                            c_strides,
                                                                            per_act_token,
                                                                            per_out_ch,
                                                                            num_sms);
        }
        return;
    }
    // #endif
    //     TORCH_CHECK_NOT_IMPLEMENTED(
    //         false, "No compiled cutlass_scaled_mm for CUDA device capability: ", version_num, ". Required capability:
    //         90");
}