#include "rtp_llm/cpp/cuda/cutlass/cutlass_kernels/w4a8_group_gemm/include/heuristics_search.hpp"
#include "rtp_llm/cpp/cuda/cutlass/cutlass_kernels/w4a8_group_gemm/w4a8_group_gemm.cuh"
#include "rtp_llm/cpp/cuda/cutlass/cutlass_kernels/w4a8_group_gemm/w4a8_group_gemm.h"

static int get_sm_capability() {
    int dev_id;
    cudaGetDevice(&dev_id);

    int major, minor;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev_id);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev_id);

    int capability = major * 10 + minor;

    return capability;
}

static int get_sm_count() {
    int dev_id;
    cudaGetDevice(&dev_id);

    int sm_cnt;
    cudaDeviceGetAttribute(&sm_cnt, cudaDevAttrMultiProcessorCount, dev_id);

    return sm_cnt;
}

static CutlassGemmConfig get_best_config_sm90(int m, int n, int k, const int& num_groups, const int num_sms) {
    auto best_cluster_shape = ClusterShape::ClusterShape_1x1x1;
    auto best_tile_shape    = CutlassTileConfigSM90::CtaShape256x16x128B;
    if (m <= 16) {
        best_tile_shape = CutlassTileConfigSM90::CtaShape256x16x128B;
    } else if (m <= 32) {
        best_tile_shape = CutlassTileConfigSM90::CtaShape256x32x128B;
    } else if (m <= 64) {
        best_tile_shape = CutlassTileConfigSM90::CtaShape256x64x128B;
    } else {
        best_tile_shape = CutlassTileConfigSM90::CtaShape256x128x128B;
    }
    CutlassGemmConfig chosen_config(
        best_tile_shape, MainloopScheduleType::AUTO, EpilogueScheduleType::AUTO, best_cluster_shape);
    return chosen_config;
}

template<typename AType, typename BType, typename BScaleType, typename OutType, bool SWAP_AB, typename TileShape>
static void dispatch_cluster_shape(torch::Tensor&       output,
                                   torch::Tensor const& a,
                                   torch::Tensor const& b,
                                   torch::Tensor const& b_scales,
                                   torch::Tensor const& a_out_scales,
                                   torch::Tensor const& b_out_scales,
                                   torch::Tensor const& expert_offsets,
                                   torch::Tensor const& problem_sizes,
                                   torch::Tensor const& a_strides,
                                   torch::Tensor const& b_strides,
                                   torch::Tensor const& b_scales_strides,
                                   torch::Tensor const& c_strides,
                                   const int            group_size,
                                   bool                 per_act_token,
                                   bool                 per_out_ch,
                                   CutlassGemmConfig    gemm_config) {
    using KernelSchedule   = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
    using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;

#define CLUSTER_SHAPE_CASE(M, N, K)                                                                                    \
    case ClusterShape::ClusterShape_##M##x##N##x##K: {                                                                 \
        using ClusterShape = cute::Shape<cute::_##M, cute::_##N, cute::_##K>;                                          \
        using GroupGemm    = W4A8GroupGemm<AType,                                                                      \
                                           BType,                                                                      \
                                           BScaleType,                                                                 \
                                           OutType,                                                                    \
                                           cutlass::arch::Sm90 /*ArchTag*/,                                            \
                                           c3x::ScaledEpilogueArray /*Epilogue*/,                                      \
                                           TileShape,                                                                  \
                                           ClusterShape,                                                               \
                                           KernelSchedule,                                                             \
                                           EpilogueSchedule,                                                           \
                                           SWAP_AB>;                                                                   \
        w4a8_group_gemm_caller<GroupGemm>(output,                                                                      \
                                          a,                                                                           \
                                          b,                                                                           \
                                          b_scales,                                                                    \
                                          a_out_scales,                                                                \
                                          b_out_scales,                                                                \
                                          expert_offsets,                                                              \
                                          problem_sizes,                                                               \
                                          a_strides,                                                                   \
                                          b_strides,                                                                   \
                                          b_scales_strides,                                                            \
                                          c_strides,                                                                   \
                                          group_size,                                                                  \
                                          per_act_token,                                                               \
                                          per_out_ch);                                                                 \
        break;                                                                                                         \
    }

    switch (gemm_config.cluster_shape) {
        CLUSTER_SHAPE_CASE(1, 1, 1)
        // CLUSTER_SHAPE_CASE(1, 2, 1)
        // CLUSTER_SHAPE_CASE(2, 1, 1)
    }
#undef CLUSTER_SHAPE_CASE
}

template<typename AType, typename BType, typename BScaleType, typename OutType, bool SWAP_AB>
static void dispatch_tile_shape(torch::Tensor&       output,
                                torch::Tensor const& a,
                                torch::Tensor const& b,
                                torch::Tensor const& b_scales,
                                torch::Tensor const& a_out_scales,
                                torch::Tensor const& b_out_scales,
                                torch::Tensor const& expert_offsets,
                                torch::Tensor const& problem_sizes,
                                torch::Tensor const& a_strides,
                                torch::Tensor const& b_strides,
                                torch::Tensor const& b_scales_strides,
                                torch::Tensor const& c_strides,
                                const int            group_size,
                                bool                 per_act_token,
                                bool                 per_out_ch,
                                CutlassGemmConfig    gemm_config) {
#define TILE_SHAPE_CASE(M, N, K, SWAP_AB)                                                                              \
    case CutlassTileConfigSM90::CtaShape##M##x##N##x##K##B: {                                                          \
        using TileShape = cute::Shape<cute::_##M, cute::_##N, cute::_##K>;                                             \
        dispatch_cluster_shape<AType, BType, BScaleType, OutType, SWAP_AB, TileShape>(output,                          \
                                                                                      a,                               \
                                                                                      b,                               \
                                                                                      b_scales,                        \
                                                                                      a_out_scales,                    \
                                                                                      b_out_scales,                    \
                                                                                      expert_offsets,                  \
                                                                                      problem_sizes,                   \
                                                                                      a_strides,                       \
                                                                                      b_strides,                       \
                                                                                      b_scales_strides,                \
                                                                                      c_strides,                       \
                                                                                      group_size,                      \
                                                                                      per_act_token,                   \
                                                                                      per_out_ch,                      \
                                                                                      gemm_config);                    \
        break;                                                                                                         \
    }

    switch (gemm_config.tile_config_sm90) {
        TILE_SHAPE_CASE(256, 16, 128, SWAP_AB)
        TILE_SHAPE_CASE(256, 32, 128, SWAP_AB)
        TILE_SHAPE_CASE(256, 64, 128, SWAP_AB)
        TILE_SHAPE_CASE(256, 128, 128, SWAP_AB)
    }

#undef TILE_SHAPE_CASE
}

template<typename AType, typename BType, typename BScaleType, typename OutType>
static void dispatch_sm90(torch::Tensor&       output,
                          torch::Tensor const& a,
                          torch::Tensor const& b,
                          torch::Tensor const& b_scales,
                          torch::Tensor const& a_out_scales,
                          torch::Tensor const& b_out_scales,
                          torch::Tensor const& expert_offsets,
                          torch::Tensor const& problem_sizes,
                          torch::Tensor const& a_strides,
                          torch::Tensor const& b_strides,
                          torch::Tensor const& b_scales_strides,
                          torch::Tensor const& c_strides,
                          const int            group_size,
                          const bool           swap_ab,
                          const bool           per_act_token,
                          const bool           per_out_ch,
                          const int32_t        num_sms,
                          const bool           profile,
                          const int            m_tile,
                          const int            n_tile,
                          const int            k_tile,
                          const int            cluster_m,
                          const int            cluster_n,
                          const int            cluster_k) {
    TORCH_CHECK(a.size(0) > 0, "No input A tensors provided.");
    TORCH_CHECK(b.size(0) > 0, "No input B tensors provided.");
    TORCH_CHECK(output.size(0) > 0, "No output tensors provided.");

    TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn, "A tensors must be of type float8_e4m3fn.");
    TORCH_CHECK(b.dtype() == torch::kInt8, "B tensors must be of type int8.");
    TORCH_CHECK(b_scales.dtype() == torch::kFloat8_e4m3fn, "B scales must be of type float8_e4m3fn.");
    TORCH_CHECK(group_size > 0, "Group size must be positive.");

    static_assert(std::is_same<AType, cutlass::float_e4m3_t>());
    static_assert(std::is_same<BType, cutlass::int4b_t>());
    static_assert(std::is_same<BScaleType, cutlass::float_e4m3_t>());

    int const m           = a.size(0);
    int const n           = output.size(1);
    int const k           = a.size(1);
    int const num_experts = b.size(0);

    if (profile) {
        // auto tile_config = tile_shape_converter(m_tile, n_tile, k_tile);
        // auto cluster_config = cluster_shape_converter(cluster_m, cluster_n, cluster_k);

        // CutlassGemmConfig profile_config(tile_config, MainloopScheduleType::AUTO, EpilogueScheduleType::AUTO,
        //                                  cluster_config);

        // if (swap_ab) {
        //     dispatch_tile_shape<AType, BType, BScaleType, OutType, true>(
        //         output, a, b, b_scales, a_out_scales, b_out_scales, expert_offsets,
        //         problem_sizes, a_strides, b_strides, b_scales_strides, c_strides, group_size, per_act_token,
        //         per_out_ch, profile_config);
        // } else {
        //     dispatch_tile_shape<AType, BType, BScaleType, OutType, false>(
        //         output, a, b, b_scales, a_out_scales, b_out_scales, expert_offsets,
        //         problem_sizes, a_strides, b_strides, b_scales_strides, c_strides, group_size, per_act_token,
        //         per_out_ch, profile_config);
        // }
    } else {
        CutlassGemmConfig estimate_best_config = get_best_config_sm90(m, n, k, 1, num_sms);

        if (swap_ab) {
            dispatch_tile_shape<AType, BType, BScaleType, OutType, true>(output,
                                                                         a,
                                                                         b,
                                                                         b_scales,
                                                                         a_out_scales,
                                                                         b_out_scales,
                                                                         expert_offsets,
                                                                         problem_sizes,
                                                                         a_strides,
                                                                         b_strides,
                                                                         b_scales_strides,
                                                                         c_strides,
                                                                         group_size,
                                                                         per_act_token,
                                                                         per_out_ch,
                                                                         estimate_best_config);
        } else {
            // dispatch_tile_shape<AType, BType, BScaleType, OutType, false>(
            //     output, a, b, b_scales, a_out_scales, b_out_scales, expert_offsets, problem_sizes, a_strides,
            //     b_strides, b_scales_strides, c_strides, group_size, per_act_token, per_out_ch, estimate_best_config);
        }
    }
}

void rtp_llm::run_w4a8_group_gemm(torch::Tensor&       output,
                                  torch::Tensor const& a,
                                  torch::Tensor const& b,
                                  torch::Tensor const& b_scales,
                                  torch::Tensor const& a_out_scales,
                                  torch::Tensor const& b_out_scales,
                                  torch::Tensor const& expert_offsets,
                                  torch::Tensor const& problem_sizes,
                                  torch::Tensor const& a_strides,
                                  torch::Tensor const& b_strides,
                                  torch::Tensor const& b_scales_strides,
                                  torch::Tensor const& c_strides,
                                  const int            group_size,
                                  const bool           swap_ab,
                                  const bool           per_act_token,
                                  const bool           per_out_ch,
                                  const bool           profile,
                                  const int            m_tile,
                                  const int            n_tile,
                                  const int            k_tile,
                                  const int            cluster_m,
                                  const int            cluster_n,
                                  const int            cluster_k) {
    int sm_capability = get_sm_capability();
    int sm_cnt        = get_sm_count();
    if (sm_capability >= 90) {
        TORCH_CHECK(output.dtype() == torch::kBFloat16 || output.dtype() == torch::kFloat16,
                    "Output type must be kBFloat16 or kFloat16.");
        if (output.dtype() == torch::kBFloat16) {
            dispatch_sm90<cutlass::float_e4m3_t, cutlass::int4b_t, cutlass::float_e4m3_t, cutlass::bfloat16_t>(
                output,
                a,
                b,
                b_scales,
                a_out_scales,
                b_out_scales,
                expert_offsets,
                problem_sizes,
                a_strides,
                b_strides,
                b_scales_strides,
                c_strides,
                group_size,
                swap_ab,
                per_act_token,
                per_out_ch,
                sm_cnt,
                profile,
                m_tile,
                n_tile,
                k_tile,
                cluster_m,
                cluster_n,
                cluster_k);
        } else {
            dispatch_sm90<cutlass::float_e4m3_t, cutlass::int4b_t, cutlass::float_e4m3_t, cutlass::half_t>(
                output,
                a,
                b,
                b_scales,
                a_out_scales,
                b_out_scales,
                expert_offsets,
                problem_sizes,
                a_strides,
                b_strides,
                b_scales_strides,
                c_strides,
                group_size,
                swap_ab,
                per_act_token,
                per_out_ch,
                sm_cnt,
                profile,
                m_tile,
                n_tile,
                k_tile,
                cluster_m,
                cluster_n,
                cluster_k);
        }
    }
}
