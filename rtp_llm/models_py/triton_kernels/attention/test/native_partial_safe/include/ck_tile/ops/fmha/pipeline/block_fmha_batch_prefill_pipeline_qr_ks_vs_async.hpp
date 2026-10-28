// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"
#include "ck_tile/ops/fmha/block/block_attention_kv_load_mode_enum.hpp"
#include "ck_tile/ops/fmha/block/block_attention_kvcache_layout_enum.hpp"
#include "ck_tile/ops/fmha/block/block_attention_quant_scale_enum.hpp"
#include "ck_tile/ops/fmha/block/block_dropout.hpp"
#include "ck_tile/ops/fmha/block/variants.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_batch_prefill_pipeline_qr_ks_vs_async_default_policy.hpp"
#include "ck_tile/ops/reduce/block/block_reduce.hpp"

namespace ck_tile {

// Load physical pages from page_idx lookup table.
// K cache: per-token lookup (each k0 may have different page_id)
// V cache: depends on whether V tile crosses pages
//   - Crosses pages: per-token lookup
//   - Single page: lane0 lookup once, broadcast to all
// Output: physical_pages array with kLoopCount elements
template <typename IndexArrayType,
          typename CoordVecType,
          index_t kCoordAxis,
          index_t kPageBlockSize,
          index_t kLoopStart,
          index_t kLoopCount,
          index_t kLoopStride,
          BlockAttentionKVCacheMemoryLayoutEnum kKVMemoryLayout,
          bool kIsKcache,
          index_t kN0>
CK_TILE_DEVICE void load_physical_pages(const index_t* page_idx,
                                        const CoordVecType& coord_vec,
                                        index_t global_seq_offset,
                                        IndexArrayType& physical_pages,
                                        index_t max_page_table_idx)
{
    static constexpr index_t kLog2PageSize = [] {
        index_t shift = 0;
        index_t val   = kPageBlockSize;
        while(val > 1)
        {
            val >>= 1;
            shift++;
        }
        return shift;
    }();

    const index_t& thread_coord_start = coord_vec[kCoordAxis];

    if constexpr(kIsKcache)
    {
        // K cache: per-token lookup (all tokens may be on different pages)
        static_for<0, kLoopCount, 1>{}([&](auto k0) {
            const index_t global_token_idx =
                global_seq_offset + thread_coord_start + kLoopStart + kLoopStride * k0.value;
            const index_t page_id =
                ck_tile::min(global_token_idx >> kLog2PageSize, max_page_table_idx);
            physical_pages[k0] = page_idx[page_id];
        });
    }
    else
    {
        // for v offsets
        // for page_size > 1, the V tile crosses pages when page_size is not a multiple of kN0.
        static constexpr bool kVTileCrossesPages =
            (kPageBlockSize > 1) && (kPageBlockSize % kN0 != 0);
        if constexpr(kPageBlockSize == 1 &&
                     kKVMemoryLayout == BlockAttentionKVCacheMemoryLayoutEnum::LINEAR_LAYOUT)
        {
            // page size = 1, per-token page lookup.
            // Here page_idx maps token_idx -> physical_page_id, so global_seq_offset must be
            // the absolute token index within the batch's kv_page_indices slice.
            static_for<0, kLoopCount, 1>{}([&](auto k0) {
                const index_t global_token_idx =
                    global_seq_offset + thread_coord_start + kLoopStart + kLoopStride * k0.value;
                physical_pages[k0] = page_idx[ck_tile::min(global_token_idx, max_page_table_idx)];
            });
        }
        else if constexpr(kVTileCrossesPages)
        {
            // V tile crosses multiple pages (e.g., page_size < kN0), so page_id must be computed
            // per token.
            static_for<0, kLoopCount, 1>{}([&](auto k0) {
                const index_t global_token_idx =
                    global_seq_offset + thread_coord_start + kLoopStart + kLoopStride * k0.value;
                const index_t page_id =
                    ck_tile::min(global_token_idx >> kLog2PageSize, max_page_table_idx);
                physical_pages[k0] = page_idx[page_id];
            });
        }
        else
        {
            // V tile fully contained in one page: lane0 lookup, broadcast to all
            const index_t lane0_start = __builtin_amdgcn_readfirstlane(thread_coord_start);
            const index_t lane0_page_id =
                ck_tile::min((global_seq_offset + lane0_start + kLoopStart) >> kLog2PageSize,
                             max_page_table_idx);
            const index_t shared_physical_page = page_idx[lane0_page_id];

            static_for<0, kLoopCount, 1>{}(
                [&](auto k0) { physical_pages[k0] = shared_physical_page; });
        }
    }
}

// kv_offset_array_transform: Converts logical token indices to physical memory offsets
// for paged KV cache access.
//
// This version uses pre-loaded physical_pages array from load_physical_pages().
// Benefits:
//   - page_idx is read only once (by load_physical_pages)
//   - physical_pages can be prefetched before GEMM to hide memory latency
//   - physical_pages can be reused for descale lookup (KV_BLOCKSCALE)
//
// Template parameters:
//   - kCoordAxis: Which axis of coord_vec contains the thread's token coordinate
//   - kPageBlockSize: Number of tokens per page (must be power of 2)
//   - kLoopStart/kLoopCount/kLoopStride: Loop iteration parameters for static_for
//   - kKVMemoryLayout: VECTORIZED_LAYOUT or LINEAR_LAYOUT
//   - kIsKcache: true for K cache, false for V cache
//   - kN0: Tile size in N dimension (used for page crossing detection)
//   - kVectorSize: Vector size for vectorized layout (e.g., 8 for fp8)
//
// Memory layout for V cache:
//   LINEAR_LAYOUT:     [page, token_in_page, head_dim]
//   VECTORIZED_LAYOUT: [page, token_in_page/kVectorSize, head_dim, kVectorSize]
//
template <typename IndexArrayType,
          typename CoordVecType,
          index_t kCoordAxis,
          index_t kPageBlockSize,
          index_t kLoopStart,
          index_t kLoopCount,
          index_t kLoopStride,
          BlockAttentionKVCacheMemoryLayoutEnum kKVMemoryLayout,
          bool kIsKcache,
          index_t kN0,
          index_t kVectorSize,
          bool kUseGlobalLoad_ = false>
CK_TILE_HOST_DEVICE void kv_offset_array_transform(const IndexArrayType& physical_pages,
                                                   const index_t& stride_token,
                                                   const index_t& stride_page_block,
                                                   const CoordVecType& coord_vec,
                                                   IndexArrayType& kv_offset_vec,
                                                   index_t global_seq_offset = 0)
{
    static constexpr index_t kLog2PageSize = [] {
        index_t shift = 0;
        index_t val   = kPageBlockSize;
        while(val > 1)
        {
            val >>= 1;
            shift++;
        }
        return shift;
    }();

    const index_t& thread_coord_start   = coord_vec[kCoordAxis];
    constexpr index_t kInPageOffsetMask = (1 << kLog2PageSize) - 1;

    // Addressing strategy — four cases controlled by (kPageBlockSize vs kN0, kUseGlobalLoad_):
    //
    //   Case 1: kPageBlockSize >= kN0
    //     SRD is rebased per-tile to the page base (rebase_{k,v}_window in caller).
    //     Page base is absorbed into the SRD's 48-bit base pointer (SGPR-resident).
    //     This function writes within-page offset only.
    //
    //   Case 2: kPageBlockSize <  kN0 && kUseGlobalLoad_
    //     SRD cannot be rebased (multi-page wave). Loads use global_load_lds_*; the full
    //     64-bit address is computed by tile_scatter_gather::load() in
    //     include/ck_tile/core/tensor/tile_scatter_gather.hpp from physical_pages_ +
    //     page_stride_elements_. This function writes within-page offset only.
    //
    //   Case 3: kPageBlockSize <  kN0 && !kUseGlobalLoad_   (kNeedFullOffset == true)
    //     SRD base is the entire KV buffer; the only place to encode page identity
    //     is the voffset itself. This function writes the FULL offset:
    //       page * stride_page_block + within_page
    //     Limited to <2GB total KV bytes by 32-bit voffset hardware width.
    //
    //   Case 4: kPageBlockSize >= kN0 && kUseGlobalLoad_
    //     Not emitted by codegen. Backstop static_assert in
    //     BlockFmhaBatchPrefillPipelineQRKSVSAsync.
    constexpr bool kNeedFullOffset = (kPageBlockSize < kN0) && !kUseGlobalLoad_;

    static_for<0, kLoopCount, 1>{}([&](auto k0) {
        const index_t global_token_idx =
            global_seq_offset + thread_coord_start + kLoopStart + kLoopStride * k0.value;
        const index_t token_idx_in_page = global_token_idx & kInPageOffsetMask;

        // Within-page offset (layout-dependent for V cache with VECTORIZED_LAYOUT)
        const index_t within_page = [&]() {
            if constexpr(!kIsKcache && kKVMemoryLayout ==
                                           BlockAttentionKVCacheMemoryLayoutEnum::VECTORIZED_LAYOUT)
            {
                return (token_idx_in_page / kVectorSize) * (stride_token * kVectorSize) +
                       (token_idx_in_page % kVectorSize);
            }
            else
            {
                return token_idx_in_page * stride_token;
            }
        }();

        // SRD + page_size < kN0: add page base to form complete voffset for buffer_load.
        //
        // 32-bit by hardware: SRD buffer_load voffset is fundamentally 32-bit (CDNA3 MUBUF
        // microcode format), so this branch is only reachable when total KV bytes fit in
        // INT32_MAX. The kUseGlobalLoad_ template path handles the >2GB case via 64-bit
        // global_load_lds_*; widening kv_offset_vec here would not lift the 2GB ceiling
        // because the hardware truncates voffset regardless.
        if constexpr(kNeedFullOffset)
        {
            kv_offset_vec[k0] = physical_pages[k0] * stride_page_block + within_page;
        }
        else
        {
            kv_offset_vec[k0] = within_page;
        }
    });
}

// a variation of qr/ks/vs, where we use async copy to load k (potentially v in the future)
template <typename Problem_,
          typename Policy_ = BlockFmhaBatchPrefillPipelineQRKSVSAsyncDefaultPolicy>
struct BlockFmhaBatchPrefillPipelineQRKSVSAsync
{
    using Problem               = remove_cvref_t<Problem_>;
    using Policy                = remove_cvref_t<Policy_>;
    using QDataType             = remove_cvref_t<typename Problem::QDataType>;
    using KDataType             = remove_cvref_t<typename Problem::KDataType>;
    using VDataType             = remove_cvref_t<typename Problem::VDataType>;
    using SaccDataType          = remove_cvref_t<typename Problem::SaccDataType>;
    using SMPLComputeDataType   = remove_cvref_t<typename Problem::SMPLComputeDataType>;
    using BiasDataType          = remove_cvref_t<typename Problem::BiasDataType>;
    using RandValOutputDataType = remove_cvref_t<typename Problem::RandValOutputDataType>;
    using LSEDataType           = remove_cvref_t<typename Problem::LSEDataType>;
    using PDataType             = remove_cvref_t<typename Problem::PDataType>;
    using OaccDataType          = remove_cvref_t<typename Problem::OaccDataType>;
    using ODataType             = remove_cvref_t<typename Problem::ODataType>;
    using AttentionVariant      = remove_cvref_t<typename Problem::AttentionVariant>;
    using FmhaMask              = remove_cvref_t<typename Problem::FmhaMask>;

    using BlockFmhaShape             = remove_cvref_t<typename Problem::BlockFmhaShape>;
    using VLayout                    = remove_cvref_t<typename BlockFmhaShape::VLayout>;
    static constexpr bool kQLoadOnce = true; // if q_tile load whole block length (hdim) at once
    static_assert(kQLoadOnce == Policy::QLoadOnce);

    static constexpr index_t kBlockSize = Problem::kBlockSize;

    static constexpr index_t kM0            = BlockFmhaShape::kM0;
    static constexpr index_t kN0            = BlockFmhaShape::kN0;
    static constexpr index_t kK0            = BlockFmhaShape::kK0;
    static constexpr index_t kN1            = BlockFmhaShape::kN1;
    static constexpr index_t kK1            = BlockFmhaShape::kK1;
    static constexpr index_t kQKHeaddim     = BlockFmhaShape::kQKHeaddim;
    static constexpr index_t kSubQKHeaddim  = BlockFmhaShape::kSubQKHeaddim;
    static constexpr index_t kPageBlockSize = Problem::kPageBlockSize;
    static constexpr index_t kVectorSize    = Problem::kVectorSize;
    // Single load-mode selector for the whole pipeline. GLOBAL_LOAD_LDS routes K/V
    // tiles through global_load_lds_* (handles >2GB KV cache); BUFFER_LOAD uses SRD
    // buffer_load_*. The enum is named at the trait/Problem level; internally we
    // derive a bool helper to keep `if constexpr` sites narrow. Codegen only emits
    // GLOBAL_LOAD_LDS arms when page_size < kN0; the static_assert is a backstop.
    static constexpr auto kKVLoadMode = Problem::kKVLoadMode;
    static constexpr bool kUseGlobalLoad =
        (kKVLoadMode == BlockAttentionKVCacheLoadModeEnum::GLOBAL_LOAD_LDS);
    static_assert(!kUseGlobalLoad || (kPageBlockSize < kN0),
                  "GLOBAL_LOAD_LDS load mode is only valid when kPageBlockSize < kN0; "
                  "codegen should not emit this instantiation otherwise.");
    static constexpr auto I0 = number<0>{};
    static constexpr auto I1 = number<1>{};
    static constexpr auto I2 = number<2>{};
    static constexpr auto I3 = number<3>{};

    static_assert(kSubQKHeaddim <= 256, "hdim bigger than 256 is not suitable for this pipeline!");
    static constexpr bool kIsGroupMode = Problem::kIsGroupMode;
    // TODO: seq_q always support padding, hdim_q/v support multiple of vector(like 8x)
    //       only need special care about seq_k padding (oob need set -INF of p instead of zero)
    static_assert(Problem::kPadSeqLenQ == true && Problem::kPadHeadDimQ == true &&
                  Problem::kPadHeadDimV == true);
    static constexpr bool kPadSeqLenQ       = true;
    static constexpr bool kPadSeqLenK       = Problem::kPadSeqLenK;
    static constexpr bool kPadHeadDimQ      = true; // support multiple of vector(like 8x)
    static constexpr bool kPadHeadDimV      = true; // support multiple of vector(like 8x)
    static constexpr bool kHasLogitsSoftCap = Problem::kHasLogitsSoftCap;
    static constexpr auto BiasEnum          = Problem::BiasEnum;
    static constexpr bool kStoreLSE         = Problem::kStoreLSE;
    static constexpr bool kHasDropout       = Problem::kHasDropout;
    static constexpr auto kKVMemoryLayout   = Problem::kKVMemoryLayout;
    static constexpr auto QScaleEnum        = Problem::QScaleEnum;
    static constexpr bool kHasSink          = Problem::kHasSink;

    // For KV_BLOCKSCALE: shift value for exp2(x + shift) to scale P to [0, 2^shift]
    // This avoids explicit P *= scale_p and v_descale /= scale_p operations
    static constexpr float OCP_FP8_SHIFT  = 8.0f;
    static constexpr float FNUZ_FP8_SHIFT = 7.0f;

    static_assert((CK_TILE_FMHA_FWD_FAST_EXP2 &&
                   (kHasLogitsSoftCap && Problem::BiasEnum == BlockAttentionBiasEnum::NO_BIAS ||
                    !kHasLogitsSoftCap)) ||
                  (!CK_TILE_FMHA_FWD_FAST_EXP2 && !kHasLogitsSoftCap));

    // last dimension vector length used to create tensor view(and decide buffer_load vector length)
    // ... together with tensor distribution. tensor dist should able to overwrite this
    static constexpr index_t kAlignmentQ = Policy::template GetAlignmentQ<Problem>();
    static constexpr index_t kAlignmentK = Policy::template GetAlignmentK<Problem>();
    static constexpr index_t kAlignmentV = []() {
        if constexpr(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>)
            return Policy::template GetAlignmentV<Problem>();
        else
            return kPadSeqLenK ? 1 : Policy::template GetAlignmentV<Problem>();
    }();
    static constexpr index_t kAlignmentO = Policy::template GetAlignmentO<Problem>();
    static constexpr index_t kAlignmentBias =
        kPadSeqLenK ? 1 : Policy::template GetAlignmentBias<Problem>();
    static constexpr index_t kAlignmentRandVal =
        kPadSeqLenK ? 1 : Policy::template GetAlignmentRandVal<Problem>();

#if CK_TILE_FMHA_FWD_FAST_EXP2
    static constexpr auto R_LOG2E = 1.0 / log2e_v<SaccDataType>;
    static constexpr auto LOG2E   = log2e_v<SaccDataType>;
#endif

    static constexpr index_t kBlockPerCu = []() {
        if constexpr(Problem::kBlockPerCu != -1)
            return Problem::kBlockPerCu;
        else
        {
            // minimize occupancy
            if constexpr(BiasEnum != BlockAttentionBiasEnum::NO_BIAS && kHasDropout)
            {
                return 1;
            }

            if constexpr(kQKHeaddim <= 32)
            {
                if constexpr(kPadSeqLenK && BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS &&
                             FmhaMask::IsMasking)
                    return 1;
                else
                    return 2;
            }
            else if constexpr(kQKHeaddim <= 64)
            {
                if constexpr(kPadSeqLenK && BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
                    return 2;
                else
                    return 3;
            }
            else if constexpr(kQKHeaddim <= 128)
            {
                if constexpr(kPadSeqLenK && BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
                    return 1;
                else
                    return 2;
            }
            else if constexpr(kQKHeaddim <= 192)
            {
                if constexpr(kPadSeqLenK && BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
                    return 1;
                else
                    return 2;
            }
            else if constexpr(kQKHeaddim <= 256)
            {
                return 1;
            }
            else
            {
                return 1;
            };
        }
    }();

    static constexpr const char* name = "qr_async";

    using DropoutType = std::conditional_t<kHasDropout, BlockDropout, NullBlockDropout>;

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename BiasDramBlockWindowTmp,
              typename RandValDramBlockWindowTmp,
              typename LSEDramBlockWindowTmp,
              typename QElementFunction,
              typename KElementFunction,
              typename VElementFunction,
              typename BiasElementFunction,
              typename LSEElementFunction,
              typename SAccElementFunction,
              typename PComputeElementFunction,
              typename OAccElementFunction,
              typename PositionEncoding,
              typename AttentionVariantParams,
              typename BlockIndices>
    CK_TILE_HOST_DEVICE auto
    operator()(const QDramBlockWindowTmp& q_dram_block_window_tmp, // M0*K0 tile
               const QElementFunction& q_element_func,
               const KDramBlockWindowTmp& k_dram_block_window_tmp, // N0*K0 tile
               const KElementFunction& /*k_element_func*/,
               const VDramBlockWindowTmp& v_dram_block_window_tmp, // N1*K1 tile
               const VElementFunction& v_element_func,
               const BiasDramBlockWindowTmp& bias_dram_block_window_tmp, // M0*N0 tile
               const BiasElementFunction& bias_element_func,
               RandValDramBlockWindowTmp& randval_dram_block_window_tmp,
               LSEDramBlockWindowTmp& lse_dram_window_tmp, // M0*1 tile
               const LSEElementFunction& lse_element_func,
               const SAccElementFunction& s_acc_element_func,
               const PComputeElementFunction& p_compute_element_func,
               const OAccElementFunction& o_acc_element_func,
               FmhaMask mask,
               PositionEncoding position_encoding,
               float scale_s,
               const AttentionVariant& variant,
               const AttentionVariantParams& variant_params,
               const BlockIndices& block_indices,
               void* smem_ptr,
               const index_t* page_idx,
               const index_t stride_k,
               const index_t stride_v,
               const index_t page_stride_k,
               const index_t page_stride_v,
               DropoutType& dropout,
               const float sink_v,
               const index_t max_page_table_idx,
               // KV_BLOCKSCALE parameters (only used when QScaleEnum == KV_BLOCKSCALE)
               const float* k_descale_ptr             = nullptr,
               const float* v_descale_ptr             = nullptr,
               index_t nblock_stride_kv_block_descale = 0,
               index_t nhead_stride_kv_block_descale  = 0) const
    {
        // KV_BLOCKSCALE requires page_block_size >= kN0 to ensure
        // all tokens in a main loop iteration belong to the same page
        if constexpr(QScaleEnum == BlockAttentionQuantScaleEnum::KV_BLOCKSCALE)
        {
            static_assert(kPageBlockSize >= kN0, "KV_BLOCKSCALE requires kPageBlockSize >= kN0");
        }

        static_assert(
            std::is_same_v<QDataType, remove_cvref_t<typename QDramBlockWindowTmp::DataType>> &&
                std::is_same_v<KDataType, remove_cvref_t<typename KDramBlockWindowTmp::DataType>> &&
                std::is_same_v<VDataType, remove_cvref_t<typename VDramBlockWindowTmp::DataType>>,
            "wrong!");

        static_assert(kM0 == QDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kN0 == KDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kK0 == KDramBlockWindowTmp{}.get_window_lengths()[number<1>{}] &&
                          kN1 == VDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kK1 == VDramBlockWindowTmp{}.get_window_lengths()[number<1>{}] &&
                          kM0 == BiasDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kN0 == BiasDramBlockWindowTmp{}.get_window_lengths()[number<1>{}],
                      "wrong!");

        constexpr auto LdsSeq = Policy::template GetLdsBufferSequence<Problem>();

        // K tile in LDS
        auto k_lds_ptr   = reinterpret_cast<KDataType*>(smem_ptr);
        auto k_lds_store = generate_tuple(
            [&](auto i_buf) {
                return make_tile_window(
                    make_tensor_view<address_space_enum::lds>(
                        k_lds_ptr, Policy::template MakeKLdsStoreBlockDescriptor<Problem>(i_buf)),
                    Policy::template MakeKLdsStoreBlockDescriptor<Problem>(i_buf).get_lengths(),
                    {0, 0, 0});
            },
            number<Policy::NumKVLdsBuffers>{});

        auto k_lds_Load_view = make_tensor_view<address_space_enum::lds>(
            k_lds_ptr, Policy::template MakeKLdsLoadBlockDescriptor<Problem>());

        auto k_lds_load =
            make_tile_window(k_lds_Load_view,
                             Policy::template MakeKLdsLoadBlockDescriptor<Problem>().get_lengths(),
                             {0, 0});

        // V tile in LDS
        auto v_lds = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<VDataType*>(smem_ptr),
            Policy::template MakeVLdsBlockDescriptor<Problem>());
        auto v_lds_window = make_tile_window(
            v_lds, Policy::template MakeVLdsBlockDescriptor<Problem>().get_lengths(), {0, 0});

        // Block GEMM
        constexpr auto gemm_0 = Policy::template GetQKBlockGemm<Problem>();
        constexpr auto gemm_1 = Policy::template GetKVBlockGemm<Problem>();

        auto q_dram_window = make_tile_window(q_dram_block_window_tmp.get_bottom_tensor_view(),
                                              q_dram_block_window_tmp.get_window_lengths(),
                                              q_dram_block_window_tmp.get_window_origin(),
                                              Policy::template MakeQRegTileDistribution<Problem>());
        q_dram_window.init_raw();

        // TODO: we use async Copy for K, which is inline asm
        // a side effect is we have to use inline asm for q as well
        auto q = decltype(load_tile(q_dram_window)){};
        // TODO: start from rocm-6.2, compiler will have problem if manually set clear of q.
        // however, q would be cleared in the constructor of static distributed tensor
        // set_tile(q, number<0>{}); // use per-dword clear to avoid scratch
        load_tile_raw(q, q_dram_window);
        __builtin_amdgcn_sched_barrier(0);

        using SaccBlockTileType = decltype(gemm_0.MakeCBlockTile());
        auto s_acc              = SaccBlockTileType{};

        // reduction function for softmax
        const auto f_max = [](auto e0, auto e1) { return max(e0, e1); };
        const auto f_sum = [](auto e0, auto e1) { return e0 + e1; };

        // infer Sacc, S, P, M, L, Oacc type
        using SBlockTileType = decltype(cast_tile<SMPLComputeDataType>(s_acc));

        using MLBlockTileType = decltype(block_tile_reduce<SMPLComputeDataType>(
            SBlockTileType{}, sequence<1>{}, f_max, SMPLComputeDataType{0}));

        using OaccBlockTileType = decltype(gemm_1.MakeCBlockTile());

        // init Oacc, M, L
        auto o_acc = OaccBlockTileType{};
        auto m     = MLBlockTileType{};
        auto l     = MLBlockTileType{};

        clear_tile(o_acc);
        if(__builtin_isinf_sign(sink_v) >= 0)
        {
#if CK_TILE_FMHA_FWD_FAST_EXP2
            if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                         BiasEnum == BlockAttentionBiasEnum::ALIBI)
                set_tile(m, sink_v * LOG2E * scale_s);
            else
                set_tile(m, sink_v * LOG2E);
#else
            set_tile(m, sink_v);
#endif
            set_tile(l, SMPLComputeDataType{1.0f});
        }
        else
        {
            set_tile(m, -numeric<SMPLComputeDataType>::infinity());
            clear_tile(l);
        }

        __builtin_amdgcn_sched_barrier(0);
        const auto q_origin          = q_dram_window.get_window_origin();
        const auto tile_range_result = [&mask, &q_origin]() {
            if constexpr(kHasSink)
                return mask.GetSinkTileRangeAlongX(
                    q_origin.at(number<0>{}), number<kM0>{}, number<kN0>{});
            else
            {
                auto [start, end] =
                    mask.GetTileRangeAlongX(q_origin.at(number<0>{}), number<kM0>{}, number<kN0>{});
                return ck_tile::make_tuple(0, start, end);
            }
        }();
        const auto sink_seq_end   = tile_range_result.get(ck_tile::number<0>{});
        const auto seqlen_k_start = tile_range_result.get(ck_tile::number<1>{});
        const auto seqlen_k_end   = tile_range_result.get(ck_tile::number<2>{});
        const auto num_sink_loop  = integer_divide_ceil(sink_seq_end, kN0);
        const auto kv_load_start  = (sink_seq_end == 0 && seqlen_k_start > 0) ? seqlen_k_start : 0;
        const auto num_total_loop =
            integer_divide_ceil(seqlen_k_end - seqlen_k_start, kN0) + num_sink_loop;

        // check early exit if no work to do
        if constexpr(FmhaMask::IsMasking || kPadSeqLenK)
        {
            if(num_total_loop <= 0)
            {
                if constexpr(kStoreLSE)
                {
                    auto lse =
                        make_static_distributed_tensor<LSEDataType>(m.get_tile_distribution());
                    set_tile(lse, SMPLComputeDataType{sink_v * scale_s});
                    store_tile(lse_dram_window_tmp, tile_elementwise_in(lse_element_func, lse));
                }
                buffer_load_fence(0); // rocm-6.1, if whole tile is masked out, need to fence(0)
                                      // otherwise will have compute error(maybe compiler bug?)

                // Note: here occ are all cleard, return it
                return o_acc;
            }
            __builtin_amdgcn_sched_barrier(0); // make sure sched_barrier(0) for this check
        }

        auto k_dram_block_window =
            make_tile_window(k_dram_block_window_tmp.get_bottom_tensor_view(),
                             k_dram_block_window_tmp.get_window_lengths(),
                             {kv_load_start, 0});

        auto k_dist               = Policy::template MakeKDramTileDistribution<Problem>();
        auto k_coord              = k_dist.calculate_index();
        using KDstrEncode         = typename decltype(k_dist)::DstrEncode;
        constexpr index_t NRepeat = KDstrEncode::hs_lengthss_[I0][I0];
        // kPageBlockSize >= kN0: within-page offset only (SRD rebased per page via rebase_k_window)
        // kPageBlockSize <  kN0: global offset, must fit int32
        statically_indexed_array<index_t, NRepeat> k_offsets;
        index_t current_seq_k = kv_load_start;

        // Load physical pages first, then compute offsets.
        // k_physical_pages can be reused for descale lookup later.
        statically_indexed_array<index_t, NRepeat> k_physical_pages{};
        load_physical_pages<statically_indexed_array<index_t, NRepeat>,
                            decltype(k_coord),
                            0,
                            kPageBlockSize,
                            0,
                            NRepeat,
                            kN0 / NRepeat,
                            kKVMemoryLayout,
                            true,
                            kN0>(
            page_idx, k_coord, current_seq_k, k_physical_pages, max_page_table_idx);

        kv_offset_array_transform<statically_indexed_array<index_t, NRepeat>,
                                  decltype(k_coord),
                                  0,
                                  kPageBlockSize,
                                  0,
                                  NRepeat,
                                  kN0 / NRepeat,
                                  kKVMemoryLayout,
                                  true,
                                  kN0,
                                  kVectorSize,
                                  kUseGlobalLoad>(
            k_physical_pages, stride_k, page_stride_k, k_coord, k_offsets, current_seq_k);

        auto k_dram_window = make_tile_scatter_gather(k_dram_block_window.get_bottom_tensor_view(),
                                                      k_dram_block_window.get_window_lengths(),
                                                      k_dram_block_window.get_window_origin(),
                                                      k_dist,
                                                      k_offsets,
                                                      bool_constant<kUseGlobalLoad>{},
                                                      page_stride_k);
        if constexpr(kUseGlobalLoad)
        {
            k_dram_window.update_physical_pages(k_physical_pages);
        }
        k_dram_window.init_raw();

        // SRD rebasing for K: only for page_size >= kN0 (all threads on same page).
        // For page_size < kN0: either flat loads (kUseGlobalLoad) or full offsets handle
        // addressing.
        auto rebase_k_window = [&](auto& window, index_t physical_page) {
            if constexpr(kPageBlockSize >= kN0)
            {
                // readfirstlane: make physical_page provably wave-uniform so the
                // resulting SRD lands in SGPRs (required by buffer load instructions).
                physical_page        = __builtin_amdgcn_readfirstlane(physical_page);
                const auto* base_ptr = k_dram_block_window.get_bottom_tensor_view().buf_.p_data_;
                const auto* page_ptr =
                    base_ptr + static_cast<long_index_t>(physical_page) * page_stride_k;
                window.set_bottom_tensor_view_data_ptr(page_ptr);
                // Limit SRD num_records to one page worth of elements.
                // Without this, the SRD claims validity for [page_ptr, page_ptr +
                // full_buffer_size), which extends far beyond the allocated buffer when rebased to
                // high pages. On gfx950, the hardware may validate the full SRD range against page
                // table permissions, causing faults on freed/protected memory beyond the buffer.
                window.set_bottom_tensor_view_buffer_size(page_stride_k);
                window.init_raw();
            }
        };

        // SRD rebasing for V: only for page_size >= kN0 (all threads on same page).
        // For page_size < kN0: either flat loads (kUseGlobalLoad) or full offsets handle
        // addressing.
        auto rebase_v_window = [&](auto& window, index_t physical_page) {
            if constexpr(kPageBlockSize >= kN0)
            {
                // readfirstlane: make physical_page provably wave-uniform so the
                // resulting SRD lands in SGPRs (required by buffer load instructions).
                physical_page = __builtin_amdgcn_readfirstlane(physical_page);
                const auto* base_ptr =
                    v_dram_block_window_tmp.get_bottom_tensor_view().buf_.p_data_;
                const auto* page_ptr =
                    base_ptr + static_cast<long_index_t>(physical_page) * page_stride_v;
                window.set_bottom_tensor_view_data_ptr(page_ptr);
                window.set_bottom_tensor_view_buffer_size(page_stride_v);
                window.init_raw();
            }
        };

        // Initial K SRD rebase (no-op for page_size < kN0, uses flat loads instead)
        rebase_k_window(k_dram_window, k_physical_pages[number<0>{}]);

        constexpr auto k_oob_ck = bool_constant<true>{};
        constexpr auto k_pre_np = [&]() {
            if constexpr(kPadSeqLenK &&
                         (BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                          (BiasEnum != BlockAttentionBiasEnum::NO_BIAS && kHasDropout)))
                return bool_constant<true>{};
            else
                return bool_constant<false>{};
        }();

        const auto bias_origin = bias_dram_block_window_tmp.get_window_origin();
        auto bias_dram_window =
            make_tile_window(bias_dram_block_window_tmp.get_bottom_tensor_view(),
                             bias_dram_block_window_tmp.get_window_lengths(),
                             {bias_origin.at(number<0>{}), kv_load_start}, // M/N
                             Policy::template MakeBiasDramTileDistribution<decltype(gemm_0)>());

        auto randval_dram_window = dropout.template MakeRandvalDramWindow<decltype(gemm_0)>(
            randval_dram_block_window_tmp, kv_load_start);

        auto v_dist       = Policy::template MakeVDramTileDistribution<Problem>();
        auto v_coord      = v_dist.calculate_index();
        using VDstrEncode = typename decltype(v_dist)::DstrEncode;

        // V tensor K-dimension decomposition for page index computation
        // ============================================================
        // The K dimension (seqlen_k) in V distribution is decomposed into multiple sub-dimensions.
        // This decomposition determines how threads iterate over the K dimension and how page
        // indices are computed for paged KV cache.
        //
        // The decomposition pattern differs by memory layout:
        //
        // VECTORIZED_LAYOUT (ColumnMajor, custom distribution):
        //   3D decomposition: K = K2 × K0 × K1
        //   - K2 (V_KIterOuter): Outer iteration count
        //   - K0 (V_KLanes):     Lanes for K dimension (matches GEMM kABKLane)
        //   - K1 (V_KIterInner): Vector load size (matches GEMM kKPerThread)
        //   - hs_lengthss_[I1] = {K2, K0, K1}, size = 3 (or {K0, K1} size = 2 if no outer iter)
        //
        // LINEAR_LAYOUT ColumnMajor (base class distribution):
        //   2D decomposition: K = K0 × K1
        //   - K0: Lanes for K dimension (may not match GEMM kABKLane)
        //   - K1: Vector load size
        //   - hs_lengthss_[I1] = {K0, K1}, size = 2
        //
        // LINEAR_LAYOUT RowMajor (base class distribution):
        //   4D decomposition: K = K0 × K1 × K2 × K3 (uses shuffle_tile for GEMM alignment)
        //   3D decomposition: K = K0 × K1 × K2 (fallback case)
        //   - Page lookup uses Y-space's last dimension only (inner iteration)
        //
        // V_PageIdxRepeat = total number of page lookups per thread = V_KIterOuter × V_KIterInner
        constexpr index_t V_KIterInner = VDstrEncode::hs_lengthss_[I1].back();

        // Compute V_KIterOuter and V_KLanes based on memory layout and K decomposition
        constexpr index_t V_KIterOuter = [] {
            if constexpr(kKVMemoryLayout ==
                         BlockAttentionKVCacheMemoryLayoutEnum::VECTORIZED_LAYOUT)
            {
                // VECTORIZED_LAYOUT: 3D decomposition {K2, K0, K1} when outer iteration is needed
                if constexpr(VDstrEncode::hs_lengthss_[I1].size() == 3)
                    return static_cast<index_t>(VDstrEncode::hs_lengthss_[I1][I0]);
                else
                    return index_t{1};
            }
            else
            {
                // LINEAR_LAYOUT: No outer iteration for page lookup
                // RowMajor uses shuffle_tile, ColumnMajor has simple 2D decomposition
                // Both cases use single-dimension Y-space page lookup
                return index_t{1};
            }
        }();

        constexpr index_t V_KLanes = [] {
            if constexpr(kKVMemoryLayout ==
                         BlockAttentionKVCacheMemoryLayoutEnum::VECTORIZED_LAYOUT)
            {
                // VECTORIZED_LAYOUT: K0 is the lanes dimension
                if constexpr(V_KIterOuter > 1)
                    return static_cast<index_t>(VDstrEncode::hs_lengthss_[I1][I1]);
                else
                    return static_cast<index_t>(VDstrEncode::hs_lengthss_[I1][I0]);
            }
            else
            {
                // LINEAR_LAYOUT: First dimension is K0 (lanes)
                return static_cast<index_t>(VDstrEncode::hs_lengthss_[I1][I0]);
            }
        }();

        // This affects page offset computation - need to track offsets for each (k2, k1)
        // combination
        constexpr index_t V_PageIdxRepeat = V_KIterInner * V_KIterOuter;

        // VPageIndexYDims: Y-space dimension indices that participate in page index computation
        // ================================================================================
        // In tile_scatter_gather, the gather index is computed from Y-space coordinates.
        // This sequence specifies which Y dimensions should be linearized to form the page lookup
        // index.
        //
        // VECTORIZED_LAYOUT with outer iteration: sequence<Y_K1, Y_K2>
        //   - Both K1 and K2 are in Y-space (thread iteration dimensions)
        //   - gather_index = y_k1 + y_k2 * len(Y_K1)  (linearized 2D -> 1D)
        //
        // VECTORIZED_LAYOUT without outer iteration / LINEAR_LAYOUT: sequence<Y_K1>
        //   - Only the innermost K dimension is used for page lookup (single dimension)
        //
        constexpr auto VPageIndexYDims = []() {
            // K1Minor is always the last element index in hs_lengthss_[I1]
            constexpr index_t K1Minor = VDstrEncode::hs_lengthss_[I1].size() - 1;
            constexpr index_t Y_K1    = VDstrEncode::detail::rhs_major_minor_to_ys_[2][K1Minor];

            if constexpr(kKVMemoryLayout ==
                             BlockAttentionKVCacheMemoryLayoutEnum::VECTORIZED_LAYOUT &&
                         V_KIterOuter > 1)
            {
                // VECTORIZED_LAYOUT with outer iteration: need 2D page lookup
                constexpr index_t Y_K2 = VDstrEncode::detail::rhs_major_minor_to_ys_[2][I0];
                return sequence<Y_K1, Y_K2>{};
            }
            else
            {
                // LINEAR_LAYOUT or VECTORIZED_LAYOUT without outer iteration: 1D page lookup
                return sequence<Y_K1>{};
            }
        }();

        static_assert(decltype(VPageIndexYDims)::at(0) < VDstrEncode::NDimY,
                      "V page-index Y dim must be valid");

        // kPageBlockSize >= kN0: within-page offset only (SRD rebased per page via rebase_v_window)
        // kPageBlockSize <  kN0: global offset, must fit int32
        statically_indexed_array<index_t, V_PageIdxRepeat> v_offsets;
        // V physical pages array for use with kv_offset_array_transform
        // For V_KIterOuter > 1, we need V_PageIdxRepeat elements; otherwise V_KIterInner
        statically_indexed_array<index_t, V_PageIdxRepeat> v_physical_pages{};

        // Prefetch V physical pages - can be called early to hide buffer load latency
        auto prefetch_v_physical_pages = [&](auto k_loop_start) {
            constexpr index_t kLoopStart = decltype(k_loop_start)::value;
            if constexpr(V_KIterOuter > 1)
            {
                static_for<0, V_KIterOuter, 1>{}([&](auto k2) {
                    // Load physical pages for this k2 slice into the appropriate portion of array
                    statically_indexed_array<index_t, V_KIterInner> v_physical_pages_k2{};
                    load_physical_pages<statically_indexed_array<index_t, V_KIterInner>,
                                        decltype(v_coord),
                                        I1,
                                        kPageBlockSize,
                                        kLoopStart + k2.value * V_KLanes * V_KIterInner,
                                        V_KIterInner,
                                        1,
                                        kKVMemoryLayout,
                                        false,
                                        kN0>(
                        page_idx, v_coord, current_seq_k, v_physical_pages_k2, max_page_table_idx);

                    // Copy to merged array
                    static_for<0, V_KIterInner, 1>{}([&](auto k1) {
                        constexpr auto idx    = number<k1.value + k2.value * V_KIterInner>{};
                        v_physical_pages[idx] = v_physical_pages_k2[k1];
                    });
                });
            }
            else
            {
                load_physical_pages<statically_indexed_array<index_t, V_KIterInner>,
                                    decltype(v_coord),
                                    I1,
                                    kPageBlockSize,
                                    kLoopStart,
                                    V_KIterInner,
                                    1,
                                    kKVMemoryLayout,
                                    false,
                                    kN0>(
                    page_idx, v_coord, current_seq_k, v_physical_pages, max_page_table_idx);
            }
        };

        // Update V offsets using pre-loaded physical pages
        auto update_v_offsets = [&](auto k_loop_start) {
            constexpr index_t kLoopStart = decltype(k_loop_start)::value;
            // For 3D K decomposition (K2, K0, K1), compute offsets for each K2 slice
            // The global K offset for (k2, k1) is: kLoopStart + k2 * (K0 * K1) + k1
            // We iterate K2 outer, K1 inner, and merge into 1D v_offsets array
            if constexpr(V_KIterOuter > 1)
            {
                static_for<0, V_KIterOuter, 1>{}([&](auto k2) {
                    statically_indexed_array<index_t, V_KIterInner> v_offsets_k2;
                    // Extract physical pages for this k2 slice
                    statically_indexed_array<index_t, V_KIterInner> v_physical_pages_k2;
                    static_for<0, V_KIterInner, 1>{}([&](auto k1) {
                        constexpr auto idx      = number<k1.value + k2.value * V_KIterInner>{};
                        v_physical_pages_k2[k1] = v_physical_pages[idx];
                    });

                    kv_offset_array_transform<statically_indexed_array<index_t, V_KIterInner>,
                                              decltype(v_coord),
                                              I1,
                                              kPageBlockSize,
                                              kLoopStart + k2.value * V_KLanes * V_KIterInner,
                                              V_KIterInner,
                                              1,
                                              kKVMemoryLayout,
                                              false,
                                              kN0,
                                              kVectorSize,
                                              kUseGlobalLoad>(v_physical_pages_k2,
                                                              stride_v,
                                                              page_stride_v,
                                                              v_coord,
                                                              v_offsets_k2,
                                                              current_seq_k);

                    static_for<0, V_KIterInner, 1>{}([&](auto k1) {
                        constexpr auto idx = number<k1.value + k2.value * V_KIterInner>{};
                        v_offsets[idx]     = v_offsets_k2[k1];
                    });
                });
            }
            else
            {
                kv_offset_array_transform<statically_indexed_array<index_t, V_KIterInner>,
                                          decltype(v_coord),
                                          I1,
                                          kPageBlockSize,
                                          kLoopStart,
                                          V_KIterInner,
                                          1,
                                          kKVMemoryLayout,
                                          false,
                                          kN0,
                                          kVectorSize,
                                          kUseGlobalLoad>(
                    v_physical_pages, stride_v, page_stride_v, v_coord, v_offsets, current_seq_k);
            }

            // v_offsets semantics — see the four-case addressing-strategy block above
            // kNeedFullOffset in kv_offset_array_transform. Three cases reach this lambda:
            //   Case 1 (kPageBlockSize >= kN0):           within-page offset; page base in SRD.
            //   Case 2 (page_size < kN0, kUseGlobalLoad): within-page offset; page base computed
            //                                             by tile_scatter_gather::load() from
            //                                             physical_pages_.
            //   Case 3 (page_size < kN0, !kUseGlobalLoad, == kNeedFullOffset):
            //                                             FULL offset (page * stride + within),
            //                                             carried in the 32-bit voffset (<2GB cap).
        };

        // Prefetch V physical pages early to hide buffer load latency
        prefetch_v_physical_pages(number<0>{});
        update_v_offsets(number<0>{});
        auto v_dram_window =
            make_tile_scatter_gather(v_dram_block_window_tmp.get_bottom_tensor_view(),
                                     v_dram_block_window_tmp.get_window_lengths(),
                                     {0, kv_load_start}, // TODO: hdim split?
                                     v_dist,
                                     v_offsets,
                                     number<1>{}, // HsGatherDim
                                     number<1>{}, // NumCoord
                                     VPageIndexYDims,
                                     bool_constant<kUseGlobalLoad>{},
                                     page_stride_v);
        if constexpr(kUseGlobalLoad)
        {
            v_dram_window.update_physical_pages(v_physical_pages);
        }

        // Initial V SRD rebase. Single source of truth: rebase_v_window's own
        // `if constexpr(kPageBlockSize >= kN0)` makes this a no-op for case 2/3.
        // Do not re-add an outer guard here — it would duplicate the inner check
        // and drift if the lambda's gating condition ever changes.
        rebase_v_window(v_dram_window, v_physical_pages[number<0>{}]);

        // Save the *current* tile's V physical pages into v_dram_window before
        // prefetch_v_physical_pages overwrites the v_physical_pages buffer with the
        // *next* tile's pages. Case-2 only (kUseGlobalLoad); case-1/3 don't read
        // physical_pages_ from the window. Encapsulating the save+prefetch pair
        // here makes the ordering invariant unmissable when a fourth prefetch site
        // is added later.
        auto save_and_prefetch_v_pages = [&](auto k_loop_start) {
            if constexpr(kUseGlobalLoad)
                v_dram_window.update_physical_pages(v_physical_pages);
            prefetch_v_physical_pages(k_loop_start);
        };

        // prefetch K tile
        async_load_tile_raw(
            k_lds_store(LdsSeq.at(number<0>{})), k_dram_window, number<-1>{}, k_oob_ck, k_pre_np);
        move_tile_window(k_dram_window, {0, kK0});
        __builtin_amdgcn_sched_barrier(0);

        buffer_load_fence(k_dram_window.get_num_of_access(), q.get_thread_buffer());
        (void)q_element_func; // ??? rocm-6.x if use q element func will have scratch on hdim=64/32
        // auto q_tile = q;      // tile_elementwise_in(q_element_func, q);

        index_t i_total_loops      = 0;
        constexpr index_t k0_loops = kQKHeaddim / kK0;
        constexpr index_t k1_loops = kN0 / kK1;

        // Clear logical sequence padding before any V value reaches LDS/PV.
        // v_tile is [N1, K1]; its local dimension 1 is the logical KV token.
        // Use the pre-clamp sequence coordinate, never the gathered physical page.
        auto zero_logical_invalid_v = [&](auto& v_tile, auto slice_start) {
            const index_t logical_begin = current_seq_k + decltype(slice_start)::value;
            if(mask.IsOutOfBound(0, logical_begin + kK1 - 1))
            {
                set_tile_if(v_tile, VDataType{0}, [&](auto tile_idx) {
                    return mask.IsOutOfBound(0, logical_begin + tile_idx.at(number<1>{}));
                });
            }
        };

        static_assert(1 <= k0_loops);
        static_assert(1 <= k1_loops);
        // main loop
        do
        {
            // KV_BLOCKSCALE: load per-page K/V descale factors
            // Uses k_physical_pages[0] from load_physical_pages to avoid redundant page_idx reads.
            // Assumes kPageBlockSize >= kN0, so all tokens in one main loop iteration belong to
            // the same page (single scale pair).
            //
            // TODO: Cross-page KV_BLOCKSCALE support
            // Currently only supports kPageBlockSize >= kN0 (all tokens in tile on same page).
            // To support smaller page sizes (cross-page tiles), need:
            //
            // 1. K descale: Load per-token k_descale_vec[NRepeat] based on k_physical_pages[k0]
            //    - After GEMM0 (S = Q × K^T), apply column-wise scaling: S[:,j] *= k_descale[j]
            //    - Requires modifying s_acc_element_func to accept column index
            //
            // 2. V descale: Load per-token v_descale_vec[V_PageIdxRepeat] based on
            // v_physical_pages[k0]
            //    - Before GEMM1 (O = P × V), apply row-wise scaling to P: P[i,j] *= v_descale[j]
            //    - Or pre-scale V in LDS (more complex)
            //
            // 3. K and V may be on different pages for the same token index, so need separate
            // lookups
            //
            [[maybe_unused]] float k_descale = 1.0f;
            [[maybe_unused]] float v_descale = 1.0f;
            if constexpr(QScaleEnum == BlockAttentionQuantScaleEnum::KV_BLOCKSCALE)
            {
                const index_t scale_offset =
                    k_physical_pages[number<0>{}] * nblock_stride_kv_block_descale +
                    block_indices.kv_head_idx * nhead_stride_kv_block_descale;
                k_descale = k_descale_ptr[scale_offset];
                v_descale = v_descale_ptr[scale_offset];
            }

            // Prefetch V physical pages early - overlaps with GEMM0 computation
            save_and_prefetch_v_pages(number<kK1>{});

            // STAGE 1, QK gemm
            clear_tile(s_acc); // initialize C
            if constexpr(k0_loops > 1)
            {
                static_for<0, k0_loops - 1, 1>{}([&](auto i_k0) {
                    async_load_tile_raw(k_lds_store(number<LdsSeq.at(number<i_k0 + 1>{})>{}),
                                        k_dram_window,
                                        number<-1>{},
                                        k_oob_ck,
                                        k_pre_np);
                    if constexpr(i_k0 < k0_loops - 1)
                        move_tile_window(k_dram_window, {0, kK0});

                    async_load_fence(k_dram_window.get_num_of_access());
                    __builtin_amdgcn_s_barrier();
                    __builtin_amdgcn_sched_barrier(0);
                    gemm_0(s_acc,
                           get_slice_tile(
                               q, sequence<0, i_k0 * kK0>{}, sequence<kM0, (i_k0 + 1) * kK0>{}),
                           get_slice_tile(k_lds_load,
                                          sequence<(LdsSeq.at(number<i_k0>{})) * kN0, 0>{},
                                          sequence<(LdsSeq.at(number<i_k0>{}) + 1) * kN0, kK0>{}));
                });
            }

            // TODO: this to fix a bug when loop smaller than 2,
            // the following fence/barrier will be scheduled inside 1st loop
            if constexpr(k0_loops <= 2)
                __builtin_amdgcn_sched_barrier(0);

            async_load_fence();
            __builtin_amdgcn_s_barrier();

            __builtin_amdgcn_sched_barrier(0);
            { // tail
                gemm_0(
                    s_acc,
                    get_slice_tile(
                        q, sequence<0, (k0_loops - 1) * kK0>{}, sequence<kM0, k0_loops * kK0>{}),
                    get_slice_tile(k_lds_load,
                                   sequence<(LdsSeq.at(number<k0_loops - 1>{})) * kN0, 0>{},
                                   sequence<(LdsSeq.at(number<k0_loops - 1>{}) + 1) * kN0, kK0>{}));
            }
            __builtin_amdgcn_sched_barrier(1);

            auto v_buf = load_tile(v_dram_window, number<-1>{}, bool_constant<false>{});
            zero_logical_invalid_v(v_buf, number<0>{});
            // V physical pages already prefetched before GEMM0
            update_v_offsets(number<kK1>{});
            v_dram_window.update_page_idx(v_offsets);
            rebase_v_window(v_dram_window, v_physical_pages[number<0>{}]);

            // KV_BLOCKSCALE: apply k_descale to s_acc (dequantize QK result)
            if constexpr(QScaleEnum == BlockAttentionQuantScaleEnum::KV_BLOCKSCALE)
            {
                tile_elementwise_inout([&k_descale](auto& x) { x *= k_descale; }, s_acc);
            }

            const auto p = [&]() {
                const auto bias_tile = load_tile(bias_dram_window); // load bias tile

                // STAGE 2, scale_s, add bias, mask, softmax
                if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
                {
                    s_acc = tile_elementwise_in(s_acc_element_func, s_acc);
                    tile_elementwise_inout([&scale_s](auto& x) { x = x * scale_s; }, s_acc);
                    tile_elementwise_inout(
                        [&](auto& x, const auto& y) {
#if !CK_TILE_FMHA_FWD_FAST_EXP2
                            x += type_convert<SaccDataType>(bias_element_func(y));
#else
                            x += log2e_v<SaccDataType> *
                                 type_convert<SaccDataType>(bias_element_func(y));
#endif
                        },
                        s_acc,
                        bias_tile);
                }
                else if constexpr(BiasEnum == BlockAttentionBiasEnum::ALIBI)
                {
                    const auto k_origin    = k_dram_block_window.get_window_origin();
                    constexpr auto s_spans = decltype(s_acc)::get_distributed_spans();
                    s_acc                  = tile_elementwise_in(s_acc_element_func, s_acc);
                    sweep_tile_span(s_spans[number<0>{}], [&](auto idx0) {
                        sweep_tile_span(s_spans[number<1>{}], [&](auto idx1) {
                            const auto tile_idx = get_x_indices_from_distributed_indices(
                                s_acc.get_tile_distribution(), make_tuple(idx0, idx1));

                            const auto row = q_origin.at(number<0>{}) + tile_idx.at(number<0>{});
                            const auto col = k_origin.at(number<0>{}) + tile_idx.at(number<1>{});
                            constexpr auto i_j_idx = make_tuple(idx0, idx1);

                            s_acc(i_j_idx) *= scale_s;
                            position_encoding.update(s_acc(i_j_idx), row, col);
                        });
                    });
                }
                else
                {
                    s_acc = tile_elementwise_in(s_acc_element_func, s_acc);
                    if constexpr(kHasLogitsSoftCap)
                    {
                        auto apply_logits_transform = [&variant, &variant_params, &block_indices](
                                                          auto& x) {
                            x = variant.LogitsTransform(variant_params,
                                                        variant.QueryTransform(variant_params, x),
                                                        block_indices.batch_idx,
                                                        block_indices.qo_head_idx,
                                                        block_indices.kv_head_idx);
                        };
#if !CK_TILE_FMHA_FWD_FAST_EXP2
                        for(index_t i = 0; i < s_acc.thread_buf_.size(); ++i)
                        {
                            apply_logits_transform(s_acc.thread_buf_[i]);
                        }
#else
                        for(index_t i = 0; i < s_acc.thread_buf_.size(); ++i)
                        {
#if(defined(__gfx90a__) || defined(__gfx94__)) &&                                               \
    (CK_TILE_ATTENTION_LOGITS_SOFT_CAP_DEFAULT == CK_TILE_ATTENTION_LOGITS_SOFT_CAP_SOFTSIGN && \
     CK_TILE_ATTENTION_USE_SOFTSIGN_ASM)
                            // Avoid data hazard if v_mfma is followed by inline asm consumer
                            // instructions. In this case, compiler won't add s_nop for us
                            if(i == s_acc.thread_buf_.size() / 2)
                            {
                                __builtin_amdgcn_sched_barrier(0);
                            }
#endif
                            apply_logits_transform(s_acc.thread_buf_[i]);
                        }
#endif
                    }
                    else
                    {
#if !CK_TILE_FMHA_FWD_FAST_EXP2
                        tile_elementwise_inout([&scale_s](auto& x) { x = x * scale_s; }, s_acc);
#endif
                    }
                }
                if constexpr(kHasSink)
                {
                    if(i_total_loops == num_sink_loop - 1)
                        move_tile_window(bias_dram_window, {0, seqlen_k_start - sink_seq_end});
                }
                move_tile_window(bias_dram_window, {0, kN0});
                if constexpr(kPadSeqLenK || FmhaMask::IsMasking)
                {
                    const auto k_origin      = k_dram_block_window.get_window_origin();
                    bool need_perpixel_check = mask.IsEdgeTile(q_origin.at(number<0>{}),
                                                               k_origin.at(number<0>{}),
                                                               number<kM0>{},
                                                               number<kN0>{});

                    if(need_perpixel_check)
                    {
                        auto apply_mask = [&](auto&& mask_func) {
                            set_tile_if(s_acc,
                                        -numeric<SMPLComputeDataType>::infinity(),
                                        [&](auto tile_idx) {
                                            const auto row =
                                                q_origin.at(number<0>{}) + tile_idx.at(number<0>{});
                                            const auto col =
                                                k_origin.at(number<0>{}) + tile_idx.at(number<1>{});
                                            return !mask_func(variant_params,
                                                              block_indices.batch_idx,
                                                              row,
                                                              col,
                                                              block_indices.qo_head_idx,
                                                              block_indices.kv_head_idx);
                                        });
                        };

                        if constexpr(kHasSink)
                        {
                            apply_mask([&](auto&&... args) {
                                return variant.LogitsSinkMask(
                                    std::forward<decltype(args)>(args)...);
                            });
                        }
                        else
                        {
                            apply_mask([&](auto&&... args) {
                                return variant.LogitsMask(std::forward<decltype(args)>(args)...);
                            });
                        }
                    }
                }

                const auto s = cast_tile<SMPLComputeDataType>(s_acc); // S{j}

                // Prefetch V physical pages early - overlaps with softmax computation
                if constexpr(k1_loops > 1)
                {
                    save_and_prefetch_v_pages(number<2 * kK1>{});
                }

                auto m_local = block_tile_reduce<SMPLComputeDataType>(
                    s,
                    sequence<1>{},
                    f_max,
                    -numeric<SMPLComputeDataType>::infinity()); // m_local = rowmax(S{j})
                block_tile_reduce_sync(m_local, f_max, bool_constant<false>{});

                const auto m_old = m; // m{j-1}
                tile_elementwise_inout([](auto& e0, auto e1, auto e2) { e0 = max(e1, e2); },
                                       m,
                                       m_old,
                                       m_local); // m{j}

                auto p_compute = make_static_distributed_tensor<SMPLComputeDataType>(
                    s.get_tile_distribution()); // Pcompute{j}

                __builtin_amdgcn_sched_barrier(0x7F);
                // store & prefetch next v, after the max reduction
                if constexpr(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor> &&
                             kKVMemoryLayout ==
                                 BlockAttentionKVCacheMemoryLayoutEnum::LINEAR_LAYOUT)
                {
                    auto v_shuffle_tmp = make_static_distributed_tensor<VDataType>(
                        Policy::template MakeShuffledVRegBlockDescriptor<Problem>());
                    shuffle_tile(v_shuffle_tmp, v_buf);

                    auto v_lds_window_tmp =
                        get_slice_tile(v_lds_window,
                                       sequence<(LdsSeq.at(number<k0_loops>{})) * kN1, 0>{},
                                       sequence<(LdsSeq.at(number<k0_loops>{}) + 1) * kN1, kK1>{});

                    store_tile(
                        v_lds_window_tmp,
                        tile_elementwise_in(v_element_func, v_shuffle_tmp)); // store the prefetch
                }
                else
                {
                    auto v_lds_window_tmp =
                        get_slice_tile(v_lds_window,
                                       sequence<(LdsSeq.at(number<k0_loops>{})) * kN1, 0>{},
                                       sequence<(LdsSeq.at(number<k0_loops>{}) + 1) * kN1, kK1>{});
                    const auto v_store_tile = tile_elementwise_in(v_element_func, v_buf);
                    store_tile(v_lds_window_tmp, v_store_tile); // store the prefetch
                }

                if constexpr(k1_loops > 1)
                {
                    move_tile_window(
                        v_dram_window,
                        {0,
                         kK1}); // will have scratch if move this right after load_tile(v_dram)...
                    v_buf = load_tile(v_dram_window, number<-1>{}, bool_constant<false>{});
                    zero_logical_invalid_v(v_buf, number<kK1>{});
                    update_v_offsets(number<2 * kK1>{});
                    v_dram_window.update_page_idx(v_offsets);
                    rebase_v_window(v_dram_window, v_physical_pages[number<0>{}]);
                }
                __builtin_amdgcn_sched_barrier(0);

                static const auto get_validated_m = [](SMPLComputeDataType raw_m) {
                    /// NOTICE: bias might be materialized mask including -inf values, need
                    /// consideration. alibi does not have this problem
                    if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                                 FmhaMask::IsMasking)
                    {
                        return raw_m == -numeric<SMPLComputeDataType>::infinity()
                                   ? type_convert<SMPLComputeDataType>(0.f)
                                   : raw_m;
                    }
                    else
                    {
                        return raw_m;
                    }
                };

                constexpr auto p_spans = decltype(p_compute)::get_distributed_spans();
                sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
                    constexpr auto i_idx = make_tuple(idx0);
#if CK_TILE_FMHA_FWD_FAST_EXP2
                    // For KV_BLOCKSCALE: precompute (m - shift) once per row
                    // exp2(s - (m - shift)) = exp2(s - m + shift) = exp2(s - m) * 2^shift
                    // This scales P by 2^shift (≈448 for fp8_e4m3) without explicit multiply
                    auto validated_m = get_validated_m(m[i_idx]);
                    auto row_max     = scale_s * validated_m;
                    if constexpr(QScaleEnum == BlockAttentionQuantScaleEnum::KV_BLOCKSCALE)
                    {
#if CK_TILE_USE_OCP_FP8
                        validated_m -= OCP_FP8_SHIFT; // for Bias/Alibi/SoftCap
                        row_max -= OCP_FP8_SHIFT;     // for else branch
#else
                        validated_m -= FNUZ_FP8_SHIFT;
                        row_max -= FNUZ_FP8_SHIFT;
#endif
                    }
#endif
                    sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);
#if CK_TILE_FMHA_FWD_FAST_EXP2
                        if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                                     BiasEnum == BlockAttentionBiasEnum::ALIBI)
                        {
                            p_compute(i_j_idx) = exp2(s[i_j_idx] - validated_m);
                        }
                        else
                        {
                            if constexpr(kHasLogitsSoftCap)
                            {
                                p_compute(i_j_idx) = exp2(s[i_j_idx] - validated_m);
                            }
                            else
                            {
                                p_compute(i_j_idx) = exp2(scale_s * s[i_j_idx] - row_max);
                            }
                        }
#else
                        p_compute(i_j_idx)     = exp(s[i_j_idx] - get_validated_m(m[i_idx]));
#endif
                    });
                });

                auto rowsum_p = block_tile_reduce<SMPLComputeDataType>(
                    p_compute, sequence<1>{}, f_sum, SMPLComputeDataType{0}); // rowsum(Pcompute{j})

                block_tile_reduce_sync(rowsum_p, f_sum, bool_constant<false>{});
                // l{j}, Oacc{j}
                constexpr auto o_spans = decltype(o_acc)::get_distributed_spans();
                sweep_tile_span(o_spans[number<0>{}], [&](auto idx0) {
                    constexpr auto i_idx = make_tuple(idx0);
#if CK_TILE_FMHA_FWD_FAST_EXP2
                    const auto tmp = [&]() {
                        if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                                     BiasEnum == BlockAttentionBiasEnum::ALIBI)
                        {
                            return exp2(m_old[i_idx] - get_validated_m(m[i_idx]));
                        }
                        else
                        {
                            if constexpr(kHasLogitsSoftCap)
                            {
                                return exp2(m_old[i_idx] - get_validated_m(m[i_idx]));
                            }
                            else
                            {
                                auto row_max = scale_s * get_validated_m(m[i_idx]);
                                return exp2(scale_s * m_old[i_idx] - row_max);
                            }
                        }
                    }();
#else
                    const auto tmp = exp(m_old[i_idx] - get_validated_m(m[i_idx]));
#endif
                    l(i_idx) = tmp * l[i_idx] + rowsum_p[i_idx];
                    sweep_tile_span(o_spans[number<1>{}], [&](auto idx1) {
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);
                        // FIXME: this use different equation from FA v2 paper,
                        // but produce correc result.
                        // Is the equation wrong?
                        o_acc(i_j_idx) *= tmp;
                    });
                });

                if constexpr(kHasDropout)
                {
                    auto randval_ptr = reinterpret_cast<char*>(smem_ptr) +
                                       Policy::template GetSmemSizeKV<Problem>();
                    index_t seq_offset = [&]() {
                        if constexpr(kHasSink)
                        {
                            const bool in_sink_phase = (num_sink_loop > i_total_loops);
                            if(i_total_loops == num_sink_loop)
                                move_tile_window(randval_dram_window,
                                                 {0, seqlen_k_start - sink_seq_end});
                            return in_sink_phase
                                       ? (kv_load_start + i_total_loops * kN0)
                                       : (seqlen_k_start + (i_total_loops - num_sink_loop) * kN0);
                        }
                        else
                            return seqlen_k_start + i_total_loops * kN0;
                    }();
                    dropout
                        .template Run<decltype(gemm_0), SMPLComputeDataType, RandValOutputDataType>(
                            randval_ptr, seq_offset, p_compute, randval_dram_window);
                }

#if CK_TILE_FMHA_FLOAT_TO_FLOAT16_RTN
                // For fp32 to fp16,
                // impl::cast_tile_pkrtz_fp16_fp32 would cause precision issue,
                // since it uses __builtin_amdgcn_cvt_pkrtz, which is round to zero.
                return cast_tile<PDataType>(tile_elementwise_in(p_compute_element_func, p_compute));
#else
                if constexpr(std::is_same_v<PDataType, fp16_t>)
                    return impl::cast_tile_pkrtz_fp16_fp32<PDataType>(
                        tile_elementwise_in(p_compute_element_func, p_compute));
                else
                    return cast_tile<PDataType>(
                        tile_elementwise_in(p_compute_element_func, p_compute));
#endif
            }();

            // STAGE 3, KV gemm
            // KV_BLOCKSCALE: accumulate P*V into temporary tile before applying v_descale
            auto o_acc_unscaled = decltype(o_acc){};
            if constexpr(QScaleEnum == BlockAttentionQuantScaleEnum::KV_BLOCKSCALE)
            {
                clear_tile(o_acc_unscaled);
            }

            // Select GEMM1 target: o_acc_unscaled for KV_BLOCKSCALE (needs v_descale), o_acc
            // otherwise
            auto& gemm1_acc = [&]() -> auto& {
                if constexpr(QScaleEnum == BlockAttentionQuantScaleEnum::KV_BLOCKSCALE)
                    return o_acc_unscaled;
                else
                    return o_acc;
            }();

            if constexpr(k1_loops > 1)
            {
                static_for<0, k1_loops - 1, 1>{}([&](auto i_k1) {
                    if constexpr(i_k1 != 0 && i_k1 < k1_loops - 1)
                    {
                        v_buf = load_tile(v_dram_window, number<-1>{}, bool_constant<false>{});
                        zero_logical_invalid_v(v_buf, number<(1 + i_k1.value) * kK1>{});
                        // Update V offsets using previously prefetched physical pages
                        update_v_offsets(number<(2 + i_k1.value) * kK1>{});
                        v_dram_window.update_page_idx(v_offsets);
                        rebase_v_window(v_dram_window, v_physical_pages[number<0>{}]);
                    }

                    // Prefetch V physical pages for NEXT iteration - overlaps with GEMM1
                    if constexpr(i_k1 + 1 < k1_loops - 1)
                    {
                        save_and_prefetch_v_pages(number<(2 + i_k1.value + 1) * kK1>{});
                    }

                    block_sync_lds();
                    gemm_1(gemm1_acc,
                           get_slice_tile(
                               p, sequence<0, i_k1 * kK1>{}, sequence<kM0, (i_k1 + 1) * kK1>{}),
                           get_slice_tile(
                               v_lds_window,
                               sequence<(LdsSeq.at(number<k0_loops + i_k1>{})) * kN1, 0>{},
                               sequence<(LdsSeq.at(number<k0_loops + i_k1>{}) + 1) * kN1, kK1>{}));

                    if constexpr(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor> &&
                                 kKVMemoryLayout ==
                                     BlockAttentionKVCacheMemoryLayoutEnum::LINEAR_LAYOUT)
                    {
                        auto v_shuffle_tmp = make_static_distributed_tensor<VDataType>(
                            Policy::template MakeShuffledVRegBlockDescriptor<Problem>());
                        shuffle_tile(v_shuffle_tmp, v_buf);
                        auto v_lds_window_tmp = get_slice_tile(
                            v_lds_window,
                            sequence<(LdsSeq.at(number<k0_loops + i_k1 + 1>{})) * kN1, 0>{},
                            sequence<(LdsSeq.at(number<k0_loops + i_k1 + 1>{}) + 1) * kN1, kK1>{});
                        store_tile(v_lds_window_tmp,
                                   tile_elementwise_in(v_element_func,
                                                       v_shuffle_tmp)); // store the prefetch
                    }
                    else
                    {
                        auto v_lds_window_tmp = get_slice_tile(
                            v_lds_window,
                            sequence<(LdsSeq.at(number<k0_loops + i_k1 + 1>{})) * kN1, 0>{},
                            sequence<(LdsSeq.at(number<k0_loops + i_k1 + 1>{}) + 1) * kN1, kK1>{});
                        store_tile(v_lds_window_tmp,
                                   tile_elementwise_in(v_element_func, v_buf)); // store next v_buf
                    }
                    if constexpr(i_k1 < k1_loops - 1)
                        move_tile_window(v_dram_window, {0, kK1});
                });
            }
            i_total_loops++;
            if(i_total_loops < num_total_loop)
            {
                // For sink: after the last sink tile, jump K/V to seqlen_k_start;
                // otherwise advance by one normal tile.
                const index_t k_advance = [&]() -> index_t {
                    if constexpr(kHasSink)
                        return (i_total_loops == num_sink_loop)
                                   ? (seqlen_k_start - sink_seq_end + kN0)
                                   : kN0;
                    else
                        return kN0;
                }();
                current_seq_k += k_advance;
                // move K tile windows
                move_tile_window(k_dram_block_window, {k_advance, 0});
                k_dram_window.set_window_origin(k_dram_block_window.get_window_origin());

                // KV_BLOCKSCALE: reload physical pages for the new tile
                load_physical_pages<statically_indexed_array<index_t, NRepeat>,
                                    decltype(k_coord),
                                    0,
                                    kPageBlockSize,
                                    0,
                                    NRepeat,
                                    kN0 / NRepeat,
                                    kKVMemoryLayout,
                                    true,
                                    kN0>(
                    page_idx, k_coord, current_seq_k, k_physical_pages, max_page_table_idx);

                kv_offset_array_transform<statically_indexed_array<index_t, NRepeat>,
                                          decltype(k_coord),
                                          0,
                                          kPageBlockSize,
                                          0,
                                          NRepeat,
                                          kN0 / NRepeat,
                                          kKVMemoryLayout,
                                          true,
                                          kN0,
                                          kVectorSize,
                                          kUseGlobalLoad>(
                    k_physical_pages, stride_k, page_stride_k, k_coord, k_offsets, current_seq_k);
                k_dram_window.update_page_idx(k_offsets);
                if constexpr(kUseGlobalLoad)
                    k_dram_window.update_physical_pages(k_physical_pages);
                rebase_k_window(k_dram_window, k_physical_pages[number<0>{}]);

                // After sink→window transition (i_total_loops == num_sink_loop), V window
                // was advanced by kN0 (one normal iter), but current_seq_k jumped by k_advance
                // = seqlen_k_start - sink_seq_end + kN0 > kN0.  Re-init V to current_seq_k.
                if constexpr(kHasSink)
                {
                    if(i_total_loops == num_sink_loop && num_sink_loop > 0)
                    {
                        prefetch_v_physical_pages(number<0>{});
                        update_v_offsets(number<0>{});
                        v_dram_window.update_page_idx(v_offsets);
                        rebase_v_window(v_dram_window, v_physical_pages[number<0>{}]);
                    }
                }

                if constexpr(k1_loops >= 2 &&
                             LdsSeq.at(number<0>{}) == LdsSeq.at(number<k0_loops + k1_loops - 2>{}))
                    __builtin_amdgcn_s_barrier();
                async_load_tile_raw(k_lds_store(LdsSeq.at(number<0>{})),
                                    k_dram_window,
                                    number<-1>{},
                                    k_oob_ck,
                                    k_pre_np);
                move_tile_window(k_dram_window, {0, kK0});
            }
            // tail
            {
                block_sync_lds();
                gemm_1(
                    gemm1_acc,
                    get_slice_tile(p, sequence<0, (k1_loops - 1) * kK1>{}, sequence<kM0, kN0>{}),
                    get_slice_tile(
                        v_lds_window,
                        sequence<(LdsSeq.at(number<k0_loops + k1_loops - 1>{})) * kN1, 0>{},
                        sequence<(LdsSeq.at(number<k0_loops + k1_loops - 1>{}) + 1) * kN1, kK1>{}));
            }

            // KV_BLOCKSCALE: apply v_descale and accumulate o_acc_unscaled into o_acc
            // Note: No division by scale_p needed because:
            // 1. P was scaled by 2^shift through exp2 shift trick
            // 2. rowsum l was also scaled by 2^shift
            // 3. Final O = sum(P*V) / l, so the 2^shift cancels out
            if constexpr(QScaleEnum == BlockAttentionQuantScaleEnum::KV_BLOCKSCALE)
            {
                tile_elementwise_inout(
                    [&v_descale](auto& o, auto& o_unscaled) { o += o_unscaled * v_descale; },
                    o_acc,
                    o_acc_unscaled);
            }
        } while(i_total_loops < num_total_loop);

        // store lse
        if constexpr(kStoreLSE)
        {
            auto lse = make_static_distributed_tensor<LSEDataType>(m.get_tile_distribution());

            constexpr auto lse_spans = decltype(lse)::get_distributed_spans();
            sweep_tile_span(lse_spans[number<0>{}], [&, m_ = m, l_ = l](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
#if CK_TILE_FMHA_FWD_FAST_EXP2
                if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                             BiasEnum == BlockAttentionBiasEnum::ALIBI)
                {
                    lse(i_idx) = m_[i_idx] * R_LOG2E + log(l_[i_idx]);
                }
                else
                {
                    if constexpr(kHasLogitsSoftCap)
                    {
                        lse(i_idx) = m_[i_idx] * R_LOG2E + log(l_[i_idx]);
                    }
                    else
                    {
                        lse(i_idx) = m_[i_idx] * scale_s * R_LOG2E + log(l_[i_idx]);
                    }
                }
#else
                lse(i_idx) = m_[i_idx] + log(l_[i_idx]);
#endif
            });

            store_tile(lse_dram_window_tmp, tile_elementwise_in(lse_element_func, lse));
        }

        // finally, O
        constexpr auto o_spans = decltype(o_acc)::get_distributed_spans();

        sweep_tile_span(o_spans[number<0>{}], [&](auto idx0) {
            constexpr auto i_idx = make_tuple(idx0);
            const auto tmp       = [&]() {
                if constexpr(FmhaMask::IsMasking)
                {
                    return l[i_idx] == 0.f ? 0.f : 1 / l[i_idx];
                }
                else
                    return 1 / l[i_idx];
            }();
            sweep_tile_span(o_spans[number<1>{}], [&](auto idx1) {
                constexpr auto i_j_idx = make_tuple(idx0, idx1);
                o_acc(i_j_idx) *= tmp;
            });
        });

        o_acc = tile_elementwise_in(o_acc_element_func, o_acc);

        return o_acc;
    }

    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename BiasDramBlockWindowTmp,
              typename RandValDramBlockWindowTmp,
              typename LSEDramBlockWindowTmp,
              typename PositionEncoding,
              typename AttentionVariantParams,
              typename BlockIndices>
    CK_TILE_HOST_DEVICE auto
    operator()(const QDramBlockWindowTmp& q_dram_block_window_tmp,       // M0*K0 tile
               const KDramBlockWindowTmp& k_dram_block_window_tmp,       // N0*K0 tile
               const VDramBlockWindowTmp& v_dram_block_window_tmp,       // N1*K1 tile
               const BiasDramBlockWindowTmp& bias_dram_block_window_tmp, // M0*N0 tile
               RandValDramBlockWindowTmp& randval_dram_block_window_tmp, // M0*N0 tile
               LSEDramBlockWindowTmp& lse_dram_block_window_tmp,         // M0*1 tile
               FmhaMask mask,
               PositionEncoding position_encoding,
               float scale_s,
               const AttentionVariant& variant,
               const AttentionVariantParams& variant_params,
               const BlockIndices& block_indices,
               void* smem_ptr,
               const index_t* page_idx,
               const index_t stride_k,
               const index_t stride_v,
               const index_t page_stride_k,
               const index_t page_stride_v,
               DropoutType& dropout,
               float sink_v,
               const index_t max_page_table_idx) const
    {
        return operator()(q_dram_block_window_tmp,
                          identity{},
                          k_dram_block_window_tmp,
                          identity{},
                          v_dram_block_window_tmp,
                          identity{},
                          bias_dram_block_window_tmp,
                          identity{},
                          randval_dram_block_window_tmp,
                          lse_dram_block_window_tmp,
                          identity{},
                          identity{},
                          identity{},
                          identity{},
                          mask,
                          position_encoding,
                          scale_s,
                          variant,
                          variant_params,
                          block_indices,
                          smem_ptr,
                          page_idx,
                          stride_k,
                          stride_v,
                          page_stride_k,
                          page_stride_v,
                          dropout,
                          sink_v,
                          max_page_table_idx);
    }

    // Overload for KV_BLOCKSCALE: K/V descale is per-page
    // This is a convenience overload that forwards to the main operator() with kv_scale parameters
    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename BiasDramBlockWindowTmp,
              typename RandValDramBlockWindowTmp,
              typename LSEDramBlockWindowTmp,
              typename PositionEncoding,
              typename AttentionVariantParams,
              typename BlockIndices>
    CK_TILE_HOST_DEVICE auto
    operator()(const QDramBlockWindowTmp& q_dram_block_window_tmp,       // M0*K0 tile
               const KDramBlockWindowTmp& k_dram_block_window_tmp,       // N0*K0 tile
               const VDramBlockWindowTmp& v_dram_block_window_tmp,       // N1*K1 tile
               const BiasDramBlockWindowTmp& bias_dram_block_window_tmp, // M0*N0 tile
               RandValDramBlockWindowTmp& randval_dram_block_window_tmp, // M0*N0 tile
               LSEDramBlockWindowTmp& lse_dram_block_window_tmp,         // M0*1 tile
               FmhaMask mask,
               PositionEncoding position_encoding,
               float scale_s,
               const AttentionVariant& variant,
               const AttentionVariantParams& variant_params,
               const BlockIndices& block_indices,
               void* smem_ptr,
               const index_t* page_idx,
               const index_t stride_k,
               const index_t stride_v,
               const index_t page_stride_k,
               const index_t page_stride_v,
               DropoutType& dropout,
               float sink_v,
               const index_t max_page_table_idx,
               const float* k_descale_ptr,
               const float* v_descale_ptr,
               index_t nblock_stride_kv_block_descale,
               index_t nhead_stride_kv_block_descale) const
    {
        return operator()(q_dram_block_window_tmp,
                          identity{},
                          k_dram_block_window_tmp,
                          identity{},
                          v_dram_block_window_tmp,
                          identity{},
                          bias_dram_block_window_tmp,
                          identity{},
                          randval_dram_block_window_tmp,
                          lse_dram_block_window_tmp,
                          identity{},
                          identity{},
                          identity{},
                          identity{},
                          mask,
                          position_encoding,
                          scale_s,
                          variant,
                          variant_params,
                          block_indices,
                          smem_ptr,
                          page_idx,
                          stride_k,
                          stride_v,
                          page_stride_k,
                          page_stride_v,
                          dropout,
                          sink_v,
                          max_page_table_idx,
                          k_descale_ptr,
                          v_descale_ptr,
                          nblock_stride_kv_block_descale,
                          nhead_stride_kv_block_descale);
    }
};

} // namespace ck_tile
