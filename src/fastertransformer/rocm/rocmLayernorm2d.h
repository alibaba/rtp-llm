// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/layernorm2d.hpp"
#include <string>
#include <ck_tile/ops/epilogue.hpp>
#include <iostream>

namespace fastertransformer {

template<typename InType, typename OutType, typename XScaleDataType_, typename YScaleDataType_>
struct LayerNormTypeConfig;

template<typename OutType, typename XScaleDataType_, typename YScaleDataType_>
struct LayerNormTypeConfig<ck_tile::half_t, OutType, XScaleDataType_, YScaleDataType_> {
    using XDataType       = ck_tile::half_t;
    using YDataType       = OutType;
    using GammaDataType   = ck_tile::half_t;
    using BetaDataType    = ck_tile::half_t;
    using MeanDataType    = ck_tile::half_t;
    using InvStdDataType  = ck_tile::half_t;
    using ComputeDataType = float;
    using XScaleDataType  = XScaleDataType_;
    using YScaleDataType  = YScaleDataType_;
};

template<typename OutType, typename XScaleDataType_, typename YScaleDataType_>
struct LayerNormTypeConfig<ck_tile::bf16_t, OutType, XScaleDataType_, YScaleDataType_> {
    using XDataType       = ck_tile::bf16_t;
    using YDataType       = OutType;
    using GammaDataType   = ck_tile::bf16_t;
    using BetaDataType    = ck_tile::bf16_t;
    using MeanDataType    = ck_tile::bf16_t;
    using InvStdDataType  = ck_tile::bf16_t;
    using ComputeDataType = float;
    using XScaleDataType  = XScaleDataType_;
    using YScaleDataType  = YScaleDataType_;
};

// runtime args
struct layernorm2d_fwd_args: public ck_tile::Layernorm2dFwdHostArgs {};

// This is the public API, will be generated by script
struct layernorm2d_fwd_traits {
    std::string prec_i;  // input precision
    std::string prec_o;  // output precision

    // if fused_quant == 1, need set prec_sx/prec_sy to proper string, otherwise can set
    // arbitrary(will skip check) if fused_quant == 2, need set prec_sy to proper string, otherwise
    // can set arbitrary(will skip check)
    std::string prec_sx;  // x-scale, used for [1*N] input smooth quant
    std::string prec_sy;  // y-scale, used for [M*1] output for next layer

    bool save_mean_var;  //
    int  fused_add;      // 0:no-add, 1:pre-add-store, 2:pre-add
    int  fused_quant;    // 0:no-sweep, 1:smooth-dynamic-quant, 2:dynamic-quant
};

float layernorm2d_fwd(layernorm2d_fwd_traits, layernorm2d_fwd_args, const ck_tile::stream_config&);

using S = ck_tile::stream_config;
using A = layernorm2d_fwd_args;

// this is used to pattern-match internl kernel implementation, not to instantiate kernel
template<typename XDataType_,
         typename YDataType_,
         typename XScaleDataType_,
         typename YScaleDataType_,
         ck_tile::index_t Repeat_M_,          // each thread repeat along M
         ck_tile::index_t Repeat_N_,          // each thread repeat along N
         ck_tile::index_t ThreadPerBlock_M_,  // num threads along M
         ck_tile::index_t ThreadPerBlock_N_,  // num threads along N
         ck_tile::index_t Vector_N_,          // vector size along N
         bool             kPadN_,
         bool             kSaveMeanInvStd_,
         bool             kFastFDiv_,
         bool             kTwoPass_,
         ck_tile::index_t kFusedAdd_   = 0,
         ck_tile::index_t kFusedQuant_ = 0>
struct layernorm2d_fwd_traits_ {
    using XDataType      = ck_tile::remove_cvref_t<XDataType_>;
    using YDataType      = ck_tile::remove_cvref_t<YDataType_>;
    using XScaleDataType = ck_tile::remove_cvref_t<XScaleDataType_>;
    using YScaleDataType = ck_tile::remove_cvref_t<YScaleDataType_>;

    static constexpr bool is_warp_per_row = ThreadPerBlock_N_ <= warpSize;
    static_assert((ThreadPerBlock_M_ * ThreadPerBlock_N_) % warpSize == 0);
    static constexpr ck_tile::index_t total_warps = (ThreadPerBlock_M_ * ThreadPerBlock_N_) / warpSize;

    // num of warps along m
    static constexpr ck_tile::index_t BlockWarps_M = []() {
        if constexpr (is_warp_per_row) {
            static_assert(warpSize % ThreadPerBlock_N_ == 0);
            return total_warps * (warpSize / ThreadPerBlock_N_);
        } else {
            // static_assert(warpSize % ThreadPerBlock_M_ == 0);
            return total_warps / (ThreadPerBlock_N_ / warpSize);
        }
    }();

    // num of warps along n
    static constexpr ck_tile::index_t BlockWarps_N = []() {
        if constexpr (is_warp_per_row) {
            static_assert(warpSize % ThreadPerBlock_N_ == 0);
            return 1;
        } else {
            static_assert(ThreadPerBlock_N_ % warpSize == 0);
            return ThreadPerBlock_N_ / warpSize;
        }
    }();

    static constexpr ck_tile::index_t Repeat_M = Repeat_M_;
    static constexpr ck_tile::index_t Repeat_N = Repeat_N_;

    static constexpr ck_tile::index_t Block_M = Repeat_M_ * ThreadPerBlock_M_;
    static constexpr ck_tile::index_t Block_N = Repeat_N_ * ThreadPerBlock_N_ * Vector_N_;

    static constexpr ck_tile::index_t Warp_M = ThreadPerBlock_M_ / BlockWarps_M;
    static constexpr ck_tile::index_t Warp_N = ThreadPerBlock_N_ / BlockWarps_N * Vector_N_;

    using BlockTile  = ck_tile::sequence<Block_M, Block_N>;
    using BlockWarps = ck_tile::sequence<BlockWarps_M, BlockWarps_N>;
    using WarpTile   = ck_tile::sequence<Warp_M, Warp_N>;
    using Vector     = ck_tile::sequence<1, Vector_N_>;

    using Shape = ck_tile::Generic2dBlockShape<BlockTile, BlockWarps, WarpTile, Vector>;

    static constexpr bool             kPadN           = kPadN_;
    static constexpr bool             kSaveMeanInvStd = kSaveMeanInvStd_;
    static constexpr bool             kFastFDiv       = kFastFDiv_;
    static constexpr bool             kTwoPass        = kTwoPass_;
    static constexpr ck_tile::index_t kFusedAdd       = kFusedAdd_;
    static constexpr ck_tile::index_t kFusedQuant     = kFusedQuant_;
};

template<typename XDataType_,
         typename YDataType_,
         typename XScaleDataType_,
         typename YScaleDataType_,
         ck_tile::index_t Repeat_M_,          // each thread repeat along M
         ck_tile::index_t Repeat_N_,          // each thread repeat along N
         ck_tile::index_t ThreadPerBlock_M_,  // num threads along M
         ck_tile::index_t ThreadPerBlock_N_,  // num threads along N
         ck_tile::index_t Vector_N_,          // vector size along N
         bool             kPadN_,
         bool             kSaveMeanInvStd_,
         bool             kFastFDiv_,
         bool             kTwoPass_,
         int              kFusedAdd_,
         int              kFusedQuant_>
using traits_ = layernorm2d_fwd_traits_<XDataType_,
                                        YDataType_,
                                        XScaleDataType_,
                                        YScaleDataType_,
                                        Repeat_M_,
                                        Repeat_N_,
                                        ThreadPerBlock_M_,
                                        ThreadPerBlock_N_,
                                        Vector_N_,
                                        kPadN_,
                                        kSaveMeanInvStd_,
                                        kFastFDiv_,
                                        kTwoPass_,
                                        kFusedAdd_,
                                        kFusedQuant_>;

}  // namespace fastertransformer
