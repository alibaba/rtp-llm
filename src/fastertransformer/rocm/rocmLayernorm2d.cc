
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include <ck_tile/core.hpp>
#include "rocmLayernorm2d.h"
#include "maga_transformer/cpp/utils/Logger.h"

namespace fastertransformer {

// Note: this internal API only declare, not define here, otherwise will block `make -j`
template <typename Traits_>
float layernorm2d_fwd_(const ck_tile::stream_config& s, layernorm2d_fwd_args a)
{
    using XDataType = typename Traits_::XDataType;
    using YDataType = typename Traits_::YDataType;
    using XScaleDataType = typename Traits_::XScaleDataType;
    using YScaleDataType = typename Traits_::YScaleDataType;
    using ComputeDataType = typename LayerNormTypeConfig<XDataType, YDataType, XScaleDataType, YScaleDataType>::ComputeDataType;

    using PipelineTraits = ck_tile::Layernorm2dFwdTraits<Traits_::kPadN,
        Traits_::kSaveMeanInvStd,
        Traits_::kFastFDiv,
        Traits_::kTwoPass,
        static_cast<ck_tile::Layernorm2dFusedAddEnum>(Traits_::kFusedAdd),
        static_cast<ck_tile::Layernorm2dFusedQuantEnum>(Traits_::kFusedQuant)>;
    using PipelineProblem = ck_tile::Layernorm2dFwdPipelineProblem<
        typename LayerNormTypeConfig<XDataType, YDataType, XScaleDataType, YScaleDataType>::XDataType,
        typename LayerNormTypeConfig<XDataType, YDataType, XScaleDataType, YScaleDataType>::GammaDataType,
        typename LayerNormTypeConfig<XDataType, YDataType, XScaleDataType, YScaleDataType>::BetaDataType,
        typename LayerNormTypeConfig<XDataType, YDataType, XScaleDataType, YScaleDataType>::ComputeDataType,
        typename LayerNormTypeConfig<XDataType, YDataType, XScaleDataType, YScaleDataType>::YDataType,
        typename LayerNormTypeConfig<XDataType, YDataType, XScaleDataType, YScaleDataType>::MeanDataType,
        typename LayerNormTypeConfig<XDataType, YDataType, XScaleDataType, YScaleDataType>::InvStdDataType,
        typename LayerNormTypeConfig<XDataType, YDataType, XScaleDataType, YScaleDataType>::XScaleDataType,
        typename LayerNormTypeConfig<XDataType, YDataType, XScaleDataType, YScaleDataType>::YScaleDataType,
        typename Traits_::Shape,
        PipelineTraits>;

    using OnePassPipeline = ck_tile::Layernorm2dFwdPipelineOnePass<PipelineProblem>;
    using TwoPassPipeline = ck_tile::Layernorm2dFwdPipelineTwoPass<PipelineProblem>;
    using Pipeline        = std::conditional_t<Traits_::kTwoPass, TwoPassPipeline, OnePassPipeline>;

    using Default2DEpilogueProblem = ck_tile::Default2DEpilogueProblem<ComputeDataType, YDataType, false, Traits_::kPadN, false>;
    using Default2DEpilogue = ck_tile::Default2DEpilogue<Default2DEpilogueProblem>;

    static constexpr bool UseSmoothInputScale = Traits_::kFusedQuant == 1;
    using DynamicQuantEpilogueProblem = ck_tile::DynamicQuantEpilogueProblem<ComputeDataType, XScaleDataType, YScaleDataType, YDataType, typename Traits_::Shape,
            ck_tile::DynamicQuantEpilogueTraits<false, Traits_::kPadN, UseSmoothInputScale, false,  true/*max3*/>>;

    using DynamicQuantEpilogue = ck_tile::DynamicQuantEpilogue<DynamicQuantEpilogueProblem>;

    using Epilogue = std::conditional_t<Traits_::kFusedQuant == 1, DynamicQuantEpilogue,  Default2DEpilogue>;

    using Kernel = ck_tile::Layernorm2dFwd<Pipeline, Epilogue>;

    const dim3 grids                       = Kernel::GridSize(a);
    constexpr dim3 blocks                  = Kernel::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = 1;

    auto kargs = Kernel::MakeKargs(a);
    if(s.log_level_ > 0)
        std::cout << ", " << Kernel::GetName() << std::flush;

    return ck_tile::launch_kernel(
        s, ck_tile::make_kernel<blocks.x, kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
};

float layernorm2d_fwd(layernorm2d_fwd_traits t,
                      layernorm2d_fwd_args a,
                      const ck_tile::stream_config& s)
{
    float r = -1;
    if(t.prec_i == "fp16" && t.prec_o == "fp16"){
        if (a.n <= 768) {
            if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  3,  4,   64,  4, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  6,  4,   64,  2, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1, 12,  4,   64,  1, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  3,  4,   64,  4, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  6,  4,   64,  2, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1, 12,  4,   64,  1, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  3,  4,   64,  4, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  6,  4,   64,  2, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1, 12,  4,   64,  1, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  3,  4,   64,  4, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  6,  4,   64,  2, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1, 12,  4,   64,  1, true , false, true , false,    1,    1>>(s, a);
        } else {

            FT_LOG_ERROR("layernorm size not supported.");
        }
    } else {

        FT_LOG_ERROR("layernorm type not supported.");
    }

    return r;
}

}  // namespace fastertransformer

