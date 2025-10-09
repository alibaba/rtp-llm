/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#ifndef _WIN32
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif // #ifndef _WIN32

#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass_extensions/compute_occupancy.h"
#include "cutlass_extensions/gemm/device/gemm_universal_base_compat.h"

#include "cutlass_extensions/epilogue_helpers.h"
#include "cutlass_extensions/gemm/kernel/default_fpA_intB_traits.h"
#include "cutlass_extensions/gemm/kernel/fpA_intB_gemm.h"
#include "cutlass_extensions/gemm/threadblock/default_mma.h"
#include "rtp_llm/cpp/cuda/cutlass/cutlass_kernels/gemm_configs.h"
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"

#ifndef _WIN32
#pragma GCC diagnostic pop
#endif // #ifndef _WIN32

#include "rtp_llm/cpp/cuda/cutlass/cutlass_kernels/gemm_lut.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/cuda/cutlass/cutlass_kernels/cutlass_heuristic.h"
#include "rtp_llm/cpp/cuda/cutlass/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm.h"
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"

namespace tc = tensorrt_llm::cutlass_extensions;

namespace tensorrt_llm
{
namespace kernels
{
namespace cutlass_kernels
{

template <typename T, typename WeightType, typename arch, cutlass::WeightOnlyQuantOp QuantOp, typename EpilogueTag,
    typename ThreadblockShape, typename WarpShape, int Stages>
void generic_mixed_gemm_kernelLauncher(const T* A, const WeightType* B, const T* weight_scales,
    const T* weight_zero_points, const T* biases, T* C, int m, int n, int k, const int group_size,
    tc::CutlassGemmConfig gemm_config, char* workspace, size_t workspace_bytes, cudaStream_t stream,
    int* occupancy = nullptr)
{
    RTP_LLM_LOG_TRACE(__PRETTY_FUNCTION__);

#ifdef ENABLE_BF16
    static_assert(cutlass::platform::is_same<T, __nv_bfloat16>::value || cutlass::platform::is_same<T, half>::value
            || cutlass::platform::is_same<T, float>::value,
        "Specialized for bfloat16, half, float");
#else
    static_assert(cutlass::platform::is_same<T, half>::value || cutlass::platform::is_same<T, float>::value,
        "Specialized for half, float");
#endif

    static_assert(cutlass::platform::is_same<T, WeightType>::value
            || cutlass::platform::is_same<WeightType, uint8_t>::value
            || cutlass::platform::is_same<WeightType, cutlass::uint4b_t>::value,
        "");

    // The cutlass type for the input elements. This is needed to convert to cutlass::half_t if necessary.
    using ElementType_ =
        typename cutlass::platform::conditional<cutlass::platform::is_same<T, half>::value, cutlass::half_t, T>::type;
#ifdef ENABLE_BF16
    using ElementType =
        typename cutlass::platform::conditional<cutlass::platform::is_same<ElementType_, __nv_bfloat16>::value,
            cutlass::bfloat16_t, ElementType_>::type;
#else
    using ElementType = ElementType_;
#endif

    using CutlassWeightType_ =
        typename cutlass::platform::conditional<cutlass::platform::is_same<WeightType, half>::value, cutlass::half_t,
            WeightType>::type;
#ifdef ENABLE_BF16
    using CutlassWeightType =
        typename cutlass::platform::conditional<cutlass::platform::is_same<CutlassWeightType_, __nv_bfloat16>::value,
            cutlass::bfloat16_t, CutlassWeightType_>::type;
#else
    using CutlassWeightType = CutlassWeightType_;
#endif

    // We need separate config for each architecture since we will target different tensorcore instructions. For float,
    // we do not target TCs.
    using MixedGemmArchTraits = cutlass::gemm::kernel::MixedGemmArchTraits<ElementType, CutlassWeightType, arch>;
    using ElementAccumulator = typename MixedGemmArchTraits::AccType;

    using EpilogueOp = typename tc::Epilogue<ElementType, MixedGemmArchTraits::ElementsPerAccessC, ElementAccumulator,
        EpilogueTag>::Op;

    // useless, just for compiler
    using Operator = typename MixedGemmArchTraits::Operator;
    using TaggedOperator = typename cutlass::arch::TagOperator<Operator, QuantOp>::TaggedOperator;

    using GemmKernel_ = typename cutlass::gemm::kernel::DefaultGemm<ElementType, cutlass::layout::RowMajor,
        MixedGemmArchTraits::ElementsPerAccessA, CutlassWeightType, typename MixedGemmArchTraits::LayoutB,
        MixedGemmArchTraits::ElementsPerAccessB, ElementType, cutlass::layout::RowMajor, ElementAccumulator,
        cutlass::arch::OpClassTensorOp, arch, ThreadblockShape, WarpShape,
        typename MixedGemmArchTraits::InstructionShape, EpilogueOp,
        typename cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, Stages, true,
        TaggedOperator>::GemmKernel;

    using GemmKernel = cutlass::gemm::kernel::GemmFpAIntB<typename GemmKernel_::Mma, typename GemmKernel_::Epilogue,
        typename GemmKernel_::ThreadblockSwizzle,
        arch, // Ensure top level arch is used for dispatch
        GemmKernel_::kSplitKSerial>;

    if (occupancy != nullptr)
    {
        thread_local int cached_kernel_occupancy = -1;
        if (cached_kernel_occupancy == -1) {
            cached_kernel_occupancy = tensorrt_llm::cutlass_extensions::compute_occupancy_for_kernel<GemmKernel>();
        }
        *occupancy = cached_kernel_occupancy;
        return;
    }

    using Gemm = cutlass::gemm::device::GemmUniversalBaseCompat<GemmKernel>;

    const int ldb = cutlass::platform::is_same<cutlass::layout::RowMajor, typename MixedGemmArchTraits::LayoutB>::value
        ? n
        : k * GemmKernel::kInterleave;

    if (weight_scales == nullptr)
    {
        throw std::runtime_error("Weight scales must always be set to a non-null value.");
    }

    if constexpr (cutlass::isFinegrained(QuantOp))
    {
        if (group_size != 64 && group_size != 128)
        {
            throw std::runtime_error("Only group size 64 and 128 supported for fine grained kernels.");
        }

        if constexpr (QuantOp == cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY)
        {
            if (weight_zero_points != nullptr)
            {
                throw std::runtime_error("Weight zero pointer must be a nullptr for scale only fine grained");
            }
        }
        else if constexpr (QuantOp == cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS)
        {
            if (weight_zero_points == nullptr)
            {
                throw std::runtime_error("Weight zero pointer must be valid for scale and bias fine grained");
            }
        }
    }
    else
    {
        if (group_size != k)
        {
            throw std::runtime_error("Invalid group size for per col'umn scaling kernels.");
        }

        if (weight_zero_points != nullptr)
        {
            throw std::runtime_error("Weight zero-points must be null when running per column scaling");
        }
    }

    const int ld_scale_zero = cutlass::isFinegrained(QuantOp) ? n : 0;
    ElementAccumulator output_op_beta = (biases == nullptr) ? ElementAccumulator(0.f) : ElementAccumulator(1.f);
    typename Gemm::Arguments args({m, n, k}, group_size, {reinterpret_cast<ElementType*>(const_cast<T*>(A)), k},
        {reinterpret_cast<CutlassWeightType*>(const_cast<WeightType*>(B)), ldb},
        {reinterpret_cast<ElementType*>(const_cast<T*>(weight_scales)), ld_scale_zero},
        {reinterpret_cast<ElementType*>(const_cast<T*>(weight_zero_points)), ld_scale_zero},
        {reinterpret_cast<ElementType*>(const_cast<T*>(biases)), 0}, {reinterpret_cast<ElementType*>(C), n},
        gemm_config.split_k_factor, {ElementAccumulator(1.f), output_op_beta});

    // This assertion is enabled because because for the column interleaved layout, K MUST be a multiple of
    // threadblockK. The reason for this is that the default pitchlinear iterators are used to handle walking over the
    // interleaved matrix. The way masking in handled in these do not map to the interleaved layout. We need to write
    // our own predicated iterator in order to relax this limitation.
    if (GemmKernel::kInterleave > 1
        && ((k % MixedGemmArchTraits::ThreadblockK)
            || ((k / gemm_config.split_k_factor) % MixedGemmArchTraits::ThreadblockK)))
    {
        throw std::runtime_error("Temp assertion: k must be multiple of threadblockK");
    }

    Gemm gemm;
    if (gemm.get_workspace_size(args) > workspace_bytes)
    {
        RTP_LLM_LOG_WARNING(
            "Requested split-k but workspace size insufficient. Falling back to non-split-k implementation.");
        // If requested split-k factor will require more workspace bytes, revert to standard gemm.
        args.batch_count = 1;
    }

    auto can_implement = gemm.can_implement(args);
    if (can_implement != cutlass::Status::kSuccess)
    {
        std::string err_msg = "fpA_intB cutlass kernel will fail for params. Error: "
            + std::string(cutlassGetStatusString(can_implement));
        throw std::runtime_error("[TensorRT-LLm Error][fpA_intB Runner] " + err_msg);
    }

    auto init_status = gemm.initialize(args, workspace, stream);
    if (init_status != cutlass::Status::kSuccess)
    {
        std::string err_msg
            = "Failed to initialize cutlass fpA_intB gemm. Error: " + std::string(cutlassGetStatusString(init_status));
        throw std::runtime_error("[TensorRT-LLm Error][fpA_intB Runner] " + err_msg);
    }

    auto run_status = gemm.run(stream);
    if (run_status != cutlass::Status::kSuccess)
    {
        std::string err_msg
            = "Failed to run cutlass fpA_intB gemm. Error: " + std::string(cutlassGetStatusString(run_status));
        throw std::runtime_error("[TensorRT-LLm Error][fpA_intB Runner] " + err_msg);
    }
}

// This filters out invalid template combinations that we DON'T want instantiated in CUTLASS. For example,
// instantiating SM=75, Stages=3 is invalid so we would need to filter that out. Fine grained
// quanitzation is only supported on Ampere+ GPUs.
template <typename T, typename WeightType, typename arch, cutlass::WeightOnlyQuantOp QuantOp, typename EpilogueTag,
    typename ThreadblockShape, typename WarpShape, int Stages>
void filter_and_run_mixed_gemm(const T* A, const WeightType* B, const T* weight_scales, const T* weight_zero_points,
    const T* biases, T* C, int m, int n, int k, const int group_size, tc::CutlassGemmConfig gemm_config,
    char* workspace, size_t workspace_bytes, cudaStream_t stream, int* occupancy = nullptr)
{

    RTP_LLM_LOG_TRACE(__PRETTY_FUNCTION__);
    if constexpr (arch::kMinComputeCapability < 70)
    {
        // Finegrained only supported on Ampere
        std::string err_msg = "Cutlass fpA_intB gemm not implemented for arch "
                              + std::to_string(arch::kMinComputeCapability);
        throw std::runtime_error("[TensorRT-LLm Error][filter_and_run_mixed_gemm] " + err_msg);
    }
    else if constexpr (Stages > 2 && arch::kMinComputeCapability < 80)
    {
        // Multistage only supported on Ampere
        std::string err_msg = "Cutlass fpA_intB gemm not supported for arch "
            + std::to_string(arch::kMinComputeCapability) + " with stages set to " + std::to_string(Stages);
        throw std::runtime_error("[TensorRT-LLm Error][filter_and_run_mixed_gemm] " + err_msg);
    }
    else if constexpr (cutlass::platform::is_same<T, __nv_bfloat16>::value &&
                       arch::kMinComputeCapability < 80)
    {
        throw std::runtime_error("[TensorRT-LLm Error][filter_and_run_mixed_gemm] " + std::to_string(arch::kMinComputeCapability) + " not support bf16");
    }
    else
    {
        generic_mixed_gemm_kernelLauncher<T, WeightType, arch, QuantOp, EpilogueTag, ThreadblockShape, WarpShape,
            Stages>(A, B, weight_scales, weight_zero_points, biases, C, m, n, k, group_size, gemm_config, workspace,
            workspace_bytes, stream, occupancy);
    }
}

template <typename T, typename WeightType, typename arch, cutlass::WeightOnlyQuantOp QuantOp, typename EpilogueTag,
    typename ThreadblockShape, typename WarpShape>
void dispatch_gemm_config(const T* A, const WeightType* B, const T* weight_scales, const T* weight_zero_points,
    const T* biases, T* C, int m, int n, int k, const int group_size, tc::CutlassGemmConfig gemm_config,
    char* workspace, size_t workspace_bytes, cudaStream_t stream, int* occupancy = nullptr)
{

    RTP_LLM_LOG_TRACE(__PRETTY_FUNCTION__);
    switch (gemm_config.stages)
    {
    case 2:
        filter_and_run_mixed_gemm<T, WeightType, arch, QuantOp, EpilogueTag, ThreadblockShape, WarpShape, 2>(A, B,
            weight_scales, weight_zero_points, biases, C, m, n, k, group_size, gemm_config, workspace, workspace_bytes,
            stream, occupancy);
        break;
    case 3:
        filter_and_run_mixed_gemm<T, WeightType, arch, QuantOp, EpilogueTag, ThreadblockShape, WarpShape, 3>(A, B,
            weight_scales, weight_zero_points, biases, C, m, n, k, group_size, gemm_config, workspace, workspace_bytes,
            stream, occupancy);
        break;
    case 4:
        filter_and_run_mixed_gemm<T, WeightType, arch, QuantOp, EpilogueTag, ThreadblockShape, WarpShape, 4>(A, B,
            weight_scales, weight_zero_points, biases, C, m, n, k, group_size, gemm_config, workspace, workspace_bytes,
            stream, occupancy);
        break;
    default:
        std::string err_msg = "dispatch_gemm_config does not support stages " + std::to_string(gemm_config.stages);
        throw std::runtime_error("[TensorRT-LLm Error][dispatch_gemm_config] " + err_msg);
        break;
    }
}

template <typename T, typename WeightType, typename arch, cutlass::WeightOnlyQuantOp QuantOp, typename EpilogueTag>
void dispatch_gemm_to_cutlass(const T* A, const WeightType* B, const T* weight_scales, const T* weight_zero_points,
    const T* biases, T* C, int m, int n, int k, const int group_size, char* workspace, size_t workspace_bytes,
    tc::CutlassGemmConfig gemm_config, cudaStream_t stream, int* occupancy = nullptr)
{

    RTP_LLM_LOG_TRACE(__PRETTY_FUNCTION__);

    // Note that SIMT configs are omitted here since they are not supported for fpA_intB.
    // We also only instantiate configs here where threadblockShapeM == warpShapeM since those usually perform the best
    // for mixed type gemms.
    switch (gemm_config.tile_config)
    {
    case tc::CutlassTileConfig::CtaShape16x128x64_WarpShape16x32x64:
        RTP_LLM_CHECK_WITH_INFO(arch::kMinComputeCapability >= 75, "Invalid config on Volta");
        if constexpr (arch::kMinComputeCapability >= 75)
        {
        dispatch_gemm_config<T, WeightType, arch, QuantOp, EpilogueTag, cutlass::gemm::GemmShape<16, 128, 64>,
            cutlass::gemm::GemmShape<16, 32, 64>>(A, B, weight_scales, weight_zero_points, biases, C, m, n, k,
            group_size, gemm_config, workspace, workspace_bytes, stream, occupancy);
        }
        break;
    case tc::CutlassTileConfig::CtaShape16x256x64_WarpShape16x64x64:
        RTP_LLM_CHECK_WITH_INFO(arch::kMinComputeCapability >= 75, "Invalid config on Volta");
        if constexpr (arch::kMinComputeCapability >= 75)
        {
        dispatch_gemm_config<T, WeightType, arch, QuantOp, EpilogueTag, cutlass::gemm::GemmShape<16, 256, 64>,
            cutlass::gemm::GemmShape<16, 64, 64>>(A, B, weight_scales, weight_zero_points, biases, C, m, n, k,
            group_size, gemm_config, workspace, workspace_bytes, stream, occupancy);
        }
        break;
    case tc::CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64:
        dispatch_gemm_config<T, WeightType, arch, QuantOp, EpilogueTag, cutlass::gemm::GemmShape<32, 128, 64>,
            cutlass::gemm::GemmShape<32, 32, 64>>(A, B, weight_scales, weight_zero_points, biases, C, m, n, k,
            group_size, gemm_config, workspace, workspace_bytes, stream, occupancy);
        break;
    case tc::CutlassTileConfig::CtaShape64x128x64_WarpShape64x32x64:
        dispatch_gemm_config<T, WeightType, arch, QuantOp, EpilogueTag, cutlass::gemm::GemmShape<64, 128, 64>,
            cutlass::gemm::GemmShape<64, 32, 64>>(A, B, weight_scales, weight_zero_points, biases, C, m, n, k,
            group_size, gemm_config, workspace, workspace_bytes, stream, occupancy);
        break;
    case tc::CutlassTileConfig::CtaShape128x128x64_WarpShape128x32x64:
        RTP_LLM_CHECK_WITH_INFO(arch::kMinComputeCapability >= 75, "Invalid config on Volta");
        if constexpr (arch::kMinComputeCapability >= 75)
        {
            dispatch_gemm_config<T, WeightType, arch, QuantOp, EpilogueTag, cutlass::gemm::GemmShape<128, 128, 64>,
                cutlass::gemm::GemmShape<128, 32, 64>>(A, B, weight_scales, weight_zero_points, biases, C, m, n, k,
                group_size, gemm_config, workspace, workspace_bytes, stream, occupancy);
        }
        break;
    case tc::CutlassTileConfig::Undefined:
        throw std::runtime_error("[TensorRT-LLm Error][fpA_intB][dispatch_gemm_to_cutlass] gemm config undefined.");
        break;
    case tc::CutlassTileConfig::ChooseWithHeuristic:
        throw std::runtime_error(
            "[TensorRT-LLm Error][fpA_intB][dispatch_gemm_to_cutlass] gemm config should have already been set by "
            "heuristic.");
        break;
    default:
        throw std::runtime_error(
            "[TensorRT-LLm Error][fpA_intB][dispatch_gemm_to_cutlass] Config is invalid for mixed type GEMM.");
        break;
    }
}

template <typename T, typename WeightType, cutlass::WeightOnlyQuantOp QuantOp>
CutlassFpAIntBGemmRunner<T, WeightType, QuantOp>::CutlassFpAIntBGemmRunner()
{
    RTP_LLM_LOG_TRACE(__PRETTY_FUNCTION__);
    sm_ = rtp_llm::get_sm();
    multi_processor_count_ = rtp_llm::getMultiProcessorCount();
    gemm_lut_ = get_gemm_lut<T, WeightType>();
}

template <typename T, typename WeightType, cutlass::WeightOnlyQuantOp QuantOp>
CutlassFpAIntBGemmRunner<T, WeightType, QuantOp>::~CutlassFpAIntBGemmRunner()
{
    RTP_LLM_LOG_TRACE(__PRETTY_FUNCTION__);
}

template <typename T, typename WeightType, cutlass::WeightOnlyQuantOp QuantOp>
template <typename EpilogueTag>
void CutlassFpAIntBGemmRunner<T, WeightType, QuantOp>::dispatch_to_arch<EpilogueTag>(const T* A, const WeightType* B,
    const T* weight_scales, const T* weight_zero_points, const T* biases, T* C, int m, int n, int k,
    const int group_size, tc::CutlassGemmConfig gemm_config, char* workspace_ptr, const size_t workspace_bytes,
    cudaStream_t stream, int* occupancy)
{
    RTP_LLM_LOG_TRACE(__PRETTY_FUNCTION__);
    if (sm_ >= 70 && sm_ < 75)
    {
        dispatch_gemm_to_cutlass<T, WeightType, cutlass::arch::Sm70, QuantOp, EpilogueTag>(A, B, weight_scales,
            weight_zero_points, biases, C, m, n, k, group_size, workspace_ptr, workspace_bytes, gemm_config, stream,
            occupancy);
    }
    else if (sm_ >= 75 && sm_ < 80)
    {
        dispatch_gemm_to_cutlass<T, WeightType, cutlass::arch::Sm75, QuantOp, EpilogueTag>(A, B, weight_scales,
            weight_zero_points, biases, C, m, n, k, group_size, workspace_ptr, workspace_bytes, gemm_config, stream,
            occupancy);
    }
    else if (sm_ >= 80 && sm_ <= 90)
    {
        dispatch_gemm_to_cutlass<T, WeightType, cutlass::arch::Sm80, QuantOp, EpilogueTag>(A, B, weight_scales,
            weight_zero_points, biases, C, m, n, k, group_size, workspace_ptr, workspace_bytes, gemm_config, stream,
            occupancy);
    }
    else
    {
        throw std::runtime_error(
            "[TensorRT-LLm Error][CutlassFpAIntBGemmRunner][GEMM Dispatch] Arch unsupported for CUTLASS mixed type "
            "GEMM");
    }
}


template <typename T, typename WeightType, cutlass::WeightOnlyQuantOp QuantOp>
void CutlassFpAIntBGemmRunner<T, WeightType, QuantOp>::gemm(const void* A, const void* B, const void* weight_scales,
    const void* weight_zero_points, const void* biases, void* C, int m, int n, int k, const int group_size,
    tc::CutlassGemmConfig gemmConfig, char* workspace_ptr, const size_t workspace_bytes, cudaStream_t stream)
{
    RTP_LLM_LOG_TRACE(__PRETTY_FUNCTION__);
    if constexpr ((QuantOp == cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS)
        || (QuantOp == cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY))
    {
        dispatch_to_arch<tc::EpilogueOpBias>((const T*) A, (const WeightType*) B, (const T*) weight_scales,
            (const T*) weight_zero_points, (const T*) biases, (T*) C, m, n, k, group_size, gemmConfig, workspace_ptr,
            workspace_bytes, stream, nullptr);
    }
    else
    {
        throw std::runtime_error(
            "Overload with scale, zero and group size only supported for fine grained bias template.");
    }
}

template <typename T, typename WeightType, cutlass::WeightOnlyQuantOp QuantOp>
void CutlassFpAIntBGemmRunner<T, WeightType, QuantOp>::gemm(const void* A, const void* B, const void* weight_scales,
    void* C, int m, int n, int k, tc::CutlassGemmConfig gemmConfig, char* workspace_ptr, const size_t workspace_bytes,
    cudaStream_t stream)
{
    RTP_LLM_LOG_TRACE(__PRETTY_FUNCTION__);

    if constexpr (QuantOp == cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY)
    {
        dispatch_to_arch<tc::EpilogueOpDefault>((const T*) A, (const WeightType*) B, (const T*) weight_scales, nullptr,
            nullptr, (T*) C, m, n, k, k, gemmConfig, workspace_ptr, workspace_bytes, stream, nullptr);
    }
    else
    {
        throw std::runtime_error("Overload with scale only (and no group size) only supported for per column scaling.");
    }
}




template <typename T, typename WeightType, cutlass::WeightOnlyQuantOp QuantOp>
std::vector<tc::CutlassGemmConfig> CutlassFpAIntBGemmRunner<T, WeightType, QuantOp>::getConfigs() const
{
    static constexpr bool is_weight_only = !std::is_same<T, WeightType>::value;
    std::vector<tc::CutlassGemmConfig> candidateConfigs = get_candidate_configs(sm_, is_weight_only, false, false, SPLIT_K_LIMIT);
    return candidateConfigs;
}


template <typename T, typename WeightType, cutlass::WeightOnlyQuantOp QuantOp>
std::vector<tc::CutlassGemmConfig> CutlassFpAIntBGemmRunner<T, WeightType, QuantOp>::getValidConfigs(const void* A, const void* B,
    const void* weight_scales, const void* weight_zero_points, const void* biases, void* C, int m, int n, int k,
    const int group_size, char* workspace_ptr, const size_t workspace_bytes, cudaStream_t stream)
{

    static constexpr bool is_weight_only = !std::is_same<T, WeightType>::value;

    std::vector<tc::CutlassGemmConfig> candidate_configs
        = get_candidate_configs(sm_, is_weight_only, false, false, SPLIT_K_LIMIT);
    std::vector<tc::CutlassGemmConfig> valid_splitk_configs;
    for (int i = 0; i < candidate_configs.size(); i++)
    {
        if (is_valid_split_k_factor(m, n, k, candidate_configs[i], workspace_bytes, is_weight_only))
        {
            valid_splitk_configs.push_back(candidate_configs[i]);
        }
    }
    std::vector<int> occupancies(valid_splitk_configs.size());

    for (size_t ii = 0; ii < valid_splitk_configs.size(); ++ii)
    {
        dispatch_to_arch<tc::EpilogueOpDefault>((const T*) A, (const WeightType*) B, (const T*) weight_scales,
            (const T*) weight_zero_points, (const T*) biases, (T*) C, m, n, k, group_size, valid_splitk_configs[ii],
            workspace_ptr, workspace_bytes, stream, &(occupancies[ii]));
    }
    std::vector<tc::CutlassGemmConfig> valid_configs
        = get_valid_config_from_occupancies(valid_splitk_configs, occupancies);

    return valid_configs;

}

template <typename T, typename WeightType, cutlass::WeightOnlyQuantOp QuantOp>
tc::CutlassGemmConfig CutlassFpAIntBGemmRunner<T, WeightType, QuantOp>::getChosenConfig(const void* A, const void* B,
    const void* weight_scales, const void* weight_zero_points, const void* biases, void* C, int m, int n, int k,
    const int group_size, char* workspace_ptr, const size_t workspace_bytes, cudaStream_t stream)
{
    // Standard GEMM, so 1 "expert". We use the same function for MoE and regular FFN.
    RTP_LLM_LOG_TRACE(__PRETTY_FUNCTION__);

    GemmParamKey cur_key{m, n, k};
    if (gemm_lut_ != nullptr)
    {
        auto iter = gemm_lut_->find({cur_key});
        if (iter != gemm_lut_->end())
        {
            CutlassGemmConfig specified_config = iter->second;
            return specified_config;
        }
    }

    static constexpr bool is_weight_only = !std::is_same<T, WeightType>::value;

    std::vector<tc::CutlassGemmConfig> candidate_configs
        = get_candidate_configs(sm_, is_weight_only, false, false, SPLIT_K_LIMIT);
    std::vector<tc::CutlassGemmConfig> valid_configs;
    for (int i = 0; i < candidate_configs.size(); i++)
    {
        if (is_valid_split_k_factor(m, n, k, candidate_configs[i], workspace_bytes, is_weight_only))
        {
            valid_configs.push_back(candidate_configs[i]);
        }
    }
    std::vector<int> occupancies(valid_configs.size());

    for (size_t ii = 0; ii < valid_configs.size(); ++ii)
    {
        dispatch_to_arch<tc::EpilogueOpDefault>((const T*) A, (const WeightType*) B, (const T*) weight_scales,
            (const T*) weight_zero_points, (const T*) biases, (T*) C, m, n, k, group_size, valid_configs[ii],
            workspace_ptr, workspace_bytes, stream, &(occupancies[ii]));
    }
    tc::CutlassGemmConfig chosen_config
        = estimate_best_config_from_occupancies(valid_configs, occupancies, m, n, k, multi_processor_count_);

    return chosen_config;
}

template <typename T, typename WeightType, cutlass::WeightOnlyQuantOp QuantOp>
size_t CutlassFpAIntBGemmRunner<T, WeightType, QuantOp>::getWorkspaceSize(const int m, const int n, const int k)
{
    RTP_LLM_LOG_TRACE(__PRETTY_FUNCTION__);
    // These are the min tile sizes for each config, which would launch the maximum number of blocks
    const int max_grid_m = cutlass::ceil_div(m, MIN_M_TILE);
    const int max_grid_n = cutlass::ceil_div(n, MIN_N_TILE);
    // We need 4 bytes per block in the worst case. We launch split_k_limit in z dim.
    return static_cast<size_t>(max_grid_m * max_grid_n * SPLIT_K_LIMIT * 4);
}

template <typename WeightType, cutlass::WeightOnlyQuantOp QuantOp>
void CutlassFpAIntBGemmRunner<float, WeightType, QuantOp>::gemm(const void* A, const void* B, const void* weight_scales,
    const void* weight_zero_points, const void* biases, void* C, int m, int n, int k, const int group_size,
    tc::CutlassGemmConfig gemmConfig, char* workspace_ptr, const size_t workspace_bytes, cudaStream_t stream)
{
    RTP_LLM_LOG_TRACE(__PRETTY_FUNCTION__);
}

template <typename WeightType, cutlass::WeightOnlyQuantOp QuantOp>
void CutlassFpAIntBGemmRunner<float, WeightType, QuantOp>::gemm(const void* A, const void* B, const void* weight_scales,
    void* C, int m, int n, int k, tc::CutlassGemmConfig gemmConfig, char* workspace_ptr, const size_t workspace_bytes,
    cudaStream_t stream)
{
    RTP_LLM_LOG_TRACE(__PRETTY_FUNCTION__);
}

template <typename WeightType, cutlass::WeightOnlyQuantOp QuantOp>
size_t CutlassFpAIntBGemmRunner<float, WeightType, QuantOp>::getWorkspaceSize(const int m, const int n, const int k)
{
    RTP_LLM_LOG_TRACE(__PRETTY_FUNCTION__);
    return 0;
}

template <typename WeightType, cutlass::WeightOnlyQuantOp QuantOp>
std::vector<tc::CutlassGemmConfig> CutlassFpAIntBGemmRunner<float, WeightType, QuantOp>::getConfigs() const
{
    RTP_LLM_LOG_TRACE(__PRETTY_FUNCTION__);
    std::vector<tc::CutlassGemmConfig> configs={};
    return configs;
}

template <typename WeightType, cutlass::WeightOnlyQuantOp QuantOp>
tc::CutlassGemmConfig CutlassFpAIntBGemmRunner<float, WeightType, QuantOp>::getChosenConfig(void const* A,
    void const* B, void const* weight_scales, void const* weight_zero_points, void const* biases, void* C, int m, int n,
    int k, int const group_size, char* workspace_ptr, size_t const workspace_bytes, cudaStream_t stream)
{
    RTP_LLM_LOG_TRACE(__PRETTY_FUNCTION__);
    tc::CutlassGemmConfig config;
    return config;

}

} // namespace cutlass_kernels
} // namespace kernels
} // namespace tensorrt_llm
