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

// clang-format off
#include <cutlass/gemm/device/default_gemm_configuration.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass_extensions/gemm/device/gemm_universal_base_compat.h>
#include <cutlass/gemm/kernel/default_gemm.h>
#include <cutlass/epilogue/threadblock/epilogue_with_visitor.h>
#include <cutlass/epilogue/thread/linear_combination_generic.h>
// clang-format on

#include "cutlass_extensions/compute_occupancy.h"
#include "cutlass_extensions/epilogue/threadblock/epilogue_per_row_per_col_scale.h"
#include "cutlass_extensions/epilogue/threadblock/epilogue_tensor_op_int32.h"
#include "cutlass_extensions/epilogue_helpers.h"
#include "rtp_llm/cpp/cuda/cutlass/cutlass_kernels/gemm_configs.h"

#include "cutlass_extensions/gemm/kernel/default_int8_traits.h"
#include "cutlass_extensions/gemm/kernel/gemm_with_epilogue_visitor.h"

#ifndef _WIN32
#pragma GCC diagnostic pop
#endif // #ifndef _WIN32

#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#include "rtp_llm/cpp/cuda/trt_utils.h"
#include "rtp_llm/cpp/cuda/cutlass/cutlass_kernels/cutlass_heuristic.h"
#include "rtp_llm/cpp/cuda/cutlass/cutlass_kernels/int8_gemm/int8_gemm.h"

#include <chrono>
#include <sstream>

namespace tk = tensorrt_llm::common;
namespace tc = tensorrt_llm::cutlass_extensions;


namespace tensorrt_llm
{
namespace kernels
{
namespace cutlass_kernels
{

template <typename T, typename arch, template<typename T1> class ActivationFunctor, typename ThreadblockShape, typename WarpShape, int Stages>
void genericInt8GemmKernelLauncher(const int8_t* A, const int8_t* B, tk::QuantMode quantOption, const float* alphaCol,
    const float* alphaRow, T* C, T* bias, int m, int n, int k, tc::CutlassGemmConfig gemmConfig, char* workspace,
    size_t workspaceBytes, cudaStream_t stream, int* occupancy = nullptr)
{
    TLLM_LOG_TRACE(__PRETTY_FUNCTION__);

    using ElementInput = int8_t;

    using ElementOutput_ =
        typename cutlass::platform::conditional<cutlass::platform::is_same<T, half>::value, cutlass::half_t, T>::type;
#ifdef ENABLE_BF16
    using ElementOutput =
        typename cutlass::platform::conditional<cutlass::platform::is_same<ElementOutput_, __nv_bfloat16>::value,
            cutlass::bfloat16_t, ElementOutput_>::type;
#else
    using ElementOutput = ElementOutput_;
#endif

    using ElementAccumulator = int32_t;
    using ElementCompute = float;

    using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

    using OperatorClass = typename cutlass::gemm::kernel::Int8GemmArchTraits<arch>::OperatorClass;
    using InstructionShape = typename cutlass::gemm::kernel::Int8GemmArchTraits<arch>::InstructionShape;

    using DefaultGemmConf = typename cutlass::gemm::device::DefaultGemmConfiguration<OperatorClass, arch, ElementInput,
        ElementInput, ElementOutput, ElementCompute>;
    using GemmOp = typename DefaultGemmConf::Operator;
    using EpilogueCompute =
        typename cutlass::platform::conditional<cutlass::platform::is_same<T, int32_t>::value, ElementCompute, ElementOutput>::type;

    using EpilogueOp = cutlass::epilogue::thread::LinearCombinationGeneric<ActivationFunctor, ElementOutput, DefaultGemmConf::EpilogueOutputOp::kCount, EpilogueCompute, EpilogueCompute, cutlass::epilogue::thread::ScaleType::NoBetaScaling>;

    // only TN is supported (s8 * s8 + s32)
    using GemmKernel_ = typename cutlass::gemm::kernel::DefaultGemm<ElementInput, cutlass::layout::RowMajor,
        DefaultGemmConf::kAlignmentA, ElementInput, cutlass::layout::ColumnMajor, DefaultGemmConf::kAlignmentB,
        ElementOutput, cutlass::layout::RowMajor, ElementAccumulator, OperatorClass, arch, ThreadblockShape, WarpShape,
        InstructionShape, EpilogueOp, ThreadblockSwizzle, Stages, true, GemmOp>::GemmKernel;

    using AlphaColTileIterator = cutlass::epilogue::threadblock::PredicatedTileIterator<
        cutlass::epilogue::threadblock::OutputTileOptimalThreadMap<
            typename GemmKernel_::Epilogue::OutputTileIterator::ThreadMap::Shape,
            typename GemmKernel_::Epilogue::OutputTileIterator::ThreadMap::Count,
            GemmKernel_::Epilogue::OutputTileIterator::ThreadMap::kThreads,
            GemmKernel_::Epilogue::OutputTileIterator::kElementsPerAccess, cutlass::sizeof_bits<ElementOutput>::value>,
        ElementCompute>;

    // Epilogue visitor
    using EpilogueVisitor = typename cutlass::epilogue::threadblock::EpilogueVisitorPerRowPerCol<ThreadblockShape,
        GemmKernel_::kThreadCount, AlphaColTileIterator, typename GemmKernel_::Epilogue::OutputTileIterator,
        ElementAccumulator, ElementCompute, EpilogueOp, EpilogueCompute>;

    /// Epilogue
    using Epilogue = typename cutlass::epilogue::threadblock::EpilogueWithVisitorFromExistingEpilogue<EpilogueVisitor,
        typename GemmKernel_::Epilogue>::Epilogue;

    // GEMM
    using GemmKernel
        = cutlass::gemm::kernel::GemmWithEpilogueVisitor<typename GemmKernel_::Mma, Epilogue, ThreadblockSwizzle>;

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

    typename EpilogueOp::Params linearScalingParams; // use default since no beta and gamma
    typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kBatched, {m, n, k}, 1,
        {reinterpret_cast<ElementInput*>(const_cast<ElementInput*>(A)), k},
        {reinterpret_cast<ElementInput*>(const_cast<ElementInput*>(B)), k}, quantOption,
        {reinterpret_cast<ElementCompute*>(const_cast<float*>(alphaCol)), 0},
        {reinterpret_cast<ElementCompute*>(const_cast<float*>(alphaRow)), 0}, {nullptr, 0},
        {reinterpret_cast<ElementOutput*>(C), n},
        {reinterpret_cast<ElementOutput*>(bias), 0}, 0, 0,
        typename EpilogueVisitor::Arguments(linearScalingParams, 0, 0, 0)};

    Gemm gemm;
    // TODO: handle that
    if (gemm.get_workspace_size(args) > workspaceBytes)
    {
        TLLM_LOG_WARNING(
            "Requested split-k but workspace size insufficient. Falling back to non-split-k implementation.");
        // If requested split-k factor will require more workspace bytes, revert to standard gemm.
        args.batch_count = 1;
    }

    auto can_implement = gemm.can_implement(args);
    if (can_implement != cutlass::Status::kSuccess)
    {
        std::string errMsg = "int8gemm cutlass kernel will fail for params. Error: "
            + std::string(cutlassGetStatusString(can_implement));
        throw std::runtime_error("[TensorRT-LLM Error][int8gemm Runner] " + errMsg);
    }

    auto initStatus = gemm.initialize(args, workspace, stream);
    if (initStatus != cutlass::Status::kSuccess)
    {
        std::string errMsg
            = "Failed to initialize cutlass int8 gemm. Error: " + std::string(cutlassGetStatusString(initStatus));
        throw std::runtime_error("[TensorRT-LLM Error][int8gemm Runner] " + errMsg);
    }

    auto runStatus = gemm.run(stream);
    if (runStatus != cutlass::Status::kSuccess)
    {
        std::string errMsg
            = "Failed to run cutlass int8 gemm. Error: " + std::string(cutlassGetStatusString(runStatus));
        throw std::runtime_error("[TensorRT-LLM Error][int8gemm Runner] " + errMsg);
    }
}

template <typename T, typename arch, template<typename T1> class ActivationFunctor, typename ThreadblockShape, typename WarpShape, int Stages, typename Enable = void>
struct dispatchStages
{
    static void dispatch(const int8_t* A, const int8_t* B, tk::QuantMode quantOption, const float* alphaCol,
        const float* alphaRow, T* C, T* bias, int m, int n, int k, tc::CutlassGemmConfig gemmConfig, char* workspace,
        size_t workspaceBytes, cudaStream_t stream, int* occupancy = nullptr)
    {
        TLLM_LOG_TRACE(__PRETTY_FUNCTION__);
        std::string errMsg = "Cutlass int8 gemm. Not instantiates for arch "
            + std::to_string(arch::kMinComputeCapability) + " with stages set to " + std::to_string(Stages);
        throw std::runtime_error("[TensorRT-LLM Error][dispatchStages::dispatch] " + errMsg);
    }
};

template <typename T, typename arch, template<typename T1> class ActivationFunctor, typename ThreadblockShape, typename WarpShape>
struct dispatchStages<T, arch, ActivationFunctor, ThreadblockShape, WarpShape, 2>
{
    static void dispatch(const int8_t* A, const int8_t* B, tk::QuantMode quantOption, const float* alphaCol,
        const float* alphaRow, T* C, T* bias, int m, int n, int k, tc::CutlassGemmConfig gemmConfig, char* workspace,
        size_t workspaceBytes, cudaStream_t stream, int* occupancy = nullptr)
    {
        TLLM_LOG_TRACE(__PRETTY_FUNCTION__);
        genericInt8GemmKernelLauncher<T, arch, ActivationFunctor, ThreadblockShape, WarpShape, 2>(A, B, quantOption, alphaCol, alphaRow, C,
            bias, m, n, k, gemmConfig, workspace, workspaceBytes, stream, occupancy);
    }
};

template <typename T, template<typename T1> class ActivationFunctor, typename ThreadblockShape, typename WarpShape, int Stages>
struct dispatchStages<T, cutlass::arch::Sm80, ActivationFunctor, ThreadblockShape, WarpShape, Stages,
    typename std::enable_if<(Stages > 2)>::type>
{
    static void dispatch(const int8_t* A, const int8_t* B, tk::QuantMode quantOption, const float* alphaCol,
        const float* alphaRow, T* C, T* bias, int m, int n, int k, tc::CutlassGemmConfig gemmConfig, char* workspace,
        size_t workspaceBytes, cudaStream_t stream, int* occupancy = nullptr)
    {

        TLLM_LOG_TRACE(__PRETTY_FUNCTION__);
        genericInt8GemmKernelLauncher<T, cutlass::arch::Sm80, ActivationFunctor, ThreadblockShape, WarpShape, Stages>(A, B, quantOption,
            alphaCol, alphaRow, C, bias, m, n, k, gemmConfig, workspace, workspaceBytes, stream, occupancy);
    }
};

template <typename T, typename arch, template<typename T1> class ActivationFunctor, typename ThreadblockShape, typename WarpShape>
void dispatchGemmConfig(const int8_t* A, const int8_t* B, tk::QuantMode quantOption, const float* alphaCol,
    const float* alphaRow, T* C, T* bias, int m, int n, int k, tc::CutlassGemmConfig gemmConfig, char* workspace,
    size_t workspaceBytes, cudaStream_t stream, int* occupancy = nullptr)
{

    TLLM_LOG_TRACE(__PRETTY_FUNCTION__);
    switch (gemmConfig.stages)
    {
    case 2:
        using DispatcherStages2 = dispatchStages<T, arch, ActivationFunctor, ThreadblockShape, WarpShape, 2>;
        DispatcherStages2::dispatch(A, B, quantOption, alphaCol, alphaRow, C, bias, m, n, k, gemmConfig, workspace,
            workspaceBytes, stream, occupancy);
        break;
    case 3:
        using DispatcherStages3 = dispatchStages<T, arch, ActivationFunctor, ThreadblockShape, WarpShape, 3>;
        DispatcherStages3::dispatch(A, B, quantOption, alphaCol, alphaRow, C, bias, m, n, k, gemmConfig, workspace,
            workspaceBytes, stream, occupancy);
        break;
    case 4:
        using DispatcherStages4 = dispatchStages<T, arch, ActivationFunctor, ThreadblockShape, WarpShape, 4>;
        DispatcherStages4::dispatch(A, B, quantOption, alphaCol, alphaRow, C, bias, m, n, k, gemmConfig, workspace,
            workspaceBytes, stream, occupancy);
        break;
    case 5:
        using DispatcherStages5 = dispatchStages<T, arch, ActivationFunctor, ThreadblockShape, WarpShape, 5>;
        DispatcherStages5::dispatch(A, B, quantOption, alphaCol, alphaRow, C, bias, m, n, k, gemmConfig, workspace,
            workspaceBytes, stream, occupancy);
        break;
    case 6:
        using DispatcherStages6 = dispatchStages<T, arch, ActivationFunctor, ThreadblockShape, WarpShape, 6>;
        DispatcherStages6::dispatch(A, B, quantOption, alphaCol, alphaRow, C, bias, m, n, k, gemmConfig, workspace,
            workspaceBytes, stream, occupancy);
        break;
    default:
        std::string errMsg = "dispatchGemmConfig does not support stages " + std::to_string(gemmConfig.stages);
        throw std::runtime_error("[TensorRT-LLM Error][dispatch_gemm_config] " + errMsg);
        break;
    }
}

template <typename T, typename arch, template<typename T1> class ActivationFunctor>
void dispatchGemmToCutlass(const int8_t* A, const int8_t* B, tk::QuantMode quantOption, const float* alphaCol,
    const float* alphaRow, T* C, T* bias, int m, int n, int k, char* workspace, size_t workspaceBytes,
    tc::CutlassGemmConfig gemmConfig, cudaStream_t stream, int* occupancy = nullptr)
{

    TLLM_LOG_TRACE(__PRETTY_FUNCTION__);

    switch (gemmConfig.tile_config)
    {
    case tc::CutlassTileConfig::CtaShape128x64x64_WarpShape64x32x64:
        dispatchGemmConfig<T, arch, ActivationFunctor, cutlass::gemm::GemmShape<128, 128, 64>, cutlass::gemm::GemmShape<64, 32, 64>>(A, B,
            quantOption, alphaCol, alphaRow, C, bias, m, n, k, gemmConfig, workspace, workspaceBytes, stream, occupancy);
        break;
    case tc::CutlassTileConfig::CtaShape256x128x64_WarpShape64x64x64:
        dispatchGemmConfig<T, arch, ActivationFunctor, cutlass::gemm::GemmShape<256, 128, 64>, cutlass::gemm::GemmShape<64, 64, 64>>(A, B,
            quantOption, alphaCol, alphaRow, C, bias, m, n, k, gemmConfig, workspace, workspaceBytes, stream, occupancy);
        break;
    case tc::CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64:
        dispatchGemmConfig<T, arch, ActivationFunctor, cutlass::gemm::GemmShape<32, 128, 64>, cutlass::gemm::GemmShape<32, 32, 64>>(A, B,
            quantOption, alphaCol, alphaRow, C, bias, m, n, k, gemmConfig, workspace, workspaceBytes, stream, occupancy);
        break;
    case tc::CutlassTileConfig::CtaShape64x128x64_WarpShape64x32x64:
        dispatchGemmConfig<T, arch, ActivationFunctor, cutlass::gemm::GemmShape<64, 128, 64>, cutlass::gemm::GemmShape<64, 32, 64>>(A, B,
            quantOption, alphaCol, alphaRow, C, bias, m, n, k, gemmConfig, workspace, workspaceBytes, stream, occupancy);
        break;
    case tc::CutlassTileConfig::CtaShape64x64x128_WarpShape32x64x64:
        dispatchGemmConfig<T, arch, ActivationFunctor, cutlass::gemm::GemmShape<64, 64, 128>, cutlass::gemm::GemmShape<32, 64, 64>>(A, B,
            quantOption, alphaCol, alphaRow, C, bias, m, n, k, gemmConfig, workspace, workspaceBytes, stream, occupancy);
        break;
    case tc::CutlassTileConfig::CtaShape128x256x64_WarpShape64x64x64:
        dispatchGemmConfig<T, arch, ActivationFunctor, cutlass::gemm::GemmShape<128, 256, 64>, cutlass::gemm::GemmShape<64, 64, 64>>(A, B,
            quantOption, alphaCol, alphaRow, C, bias, m, n, k, gemmConfig, workspace, workspaceBytes, stream, occupancy);
        break;
    case tc::CutlassTileConfig::Undefined:
        throw std::runtime_error("[TensorRT-LLM Error][int8][dispatch_gemm_to_cutlass] gemm config undefined.");
        break;
    case tc::CutlassTileConfig::ChooseWithHeuristic:
        throw std::runtime_error(
            "[TensorRT-LLM Error][int8][dispatch_gemm_to_cutlass] gemm config should have already been set by "
            "heuristic.");
        break;
    default:
        throw std::runtime_error(
            "[TensorRT-LLM Error][int8][dispatch_gemm_to_cutlass] Config is invalid for int8 GEMM.");
        break;
    }
}

template <typename T, typename arch>
void dispatchGemmActivationFunc(const int8_t* A, const int8_t* B, tk::QuantMode quantOption, const float* alphaCol,
    const float* alphaRow, T* C, T* bias, CutlassActivationType activation, int m, int n, int k, char* workspace, size_t workspaceBytes,
    tc::CutlassGemmConfig gemmConfig, cudaStream_t stream, int* occupancy = nullptr) {
    TLLM_LOG_TRACE(__PRETTY_FUNCTION__);
    switch (activation) {
        case CutlassActivationType::IDENTITY:
            dispatchGemmToCutlass<T, arch, cutlass::epilogue::thread::Identity>(A, B, quantOption, alphaCol, alphaRow, C, bias, m, n, k, workspace,
                workspaceBytes, gemmConfig, stream, occupancy);
            break;
        case CutlassActivationType::GELU_FAST:
            dispatchGemmToCutlass<T, arch, cutlass::epilogue::thread::GELU_taylor>(A, B, quantOption, alphaCol, alphaRow, C, bias, m, n, k, workspace,
                workspaceBytes, gemmConfig, stream, occupancy);
            break;
        default:
        throw std::runtime_error(
            "[TensorRT-LLM Error][int8][dispatchGemmActivationFunc] Config is invalid for int8 GEMM.");
        break;
    }
}


template <typename T>
CutlassInt8GemmRunner<T>::CutlassInt8GemmRunner()
{
    TLLM_LOG_TRACE(__PRETTY_FUNCTION__);
    mSm = rtp_llm::get_sm();
    mMultiProcessorCount = rtp_llm::getMultiProcessorCount();
    gemm_lut_ = get_gemm_lut<uint8_t, uint8_t>();
}

template <typename T>
CutlassInt8GemmRunner<T>::~CutlassInt8GemmRunner()
{
    TLLM_LOG_TRACE(__PRETTY_FUNCTION__);
}

template <typename T>
void CutlassInt8GemmRunner<T>::dispatchToArch(const int8_t* A, const int8_t* B, tk::QuantMode quantOption,
    const float* alphaCol, const float* alphaRow, T* C, T* bias, CutlassActivationType activation, int m, int n, int k, tc::CutlassGemmConfig gemmConfig,
    char* workspacePtr, const size_t workspaceBytes, cudaStream_t stream, int* occupancy)
{
    TLLM_LOG_TRACE(__PRETTY_FUNCTION__) ;
    if (mSm >= 70 && mSm < 72)
    {
        dispatchGemmActivationFunc<T, cutlass::arch::Sm70>(A, B, quantOption, alphaCol, alphaRow, C, bias, activation, m, n, k, workspacePtr,
            workspaceBytes, gemmConfig, stream, occupancy);
    }
    else if (mSm >= 72 && mSm < 75)
    {
        dispatchGemmActivationFunc<T, cutlass::arch::Sm72>(A, B, quantOption, alphaCol, alphaRow, C, bias, activation, m, n, k, workspacePtr,
            workspaceBytes, gemmConfig, stream, occupancy);
    }
    else if (mSm >= 75 && mSm < 80)
    {
        dispatchGemmActivationFunc<T, cutlass::arch::Sm75>(A, B, quantOption, alphaCol, alphaRow, C, bias, activation, m, n, k, workspacePtr,
            workspaceBytes, gemmConfig, stream, occupancy);
    }
    else if (mSm >= 80 && mSm <= 90)
    {
        dispatchGemmActivationFunc<T, cutlass::arch::Sm80>(A, B, quantOption, alphaCol, alphaRow, C, bias, activation, m, n, k, workspacePtr,
            workspaceBytes, gemmConfig, stream, occupancy);
    }
    else
    {
        throw std::runtime_error(
            "[TensorRT-LLM Error][CutlassInt8GemmRunner][GEMM Dispatch] Arch unsupported for CUTLASS int8 GEMM");
    }
}

template <typename T>
void CutlassInt8GemmRunner<T>::gemm(const void* A, const void* B, tk::QuantMode quantOption, const float* alphaCol,
    const float* alphaRow, void* C, void* bias, CutlassActivationType activation_type, int m, int n, int k, tc::CutlassGemmConfig gemmConfig, char* workspacePtr,
    const size_t workspaceBytes, cudaStream_t stream)
{
    TLLM_LOG_TRACE(__PRETTY_FUNCTION__);

    dispatchToArch(reinterpret_cast<const int8_t*>(A), reinterpret_cast<const int8_t*>(B), quantOption, alphaCol,
        alphaRow, reinterpret_cast<T*>(C), (T*)bias, activation_type, m, n, k, gemmConfig, workspacePtr, workspaceBytes, stream);
}

template <typename T>
std::vector<tc::CutlassGemmConfig> CutlassInt8GemmRunner<T>::getConfigs() const
{
    static constexpr bool isWeightOnly = false;
    std::vector<tc::CutlassGemmConfig> candidateConfigs
        = get_candidate_configs(mSm, isWeightOnly, false, /* SIMT configs */
            true, SPLIT_K_LIMIT);                             /* INT8 configs */
    return candidateConfigs;
}

template <typename T>
std::vector<tc::CutlassGemmConfig> CutlassInt8GemmRunner<T>::getValidConfigs(const void* A, const void* B,
    tk::QuantMode quantOption, const float* alphaCol, const float* alphaRow, void* C, int m, int n, int k,
    char* workspacePtr, const size_t workspaceBytes, cudaStream_t stream)
{
    // Standard GEMM, so 1 "expert". We use the same function for MoE and regular FFN.
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);

    std::vector<tc::CutlassGemmConfig> candidate_configs = getConfigs();
    std::vector<tc::CutlassGemmConfig> valid_splitk_configs;
    for (int i = 0; i < candidate_configs.size(); i++)
    {
        if (is_valid_split_k_factor(m, n, k, candidate_configs[i], workspaceBytes, false))
        {
            valid_splitk_configs.push_back(candidate_configs[i]);
        }
    }
    std::vector<int> occupancies(valid_splitk_configs.size());

    for (size_t ii = 0; ii < valid_splitk_configs.size(); ++ii)
    {
        dispatchToArch(reinterpret_cast<const int8_t*>(A), reinterpret_cast<const int8_t*>(B), quantOption, alphaCol,
            alphaRow, reinterpret_cast<T*>(C), (T*)nullptr, CutlassActivationType::IDENTITY, m, n, k, valid_splitk_configs[ii], workspacePtr, workspaceBytes, stream,
            &(occupancies[ii]));
    }
    std::vector<tc::CutlassGemmConfig> valid_configs
        = get_valid_config_from_occupancies(valid_splitk_configs, occupancies);

    return valid_configs;
}

template <typename T>
tc::CutlassGemmConfig CutlassInt8GemmRunner<T>::getChosenConfig(const void* A, const void* B, tk::QuantMode quantOption,
        const float* alphaCol, const float* alphaRow, void* C, int m, int n, int k, char* workspacePtr,
        const size_t workspaceBytes, cudaStream_t stream)
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

    std::vector<tc::CutlassGemmConfig> candidate_configs = getConfigs();
    std::vector<tc::CutlassGemmConfig> valid_configs;
    for (int i = 0; i < candidate_configs.size(); i++)
    {
        if (is_valid_split_k_factor(
                m, n, k, candidate_configs[i], workspaceBytes, false))
        {
            valid_configs.push_back(candidate_configs[i]);
        }
    }
    std::vector<int> occupancies(valid_configs.size());

    for (size_t ii = 0; ii < valid_configs.size(); ++ii)
    {
        dispatchToArch(reinterpret_cast<const int8_t*>(A), reinterpret_cast<const int8_t*>(B), quantOption, alphaCol,
            alphaRow, reinterpret_cast<T*>(C), nullptr, CutlassActivationType::IDENTITY, m, n, k, valid_configs[ii], workspacePtr, workspaceBytes, stream,
            &(occupancies[ii]));
    }
    tc::CutlassGemmConfig chosen_config = estimate_best_config_from_occupancies(valid_configs, occupancies, m, n, k,
        mMultiProcessorCount);

    return chosen_config;
}

template <typename T>
size_t CutlassInt8GemmRunner<T>::getWorkspaceSize(const int m, const int n, const int k)
{
    TLLM_LOG_TRACE(__PRETTY_FUNCTION__);
    // These are the min tile sizes for each config, which would launch the maximum number of blocks
    const int maxGridM = cutlass::ceil_div(m, MIN_M_TILE);
    const int maxGridN = cutlass::ceil_div(n, MIN_N_TILE);
    // We need 4 bytes per block in the worst case. We launch SPLIT_K_LIMIT in z dim.
    return static_cast<size_t>(maxGridM * maxGridN * SPLIT_K_LIMIT * 4);
}

} // namespace cutlass_kernels
} // namespace kernels
} // namespace tensorrt_llm
