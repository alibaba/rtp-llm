/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <algorithm>
#include <cuda.h>
#include <cuda_fp16.h>
#include <float.h>
#include <math.h>
#include <numeric>
#include <random>
#include <sstream>

// Ignore CUTLASS warnings about type punning
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"

#include "cute/tensor.hpp"
#include "cutlass/conv/convolution.h"
// Order matters here, packed_stride.hpp is missing cute and convolution includes
#include "cutlass/util/packed_stride.hpp"

#include "cutlass/array.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"

#include "cutlass_extensions/epilogue/thread/fused_activations.h"

#pragma GCC diagnostic pop

#include "rtp_llm/cpp/cuda/trt_utils.h"
#include "rtp_llm/cpp/cuda/cutlass/cutlass_kernels/moe_gemm/moe_gemm_kernels.h"
#include "rtp_llm/cpp/cuda/cutlass/cutlass_kernels/moe_gemm/moe_kernels.h"

#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#elif (CUDART_VERSION >= 11050)
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/util_type.cuh>
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>
#else
#include "3rdparty/cub/cub.cuh"
#include "3rdparty/cub/device/device_radix_sort.cuh"
#include "3rdparty/cub/util_type.cuh"
#endif

using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::common;

namespace tensorrt_llm::kernels
{

// TODO Could linear search be better for small # experts
template <class T>
__device__ inline int64_t findTotalEltsLessThanTarget(T const* sorted_indices, int64_t const arr_length, T const target)
{
    int64_t low = 0, high = arr_length - 1, target_location = -1;
    while (low <= high)
    {
        int64_t mid = (low + high) / 2;

        if (sorted_indices[mid] >= target)
        {
            high = mid - 1;
        }
        else
        {
            low = mid + 1;
            target_location = mid;
        }
    }
    return target_location + 1;
}

void computeExpertFirstTokenOffset(int const* sorted_indices, int const total_indices, int const num_experts,
    int64_t* expert_first_token_offset, cudaStream_t stream);

float const** computeFP8DequantScale(
    float const** alpha_scale_ptr_array, int const num_experts, float const* fp8_dequant, cudaStream_t stream);

// ========================== Permutation things =======================================

template <class T, class U>
__host__ __device__ constexpr static U arrayConvert(T const& input)
{
    using Type = typename U::Element;
    static_assert(T::kElements == U::kElements);
    U u;
#pragma unroll
    for (int i = 0; i < U::kElements; i++)
    {
        u[i] = static_cast<Type>(input[i]);
    }
    return u;
}

// Duplicated and permutes rows for MoE. In addition, reverse the permutation map to help with finalizing routing.

// "expanded_x_row" simply means that the number of values is num_rows x k. It is "expanded" since we will have to
// duplicate some rows in the input matrix to match the dimensions. Duplicates will always get routed to separate
// experts in the end.

// Note that the expanded_dest_row_to_expanded_source_row map referred to here has indices in the range (0,
// k*rows_in_input - 1). However, it is set up so that index 0, rows_in_input, 2*rows_in_input ... (k-1)*rows_in_input
// all map to row 0 in the original matrix. Thus, to know where to read in the source matrix, we simply take the modulus
// of the expanded index.

constexpr static int EXPAND_THREADS_PER_BLOCK = 256;

template <typename T, bool CHECK_SKIPPED>
__global__ void expandInputRowsKernel(T const* unpermuted_input, T* permuted_output, float* unpermuted_scales,
    float* permuted_scales, int const* expanded_dest_row_to_expanded_source_row,
    int* expanded_source_row_to_expanded_dest_row, int64_t const num_rows, int64_t const* num_dest_rows,
    int64_t const cols, int64_t k)
{

    // Reverse permutation map.
    // I do this so that later, we can use the source -> dest map to do the k-way reduction and unpermuting. I need the
    // reverse map for that reduction to allow each threadblock to do 1 k-way reduce without atomics later in MoE. 1
    // thread block will be responsible for all k summations.
    int64_t const expanded_dest_row = blockIdx.x;
    int64_t const expanded_source_row = expanded_dest_row_to_expanded_source_row[expanded_dest_row];
    if (threadIdx.x == 0)
    {
        assert(expanded_dest_row <= INT32_MAX);
        expanded_source_row_to_expanded_dest_row[expanded_source_row] = static_cast<int>(expanded_dest_row);
    }

    if (!CHECK_SKIPPED || blockIdx.x < *num_dest_rows)
    {
        // Load 128-bits per thread
        constexpr int64_t ELEM_PER_THREAD = 128 / cutlass::sizeof_bits<T>::value;
        using DataElem = cutlass::Array<T, ELEM_PER_THREAD>;

        // Duplicate and permute rows
        int64_t const source_k_rank = expanded_source_row / num_rows;
        int64_t const source_row = expanded_source_row % num_rows;

        auto const* source_row_ptr = reinterpret_cast<DataElem const*>(unpermuted_input + source_row * cols);
        auto* dest_row_ptr = reinterpret_cast<DataElem*>(permuted_output + expanded_dest_row * cols);

        int64_t const start_offset = threadIdx.x;
        int64_t const stride = EXPAND_THREADS_PER_BLOCK;
        int64_t const num_elems_in_col = cols / ELEM_PER_THREAD;

        for (int elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride)
        {
            dest_row_ptr[elem_index] = source_row_ptr[elem_index];
        }

        if (permuted_scales && threadIdx.x == 0)
        {
            int64_t const source_k_idx = source_row * k + source_k_rank;
            permuted_scales[expanded_dest_row] = unpermuted_scales[source_k_idx];
        }
    }
}

template <typename T>
void expandInputRowsKernelLauncher(T const* unpermuted_input, T* permuted_output, float* unpermuted_scales,
    float* permuted_scales, int const* expanded_dest_row_to_expanded_source_row,
    int* expanded_source_row_to_expanded_dest_row, int64_t const num_rows, int64_t const* num_valid_tokens_ptr,
    int64_t const cols, int const k, cudaStream_t stream)
{
    int64_t const blocks = num_rows * k;
    int64_t const threads = EXPAND_THREADS_PER_BLOCK;
    auto func = (num_valid_tokens_ptr != nullptr) ? expandInputRowsKernel<T, true> : expandInputRowsKernel<T, false>;
    func<<<blocks, threads, 0, stream>>>(unpermuted_input, permuted_output, unpermuted_scales, permuted_scales,
        expanded_dest_row_to_expanded_source_row, expanded_source_row_to_expanded_dest_row, num_rows,
        num_valid_tokens_ptr, cols, k);
}

enum class ScaleMode : int
{
    NO_SCALE = 0,
    DEFAULT = 1,
    RENORM_SCALE = 2,
};

constexpr static int FINALIZE_THREADS_PER_BLOCK = 256;

// Final kernel to unpermute and scale
// This kernel unpermutes the original data, does the k-way reduction and performs the final skip connection.
template <typename T, typename OutputType, class GemmOutputType, class ScaleBiasType, ScaleMode SCALE_MODE,
    bool CHECK_SKIPPED>
__global__ void finalizeMoeRoutingKernel(GemmOutputType const* expanded_permuted_rows,
    OutputType* reduced_unpermuted_output, ScaleBiasType const* bias, float const* scales,
    int const* expanded_source_row_to_expanded_dest_row, int const* expert_for_source_row, int64_t const orig_cols,
    int64_t const k, int64_t const* num_valid_ptr)
{
    assert(orig_cols % 4 == 0);
    int64_t const original_row = blockIdx.x;
    int64_t const num_rows = gridDim.x;
    auto const offset = original_row * orig_cols;
    OutputType* reduced_row_ptr = reduced_unpermuted_output + offset;
    int64_t const num_valid = num_valid_ptr ? *num_valid_ptr: 0;

    // Load 128-bits per thread, according to the smallest data type we read/write
    constexpr int64_t FINALIZE_ELEM_PER_THREAD
        = 128 / std::min(cutlass::sizeof_bits<OutputType>::value, cutlass::sizeof_bits<GemmOutputType>::value);

    int64_t const start_offset = threadIdx.x;
    int64_t const stride = FINALIZE_THREADS_PER_BLOCK;
    int64_t const num_elems_in_col = orig_cols / FINALIZE_ELEM_PER_THREAD;

    using BiasElem = cutlass::Array<ScaleBiasType, FINALIZE_ELEM_PER_THREAD>;
    using InputElem = cutlass::Array<GemmOutputType, FINALIZE_ELEM_PER_THREAD>;
    using OutputElem = cutlass::Array<OutputType, FINALIZE_ELEM_PER_THREAD>;
    using ComputeElem = cutlass::Array<float, FINALIZE_ELEM_PER_THREAD>;
    auto const* bias_v = reinterpret_cast<BiasElem const*>(bias);
    auto const* expanded_permuted_rows_v = reinterpret_cast<InputElem const*>(expanded_permuted_rows);
    auto* reduced_row_ptr_v = reinterpret_cast<OutputElem*>(reduced_row_ptr);

#pragma unroll
    for (int elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride)
    {
        bool has_valid = false;
        ComputeElem thread_output;
        thread_output.fill(0);
        float row_rescale{0.f};
        for (int k_idx = 0; k_idx < k; ++k_idx)
        {
            int64_t const expanded_original_row = original_row + k_idx * num_rows;
            int64_t const expanded_permuted_row = expanded_source_row_to_expanded_dest_row[expanded_original_row];

            int64_t const k_offset = original_row * k + k_idx;
            float const row_scale = (SCALE_MODE == ScaleMode::NO_SCALE) ? 1.f : scales[k_offset];
            if constexpr (SCALE_MODE == ScaleMode::RENORM_SCALE)
            {
                row_rescale = row_rescale + row_scale;
            }
            if (expanded_permuted_row < 0) {
                continue;
            }
            // Check after row_rescale has accumulated
            if (CHECK_SKIPPED && expanded_permuted_row >= num_valid)
            {
                continue;
            }

            auto const* expanded_permuted_rows_row_ptr
                = expanded_permuted_rows_v + expanded_permuted_row * num_elems_in_col;

            int64_t const expert_idx = expert_for_source_row[k_offset];

            auto const* bias_ptr = bias_v + expert_idx * num_elems_in_col;
            ComputeElem bias_value;
            if (bias)
            {
                bias_value = arrayConvert<BiasElem, ComputeElem>(bias_ptr[elem_index]);
            }
            else
            {
                bias_value.fill(0);
            }

            ComputeElem expert_result
                = arrayConvert<InputElem, ComputeElem>(expanded_permuted_rows_row_ptr[elem_index]);
            thread_output = thread_output + row_scale * (expert_result + bias_value);
            has_valid = true;
        }

        if (SCALE_MODE == ScaleMode::RENORM_SCALE && (!CHECK_SKIPPED || has_valid))
        {
            assert(row_rescale != 0.f);
            for (auto& elem : thread_output)
            {
                elem /= row_rescale;
            }
        }

        OutputElem output_elem = arrayConvert<ComputeElem, OutputElem>(thread_output);
        reduced_row_ptr_v[elem_index] = output_elem;
    }
}

template <class T, class OutputType, class GemmOutputType, class ScaleBiasType>
void finalizeMoeRoutingKernelLauncher(GemmOutputType const* expanded_permuted_rows,
    OutputType* reduced_unpermuted_output, ScaleBiasType const* bias, float const* scales,
    int const* expanded_source_row_to_expanded_dest_row, int const* expert_for_source_row, int64_t const num_rows,
    int64_t const cols, int64_t const k, int64_t const* num_valid_ptr, MOEParallelismConfig parallelism_config,
    MOEExpertScaleNormalizationMode normalization_mode, cudaStream_t stream)
{
    int64_t const blocks = num_rows;
    int64_t const threads = FINALIZE_THREADS_PER_BLOCK;

    // Only add bias on rank 0 for tensor parallelism
    bool const is_rank_0 = parallelism_config.tp_rank == 0;
    ScaleBiasType const* bias_ptr = is_rank_0 ? bias : nullptr;

    bool const check_finished = num_valid_ptr != nullptr;

    ScaleMode renorm_scales = ScaleMode::DEFAULT;
    if (normalization_mode == MOEExpertScaleNormalizationMode::RENORMALIZE)
    {
        renorm_scales = k == 1 ? ScaleMode::NO_SCALE : ScaleMode::RENORM_SCALE;
    }

    using FuncPtr
        = decltype(&finalizeMoeRoutingKernel<T, OutputType, GemmOutputType, ScaleBiasType, ScaleMode::DEFAULT, false>);
    FuncPtr func_map[2][3] = {
        {
            &finalizeMoeRoutingKernel<T, OutputType, GemmOutputType, ScaleBiasType, ScaleMode::NO_SCALE, false>,
            &finalizeMoeRoutingKernel<T, OutputType, GemmOutputType, ScaleBiasType, ScaleMode::DEFAULT, false>,
            &finalizeMoeRoutingKernel<T, OutputType, GemmOutputType, ScaleBiasType, ScaleMode::RENORM_SCALE, false>,
        },
        {
            &finalizeMoeRoutingKernel<T, OutputType, GemmOutputType, ScaleBiasType, ScaleMode::NO_SCALE, true>,
            &finalizeMoeRoutingKernel<T, OutputType, GemmOutputType, ScaleBiasType, ScaleMode::DEFAULT, true>,
            &finalizeMoeRoutingKernel<T, OutputType, GemmOutputType, ScaleBiasType, ScaleMode::RENORM_SCALE, true>,
        },
    };
    auto* const func = func_map[check_finished][int(renorm_scales)];
    func<<<blocks, threads, 0, stream>>>(expanded_permuted_rows, reduced_unpermuted_output, bias_ptr, scales,
        expanded_source_row_to_expanded_dest_row, expert_for_source_row, cols, k, num_valid_ptr);
}

// ============================== Gated Activation =================================
constexpr static int ACTIVATION_THREADS_PER_BLOCK = 256;

template <class T, class OutputType, template <class> class ActFn>
__global__ void doGatedActivationKernel(
    T* output, OutputType const* gemm_result, int64_t const* num_valid_tokens_ptr, int64_t inter_size)
{
    int64_t const tid = threadIdx.x;
    int64_t const token = blockIdx.x;
    if (num_valid_tokens_ptr && token >= *num_valid_tokens_ptr)
    {
        return;
    }

    output = output + token * inter_size;
    gemm_result = gemm_result + token * inter_size * 2;

    constexpr int64_t ACTIVATION_ELEM_PER_THREAD = 128 / cutlass::sizeof_bits<T>::value;

    using DataElem = cutlass::Array<T, ACTIVATION_ELEM_PER_THREAD>;
    using ComputeElem = cutlass::Array<float, ACTIVATION_ELEM_PER_THREAD>;
    auto gemm_result_vec = reinterpret_cast<DataElem const*>(gemm_result);
    auto output_vec = reinterpret_cast<DataElem*>(output);
    int64_t const start_offset = tid;
    int64_t const stride = ACTIVATION_THREADS_PER_BLOCK;
    assert(inter_size % ACTIVATION_ELEM_PER_THREAD == 0);
    int64_t const num_elems_in_col = inter_size / ACTIVATION_ELEM_PER_THREAD;
    int64_t const inter_size_vec = inter_size / ACTIVATION_ELEM_PER_THREAD;

    ActFn<ComputeElem> fn{};
    for (int64_t elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride)
    {
        auto fc1_value = arrayConvert<DataElem, ComputeElem>(gemm_result_vec[elem_index]);
        // BF16 isn't supported, use FP32 for activation function
        auto gate_value = arrayConvert<DataElem, ComputeElem>(gemm_result_vec[elem_index + inter_size_vec]);
        auto gate_act = fn(gate_value);
        output_vec[elem_index] = arrayConvert<ComputeElem, DataElem>(fc1_value * gate_act);
    }
}

template <typename T, typename OutputType>
void doGatedActivation(T* output, OutputType const* gemm_result, int64_t const* num_valid_tokens_ptr,
    int64_t inter_size, int64_t num_tokens, ActivationType activation_type, cudaStream_t stream)
{
    int64_t const blocks = num_tokens;
    int64_t const threads = ACTIVATION_THREADS_PER_BLOCK;

    // TODO For some reason Volta fails on GELU_taylor here with Warp Illegal Instruction.

    auto* fn = activation_type == ActivationType::Swiglu
        ? &doGatedActivationKernel<T, OutputType, cutlass::epilogue::thread::SiLu>
        : &doGatedActivationKernel<T, OutputType, cutlass::epilogue::thread::GELU>;
    fn<<<blocks, threads, 0, stream>>>(output, gemm_result, num_valid_tokens_ptr, inter_size);
}

// ============================== Activation =================================

template <class T, class GemmOutputType, class ScaleBiasType, template <class> class ActFn>
__global__ void doActivationKernel(T* output, GemmOutputType const* gemm_result, float const* fp8_quant,
    ScaleBiasType const* bias_ptr, bool bias_is_broadcast, int64_t const* expert_first_token_offset, int num_experts,
    int64_t inter_size, bool gated)
{
    int64_t const tid = threadIdx.x;
    int64_t const token = blockIdx.x;
    if (token >= expert_first_token_offset[num_experts])
    {
        return;
    }

    size_t gated_size_mul = gated ? 2 : 1;
    size_t gated_off = gated ? inter_size : 0;

    gemm_result = gemm_result + token * inter_size * gated_size_mul;
    output = output + token * inter_size; // Aliases gemm_result for non-gated, non-fp8 cases

    int64_t expert = 0;
    if (bias_ptr)
    {
        // TODO this is almost certainly faster as a linear scan
        expert = findTotalEltsLessThanTarget(expert_first_token_offset, num_experts, (int64_t) token + 1) - 1;
    }

    float const quant_scale = fp8_quant ? *fp8_quant : 1.f;

    if (bias_ptr)
    {
        size_t bias_offset
            = (bias_is_broadcast ? expert * inter_size * gated_size_mul : token * inter_size * gated_size_mul);
        bias_ptr = bias_ptr + bias_offset;
    }

    // Load 128-bits per thread, according to the smallest data type we read/write
    constexpr int64_t ACTIVATION_ELEM_PER_THREAD
        = 128 / std::min(cutlass::sizeof_bits<T>::value, cutlass::sizeof_bits<GemmOutputType>::value);

    using BiasElem = cutlass::Array<ScaleBiasType, ACTIVATION_ELEM_PER_THREAD>;
    using GemmResultElem = cutlass::Array<GemmOutputType, ACTIVATION_ELEM_PER_THREAD>;
    using OutputElem = cutlass::Array<T, ACTIVATION_ELEM_PER_THREAD>;
    using ComputeElem = cutlass::Array<float, ACTIVATION_ELEM_PER_THREAD>;
    auto gemm_result_vec = reinterpret_cast<GemmResultElem const*>(gemm_result);
    auto output_vec = reinterpret_cast<OutputElem*>(output);
    auto bias_ptr_vec = reinterpret_cast<BiasElem const*>(bias_ptr);
    int64_t const start_offset = tid;
    int64_t const stride = ACTIVATION_THREADS_PER_BLOCK;
    assert(inter_size % ACTIVATION_ELEM_PER_THREAD == 0);
    int64_t const num_elems_in_col = inter_size / ACTIVATION_ELEM_PER_THREAD;
    assert(gated_off % ACTIVATION_ELEM_PER_THREAD == 0);
    int64_t const gated_off_vec = gated_off / ACTIVATION_ELEM_PER_THREAD;

    ActFn<ComputeElem> fn{};
    for (int64_t elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride)
    {
        auto fc1_value = arrayConvert<GemmResultElem, ComputeElem>(gemm_result_vec[elem_index + gated_off_vec]);
        if (bias_ptr)
        {
            fc1_value = fc1_value + arrayConvert<BiasElem, ComputeElem>(bias_ptr_vec[elem_index + gated_off_vec]);
        }

        auto gate_act = fn(fc1_value);

        if (gated)
        {
            auto gate_mul = arrayConvert<GemmResultElem, ComputeElem>(gemm_result_vec[elem_index]);
            if (bias_ptr_vec)
            {
                gate_mul = gate_mul + arrayConvert<BiasElem, ComputeElem>(bias_ptr_vec[elem_index]);
            }
            gate_act = gate_act * gate_mul;
        }

        output_vec[elem_index] = arrayConvert<ComputeElem, OutputElem>(gate_act * quant_scale);
    }
}

template <class T, class GemmOutputType, class ScaleBiasType>
void doActivation(T* output, GemmOutputType const* gemm_result, float const* fp8_quant, ScaleBiasType const* bias,
    bool bias_is_broadcast, int64_t const* expert_first_token_offset, int num_experts, int64_t inter_size,
    int64_t num_tokens, ActivationType activation_type, cudaStream_t stream)
{
    int64_t const blocks = num_tokens;
    int64_t const threads = ACTIVATION_THREADS_PER_BLOCK;

    auto fn_list = std::array{
        &doActivationKernel<T, GemmOutputType, ScaleBiasType, cutlass::epilogue::thread::GELU>,    // Gelu
        &doActivationKernel<T, GemmOutputType, ScaleBiasType, cutlass::epilogue::thread::ReLu>,    // Relu
        &doActivationKernel<T, GemmOutputType, ScaleBiasType, cutlass::epilogue::thread::SiLu>,    // Silu
        &doActivationKernel<T, GemmOutputType, ScaleBiasType, cutlass::epilogue::thread::SiLu>,    // Swiglu
        &doActivationKernel<T, GemmOutputType, ScaleBiasType, cutlass::epilogue::thread::GELU>,    // Geglu
        &doActivationKernel<T, GemmOutputType, ScaleBiasType, cutlass::epilogue::thread::Identity> // Identity
    };
    auto fn = fn_list[static_cast<int>(activation_type)];
    fn<<<blocks, threads, 0, stream>>>(output, gemm_result, fp8_quant, bias, bias_is_broadcast,
        expert_first_token_offset, num_experts, inter_size, isGatedActivation(activation_type));
}

// ============================== Lora Add Bias =================================
constexpr static int LORA_KERNELS_THREADS_PER_BLOCK = 256;

template <class ScaleBiasType, class LoraType, bool IsGated>
__global__ void loraAddBiasKernel(ScaleBiasType* output, LoraType const* lora_result, ScaleBiasType const* bias,
    int64_t const* num_valid_tokens_ptr, int* permuted_experts, int64_t inter_size)
{
    int64_t const tid = threadIdx.x;
    int64_t const token = blockIdx.x;
    int64_t const num_tokens = gridDim.x;
    if (num_valid_tokens_ptr && token >= *num_valid_tokens_ptr)
    {
        return;
    }

    LoraType const* lora_result_1 = lora_result + token * inter_size;
    int expert_id = permuted_experts[token];
    if constexpr (IsGated)
    {
        output = output + token * inter_size * 2;
        bias = bias + expert_id * inter_size * 2;
    }
    else
    {
        output = output + token * inter_size;
        bias = bias + expert_id * inter_size;
    }

    constexpr int64_t LORA_ADD_BIAS_ELEM_PER_THREAD = 128 / cutlass::sizeof_bits<LoraType>::value;

    using DataElem = cutlass::Array<LoraType, LORA_ADD_BIAS_ELEM_PER_THREAD>;
    using BiasElem = cutlass::Array<ScaleBiasType, LORA_ADD_BIAS_ELEM_PER_THREAD>;
    auto lora_result_1_vec = reinterpret_cast<DataElem const*>(lora_result_1);
    auto bias_vec = reinterpret_cast<BiasElem const*>(bias);
    auto output_vec = reinterpret_cast<BiasElem*>(output);

    int64_t const start_offset = tid;
    int64_t const stride = LORA_KERNELS_THREADS_PER_BLOCK;
    assert(inter_size % LORA_ADD_BIAS_ELEM_PER_THREAD == 0);
    int64_t const num_elems_in_col = inter_size / LORA_ADD_BIAS_ELEM_PER_THREAD;

    for (int64_t elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride)
    {
        auto lora_value = lora_result_1_vec[elem_index];
        auto bias_value = bias_vec[elem_index];
        output_vec[elem_index] = bias_value + arrayConvert<DataElem, BiasElem>(lora_value);
    }

    if constexpr (IsGated)
    {
        auto lora_result_2_vec = reinterpret_cast<DataElem const*>(lora_result_1 + num_tokens * inter_size);
        int64_t const inter_size_vec = inter_size / LORA_ADD_BIAS_ELEM_PER_THREAD;
        for (int64_t elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride)
        {
            auto lora_value = lora_result_2_vec[elem_index];
            auto bias_value = bias_vec[elem_index + inter_size_vec];
            output_vec[elem_index + inter_size_vec] = bias_value + arrayConvert<DataElem, BiasElem>(lora_value);
        }
    }
}

template <class ScaleBiasType, class LoraType>
void loraAddBias(ScaleBiasType* output, LoraType const* lora_result, ScaleBiasType const* bias,
    int64_t const* num_valid_tokens_ptr, int64_t inter_size, int* permuted_experts, int64_t num_tokens,
    bool is_gated_activation, cudaStream_t stream)
{
    int64_t const blocks = num_tokens;
    int64_t const threads = LORA_KERNELS_THREADS_PER_BLOCK;

    auto selected_fn = is_gated_activation ? loraAddBiasKernel<ScaleBiasType, LoraType, true>
                                           : loraAddBiasKernel<ScaleBiasType, LoraType, false>;
    selected_fn<<<blocks, threads, 0, stream>>>(
        output, lora_result, bias, num_valid_tokens_ptr, permuted_experts, inter_size);
}

template <class T, class WeightType, class OutputType>
__device__ void computeHopperInputPointers(HopperGroupedGemmInput layout_info, int64_t gemm_m, int64_t gemm_n,
    int64_t gemm_k, int num_tokens_before_expert, int64_t expert, T const* in, WeightType const* weights, T const* bias,
    OutputType* output, int64_t const out_idx)
{
    // The input prior to this contains K elements per token, with `num_tokens_before_expert` tokens
    layout_info.ptr_a[out_idx] = in + num_tokens_before_expert * gemm_k;

    // Each expert's weight matrix is a constant size NxK, with `expert` experts
    layout_info.ptr_b[out_idx] = weights + expert * (gemm_n * gemm_k);

    //    if (bias)
    //    {
    //        // Each expert's bias is a constant size N, with `expert` experts
    //        layout_info.ptr_c[out_idx] = bias + expert * gemm_n;
    //    }

    if (layout_info.fusion == HopperGroupedGemmInput::EpilogueFusion::NONE)
    {
        // The output prior to this contains N elements per token, with `num_tokens_before_expert` tokens
        layout_info.default_epilogue.ptr_d[out_idx] = output + num_tokens_before_expert * gemm_n;
    }
}

__device__ __inline__ void computeHopperInputStrides(
    HopperGroupedGemmInput layout_info, int gemm_m, int gemm_n, int gemm_k, int64_t out_idx)
{
    layout_info.stride_a[out_idx]
        = cutlass::make_cute_packed_stride(HopperGroupedGemmInput::StrideA{}, cute::make_shape(gemm_m, gemm_k, 1));
    layout_info.stride_b[out_idx]
        = cutlass::make_cute_packed_stride(HopperGroupedGemmInput::StrideB{}, cute::make_shape(gemm_n, gemm_k, 1));
    if (layout_info.stride_c)
    {
        assert(false && "CUTLASS does not support a 1xN bias");
        //        layout_info.stride_c[out_idx] = cute::make_stride(0, cute::Int<1>{}, 0);
        layout_info.stride_c[out_idx]
            = cutlass::make_cute_packed_stride(HopperGroupedGemmInput::StrideC{}, cute::make_shape(1, gemm_n, 1));
    }
    if (layout_info.fusion == HopperGroupedGemmInput::EpilogueFusion::NONE)
    {
        layout_info.default_epilogue.stride_d[out_idx] = cutlass::make_cute_packed_stride(
            HopperGroupedGemmInput::DefaultEpilogue::StrideD{}, cute::make_shape(gemm_n, gemm_m, 1));
    }
}

// TODO Some of this setup could be cached
template <class T, class WeightType, class OutputType>
__global__ void computeStridesHopperKernel(int64_t const* expert_first_token_offset, HopperGroupedGemmInput layout_info,
    int64_t gemm_n, int64_t gemm_k, int64_t const num_experts, T const* in, WeightType const* weights,
    float const* fp8_dequant, T const* bias, OutputType* output)
{
    // First, compute the global tid. We only need 1 thread per expert.
    int const expert = blockIdx.x * blockDim.x + threadIdx.x;
    if (expert >= num_experts)
    {
        return;
    }

    auto const num_tokens_before_expert = expert_first_token_offset[expert];
    auto const num_tokens_including_expert = expert_first_token_offset[expert + 1];
    auto const num_tokens_to_expert = num_tokens_including_expert - num_tokens_before_expert;
    auto const gemm_m = num_tokens_to_expert;

    // M and N transposed since we are using the #tokens as the N dimension
    layout_info.shape_info.problem_shapes[expert]
        = HopperGroupedGemmInput::ProblemShape::UnderlyingProblemShape(gemm_n, gemm_m, gemm_k);

    if (fp8_dequant)
    {
        layout_info.alpha_scale_ptr_array[expert] = fp8_dequant + expert;
    }

    assert(gemm_m <= INT32_MAX);
    assert(gemm_n <= INT32_MAX);
    assert(gemm_k <= INT32_MAX);
    computeHopperInputStrides(layout_info, gemm_m, gemm_n, gemm_k, expert);

    computeHopperInputPointers(
        layout_info, gemm_m, gemm_n, gemm_k, num_tokens_before_expert, expert, in, weights, bias, output, expert);
}

template <class T, class WeightType, cutlass::WeightOnlyQuantOp QuantOp, class OutputType, class ScaleBiasType,
    class Enable>
std::vector<size_t>
CutlassMoeFCRunner<T, WeightType, QuantOp, OutputType, ScaleBiasType, Enable>::getWorkspaceDeviceBufferSizes(
    int64_t const num_rows, int64_t const hidden_size, int64_t const inter_size, int const num_experts,
    int const num_experts_per_node, int const k, ActivationType activation_type,
    MOEExpertScaleNormalizationMode norm_mode, bool use_lora) const
{
    size_t const num_moe_inputs = k * num_rows;
    size_t const permuted_elems = num_moe_inputs * hidden_size;
    size_t const interbuf_elems = num_moe_inputs * inter_size;
    size_t glu_inter_elems = 0;
    bool is_gated_activation = isGatedActivation(activation_type);
    if (is_gated_activation)
    {
        glu_inter_elems = interbuf_elems * 2;
    }
    else if (mayHaveDifferentGEMMOutputType())
    {
        // In this case we are using activation quantization, and some intermediate buffers will be unquantized
        // We need to have separate memory for these as we can no longer alias the output buffer for reuse
        glu_inter_elems = interbuf_elems;
    }

    bool using_hopper = moe_gemm_runner_.supportsHopperSpecialisation();

    size_t const gemm_output_dtype = sizeof(UnfusedGemmOutputType);

    size_t sparse_mixer_outs = 0;
    if (norm_mode == MOEExpertScaleNormalizationMode::SPARSE_MIXER)
    {
        sparse_mixer_outs = num_rows * num_experts;
    }

    size_t num_softmax_outs = 0;
    bool const is_pow_2 = (num_experts != 0) && ((num_experts & (num_experts - 1)) == 0);
    if (!is_pow_2 || num_experts > 256)
    {
        num_softmax_outs = num_rows * num_experts;
    }

    size_t const source_rows_size = num_moe_inputs * sizeof(int);
    size_t const permuted_rows_size = num_moe_inputs * sizeof(int);
    size_t const permuted_experts_size = num_moe_inputs * sizeof(int);
    size_t const permuted_data_size = permuted_elems * sizeof(T);
    size_t const expert_first_token_offset_size = (num_experts_per_node + 1) * sizeof(int64_t);
    size_t const sparse_mixer_out_size = sparse_mixer_outs * sizeof(float);
    size_t const softmax_out_size = num_softmax_outs * sizeof(float);
    size_t const permuted_scales_size = mayHaveFinalizeFused() ? num_moe_inputs * sizeof(float) : 0;
    size_t const glu_inter_size = glu_inter_elems * gemm_output_dtype; // May be an intermediate type for quantization
    size_t const fc1_result_size = interbuf_elems * sizeof(T);         // Acitvation quantizes so back to sizeof(T)
    size_t const sorter_size = CubKeyValueSorter::getWorkspaceSize(num_rows, num_experts);
    size_t const fc2_result_size = permuted_elems * gemm_output_dtype; // May be an intermediate type for quantization

    size_t const hopper_size = using_hopper ? HopperGroupedGemmInput::workspaceSize(num_experts_per_node) : 0;

    size_t const gemm_workspace_size = moe_gemm_runner_.getMaxWorkspaceSize(num_experts_per_node);

    // lora related
    size_t const lora_input_size
        = (use_lora && use_fp8) ? std::max(permuted_elems, interbuf_elems) * sizeof(ScaleBiasType) : 0;
    size_t const lora_fc1_result_size = use_lora
        ? (is_gated_activation ? 2 * interbuf_elems * sizeof(ScaleBiasType) : interbuf_elems * sizeof(ScaleBiasType))
        : 0;
    size_t const lora_add_bias_size = use_lora ? lora_fc1_result_size : 0;
    size_t const lora_fc2_result_size = use_lora ? permuted_elems * sizeof(ScaleBiasType) : 0;

    // We do some overlapping of the large workspace buffers. Although we could overlap some of the other buffers, they
    // are small enough (i.e no factor of hidden size) they will only be a couple MiB at most, so we don't bother
    // in the case of fused activation we overlap permuted_data and fc2_result
    // in the case of unfused activation we overlap permuted_data and fc1_result
    // we need to calculate the max possible size, so use the max of all three
    size_t overlapped_gemm1_gemm2_inputs = std::max(permuted_data_size, fc2_result_size);
    // When glu_inter_elems is 0 we are always fused, otherwise we may need the un-fused case
    if (glu_inter_elems > 0)
    {
        overlapped_gemm1_gemm2_inputs = std::max(overlapped_gemm1_gemm2_inputs, fc1_result_size);
    }

    size_t const alpha_scale_ptr_array_size = num_experts_per_node * sizeof(float**);

    // if we have glu_inter we overlap it with fc2_result, otherwise we use fc1_result by itself
    size_t overlapped_gemm1_gemm2_outputs = fc1_result_size;
    if (glu_inter_elems > 0)
    {
        overlapped_gemm1_gemm2_outputs
            = std::max(std::max(glu_inter_size, fc2_result_size), overlapped_gemm1_gemm2_outputs);
    }

    std::vector<size_t> workspace{source_rows_size, //
        permuted_rows_size,                         //
        permuted_experts_size,                      //
        expert_first_token_offset_size,             //
        sparse_mixer_out_size,                      //
        softmax_out_size,                           //
        permuted_scales_size,                       //
        sorter_size,                                //
        // These pointers reuse the same memory
        overlapped_gemm1_gemm2_inputs,  //
        overlapped_gemm1_gemm2_outputs, //
        alpha_scale_ptr_array_size,     //
        hopper_size,                    //
        gemm_workspace_size,            //
        lora_input_size,                //
        lora_fc1_result_size,           //
        lora_add_bias_size,             //
        lora_fc2_result_size};
    return workspace;
}

template <class T, class WeightType, cutlass::WeightOnlyQuantOp QuantOp, class OutputType, class ScaleBiasType,
    class Enable>
size_t CutlassMoeFCRunner<T, WeightType, QuantOp, OutputType, ScaleBiasType, Enable>::getWorkspaceSize(
    int64_t const num_rows, int64_t const hidden_size, int64_t const inter_size, int const num_experts, int const k,
    ActivationType activation_type, MOEExpertScaleNormalizationMode norm_mode, MOEParallelismConfig parallelism_config,
    bool use_lora) const
{
    int const ep_size = parallelism_config.ep_size;
    TLLM_CHECK_WITH_INFO(num_experts % ep_size == 0, "Number of experts must be a multiple of ep size");
    auto workspace = getWorkspaceDeviceBufferSizes(
        num_rows, hidden_size, inter_size, num_experts, num_experts / ep_size, k, activation_type, norm_mode, use_lora);
    auto ws_size = tensorrt_llm::common::calculateTotalWorkspaceSize(workspace.data(), workspace.size());
    TLLM_LOG_DEBUG("Mixture Of Experts Plugin requires workspace of %2f MiB", ws_size / 1024.f / 1024.f);
    return ws_size;
}

template <class T, class WeightType, cutlass::WeightOnlyQuantOp QuantOp, class OutputType, class ScaleBiasType,
    class Enable>
void CutlassMoeFCRunner<T, WeightType, QuantOp, OutputType, ScaleBiasType, Enable>::configureWsPtrs(char* ws_ptr,
    int64_t const num_rows, int64_t const hidden_size, int64_t const inter_size, int const num_experts,
    int const num_experts_per_node, int const k, ActivationType activation_type,
    MOEExpertScaleNormalizationMode norm_mode, bool use_lora)

{
    auto ws_sizes = getWorkspaceDeviceBufferSizes(
        num_rows, hidden_size, inter_size, num_experts, num_experts_per_node, k, activation_type, norm_mode, use_lora);

    std::vector<int8_t*> ws_sliced{(int8_t*) ws_ptr};
    for (auto size : ws_sizes)
    {
        ws_sliced.push_back(nextWorkspacePtr(ws_sliced.back(), size));
    }
    ws_sliced.pop_back();

    source_rows_ = (int*) ws_sliced[0];
    permuted_rows_ = (int*) ws_sliced[1];
    permuted_experts_ = (int*) ws_sliced[2];

    expert_first_token_offset_ = (int64_t*) ws_sliced[3];

    sparse_mixer_out_ = nullptr;
    if (norm_mode == MOEExpertScaleNormalizationMode::SPARSE_MIXER)
    {
        sparse_mixer_out_ = (float*) ws_sliced[4];
    }

    softmax_out_ = nullptr;
    bool const is_pow_2 = (num_experts != 0) && ((num_experts & (num_experts - 1)) == 0);
    if (!is_pow_2 || num_experts > 256)
    {
        softmax_out_ = (float*) ws_sliced[5];
    }

    bool const gemm2_using_hopper = moe_gemm_runner_.isHopperSpecialised(*gemm2_config_);
    permuted_scales_ = (gemm2_using_hopper && mayHaveFinalizeFused()) ? (float*) ws_sliced[6] : nullptr;

    sorter_ws_ = (char*) ws_sliced[7];

    // Always same index, but overlapped with either fc1_result_ or fc2_result_
    permuted_data_ = (T*) ws_sliced[8];

    bool const is_gated_activation = isGatedActivation(activation_type);
    bool const gemm1_using_fused_moe
        = moe_gemm_runner_.isFusedGatedActivation(is_gated_activation, inter_size, hidden_size);
    bool const gemm1_using_hopper = moe_gemm_runner_.isHopperSpecialised(*gemm1_config_);
    bool const hopper_has_glu = gemm1_using_hopper && (mayHaveDifferentGEMMOutputType() || is_gated_activation);
    // We always use fused path if we can
    bool const non_hopper_has_glu = !gemm1_using_fused_moe && is_gated_activation;
    bool const has_glu_inter_result = hopper_has_glu || non_hopper_has_glu || use_fp8;
    // Always same index, ignored if not needed
    glu_inter_result_ = has_glu_inter_result ? (T*) ws_sliced[9] : nullptr;

    // fc1 and fc2 alias one of the above pointers, but it depends on if actfn is fused/unfused which is overlapped
    // NOTE: It is important to get the order of these correct as the wrong order will cause the buffer to be used as an
    // input and output for the same gemm, which will cause corruption
    fc1_result_ = has_glu_inter_result ? (T*) ws_sliced[8] : (T*) ws_sliced[9];
    fc2_result_ = has_glu_inter_result ? (T*) ws_sliced[9] : (T*) ws_sliced[8];

    alpha_scale_ptr_array_ = reinterpret_cast<float const**>(ws_sliced[10]);

    hopper_grouped_gemm_input_ = {};
    if (moe_gemm_runner_.supportsHopperSpecialisation())
    {
        hopper_grouped_gemm_input_.configureWorkspace(ws_sliced[11], num_experts_per_node, ws_sliced[12], ws_sizes[12]);
    }

    lora_fc1_result_ = {};
    lora_add_bias_ = {};
    lora_fc2_result_ = {};

    if (use_lora)
    {
        lora_input_ = (ScaleBiasType*) ws_sliced[13];
        lora_fc1_result_ = (ScaleBiasType*) ws_sliced[14];
        lora_add_bias_ = (ScaleBiasType*) ws_sliced[15];
        lora_fc2_result_ = (ScaleBiasType*) ws_sliced[16];
    }
}

template <class T, class WeightType, cutlass::WeightOnlyQuantOp QuantOp, class OutputType, class ScaleBiasType,
    class Enable>
void CutlassMoeFCRunner<T, WeightType, QuantOp, OutputType, ScaleBiasType, Enable>::gemm1(
    MoeGemmRunner<T, WeightType, QuantOp, OutputType, ScaleBiasType>& gemm_runner, T const* const input,
    T* const output, void* const intermediate_result, int64_t const* const expert_first_token_offset,
    HopperGroupedGemmInput const hopper_input_template, WeightType const* const fc1_expert_weights,
    ScaleBiasType const* const fc1_expert_biases, int64_t const* const num_valid_tokens_ptr,
    ScaleBiasType const* const fc1_int_scales, ScaleBiasType const* const fc1_int_zeros, int group_size,
    float const* const fc1_fp8_dequant, float const* const fc2_fp8_quant, int64_t const expanded_num_rows,
    int64_t const hidden_size, int64_t const inter_size, int const num_experts_per_node,
    ActivationType fc1_activation_type, float const** alpha_scale_ptr_array, bool bias_is_broadcast,
    cudaStream_t stream, cutlass_extensions::CutlassGemmConfig config)
{
    bool const using_hopper_gemm1 = gemm_runner.isHopperSpecialised(config);
    bool const is_gated_activation = isGatedActivation(fc1_activation_type);
    bool const use_ampere_activation_fusion
        = gemm_runner.isFusedGatedActivation(is_gated_activation, inter_size, hidden_size);
    size_t const fc1_out_size = ((!use_ampere_activation_fusion) && is_gated_activation) ? inter_size * 2 : inter_size;

    int64_t const* total_tokens_including_expert = expert_first_token_offset + 1;

    if (using_hopper_gemm1)
    {
        TLLM_CHECK(config.is_sm90);
        TLLM_CHECK(!use_ampere_activation_fusion);
        bool has_different_gemm_output_type = using_hopper_gemm1 && !std::is_same_v<T, OutputType>;
        bool const has_intermediate = has_different_gemm_output_type || is_gated_activation;
        TLLM_CHECK_WITH_INFO(has_intermediate || input != output, "Input and output buffers are overlapping");
        auto* gemm_output = has_intermediate ? intermediate_result : static_cast<void*>(output);

        auto hopper_input = hopper_input_template;
        hopper_input.fusion = HopperGroupedGemmInput::EpilogueFusion::NONE;
        hopper_input = computeStridesHopper(expert_first_token_offset, hopper_input, fc1_out_size, hidden_size,
            num_experts_per_node, input, fc1_expert_weights, fc1_fp8_dequant, nullptr,
            static_cast<UnfusedGemmOutputType*>(gemm_output), stream);
        check_cuda_error();

        gemm_runner.moeGemm(input, nullptr, nullptr, nullptr, group_size, nullptr, total_tokens_including_expert,
            hopper_input, expanded_num_rows, fc1_out_size, hidden_size, num_experts_per_node, false,
            alpha_scale_ptr_array, stream, config);

        check_cuda_error();
        // TODO: when bias_is_broadcast is false, fuse bias to gemm
        doActivation<T, UnfusedGemmOutputType>(output, static_cast<UnfusedGemmOutputType const*>(gemm_output),
            fc2_fp8_quant, fc1_expert_biases, bias_is_broadcast, expert_first_token_offset, num_experts_per_node,
            inter_size, expanded_num_rows, fc1_activation_type, stream);

        check_cuda_error();
    }
    else if (use_fp8)
    {
        TLLM_CHECK(!use_ampere_activation_fusion);
        TLLM_CHECK(!config.is_sm90);

        alpha_scale_ptr_array
            = computeFP8DequantScale(alpha_scale_ptr_array, num_experts_per_node, fc1_fp8_dequant, stream);

        gemm_runner.moeGemm(input, fc1_expert_weights, nullptr, nullptr, group_size,
            reinterpret_cast<UnfusedGemmOutputType*>(intermediate_result), total_tokens_including_expert,
            HopperGroupedGemmInput{}, expanded_num_rows, fc1_out_size, hidden_size, num_experts_per_node, false,
            alpha_scale_ptr_array, stream, config);

        doActivation<T, UnfusedGemmOutputType>(output, static_cast<UnfusedGemmOutputType const*>(intermediate_result),
            fc2_fp8_quant, fc1_expert_biases, bias_is_broadcast, expert_first_token_offset, num_experts_per_node,
            inter_size, expanded_num_rows, fc1_activation_type, stream);

        check_cuda_error();
    }
    else if (!is_gated_activation)
    {
        TLLM_CHECK(!use_ampere_activation_fusion);
        TLLM_CHECK(!config.is_sm90);
        gemm_runner.moeGemmBiasAct(input, fc1_expert_weights, fc1_int_scales, fc1_int_zeros, group_size,
            fc1_expert_biases, bias_is_broadcast, output, total_tokens_including_expert, HopperGroupedGemmInput{},
            expanded_num_rows, fc1_out_size, hidden_size, num_experts_per_node, fc1_activation_type, false,
            alpha_scale_ptr_array, stream, config);

        check_cuda_error();
    }
    else
    {
        TLLM_CHECK(!config.is_sm90);
        TLLM_CHECK(is_gated_activation);
        TLLM_CHECK_WITH_INFO(
            !use_ampere_activation_fusion || input != output, "Input and output buffers are overlapping");

        // Run the GEMM with activation function overridden with `Identity`, we do the activation separately
        ActivationType activation_type = use_ampere_activation_fusion ? fc1_activation_type : ActivationType::Identity;
        void* gemm_result = use_ampere_activation_fusion ? static_cast<void*>(output) : intermediate_result;
        gemm_runner.moeGemmBiasAct(input, fc1_expert_weights, fc1_int_scales, fc1_int_zeros, group_size,
            fc1_expert_biases, bias_is_broadcast, gemm_result, total_tokens_including_expert, HopperGroupedGemmInput{},
            expanded_num_rows, fc1_out_size, hidden_size, num_experts_per_node, activation_type,
            use_ampere_activation_fusion, alpha_scale_ptr_array, stream, config);

        check_cuda_error();

        if (!use_ampere_activation_fusion)
        {
            doGatedActivation<T, UnfusedGemmOutputType>(output,
                static_cast<UnfusedGemmOutputType const*>(intermediate_result), num_valid_tokens_ptr, inter_size,
                expanded_num_rows, fc1_activation_type, stream);

            check_cuda_error();
        }
    }
}

template <class T, class WeightType, cutlass::WeightOnlyQuantOp QuantOp, class OutputType, class ScaleBiasType,
    class Enable>
void CutlassMoeFCRunner<T, WeightType, QuantOp, OutputType, ScaleBiasType, Enable>::gemm2(
    MoeGemmRunner<T, WeightType, QuantOp, OutputType, ScaleBiasType>& gemm_runner, T const* const input,
    void* const gemm_output, OutputType* const final_output, int64_t const* const expert_first_token_offset,
    HopperGroupedGemmInput const hopper_input_template, WeightType const* const fc2_expert_weights,
    ScaleBiasType const* const fc2_expert_biases, ScaleBiasType const* const fc2_int_scales,
    ScaleBiasType const* const fc2_int_zeros, int group_size, float const* const fc2_fp8_dequant,
    float const* const token_topk_unpermuted_scales, float const* const token_topk_permuted_scales,
    int const* const expanded_source_row_to_expanded_dest_row, int const* expanded_dest_row_to_expanded_source_row,
    int const* const expert_for_source_row, int64_t const* const num_valid_tokens_ptr, int64_t const num_rows,
    int64_t const expanded_num_rows, int64_t const hidden_size, int64_t const inter_size,
    int const num_experts_per_node, int64_t const k, bool using_hopper_fused_finalize,
    float const** alpha_scale_ptr_array, bool use_lora, void* fc2_lora, cudaStream_t stream,
    MOEParallelismConfig parallelism_config, cutlass_extensions::CutlassGemmConfig config)
{
    int64_t const* total_tokens_including_expert = expert_first_token_offset + 1;

    bool const using_hopper_gemm2 = gemm_runner.isHopperSpecialised(config);
    HopperGroupedGemmInput hopper_input{};
    if (using_hopper_gemm2)
    {
        bool apply_bias = parallelism_config.tp_rank == 0;
        hopper_input = hopper_input_template;
        hopper_input.fusion = HopperGroupedGemmInput::EpilogueFusion::NONE;
        if (using_hopper_fused_finalize)
        {
            hopper_input.fusion = HopperGroupedGemmInput::EpilogueFusion::FINALIZE;
            hopper_input.setFinalizeFusionParams(final_output, token_topk_permuted_scales, expert_first_token_offset,
                expanded_dest_row_to_expanded_source_row, apply_bias ? fc2_expert_biases : nullptr, hidden_size,
                num_rows);
            check_cuda_value(cudaMemsetAsync(final_output, 0x0, sizeof(OutputType) * num_rows * hidden_size, stream));
        }

        hopper_input = computeStridesHopper(expert_first_token_offset, hopper_input, hidden_size, inter_size,
            num_experts_per_node, input, fc2_expert_weights, fc2_fp8_dequant, nullptr,
            static_cast<UnfusedGemmOutputType*>(gemm_output), stream);

        check_cuda_error();
    }
    else if (use_fp8)
    {
        alpha_scale_ptr_array
            = computeFP8DequantScale(alpha_scale_ptr_array, num_experts_per_node, fc2_fp8_dequant, stream);
    }

    bool fuse_lora_bias = use_lora && !(use_fp8 || using_hopper_gemm2);

    gemm_runner.moeGemmBiasAct(input, fc2_expert_weights, fc2_int_scales, fc2_int_zeros, group_size,
        fuse_lora_bias ? static_cast<ScaleBiasType const*>(fc2_lora) : nullptr, false, gemm_output,
        total_tokens_including_expert, hopper_input, expanded_num_rows, hidden_size, inter_size, num_experts_per_node,
        ActivationType::Identity, false, alpha_scale_ptr_array, stream, config);
    check_cuda_error();

    if (use_lora && !fuse_lora_bias)
    {
        auto loraBiasApplyFunc = doActivation<UnfusedGemmOutputType, UnfusedGemmOutputType, ScaleBiasType>;
        loraBiasApplyFunc(static_cast<UnfusedGemmOutputType*>(gemm_output),
            static_cast<UnfusedGemmOutputType const*>(gemm_output), nullptr,
            static_cast<ScaleBiasType const*>(fc2_lora), false, expert_first_token_offset, num_experts_per_node,
            hidden_size, expanded_num_rows, ActivationType::Identity, stream);
        check_cuda_error();
    }

    bool has_different_output_type_ampere = use_fp8 && !using_hopper_gemm2;
    bool has_different_output_type_hopper = !using_hopper_fused_finalize && using_hopper_gemm2;

    if (has_different_output_type_ampere || has_different_output_type_hopper)
    {
        finalizeMoeRoutingKernelLauncher<T, OutputType, UnfusedGemmOutputType>(
            static_cast<UnfusedGemmOutputType const*>(gemm_output), final_output, fc2_expert_biases,
            token_topk_unpermuted_scales, expanded_source_row_to_expanded_dest_row, expert_for_source_row, num_rows,
            hidden_size, k, num_valid_tokens_ptr, parallelism_config, MOEExpertScaleNormalizationMode::NONE, stream);
    }
    else if (!using_hopper_gemm2)
    {
        finalizeMoeRoutingKernelLauncher<T, OutputType, T>(static_cast<T const*>(gemm_output), final_output,
            fc2_expert_biases, token_topk_unpermuted_scales, expanded_source_row_to_expanded_dest_row,
            expert_for_source_row, num_rows, hidden_size, k, num_valid_tokens_ptr, parallelism_config,
            MOEExpertScaleNormalizationMode::NONE, stream);
    }

    check_cuda_error();
}

template<class T,
         class WeightType,
         cutlass::WeightOnlyQuantOp QuantOp,
         class OutputType,
         class ScaleBiasType,
         class Enable>
void CutlassMoeFCRunner<T, WeightType, QuantOp, OutputType, ScaleBiasType, Enable>::runMoe(
    void const*                     input_activations_void,
    float const*                    gating_output,
    float const*                    gating_output_with_bias,
    void const*                     fc1_expert_weights_void,
    void const*                     fc1_expert_biases_void,
    ActivationType                  fc1_activation_type,
    void const*                     fc2_expert_weights_void,
    void const*                     fc2_expert_biases_void,
    QuantParams                     quant_params,
    int64_t const                   num_rows,
    int64_t const                   hidden_size,
    int64_t const                   inter_size,
    int const                       num_experts,
    int const                       k,
    char*                           workspace_ptr,
    void*                           final_output_void,
    bool const*                     finished,
    int64_t const                   active_rows,
    void*                           token_topk_final_scales_void,
    int*                            expanded_source_row_to_expanded_dest_row,
    int*                            expert_for_source_row,
    float                           sparse_mixer_epsilon,
    MOEParallelismConfig            parallelism_config,
    MOEExpertScaleNormalizationMode normalization_mode,
    bool                            use_lora,
    LoraParams&                     lora_params,
    cudaStream_t                    stream) {
    static constexpr bool int_scales_required
        = std::is_same<WeightType, uint8_t>::value || std::is_same<WeightType, cutlass::uint4b_t>::value;
    static constexpr bool fp8_scales_required
        = std::is_same<WeightType, __nv_fp8_e4m3>::value || std::is_same<WeightType, __nv_fp8_e5m2>::value;

    auto const* input_activations = static_cast<T const*>(input_activations_void);
    auto const* fc1_expert_weights = static_cast<WeightType const*>(fc1_expert_weights_void);
    auto const* fc1_expert_biases = reinterpret_cast<ScaleBiasType const*>(fc1_expert_biases_void);
    auto const* fc2_expert_weights = static_cast<WeightType const*>(fc2_expert_weights_void);
    auto const* fc1_int_scales = reinterpret_cast<ScaleBiasType const*>(quant_params.fc1_weight_scales);
    auto const* fc1_int_zeros = reinterpret_cast<ScaleBiasType const*>(quant_params.fc1_weight_zeros);
    auto const* fc2_int_scales = reinterpret_cast<ScaleBiasType const*>(quant_params.fc2_weight_scales);
    auto const* fc2_int_zeros = reinterpret_cast<ScaleBiasType const*>(quant_params.fc2_weight_zeros);
    int const group_size = quant_params.group_size;

    auto const* fc1_fp8_dequant = quant_params.dequant_fc1;
    auto const* fc2_fp8_quant = quant_params.quant_fc2;
    auto const* fc2_fp8_dequant = quant_params.dequant_fc2;
    auto const* input_fp8_dequant = quant_params.dequant_input;
    auto const* fc2_expert_biases = reinterpret_cast<ScaleBiasType const*>(fc2_expert_biases_void);
    auto* final_output = static_cast<OutputType*>(final_output_void);
    auto* token_topk_unpermuted_scales = static_cast<float*>(token_topk_final_scales_void);

    TLLM_CHECK_WITH_INFO(finished == nullptr, "Using 'finished' is deprecated and will be removed in future versions");
    TLLM_CHECK_WITH_INFO(
        num_rows == active_rows, "Using 'finished' is deprecated and will be removed in future versions");
    TLLM_CHECK(input_activations);
    TLLM_CHECK(fc1_expert_weights);
    TLLM_CHECK(fc2_expert_weights);
    TLLM_CHECK(workspace_ptr);
    TLLM_CHECK(token_topk_unpermuted_scales);
    TLLM_CHECK(expanded_source_row_to_expanded_dest_row);
    TLLM_CHECK(expert_for_source_row);
    TLLM_CHECK(num_experts % parallelism_config.ep_size == 0);
    TLLM_CHECK_WITH_INFO(hidden_size >= 128 / cutlass::sizeof_bits<WeightType>::value,
        "Hidden size is too small to meet alignment requirements for MOE GEMM");
    TLLM_CHECK_WITH_INFO(hidden_size % (128 / cutlass::sizeof_bits<WeightType>::value) == 0,
        "Hidden size does not meet minimum alignment requirements for MOE GEMM");
    TLLM_CHECK_WITH_INFO(inter_size % (128 / cutlass::sizeof_bits<WeightType>::value) == 0,
        "Inter size does not meet minimum alignment requirements for MOE GEMM");

    // These values must fit into an int for building the source maps
    TLLM_CHECK_WITH_INFO(num_rows <= std::numeric_limits<int>::max(), "Number of rows is too large");
    TLLM_CHECK_WITH_INFO(
        num_rows * num_experts <= std::numeric_limits<int>::max(), "Number of rows * num_experts is too large");
    TLLM_CHECK_WITH_INFO(k * num_experts <= std::numeric_limits<int>::max(), "k * num_experts is too large");

    TLLM_CHECK_WITH_INFO(gemm1_config_.has_value(), "MOE GEMM1 Config is not set");
    TLLM_CHECK_WITH_INFO(gemm2_config_.has_value(), "MOE GEMM2 Config is not set");

    if (int_scales_required)
    {
        TLLM_CHECK_WITH_INFO(
            fc1_int_scales != nullptr, "Weight scales expected but scale for first matmul is a null pointer");
        TLLM_CHECK_WITH_INFO(
            fc2_int_scales != nullptr, "Weight scales expected but scale for second matmul is a null pointer");

        TLLM_CHECK_WITH_INFO(fc1_fp8_dequant == nullptr && fc2_fp8_quant == nullptr && fc2_fp8_dequant == nullptr,
            "FP8 scales are provided for integer quantization");
    }
    else if (fp8_scales_required)
    {
        TLLM_CHECK_WITH_INFO(fc1_expert_biases == nullptr, "Bias is not supported with FP8");
        TLLM_CHECK_WITH_INFO(fc2_expert_biases == nullptr, "Bias is not supported with FP8");

        TLLM_CHECK_WITH_INFO(
            fc1_fp8_dequant != nullptr, "FP8 scales expected but dequant scale for FC1 is a null pointer");
        TLLM_CHECK_WITH_INFO(fc2_fp8_quant != nullptr, "FP8 scales expected but quant scale for FC2 is a null pointer");
        TLLM_CHECK_WITH_INFO(
            fc2_fp8_dequant != nullptr, "FP8 scales expected but quant scale for FC2 is a null pointer");

        TLLM_CHECK_WITH_INFO(
            fc1_int_scales == nullptr && fc2_int_scales == nullptr, "Integer scales are provided for FP8 quantization");
    }
    else if (use_lora && use_fp8)
    {
        TLLM_CHECK_WITH_INFO(
            input_fp8_dequant != nullptr, "FP8 scales expected but quant scale for input is a null pointer");
    }
    else
    {
        TLLM_CHECK_WITH_INFO(
            fc1_int_scales == nullptr, "Scales are ignored for fp32/fp16/bf16 but received weight scale for FC1");
        TLLM_CHECK_WITH_INFO(
            fc2_int_scales == nullptr, "Scales are ignored for fp32/fp16/bf16 but received weight scale for FC2");
        TLLM_CHECK_WITH_INFO(
            fc1_fp8_dequant == nullptr, "Scales are ignored for fp32/fp16/bf16 but received dequant scale for FC1");
        TLLM_CHECK_WITH_INFO(
            fc2_fp8_quant == nullptr, "Scales are ignored for fp32/fp16/bf16 but received quant scale for FC2");
        TLLM_CHECK_WITH_INFO(
            fc2_fp8_dequant == nullptr, "Scales are ignored for fp32/fp16/bf16 but received quant scale for FC2");
    }

    int const num_experts_per_node = num_experts / parallelism_config.ep_size;

    configureWsPtrs(workspace_ptr, num_rows, hidden_size, inter_size, num_experts, num_experts_per_node, k,
        fc1_activation_type, normalization_mode, use_lora);

    int const start_expert = num_experts_per_node * parallelism_config.ep_rank;
    int const end_expert = start_expert + num_experts_per_node;

    genSourceRow(expert_for_source_row, source_rows_, num_rows, k, num_experts, start_expert, end_expert, stream);
    check_cuda_error();

    sortAndScanSoftmaxOutput(expert_for_source_row, source_rows_, permuted_experts_, permuted_rows_,
        expert_first_token_offset_, num_rows, num_experts, num_experts_per_node, k, sorter_,
        static_cast<void*>(sorter_ws_), stream);

    check_cuda_error();

    int64_t const expanded_num_rows = k * num_rows;
    bool is_gated_activation = isGatedActivation(fc1_activation_type);

    if (use_lora)
    {
        std::vector<int>& host_permuted_rows = host_lora_workspace_.host_permuted_rows;
        std::vector<int64_t>& host_expert_first_token_offset = host_lora_workspace_.host_expert_first_token_offset;
        host_permuted_rows.resize(expanded_num_rows);
        TLLM_CUDA_CHECK(cudaMemcpyAsync(host_permuted_rows.data(), permuted_rows_, expanded_num_rows * sizeof(int),
            cudaMemcpyDeviceToHost, stream));
        host_expert_first_token_offset.resize(num_experts_per_node + 1);
        TLLM_CUDA_CHECK(cudaMemcpyAsync(host_expert_first_token_offset.data(), expert_first_token_offset_,
            (num_experts_per_node + 1) * sizeof(int64_t), cudaMemcpyDeviceToHost, stream));
        TLLM_CUDA_CHECK(cudaEventRecord(*(lora_params.memcpy_event_ptr), stream));
    }

    // Actually permute the data
    bool const needs_num_valid = finished || parallelism_config.ep_size > 1;
    int64_t const* num_valid_tokens_ptr = needs_num_valid ? expert_first_token_offset_ + num_experts_per_node : nullptr;
    expandInputRowsKernelLauncher(input_activations, permuted_data_, token_topk_unpermuted_scales, permuted_scales_,
        permuted_rows_, expanded_source_row_to_expanded_dest_row, num_rows, num_valid_tokens_ptr, hidden_size, k,
        stream);

    check_cuda_error();

    Self::gemm1(moe_gemm_runner_, permuted_data_, fc1_result_, glu_inter_result_, expert_first_token_offset_,
        hopper_grouped_gemm_input_, fc1_expert_weights, fc1_expert_biases, num_valid_tokens_ptr, fc1_int_scales,
        fc1_int_zeros, group_size, fc1_fp8_dequant, fc2_fp8_quant, expanded_num_rows, hidden_size, inter_size,
        num_experts_per_node, fc1_activation_type, alpha_scale_ptr_array_, !use_lora, stream, *gemm1_config_);

    check_cuda_error();

    Self::gemm2(moe_gemm_runner_, fc1_result_, fc2_result_, final_output, expert_first_token_offset_,
        hopper_grouped_gemm_input_, fc2_expert_weights, fc2_expert_biases, fc2_int_scales, fc2_int_zeros, group_size,
        fc2_fp8_dequant, token_topk_unpermuted_scales, permuted_scales_, expanded_source_row_to_expanded_dest_row,
        permuted_rows_, expert_for_source_row, num_valid_tokens_ptr, num_rows, expanded_num_rows, hidden_size,
        inter_size, num_experts_per_node, k, !use_deterministic_hopper_reduce_, alpha_scale_ptr_array_, use_lora,
        lora_fc2_result_, stream, parallelism_config, *gemm2_config_);

    check_cuda_error();
}

template <class T, class WeightType, cutlass::WeightOnlyQuantOp QuantOp, class OutputType, class ScaleBiasType,
    class Enable>
HopperGroupedGemmInput
CutlassMoeFCRunner<T, WeightType, QuantOp, OutputType, ScaleBiasType, Enable>::computeStridesHopper(
    int64_t const* expert_first_token_offset, HopperGroupedGemmInput layout_info, int64_t gemm_n, int64_t gemm_k,
    int const num_experts, T const* in, WeightType const* weights, float const* fp8_dequant, T const* bias,
    UnfusedGemmOutputType* output, cudaStream_t stream)
{
    // Always nullptr
    layout_info.ptr_c = nullptr;
    layout_info.stride_c = nullptr;

    if (!fp8_dequant)
    {
        layout_info.alpha_scale_ptr_array = nullptr;
    }

    int const threads = std::min(1024, num_experts);
    int const blocks = (num_experts + threads - 1) / threads;

    computeStridesHopperKernel<<<blocks, threads, 0, stream>>>(
        expert_first_token_offset, layout_info, gemm_n, gemm_k, num_experts, in, weights, fp8_dequant, bias, output);

    return layout_info;
}

template void finalizeMoeRoutingKernelLauncher<__nv_bfloat16>(__nv_bfloat16 const* expanded_permuted_rows,
                                                              __nv_bfloat16* reduced_unpermuted_output, __nv_bfloat16 const* bias, float const* scales,
                                                              int const* expanded_source_row_to_expanded_dest_row, int const* expert_for_source_row, int64_t const num_rows,
                                                              int64_t const cols, int64_t const k, int64_t const* num_valid_ptr, MOEParallelismConfig parallelism_config,
                                                              MOEExpertScaleNormalizationMode normalization_mode, cudaStream_t stream);

} // namespace tensorrt_llm::kernels
