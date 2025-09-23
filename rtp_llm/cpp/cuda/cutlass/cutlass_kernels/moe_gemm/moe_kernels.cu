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
#include "rtp_llm/cpp/cuda/cutlass/cutlass_kernels/moe_gemm/moe_kernels.inl"

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

static constexpr int WARP_SIZE = 32;

// ====================== Softmax things ===============================
// We have our own implementation of softmax here so we can support transposing the output
// in the softmax kernel when we extend this module to support expert-choice routing.
template <int TPB>
__launch_bounds__(TPB) __global__
    void moeSoftmax(float const* input, bool const* finished, float* output, int64_t const num_cols)
{
    using BlockReduce = cub::BlockReduce<float, TPB>;
    __shared__ typename BlockReduce::TempStorage tmpStorage;

    __shared__ float normalizing_factor;
    __shared__ float float_max;

    int64_t const thread_row_offset = blockIdx.x * num_cols;

    cub::Sum sum;
    float threadData(-FLT_MAX);

    // Don't touch finished rows.
    if ((finished != nullptr) && finished[blockIdx.x])
    {
        return;
    }

    for (int ii = threadIdx.x; ii < num_cols; ii += TPB)
    {
        int64_t const idx = thread_row_offset + ii;
        threadData = max(input[idx], threadData);
    }

    float const maxElem = BlockReduce(tmpStorage).Reduce(threadData, cub::Max());
    if (threadIdx.x == 0)
    {
        float_max = maxElem;
    }
    __syncthreads();

    threadData = 0;

    for (int ii = threadIdx.x; ii < num_cols; ii += TPB)
    {
        int64_t const idx = thread_row_offset + ii;
        threadData += exp((static_cast<float>(input[idx]) - float_max));
    }

    auto const Z = BlockReduce(tmpStorage).Reduce(threadData, sum);

    if (threadIdx.x == 0)
    {
        normalizing_factor = 1.f / Z;
    }
    __syncthreads();

    for (int ii = threadIdx.x; ii < num_cols; ii += TPB)
    {
        int64_t const idx = thread_row_offset + ii;
        float const val = exp((static_cast<float>(input[idx]) - float_max)) * normalizing_factor;
        output[idx] = val;
    }
}

template <int TPB, typename TOPK_T>
__launch_bounds__(TPB) __global__ void moeTopK(float const* inputs_after_softmax, bool const* finished, float* output,
    TOPK_T* indices, int* source_rows, int const num_experts, int const k, int const startk, int const endk,
    int const start_expert, int const end_expert, MOEExpertScaleNormalizationMode norm_mode)
{

    using cub_kvp = cub::KeyValuePair<int, float>;
    using BlockReduce = cub::BlockReduce<cub_kvp, TPB>;
    __shared__ typename BlockReduce::TempStorage tmpStorage;

    cub_kvp thread_kvp;
    cub::ArgMax arg_max;

    int64_t const num_rows = gridDim.x;
    int64_t const block_row = blockIdx.x;

    float renorm_value = 0.0f;
    bool const row_is_active = finished ? !finished[block_row] : true;
    int64_t const thread_read_offset = blockIdx.x * num_experts;
    for (int k_idx = startk; k_idx < endk; ++k_idx)
    {
        thread_kvp.key = 0;
        thread_kvp.value = -1.f; // This is OK because inputs are probabilities

        cub_kvp inp_kvp;
        for (int expert = threadIdx.x; expert < num_experts; expert += TPB)
        {
            int64_t const idx = thread_read_offset + expert;
            inp_kvp.key = expert;
            inp_kvp.value = inputs_after_softmax[idx];

            for (int prior_k = startk; prior_k < k_idx; ++prior_k)
            {
                int prior_winning_expert = indices[k * block_row + prior_k];
                // Adjust the selected index to correct for the expert parallel transformation
                prior_winning_expert = prior_winning_expert >= num_experts ? prior_winning_expert - num_experts
                                                                           : prior_winning_expert + start_expert;
                if (prior_winning_expert == expert)
                {
                    inp_kvp = thread_kvp;
                }
            }

            thread_kvp = arg_max(inp_kvp, thread_kvp);
        }

        cub_kvp const result_kvp = BlockReduce(tmpStorage).Reduce(thread_kvp, arg_max);
        if (threadIdx.x == 0)
        {
            // Ignore experts the node isn't responsible for with expert parallelism
            int const expert = result_kvp.key;
            bool const node_uses_expert = expert >= start_expert && expert < end_expert;
            bool const should_process_row = row_is_active && node_uses_expert;

            int64_t const idx = k * block_row + k_idx;
            output[idx] = result_kvp.value;
            indices[idx] = should_process_row ? (expert - start_expert) : (num_experts + expert);
            assert(indices[idx] >= 0);
            source_rows[idx] = k_idx * num_rows + block_row;

            if (norm_mode == MOEExpertScaleNormalizationMode::RENORMALIZE)
            {
                renorm_value += result_kvp.value;
            }
        }
        __syncthreads();
    }

    if (norm_mode == MOEExpertScaleNormalizationMode::RENORMALIZE && threadIdx.x == 0 && renorm_value != 0.f)
    {
        assert(startk == 0 && endk == k);
        renorm_value = 1 / renorm_value;
        for (int k_idx = 0; k_idx < k; k_idx++)
        {
            int64_t const idx = k * block_row + k_idx;
            output[idx] *= renorm_value;
        }
    }
}

// moeTopK based on input with bias
template<int TPB>
__launch_bounds__(TPB) __global__
void moeTopK(float const*                    inputs_after_softmax,
             float const*                    input_with_bias,
             bool const*                     finished,
             float*                          output,
             int*                            indices,
             int*                            source_rows,
             int const                       num_experts,
             int const                       k,
             int const                       startk,
             int const                       endk,
             int const                       start_expert,
             int const                       end_expert,
             MOEExpertScaleNormalizationMode norm_mode)
{
    using cub_kvp = cub::KeyValuePair<int, float>;
    using BlockReduce = cub::BlockReduce<cub_kvp, TPB>;
    __shared__ typename BlockReduce::TempStorage tmpStorage;

    cub_kvp thread_kvp;
    cub::ArgMax arg_max;

    int64_t const num_rows = gridDim.x;
    int64_t const block_row = blockIdx.x;

    float renorm_value = 0.0f;
    bool const row_is_active = finished ? !finished[block_row] : true;
    int64_t const thread_read_offset = blockIdx.x * num_experts;
    for (int k_idx = startk; k_idx < endk; ++k_idx)
    {
        thread_kvp.key = 0;
        thread_kvp.value = -1.f; // This is OK because inputs are probabilities

        cub_kvp inp_kvp;
        for (int expert = threadIdx.x; expert < num_experts; expert += TPB)
        {
            int64_t const idx = thread_read_offset + expert;
            inp_kvp.key = expert;
            inp_kvp.value = input_with_bias[idx];

            for (int prior_k = startk; prior_k < k_idx; ++prior_k)
            {
                int prior_winning_expert = indices[k * block_row + prior_k];
                // Adjust the selected index to correct for the expert parallel transformation
                prior_winning_expert = prior_winning_expert >= num_experts ? prior_winning_expert - num_experts
                                                                           : prior_winning_expert + start_expert;
                if (prior_winning_expert == expert)
                {
                    inp_kvp = thread_kvp;
                }
            }

            thread_kvp = arg_max(inp_kvp, thread_kvp);
        }

        cub_kvp const result_kvp = BlockReduce(tmpStorage).Reduce(thread_kvp, arg_max);
        if (threadIdx.x == 0)
        {
            // Ignore experts the node isn't responsible for with expert parallelism
            int const expert = result_kvp.key;
            int const origin_idx = thread_read_offset + expert;
            bool const node_uses_expert = expert >= start_expert && expert < end_expert;
            bool const should_process_row = row_is_active && node_uses_expert;

            int64_t const idx = k * block_row + k_idx;
            output[idx] = inputs_after_softmax[origin_idx];
            indices[idx] = should_process_row ? (expert - start_expert) : (num_experts + expert);
            assert(indices[idx] >= 0);
            source_rows[idx] = k_idx * num_rows + block_row;

            if (norm_mode == MOEExpertScaleNormalizationMode::RENORMALIZE)
            {
                renorm_value += inputs_after_softmax[origin_idx];
            }
        }
        __syncthreads();
    }

    if (norm_mode == MOEExpertScaleNormalizationMode::RENORMALIZE)
    {
        renorm_value += 1e-20;

        if (threadIdx.x == 0 && renorm_value != 0.f)
        {
            assert(startk == 0 && endk == k);
            renorm_value = 1 / renorm_value;
            for (int k_idx = 0; k_idx < k; k_idx++)
            {
                int64_t const idx = k * block_row + k_idx;
                output[idx] *= renorm_value;
            }
        }
    }
}

// ====================== TopK softmax things ===============================

/*
  A Top-K gating softmax written to exploit when the number of experts in the MoE layers
  are a small power of 2. This allows us to cleanly share the rows among the threads in
  a single warp and eliminate communication between warps (so no need to use shared mem).

  It fuses the softmax, max and argmax into a single kernel.

  Limitations:
  1) This implementation is intended for when the number of experts is a small power of 2.
  2) This implementation assumes k is small, but will work for any k.
*/

template <int VPT, int NUM_EXPERTS, int WARPS_PER_CTA, int BYTES_PER_LDG, typename TOPK_T>
__launch_bounds__(WARPS_PER_CTA* WARP_SIZE) __global__ void topkGatingSoftmax(float const* input, bool const* finished,
    float* output, int64_t const num_rows, TOPK_T* indices, int* source_rows, int const k, int const startk,
    int const endk, int const start_expert, int const end_expert, MOEExpertScaleNormalizationMode norm_mode)
{
    // We begin by enforcing compile time assertions and setting up compile time constants.
    static_assert(VPT == (VPT & -VPT), "VPT must be power of 2");
    static_assert(NUM_EXPERTS == (NUM_EXPERTS & -NUM_EXPERTS), "NUM_EXPERTS must be power of 2");
    static_assert(BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG), "BYTES_PER_LDG must be power of 2");
    static_assert(BYTES_PER_LDG <= 16, "BYTES_PER_LDG must be leq 16");

    // Number of bytes each thread pulls in per load
    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(float);
    static constexpr int ELTS_PER_ROW = NUM_EXPERTS;
    static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT;
    static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;

    // Restrictions based on previous section.
    static_assert(VPT % ELTS_PER_LDG == 0, "The elements per thread must be a multiple of the elements per ldg");
    static_assert(WARP_SIZE % THREADS_PER_ROW == 0, "The threads per row must cleanly divide the threads per warp");
    static_assert(THREADS_PER_ROW == (THREADS_PER_ROW & -THREADS_PER_ROW), "THREADS_PER_ROW must be power of 2");
    static_assert(THREADS_PER_ROW <= WARP_SIZE, "THREADS_PER_ROW can be at most warp size");

    // We have NUM_EXPERTS elements per row. We specialize for small #experts
    static constexpr int ELTS_PER_WARP = WARP_SIZE * VPT;
    static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW;
    static constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;

    // Restrictions for previous section.
    static_assert(ELTS_PER_WARP % ELTS_PER_ROW == 0, "The elts per row must cleanly divide the total elt per warp");

    // ===================== From this point, we finally start computing run-time variables. ========================

    // Compute CTA and warp rows. We pack multiple rows into a single warp, and a block contains WARPS_PER_CTA warps.
    // This, each block processes a chunk of rows. We start by computing the start row for each block.
    int64_t const cta_base_row = blockIdx.x * ROWS_PER_CTA;

    // Now, using the base row per thread block, we compute the base row per warp.
    int64_t const warp_base_row = cta_base_row + threadIdx.y * ROWS_PER_WARP;

    // The threads in a warp are split into sub-groups that will work on a row.
    // We compute row offset for each thread sub-group
    int const thread_row_in_warp = threadIdx.x / THREADS_PER_ROW;
    int64_t const thread_row = warp_base_row + thread_row_in_warp;

    // Threads with indices out of bounds should early exit here.
    if (thread_row >= num_rows)
    {
        return;
    }
    bool const row_is_active = finished ? !finished[thread_row] : true;

    // We finally start setting up the read pointers for each thread. First, each thread jumps to the start of the
    // row it will read.
    float const* thread_row_ptr = input + thread_row * ELTS_PER_ROW;

    // Now, we compute the group each thread belong to in order to determine the first column to start loads.
    int const thread_group_idx = threadIdx.x % THREADS_PER_ROW;
    int const first_elt_read_by_thread = thread_group_idx * ELTS_PER_LDG;
    float const* thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;

    // Determine the pointer type to use to read in the data depending on the BYTES_PER_LDG template param. In theory,
    // this can support all powers of 2 up to 16.
    using AccessType = cutlass::AlignedArray<float, ELTS_PER_LDG>;

    // Finally, we pull in the data from global mem
    cutlass::Array<float, VPT> row_chunk;
    AccessType* row_chunk_vec_ptr = reinterpret_cast<AccessType*>(&row_chunk);
    AccessType const* vec_thread_read_ptr = reinterpret_cast<AccessType const*>(thread_read_ptr);
#pragma unroll
    for (int ii = 0; ii < LDG_PER_THREAD; ++ii)
    {
        row_chunk_vec_ptr[ii] = vec_thread_read_ptr[ii * THREADS_PER_ROW];
    }

    // First, we perform a max reduce within the thread. We can do the max in fp16 safely (I think) and just
    // convert to float afterwards for the exp + sum reduction.
    float thread_max = row_chunk[0];
#pragma unroll
    for (int ii = 1; ii < VPT; ++ii)
    {
        thread_max = max(thread_max, row_chunk[ii]);
    }

// Now, we find the max within the thread group and distribute among the threads. We use a butterfly reduce.
#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2)
    {
        thread_max = max(thread_max, __shfl_xor_sync(0xFFFFFFFF, thread_max, mask, THREADS_PER_ROW));
    }

    // From this point, thread max in all the threads have the max within the row.
    // Now, we subtract the max from each element in the thread and take the exp. We also compute the thread local sum.
    float row_sum = 0;
#pragma unroll
    for (int ii = 0; ii < VPT; ++ii)
    {
        row_chunk[ii] = expf(row_chunk[ii] - thread_max);
        row_sum += row_chunk[ii];
    }

// Now, we perform the sum reduce within each thread group. Similar to the max reduce, we use a bufferfly pattern.
#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2)
    {
        row_sum += __shfl_xor_sync(0xFFFFFFFF, row_sum, mask, THREADS_PER_ROW);
    }

    // From this point, all threads have the max and the sum for their rows in the thread_max and thread_sum variables
    // respectively. Finally, we can scale the rows for the softmax. Technically, for top-k gating we don't need to
    // compute the entire softmax row. We can likely look at the maxes and only compute for the top-k values in the row.
    // However, this kernel will likely not be a bottle neck and it seems better to closer match torch and find the
    // argmax after computing the softmax.
    float const reciprocal_row_sum = 1.f / row_sum;

#pragma unroll
    for (int ii = 0; ii < VPT; ++ii)
    {
        row_chunk[ii] = row_chunk[ii] * reciprocal_row_sum;
    }

    // Now, softmax_res contains the softmax of the row chunk. Now, I want to find the topk elements in each row, along
    // with the max index.
    int start_col = first_elt_read_by_thread;
    static constexpr int COLS_PER_GROUP_LDG = ELTS_PER_LDG * THREADS_PER_ROW;

    float renorm_value = 0.0f;

    for (int k_idx = startk; k_idx < endk; ++k_idx)
    {
        // First, each thread does the local argmax
        float max_val = row_chunk[0];
        int expert = start_col;
#pragma unroll
        for (int ldg = 0, col = start_col; ldg < LDG_PER_THREAD; ++ldg, col += COLS_PER_GROUP_LDG)
        {
#pragma unroll
            for (int ii = 0; ii < ELTS_PER_LDG; ++ii)
            {
                float val = row_chunk[ldg * ELTS_PER_LDG + ii];

                // No check on the experts here since columns with the smallest index are processed first and only
                // updated if > (not >=)
                if (val > max_val)
                {
                    max_val = val;
                    expert = col + ii;
                }
            }
        }

// Now, we perform the argmax reduce. We use the butterfly pattern so threads reach consensus about the max.
// This will be useful for K > 1 so that the threads can agree on "who" had the max value. That thread can
// then blank out their max with -inf and the warp can run more iterations...
#pragma unroll
        for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2)
        {
            float other_max = __shfl_xor_sync(0xFFFFFFFF, max_val, mask, THREADS_PER_ROW);
            int other_expert = __shfl_xor_sync(0xFFFFFFFF, expert, mask, THREADS_PER_ROW);

            // We want lower indices to "win" in every thread so we break ties this way
            if (other_max > max_val || (other_max == max_val && other_expert < expert))
            {
                max_val = other_max;
                expert = other_expert;
            }
        }

        // Write the max for this k iteration to global memory.
        if (thread_group_idx == 0)
        {
            // Add a guard to ignore experts not included by this node
            bool const node_uses_expert = expert >= start_expert && expert < end_expert;
            bool const should_process_row = row_is_active && node_uses_expert;

            // The lead thread from each sub-group will write out the final results to global memory. (This will be a
            // single) thread per row of the input/output matrices.
            int64_t const idx = k * thread_row + k_idx;
            output[idx] = max_val;
            indices[idx] = should_process_row ? (expert - start_expert) : (NUM_EXPERTS + expert);
            source_rows[idx] = k_idx * num_rows + thread_row;

            // Accumulate renorm scalar
            if (norm_mode == MOEExpertScaleNormalizationMode::RENORMALIZE)
            {
                renorm_value += max_val;
            }
        }

        // Finally, we clear the value in the thread with the current max if there is another iteration to run.
        if (k_idx + 1 < endk)
        {
            int const ldg_group_for_expert = expert / COLS_PER_GROUP_LDG;
            int const thread_to_clear_in_group = (expert / ELTS_PER_LDG) % THREADS_PER_ROW;

            // Only the thread in the group which produced the max will reset the "winning" value to -inf.
            if (thread_group_idx == thread_to_clear_in_group)
            {
                int const offset_for_expert = expert % ELTS_PER_LDG;
                // Safe to set to any negative value since row_chunk values must be between 0 and 1.
                row_chunk[ldg_group_for_expert * ELTS_PER_LDG + offset_for_expert] = -10000.f;
            }
        }
    }

    if (norm_mode == MOEExpertScaleNormalizationMode::RENORMALIZE && thread_group_idx == 0 && renorm_value != 0.f)
    {
        assert(startk == 0 && endk == k);
        renorm_value = 1 / renorm_value;
        for (int k_idx = 0; k_idx < k; k_idx++)
        {
            int64_t const idx = k * thread_row + k_idx;
            output[idx] *= renorm_value;
        }
    }
}

namespace detail
{
// Constructs some constants needed to partition the work across threads at compile time.
template <int EXPERTS, int BYTES_PER_LDG>
struct TopkConstants
{
    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(float);
    static_assert(EXPERTS / (ELTS_PER_LDG * WARP_SIZE) == 0 || EXPERTS % (ELTS_PER_LDG * WARP_SIZE) == 0);
    static constexpr int VECs_PER_THREAD = std::max(1, EXPERTS / (ELTS_PER_LDG * WARP_SIZE));
    static constexpr int VPT = VECs_PER_THREAD * ELTS_PER_LDG;
    static constexpr int THREADS_PER_ROW = EXPERTS / VPT;
    static constexpr int ROWS_PER_WARP = WARP_SIZE / THREADS_PER_ROW;
};
} // namespace detail

template <int EXPERTS, int WARPS_PER_TB, typename TOPK_T>
void topkGatingSoftmaxLauncherHelper(float const* input, bool const* finished, float* output, TOPK_T* indices,
    int* source_row, int64_t const num_rows, int const k, int const startk, int const endk, int const start_expert,
    int const end_expert, MOEExpertScaleNormalizationMode norm_mode, cudaStream_t stream)
{
    static constexpr std::size_t MAX_BYTES_PER_LDG = 16;

    static constexpr int BYTES_PER_LDG = std::min(MAX_BYTES_PER_LDG, sizeof(float) * EXPERTS);
    using Constants = detail::TopkConstants<EXPERTS, BYTES_PER_LDG>;
    static constexpr int VPT = Constants::VPT;
    static constexpr int ROWS_PER_WARP = Constants::ROWS_PER_WARP;
    int64_t const num_warps = (num_rows + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
    int64_t const num_blocks = (num_warps + WARPS_PER_TB - 1) / WARPS_PER_TB;

    dim3 block_dim(WARP_SIZE, WARPS_PER_TB);
    topkGatingSoftmax<VPT, EXPERTS, WARPS_PER_TB, BYTES_PER_LDG><<<num_blocks, block_dim, 0, stream>>>(
        input, finished, output, num_rows, indices, source_row, k, startk, endk, start_expert, end_expert, norm_mode);
}

template <typename TOPK_T>
void topkGatingSoftmaxKernelLauncher(float const* input, float* output, float* softmax_temp_output, TOPK_T* indices,
    int* source_row, int64_t const num_rows, int const num_experts, int const k, int const startk, int const endk,
    int const start_expert, int const end_expert, MOEExpertScaleNormalizationMode norm_mode, cudaStream_t stream)
{
    static constexpr int WARPS_PER_TB = 4;

    switch (num_experts)
    {
    case 1:
    {
        topkGatingSoftmaxLauncherHelper<1, WARPS_PER_TB>(input, nullptr, output, indices, source_row, num_rows, k,
            startk, endk, start_expert, end_expert, norm_mode, stream);
        break;
    }
    case 2:
    {
        topkGatingSoftmaxLauncherHelper<2, WARPS_PER_TB>(input, nullptr, output, indices, source_row, num_rows, k,
            startk, endk, start_expert, end_expert, norm_mode, stream);
        break;
    }
    case 4:
    {
        topkGatingSoftmaxLauncherHelper<4, WARPS_PER_TB>(input, nullptr, output, indices, source_row, num_rows, k,
            startk, endk, start_expert, end_expert, norm_mode, stream);
        break;
    }
    case 8:
    {
        topkGatingSoftmaxLauncherHelper<8, WARPS_PER_TB>(input, nullptr, output, indices, source_row, num_rows, k,
            startk, endk, start_expert, end_expert, norm_mode, stream);
        break;
    }
    case 16:
    {
        topkGatingSoftmaxLauncherHelper<16, WARPS_PER_TB>(input, nullptr, output, indices, source_row, num_rows, k,
            startk, endk, start_expert, end_expert, norm_mode, stream);
        break;
    }
    case 32:
    {
        topkGatingSoftmaxLauncherHelper<32, WARPS_PER_TB>(input, nullptr, output, indices, source_row, num_rows, k,
            startk, endk, start_expert, end_expert, norm_mode, stream);
        break;
    }
    case 64:
    {
        topkGatingSoftmaxLauncherHelper<64, WARPS_PER_TB>(input, nullptr, output, indices, source_row, num_rows, k,
            startk, endk, start_expert, end_expert, norm_mode, stream);
        break;
    }
    case 128:
    {
        topkGatingSoftmaxLauncherHelper<128, WARPS_PER_TB>(input, nullptr, output, indices, source_row, num_rows, k,
            startk, endk, start_expert, end_expert, norm_mode, stream);
        break;
    }
    case 256:
    {
        topkGatingSoftmaxLauncherHelper<256, WARPS_PER_TB>(input, nullptr, output, indices, source_row, num_rows, k,
            startk, endk, start_expert, end_expert, norm_mode, stream);
        break;
    }
    default:
    {
        static constexpr int TPB = 256;
        TLLM_CHECK(softmax_temp_output != nullptr);
        moeSoftmax<TPB><<<num_rows, TPB, 0, stream>>>(input, nullptr, softmax_temp_output, num_experts);
        moeTopK<TPB><<<num_rows, TPB, 0, stream>>>(softmax_temp_output, nullptr, output, indices, source_row,
            num_experts, k, startk, endk, start_expert, end_expert, norm_mode);
    }
    }
}

template <int TPB>
void topkKernelLauncherHelper(float const* input, float const* input_with_bias, float* output, float* softmax_temp_output, int* indices,
    int* source_row, int64_t const num_rows, int const num_experts, int const k, int const startk, int const endk,
    int const start_expert, int const end_expert, MOEExpertScaleNormalizationMode norm_mode, cudaStream_t stream)
{
    moeTopK<TPB><<<num_rows, TPB, 0, stream>>>(input, input_with_bias, nullptr, output, indices, source_row,
                    num_experts, k, startk, endk, start_expert, end_expert, norm_mode);
}

template <typename TOPK_T>
void topkKernelLauncher(float const* input, float const* input_with_bias, float* output, float* softmax_temp_output, TOPK_T* indices,
    int* source_row, int64_t const num_rows, int const num_experts, int const k, int const startk, int const endk,
    int const start_expert, int const end_expert, MOEExpertScaleNormalizationMode norm_mode, cudaStream_t stream)
{
    static constexpr int WARPS_PER_TB = 4;

    switch (num_experts)
    {
    case 1:
    {
        topkKernelLauncherHelper<1>(input, input_with_bias, output, softmax_temp_output, indices, source_row, num_rows,
            num_experts, k, startk, endk, start_expert, end_expert, norm_mode, stream);
        break;
    }
    case 2:
    {
        topkKernelLauncherHelper<2>(input, input_with_bias, output, softmax_temp_output, indices, source_row, num_rows,
            num_experts, k, startk, endk, start_expert, end_expert, norm_mode, stream);
        break;
    }
    case 4:
    {
        topkKernelLauncherHelper<4>(input, input_with_bias, output, softmax_temp_output, indices, source_row, num_rows,
            num_experts, k, startk, endk, start_expert, end_expert, norm_mode, stream);
        break;
    }
    case 8:
    {
        topkKernelLauncherHelper<8>(input, input_with_bias, output, softmax_temp_output, indices, source_row, num_rows,
            num_experts, k, startk, endk, start_expert, end_expert, norm_mode, stream);
        break;
    }
    case 16:
    {
        topkKernelLauncherHelper<16>(input, input_with_bias, output, softmax_temp_output, indices, source_row, num_rows,
            num_experts, k, startk, endk, start_expert, end_expert, norm_mode, stream);
        break;
    }
    case 32:
    {
        topkKernelLauncherHelper<32>(input, input_with_bias, output, softmax_temp_output, indices, source_row, num_rows,
            num_experts, k, startk, endk, start_expert, end_expert, norm_mode, stream);
        break;
    }
    case 64:
    {
        topkKernelLauncherHelper<64>(input, input_with_bias, output, softmax_temp_output, indices, source_row, num_rows,
            num_experts, k, startk, endk, start_expert, end_expert, norm_mode, stream);
        break;
    }
    case 128:
    {
        topkKernelLauncherHelper<128>(input, input_with_bias, output, softmax_temp_output, indices, source_row, num_rows,
            num_experts, k, startk, endk, start_expert, end_expert, norm_mode, stream);
        break;
    }
    case 256:
    {
        topkKernelLauncherHelper<256>(input, input_with_bias, output, softmax_temp_output, indices, source_row, num_rows,
            num_experts, k, startk, endk, start_expert, end_expert, norm_mode, stream);
        break;
    }
    default:
    {
        static constexpr int TPB = 256;
        topkKernelLauncherHelper<256>(input, input_with_bias, output, softmax_temp_output, indices, source_row, num_rows,
            num_experts, k, startk, endk, start_expert, end_expert, norm_mode, stream);
    }
    }
}

__global__ void sparseMixerMask(float const* input, float* output, int const* indices, int k_idx, int k, int num_tokens,
    int num_experts, int start_expert, float epsilon)
{
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_idx >= num_tokens)
    {
        return;
    }

    // Mask out the largest value selected in the previous iteration
    int last_selected = (k_idx > 0) ? indices[k * token_idx + (k_idx - 1)] : INT_MIN;
    // Adjust the selected index to correct for the expert parallel transformation
    last_selected = last_selected >= num_experts ? last_selected - num_experts : last_selected + start_expert;

    // Find the max value in the current row
    float max_val = -INFINITY;
    for (int i = 0; i < num_experts; ++i)
    {
        if (i != last_selected)
        {
            float const val = input[token_idx * num_experts + i];
            max_val = max(val, max_val);
        }
    }

    // Mask out any values that fail the condition '(max - value) / std::max(abs(value), max) > 2 * epsilon'
    for (int i = 0; i < num_experts; ++i)
    {
        float val = input[token_idx * num_experts + i];
        float mask = (max_val - val) / max(abs(val), max_val);
        bool mask_value = (mask > 2 * epsilon) || i == last_selected;
        output[token_idx * num_experts + i] = mask_value ? -INFINITY : val;
    }
}

void sparseMixerTopkSoftmax(float const* input, float* output, float* mixer_temp_output, float* softmax_temp_output,
    int* indices, int* source_row, int64_t const num_rows, int const num_experts, int const k, int const start_expert,
    int const end_expert, float epsilon, cudaStream_t stream)
{
    // TODO we need to update the sparseMixerMask() function to mask all previous experts instead of just the most
    //  recent one.
    TLLM_CHECK_WITH_INFO(k <= 2, "Current sparse mixer only supports k <= 2");

    // Each thread handles one token
    constexpr int threads_per_block = 256;
    int num_blocks = ceilDiv(num_rows, threads_per_block);
    for (int k_idx = 0; k_idx < k; ++k_idx)
    {
        // Run softmax and topk in serial for each selection, recalculating the mask for each step
        sparseMixerMask<<<num_blocks, threads_per_block, 0, stream>>>(
            input, mixer_temp_output, indices, k_idx, k, num_rows, num_experts, start_expert, epsilon);

        topkGatingSoftmaxKernelLauncher(mixer_temp_output, output, softmax_temp_output, indices, source_row, num_rows,
            num_experts, k, k_idx, k_idx + 1, start_expert, end_expert, MOEExpertScaleNormalizationMode::NONE, stream);
    }
}

template <typename TOPK_T>
void invokeSelectExpertsForTokens(float const* input, float const* input_with_bias, float* output, float* mixer_temp_output, float* softmax_temp_output,
    TOPK_T* indices, int* source_row, int64_t const num_rows, int const num_experts, int const k, int const start_expert,
    int const end_expert, float mixer_epsilon, MOEExpertScaleNormalizationMode norm_mode, cudaStream_t stream)
{
    if (input == input_with_bias) {
        if (norm_mode == MOEExpertScaleNormalizationMode::SPARSE_MIXER)
        {
            TLLM_CHECK_WITH_INFO(mixer_temp_output, "Sparse mixer output is null when running sparse mixer");
            sparseMixerTopkSoftmax(input, output, mixer_temp_output, softmax_temp_output, (int *)indices, source_row, num_rows,
                num_experts, k, start_expert, end_expert, mixer_epsilon, stream);
        }
        else
        {
            topkGatingSoftmaxKernelLauncher(input, output, softmax_temp_output, indices, source_row, num_rows, num_experts,
                k, 0, k, start_expert, end_expert, norm_mode, stream);
        }
    }
    else
    {
        topkKernelLauncher(input, input_with_bias, output, softmax_temp_output, (int *)indices, source_row, num_rows, num_experts, k, 0, k,
            start_expert, end_expert, norm_mode, stream);
    }
    check_cuda_error();
}

template
void invokeSelectExpertsForTokens(float const* input, float const* input_with_bias, float* output, float* mixer_temp_output, float* softmax_temp_output,
    int* indices, int* source_row, int64_t const num_rows, int const num_experts, int const k, int const start_expert,
                                  int const end_expert, float mixer_epsilon, MOEExpertScaleNormalizationMode norm_mode, cudaStream_t stream);

template
void invokeSelectExpertsForTokens(float const* input, float const* input_with_bias, float* output, float* mixer_temp_output, float* softmax_temp_output,
    int64_t* indices, int* source_row, int64_t const num_rows, int const num_experts, int const k, int const start_expert,
                                  int const end_expert, float mixer_epsilon, MOEExpertScaleNormalizationMode norm_mode, cudaStream_t stream);

// ========================== CUB Sorting things ====================================
CubKeyValueSorter::CubKeyValueSorter()
    : num_experts_(0)
    , num_bits_(sizeof(int) * 8)
{
}

int CubKeyValueSorter::expertsToBits(int num_experts)
{
    // Max value we represent is V = num_experts + (num_experts - 1) = 2 * num_experts - 1
    // The maximum number of bits is therefore floor(log2(V)) + 1
    return static_cast<int>(log2(2 * num_experts - 1)) + 1;
}

CubKeyValueSorter::CubKeyValueSorter(int const num_experts)
    : num_experts_(num_experts)
    , num_bits_(expertsToBits(num_experts))
{
}

void CubKeyValueSorter::updateNumExperts(int const num_experts)
{
    num_experts_ = num_experts;
    num_bits_ = expertsToBits(num_experts);
}

size_t CubKeyValueSorter::getWorkspaceSize(size_t const num_key_value_pairs, int const num_experts)
{
    int num_bits = expertsToBits(num_experts);
    size_t required_storage = 0;
    int* null_int = nullptr;
    cub::DeviceRadixSort::SortPairs(
        nullptr, required_storage, null_int, null_int, null_int, null_int, num_key_value_pairs, 0, num_bits);

    // TODO: fix DeviceRadixSort
    //   when num_key_value_pairs, num_experts, num_bits, required_storage = 64, 4, 3, 0
    //   The required_storage seems to vary between 0 and 1 for the same inputs
    if (required_storage == 0)
    {
        required_storage = 1;
    }
    return required_storage;
}

void CubKeyValueSorter::run(void* workspace, size_t const workspace_size, int const* keys_in, int* keys_out,
    int const* values_in, int* values_out, size_t const num_key_value_pairs, cudaStream_t stream)
{
    size_t expected_ws_size = getWorkspaceSize(num_key_value_pairs, num_experts_);
    size_t actual_ws_size = workspace_size;

    TLLM_CHECK_WITH_INFO(expected_ws_size <= workspace_size,
        "[CubKeyValueSorter::run] The allocated workspace is too small to run this problem.");
    cub::DeviceRadixSort::SortPairs(
        workspace, actual_ws_size, keys_in, keys_out, values_in, values_out, num_key_value_pairs, 0, num_bits_, stream);
}

// ============================== Infer GEMM sizes =================================
// Calculates the start offset of the tokens for a given expert. The last element is the total number of valid tokens
__global__ void computeExpertFirstTokenOffsetKernel(int const* sorted_experts, int64_t const sorted_experts_len,
    int64_t const num_experts, int64_t* expert_first_token_offset)
{
    // First, compute the global tid. We only need 1 thread per expert.
    int const expert = blockIdx.x * blockDim.x + threadIdx.x;

    // Note that expert goes [0, num_experts] (inclusive) because we want a count for the total number of active tokens
    // at the end of the scan.
    if (expert >= num_experts + 1)
    {
        return;
    }
    expert_first_token_offset[expert] = findTotalEltsLessThanTarget(sorted_experts, sorted_experts_len, expert);
}

void computeExpertFirstTokenOffset(int const* sorted_indices, int const total_indices, int const num_experts,
    int64_t* expert_first_token_offset, cudaStream_t stream)
{
    int const num_entries = num_experts + 1;
    int const threads = std::min(1024, num_entries);
    int const blocks = (num_entries + threads - 1) / threads;

    computeExpertFirstTokenOffsetKernel<<<blocks, threads, 0, stream>>>(
        sorted_indices, total_indices, num_experts, expert_first_token_offset);
}

// ====================== Compute FP8 dequant scale only ===============================
__global__ void computeFP8DequantScaleKernel(
    float const** alpha_scale_ptr_array, int64_t const num_experts, float const* fp8_dequant)
{
    // First, compute the global tid. We only need 1 thread per expert.
    int const expert = blockIdx.x * blockDim.x + threadIdx.x;
    if (expert >= num_experts)
    {
        return;
    }

    assert(fp8_dequant != nullptr);
    alpha_scale_ptr_array[expert] = fp8_dequant + expert;
}

float const** computeFP8DequantScale(
    float const** alpha_scale_ptr_array, int const num_experts, float const* fp8_dequant, cudaStream_t stream)
{
    if (!fp8_dequant)
    {
        return nullptr;
    }

    int const threads = std::min(1024, num_experts);
    int const blocks = (num_experts + threads - 1) / threads;

    computeFP8DequantScaleKernel<<<blocks, threads, 0, stream>>>(alpha_scale_ptr_array, num_experts, fp8_dequant);

    return alpha_scale_ptr_array;
}

template <class T>
__global__ void loraReorderKernel(
    T* output, T const* lora_result, int64_t const* num_valid_tokens_ptr, int64_t inter_size)
{
    int64_t const tid = threadIdx.x;
    int64_t const token = blockIdx.x;
    int64_t const num_tokens = gridDim.x;
    if (num_valid_tokens_ptr && token >= *num_valid_tokens_ptr)
    {
        return;
    }

    T const* lora_result_1 = lora_result + token * inter_size;
    output = output + token * inter_size * 2;

    constexpr int64_t LORA_REORDER_ELEM_PER_THREAD = 128 / cutlass::sizeof_bits<T>::value;

    using DataElem = cutlass::Array<T, LORA_REORDER_ELEM_PER_THREAD>;
    auto lora_result_1_vec = reinterpret_cast<DataElem const*>(lora_result_1);
    auto output_vec = reinterpret_cast<DataElem*>(output);

    int64_t const start_offset = tid;
    int64_t const stride = LORA_KERNELS_THREADS_PER_BLOCK;
    assert(inter_size % LORA_REORDER_ELEM_PER_THREAD == 0);
    int64_t const num_elems_in_col = inter_size / LORA_REORDER_ELEM_PER_THREAD;

    for (int64_t elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride)
    {
        auto lora_value = lora_result_1_vec[elem_index];
        output_vec[elem_index] = lora_value;
    }

    auto lora_result_2_vec = reinterpret_cast<DataElem const*>(lora_result_1 + num_tokens * inter_size);
    int64_t const inter_size_vec = inter_size / LORA_REORDER_ELEM_PER_THREAD;
    for (int64_t elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride)
    {
        auto lora_value = lora_result_2_vec[elem_index];
        output_vec[elem_index + inter_size_vec] = lora_value;
    }
}

template <class T>
void loraReorder(T* output, T const* lora_result, int64_t const* num_valid_tokens_ptr, int64_t inter_size,
    int64_t num_tokens, cudaStream_t stream)
{
    int64_t const blocks = num_tokens;
    int64_t const threads = LORA_KERNELS_THREADS_PER_BLOCK;

    loraReorderKernel<T><<<blocks, threads, 0, stream>>>(output, lora_result, num_valid_tokens_ptr, inter_size);
}

// ============================== DEQUANT_FP8 =================================
constexpr static int DEQUANT_KERNELS_THREADS_PER_BLOCK = 256;

template <class OutputType, class InputType>
__global__ void dequantFP8Kernel(OutputType* output, InputType const* input, int64_t const* num_valid_tokens_ptr,
    int64_t inter_size, float const* scale, bool scale_is_dequant)
{
    int64_t const tid = threadIdx.x;
    int64_t const token = blockIdx.x;
    if (num_valid_tokens_ptr && token >= *num_valid_tokens_ptr)
    {
        return;
    }

    output = output + token * inter_size;
    input = input + token * inter_size;

    constexpr int64_t DEQUANT_ELEM_PER_THREAD = 128 / cutlass::sizeof_bits<InputType>::value;

    using DataElem = cutlass::Array<InputType, DEQUANT_ELEM_PER_THREAD>;
    using OutputElem = cutlass::Array<OutputType, DEQUANT_ELEM_PER_THREAD>;
    using ComputeElem = cutlass::Array<float, DEQUANT_ELEM_PER_THREAD>;
    auto input_vec = reinterpret_cast<DataElem const*>(input);
    auto output_vec = reinterpret_cast<OutputElem*>(output);

    int64_t const start_offset = tid;
    int64_t const stride = DEQUANT_KERNELS_THREADS_PER_BLOCK;
    assert(inter_size % DEQUANT_ELEM_PER_THREAD == 0);
    int64_t const num_elems_in_col = inter_size / DEQUANT_ELEM_PER_THREAD;

    ComputeElem deqaunt_scale_value;
    float dequant_scale = scale[0];
    if (!scale_is_dequant)
    {
        dequant_scale = 1.f / dequant_scale;
    }
    deqaunt_scale_value.fill(dequant_scale);

    for (int64_t elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride)
    {
        auto input_value = arrayConvert<DataElem, ComputeElem>(input_vec[elem_index]);
        output_vec[elem_index] = arrayConvert<ComputeElem, OutputElem>(input_value * deqaunt_scale_value);
    }
}

template <class OutputType, class InputType>
void dequantFP8(OutputType* output, InputType const* input, int64_t const* num_valid_tokens_ptr, int64_t inter_size,
    int64_t num_tokens, float const* scale, bool scale_is_dequant, cudaStream_t stream)
{
    int64_t const blocks = num_tokens;
    int64_t const threads = DEQUANT_KERNELS_THREADS_PER_BLOCK;

    dequantFP8Kernel<OutputType, InputType>
        <<<blocks, threads, 0, stream>>>(output, input, num_valid_tokens_ptr, inter_size, scale, scale_is_dequant);
}

// ==================== Helper for getting load balanced routing for profiling ==================================

template <class T>
__global__ void initRoutingKernelDiagonal(void* data_void, int num_experts, int num_tokens, int k, int stride)
{
    assert(k == 1 || (stride % num_experts) != 0);
    int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= num_tokens)
    {
        return;
    }
    T* data = reinterpret_cast<T*>(data_void) + token * num_experts;
    int start = token % num_experts;
    for (int i = 0; i < k; i++)
    {
        data[start] = T{1.f};
        start += stride;
        if (start >= num_experts) // Wrap
            start -= num_experts;
    }
}

void makeLoadBalancedRoutingConfiguration(
    void* data_void, int num_experts, int num_tokens, int k, nvinfer1::DataType type, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(type == nvinfer1::DataType::kFLOAT, "Routing configuration must be float");
    check_cuda_value(
        cudaMemsetAsync(data_void, 0x0, int64_t{num_experts} * int64_t{num_tokens} * sizeof(float), stream));

    int stride = tensorrt_llm::common::ceilDiv(num_experts, k);

    int blockDim = 256;
    int gridDim = tensorrt_llm::common::ceilDiv(num_tokens, blockDim);
    initRoutingKernelDiagonal<float><<<gridDim, blockDim, 0, stream>>>(data_void, num_experts, num_tokens, k, stride);

    check_cuda_error();
}

__global__ void prepareFakeRouterBuffers(int* unpermuted_source_rows, int* unpermuted_expert_selection,
    int64_t num_tokens, int64_t k, int64_t num_experts, int64_t num_experts_per_node)
{
    int64_t tid = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    int64_t sample = blockIdx.y;
    if (tid >= num_tokens)
    {
        return;
    }

    // Offset the buffers to the start of the sample
    unpermuted_source_rows += sample * num_tokens * k;
    unpermuted_expert_selection += sample * num_tokens * k;

    // This is not perf sensitive we just init the state here every time prepare is called
    // This means the first N tokens will always have the same distribution, regardless of num_tokens
    curandStatePhilox4_32_10_t state;
    curand_init(sample, tid, 0, &state);
    for (int k_idx = 0; k_idx < k; k_idx++)
    {
        while (true)
        {
            // curand_uniform includes 1 but not 0, so round up and subtract 1
            int expert = std::ceil(static_cast<float>(num_experts) * curand_uniform(&state)) - 1;

            bool valid = true;
            for (int prev_k = 0; prev_k < k_idx; prev_k++)
            {
                int prev_expert = unpermuted_expert_selection[k * tid + prev_k];
                if (expert == prev_expert)
                {
                    valid = false;
                    break;
                }
            }

            if (valid)
            {
                int64_t const idx = k * tid + k_idx;
                unpermuted_expert_selection[idx] = expert < num_experts_per_node ? expert : num_experts;
                unpermuted_source_rows[idx] = k_idx * num_tokens + tid;
                break;
            }
        }
    }
}

__global__ void buildReverseMap(int* expanded_source_row_to_expanded_dest_row,
    int const* expanded_dest_row_to_expanded_source_row, int64_t expanded_num_tokens)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < expanded_num_tokens)
    {
        assert(expanded_dest_row_to_expanded_source_row[tid] >= 0);
        assert(expanded_dest_row_to_expanded_source_row[tid] < expanded_num_tokens);
        expanded_source_row_to_expanded_dest_row[expanded_dest_row_to_expanded_source_row[tid]] = tid;
    }
}

__global__ void genSourceRowKernel(
    int* expert_rows, int* source_rows, int token_num, int top_k, int num_experts, int start_expert, int end_expert) {
    int const idx       = blockIdx.x * blockDim.x + threadIdx.x;
    int const token_idx = idx / top_k;
    int const k_idx     = idx % top_k;
    if (idx < token_num * top_k) {
        if (expert_rows[idx] >= start_expert && expert_rows[idx] < end_expert) {
            expert_rows[idx] = expert_rows[idx] - start_expert;
        } else if (expert_rows[idx] < 0) {
            expert_rows[idx] = num_experts;
        } else {
            expert_rows[idx] = expert_rows[idx] + num_experts;
        }
        source_rows[idx] = k_idx * token_num + token_idx;
    }
}

void genSourceRow(int*         expert_rows,
                  int*         source_rows,
                  int          token_num,
                  int          top_k,
                  int          num_experts,
                  int          start_expert,
                  int          end_expert,
                  cudaStream_t stream) {
    int const threads = 256;
    int const blocks  = token_num * top_k / 256 + 1;

    genSourceRowKernel<<<blocks, threads, 0, stream>>>(
        expert_rows, source_rows, token_num, top_k, num_experts, start_expert, end_expert);
}

void sortAndScanSoftmaxOutput(int* expert_for_source_row, int* source_rows, int* permuted_experts, int* permuted_rows,
    int64_t* expert_first_token_offset, int64_t num_rows, int64_t num_experts, int64_t num_experts_per_node, int64_t k,
    CubKeyValueSorter& sorter, void* sorter_ws, cudaStream_t stream)
{
    int64_t const expanded_num_rows = k * num_rows;
    // We need to use the full num_experts because that is the sentinel value used by topk for disabled experts
    sorter.updateNumExperts(num_experts);
    size_t const sorter_ws_size_bytes = pad_to_multiple_of_16(sorter.getWorkspaceSize(expanded_num_rows, num_experts));
    sorter.run((void*) sorter_ws, sorter_ws_size_bytes, expert_for_source_row, permuted_experts, source_rows,
        permuted_rows, expanded_num_rows, stream);

    check_cuda_error();

    // Upper bound on number of expanded rows
    computeExpertFirstTokenOffset(
        permuted_experts, expanded_num_rows, num_experts_per_node, expert_first_token_offset, stream);
}

void GemmProfilerBackend::prepare(int num_tokens, char* workspace, cudaStream_t stream)
{
    mAllTacticsSaved = mInterface->getTactics();
    mSampleIndex = 0;

    int64_t num_expanded_tokens = num_tokens * mK;

    mSorter.updateNumExperts(mNumExperts);

    auto getNext = getWorkspacePointerGenerator(workspace, num_tokens, mSM >= 90);
    int64_t* expert_first_token_offset_base = reinterpret_cast<int64_t*>(getNext());
    int* source_to_dest_base = reinterpret_cast<int*>(getNext());
    int* dest_to_source_base = reinterpret_cast<int*>(getNext());
    int* unpermuted_expert_selection_base = reinterpret_cast<int*>(getNext());
    int* unpermuted_source_rows_base = reinterpret_cast<int*>(getNext());

    int* permuted_experts = reinterpret_cast<int*>(getNext());
    int* sorter_ws = reinterpret_cast<int*>(getNext());

    uint32_t num_threads = 256;
    dim3 grid_dim{(num_tokens + num_threads - 1) / num_threads, NUM_ROUTING_SAMPLES, 1};
    prepareFakeRouterBuffers<<<grid_dim, num_threads, 0, stream>>>(
        unpermuted_source_rows_base, unpermuted_expert_selection_base, num_tokens, mK, mNumExperts, mNumExpertsPerNode);
    check_cuda_error();

    for (int64_t i = 0; i < NUM_ROUTING_SAMPLES; i++)
    {
        int64_t* expert_first_token_offset = expert_first_token_offset_base + i * (mNumExpertsPerNode + 1);
        int* source_to_dest = source_to_dest_base + i * num_expanded_tokens;
        int* dest_to_source = dest_to_source_base + i * num_expanded_tokens;
        int* unpermuted_expert_selection = unpermuted_expert_selection_base + i * num_expanded_tokens;
        int* unpermuted_source_rows = unpermuted_source_rows_base + i * num_expanded_tokens;

        sortAndScanSoftmaxOutput(unpermuted_expert_selection, unpermuted_source_rows, permuted_experts, dest_to_source,
            expert_first_token_offset, num_tokens, mNumExperts, mNumExpertsPerNode, mK, mSorter, sorter_ws, stream);

        check_cuda_error();

        int grid_dim = (num_expanded_tokens + num_threads - 1) / num_threads;
        buildReverseMap<<<grid_dim, num_threads, 0, stream>>>(source_to_dest, dest_to_source, num_expanded_tokens);
    }
}

std::vector<size_t> GemmProfilerBackend::getProfilerWorkspaces(int maxM, bool is_hopper)
{
    size_t k = mK;
    size_t num_expanded_tokens = maxM * k;

    size_t dtype_bytes = tensorrt_llm::common::getDTypeSize(mDType);
    float weight_bytes
        = mWType == nvinfer1::DataType::kINT4 ? 0.5f : static_cast<float>(tensorrt_llm::common::getDTypeSize(mWType));
    size_t output_bytes = tensorrt_llm::common::getDTypeSize(mOType);
    size_t gemm_output_bytes = (mOType == nvinfer1::DataType::kFP8)
        ? sizeof(HopperGroupedGemmInput::OutputTypeAdaptor_t<__nv_fp8_e4m3>)
        : output_bytes;

    size_t hidden_size = mExpertHiddenSize;
    size_t inter_size = mExpertInterSize; // Already divided by TP
    size_t num_experts_per_node = mNumExpertsPerNode;

    size_t fc1_out_size = inter_size;
    if (isGatedActivation(mActivationType))
    {
        fc1_out_size = inter_size * 2;
    }

    // TODO Needs updated when gather/finalize fusion is integrated
    size_t input_size1 = hidden_size * num_expanded_tokens * dtype_bytes;
    size_t output_size1 = inter_size * num_expanded_tokens * dtype_bytes;

    size_t input_size2 = inter_size * num_expanded_tokens * dtype_bytes;
    size_t output_size2 = hidden_size * output_bytes;

    size_t input_size = mGemmToProfile == GemmToProfile::GEMM_1 ? input_size1 : input_size2;
    size_t output_size = mGemmToProfile == GemmToProfile::GEMM_1 ? output_size1 : output_size2;

    // This may allocate a pointer when not required. That's fine it will be ignored at the cost of some memory
    size_t intermediate_size1 = fc1_out_size * num_expanded_tokens * gemm_output_bytes; // Note gemm_output_bytes
    size_t intermediate_size2 = hidden_size * num_expanded_tokens * gemm_output_bytes;  // Note gemm_output_bytes

    size_t intermediate_size = mGemmToProfile == GemmToProfile::GEMM_1 ? intermediate_size1 : intermediate_size2;

    size_t weights_1 = hidden_size * fc1_out_size * num_experts_per_node * weight_bytes;
    size_t bias_1 = mBias ? fc1_out_size * num_experts_per_node * dtype_bytes : 0;
    if (mUseLora && !is_hopper)
        bias_1 = output_size1;
    size_t weights_2 = hidden_size * inter_size * num_experts_per_node * weight_bytes;
    size_t bias_2 = mBias ? hidden_size * num_experts_per_node * dtype_bytes : 0;

    size_t weights = mGemmToProfile == GemmToProfile::GEMM_1 ? weights_1 : weights_2;
    size_t bias = mGemmToProfile == GemmToProfile::GEMM_1 ? bias_1 : bias_2;

    // TODO Make quant 2 & 4 bigger for FP8 if we ever change to scaling per expert
    bool is_int_w_quant = mWType == nvinfer1::DataType::kINT8 || mWType == nvinfer1::DataType::kINT4;
    bool is_fp8_w_quant = mWType == nvinfer1::DataType::kFP8;

    // Int sizes
    size_t quant_1 = is_int_w_quant ? fc1_out_size * num_experts_per_node * dtype_bytes : 0;
    size_t quant_2 = is_int_w_quant ? hidden_size * num_experts_per_node * dtype_bytes : 0;

    // FP8 sizes
    quant_1 = is_fp8_w_quant ? num_experts_per_node * sizeof(float) : quant_1;
    quant_2 = is_fp8_w_quant ? sizeof(float) : quant_2;
    size_t quant_3 = is_fp8_w_quant ? num_experts_per_node * sizeof(float) : 0;
    size_t quant_4 = 0; // Currently ignored by the GEMM

    size_t hopper_workspace_size = 0;
    if (is_hopper)
    {
        hopper_workspace_size = HopperGroupedGemmInput::workspaceSize(num_experts_per_node);
    }

    size_t alpha_scale_ptr_array_size = num_experts_per_node * sizeof(float**);
    size_t gemm_workspace_size = mInterface->getGemmWorkspaceSize(num_experts_per_node);

    // Routing info
    size_t expert_first_token_offset_size = (num_experts_per_node + 1) * sizeof(int64_t) * NUM_ROUTING_SAMPLES;
    size_t map_size = NUM_ROUTING_SAMPLES * num_expanded_tokens * sizeof(int);
    size_t unpermuted_size = NUM_ROUTING_SAMPLES * num_expanded_tokens * sizeof(int);
    size_t permuted_size = num_expanded_tokens * sizeof(int);
    size_t sorter_ws_size = mSorter.getWorkspaceSize(num_expanded_tokens, mNumExperts);
    size_t token_topk_final_scale_size = num_expanded_tokens * sizeof(float);

    // Warning: order is sensitive. Routing info must be first and gemm_workspace_size must be last
    return {// These go first because they are needed for prepare
        expert_first_token_offset_size, map_size, map_size, unpermuted_size, unpermuted_size, permuted_size,
        permuted_size, sorter_ws_size, token_topk_final_scale_size,
        // The below are for the actual run
        input_size, output_size, intermediate_size, weights, bias, quant_1, quant_2, quant_3, quant_4,
        hopper_workspace_size, alpha_scale_ptr_array_size, gemm_workspace_size};
}

size_t GemmProfilerBackend::getWorkspaceSize(int maxM)
{
    auto sizes = getProfilerWorkspaces(maxM, mSM >= 90);
    return calculateTotalWorkspaceSize(sizes.data(), sizes.size());
}

std::function<void*()> GemmProfilerBackend::getWorkspacePointerGenerator(char* ws, int maxM, bool is_hopper)
{
    int8_t* workspace_ptr = reinterpret_cast<int8_t*>(ws);
    auto workspaces = getProfilerWorkspaces(maxM, is_hopper);
    auto index = 0;
    auto getNext = [=]() mutable -> void*
    {
        TLLM_CHECK_WITH_INFO(index < workspaces.size(), "Mismatching scratch space allocation");
        auto res = workspace_ptr;
        size_t element_size_bytes = workspaces[index];
        workspace_ptr = nextWorkspacePtr(workspace_ptr, element_size_bytes);
        index++;
        // Return nullptr if size is 0
        return element_size_bytes != 0 ? res : nullptr;
    };
    return getNext;
}

void GemmProfilerBackend::runProfiler(
    int original_num_tokens, Config const& tactic, char* workspace_ptr_char, cudaStream_t const& stream)
{
    int64_t expanded_num_tokens = original_num_tokens * mK;
    int64_t num_experts_per_node = mNumExpertsPerNode;

    mSampleIndex = (mSampleIndex + 1) % NUM_ROUTING_SAMPLES;

    auto workspaces = getProfilerWorkspaces(original_num_tokens, tactic.is_sm90);
    auto getNext = getWorkspacePointerGenerator(workspace_ptr_char, original_num_tokens, tactic.is_sm90);
    // Routing goes first as we need to manually initialise it in prepare(), everything else can be uninit
    // If we didn't init routing all the values could go to one expert, causing the profile to be unreliable
    auto const* expert_first_token_offset
        = static_cast<int64_t const*>(getNext()) + mSampleIndex * (mNumExpertsPerNode + 1);
    auto const* source_to_dest = static_cast<int const*>(getNext()) + mSampleIndex * expanded_num_tokens;
    auto const* dest_to_source = static_cast<int const*>(getNext()) + mSampleIndex * expanded_num_tokens;
    auto const* expert_for_source_row = static_cast<int const*>(getNext()) + mSampleIndex * expanded_num_tokens;

    std::ignore = getNext(); // Only used in prepare()
    std::ignore = getNext(); // Only used in prepare()
    std::ignore = getNext(); // Only used in prepare()
    std::ignore = getNext(); // Only used in prepare()

    // These are uninitialised so just alias them as they don't matter for performance
    auto const* token_topk_unpermuted_scales = static_cast<float const*>(getNext());
    auto const* token_topk_permuted_scales = token_topk_unpermuted_scales;

    void const* inputs = getNext();
    void* outputs = getNext();
    void* intermediate = getNext();
    void const* weights = getNext();
    void const* bias = getNext();
    void const* scale_1 = getNext();
    void const* scale_2 = getNext();
    void const* scale_3 = getNext();
    void const* scale_4 = getNext();
    void* hopper_workspace = getNext();
    float const** alpha_scale_ptr_array = reinterpret_cast<float const**>(getNext());
    void* gemm_workspace = getNext(); // NOTE we rely on this being last below (i.e. workspaces.back())

    HopperGroupedGemmInput hopper_input_template;
    if (tactic.is_sm90)
    {
        hopper_input_template.configureWorkspace(
            static_cast<int8_t*>(hopper_workspace), num_experts_per_node, gemm_workspace, workspaces.back());
    }

    QuantParams quant_params;
    if (mWType == nvinfer1::DataType::kINT8 || mWType == nvinfer1::DataType::kINT4)
    {
        TLLM_CHECK(scale_1 && scale_2);
        quant_params = QuantParams::Int(scale_1, scale_2, scale_3, scale_4, 128);
    }
    else if (mWType == nvinfer1::DataType::kFP8)
    {
        TLLM_CHECK(scale_1 && scale_2 && scale_3);
        quant_params = QuantParams::FP8(static_cast<float const*>(scale_1), static_cast<float const*>(scale_2),
            static_cast<float const*>(scale_3), static_cast<float const*>(scale_4));
    }

    mInterface->is_profiler = true;
    if (mGemmToProfile == GemmToProfile::GEMM_1)
    {
        mInterface->gemm1(inputs, outputs, intermediate, expert_first_token_offset, hopper_input_template, weights,
            bias, expert_first_token_offset + num_experts_per_node, quant_params.fc1_weight_scales,
            quant_params.fc1_weight_zeros, quant_params.group_size, quant_params.dequant_fc1, quant_params.quant_fc2,
            expanded_num_tokens, mExpertHiddenSize, mExpertInterSize, num_experts_per_node, mActivationType,
            alpha_scale_ptr_array, !mUseLora, stream, tactic);
    }
    else
    {
        TLLM_CHECK(mGemmToProfile == GemmToProfile::GEMM_2);
        mInterface->gemm2(inputs, intermediate, outputs, expert_first_token_offset, hopper_input_template, weights,
            bias, quant_params.fc2_weight_scales, quant_params.fc2_weight_zeros, quant_params.group_size,
            quant_params.dequant_fc2, token_topk_unpermuted_scales, token_topk_permuted_scales, source_to_dest,
            dest_to_source, expert_for_source_row, expert_first_token_offset + mNumExpertsPerNode, original_num_tokens,
            expanded_num_tokens, mExpertHiddenSize, mExpertInterSize, num_experts_per_node, mK,
            !mInterface->use_deterministic_hopper_reduce_, alpha_scale_ptr_array, false, nullptr, stream,
            mParallelismConfig, tactic);
    }
    mInterface->is_profiler = false;

    check_cuda_error();
}

} // namespace tensorrt_llm::kernels
