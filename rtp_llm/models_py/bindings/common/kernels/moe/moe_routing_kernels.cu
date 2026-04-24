/*
 * Routing kernel implementations extracted from cutlass MOE code.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cuda.h>
#include <cuda_fp16.h>
#include <float.h>
#include <math.h>

#include "rtp_llm/models_py/bindings/common/kernels/moe/moe_routing_kernels.h"
#include "rtp_llm/models_py/bindings/cuda/trt_utils.h"

#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#elif (CUDART_VERSION >= 11050)
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/util_type.cuh>
#else
#include "3rdparty/cub/cub.cuh"
#include "3rdparty/cub/device/device_radix_sort.cuh"
#include "3rdparty/cub/util_type.cuh"
#endif

using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::common;

namespace tensorrt_llm::kernels {

static constexpr int WARP_SIZE = 32;

// ====================== Softmax ======================

template<int TPB>
__launch_bounds__(TPB) __global__
    void moeSoftmax(float const* input, bool const* finished, float* output, int64_t const num_cols) {
    using BlockReduce = cub::BlockReduce<float, TPB>;
    __shared__ typename BlockReduce::TempStorage tmpStorage;
    __shared__ float                             normalizing_factor;
    __shared__ float                             float_max;

    int const     tidx              = threadIdx.x;
    int64_t const bidx              = blockIdx.x;
    int64_t const thread_row_offset = bidx * num_cols;

    cub::Sum sum;
    float    threadData(-FLT_MAX);

    if ((finished != nullptr) && finished[bidx]) {
        return;
    }

    for (int ii = tidx; ii < num_cols; ii += TPB) {
        int64_t const idx = thread_row_offset + ii;
        threadData        = max(input[idx], threadData);
    }

    float const maxElem = BlockReduce(tmpStorage).Reduce(threadData, cub::Max());
    if (tidx == 0) {
        float_max = maxElem;
    }
    __syncthreads();

    threadData = 0.f;
    for (int ii = tidx; ii < num_cols; ii += TPB) {
        int64_t const idx = thread_row_offset + ii;
        threadData += exp((static_cast<float>(input[idx]) - float_max));
    }

    auto const Z = BlockReduce(tmpStorage).Reduce(threadData, sum);
    if (tidx == 0) {
        normalizing_factor = 1.f / Z;
    }
    __syncthreads();

    for (int ii = tidx; ii < num_cols; ii += TPB) {
        int64_t const idx = thread_row_offset + ii;
        float const   val = exp((static_cast<float>(input[idx]) - float_max)) * normalizing_factor;
        output[idx]       = val;
    }
}

// ====================== TopK ======================

template<int TPB, typename TOPK_T>
__launch_bounds__(TPB) __global__ void moeTopK(float const*                    inputs_after_softmax,
                                               bool const*                     finished,
                                               float*                          output,
                                               TOPK_T*                         indices,
                                               int*                            source_rows,
                                               int const                       num_experts,
                                               int const                       k,
                                               int const                       startk,
                                               int const                       endk,
                                               int const                       start_expert,
                                               int const                       end_expert,
                                               MOEExpertScaleNormalizationMode norm_mode) {
    using cub_kvp     = cub::KeyValuePair<int, float>;
    using BlockReduce = cub::BlockReduce<cub_kvp, TPB>;
    __shared__ typename BlockReduce::TempStorage tmpStorage;

    cub_kvp     thread_kvp;
    cub::ArgMax arg_max;

    int64_t const num_rows  = gridDim.x;
    int64_t const block_row = blockIdx.x;

    float         renorm_value       = 0.0f;
    bool const    row_is_active      = finished ? !finished[block_row] : true;
    int64_t const thread_read_offset = block_row * num_experts;
    int64_t const indice_offset      = block_row * k;
    for (int k_idx = startk; k_idx < endk; ++k_idx) {
        thread_kvp.key   = 0;
        thread_kvp.value = -1.f;

        cub_kvp inp_kvp;
        for (int expert = threadIdx.x; expert < num_experts; expert += TPB) {
            int64_t const idx = thread_read_offset + expert;
            inp_kvp.key       = expert;
            inp_kvp.value     = inputs_after_softmax[idx];

            for (int prior_k = startk; prior_k < k_idx; ++prior_k) {
                int prior_winning_expert = static_cast<int>(indices[indice_offset + prior_k]);
                prior_winning_expert     = prior_winning_expert >= num_experts ? prior_winning_expert - num_experts :
                                                                                 prior_winning_expert + start_expert;
                if (prior_winning_expert == expert) {
                    inp_kvp = thread_kvp;
                }
            }
            thread_kvp = arg_max(inp_kvp, thread_kvp);
        }

        cub_kvp const result_kvp = BlockReduce(tmpStorage).Reduce(thread_kvp, arg_max);
        if (threadIdx.x == 0) {
            int const  expert             = result_kvp.key;
            bool const node_uses_expert   = expert >= start_expert && expert < end_expert;
            bool const should_process_row = row_is_active && node_uses_expert;

            int64_t const idx = indice_offset + k_idx;
            output[idx]       = result_kvp.value;
            indices[idx]      = should_process_row ? static_cast<TOPK_T>(expert - start_expert) :
                                                     static_cast<TOPK_T>(num_experts + expert);
            assert(indices[idx] >= 0);
            source_rows[idx] = k_idx * num_rows + block_row;

            if (norm_mode == MOEExpertScaleNormalizationMode::RENORMALIZE) {
                renorm_value += result_kvp.value;
            }
        }
        __syncthreads();
    }

    if (norm_mode == MOEExpertScaleNormalizationMode::RENORMALIZE && threadIdx.x == 0 && renorm_value != 0.f) {
        assert(startk == 0 && endk == k);
        renorm_value = 1 / renorm_value;
        for (int k_idx = 0; k_idx < k; k_idx++) {
            int64_t const idx = indice_offset + k_idx;
            output[idx] *= renorm_value;
        }
    }
}

// moeTopK with bias
template<int TPB>
__launch_bounds__(TPB) __global__ void moeTopK(float const*                    inputs_after_softmax,
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
                                               MOEExpertScaleNormalizationMode norm_mode) {
    using cub_kvp     = cub::KeyValuePair<int, float>;
    using BlockReduce = cub::BlockReduce<cub_kvp, TPB>;
    __shared__ typename BlockReduce::TempStorage tmpStorage;

    cub_kvp     thread_kvp;
    cub::ArgMax arg_max;

    int64_t const num_rows  = gridDim.x;
    int64_t const block_row = blockIdx.x;

    float         renorm_value       = 0.0f;
    bool const    row_is_active      = finished ? !finished[block_row] : true;
    int64_t const thread_read_offset = block_row * num_experts;
    int64_t const indice_offset      = block_row * k;
    for (int k_idx = startk; k_idx < endk; ++k_idx) {
        thread_kvp.key   = 0;
        thread_kvp.value = -1.f;

        cub_kvp inp_kvp;
        for (int expert = threadIdx.x; expert < num_experts; expert += TPB) {
            int64_t const idx = thread_read_offset + expert;
            inp_kvp.key       = expert;
            inp_kvp.value     = input_with_bias[idx];

            for (int prior_k = startk; prior_k < k_idx; ++prior_k) {
                int prior_winning_expert = indices[indice_offset + prior_k];
                prior_winning_expert     = prior_winning_expert >= num_experts ? prior_winning_expert - num_experts :
                                                                                 prior_winning_expert + start_expert;
                if (prior_winning_expert == expert) {
                    inp_kvp = thread_kvp;
                }
            }
            thread_kvp = arg_max(inp_kvp, thread_kvp);
        }

        cub_kvp const result_kvp = BlockReduce(tmpStorage).Reduce(thread_kvp, arg_max);
        if (threadIdx.x == 0) {
            int const     expert             = result_kvp.key;
            int64_t const origin_idx         = thread_read_offset + expert;
            bool const    node_uses_expert   = expert >= start_expert && expert < end_expert;
            bool const    should_process_row = row_is_active && node_uses_expert;

            int64_t const idx = indice_offset + k_idx;
            output[idx]       = inputs_after_softmax[origin_idx];
            indices[idx]      = should_process_row ? (expert - start_expert) : (num_experts + expert);
            assert(indices[idx] >= 0);
            source_rows[idx] = k_idx * num_rows + block_row;

            if (norm_mode == MOEExpertScaleNormalizationMode::RENORMALIZE) {
                renorm_value += inputs_after_softmax[origin_idx];
            }
        }
        __syncthreads();
    }

    if (norm_mode == MOEExpertScaleNormalizationMode::RENORMALIZE) {
        renorm_value += 1e-20;
        if (threadIdx.x == 0 && renorm_value != 0.f) {
            assert(startk == 0 && endk == k);
            renorm_value = 1 / renorm_value;
            for (int k_idx = 0; k_idx < k; k_idx++) {
                int64_t const idx = indice_offset + k_idx;
                output[idx] *= renorm_value;
            }
        }
    }
}

// ====================== TopK Gating Softmax (fused) ======================

namespace detail {
template<int EXPERTS, int BYTES_PER_LDG>
struct TopkConstants {
    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(float);
    static_assert(EXPERTS / (ELTS_PER_LDG * WARP_SIZE) == 0 || EXPERTS % (ELTS_PER_LDG * WARP_SIZE) == 0);
    static constexpr int VECs_PER_THREAD = std::max(1, EXPERTS / (ELTS_PER_LDG * WARP_SIZE));
    static constexpr int VPT             = VECs_PER_THREAD * ELTS_PER_LDG;
    static constexpr int THREADS_PER_ROW = EXPERTS / VPT;
    static constexpr int ROWS_PER_WARP   = WARP_SIZE / THREADS_PER_ROW;
};
}  // namespace detail

template<int VPT, int NUM_EXPERTS, int WARPS_PER_CTA, int BYTES_PER_LDG, typename TOPK_T>
__launch_bounds__(WARPS_PER_CTA* WARP_SIZE) __global__
    void topkGatingSoftmax(float const*                    input,
                           bool const*                     finished,
                           float*                          output,
                           int64_t const                   num_rows,
                           TOPK_T*                         indices,
                           int*                            source_rows,
                           int const                       k,
                           int const                       startk,
                           int const                       endk,
                           int const                       start_expert,
                           int const                       end_expert,
                           MOEExpertScaleNormalizationMode norm_mode) {
    static_assert(VPT == (VPT & -VPT));
    static_assert(NUM_EXPERTS == (NUM_EXPERTS & -NUM_EXPERTS));
    static_assert(BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG));
    static_assert(BYTES_PER_LDG <= 16);

    static constexpr int ELTS_PER_LDG    = BYTES_PER_LDG / sizeof(float);
    static constexpr int ELTS_PER_ROW    = NUM_EXPERTS;
    static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT;
    static constexpr int LDG_PER_THREAD  = VPT / ELTS_PER_LDG;
    static constexpr int ELTS_PER_WARP   = WARP_SIZE * VPT;
    static constexpr int ROWS_PER_WARP   = ELTS_PER_WARP / ELTS_PER_ROW;
    static constexpr int ROWS_PER_CTA    = WARPS_PER_CTA * ROWS_PER_WARP;

    int64_t const cta_base_row       = blockIdx.x * ROWS_PER_CTA;
    int64_t const warp_base_row      = cta_base_row + threadIdx.y * ROWS_PER_WARP;
    int const     thread_row_in_warp = threadIdx.x / THREADS_PER_ROW;
    int64_t const thread_row         = warp_base_row + thread_row_in_warp;

    if (thread_row >= num_rows)
        return;
    bool const row_is_active = finished ? !finished[thread_row] : true;

    float const* thread_row_ptr           = input + thread_row * ELTS_PER_ROW;
    int const    thread_group_idx         = threadIdx.x % THREADS_PER_ROW;
    int const    first_elt_read_by_thread = thread_group_idx * ELTS_PER_LDG;
    float const* thread_read_ptr          = thread_row_ptr + first_elt_read_by_thread;

    using AccessType = cutlass::AlignedArray<float, ELTS_PER_LDG>;

    cutlass::Array<float, VPT> row_chunk;
    AccessType*                row_chunk_vec_ptr   = reinterpret_cast<AccessType*>(&row_chunk);
    AccessType const*          vec_thread_read_ptr = reinterpret_cast<AccessType const*>(thread_read_ptr);
#pragma unroll
    for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
        row_chunk_vec_ptr[ii] = vec_thread_read_ptr[ii * THREADS_PER_ROW];
    }

    float thread_max = row_chunk[0];
#pragma unroll
    for (int ii = 1; ii < VPT; ++ii) {
        thread_max = max(thread_max, row_chunk[ii]);
    }

#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
        thread_max = max(thread_max, __shfl_xor_sync(0xFFFFFFFF, thread_max, mask, THREADS_PER_ROW));
    }

    float row_sum = 0;
#pragma unroll
    for (int ii = 0; ii < VPT; ++ii) {
        row_chunk[ii] = expf(row_chunk[ii] - thread_max);
        row_sum += row_chunk[ii];
    }

#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
        row_sum += __shfl_xor_sync(0xFFFFFFFF, row_sum, mask, THREADS_PER_ROW);
    }

    float const reciprocal_row_sum = 1.f / row_sum;
#pragma unroll
    for (int ii = 0; ii < VPT; ++ii) {
        row_chunk[ii] = row_chunk[ii] * reciprocal_row_sum;
    }

    int                  start_col          = first_elt_read_by_thread;
    static constexpr int COLS_PER_GROUP_LDG = ELTS_PER_LDG * THREADS_PER_ROW;

    float renorm_value = 0.0f;

    for (int k_idx = startk; k_idx < endk; ++k_idx) {
        float max_val = row_chunk[0];
        int   expert  = start_col;
#pragma unroll
        for (int ldg = 0, col = start_col; ldg < LDG_PER_THREAD; ++ldg, col += COLS_PER_GROUP_LDG) {
#pragma unroll
            for (int ii = 0; ii < ELTS_PER_LDG; ++ii) {
                float val = row_chunk[ldg * ELTS_PER_LDG + ii];
                if (val > max_val) {
                    max_val = val;
                    expert  = col + ii;
                }
            }
        }

#pragma unroll
        for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
            float other_max    = __shfl_xor_sync(0xFFFFFFFF, max_val, mask, THREADS_PER_ROW);
            int   other_expert = __shfl_xor_sync(0xFFFFFFFF, expert, mask, THREADS_PER_ROW);
            if (other_max > max_val || (other_max == max_val && other_expert < expert)) {
                max_val = other_max;
                expert  = other_expert;
            }
        }

        if (thread_group_idx == 0) {
            bool const node_uses_expert   = expert >= start_expert && expert < end_expert;
            bool const should_process_row = row_is_active && node_uses_expert;

            int64_t const idx = k * thread_row + k_idx;
            output[idx]       = max_val;
            indices[idx]      = should_process_row ? (expert - start_expert) : (NUM_EXPERTS + expert);
            source_rows[idx]  = k_idx * num_rows + thread_row;

            if (norm_mode == MOEExpertScaleNormalizationMode::RENORMALIZE) {
                renorm_value += max_val;
            }
        }

        if (k_idx + 1 < endk) {
            int const ldg_group_for_expert     = expert / COLS_PER_GROUP_LDG;
            int const thread_to_clear_in_group = (expert / ELTS_PER_LDG) % THREADS_PER_ROW;
            if (thread_group_idx == thread_to_clear_in_group) {
                int const offset_for_expert                                        = expert % ELTS_PER_LDG;
                row_chunk[ldg_group_for_expert * ELTS_PER_LDG + offset_for_expert] = -10000.f;
            }
        }
    }

    if (norm_mode == MOEExpertScaleNormalizationMode::RENORMALIZE && thread_group_idx == 0 && renorm_value != 0.f) {
        assert(startk == 0 && endk == k);
        renorm_value = 1 / renorm_value;
        for (int k_idx = 0; k_idx < k; k_idx++) {
            int64_t const idx = k * thread_row + k_idx;
            output[idx] *= renorm_value;
        }
    }
}

// ====================== Launcher helpers ======================

template<int EXPERTS, int WARPS_PER_TB, typename TOPK_T>
void topkGatingSoftmaxLauncherHelper(float const*                    input,
                                     bool const*                     finished,
                                     float*                          output,
                                     TOPK_T*                         indices,
                                     int*                            source_row,
                                     int64_t const                   num_rows,
                                     int const                       k,
                                     int const                       startk,
                                     int const                       endk,
                                     int const                       start_expert,
                                     int const                       end_expert,
                                     MOEExpertScaleNormalizationMode norm_mode,
                                     cudaStream_t                    stream) {
    static constexpr std::size_t MAX_BYTES_PER_LDG = 16;
    static constexpr int         BYTES_PER_LDG     = std::min(MAX_BYTES_PER_LDG, sizeof(float) * EXPERTS);
    using Constants                                = detail::TopkConstants<EXPERTS, BYTES_PER_LDG>;
    static constexpr int VPT                       = Constants::VPT;
    static constexpr int ROWS_PER_WARP             = Constants::ROWS_PER_WARP;
    int64_t const        num_warps                 = (num_rows + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
    int64_t const        num_blocks                = (num_warps + WARPS_PER_TB - 1) / WARPS_PER_TB;
    dim3                 block_dim(WARP_SIZE, WARPS_PER_TB);
    topkGatingSoftmax<VPT, EXPERTS, WARPS_PER_TB, BYTES_PER_LDG><<<num_blocks, block_dim, 0, stream>>>(
        input, finished, output, num_rows, indices, source_row, k, startk, endk, start_expert, end_expert, norm_mode);
}

template<typename TOPK_T>
void topkGatingSoftmaxKernelLauncher(float const*                    input,
                                     float*                          output,
                                     float*                          softmax_temp_output,
                                     TOPK_T*                         indices,
                                     int*                            source_row,
                                     int64_t const                   num_rows,
                                     int const                       num_experts,
                                     int const                       k,
                                     int const                       startk,
                                     int const                       endk,
                                     int const                       start_expert,
                                     int const                       end_expert,
                                     MOEExpertScaleNormalizationMode norm_mode,
                                     cudaStream_t                    stream) {
    static constexpr int WARPS_PER_TB = 4;

#define CASE(N)                                                                                                        \
    case N:                                                                                                            \
        topkGatingSoftmaxLauncherHelper<N, WARPS_PER_TB>(input,                                                        \
                                                         nullptr,                                                      \
                                                         output,                                                       \
                                                         indices,                                                      \
                                                         source_row,                                                   \
                                                         num_rows,                                                     \
                                                         k,                                                            \
                                                         startk,                                                       \
                                                         endk,                                                         \
                                                         start_expert,                                                 \
                                                         end_expert,                                                   \
                                                         norm_mode,                                                    \
                                                         stream);                                                      \
        break;

    switch (num_experts) {
        CASE(1) CASE(2) CASE(4) CASE(8) CASE(16) CASE(32) CASE(64) CASE(128) CASE(256) default: {
            static constexpr int TPB = 256;
            TLLM_CHECK(softmax_temp_output != nullptr);
            moeSoftmax<TPB><<<num_rows, TPB, 0, stream>>>(input, nullptr, softmax_temp_output, num_experts);
            moeTopK<TPB><<<num_rows, TPB, 0, stream>>>(softmax_temp_output,
                                                       nullptr,
                                                       output,
                                                       indices,
                                                       source_row,
                                                       num_experts,
                                                       k,
                                                       startk,
                                                       endk,
                                                       start_expert,
                                                       end_expert,
                                                       norm_mode);
        }
    }
#undef CASE
}

template<int TPB>
void topkKernelLauncherHelper(float const*                    input,
                              float const*                    input_with_bias,
                              float*                          output,
                              float*                          softmax_temp_output,
                              int*                            indices,
                              int*                            source_row,
                              int64_t const                   num_rows,
                              int const                       num_experts,
                              int const                       k,
                              int const                       startk,
                              int const                       endk,
                              int const                       start_expert,
                              int const                       end_expert,
                              MOEExpertScaleNormalizationMode norm_mode,
                              cudaStream_t                    stream) {
    moeTopK<TPB><<<num_rows, TPB, 0, stream>>>(input,
                                               input_with_bias,
                                               nullptr,
                                               output,
                                               indices,
                                               source_row,
                                               num_experts,
                                               k,
                                               startk,
                                               endk,
                                               start_expert,
                                               end_expert,
                                               norm_mode);
}

template<typename TOPK_T>
void topkKernelLauncher(float const*                    input,
                        float const*                    input_with_bias,
                        float*                          output,
                        float*                          softmax_temp_output,
                        TOPK_T*                         indices,
                        int*                            source_row,
                        int64_t const                   num_rows,
                        int const                       num_experts,
                        int const                       k,
                        int const                       startk,
                        int const                       endk,
                        int const                       start_expert,
                        int const                       end_expert,
                        MOEExpertScaleNormalizationMode norm_mode,
                        cudaStream_t                    stream) {

#define CASE(N)                                                                                                        \
    case N:                                                                                                            \
        topkKernelLauncherHelper<N>(input,                                                                             \
                                    input_with_bias,                                                                   \
                                    output,                                                                            \
                                    softmax_temp_output,                                                               \
                                    (int*)indices,                                                                     \
                                    source_row,                                                                        \
                                    num_rows,                                                                          \
                                    num_experts,                                                                       \
                                    k,                                                                                 \
                                    startk,                                                                            \
                                    endk,                                                                              \
                                    start_expert,                                                                      \
                                    end_expert,                                                                        \
                                    norm_mode,                                                                         \
                                    stream);                                                                           \
        break;

    switch (num_experts) {
        CASE(1) CASE(2) CASE(4) CASE(8) CASE(16) CASE(32) CASE(64) CASE(128) CASE(256) default: {
            topkKernelLauncherHelper<256>(input,
                                          input_with_bias,
                                          output,
                                          softmax_temp_output,
                                          (int*)indices,
                                          source_row,
                                          num_rows,
                                          num_experts,
                                          k,
                                          startk,
                                          endk,
                                          start_expert,
                                          end_expert,
                                          norm_mode,
                                          stream);
        }
    }
#undef CASE
}

// ====================== Sparse Mixer ======================

__global__ void sparseMixerMask(float const* input,
                                float*       output,
                                int const*   indices,
                                int          k_idx,
                                int          k,
                                int          num_tokens,
                                int          num_experts,
                                int          start_expert,
                                float        epsilon) {
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_idx >= num_tokens)
        return;

    int last_selected = (k_idx > 0) ? indices[k * token_idx + (k_idx - 1)] : INT_MIN;
    last_selected     = last_selected >= num_experts ? last_selected - num_experts : last_selected + start_expert;

    float max_val = -INFINITY;
    for (int i = 0; i < num_experts; ++i) {
        if (i != last_selected) {
            float const val = input[token_idx * num_experts + i];
            max_val         = max(val, max_val);
        }
    }

    for (int i = 0; i < num_experts; ++i) {
        float val                           = input[token_idx * num_experts + i];
        float mask                          = (max_val - val) / max(abs(val), max_val);
        bool  mask_value                    = (mask > 2 * epsilon) || i == last_selected;
        output[token_idx * num_experts + i] = mask_value ? -INFINITY : val;
    }
}

void sparseMixerTopkSoftmax(float const*  input,
                            float*        output,
                            float*        mixer_temp_output,
                            float*        softmax_temp_output,
                            int*          indices,
                            int*          source_row,
                            int64_t const num_rows,
                            int const     num_experts,
                            int const     k,
                            int const     start_expert,
                            int const     end_expert,
                            float         epsilon,
                            cudaStream_t  stream) {
    TLLM_CHECK_WITH_INFO(k <= 2, "Current sparse mixer only supports k <= 2");
    constexpr int threads_per_block = 256;
    int           num_blocks        = ceilDiv(num_rows, threads_per_block);
    for (int k_idx = 0; k_idx < k; ++k_idx) {
        sparseMixerMask<<<num_blocks, threads_per_block, 0, stream>>>(
            input, mixer_temp_output, indices, k_idx, k, num_rows, num_experts, start_expert, epsilon);
        topkGatingSoftmaxKernelLauncher(mixer_temp_output,
                                        output,
                                        softmax_temp_output,
                                        indices,
                                        source_row,
                                        num_rows,
                                        num_experts,
                                        k,
                                        k_idx,
                                        k_idx + 1,
                                        start_expert,
                                        end_expert,
                                        MOEExpertScaleNormalizationMode::NONE,
                                        stream);
    }
}

// ====================== invokeSelectExpertsForTokens ======================

template<typename TOPK_T>
void invokeSelectExpertsForTokens(float const*                    input,
                                  float const*                    input_with_bias,
                                  float*                          output,
                                  float*                          mixer_temp_output,
                                  float*                          softmax_temp_output,
                                  TOPK_T*                         indices,
                                  int*                            source_row,
                                  int64_t const                   num_rows,
                                  int const                       num_experts,
                                  int const                       k,
                                  int const                       start_expert,
                                  int const                       end_expert,
                                  float                           mixer_epsilon,
                                  MOEExpertScaleNormalizationMode norm_mode,
                                  cudaStream_t                    stream) {
    if (input == input_with_bias) {
        if (norm_mode == MOEExpertScaleNormalizationMode::SPARSE_MIXER) {
            TLLM_CHECK_WITH_INFO(mixer_temp_output, "Sparse mixer output is null when running sparse mixer");
            sparseMixerTopkSoftmax(input,
                                   output,
                                   mixer_temp_output,
                                   softmax_temp_output,
                                   (int*)indices,
                                   source_row,
                                   num_rows,
                                   num_experts,
                                   k,
                                   start_expert,
                                   end_expert,
                                   mixer_epsilon,
                                   stream);
        } else {
            topkGatingSoftmaxKernelLauncher(input,
                                            output,
                                            softmax_temp_output,
                                            indices,
                                            source_row,
                                            num_rows,
                                            num_experts,
                                            k,
                                            0,
                                            k,
                                            start_expert,
                                            end_expert,
                                            norm_mode,
                                            stream);
        }
    } else {
        topkKernelLauncher(input,
                           input_with_bias,
                           output,
                           softmax_temp_output,
                           (int*)indices,
                           source_row,
                           num_rows,
                           num_experts,
                           k,
                           0,
                           k,
                           start_expert,
                           end_expert,
                           norm_mode,
                           stream);
    }
    check_cuda_error();
}

template void invokeSelectExpertsForTokens<int>(float const*,
                                                float const*,
                                                float*,
                                                float*,
                                                float*,
                                                int*,
                                                int*,
                                                int64_t const,
                                                int const,
                                                int const,
                                                int const,
                                                int const,
                                                float,
                                                MOEExpertScaleNormalizationMode,
                                                cudaStream_t);

template void invokeSelectExpertsForTokens<int64_t>(float const*,
                                                    float const*,
                                                    float*,
                                                    float*,
                                                    float*,
                                                    int64_t*,
                                                    int*,
                                                    int64_t const,
                                                    int const,
                                                    int const,
                                                    int const,
                                                    int const,
                                                    float,
                                                    MOEExpertScaleNormalizationMode,
                                                    cudaStream_t);

// ====================== CubKeyValueSorter ======================

CubKeyValueSorter::CubKeyValueSorter(): num_experts_(0), num_bits_(sizeof(int) * 8) {}

int CubKeyValueSorter::expertsToBits(int num_experts) {
    return static_cast<int>(log2(2 * num_experts - 1)) + 1;
}

CubKeyValueSorter::CubKeyValueSorter(int const num_experts):
    num_experts_(num_experts), num_bits_(expertsToBits(num_experts)) {}

void CubKeyValueSorter::updateNumExperts(int const num_experts) {
    num_experts_ = num_experts;
    num_bits_    = expertsToBits(num_experts);
}

size_t CubKeyValueSorter::getWorkspaceSize(size_t const num_key_value_pairs, int const num_experts) {
    int    num_bits         = expertsToBits(num_experts);
    size_t required_storage = 0;
    int*   null_int         = nullptr;
    cub::DeviceRadixSort::SortPairs(
        nullptr, required_storage, null_int, null_int, null_int, null_int, num_key_value_pairs, 0, num_bits);
    if (required_storage == 0) {
        required_storage = 1;
    }
    return required_storage;
}

void CubKeyValueSorter::run(void*        workspace,
                            size_t const workspace_size,
                            int const*   keys_in,
                            int*         keys_out,
                            int const*   values_in,
                            int*         values_out,
                            size_t const num_key_value_pairs,
                            cudaStream_t stream) {
    size_t expected_ws_size = getWorkspaceSize(num_key_value_pairs, num_experts_);
    size_t actual_ws_size   = workspace_size;
    TLLM_CHECK_WITH_INFO(expected_ws_size <= workspace_size,
                         "[CubKeyValueSorter::run] The allocated workspace is too small to run this problem.");
    cub::DeviceRadixSort::SortPairs(
        workspace, actual_ws_size, keys_in, keys_out, values_in, values_out, num_key_value_pairs, 0, num_bits_, stream);
}

// ====================== computeExpertFirstTokenOffset ======================

__global__ void computeExpertFirstTokenOffsetKernel(int const*    sorted_experts,
                                                    int64_t const sorted_experts_len,
                                                    int64_t const num_experts,
                                                    int64_t*      expert_first_token_offset) {
    int const expert = blockIdx.x * blockDim.x + threadIdx.x;
    if (expert >= num_experts + 1)
        return;
    expert_first_token_offset[expert] = findTotalEltsLessThanTarget(sorted_experts, sorted_experts_len, expert);
}

void computeExpertFirstTokenOffset(int const*   sorted_indices,
                                   int const    total_indices,
                                   int const    num_experts,
                                   int64_t*     expert_first_token_offset,
                                   cudaStream_t stream) {
    int const num_entries = num_experts + 1;
    int const threads     = std::min(1024, num_entries);
    int const blocks      = (num_entries + threads - 1) / threads;
    computeExpertFirstTokenOffsetKernel<<<blocks, threads, 0, stream>>>(
        sorted_indices, total_indices, num_experts, expert_first_token_offset);
}

// ====================== genSourceRow ======================

__global__ void genSourceRowKernel(int*   expert_rows,
                                   int*   source_rows,
                                   size_t token_num,
                                   size_t top_k,
                                   size_t num_experts,
                                   int    start_expert,
                                   int    end_expert) {
    size_t const idx       = blockIdx.x * blockDim.x + threadIdx.x;
    size_t const token_idx = idx / top_k;
    size_t const k_idx     = idx % top_k;
    if (idx < token_num * top_k) {
        if (expert_rows[idx] >= start_expert && expert_rows[idx] < end_expert) {
            expert_rows[idx] = static_cast<int>(expert_rows[idx] - start_expert);
        } else if (expert_rows[idx] < 0) {
            expert_rows[idx] = static_cast<int>(num_experts);
        } else {
            expert_rows[idx] = static_cast<int>(expert_rows[idx] + num_experts);
        }
        source_rows[idx] = static_cast<int>(k_idx * token_num + token_idx);
    }
}

void genSourceRow(int*         expert_rows,
                  int*         source_rows,
                  size_t       token_num,
                  size_t       top_k,
                  size_t       num_experts,
                  int          start_expert,
                  int          end_expert,
                  cudaStream_t stream) {
    size_t const threads = 256;
    size_t const blocks  = token_num * top_k / 256 + 1;
    genSourceRowKernel<<<blocks, threads, 0, stream>>>(
        expert_rows, source_rows, token_num, top_k, num_experts, start_expert, end_expert);
}

// ====================== sortAndScanSoftmaxOutput ======================

void sortAndScanSoftmaxOutput(int*               expert_for_source_row,
                              int*               source_rows,
                              int*               permuted_experts,
                              int*               permuted_rows,
                              int64_t*           expert_first_token_offset,
                              int64_t            num_rows,
                              int64_t            num_experts,
                              int64_t            num_experts_per_node,
                              int64_t            k,
                              CubKeyValueSorter& sorter,
                              void*              sorter_ws,
                              cudaStream_t       stream) {
    int64_t const expanded_num_rows = k * num_rows;
    sorter.updateNumExperts(num_experts);
    size_t const sorter_ws_size_bytes = pad_to_multiple_of_16(sorter.getWorkspaceSize(expanded_num_rows, num_experts));
    sorter.run((void*)sorter_ws,
               sorter_ws_size_bytes,
               expert_for_source_row,
               permuted_experts,
               source_rows,
               permuted_rows,
               expanded_num_rows,
               stream);
    check_cuda_error();
    computeExpertFirstTokenOffset(
        permuted_experts, expanded_num_rows, num_experts_per_node, expert_first_token_offset, stream);
}

}  // namespace tensorrt_llm::kernels
