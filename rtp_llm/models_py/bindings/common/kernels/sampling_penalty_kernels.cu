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

#include <assert.h>
#include <float.h>
#include <algorithm>
#include <cstdint>

// #include "driver_types.h"
#include "rtp_llm/models_py/bindings/common/kernels/sampling_penalty_kernels.h"

namespace rtp_llm {

// Two-dimensional row/vocabulary tiling follows vLLM's _temperature_kernel
// (Apache-2.0). Preserve RTP's reciprocal epsilon, multiplication and FTZ
// compilation unit; do not substitute vLLM's different temperature formula.
template<bool Vectorized, bool HasTempArray>
__global__ void samplerTiledTemperature(float* logits, const float* temperatures, float scalar_temperature, int vocab) {
    __shared__ float inverse;
    const int        row = blockIdx.x;
    if (threadIdx.x == 0) {
        const float temp = HasTempArray ? temperatures[row] : scalar_temperature;
        inverse          = 1.0f / (temp + 1e-6f);
    }
    __syncthreads();
    constexpr int tile       = 4096;
    float*        row_logits = logits + static_cast<int64_t>(row) * vocab;
#pragma unroll
    for (int j = 0; j < tile / (256 * 4); ++j) {
        const int col = blockIdx.y * tile + (j * 256 + threadIdx.x) * 4;
        if (col < vocab) {
            if constexpr (Vectorized) {
                float4 value = reinterpret_cast<float4*>(row_logits)[col / 4];
                value.x *= inverse;
                value.y *= inverse;
                value.z *= inverse;
                value.w *= inverse;
                reinterpret_cast<float4*>(row_logits)[col / 4] = value;
            } else {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    if (col + i < vocab) {
                        row_logits[col + i] *= inverse;
                    }
                }
            }
        }
    }
}

void invokeTiledTemperaturePenalty(
    float* logits, const float* temperatures, int batch_size, int vocab, cudaStream_t stream) {
    assert(batch_size > 0 && vocab > 0);
    const dim3 grid(batch_size, (vocab + 4095) / 4096);
    if (reinterpret_cast<uintptr_t>(logits) % 16 == 0 && vocab % 4 == 0)
        samplerTiledTemperature<true, true><<<grid, 256, 0, stream>>>(logits, temperatures, 1.0f, vocab);
    else
        samplerTiledTemperature<false, true><<<grid, 256, 0, stream>>>(logits, temperatures, 1.0f, vocab);
}

void invokeTiledTemperaturePenalty(
    float* logits, float scalar_temperature, int batch_size, int vocab, cudaStream_t stream) {
    assert(batch_size > 0 && vocab > 0);
    const dim3 grid(batch_size, (vocab + 4095) / 4096);
    if (reinterpret_cast<uintptr_t>(logits) % 16 == 0 && vocab % 4 == 0)
        samplerTiledTemperature<true, false><<<grid, 256, 0, stream>>>(logits, nullptr, scalar_temperature, vocab);
    else
        samplerTiledTemperature<false, false><<<grid, 256, 0, stream>>>(logits, nullptr, scalar_temperature, vocab);
}

template<typename T>
__global__ void batchApplyTemperaturePenalty(T*           logits,
                                             const T*     bias,
                                             const float* temperatures,
                                             const int    batch_size,
                                             const int    vocab_size,
                                             const int    vocab_size_padd) {
    // TODO: Add macro or device function to get MAX_T_VAL.
    const bool              IS_FP16   = std::is_same<T, half>::value;
    const T                 MAX_T_VAL = (IS_FP16) ? 65504.F : FLT_MAX;
    extern __shared__ float inv_temperatures[];
    for (int i = threadIdx.x; i < batch_size; i += blockDim.x) {
        inv_temperatures[i] = 1.0f / (temperatures[i] + 1e-6f);
    }
    __syncthreads();

    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < batch_size * vocab_size_padd;
         index += blockDim.x * gridDim.x) {
        int batch_idx = index / vocab_size_padd;
        int vocab_idx = index % vocab_size_padd;
        T   logit     = (vocab_idx < vocab_size) ? logits[index] : -MAX_T_VAL;
        if (vocab_idx < vocab_size) {
            if (bias != nullptr) {
                logit += bias[vocab_idx];
            }
            logit *= inv_temperatures[batch_idx];
        }
        logits[index] = logit;
    }
}

__global__ void batchApplyTemperaturePenalty_h2(half2*       logits,
                                                const half2* bias,
                                                const float* temperatures,
                                                const int    batch_size,
                                                const int    vocab_size,
                                                const int    vocab_size_padded) {
    assert(vocab_size % 2 == 0);
    assert(vocab_size_padded % 2 == 0);
    extern __shared__ half2 h2_inv_temperatures[];
    for (int i = threadIdx.x; i < batch_size; i += blockDim.x) {
        h2_inv_temperatures[i] = __float2half2_rn(1.f / (temperatures[i] + 1e-6f));
    }
    __syncthreads();

    const half2 mask_val               = __float2half2_rn(-65504.0f);
    const int   half_vocab_size        = vocab_size / 2;
    const int   half_vocab_size_padded = vocab_size_padded / 2;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < batch_size * half_vocab_size_padded;
         index += blockDim.x * gridDim.x) {
        int   batch_idx = index / half_vocab_size_padded;
        int   vocab_idx = index % half_vocab_size_padded;
        half2 logit     = vocab_idx < half_vocab_size ? __ldg(&logits[index]) : mask_val;
        if (vocab_idx < half_vocab_size) {
            if (bias != nullptr) {
                logit = __hadd2(logit, bias[vocab_idx]);
            }
            logits[index] = __hmul2(logit, h2_inv_temperatures[batch_idx]);
        }
    }
}

template<typename T>
void invokeBatchApplyTemperaturePenalty(T*           logits,
                                        const T*     bias,
                                        const float* temperatures,
                                        const int    batch_size,
                                        const int    vocab_size,
                                        const int    vocab_size_padd,
                                        cudaStream_t stream) {
    dim3 block(min(vocab_size_padd, 1024));
    dim3 grid(
        max(1, min((batch_size * vocab_size_padd + static_cast<int>(block.x) - 1) / static_cast<int>(block.x), 65536)));
    if (std::is_same<T, half>::value && vocab_size % 2 == 0 && vocab_size_padd % 2 == 0) {
        size_t smem_size = sizeof(half2) * batch_size;
        batchApplyTemperaturePenalty_h2<<<grid, block, smem_size, stream>>>(reinterpret_cast<half2*>(logits),
                                                                            reinterpret_cast<const half2*>(bias),
                                                                            temperatures,
                                                                            batch_size,
                                                                            vocab_size,
                                                                            vocab_size_padd);
    } else {
        size_t smem_size = sizeof(float) * batch_size;
        batchApplyTemperaturePenalty<T>
            <<<grid, block, smem_size, stream>>>(logits, bias, temperatures, batch_size, vocab_size, vocab_size_padd);
    }
}

template void invokeBatchApplyTemperaturePenalty(float*       logits,
                                                 const float* bias,
                                                 const float* temperatures,
                                                 const int    batch_size,
                                                 const int    vocab_size,
                                                 const int    vocab_size_padd,
                                                 cudaStream_t stream);

template void invokeBatchApplyTemperaturePenalty(half*        logits,
                                                 const half*  bias,
                                                 const float* temperatures,
                                                 const int    batch_size,
                                                 const int    vocab_size,
                                                 const int    vocab_size_padd,
                                                 cudaStream_t stream);

template<typename T>
__global__ void batchApplyPenaltyLongSeq(T*           logits,
                                         int*         penalty_ws,
                                         const float* repetition_penalties,
                                         const float* presence_penalties,
                                         const float* frequency_penalties,
                                         const int*   output_ids,
                                         const int    batch_size,
                                         const int    vocab_size,
                                         const int*   input_lengths,
                                         const int    max_input_length,
                                         const int    step) {
    const int batch_idx    = blockIdx.x;
    const int input_length = input_lengths != nullptr ? input_lengths[batch_idx] : max_input_length;
    float     repetition_penalty, presence_penalty, frequency_penalty;
    if (repetition_penalties) {
        repetition_penalty = repetition_penalties[batch_idx];
    }
    if (presence_penalties) {
        presence_penalty = presence_penalties[batch_idx];
    }
    if (frequency_penalties) {
        frequency_penalty = frequency_penalties[batch_idx];
    }
    logits += batch_idx * vocab_size;
    penalty_ws += batch_idx * vocab_size;
    // Phase 1. Find indices to penalize and keep the penalized values.
    // A vocab id can appear multiple times but should be penalized once.
    for (int index = threadIdx.x; index < step; index += blockDim.x) {
        // Skip the padding tokens in input sequences.
        if (index >= input_length && index < max_input_length) {
            continue;
        }
        // output_ids shape: (input_len + output_len, batch_size)
        int penalty_index = output_ids[index * batch_size + batch_idx];
        if (penalty_index >= vocab_size || penalty_index < 0) {
            continue;
        }
        atomicAdd(&penalty_ws[penalty_index], 1);
    }
    __syncthreads();
    // Phase 2. Replace a logit value by the penalized one.
    for (int index = threadIdx.x; index < vocab_size; index += blockDim.x) {
        // Skip the padding tokens in input sequences.
        int count = penalty_ws[index];
        if (count > 0) {
            float logit = logits[index];
            if (repetition_penalties) {
                logit = logit < 0.0f ? logit * repetition_penalty : logit / repetition_penalty;
            }
            if (presence_penalties) {
                logit -= presence_penalty;
            }
            if (frequency_penalties) {
                logit -= frequency_penalty * count;
            }
            logits[index] = logit;
        }
    }
}

template<typename T>
void invokeBatchApplyRepetitionPenalty(T*           logits,
                                       int*         penalty_ws,
                                       const float* repetition_penalty,
                                       const float* presence_penalty,
                                       const float* frequency_penalty,
                                       const int*   output_ids,
                                       const int    batch_size,
                                       const int    local_batch_size,
                                       const int    vocab_size,
                                       const int*   input_lengths,
                                       const int    max_input_length,
                                       const int    step,
                                       cudaStream_t stream) {
    // Inputs
    //   logits [local_batch_size, vocab_size] : logit values.
    //   penalties [local_batch_size] : repetition penalty factors.
    //   output_ids [step, batch_size] : output token ids (with offset ite * local_batch_size).
    //   input_lengths [local_batch_size], input lengths (optional).
    //      Padding tokens at [input_length, max_input_length) of input will not be penalized.
    dim3   block(min(step, 1024));
    dim3   grid(local_batch_size);
    size_t smem_size = step * (sizeof(float) + sizeof(int));
    batchApplyPenaltyLongSeq<T><<<grid, block, 0, stream>>>(logits,
                                                            penalty_ws,
                                                            repetition_penalty,
                                                            presence_penalty,
                                                            frequency_penalty,
                                                            output_ids,
                                                            batch_size,
                                                            vocab_size,
                                                            input_lengths,
                                                            max_input_length,
                                                            step);
}

template void invokeBatchApplyRepetitionPenalty(float*       logits,
                                                int*         penalty_ws,
                                                const float* repetition_penalty,
                                                const float* presence_penalty,
                                                const float* frequency_penalty,
                                                const int*   output_ids,
                                                const int    batch_size,
                                                const int    local_batch_size,
                                                const int    vocab_size,
                                                const int*   input_lengths,
                                                const int    max_input_length,
                                                const int    step,
                                                cudaStream_t stream);

template void invokeBatchApplyRepetitionPenalty(half*        logits,
                                                int*         penalty_ws,
                                                const float* repetition_penalty,
                                                const float* presence_penalty,
                                                const float* frequency_penalty,
                                                const int*   output_ids,
                                                const int    batch_size,
                                                const int    local_batch_size,
                                                const int    vocab_size,
                                                const int*   input_lengths,
                                                const int    max_input_length,
                                                const int    step,
                                                cudaStream_t stream);

template<typename T>
__global__ void batchApplyPenaltyRowMajorSparse(T*           logits,
                                                int*         penalty_ws,
                                                const float* repetition_penalties,
                                                const float* presence_penalties,
                                                const float* frequency_penalties,
                                                float        scalar_repetition_penalty,
                                                float        scalar_presence_penalty,
                                                float        scalar_frequency_penalty,
                                                const int*   token_ids,
                                                const int    vocab_size,
                                                const int    token_stride,
                                                const int*   input_lengths,
                                                const int    uniform_input_length,
                                                const int    max_input_length,
                                                const int    step) {
    const int   batch_idx = blockIdx.x;
    const float rep_val =
        (repetition_penalties != nullptr) ? repetition_penalties[batch_idx] : scalar_repetition_penalty;
    const float pres_val = (presence_penalties != nullptr) ? presence_penalties[batch_idx] : scalar_presence_penalty;
    const float freq_val = (frequency_penalties != nullptr) ? frequency_penalties[batch_idx] : scalar_frequency_penalty;

    // Early-exit neutral rows without touching penalty_ws, keeping workspace cleanly zeroed.
    if (rep_val == 1.0f && pres_val == 0.0f && freq_val == 0.0f) {
        return;
    }

    const int  input_length = (input_lengths != nullptr) ?
                                  input_lengths[batch_idx] :
                                  (uniform_input_length >= 0 ? uniform_input_length : max_input_length);
    const bool has_rep      = (rep_val != 1.0f);
    const bool has_pres     = (pres_val != 0.0f);
    const bool has_freq     = (freq_val != 0.0f);

    T*         row_logits = logits + static_cast<int64_t>(batch_idx) * vocab_size;
    int*       row_ws     = penalty_ws + static_cast<int64_t>(batch_idx) * vocab_size;
    const int* row_tokens = token_ids + static_cast<int64_t>(batch_idx) * token_stride;

    // Phase 1: accumulate token counts in row-major coalesced order.
    for (int index = threadIdx.x; index < step; index += blockDim.x) {
        if (index >= input_length && index < max_input_length) {
            continue;
        }
        const int tok = row_tokens[index];
        if (tok < 0 || tok >= vocab_size) {
            continue;
        }
        atomicAdd(&row_ws[tok], 1);
    }
    __syncthreads();

    // Phase 2: sparse update over history tokens, atomically resetting every touched slot back to 0.
    for (int index = threadIdx.x; index < step; index += blockDim.x) {
        if (index >= input_length && index < max_input_length) {
            continue;
        }
        const int tok = row_tokens[index];
        if (tok < 0 || tok >= vocab_size) {
            continue;
        }
        const int count = atomicExch(&row_ws[tok], 0);
        if (count > 0) {
            float logit = static_cast<float>(row_logits[tok]);
            if (has_rep) {
                logit = logit < 0.0f ? logit * rep_val : logit / rep_val;
            }
            if (has_pres) {
                logit -= pres_val;
            }
            if (has_freq) {
                logit -= freq_val * count;
            }
            row_logits[tok] = static_cast<T>(logit);
        }
    }
}

template<typename T>
void invokeBatchApplyRepetitionPenaltyRowMajorSparse(T*           logits,
                                                     int*         penalty_ws,
                                                     const float* repetition_penalty,
                                                     const float* presence_penalty,
                                                     const float* frequency_penalty,
                                                     float        scalar_repetition_penalty,
                                                     float        scalar_presence_penalty,
                                                     float        scalar_frequency_penalty,
                                                     const int*   token_ids,
                                                     const int    batch_size,
                                                     const int    token_stride,
                                                     const int    vocab_size,
                                                     const int*   input_lengths,
                                                     const int    uniform_input_length,
                                                     const int    max_input_length,
                                                     const int    step,
                                                     cudaStream_t stream) {
    if (batch_size <= 0 || step <= 0 || vocab_size <= 0) {
        return;
    }
    const int threads = std::min(((step + 31) / 32) * 32, 1024);
    dim3      block(threads);
    dim3      grid(batch_size);
    batchApplyPenaltyRowMajorSparse<T><<<grid, block, 0, stream>>>(logits,
                                                                   penalty_ws,
                                                                   repetition_penalty,
                                                                   presence_penalty,
                                                                   frequency_penalty,
                                                                   scalar_repetition_penalty,
                                                                   scalar_presence_penalty,
                                                                   scalar_frequency_penalty,
                                                                   token_ids,
                                                                   vocab_size,
                                                                   token_stride,
                                                                   input_lengths,
                                                                   uniform_input_length,
                                                                   max_input_length,
                                                                   step);
}

template<typename T>
void invokeBatchApplyRepetitionPenaltyRowMajorSparse(T*           logits,
                                                     int*         penalty_ws,
                                                     const float* repetition_penalty,
                                                     const float* presence_penalty,
                                                     const float* frequency_penalty,
                                                     const int*   token_ids,
                                                     const int    batch_size,
                                                     const int    token_stride,
                                                     const int    vocab_size,
                                                     const int*   input_lengths,
                                                     const int    max_input_length,
                                                     const int    step,
                                                     cudaStream_t stream) {
    invokeBatchApplyRepetitionPenaltyRowMajorSparse<T>(logits,
                                                       penalty_ws,
                                                       repetition_penalty,
                                                       presence_penalty,
                                                       frequency_penalty,
                                                       1.0f,
                                                       0.0f,
                                                       0.0f,
                                                       token_ids,
                                                       batch_size,
                                                       token_stride,
                                                       vocab_size,
                                                       input_lengths,
                                                       max_input_length,
                                                       max_input_length,
                                                       step,
                                                       stream);
}

template void invokeBatchApplyRepetitionPenaltyRowMajorSparse(float*       logits,
                                                              int*         penalty_ws,
                                                              const float* repetition_penalty,
                                                              const float* presence_penalty,
                                                              const float* frequency_penalty,
                                                              float        scalar_repetition_penalty,
                                                              float        scalar_presence_penalty,
                                                              float        scalar_frequency_penalty,
                                                              const int*   token_ids,
                                                              const int    batch_size,
                                                              const int    token_stride,
                                                              const int    vocab_size,
                                                              const int*   input_lengths,
                                                              const int    uniform_input_length,
                                                              const int    max_input_length,
                                                              const int    step,
                                                              cudaStream_t stream);

template void invokeBatchApplyRepetitionPenaltyRowMajorSparse(half*        logits,
                                                              int*         penalty_ws,
                                                              const float* repetition_penalty,
                                                              const float* presence_penalty,
                                                              const float* frequency_penalty,
                                                              float        scalar_repetition_penalty,
                                                              float        scalar_presence_penalty,
                                                              float        scalar_frequency_penalty,
                                                              const int*   token_ids,
                                                              const int    batch_size,
                                                              const int    token_stride,
                                                              const int    vocab_size,
                                                              const int*   input_lengths,
                                                              const int    uniform_input_length,
                                                              const int    max_input_length,
                                                              const int    step,
                                                              cudaStream_t stream);

template void invokeBatchApplyRepetitionPenaltyRowMajorSparse(float*       logits,
                                                              int*         penalty_ws,
                                                              const float* repetition_penalty,
                                                              const float* presence_penalty,
                                                              const float* frequency_penalty,
                                                              const int*   token_ids,
                                                              const int    batch_size,
                                                              const int    token_stride,
                                                              const int    vocab_size,
                                                              const int*   input_lengths,
                                                              const int    max_input_length,
                                                              const int    step,
                                                              cudaStream_t stream);

template void invokeBatchApplyRepetitionPenaltyRowMajorSparse(half*        logits,
                                                              int*         penalty_ws,
                                                              const float* repetition_penalty,
                                                              const float* presence_penalty,
                                                              const float* frequency_penalty,
                                                              const int*   token_ids,
                                                              const int    batch_size,
                                                              const int    token_stride,
                                                              const int    vocab_size,
                                                              const int*   input_lengths,
                                                              const int    max_input_length,
                                                              const int    step,
                                                              cudaStream_t stream);

}  // namespace rtp_llm
