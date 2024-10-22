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
#pragma once
#include <cstdint>
#include <curand_kernel.h>
#include <cuda_fp16.h>
#include "src/fastertransformer/utils/string_utils.h"
#include "src/fastertransformer/cuda/cuda_utils.h"
namespace tensorrt_llm
{
namespace kernels
{
static constexpr int nMaxBeamWidth = 64; // max beam width supported now
static constexpr int nBlockSizeForSmallBeamWidth = 256;
static constexpr int nMaxVocabPartForStage1FastKernel = 128;
struct BeamHypotheses
{
    // MBS: max_batch_size, BS: batch_size, BM: beam_width, MSL: max_seq_length
    // Candidate beams: a beam which generates end_id or its sequence length reaches MSL
    // Candidate-Beam-Array (CBA): The arrays to place the candidate beams and related information
    // Scalar values
    int nMaxBatchSize{0};               // max batch size by model configuration
    int nBatchSize{0};                  // batch size by runtime input data
    int nBeamWidth{0};                  //
    int nMaxSeqLen{0};                  //
    int nVocabSize{0};                  // vocab_size_padded
    // Pointers from input
    int const* inputLengths{nullptr};   // [BS, BM]         %% context_length
    int* sequenceLengths{nullptr};      // [BS, BM]         %% self.sequence_length_buffer
    float* cumLogProbs{nullptr};        // [BS, BM]         %% self.cum_log_probs
    int* outputIdsPtr{nullptr};        // [BS][BM, MSL]        %% self.output_ids
    int* beamIdsPtr{nullptr};        //  [BS, BM]        %% self.parent_ids
};
__inline__ int padToNextPowerOfTwo(int const n)
{
    // Pad n up to the nearest power of 2
    int recursor = n - 1;
    int res = 2;
    while (recursor >>= 1)
        res <<= 1;
    return res;
}
template <typename T>
__device__ __forceinline__ T applyLengthPenalty(T const log_prob, int const length, float const length_penalty)
{
    // score = log(prob) / (length ^ length_penalty)
    if (length_penalty == 0.0f || length == 1)
    {
        return log_prob;
    }
    return log_prob / static_cast<T>(powf(static_cast<float>(length), length_penalty));
}
template <typename T>
void invokeTopkSoftMax(T const* logits, void* workspace, BeamHypotheses& bh, cudaStream_t stream);
} // namespace kernels
} // namespace tensorrt_llm