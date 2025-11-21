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

#if USING_CUDA
#include <cuda_fp16.h>
#endif

#include "rtp_llm/cpp/kernels/penalty_types.h"

#if USING_ROCM
#include "rtp_llm/cpp/rocm/cuda_shims.h"
#endif

namespace rtp_llm {

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
                                       cudaStream_t stream);

template<typename T>
void invokeApplyTemperaturePenalty(T*           logits,
                                   const T*     bias,
                                   const float  temperature,
                                   const int    batch_size,
                                   const int    vocab_size,
                                   const int    vocab_size_padd,
                                   cudaStream_t stream);

template<typename T>
void invokeBatchApplyTemperaturePenalty(T*           logits,
                                        const T*     bias,
                                        const float* temperatures,
                                        const int    batch_size,
                                        const int    vocab_size,
                                        const int    vocab_size_padd,
                                        cudaStream_t stream);


template<typename T>
void invokeCopyLogits(float*       output_logits_buf,
                      int*         logit_index_buf,
                      T*           runtime_logits_buf,
                      bool*        skip_decode_buf_,
                      const int    local_batch_size,
                      const int    vocab_size_padded_,
                      cudaStream_t steam);

}  // namespace rtp_llm
