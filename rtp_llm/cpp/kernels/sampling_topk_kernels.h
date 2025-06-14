/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
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

#include "rtp_llm/cpp/utils/Logger.h"
#if USING_CUDA
#include <curand_kernel.h>
#endif
#if USING_ROCM
#include "rtp_llm/cpp/rocm/cuda_shims.h"
#endif

namespace rtp_llm {

template<typename T>
void invokeTopKSampling(void*          workspace,
                        size_t&        workspace_size,
                        const T*       log_probs,
                        int*           ids,
                        int*           sequence_length,
                        bool*          finished_buf,
                        float*         cum_log_probs,
                        float*         output_log_probs,
                        curandState_t* curandstate,
                        const int      top_k,
                        const float    top_p,
                        const int      vocab_size_padded,
                        const int*     end_ids,
                        float*         output_all_probs,
                        cudaStream_t   stream,
                        const int      batch_size,
                        const bool*    skip_decode);

template<typename T>
void invokeBatchTopKSampling(void*          workspace,
                             size_t&        workspace_size,
                             const T*       log_probs,
                             int*           ids,
                             int*           sequence_length,
                             bool*          finished,
                             float*         cum_log_probs,
                             float*         output_log_probs,
                             curandState_t* curandstate,
                             const int      max_top_k,
                             const int*     top_ks,
                             const float    top_p,
                             const float*   top_ps,
                             const int      vocab_size_padded,
                             const int*     end_ids,
                             float*         output_all_probs,
                             cudaStream_t   stream,
                             const int      batch_size,
                             const bool*    skip_decode);

void invokeCurandInitialize(curandState_t*     state,
                            const size_t       batch_size,
                            unsigned long long random_seed,
                            cudaStream_t       stream);

void invokeCurandBatchInitialize(curandState_t*            states,
                                 const size_t              batch_size,
                                 const unsigned long long* random_seeds,
                                 cudaStream_t              stream);

template<typename T>
void invokeAddBiasEndMask(T*           logits,
                          const T*     bias,
                          const int*   end_ids,
                          const bool*  finished,
                          const int    batch_size,
                          const int    vocab_size,
                          const int    vocab_size_padded,
                          cudaStream_t stream);

void invokeSetupTopKRuntimeArgs(int    batch_size,
                                uint   top_k,
                                uint*  top_ks,
                                int    top_ks_size,
                                float  top_p,
                                float* top_ps,
                                int    top_ps_size,
                                bool*  skip_decode,
                                cudaStream_t stream);


}  // namespace rtp_llm

