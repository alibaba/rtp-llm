/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include <cstdint>

#if USING_CUDA
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#endif
#if USING_ROCM
#include <hip/hip_runtime.h>
#include "rtp_llm/cpp/rocm/cuda_shims.h"
#endif

namespace rtp_llm {

template<typename T>
void invokeEmbeddingLookup(T*           from_tensor,
                            const T*     embedding_table,
                            double       input_embedding_scalar,
                            const T*     pos_table,
                            const T*     type_table,
                            const int*   input_ids,
                            const int*   input_pos,
                            const int*   input_type,
                            const int*   input_mask,
                            const int    token_num,
                            const int    hidden_units,
                            cudaStream_t stream);

template<typename T>
void invokeEmbeddingLookupVec(T*           from_tensor,
                               const T*     embedding_table,
                               double       input_embedding_scalar,
                               const T*     pos_table,
                               const T*     type_table,
                               const int*   input_ids,
                               const int*   input_pos,
                               const int*   input_type,
                               const int*   input_mask,
                               const int    token_num,
                               const int    hidden_units,
                               cudaStream_t stream);

}  // namespace rtp_llm