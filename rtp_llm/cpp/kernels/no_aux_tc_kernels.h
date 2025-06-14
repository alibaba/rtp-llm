/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "rtp_llm/cpp/cuda/cuda_utils.h"

namespace rtp_llm
{

template <typename T, typename IdxT>
void invokeNoAuxTc(T* scores, T* group_scores, T* topk_values, IdxT* topk_indices, T* scores_with_bias,
    int64_t const num_tokens, int64_t const num_experts, int64_t const n_group, int64_t const topk_group,
    int64_t const topk, int norm_node, double const routed_scaling_factor, cudaStream_t const stream = 0);

} // namespace rtp_llm
