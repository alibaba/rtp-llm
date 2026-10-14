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

namespace rtp_llm {

// Grouped biased top-k over raw router logits. The sigmoid gating math is fused
// into the kernels: each logit is widened to fp32, run through
// 1/(1+expf(-x)) in fp32, and the widened correction bias is added in fp32 —
// the same rounding sequence as the previous python path
// (scores.float().sigmoid() + correction_bias).
// InT is the router logits dtype and BiasT the correction bias dtype
// (fp32, fp16 or bf16 each); group scores and topk values are computed in
// fp32 regardless.
template<typename InT, typename BiasT, typename IdxT>
void invokeNoAuxTc(InT const*         router_logits,
                   float*             group_scores,
                   float*             topk_values,
                   IdxT*              topk_indices,
                   BiasT const*       correction_bias,
                   int64_t const      num_tokens,
                   int64_t const      num_experts,
                   int64_t const      n_group,
                   int64_t const      topk_group,
                   int64_t const      topk,
                   int                norm_node,
                   double const       routed_scaling_factor,
                   cudaStream_t const stream = 0);

}  // namespace rtp_llm
