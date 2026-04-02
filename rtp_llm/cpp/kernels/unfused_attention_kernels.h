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

#include "rtp_llm/cpp/kernels/kv_cache/kv_cache_utils.h"
#include "rtp_llm/cpp/model_utils/RopeConfig.h"
#if USING_ROCM
#include "hip/amd_detail/amd_hip_vector_types.h"
#endif

namespace rtp_llm {

struct PrefixPromptBatchWeightsParam {
    const int*   d_prefix_prompt_lengths  = nullptr;
    int          max_prefix_prompt_length = 0;
    bool         count_length             = false;
    KVBlockArray kv_block_array           = KVBlockArray();
#if USING_ROCM
    OffsetIndexedKVBlockArray offset_kv_block_array = OffsetIndexedKVBlockArray();
#endif
};

template<typename T>
void invokeAddFusedQKVBiasTranspose(T*                             q_no_transpose_buf,
                                    T*                             q_buf,
                                    T*                             k_buf,
                                    T*                             v_buf,
                                    PrefixPromptBatchWeightsParam* param,
                                    T*                             QKV,
                                    void*                          QuantizedQKV,
                                    const int*                     position_ids,
                                    const T*                       qkv_bias,
                                    const int*                     padding_offset,
                                    const int*                     cu_seqlens,
                                    const bool                     use_rope_cache,
                                    const float*                   rope_cache,
                                    const int                      batch_size,
                                    const int                      seq_len,
                                    const int                      token_num,
                                    const int                      head_num,
                                    const int                      head_num_kv,
                                    const int                      size_per_head,
                                    const RopeConfig               rope_config,
                                    const bool                     use_logn_attn,
                                    const float*                   scale,
                                    const int                      int8_mode,
                                    const bool                     use_paged_fmha,
                                    const bool                     store_qkv,
                                    const bool                     store_q_no_transpose,
                                    const bool                     store_q,
                                    const bool                     store_kv,
                                    const bool                     store_cache,
                                    cudaStream_t                   stream);

template<typename T>
void invoke_debug_kernel2(
    T* data, int strat_row, int start_col, int m, int n, int row_len, int info_id, cudaStream_t stream);

template<typename T>
void invokeDecodeAddFusedQKVBiasTranspose(T*               q_buf,
                                          T*               k_buf,
                                          T*               v_buf,
                                          KVBlockArray     kv_block_array,
                                          T*               QKV,
                                          const int*       position_ids,
                                          const T*         qkv_bias,
                                          const bool       use_rope_cache,
                                          const float*     rope_cache,
                                          const int        batch_size,
                                          const int        head_num,
                                          const int        head_num_kv,
                                          const int        size_per_head,
                                          const RopeConfig rope_config,
                                          const bool       use_logn_attn,
                                          const bool       store_q,
                                          const bool       store_kv,
                                          const bool       store_cache,
                                          cudaStream_t     stream);

#if USING_ROCM
template<typename T>
void invokeAddFusedQKVBiasTransposePrefillV1(T*                             q_buf,
                                             T*                             k_buf,
                                             T*                             v_buf,
                                             PrefixPromptBatchWeightsParam* param,
                                             T*                             QKV,
                                             void*                          QuantizedQKV,
                                             const int*                     position_ids,
                                             const T*                       qkv_bias,
                                             const int*                     padding_offset,
                                             const int*                     cu_seqlens,
                                             const int                      batch_size,
                                             const int                      seq_len,
                                             const int                      token_num,
                                             const int                      head_num,
                                             const int                      head_num_kv,
                                             const int                      size_per_head,
                                             const RopeConfig               rope_config,
                                             const bool                     use_logn_attn,
                                             const float*                   scale,
                                             const int                      int8_mode,
                                             const bool                     use_paged_fmha,
                                             const bool                     store_qkv,
                                             const bool                     store_q,
                                             const bool                     store_kv,
                                             const bool                     store_cache,
                                             const float2*                  cos_sin_cache,
                                             cudaStream_t                   stream);

template<typename T>
void invokeAddFusedQKVBiasTransposePrefill(T*                             q_buf,
                                           T*                             k_buf,
                                           T*                             v_buf,
                                           PrefixPromptBatchWeightsParam* param,
                                           T*                             QKV,
                                           void*                          QuantizedQKV,
                                           const int*                     position_ids,
                                           const T*                       qkv_bias,
                                           const int*                     padding_offset,
                                           const int*                     cu_seqlens,
                                           const int                      batch_size,
                                           const int                      seq_len,
                                           const int                      token_num,
                                           const int                      head_num,
                                           const int                      head_num_kv,
                                           const int                      size_per_head,
                                           const RopeConfig               rope_config,
                                           const bool                     use_logn_attn,
                                           const float*                   scale,
                                           const int                      int8_mode,
                                           const bool                     use_paged_fmha,
                                           const bool                     store_qkv,
                                           const bool                     store_q,
                                           const bool                     store_kv,
                                           const bool                     store_cache,
                                           const float2*                  cos_sin_cache,
                                           const bool                     pad_query,
                                           cudaStream_t                   stream);

template<typename T>
void invokeAddFusedQKVBiasTransposeDecodeV1(T*                             q_buf,
                                            T*                             k_buf,
                                            T*                             v_buf,
                                            PrefixPromptBatchWeightsParam* param,
                                            const int*                     input_lengths,
                                            T*                             QKV,
                                            void*                          QuantizedQKV,
                                            const int*                     position_ids,
                                            const T*                       qkv_bias,
                                            const int*                     padding_offset,
                                            const int*                     cu_seqlens,
                                            const int*                     sequence_lengths,
                                            const int                      batch_size,
                                            const int                      seq_len,
                                            const int                      token_num,
                                            const int                      head_num,
                                            const int                      head_num_kv,
                                            const int                      size_per_head,
                                            const RopeConfig               rope_config,
                                            const bool                     use_logn_attn,
                                            const float*                   scale,
                                            const int                      int8_mode,
                                            const bool                     use_paged_fmha,
                                            const bool                     store_qkv,
                                            const bool                     store_q,
                                            const bool                     store_kv,
                                            const bool                     store_cache,
                                            const float2*                  cos_sin_cache,
                                            cudaStream_t                   stream);

template<typename T>
void invokeAddFusedQKVBiasTransposeDecode(T*                             q_buf,
                                          T*                             k_buf,
                                          T*                             v_buf,
                                          PrefixPromptBatchWeightsParam* param,
                                          const int*                     input_lengths,
                                          T*                             QKV,
                                          void*                          QuantizedQKV,
                                          const int*                     position_ids,
                                          const T*                       qkv_bias,
                                          const int*                     padding_offset,
                                          const int*                     cu_seqlens,
                                          const int*                     sequence_lengths,
                                          const int                      batch_size,
                                          const int                      seq_len,
                                          const int                      token_num,
                                          const int                      head_num,
                                          const int                      head_num_kv,
                                          const int                      size_per_head,
                                          const RopeConfig               rope_config,
                                          const bool                     use_logn_attn,
                                          const float*                   scale,
                                          const int                      int8_mode,
                                          const bool                     use_paged_fmha,
                                          const bool                     store_qkv,
                                          const bool                     store_q,
                                          const bool                     store_kv,
                                          const bool                     store_cache,
                                          const float2*                  cos_sin_cache,
                                          cudaStream_t                   stream);
#endif

}  // namespace rtp_llm
