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

#if USING_CUDA
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "src/fastertransformer/cuda/memory_utils.h"
#endif
#if USING_ROCM
#include <hip/hip_runtime.h>
#include "src/fastertransformer/rocm/hip_utils.h"
#endif
#include <unordered_map>

namespace fastertransformer {

template<typename T>
struct inputIdsEmbeddingLookupPosEncodingSoftPromptParam {
    T*           from_tensor;
    int*         output_ids;
    int*         input_lengths;
    const T*     embedding_table;
    const T*     pos_table;
    const float* prefix_soft_prompt_embedding;
    const int*   prefix_soft_prompt_lengths;
    int*         input_ids;
    int          start_step;
    int          max_input_length;
    int          max_prefix_soft_prompt_length;
    int          batch_size;
    int          beam_width;
    int          hidden_units;
    cudaStream_t stream;
};

template<typename T>
struct pPromptTuningParam {
    // Batch number of ptrs, each ptr is the ptr of the specific p/prompt tuning weights for this sequence
    const T** p_prompt_tuning_batch_weights = nullptr;
    // The start id of p_prompt_tuning token ids (based on the tokenizer)
    // PROMPT_0 --> p_prompt_tuning_id_start; PROMPT_1 --> p_prompt_tuning_id_start + 1; ...
    const int p_prompt_tuning_id_start = 0;
    // Request prompt embeddding's max length
    const int request_prompt_max_length = 0;
    // Whether or not use the request prompt embeddings
    const bool use_request_p_prompt_embedding = false;
    // Request prompt embeddings
    const T* request_prompt_embedding = nullptr;
};

template<typename T>
void invokeInputIdsEmbeddingLookupPosEncoding(T*                    from_tensor,
                                              int*                  output_ids,
                                              const T*              embedding_table,
                                              const T*              pos_table,
                                              pPromptTuningParam<T> prompt_param,
                                              const int*            input_ids,
                                              const int             start_step,
                                              const int             length,
                                              const int             max_length,
                                              const int             batch_size,
                                              const int             hidden_units,
                                              cudaStream_t          stream);

template<typename T>
void invokeEmebeddingLookup(T*           from_tensor,
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
void invokeInputIdsEmbeddingLookupPosEncodingSoftPrompt(inputIdsEmbeddingLookupPosEncodingSoftPromptParam<T> param);

template<typename T>
void invokeTransposeAxis012(T* out, T* in, const int dim0, const int dim1, const int dim2, cudaStream_t stream);

// from [b, s, h, d] to [b, h, s, d]
template<typename T>
void invokeTransposeAxis12(T* out, T* in, const int dim0, const int dim1, const int dim2, const int dim_3, cudaStream_t stream);

template<typename T>
void invokeTransposeAxis01(
    T* out, T* in, const int dim0, const int dim1, cudaStream_t stream);

template<typename T>
void invokeBuildDecoderAttentionMask(T*           attention_mask,
                                     const int*   sequence_lengths,
                                     const int*   prefix_prompt_lengths,
                                     const int    batch_size,
                                     const int    max_seq_len,
                                     const int    max_prompt_length,
                                     const bool   is_causal,
                                     cudaStream_t stream);

template<typename T>
void invokeBuildGlmDecoderAttentionMask(
    T* attention_mask, const int* sequence_lengths, const int batch_size, const int max_seq_len, cudaStream_t stream);

template<typename T>
void invokeLookupHiddenStateOfLastToken(T*           from_tensor,
                                        const T*     hidden_state,
                                        const int*   input_lengths,
                                        const int    batch_size,
                                        const int    hidden_units,
                                        const int    idx_offset,
                                        cudaStream_t stream);

template<typename T>
void invokeLookupHiddenStateOfFirstToken(T*           from_tensor,
                                        const T*     hidden_state,
                                        const int*   input_lengths,
                                        const int    batch_size,
                                        const int    hidden_units,
                                        cudaStream_t stream);

void invokeTileGptPromptInputs(int*         tiled_input_ids,
                               int*         tiled_input_lengths,
                               int*         tiled_prompt_lengths,
                               const int*   input_ids,
                               const int*   input_lengths,
                               const int*   prefix_prompt_lengths,
                               const int    batch_size,
                               const int    beam_width,
                               const int    max_input_length,
                               cudaStream_t stream);

void invokeTileGptInputs(int*         tiled_input_ids,
                         int*         tiled_input_lengths,
                         const int*   input_ids,
                         const int*   input_lengths,
                         const int    batch_size,
                         const int    beam_width,
                         const int    max_input_length,
                         cudaStream_t stream);

void invokeFindContextDups(int*         shared_contexts,
                           int*         batch_to_compact,
                           int*         compact_to_batch,
                           int*         compact_size,
                           const int*   input_ids,
                           const size_t batch_size,
                           const size_t input_seq_len,
                           cudaStream_t stream = 0);

template<typename T>
void invokeCompactInputs(T*           compact_input,
                         T*           compact_attention_mask,
                         int*         compact_input_lengths,
                         const T*     decoder_input,
                         const T*     decoder_mask,
                         const int*   input_lengths,
                         const int*   compact_idx,
                         size_t       compact_size,
                         size_t       seq_len,
                         size_t       hidden_dimension,
                         cudaStream_t stream = 0);

template<typename T>
void invokeUnCompactOutputs(T*           uncompact_buffer,
                            const T*     compact_buffer,
                            const int*   batch_to_compact_idx,
                            size_t       batch_size,
                            size_t       buffer_stride,
                            cudaStream_t stream = 0);

template<typename T>
void invokeUnCompactCaches(T*           uncompact_k_cache,
                           T*           uncompact_v_cache,
                           const T*     compact_k_cache,
                           const T*     compact_v_cache,
                           const int*   batch_to_compact_idx,
                           size_t       batch_size,
                           size_t       num_heads,
                           size_t       max_seq_len,
                           size_t       seq_len,
                           size_t       size_per_head,
                           size_t       local_batch_size,
                           size_t       ite,
                           cudaStream_t stream = 0);

void invokeUpdatePaddingCount(int*         total_padding_count,
                              const int*   input_lengths,
                              const int*   tiled_prompt_lengths,
                              size_t       max_input_length,
                              size_t       max_prompt_length,
                              size_t       batch_size,
                              size_t       beam_width,
                              cudaStream_t stream = 0);

inline void invokeUpdatePaddingCount(int*         total_padding_count,
                                     const int*   input_lengths,
                                     size_t       max_input_length,
                                     size_t       batch_size,
                                     size_t       beam_width,
                                     cudaStream_t stream = 0) {
    invokeUpdatePaddingCount(
        total_padding_count, input_lengths, (const int*)nullptr, max_input_length, 0, batch_size, beam_width, stream);
}

void invokeMaskPaddingTokens(bool*        masked_tokens,
                             const int*   input_lengths,
                             const int*   tiled_prefix_prompt_lengths,
                             const size_t memory_len,
                             const size_t max_input_length,
                             const size_t initial_step,
                             size_t       batch_size,
                             size_t       beam_width,
                             cudaStream_t stream = 0);

inline void invokeMaskPaddingTokens(bool*        masked_tokens,
                                    const int*   input_lengths,
                                    const size_t memory_len,
                                    const size_t max_input_length,
                                    const size_t initial_step,
                                    size_t       batch_size,
                                    size_t       beam_width,
                                    cudaStream_t stream = 0) {
    invokeMaskPaddingTokens(masked_tokens,
                            input_lengths,
                            (const int*)nullptr,
                            memory_len,
                            max_input_length,
                            initial_step,
                            batch_size,
                            beam_width,
                            stream);
}

template<typename T>
void invokeSumLengthDimension(float*       out_buf,
                              const T*     in_buf,
                              const size_t batch_size,
                              const size_t input_length,
                              const size_t hidden_dim,
                              cudaStream_t stream = 0);

void invokeMaxLength(
    size_t* h_pinned_max_length, const int* lengths, size_t* max_length, const size_t size, cudaStream_t stream = 0);

void invokeConvertOffsetToAddr(uint64_t*       block_addr, // [l, b, 2, m]
                               const uint64_t* k_cache_base_addr, // [l]
                               const uint64_t* v_cache_base_addr,
                               const int*      offset, // [b, m]
                               int             layer_num,
                               int             batch_size,
                               int             max_block_num,
                               int             block_size,
                               cudaStream_t    stream);

void invokeConvertOffsetToAddrOneLayer(uint64_t*      block_addr, // [b, 2, m]
                                       const uint64_t k_cache_base_addr,
                                       const uint64_t v_cache_base_addr,
                                       const int*     offset, // [b, m]
                                       int            batch_size,
                                       int            max_block_num,
                                       int            block_size,
                                       cudaStream_t   stream);

void invokeConvertOffsetToBlockArrayData(int32_t*       offset_addr, // [b, 2, m]
                                         const int*     offset, // [b, m]
                                         int            batch_size,
                                         int            max_block_num,
                                         int            kv_block_offset,
                                         cudaStream_t   stream);

void invokeGetPaddingOffsetAndCuSeqLens(int*         tmp_mask_offset,
                                        int*         cu_seqlens,
                                        const int*   sequence_length,
                                        const int    batch_size,
                                        const int    max_seq_len,
                                        cudaStream_t stream);

void invokeGetCuSeqLens(int* cu_seqlens,
                        const int* sequence_length,
                        const int* prefix_length,
                        const int batch_size,
                        cudaStream_t stream);

// just support two dim
template<typename T>
void invokeScatterAdd(T const* src, int N, int K, int32_t const* index, T* out, bool use_stable_scatter_add, cudaStream_t stream);


template<typename T>
void invokeSliceDim1Copy(T const* src, int dim0, int dim1, int dim1_start, int dim1_size, T* out, cudaStream_t stream);

void fake_balance_expert(int* expert, float* expert_scales, int start, int expert_num, int size, cudaStream_t stream);

}  // namespace fastertransformer
