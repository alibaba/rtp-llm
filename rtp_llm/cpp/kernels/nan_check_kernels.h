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

// General NaN check operation
template<typename T>
void invokeCheckNAN(T* input, size_t nums, cudaStream_t stream);

/**
 * Prefill: check and reset NaN/Inf in KV cache.
 *
 * Layout: [layer_num, block_num, 2, local_head_num_kv, seq_size_per_block, k_token_size]
 * Within a block: [2, local_head_num_kv, seq_size_per_block, k_token_size]
 * Memory organization: All K first (organized by [head, token, head_dim]), then all V.
 *
 * For MLA (Multi-head Latent Attention) KV cache:
 * - Layout: [layer_num, block_num, seq_size_per_block, k_token_size + v_token_size + layer_out_size]
 * - Within a block: K part (kv_lora_rank), V part (rope_head_dim), then layer_out part.
 * - Memory organization: All K first, then all V, then all layer_out.
 *
 * All pointer arguments are device pointers.
 *
 * Tensor / buffer contracts:
 * - layer_base_addr:
 *     shape [layer_num]. Each element is a base pointer to the KV cache region for that layer.
 * - kv_cache_block_id:
 *     Always interpreted as 3D [num_groups, batch_dim, max_blocks_per_batch]. Index:
 *     block_id(g, b, m) = base + g * (batch_dim * max_blocks_per_batch) + b * max_blocks_per_batch + m.
 *     For 2D input [B, M], treat as [1, B, M] (num_groups=1, batch_dim=B).
 *     A value of -1 indicates a null/unallocated block and is skipped.
 * - num_groups: number of groups (1 when 2D).
 * - layer_to_group: [layer_num], layer_id -> group_id; nullptr means all layers use group 0.
 * - group_types: [num_groups], int32: 0=LINEAR (skip check), 1=FULL (check); nullptr means all FULL.
 * - batch_dim: second dimension of 3D block_id (total batch size).
 * - batch_start: start batch index for this phase (0 for decode, decoder_batch_size for prefill).
 * - prefix_lengths:
 *     shape [batch_size]. The checked region starts at prefix_lengths[batch].
 * - input_lengths:
 *     shape [batch_size]. The checked region ends at input_lengths[batch] (exclusive).
 * - nan_flag:
 *     shape [batch_size]. Set to 1 if any NaN/Inf is detected and reset for the batch.
 *
 * Size units:
 * - k_token_size: elements per head for K (not bytes)
 * - v_token_size: elements per head for V (not bytes)
 * - k_block_size_bytes: total bytes for K part of one block
 * - v_block_size_bytes: total bytes for V part of one block
 * - seq_size_per_block: number of tokens in one block
 */
template<typename T>
void invokeCheckAndResetNANKvCachePrefill(const void* const* layer_base_addr,
                                          const int32_t*     kv_cache_block_id,
                                          const int32_t*     prefix_lengths,
                                          const int32_t*     input_lengths,
                                          size_t             batch_size,
                                          size_t             layer_num,
                                          size_t             num_groups,
                                          const int32_t*     layer_to_group,
                                          const int32_t*     group_types,
                                          size_t             batch_dim,
                                          size_t             batch_start,
                                          size_t             max_blocks_per_batch,
                                          size_t             local_head_num_kv,
                                          size_t             k_token_size,
                                          size_t             v_token_size,
                                          size_t             k_block_size_bytes,
                                          size_t             v_block_size_bytes,
                                          size_t             k_token_bytes,
                                          size_t             v_token_bytes,
                                          size_t             block_size_bytes,
                                          size_t             seq_size_per_block,
                                          float*             nan_flag,
                                          cudaStream_t       stream);

/**
 * Decode: check and reset NaN/Inf in KV cache.
 *
 * This checks only the last token for each batch, based on sequence_lengths[batch].
 *
 * For MLA (Multi-head Latent Attention) KV cache:
 * - Layout: [layer_num, block_num, seq_size_per_block, k_token_size + v_token_size + layer_out_size]
 * - Within a block: K part (kv_lora_rank), V part (rope_head_dim), then layer_out part.
 * - Memory organization: All K first, then all V, then all layer_out.
 *
 * See invokeCheckAndResetNANKvCachePrefill() for layout and buffer contracts.
 */
template<typename T>
void invokeCheckAndResetNANKvCacheDecode(const void* const* layer_base_addr,
                                         const int32_t*     kv_cache_block_id,
                                         const int32_t*     sequence_lengths,
                                         size_t             batch_size,
                                         size_t             layer_num,
                                         size_t             num_groups,
                                         const int32_t*     layer_to_group,
                                         const int32_t*     group_types,
                                         size_t             batch_dim,
                                         size_t             batch_start,
                                         size_t             max_blocks_per_batch,
                                         size_t             local_head_num_kv,
                                         size_t             k_token_size,
                                         size_t             v_token_size,
                                         size_t             k_block_size_bytes,
                                         size_t             v_block_size_bytes,
                                         size_t             k_token_bytes,
                                         size_t             v_token_bytes,
                                         size_t             block_size_bytes,
                                         size_t             seq_size_per_block,
                                         float*             nan_flag,
                                         cudaStream_t       stream);

}  // namespace rtp_llm
