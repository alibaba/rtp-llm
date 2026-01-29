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

// Tensor transpose operations
template<typename T>
void invokeTransposeAxis012(
    T* out, T* in, const size_t dim0, const size_t dim1, const size_t dim2, cudaStream_t stream);

// from [b, s, h, d] to [b, h, s, d]
template<typename T>
void invokeTransposeAxis12(
    T* out, T* in, const size_t dim0, const size_t dim1, const size_t dim2, const size_t dim_3, cudaStream_t stream);

template<typename T>
void invokeTransposeAxis01(T* out, T* in, const size_t dim0, const size_t dim1, cudaStream_t stream);

// Sequence operations
template<typename T>
void invokeLookupHiddenStateOfLastToken(T*           from_tensor,
                                        const T*     hidden_state,
                                        const int*   input_lengths,
                                        const size_t batch_size,
                                        const size_t hidden_units,
                                        const size_t idx_offset,
                                        cudaStream_t stream);

template<typename T>
void invokeCheckNAN(T* input, size_t nums, cudaStream_t stream);

/**
 * Prefill: check and reset NaN/Inf in KV cache (interleaved layout).
 *
 * Layout assumption (per token, within a block): [K(token), V(token)] interleaved, i.e.
 *   [Token0_K, Token0_V, Token1_K, Token1_V, ...]
 *
 * All pointer arguments are device pointers.
 *
 * Tensor / buffer contracts:
 * - layer_base_addr:
 *     shape [layer_num] .
 *     Each element is a base pointer to the KV cache region for that layer,
 *     covering all physical blocks for that layer. The kernel computes:
 *       token_stride_bytes = block_size_bytes / seq_size_per_block
 *       token_ptr = base + physical_block_id * block_size_bytes + token_in_block * token_stride_bytes
 * - kv_cache_block_id:
 *     shape [batch_size, max_blocks_per_batch], mapping logical_block_id -> physical_block_id.
 *     A value of -1 indicates a null/unallocated block and is skipped.
 * - prefix_lengths:
 *     shape [batch_size]. The checked region starts at prefix_lengths[batch].
 * - seq_len_cu:
 *     shape [batch_size]. The checked region ends at seq_len_cu[batch] (exclusive).
 * - nan_flag:
 *     shape [batch_size]. Set to 1 if any NaN/Inf is detected and reset for the batch.
 *     The kernel only ever writes 1 (never writes 0), so non-atomic stores are sufficient.
 *
 * Size units:
 * - block_size_bytes: total bytes of one KV block (K + V) for a single layer.
 * - seq_size_per_block: number of tokens in one block.
 */
template<typename T>
void invokeCheckAndResetNANKvCachePrefill(const void* const* layer_base_addr,
                                          const int32_t*     kv_cache_block_id,
                                          const int32_t*     prefix_lengths,
                                          const int32_t*     seq_len_cu,
                                          size_t             batch_size,
                                          size_t             layer_num,
                                          size_t             max_blocks_per_batch,
                                          size_t             block_size_bytes,
                                          size_t             seq_size_per_block,
                                          int32_t*           nan_flag,
                                          cudaStream_t       stream);

/**
 * Decode: check and reset NaN/Inf in KV cache (interleaved layout).
 *
 * This checks only the last token for each batch, based on sequence_lengths[batch].
 *
 * See invokeCheckAndResetNANKvCachePrefill() for layout and buffer contracts.
 */
template<typename T>
void invokeCheckAndResetNANKvCacheDecode(const void* const* layer_base_addr,
                                         const int32_t*     kv_cache_block_id,
                                         const int32_t*     sequence_lengths,
                                         size_t             batch_size,
                                         size_t             layer_num,
                                         size_t             max_blocks_per_batch,
                                         size_t             block_size_bytes,
                                         size_t             seq_size_per_block,
                                         int32_t*           nan_flag,
                                         cudaStream_t       stream);

}  // namespace rtp_llm
