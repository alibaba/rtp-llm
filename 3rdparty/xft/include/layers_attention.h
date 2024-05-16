// Copyright (c) 2024 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================================
#pragma once

#include "dtype.h"

namespace xft {

/**
 * @brief forward function of attention, designed for continous batching inference
 * @param output output tensor
 * @param query query tensor
 * @param key key tensor
 * @param value value tensor
 * @param query_shape shape for query, in [total_tokens, query_head_num, head_size]
 * @param kv_shape shape for key and value, in [total_tokens, kv_head_num, head_size]
 * @param q_stride, kv_stride stride for query/key/value
 * @param scale scale value before softmax, generally equals to 1/sqrt(head_size)
 * @param batch_size batch size, how many inputs
 * @param token_lens token sizes for each input in the batch
 * @param kcache, vcache tensor of key/value cache
 * @param kvcache_shape shape of the KV cache, in [block_num, block_size, head_num, head_size]
 * @param block_tables which blocks are using, for example [[1, 4], [2], [3]]
 * @param block_nums how many blocks for each input, in above case, it is [2, 1, 1]
 * @param context_lens context length for each input
 * @param layer_id layer ID, we could do some preparation if 0
 * @param is_prefill prefill phase or generation phase
 * @param slot_mapping which slot in KV cache to fill in current key/value
 * 
 * KV cache is like below example (block_num=3, block_size=4):
 *  ________________ ________________ ________________ ________________ 
 * |      slot0     |     slot1      |                |                | block0
 * |________________|________________|________________|________________|
 * |                |                |                |                | block1
 * |________________|________________|________________|________________|
 * |                |                |                |                | block2
 * |________________|________________|________________|________________|
 * 
*/
void invokeAttention(DataType dt,
        void *__restrict__ output, // [num_tokens]
        const void *__restrict__ query, // [total_tokens, num_heads, head_size]
        const void *__restrict__ key, // [total_tokens, num_kv_heads, head_size]
        const void *__restrict__ value, // [total_tokens, num_kv_heads, head_size]
        int *query_shape, int *kv_shape, const int q_stride, const int kv_stride, const float scale,
        const int batch_size, const int *token_lens, const void *kcache, const void *vcache, int *kvcache_shape,
        int *block_tables, int *block_nums, int *context_lens, int layer_id, bool is_prefill, int *slot_mapping);

void invokeAttentionLLaMA(DataType dt, int batchSize, int inputSeqLen, int attHeadDim, int attHeadNum, int kvHeadNum,
        int maxPositions, int maxPosEmbed, int pastSeqLen, int currentSeqLen, int step, int hiddenSize, void *output,
        int outputStride, const void *input, int inputStride, const void *queryWeight, const void *keyWeight,
        const void *valueWeight, const void *attnOutWeight, const float *queryBias = nullptr,
        const float *keyBias = nullptr, const float *valueBias = nullptr, const float *attnOutBias = nullptr);

} // namespace xft