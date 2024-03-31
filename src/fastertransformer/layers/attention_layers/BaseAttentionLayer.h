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

#include <assert.h>
#include <vector>

#include "src/fastertransformer/layers/BaseLayer.h"
#include "src/fastertransformer/layers/attention_layers/AttentionWeight.h"
#include "src/fastertransformer/core/Tensor.h"
#include "src/fastertransformer/core/allocator.h"
#include "src/fastertransformer/cuda/cublas/cublas.h"
#include "src/fastertransformer/cuda/cuda_fp8_utils.h"
#include "src/fastertransformer/cuda/memory_utils.h"

namespace fastertransformer {

enum class AttentionType {
    UNFUSED_MHA,
    UNFUSED_PADDED_MHA,
    FUSED_MHA,
    FUSED_PADDED_MHA
};

/* NOTE:
1. only swin-style relative position bias is supported currently
2. gpt-style (causal-mask) models support any-sequence-length fmha, so we don't need to call isValidSeqLen at run-time
3. bert/vit can also support any-seq-length fmha
*/
template<typename T>
AttentionType getAttentionType(size_t     size_per_head,
                               const int  sm,
                               const bool remove_padding,
                               const int  max_seq_len,
                               const bool is_fuse                          = true,
                               const bool with_swin_relative_position_bias = false,
                               const bool causal_mask                      = false)
{
    return remove_padding ? AttentionType::UNFUSED_MHA : AttentionType::UNFUSED_PADDED_MHA;
}

template<typename T>
AttentionType getAttentionTypeINT8(
    size_t size_per_head, const int sm, const bool remove_padding, const int max_seq_len, const int int8_mode)
{
    return remove_padding ? AttentionType::UNFUSED_MHA : AttentionType::UNFUSED_PADDED_MHA;    
}

inline bool isFusedMHA(AttentionType attention_type)
{
    return attention_type == AttentionType::FUSED_MHA || attention_type == AttentionType::FUSED_PADDED_MHA;
}

inline bool isUnPaddedMHA(AttentionType attention_type)
{
    return attention_type == AttentionType::FUSED_MHA || attention_type == AttentionType::UNFUSED_MHA;
}

inline bool isPaddedMHA(AttentionType attention_type)
{
    return attention_type == AttentionType::FUSED_PADDED_MHA || attention_type == AttentionType::UNFUSED_PADDED_MHA;
}

inline AttentionType getUnfusedAttentionType(AttentionType attention_type)
{
    if (attention_type == AttentionType::FUSED_MHA) {
        return AttentionType::UNFUSED_MHA;
    }
    else if (attention_type == AttentionType::FUSED_PADDED_MHA) {
        return AttentionType::UNFUSED_PADDED_MHA;
    }
    return attention_type;
}

template<typename T>
class BaseAttentionLayer: public BaseLayer {

public:
    virtual void
    forward(TensorMap* output_tensors, TensorMap* input_tensors, const AttentionWeight<T>* attention_weights) = 0;

    virtual bool UseFMHA() = 0;

    BaseAttentionLayer(cudaStream_t     stream,
                       cublasMMWrapper* cublas_wrapper,
                       IAllocator*      allocator,
                       bool             is_free_buffer_after_forward,
                       bool             sparse = false):
        BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, sparse)
    {
    }
    virtual ~BaseAttentionLayer() = default;
    virtual bool isValidSeqLen(const size_t seq_len)
    {
        return true;
    }
    virtual void preAllocate()
    {
        return ;
    }
};

}  // namespace fastertransformer
