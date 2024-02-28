/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/cutlass/interface.h"
#include "src/fastertransformer/layers/GemmRunner.h"
#include "src/fastertransformer/layers/LoraGemm.h"
#include "src/fastertransformer/layers/attention_layers/BaseAttentionLayer.h"

namespace fastertransformer {

template<typename T>
class GptContextAttentionLayer: public BaseAttentionLayer<T> {
private:
    // buffer handling
    size_t max_batch_size_ = 0;
    size_t max_seq_len_    = 0;

    // metadata
    const size_t head_num_;
    const size_t head_num_kv_;
    const size_t size_per_head_;
    const size_t hidden_units_;
    const size_t local_head_num_;
    const size_t local_head_num_kv_;
    const size_t local_hidden_units_;
    const size_t rotary_embedding_dim_;
    const int    rotary_embedding_style_;
    const int    rotary_embedding_base_;
    const float  dynamic_embedding_scalar_;
    const int    dynamic_embedding_max_pos_;
    const int    position_embeddings_scale_;
    const int    base_scale_;

    const bool use_logn_attn_ = false;
    const int  logn_seq_len_ = 2048;
 
    // for sparse
    const bool is_sparse_head_ = false;
    const std::vector<int64_t> local_layer_head_num_;
    const std::vector<int64_t> local_layer_head_num_kv_;

    // fmha runner
    int                        sm_ = getSMVersion();

    void allocateBuffer() override;
    void allocateBuffer(size_t batch_size, size_t token_num, size_t seq_len, bool allocate_qk_buf);
    void freeBuffer() override;

    using BaseAttentionLayer<T>::is_free_buffer_after_forward_;
    using BaseAttentionLayer<T>::is_allocate_buffer_;
    using BaseAttentionLayer<T>::cublas_wrapper_;

    bool is_qk_buf_float_;

    std::shared_ptr<tensorrt_llm::kernels::cutlass_kernels::
                        CutlassFpAIntBGemmRunner<T, uint8_t, cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY>>
                                   weight_only_int8_fc_runner_;
    std::shared_ptr<GemmRunner<T>> gemm_runner_;
    std::shared_ptr<LoraGemm<T>>   lora_gemm_;

protected:
    using BaseAttentionLayer<T>::allocator_;
    using BaseAttentionLayer<T>::stream_;
    using BaseAttentionLayer<T>::sparse_;
    T*     qkv_buf_              = nullptr;
    T*     q_buf_2_              = nullptr;
    T*     k_buf_2_              = nullptr;
    T*     v_buf_2_              = nullptr;
    T*     qk_buf_               = nullptr;
    float* qk_buf_float_         = nullptr;
    T*     qkv_buf_2_            = nullptr;
    T*     qkv_buf_3_            = nullptr;
    T*     qkv_buf_3_normed_     = nullptr;
    char*  mixed_gemm_workspace_ = nullptr;
    size_t mixed_gemm_ws_bytes_  = 0;
    char*  int8_gemm_workspace_  = nullptr;
    size_t int8_gemm_ws_bytes_   = 0;

    // int8_mode_ == 0 means we don't use any mechanism related to INT8.
    // int8_mode_ == 1 for weight quantized only gemm for GPT
    // int8_mode_ == 2 for SmoothQuant O3 (per tensor scales)
    const int  int8_mode_ = 0;
    const bool int4_mode_ = false;

public:
    GptContextAttentionLayer(size_t               max_batch_size,
                             size_t               max_seq_len,
                             size_t               head_num,
                             size_t               head_num_kv,
                             size_t               size_per_head,
                             size_t               local_head_num,
                             size_t               local_head_num_kv,
                             std::vector<int64_t> local_layer_head_num,
                             std::vector<int64_t> local_layer_head_num_kv,
                             size_t               rotary_embedding_dim,
                             int                  rotary_embedding_style,
                             int                  rotary_embedding_base,
                             float                dynamic_embedding_scalar,
                             int                  dynamic_embedding_max_pos,
                             int                  position_embeddings_scale,
                             int                  base_scale,
                             int                  logn_seq_len,
                             cudaStream_t         stream,
                             cublasMMWrapper*     cublas_wrapper,
                             IAllocator*          allocator,
                             bool                 use_logn_attn,
                             bool                 is_free_buffer_after_forward,
                             bool                 is_qk_buf_float,
                             bool                 sparse          = false,
                             bool                 is_sparse_head_ = false,
                             int                  int8_mode       = 0,
                             bool                 int4_mode       = false);

    GptContextAttentionLayer(GptContextAttentionLayer<T> const& attention_layer);

    virtual ~GptContextAttentionLayer();

    void preAllocate() override;
    void
    forward(TensorMap* output_tensors, TensorMap* input_tensors, const AttentionWeight<T>* attention_weights) override;
    void Attention(TensorMap*                output_tensors,
                   TensorMap*                input_tensors,
                   const AttentionWeight<T>* attention_weights);
};

}  // namespace fastertransformer
