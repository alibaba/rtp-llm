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

#include "src/fastertransformer/layers/attention_layers/DecoderSelfAttentionLayer.h"
#include "src/fastertransformer/cuda/custom_ar_comm.h"
#include "src/fastertransformer/cuda/nccl/nccl_utils.h"

namespace fastertransformer {

template<typename T>
class TensorParallelDecoderSelfAttentionLayer: public DecoderSelfAttentionLayer<T> {
private:
    NcclParam                           tensor_para_;
    std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm_;
    int                                 enable_custom_all_reduce_;
    bool                                do_all_reduce_;

protected:
public:
    TensorParallelDecoderSelfAttentionLayer(size_t                              max_batch_size,
                                            size_t                              head_num,
                                            size_t                              head_num_kv,
                                            size_t                              size_per_head,
                                            std::vector<int64_t>                layer_head_num,
                                            std::vector<int64_t>                layer_head_num_kv,
                                            size_t                              rotary_embedding_dim,
                                            int                                 rotary_embedding_style,
                                            int                                 rotary_embedding_base,
                                            float                               dynamic_embedding_scalar,
                                            int                                 dynamic_embedding_max_pos,
                                            int                                 position_embeddings_scale,
                                            int                                 base_scale,
                                            size_t                              d_model,
                                            int                                 logn_seq_len,
                                            float                               q_scaling,
                                            NcclParam                           tensor_para,
                                            cudaStream_t                        stream,
                                            cublasMMWrapper*                    cublas_wrapper,
                                            IAllocator*                         allocator,
                                            bool                                use_logn_attn,
                                            bool                                do_all_reduce,
                                            bool                                is_free_buffer_after_forward,
                                            bool                                is_sparse                = false,
                                            bool                                is_sparse_head           = false,
                                            int                                 int8_mode                = 0,
                                            bool                                int4_mode                = false,
                                            std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm   = nullptr,
                                            int                                 enable_custom_all_reduce = 0);

    TensorParallelDecoderSelfAttentionLayer(TensorParallelDecoderSelfAttentionLayer<T> const& attention_layer);

    void
    forward(TensorMap* output_tensors, TensorMap* input_tensors, const AttentionWeight<T>* attention_weights) override;
};

}  // namespace fastertransformer
