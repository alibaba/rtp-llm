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

#include "src/fastertransformer/layers/attention_layers/TensorParallelDecoderSelfAttentionLayer.h"
#include "src/fastertransformer/utils/nvtx_utils.h"

namespace fastertransformer {

template<typename T>
TensorParallelDecoderSelfAttentionLayer<T>::TensorParallelDecoderSelfAttentionLayer(
    size_t                              max_batch_size,
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
    bool                                is_sparse,
    bool                                is_sparse_head,
    int                                 int8_mode,
    std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm,
    int                                 enable_custom_all_reduce):
    DecoderSelfAttentionLayer<T>(max_batch_size,
                                 head_num,
                                 head_num_kv,
                                 size_per_head,
                                 head_num / tensor_para.world_size_,
                                 head_num_kv == 1 ? 1 : head_num_kv / tensor_para.world_size_,
                                 getLocalParameter(layer_head_num, tensor_para.world_size_),
                                 getLocalParameter(layer_head_num_kv, tensor_para.world_size_),
                                 rotary_embedding_dim,
                                 rotary_embedding_style,
                                 rotary_embedding_base,
                                 dynamic_embedding_scalar,
                                 dynamic_embedding_max_pos,
                                 position_embeddings_scale,
                                 base_scale,
                                 d_model,
                                 logn_seq_len,
                                 q_scaling,  // NOTE
                                 stream,
                                 cublas_wrapper,
                                 allocator,
                                 use_logn_attn,
                                 is_free_buffer_after_forward,
                                 is_sparse,
                                 is_sparse_head,
                                 int8_mode),
    do_all_reduce_(do_all_reduce),
    tensor_para_(tensor_para),
    custom_all_reduce_comm_(custom_all_reduce_comm),
    enable_custom_all_reduce_(enable_custom_all_reduce)
{
    FT_CHECK(head_num % tensor_para_.world_size_ == 0);
}

template<typename T>
TensorParallelDecoderSelfAttentionLayer<T>::TensorParallelDecoderSelfAttentionLayer(
    TensorParallelDecoderSelfAttentionLayer<T> const& attention_layer):
    DecoderSelfAttentionLayer<T>(attention_layer),
    do_all_reduce_(attention_layer.do_all_reduce_),
    tensor_para_(attention_layer.tensor_para_),
    custom_all_reduce_comm_(attention_layer.custom_all_reduce_comm_),
    enable_custom_all_reduce_(attention_layer.enable_custom_all_reduce_)
{
}

template<typename T>
void TensorParallelDecoderSelfAttentionLayer<T>::forward(TensorMap*                output_tensors,
                                                         TensorMap*                input_tensors,
                                                         const AttentionWeight<T>* attention_weights)
{
    // input tensors:
    //      attention_input [batch_size, hidden_dimension],
    //      finished [batch_size],
    //      sequence_lengths [batch_size]
    //      input_lengths [batch_size]
    //      max_input_length [1] on cpu
    //      step [1] on cpu

    // output tensors:
    //      attention_output [batch_size, hidden_dimension],
    //      key_cache [batch, head_num, size_per_head // x, max_seq_len, x]
    //      value_cache [batch, head_num, max_seq_len, size_per_head]

    const size_t size = output_tensors->at("hidden_features").size();

    bool use_custom_all_reduce_kernel = false;
    if (enable_custom_all_reduce_ && custom_all_reduce_comm_ != nullptr && do_all_reduce_) {
        std::vector<Tensor> reduce_tensor{output_tensors->at("hidden_features")};
        use_custom_all_reduce_kernel = custom_all_reduce_comm_->swapInternalBuffer(&reduce_tensor, size);
    }

    DecoderSelfAttentionLayer<T>::forward(output_tensors, input_tensors, attention_weights);

    // PUSH_RANGE(stream_, "all reduce sum");
    T* attention_out = output_tensors->getPtr<T>("hidden_features");
    if (tensor_para_.world_size_ > 1 && do_all_reduce_) {
        if (!use_custom_all_reduce_kernel) {
            ftNcclAllReduceSum(attention_out, attention_out, size, tensor_para_, DecoderSelfAttentionLayer<T>::stream_);
        }
        else {
            custom_all_reduce_comm_->customAllReduce(size, DecoderSelfAttentionLayer<T>::stream_);
        }
        sync_check_cuda_error();
    }
    // POP_RANGE;
}

template class TensorParallelDecoderSelfAttentionLayer<float>;
template class TensorParallelDecoderSelfAttentionLayer<half>;
#ifdef ENABLE_BF16
template class TensorParallelDecoderSelfAttentionLayer<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
