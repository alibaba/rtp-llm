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

#include "src/fastertransformer/layers/attention_layers/DecoderSelfAttentionLayer.h"
#include "src/fastertransformer/kernels/decoder_masked_multihead_attention.h"
#include "src/fastertransformer/utils/logger.h"
#include "src/fastertransformer/cuda/memory_utils.h"
#include "src/fastertransformer/cuda/nvtx/nvtx_utils.h"
#include "src/fastertransformer/cuda/cuda_utils.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/kernels/kv_cache_utils.h"

namespace fastertransformer {

template<typename T>
void DecoderSelfAttentionLayer<T>::allocateBuffer()
{
    FT_CHECK_WITH_INFO(false, "Deprecated. Use `allocateBuffer(size_t batch_size)` instead");
}

template<typename T>
void DecoderSelfAttentionLayer<T>::allocateBuffer(size_t batch_size)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    qkv_buf_ =
        reinterpret_cast<T*>(allocator_->reMalloc(qkv_buf_, sizeof(T) * batch_size * (local_hidden_units_ + 2 * size_per_head_ * local_head_num_kv_), false));
    context_buf_ =
        reinterpret_cast<T*>(allocator_->reMalloc(context_buf_, sizeof(T) * batch_size * local_hidden_units_, false));

    if (int8_mode_ == 1) {
        // We use max_size for n and k since we reuse buffers for both FCs and want to allocate the max
        // possible memory that would be required by any of the individual gemms.
        const int max_size    = std::max(d_model_, local_hidden_units_ + 2 * size_per_head_ * local_head_num_kv_);
        mixed_gemm_ws_bytes_  = weight_only_int8_fc_runner_->getWorkspaceSize(batch_size, max_size, max_size);
        mixed_gemm_workspace_ = (char*)allocator_->reMalloc(mixed_gemm_workspace_, mixed_gemm_ws_bytes_, false);
    } else if (int8_mode_ == 2){
        FT_LOG_ERROR("int8_mode == 2 not support");
        abort();
    }

    is_allocate_buffer_ = true;
}

template<typename T>
void DecoderSelfAttentionLayer<T>::freeBuffer()
{
    if (is_allocate_buffer_) {
        allocator_->free((void**)(&qkv_buf_));
        allocator_->free((void**)(&context_buf_));
        is_allocate_buffer_ = false;

        if (mixed_gemm_workspace_) {
            allocator_->free((void**)(&mixed_gemm_workspace_));
            mixed_gemm_ws_bytes_ = 0;
        }
    }
}

template<typename T>
bool DecoderSelfAttentionLayer<T>::isValidBatchSize(size_t batch_size)
{
    if (batch_size <= max_batch_size_) {
        return true;
    }
    else {
        freeBuffer();
        max_batch_size_ = batch_size * 1.2;
        return true;
    }
}

template<typename T>
DecoderSelfAttentionLayer<T>::DecoderSelfAttentionLayer(size_t               max_batch_size,
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
                                                        size_t               d_model,
                                                        int                  logn_seq_len,
                                                        const float          q_scaling,
                                                        cudaStream_t         stream,
                                                        cublasMMWrapper*     cublas_wrapper,
                                                        IAllocator*          allocator,
                                                        bool                 use_logn_attn,
                                                        bool                 is_free_buffer_after_forward,
                                                        bool                 sparse,
                                                        bool                 is_sparse_head,
                                                        int                  int8_mode,
                                                        bool                 int4_mode):
    BaseAttentionLayer<T>(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, sparse),
    max_batch_size_(max_batch_size),
    head_num_(head_num),
    head_num_kv_(head_num_kv),
    size_per_head_(size_per_head),
    hidden_units_(head_num_ * size_per_head_),
    local_head_num_(local_head_num),
    local_head_num_kv_(local_head_num_kv),
    local_layer_head_num_(local_layer_head_num),
    local_layer_head_num_kv_(local_layer_head_num_kv),
    local_hidden_units_(local_head_num_ * size_per_head_),
    rotary_embedding_dim_(rotary_embedding_dim),
    rotary_embedding_style_(rotary_embedding_style),
    rotary_embedding_base_(rotary_embedding_base),
    dynamic_embedding_scalar_(dynamic_embedding_scalar),
    dynamic_embedding_max_pos_(dynamic_embedding_max_pos),
    position_embeddings_scale_(position_embeddings_scale),
    base_scale_(base_scale),
    logn_seq_len_(logn_seq_len),
    use_logn_attn_(use_logn_attn),
    d_model_(d_model),
    q_scaling_(q_scaling),
    is_sparse_head_(is_sparse_head),
    int8_mode_(int8_mode),
    int4_mode_(int4_mode) {
    if (int8_mode_ == 1) {
        FT_CHECK_WITH_INFO(!(std::is_same<T, float>::value), "Weight only quant not supported for fp32.");
        weight_only_int8_fc_runner_ = std::make_shared<
            tensorrt_llm::kernels::cutlass_kernels::
                CutlassFpAIntBGemmRunner<T, uint8_t, cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY>>();
    } else if (int8_mode_ == 2) {
        FT_LOG_ERROR("int8_mode == 2 not support");
        abort();
    }
    gemm_runner_ = std::make_shared<GemmRunner<T>>(stream, allocator, cublas_wrapper, int8_mode_);
}

template<typename T>
DecoderSelfAttentionLayer<T>::DecoderSelfAttentionLayer(DecoderSelfAttentionLayer<T> const& attention_layer):
    DecoderSelfAttentionLayer<T>(attention_layer.max_batch_size_,
                                 attention_layer.head_num_,
                                 attention_layer.head_num_kv_,
                                 attention_layer.size_per_head_,
                                 attention_layer.local_head_num_,
                                 attention_layer.local_head_num_kv_,
                                 attention_layer.local_layer_head_num_,
                                 attention_layer.local_layer_head_num_kv_,
                                 attention_layer.rotary_embedding_dim_,
                                 attention_layer.rotary_embedding_style_,
                                 attention_layer.rotary_embedding_base_,
                                 attention_layer.dynamic_embedding_scalar_,
                                 attention_layer.dynamic_embedding_max_pos_,
                                 attention_layer.position_embeddings_scale_,
                                 attention_layer.base_scale_,
                                 attention_layer.d_model_,
                                 attention_layer.logn_seq_len_,
                                 attention_layer.q_scaling_,
                                 attention_layer.stream_,
                                 attention_layer.cublas_wrapper_,
                                 attention_layer.allocator_,
                                 attention_layer.use_logn_attn_,
                                 attention_layer.is_free_buffer_after_forward_,
                                 attention_layer.sparse_,
                                 attention_layer.is_sparse_head_,
                                 attention_layer.int8_mode_)
{
}

template<typename T>
DecoderSelfAttentionLayer<T>::~DecoderSelfAttentionLayer()
{
    cublas_wrapper_ = nullptr;
    freeBuffer();
}

template<typename T>
void DecoderSelfAttentionLayer<T>::preAllocate()
{
    if (max_batch_size_ > 0) {
        allocateBuffer(max_batch_size_);
    }
}

template<typename T>
void DecoderSelfAttentionLayer<T>::forward(TensorMap*                output_tensors,
                                           TensorMap*                input_tensors,
                                           const AttentionWeight<T>* attention_weights)
{
    // input tensors:
    //      input_query [batch_size, d_model_],
    //      sequence_lengths [batch_size]
    //      step [1] on cpu
    //      finished [batch_size] (optional)
    //      input_lengths [batch_size] (optional)
    //      max_input_length [1] on cpu (optional)
    //      masked_tokens [batch_size, memory_len], (optional)
    //      cache_indirection [batch_size / beam_width, beam_width, memory_max_len] (optional)
    //      d_prefix_prompt_lengths [batch_size] (optional)
    //      max_prefix_prompt_length [1] on cpu (optional)
    //      relative_attention_bias [1, head_num, step, step] or [1, head_num, max_seq_len, max_seq_len] (optional)
    //      linear_bias_slopes [head_num] (optional)
    //      ia3_tasks [batch_size] (optional)

    // output tensors:
    //      attention_output [batch_size, d_model_],
    //      key_cache [batch, local_head_num, size_per_head // x, memory_max_len, x]
    //      value_cache [batch, local_head_num, memory_max_len, size_per_head]

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    FT_CHECK(input_tensors->isExist("sequence_lengths"));
    // FT_CHECK(output_tensors->at("key_cache").shape().size() == 5 || output_tensors->at("key_cache").shape().size() == 3);
    // FT_CHECK(output_tensors->at("value_cache").shape().size() == 4
    //          || output_tensors->at("value_cache").shape().size() == 3);
    allocateBuffer(input_tensors->at("input_query").shape()[0]);

    const T*    attention_input         = input_tensors->getPtr<T>("input_query");
    const int*  sequence_lengths        = input_tensors->getPtr<int>("sequence_lengths");
    const bool* finished                = input_tensors->getPtr<bool>("finished", nullptr);
    const bool* masked_tokens           = input_tensors->getPtr<bool>("masked_tokens", nullptr);
    const int*  cache_indir             = input_tensors->getPtr<int>("cache_indirection", nullptr);
    const int*  block_index_map         = input_tensors->getPtr<int>("block_index_map", nullptr);
    const int   layer_id                = input_tensors->getVal<int>("layer_id");
    const T*    relative_attention_bias = input_tensors->getPtr<T>("relative_attention_bias", nullptr);
    const int   relative_attention_bias_stride =
        input_tensors->isExist("relative_attention_bias") ? input_tensors->at("relative_attention_bias").shape()[3] : 0;
    const T*   linear_bias_slopes = input_tensors->getPtr<T>("linear_bias_slopes", nullptr);

    T* attention_out = output_tensors->getPtr<T>("hidden_features");
    T* key_cache     = output_tensors->getPtr<T>("key_cache");
    T* value_cache   = output_tensors->getPtr<T>("value_cache");
    auto& key_cache_tensor   = output_tensors->at("key_cache");
    auto& value_cache_tensor = output_tensors->at("value_cache");

    const int batch_size     = input_tensors->at("input_query").shape()[0];
    const int beam_width     = cache_indir != nullptr ? input_tensors->at("cache_indirection").shape()[1] : 1;
    const int memory_max_len = output_tensors->at("key_cache").shape()[2];
    const int* d_prefix_prompt_lengths  = input_tensors->getPtr<int>("d_prefix_prompt_lengths", nullptr);
    const int  max_prefix_prompt_length = input_tensors->getVal<int>("max_prefix_prompt_length", 0);
    const bool count_prefix_length      = input_tensors->getVal<bool>("count_prefix_length", false);

    const int local_hidden_units_rt = (is_sparse_head_ ? local_layer_head_num_[layer_id]: local_head_num_) * size_per_head_;
    const int local_hidden_units_kv_rt = (is_sparse_head_ ? local_layer_head_num_kv_[layer_id]: local_head_num_kv_) * size_per_head_;
    const int local_head_num = is_sparse_head_? local_layer_head_num_[layer_id] : local_head_num_;
    const int local_head_num_kv = is_sparse_head_? local_layer_head_num_kv_[layer_id] : local_head_num_kv_;

    // lora
    int* lora_ids = input_tensors->getPtr<int>("lora_ids", nullptr);
    const int* lora_input_lengths = input_tensors->getPtr<int>("lora_input_lengths", nullptr);

    int max_blocks_per_batch = 0;
    if (block_index_map) {
        max_blocks_per_batch = input_tensors->at("block_index_map").shape()[1];
    }
    const int m_padded = 8 * div_up(batch_size, 8);
#ifdef SPARSITY_ENABLED
    bool use_sparse_gemm = sparse_ && cublas_wrapper_->isUseSparse(1, local_hidden_units_rt + 2 * local_hidden_units_kv_rt, m_padded, d_model_);
#else
    constexpr bool use_sparse_gemm = false;
#endif

    PUSH_RANGE(stream_, "qkv_gemm");
    // QKV gemm: [batch_size, hidden_units] * [hidden_units, qkv_dim] -> [batch_size, qkv_dim]
    gemm_runner_->Gemm(batch_size,
                       local_hidden_units_rt + 2 * local_hidden_units_kv_rt,
                       d_model_,
                       attention_input,
                       &attention_weights->query_weight,
                       qkv_buf_);
    
    // lora

    lora_gemm_->applyLoRA(batch_size,
                          batch_size,
                          lora_input_lengths,
                          d_model_,
                          local_hidden_units_rt + 2 * local_hidden_units_kv_rt,
                          lora_ids,
                          attention_weights->query_weight.lora_weights,
                          attention_input,
                          qkv_buf_);

    int k_start = local_hidden_units_rt;
    int v_start = local_hidden_units_rt + local_hidden_units_kv_rt;
    print_bsd(layer_id, "self q", qkv_buf_, 1, 1, local_hidden_units_rt + 2 * local_hidden_units_kv_rt,
              0, 20);
    print_bsd(layer_id, "self k", qkv_buf_, 1, 1, local_hidden_units_rt + 2 * local_hidden_units_kv_rt,
              k_start, k_start + 20);
    print_bsd(layer_id, "self v", qkv_buf_, 1, 1, local_hidden_units_rt + 2 * local_hidden_units_kv_rt,
              v_start, v_start + 20);

    sync_check_cuda_error();
    POP_RANGE;
    KVLinearBuffer kv_cache_buffer(batch_size, 1, memory_max_len, local_head_num_kv_ * size_per_head_ * sizeof(T));
    kv_cache_buffer.k_data = reinterpret_cast<int8_t*>(key_cache);
    kv_cache_buffer.v_data = reinterpret_cast<int8_t*>(value_cache);
    fusedQKV_masked_attention_dispatch<T>(
        qkv_buf_,
        attention_weights->query_weight.bias,
        relative_attention_bias,
        cache_indir,
        context_buf_,
        finished,
        sequence_lengths,  // NOTE: current seq len including padding (fixed after meeting the finished id)
        batch_size,
        beam_width,
        local_head_num,
        local_head_num_kv,
        size_per_head_,
        rotary_embedding_dim_,
        rotary_embedding_style_,
        rotary_embedding_base_,
        logn_seq_len_,
        use_logn_attn_,
        dynamic_embedding_scalar_,
        dynamic_embedding_max_pos_,
        position_embeddings_scale_,
        base_scale_,
        memory_max_len,
        d_prefix_prompt_lengths,
        max_prefix_prompt_length,
        count_prefix_length,
        input_tensors->getPtr<int>("input_lengths", nullptr),
        input_tensors->getVal<int>("step"),
        q_scaling_,
        relative_attention_bias_stride,
        linear_bias_slopes,
        masked_tokens,
        int8_mode_ == 2 ? attention_weights->query_weight.scale_out : nullptr,
        int8_mode_ == 2 ? attention_weights->attention_output_weight.scale : nullptr,
        int8_mode_,
        false,
        0,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        kv_cache_buffer,
        stream_);
    sync_check_cuda_error();

    // assert(key_cache_tensor.shape().size() == 6);
    // assert(value_cache_tensor.shape().size() == 5);
    // auto& kc_shape = key_cache_tensor.shape();
    // print_kv_cache(layer_id, "key_cache", key_cache, kc_shape[0],
    //     kc_shape[1], kc_shape[2], kc_shape[3], kc_shape[4], kc_shape[5]);
    // auto& vc_shape = value_cache_tensor.shape();
    // print_kv_cache(layer_id, "value_cache", value_cache, 1, vc_shape[0],
    //     vc_shape[1], vc_shape[2], vc_shape[3], vc_shape[4]);

    PUSH_RANGE(stream_, "proj_gemm");
#ifdef SPARSITY_ENABLED
    use_sparse_gemm = sparse_ && cublas_wrapper_->isUseSparse(1, d_model_, m_padded, local_hidden_units_rt);
#endif

    print_bshd(layer_id, "qkv_weighted_t", context_buf_, 1, 1, local_head_num, size_per_head_);

    float layernorm_eps = 1E-5;
    T* context_buf_input = nullptr;
    if (attention_weights->attention_layernorm.gamma && attention_weights->attention_layernorm.beta) {
        invokeGeneralLayerNorm(qkv_buf_,
                                context_buf_,
                                attention_weights->attention_layernorm.gamma,
                                attention_weights->attention_layernorm.beta,
                                layernorm_eps,
                                batch_size,
                                local_hidden_units_rt,
                                nullptr,
                                nullptr,
                                int8_mode_,
                                stream_);
        context_buf_input = qkv_buf_;
        sync_check_cuda_error();
    } else {
        context_buf_input = context_buf_;
    }

    print_bsd(layer_id, "decoder attn before o", context_buf_input, 1, 1, local_hidden_units_rt);
    print_bsd_sum_and_square(layer_id, "decoder attn before o", context_buf_input, 1, 1, local_hidden_units_rt);

    // attention out gemm: [batch_size, local_hidden_units_rt] * [local_hidden_units_rt, hidden_units_]
    gemm_runner_->Gemm( batch_size,
                        d_model_,
                        local_hidden_units_rt,
                        context_buf_input,
                        &attention_weights->attention_output_weight,
                        attention_out);
    
    // lora
    lora_gemm_->applyLoRA(batch_size,
                          batch_size,
                          lora_input_lengths,
                          local_hidden_units_rt,
                          d_model_,
                          lora_ids,
                          attention_weights->attention_output_weight.lora_weights,
                          context_buf_input,
                          attention_out);

    POP_RANGE;

    sync_check_cuda_error();
    print_bsd(layer_id, "attn output", attention_out, 1, 1, hidden_units_);

    if (is_free_buffer_after_forward_) {
        freeBuffer();
    }
}

template class DecoderSelfAttentionLayer<float>;
template class DecoderSelfAttentionLayer<half>;
#ifdef ENABLE_BF16
template class DecoderSelfAttentionLayer<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
