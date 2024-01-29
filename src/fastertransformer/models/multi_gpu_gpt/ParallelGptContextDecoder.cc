/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2022, SK Telecom Authored by A. Dialog
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

#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGptContextDecoder.h"
#include "src/fastertransformer/kernels/bert_preprocess_kernels.h"
#include "src/fastertransformer/kernels/gpt_kernels.h"
#include "src/fastertransformer/cuda/nvtx/nvtx_utils.h"
#include <thread>

namespace fastertransformer {

template<typename T>
void ParallelGptContextDecoder<T>::initialize()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    self_attention_layer_.reset(new TensorParallelGptContextAttentionLayer<T>(
                                        max_batch_size_,
                                        max_seq_len_,
                                        gpt_init_parameter_.head_num_,
                                        gpt_init_parameter_.head_num_kv_,
                                        gpt_init_parameter_.size_per_head_,
                                        gpt_init_parameter_.layer_head_num_,
                                        gpt_init_parameter_.layer_head_num_kv_,
                                        gpt_init_parameter_.rotary_embedding_dim_,
                                        gpt_init_parameter_.rotary_embedding_style_,
                                        gpt_init_parameter_.rotary_embedding_base_,
                                        gpt_init_parameter_.dynamic_embedding_scalar_,
                                        gpt_init_parameter_.dynamic_embedding_max_pos_,
                                        gpt_init_parameter_.position_embeddings_scale_,
                                        gpt_init_parameter_.base_scale_,
                                        gpt_init_parameter_.logn_seq_len_,
                                        tensor_para_,
                                        stream_,
                                        cublas_wrapper_,
                                        allocator_,
                                        gpt_init_parameter_.use_logn_attn_,
                                        true,
                                        is_free_buffer_after_forward_,
                                        is_qk_buf_float_,
                                        sparse_,
                                        gpt_init_parameter_.is_sparse_head_,
                                        gpt_init_parameter_.int8_mode_,
                                        custom_all_reduce_comm_,
                                        enable_custom_all_reduce_));

    ffn_layer_.reset(new TensorParallelFfnLayer<T>(
                             max_batch_size_,
                             1,
                             gpt_init_parameter_.head_num_,
                             gpt_init_parameter_.size_per_head_,
                             gpt_init_parameter_.expert_num_,  // expert_num
                             gpt_init_parameter_.inter_size_,
                             gpt_init_parameter_.inter_padding_size_,
                             gpt_init_parameter_.layer_inter_size_,
                             gpt_init_parameter_.layer_inter_padding_size_,
                             tensor_para_,
                             stream_,
                             cublas_wrapper_,
                             allocator_,
                             true,
                             is_free_buffer_after_forward_,
                             sparse_,
                             gpt_init_parameter_.is_sparse_head_,
                             gpt_init_parameter_.int8_mode_,
                             gpt_init_parameter_.activation_type_,
                             gpt_init_parameter_.layernorm_eps_,
                             custom_all_reduce_comm_,
                             enable_custom_all_reduce_));

    norm_wrapper_.reset(new NormWrapper<T>(gpt_init_parameter_.layernorm_type_,
                                           gpt_init_parameter_.norm_type_,
                                           T(sqrt(2 * gpt_init_parameter_.num_layers_))));
}

template<typename T>
void ParallelGptContextDecoder<T>::allocateBuffer()
{
    FT_CHECK(false);
}

template<typename T>
void ParallelGptContextDecoder<T>::allocateBuffer(size_t batch_size, size_t seq_len,
                                                  bool use_shared_contexts,
                                                  bool reuse_buf,
                                                  bool pre_attn_ln)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    size_t hidden_units = gpt_init_parameter_.size_per_head_ * gpt_init_parameter_.head_num_;
    decoder_layer_output_ = reinterpret_cast<T*>(
        allocator_->reMalloc(decoder_layer_output_, sizeof(T) * batch_size * seq_len * hidden_units, false));
    decoder_normed_input_ = reinterpret_cast<T*>(
        allocator_->reMalloc(decoder_normed_input_, sizeof(T) * batch_size * seq_len * hidden_units, false));
    self_attn_output_ = reinterpret_cast<T*>(
        allocator_->reMalloc(self_attn_output_, sizeof(T) * batch_size * seq_len * hidden_units, false));
    if (!reuse_buf) {
        normed_self_attn_output_ = reinterpret_cast<T*>(
                allocator_->reMalloc(normed_self_attn_output_, sizeof(T) * batch_size * seq_len * hidden_units, false));
    } else {
        normed_self_attn_output_ = decoder_normed_input_;
    }
    if (pre_attn_ln) {
        attn_normed_input_ = reinterpret_cast<T*>(
                allocator_->reMalloc(attn_normed_input_, sizeof(T) * batch_size * seq_len * hidden_units, false));
    }
    if (gpt_init_parameter_.int8_mode_ == 2) {
        FT_LOG_ERROR("int8_mode == 2 not support");
        abort();
    }
    h_pinned_token_num_ptr_ = (size_t*)allocator_->reMalloc(h_pinned_token_num_ptr_, sizeof(size_t), true, true);
    padding_offset_ =
        reinterpret_cast<int*>(allocator_->reMalloc(padding_offset_, sizeof(int) * batch_size * seq_len, false));
    cu_seqlens_ = reinterpret_cast<int*>(allocator_->reMalloc(cu_seqlens_, sizeof(int) * (batch_size + 1), false));
    // for moe
    expert_scales_ = reinterpret_cast<T*>(
        allocator_->malloc(sizeof(T) * pad_to_multiple_of_16(gpt_init_parameter_.moe_k_ * batch_size * seq_len), false));
    expanded_source_row_to_expanded_dest_row_ = reinterpret_cast<int*>(
        allocator_->malloc(sizeof(int) * pad_to_multiple_of_16(gpt_init_parameter_.moe_k_ * batch_size * seq_len), false));
    expert_for_source_row_ = reinterpret_cast<int*>(
        allocator_->malloc(sizeof(int) * pad_to_multiple_of_16(gpt_init_parameter_.moe_k_ * batch_size * seq_len), false));
    fc2_result_ = reinterpret_cast<T*>(
        allocator_->malloc(sizeof(T) * pad_to_multiple_of_16(gpt_init_parameter_.moe_k_ * batch_size * seq_len * hidden_units), false));

    is_allocate_buffer_ = true;

    if (use_shared_contexts) {
        compact_decoder_features_ = reinterpret_cast<T*>(
            allocator_->reMalloc(compact_decoder_features_, sizeof(T) * batch_size * seq_len * hidden_units, false));
        compact_attention_mask_ = reinterpret_cast<T*>(
            allocator_->reMalloc(compact_attention_mask_, sizeof(T) * batch_size * seq_len * seq_len, false));
        compact_input_lengths_ =
            reinterpret_cast<int*>(allocator_->reMalloc(compact_input_lengths_, sizeof(int) * batch_size, false));
        k_cache_layer_ = reinterpret_cast<T*>(
            allocator_->reMalloc(k_cache_layer_, sizeof(T) * batch_size * seq_len * (gpt_init_parameter_.size_per_head_ * gpt_init_parameter_.head_num_kv_), false));
        v_cache_layer_ = reinterpret_cast<T*>(
            allocator_->reMalloc(v_cache_layer_, sizeof(T) * batch_size * seq_len * (gpt_init_parameter_.size_per_head_ * gpt_init_parameter_.head_num_kv_), false));
    }
}

template<typename T>
void ParallelGptContextDecoder<T>::freeBuffer()
{
    if (is_allocate_buffer_) {
        FT_LOG_DEBUG(__PRETTY_FUNCTION__);
        allocator_->free((void**)(&decoder_normed_input_));
        allocator_->free((void**)(&self_attn_output_));
        allocator_->free((void**)(&decoder_layer_output_));
        allocator_->free((void**)(&h_pinned_token_num_ptr_), true);
        allocator_->free((void**)(&padding_offset_));
        allocator_->free((void**)(&cu_seqlens_));

        allocator_->free((void**)(&expert_scales_));
        allocator_->free((void**)(&expanded_source_row_to_expanded_dest_row_));
        allocator_->free((void**)(&expert_for_source_row_));
        allocator_->free((void**)(&fc2_result_));

        if (compact_attention_mask_ != nullptr) {
            allocator_->free((void**)(&compact_decoder_features_));
            allocator_->free((void**)(&compact_attention_mask_));
            allocator_->free((void**)(&compact_input_lengths_));
            allocator_->free((void**)(&k_cache_layer_));
            allocator_->free((void**)(&v_cache_layer_));
        }
        if (gpt_init_parameter_.int8_mode_ == 2) {
            allocator_->free((void**)(&attention_query_dynamic_scale_));
            allocator_->free((void**)(&ffn_intermediate_dynamic_scale_));
        }
        is_allocate_buffer_ = false;
    }
}

template<typename T>
bool ParallelGptContextDecoder<T>::isValidLayerParallelId(uint l)
{
    uint local_num_layer = (uint)(ceil(gpt_init_parameter_.num_layers_ * 1.0f / pipeline_para_.world_size_));
    return l < gpt_init_parameter_.num_layers_ && (l >= local_num_layer * pipeline_para_.rank_)
           && (l < local_num_layer * (pipeline_para_.rank_ + 1));
}

template<typename T>
bool ParallelGptContextDecoder<T>::isFirstLayerParallelId(uint l)
{
    uint local_num_layer = (uint)(ceil(gpt_init_parameter_.num_layers_ * 1.0f / pipeline_para_.world_size_));
    return l < gpt_init_parameter_.num_layers_ && (l == local_num_layer * pipeline_para_.rank_);
}

template<typename T>
bool ParallelGptContextDecoder<T>::isLastLayerParallelId(uint l)
{
    uint local_num_layer = (uint)(ceil(gpt_init_parameter_.num_layers_ * 1.0f / pipeline_para_.world_size_));
    return l < gpt_init_parameter_.num_layers_ && (l == local_num_layer * (pipeline_para_.rank_ + 1) - 1);
}

template<typename T>
int ParallelGptContextDecoder<T>::getFirstLayerParallelId()
{
    uint local_num_layer = (uint)(ceil(gpt_init_parameter_.num_layers_ * 1.0f / pipeline_para_.world_size_));
    return local_num_layer * pipeline_para_.rank_;
}

template<typename T>
ParallelGptContextDecoder<T>::ParallelGptContextDecoder(size_t                              max_batch_size,
                                                        size_t                              max_seq_len,
                                                        const GptInitParameter&             gpt_init_parameter,
                                                        NcclParam                           tensor_para,
                                                        NcclParam                           pipeline_para,
                                                        cudaStream_t                        stream,
                                                        cublasMMWrapper*                    cublas_wrapper,
                                                        IAllocator*                         allocator,
                                                        bool                                is_free_buffer_after_forward,
                                                        bool                                is_qk_buf_float,
                                                        AttentionType                       attention_type,
                                                        bool                                sparse,
                                                        std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm,
                                                        int                                 enable_custom_all_reduce):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, sparse),
    max_batch_size_(max_batch_size),
    max_seq_len_(max_seq_len),
    gpt_init_parameter_(gpt_init_parameter),
    tensor_para_(tensor_para),
    pipeline_para_(pipeline_para),
    custom_all_reduce_comm_(custom_all_reduce_comm),
    enable_custom_all_reduce_(enable_custom_all_reduce),
    attention_type_(attention_type),
    is_qk_buf_float_(is_qk_buf_float)
{
    initialize();
}


template<typename T>
ParallelGptContextDecoder<T>::ParallelGptContextDecoder(ParallelGptContextDecoder<T> const& decoder):
    BaseLayer(decoder.stream_, decoder.cublas_wrapper_, decoder.allocator_, decoder.is_free_buffer_after_forward_),
    max_batch_size_(decoder.max_batch_size_),
    max_seq_len_(decoder.max_seq_len_),
    gpt_init_parameter_(decoder.gpt_init_parameter_),
    tensor_para_(decoder.tensor_para_),
    pipeline_para_(decoder.pipeline_para_),
    custom_all_reduce_comm_(decoder.custom_all_reduce_comm_),
    enable_custom_all_reduce_(decoder.enable_custom_all_reduce_),
    attention_type_(decoder.attention_type_),
    is_qk_buf_float_(decoder.is_qk_buf_float_)
{
    initialize();
}

template<typename T>
ParallelGptContextDecoder<T>::~ParallelGptContextDecoder()
{
    freeBuffer();
}

template<typename T>
void ParallelGptContextDecoder<T>::forward(
    TensorMap*                                            output_tensors,
    const TensorMap*                                      input_tensors,
    const std::vector<ParallelGptDecoderLayerWeight<T>*>* gpt_decoder_layer_weight)
{
    // input tensors:
    //      decoder_input [batch_size, seq_len, hidden_dimension],
    //      attention_mask [batch_size, 1, seq_len, seq_len]
    //      input_lengths [batch_size]
    //      compact_idx [compact_size], optional
    //      batch_to_compact_idx [batch_size], optional
    //      linear_bias_slopes [head_num], optional

    // output tensors:
    //      decoder_output [batch_size, seq_len, hidden_dimension],
    //      key_cache [num_layer, batch, local_head_num, size_per_head // x, max_seq_len, x]
    //      value_cache [num_layer, batch, local_head_num, max_seq_len, size_per_head]
    //      last_token_hidden_units [batch_size, hidden_dimension]

    // To use layer/pipeline parallelism, we view the shape of 'batch_size' to 'ite * local_batch_size'.
    // For example, the shape of decoder_input becomes [ite, batch_size, seq_len, hidden_dimension] during
    // computing.

    FT_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    FT_CHECK(input_tensors->isExist("decoder_input"));
    FT_CHECK(input_tensors->isExist("attention_mask"));
    FT_CHECK(input_tensors->isExist("input_lengths"));
    FT_CHECK(input_tensors->isExist("lora_ids"));
    FT_CHECK(input_tensors->isExist("lora_input_lengths"));
    FT_CHECK(output_tensors->isExist("decoder_output"));
    FT_CHECK(output_tensors->isExist("key_cache"));
    FT_CHECK(output_tensors->isExist("value_cache"));
    FT_CHECK(output_tensors->isExist("last_token_hidden_units"));

    const bool use_shared_contexts = input_tensors->isExist("compact_idx");
    FT_CHECK(!use_shared_contexts || input_tensors->isExist("batch_to_compact_idx"));

    size_t hidden_units = gpt_init_parameter_.size_per_head_ * gpt_init_parameter_.head_num_;
    Tensor decoder_input_tensor = input_tensors->at("decoder_input");
    FT_CHECK(decoder_input_tensor.shape()[2] == hidden_units);

    // Request batch size
    const size_t request_batch_size = decoder_input_tensor.shape()[0];
    // Maybe compacted batch size.
    const size_t batch_size =
        use_shared_contexts ? input_tensors->at("compact_idx").shape()[0] : decoder_input_tensor.shape()[0];
    // Request input length
    const size_t seq_len = decoder_input_tensor.shape()[1];
    // The maximum length of generation.
    const size_t max_seq_len = output_tensors->at("value_cache").shape()[3];

    const DataType data_type = getTensorType<T>();

    int max_prefix_prompt_length = 0;

    if (input_tensors->isExist("d_prefix_prompt_lengths")) {
        auto attention_mask_tensor = input_tensors->at("attention_mask");
        int seq_len = attention_mask_tensor.shape()[1];
        int total_len = attention_mask_tensor.shape()[2];
        max_prefix_prompt_length = total_len - seq_len;
    }

    PUSH_RANGE(stream_, "buffer_allocation");
    bool reuse_buf = !gpt_init_parameter_.use_norm_input_residual_;
    bool pre_attn_ln = gpt_decoder_layer_weight->at(0)->pre_attn_layernorm_weights.gamma;
    allocateBuffer(batch_size, seq_len, use_shared_contexts, reuse_buf, pre_attn_ln);
    POP_RANGE;

    PUSH_RANGE(stream_, "compact_inputs");
    if (use_shared_contexts) {
        invokeCompactInputs(compact_decoder_features_,
                            compact_attention_mask_,
                            compact_input_lengths_,
                            decoder_input_tensor.getPtr<T>(),
                            input_tensors->at("attention_mask").getPtr<T>(),
                            input_tensors->at("input_lengths").getPtr<int>(),
                            input_tensors->at("compact_idx").getPtr<int>(),
                            batch_size,
                            seq_len,
                            hidden_units,
                            stream_);
    }
    POP_RANGE;

    const size_t local_batch_size = getLocalBatchSize(batch_size, seq_len, pipeline_para_.world_size_);
    FT_CHECK(batch_size % local_batch_size == 0);
    const size_t iteration_num = batch_size / local_batch_size;

    Tensor k_cache = output_tensors->at("key_cache");
    Tensor v_cache = output_tensors->at("value_cache");

    const auto activation_in_type  = gpt_init_parameter_.int8_mode_ == 2 ? TYPE_INT8 : data_type;
    const auto activation_out_type = data_type;

    // The resize of the key cache buffer by
    //   (local_batch_size, local_head_num, size_per_head // x, max_seq_len, x) where x is constant.
    std::vector<size_t> self_k_cache_size(k_cache.shape().begin() + 2, k_cache.shape().end());
    self_k_cache_size.insert(self_k_cache_size.begin(), local_batch_size);

    // The resize of the value cache buffer by
    //   (local_batch_size, local_head_num, max_seq_len, size_per_head).
    std::vector<size_t> self_v_cache_size(v_cache.shape().begin() + 2, v_cache.shape().end());
    self_v_cache_size.insert(self_v_cache_size.begin(), local_batch_size);

    if (use_shared_contexts) {
        // we use k_cache_layer_ and v_cache_layer_
        self_k_cache_size[3] = seq_len;
        self_v_cache_size[2] = seq_len;
    }

    AttentionType attention_type =
        (input_tensors->isExist("linear_bias_slopes") || gpt_init_parameter_.int8_mode_ == 2) ?
            getUnfusedAttentionType(attention_type_) :
            attention_type_;
    const bool is_unpadded_mha = isUnPaddedMHA(attention_type);

    PUSH_RANGE(stream_, "context_generation");
    for (uint ite = 0; ite < iteration_num; ite++) {
        size_t h_token_num = local_batch_size * seq_len;

        if (is_unpadded_mha) {
            const int* base_input_lengths =
                (use_shared_contexts ? compact_input_lengths_ : input_tensors->at("input_lengths").getPtr<int>());
            invokeGetPaddingOffsetAndCuSeqLens(h_pinned_token_num_ptr_,
                                               &h_token_num,
                                               padding_offset_,
                                               cu_seqlens_,
                                               base_input_lengths + ite * local_batch_size,
                                               local_batch_size,
                                               seq_len,
                                               stream_);
        }

        for (uint l = 0; l < gpt_init_parameter_.num_layers_; l++) {
            PUSH_RANGE(stream_, fmtstr("layer_%u", l));
            bool use_moe = std::find(gpt_init_parameter_.moe_layer_index_.begin(), gpt_init_parameter_.moe_layer_index_.end(), l) != gpt_init_parameter_.moe_layer_index_.end();
            if (isValidLayerParallelId(l) == false) {
                continue;
            }

            if (l == 0 && is_unpadded_mha) {
                PUSH_RANGE(stream_, "remove_padding");
                const T* base_input =
                    (use_shared_contexts ? compact_decoder_features_ : decoder_input_tensor.getPtr<T>());

                invokeRemovePadding(decoder_layer_output_,
                                    base_input + ite * local_batch_size * seq_len * hidden_units,
                                    padding_offset_,
                                    h_token_num,
                                    hidden_units,
                                    stream_);
                POP_RANGE;
            }

            ParallelGptDecoderLayerWeight<T>* layer_weight = gpt_decoder_layer_weight->at(l);

            T* decoder_input  = decoder_layer_output_;
            T* decoder_output = decoder_layer_output_;
            if (!is_unpadded_mha) {
                if (l == 0) {
                    decoder_input = use_shared_contexts ? compact_decoder_features_ : decoder_input_tensor.getPtr<T>();
                    decoder_input += ite * local_batch_size * seq_len * hidden_units;
                }
                if (l == gpt_init_parameter_.num_layers_ - 1) {
                    decoder_output = use_shared_contexts ? compact_decoder_features_ :
                                                           output_tensors->at("decoder_output").getPtr<T>();
                    decoder_output += ite * local_batch_size * seq_len * hidden_units;
                }
            }

            if (isFirstLayerParallelId(l) && pipeline_para_.rank_ != 0) {
                PUSH_RANGE(stream_, "input communication");
                const int data_size = h_token_num * hidden_units / tensor_para_.world_size_;
                ftNcclRecv(decoder_input + data_size * tensor_para_.rank_,
                           data_size,
                           pipeline_para_.rank_ - 1,
                           pipeline_para_,
                           stream_);
                if (tensor_para_.world_size_ > 1) {
                    PUSH_RANGE(stream_, "all gather");
                    ftNcclAllGather(decoder_input, decoder_input, data_size, tensor_para_.rank_, tensor_para_, stream_);
                    POP_RANGE;
                }
                POP_RANGE;
            }

            print_bsd(l, "decoder input", decoder_input, request_batch_size, seq_len, hidden_units);

            PUSH_RANGE(stream_, "pre_mha_ln");
            norm_wrapper_->initDecoderLayerNorm(decoder_normed_input_,
                                                decoder_input,
                                                layer_weight->pre_layernorm_weights.gamma,
                                                layer_weight->pre_layernorm_weights.beta,
                                                gpt_init_parameter_.layernorm_eps_,
                                                h_token_num,
                                                hidden_units,
                                                const_cast<float*>(layer_weight->self_attention_weights.query_weight.scale),
                                                nullptr,
                                                gpt_init_parameter_.int8_mode_,
                                                stream_);
            if (gpt_init_parameter_.layernorm_type_ == LayerNormType::pre_layernorm) {
                print_bsd(l, "pre ln", decoder_normed_input_, request_batch_size, seq_len, hidden_units);
            }

            if (pre_attn_ln) {
                    norm_wrapper_->preAttentionLayerNorm(attn_normed_input_,
                                                         decoder_input,
                                                         layer_weight->pre_attn_layernorm_weights.gamma,
                                                         layer_weight->pre_attn_layernorm_weights.beta,
                                                         gpt_init_parameter_.layernorm_eps_,
                                                         h_token_num,
                                                         hidden_units,
                                                         nullptr,
                                                         nullptr,
                                                         gpt_init_parameter_.int8_mode_,
                                                         stream_);
                    print_bsd(l, "pre attn ln", attn_normed_input_, request_batch_size, seq_len, hidden_units);
            }

            sync_check_cuda_error();
            POP_RANGE;

            const bool is_final = false;  // TODO(bhsueh) remove this flag

            const T* attention_ptr =
                use_shared_contexts ? compact_attention_mask_ : input_tensors->at("attention_mask").getPtr<T>();

            const T* input_query = nullptr;
            if (pre_attn_ln) {
                input_query = attn_normed_input_;
            } else if (gpt_init_parameter_.layernorm_type_ == LayerNormType::pre_layernorm) {
                input_query = decoder_normed_input_;
            } else {
                input_query = decoder_input;
            }
            TensorMap self_attention_input_tensors{
                {"input_query",
                 Tensor{MEMORY_GPU,
                        activation_in_type,
                        {h_token_num, hidden_units}, input_query}},
                {"attention_mask",
                 Tensor{MEMORY_GPU,
                        data_type,
                        {local_batch_size, 1, seq_len, (size_t)(seq_len + max_prefix_prompt_length)},
                        attention_ptr + local_batch_size * ite * (seq_len * seq_len + max_prefix_prompt_length)}},
                {"attention_type", Tensor{MEMORY_CPU, TYPE_VOID, {1}, &attention_type}},
                {"is_final_layer", Tensor{MEMORY_CPU, TYPE_BOOL, {1}, &is_final}},
                {"input_lengths", Tensor{MEMORY_GPU, TYPE_INT32, {batch_size}, input_tensors->at("input_lengths").getPtr<int>()}},
                {"layer_id", Tensor{MEMORY_CPU, TYPE_INT32, {(size_t)1}, &l}},
		        {"lora_ids", input_tensors->at("lora_ids")},
                {"lora_input_lengths", input_tensors->at("lora_input_lengths")}};

            if (is_unpadded_mha) {
                self_attention_input_tensors.insert("padding_offset",
                                                    Tensor{MEMORY_GPU, TYPE_INT32, {h_token_num}, padding_offset_});
                self_attention_input_tensors.insert(
                    "cu_seqlens", Tensor{MEMORY_GPU, TYPE_INT32, {size_t(local_batch_size + 1)}, cu_seqlens_});
            }

            /* if (dynamic_quant_) { */
            /*     self_attention_input_tensors.insert("attention_query_dynamic_scale", */
            /*         Tensor{MEMORY_GPU, TYPE_FP32, {h_token_num}, attention_query_dynamic_scale_}); */
            /* } */

            if (input_tensors->isExist("linear_bias_slopes")) {
                self_attention_input_tensors.insert("linear_bias_slopes", input_tensors->at("linear_bias_slopes"));
            }
            if (input_tensors->isExist("d_prefix_prompt_batch") && input_tensors->isExist("d_prefix_prompt_lengths")) {
                const T**  d_prefix_prompt_batch   = input_tensors->at("d_prefix_prompt_batch").getPtr<const T*>();
                const int* d_prefix_prompt_lengths = input_tensors->at("d_prefix_prompt_lengths").getPtr<const int>();
                self_attention_input_tensors.insert(
                    "d_prefix_prompt_batch", Tensor{MEMORY_GPU, data_type, {local_batch_size}, d_prefix_prompt_batch}
                );
                self_attention_input_tensors.insert(
                    "d_prefix_prompt_lengths", Tensor{MEMORY_GPU, TYPE_INT32, {local_batch_size}, d_prefix_prompt_lengths}
                );
            }

            // The key/value cache stride per batch.
            // auto local_head_num_kv = gpt_init_parameter_.head_num_kv_;
            // if (gpt_init_parameter_.head_num_kv_ > 1) {
            //     local_head_num_kv = gpt_init_parameter_.head_num_kv_ / tensor_para_.world_size_;
            // }
            size_t cache_layer_offset = l - getFirstLayerParallelId();
            for (auto t = k_cache.shape().begin() + 1; t != k_cache.shape().end(); ++t) {
                cache_layer_offset *= *t;
            };
            size_t ite_cache_offset = ite * local_batch_size;
            for (auto t = k_cache.shape().begin() + 2; t != k_cache.shape().end(); ++t) {
                ite_cache_offset *= *t;
            }
            size_t cache_offset = cache_layer_offset + ite_cache_offset;


            T* k_cache_ptr = use_shared_contexts ? k_cache_layer_ : k_cache.getPtrWithOffset<T>(cache_offset);
            T* v_cache_ptr = use_shared_contexts ? v_cache_layer_ : v_cache.getPtrWithOffset<T>(cache_offset);

            TensorMap self_attention_output_tensors{
                {"hidden_features",
                 Tensor{MEMORY_GPU, activation_out_type, {h_token_num, hidden_units}, self_attn_output_}},
                {"key_cache", Tensor{MEMORY_GPU, data_type, self_k_cache_size, k_cache_ptr}},
                {"value_cache", Tensor{MEMORY_GPU, data_type, self_v_cache_size, v_cache_ptr}}};

            if (gpt_init_parameter_.is_sparse_head_ && gpt_init_parameter_.layer_head_num_[l] == 0) {
                check_cuda_error(cudaMemcpyAsync(self_attn_output_,
                                                 input_query,
                                                 sizeof(T) * batch_size * seq_len * hidden_units,
                                                 cudaMemcpyDeviceToDevice,
                                                 stream_));
            }
            else {
                self_attention_layer_->forward(&self_attention_output_tensors,
                                               &self_attention_input_tensors,
                                               &layer_weight->self_attention_weights);
            }

            print_bsd(l, "attn out", self_attn_output_, request_batch_size, seq_len, hidden_units);

            PUSH_RANGE(stream_, "KV_cache_clean");
            if (use_shared_contexts) {
                // Even with local batches, we must process the whole K/V caches as any
                // element in batch_idx_to_compact_idx may reference the local batch
                // we're processing. We also need to discard references that aren't in
                // that particular local batch.
                invokeUnCompactCaches(k_cache.getPtrWithOffset<T>(cache_layer_offset),
                                      v_cache.getPtrWithOffset<T>(cache_layer_offset),
                                      k_cache_layer_,
                                      v_cache_layer_,
                                      input_tensors->at("batch_to_compact_idx").getPtr<int>(),
                                      request_batch_size,  // batch_size (uncompact)
                                      v_cache.shape()[2],    // local_head_num
                                      max_seq_len,
                                      seq_len,
                                      gpt_init_parameter_.size_per_head_,
                                      local_batch_size,
                                      ite,
                                      stream_);
                sync_check_cuda_error();
            }
            POP_RANGE;

            // the adapter after attention (only pre layernorm currently)
            PUSH_RANGE(stream_, "post_mha_ln");

            T *input_residual = nullptr;
            if (!layer_weight->self_attn_layernorm_weights.gamma) {
                // falcon7b
                // output = attn(norm1(in)) + mlp(norm1(in)) + in
                // falcon40b
                // output = attn(norm2(in)) + mlp(norm1(in)) + in
                input_residual = decoder_input;
                std::swap(normed_self_attn_output_, decoder_normed_input_);
            } else {
                norm_wrapper_->attentionAddBiasResidualLayerNorm(
                    self_attn_output_,
                    normed_self_attn_output_,
                    self_attn_output_,
                    gpt_init_parameter_.use_norm_input_residual_ ? decoder_normed_input_ : decoder_input,
                    (T*)nullptr,
                    layer_weight->self_attn_layernorm_weights.gamma,
                    layer_weight->self_attn_layernorm_weights.beta,
                    layer_weight->self_attention_weights.attention_output_weight.bias,
                    gpt_init_parameter_.layernorm_eps_,
                    h_token_num,
                    hidden_units,
                    nullptr,
                    nullptr,
                    const_cast<float*>(layer_weight->ffn_weights.intermediate_weight.scale),
                    nullptr,  // NOTE (perkzz): dynamic_quant_ ? ffn_intermediate_dynamic_scale_ : nullptr,
                    gpt_init_parameter_.int8_mode_,
                    stream_);
            }
            sync_check_cuda_error();
            POP_RANGE;

            T* ffn_output_ptr = decoder_normed_input_;

            print_bsd(l, "post_attn_ln", normed_self_attn_output_, request_batch_size, seq_len, hidden_units);

            TensorMap ffn_input_tensors(
                {{"ffn_input",
                  Tensor{MEMORY_GPU,
                         activation_in_type,
                         {h_token_num, hidden_units},
                         gpt_init_parameter_.layernorm_type_ == LayerNormType::pre_layernorm ? normed_self_attn_output_ :
                                                                           self_attn_output_}},
		        {"layer_id", Tensor{MEMORY_CPU, TYPE_INT32, {(size_t)1}, &l}},
		        {"lora_ids", input_tensors->at("lora_ids")},
                {"lora_input_lengths", input_tensors->at("lora_input_lengths")},
                {"input_lengths", Tensor{MEMORY_GPU, TYPE_INT32, {batch_size}, input_tensors->at("input_lengths").getPtr<int>()}},
                {"batch_size", Tensor{MEMORY_CPU, TYPE_INT32, {(size_t)1}, &local_batch_size}}});
            TensorMap ffn_output_tensors;
            if (!use_moe) {
                ffn_output_tensors.insert(
                    "ffn_output",
                    Tensor{MEMORY_GPU, activation_out_type, {h_token_num, hidden_units}, ffn_output_ptr});
            }

            ffn_layer_->forward(&ffn_output_tensors, &ffn_input_tensors, &layer_weight->ffn_weights);

            print_bsd(l, "ffn out decoder", ffn_output_ptr, request_batch_size, seq_len, hidden_units);

            // the adapter after ffn (only pre layernorm currently)
            PUSH_RANGE(stream_, "post_ffn_ln");

            if (!use_moe) {
                print_bsd(l, "ffn add res", self_attn_output_, request_batch_size, seq_len, hidden_units);
                norm_wrapper_->ffnAddBiasResidualLayerNorm(decoder_output,
                                                           gpt_init_parameter_.use_norm_attn_out_residual_ ? normed_self_attn_output_ : self_attn_output_,
                                                           ffn_output_ptr,
                                                           input_residual,
                                                           layer_weight->ffn_weights.output_weight.bias,
                                                           layer_weight->self_attn_layernorm_weights.gamma,
                                                           layer_weight->self_attn_layernorm_weights.beta,
                                                           gpt_init_parameter_.layernorm_eps_,
                                                           h_token_num,
                                                           hidden_units,
                                                           nullptr,
                                                           nullptr,
                                                           stream_);
            } else {
                abort();
            }

            print_bsd(l, "decoder out", decoder_output, request_batch_size, seq_len, hidden_units);

            sync_check_cuda_error();
            POP_RANGE;
            PUSH_RANGE(stream_, "Nccl send");
            if (isLastLayerParallelId(l) == true && (pipeline_para_.rank_ != pipeline_para_.world_size_ - 1)) {
                const int data_size = h_token_num * hidden_units / tensor_para_.world_size_;
                ftNcclSend(decoder_output + data_size * tensor_para_.rank_,
                           data_size,
                           pipeline_para_.rank_ + 1,
                           pipeline_para_,
                           stream_);
            }
            POP_RANGE;

            PUSH_RANGE(stream_, "Rebuild_padding");
            if ((l == gpt_init_parameter_.num_layers_ - 1) && is_unpadded_mha) {
                T* base_ptr =
                    use_shared_contexts ? compact_decoder_features_ : output_tensors->at("decoder_output").getPtr<T>();
                invokeRebuildPadding(base_ptr + ite * local_batch_size * seq_len * hidden_units,
                                     decoder_layer_output_,
                                     padding_offset_,
                                     h_token_num,
                                     gpt_init_parameter_.head_num_ * gpt_init_parameter_.size_per_head_,
                                     stream_);
            }
            POP_RANGE;

            POP_RANGE;
        }
    }
    POP_RANGE;

    PUSH_RANGE(stream_, "uncompact_outputs");
    if (use_shared_contexts) {
        invokeUnCompactOutputs(output_tensors->at("decoder_output").getPtr<T>(),
                               compact_decoder_features_,
                               input_tensors->at("batch_to_compact_idx").getPtr<int>(),
                               request_batch_size,  // batch
                               seq_len * hidden_units,
                               stream_);
    }
    POP_RANGE;

    PUSH_RANGE(stream_, "last_token_hidden_state_lookup");
    // TODO(bhsueh) We could optimize this point by only computing the last token for the last layer
    invokeLookupHiddenStateOfLastToken(output_tensors->at("last_token_hidden_units").getPtr<T>(),
                                       output_tensors->at("decoder_output").getPtr<T>(),
                                       input_tensors->at("input_lengths").getPtr<int>(),
                                       seq_len,
                                       request_batch_size,
                                       hidden_units,
                                       stream_);

    print_bsd(gpt_init_parameter_.num_layers_, "last token", output_tensors->at("last_token_hidden_units").getPtr<T>(), request_batch_size, 1, hidden_units);

    POP_RANGE;
    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
    FT_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
    final_check_error();
}

template class ParallelGptContextDecoder<float>;
template class ParallelGptContextDecoder<half>;
#ifdef ENABLE_BF16
template class ParallelGptContextDecoder<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
