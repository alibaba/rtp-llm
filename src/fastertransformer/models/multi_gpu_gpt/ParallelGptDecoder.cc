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

#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGptDecoder.h"
#include "src/fastertransformer/cuda/nvtx/nvtx_utils.h"

namespace fastertransformer {

template<typename T>
void ParallelGptDecoder<T>::initialize()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    quant_algo_ = tc::QuantAlgo(gpt_init_parameter_.int8_mode_, gpt_init_parameter_.int4_mode_, gpt_init_parameter_.has_pre_scale_, gpt_init_parameter_.has_zeros_, gpt_init_parameter_.weight_only_group_size_);
    self_attention_layer_.reset(new TensorParallelDecoderSelfAttentionLayer<T>(
                                        max_batch_size_,
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
                                        gpt_init_parameter_.head_num_ * gpt_init_parameter_.size_per_head_,
                                        gpt_init_parameter_.logn_seq_len_,
                                        1.0f,
                                        tensor_para_,
                                        stream_,
                                        cublas_wrapper_,
                                        allocator_,
                                        gpt_init_parameter_.use_logn_attn_,
                                        true,
                                        is_free_buffer_after_forward_,
                                        sparse_,
                                        gpt_init_parameter_.is_sparse_head_,
                                        gpt_init_parameter_.int8_mode_,
                                        gpt_init_parameter_.int4_mode_,
                                        custom_all_reduce_comm_,
                                        enable_custom_all_reduce_));

    ffn_layer_.reset(new TensorParallelFfnLayer<T>(
                             max_batch_size_,
                             1,
                             gpt_init_parameter_.head_num_,
                             gpt_init_parameter_.size_per_head_,
                             gpt_init_parameter_.expert_num_,  // expert_num
                             gpt_init_parameter_.moe_k_,
                             gpt_init_parameter_.inter_size_,
                             gpt_init_parameter_.inter_padding_size_,
                             gpt_init_parameter_.layer_inter_size_,
                             gpt_init_parameter_.layer_inter_padding_size_,
                             tensor_para_,
                             stream_,
                             cublas_wrapper_,
                             quant_algo_,
                             allocator_,
                             true,
                             is_free_buffer_after_forward_,
                             sparse_,
                             gpt_init_parameter_.is_sparse_head_,
                             gpt_init_parameter_.activation_type_,
                             gpt_init_parameter_.layernorm_eps_,
                             custom_all_reduce_comm_,
                             enable_custom_all_reduce_));

    norm_wrapper_.reset(new NormWrapper<T>(gpt_init_parameter_.layernorm_type_,
                                           gpt_init_parameter_.norm_type_,
                                           T(sqrt(2 * gpt_init_parameter_.num_layers_))));
}

template<typename T>
ParallelGptDecoder<T>::ParallelGptDecoder(size_t                              max_batch_size,
                                          const GptInitParameter&             gpt_init_parameter,
                                          NcclParam                           tensor_para,
                                          NcclParam                           pipeline_para,
                                          cudaStream_t                        stream,
                                          cublasMMWrapper*                    cublas_wrapper,
                                          IAllocator*                         allocator,
                                          bool                                is_free_buffer_after_forward,
                                          bool                                sparse,
                                          std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm,
                                          int                                 enable_custom_all_reduce):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, sparse),
    max_batch_size_(max_batch_size),
    gpt_init_parameter_(gpt_init_parameter),
    tensor_para_(tensor_para),
    pipeline_para_(pipeline_para),
    custom_all_reduce_comm_(custom_all_reduce_comm),
    enable_custom_all_reduce_(enable_custom_all_reduce)
{
    initialize();
}

template<typename T>
void ParallelGptDecoder<T>::allocateBuffer()
{
    FT_CHECK(false);
}

template<typename T>
void ParallelGptDecoder<T>::allocateBuffer(size_t batch_size, bool reuse_buf, bool pre_attn_ln)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    size_t hidden_units = gpt_init_parameter_.hidden_size_;
    decoder_layer_output_ = reinterpret_cast<T*>(
        allocator_->reMalloc(decoder_layer_output_, sizeof(T) * batch_size * hidden_units, false));
    decoder_normed_input_ = reinterpret_cast<T*>(
        allocator_->reMalloc(decoder_normed_input_, sizeof(T) * batch_size * hidden_units, false));
    self_attn_output_ =
        reinterpret_cast<T*>(allocator_->reMalloc(self_attn_output_, sizeof(T) * batch_size * hidden_units, false));
    if (!reuse_buf) {
        normed_self_attn_output_ = reinterpret_cast<T*>(
                allocator_->reMalloc(normed_self_attn_output_, sizeof(T) * batch_size * hidden_units, false));
    } else {
        normed_self_attn_output_ = decoder_normed_input_;
    }
    if (pre_attn_ln) {
        attn_normed_input_ = reinterpret_cast<T*>(
                allocator_->reMalloc(attn_normed_input_, sizeof(T) * batch_size * hidden_units, false));
    }

    // for moe
    expert_scales_ = reinterpret_cast<T*>(
        allocator_->reMalloc(expert_scales_, sizeof(T) * pad_to_multiple_of_16(gpt_init_parameter_.moe_k_ * batch_size), false));
    expanded_source_row_to_expanded_dest_row_ = reinterpret_cast<int*>(allocator_->reMalloc(
        expanded_source_row_to_expanded_dest_row_, sizeof(int) * pad_to_multiple_of_16(gpt_init_parameter_.moe_k_ * batch_size), false));
    expert_for_source_row_                    = reinterpret_cast<int*>(
        allocator_->reMalloc(expert_for_source_row_, sizeof(int) * pad_to_multiple_of_16(gpt_init_parameter_.moe_k_ * batch_size), false));
    fc2_result_ = reinterpret_cast<T*>(allocator_->reMalloc(
        fc2_result_, sizeof(T) * pad_to_multiple_of_16(gpt_init_parameter_.moe_k_ * batch_size * hidden_units), false));

    is_allocate_buffer_ = true;
}

template<typename T>
void ParallelGptDecoder<T>::freeBuffer()
{
    if (is_allocate_buffer_) {
        FT_LOG_DEBUG(__PRETTY_FUNCTION__);
        allocator_->free((void**)(&decoder_layer_output_));
        allocator_->free((void**)(&decoder_normed_input_));
        allocator_->free((void**)(&self_attn_output_));
        allocator_->free((void**)(&normed_self_attn_output_));

        is_allocate_buffer_ = false;

        allocator_->free((void**)(&expert_scales_));
        allocator_->free((void**)(&expanded_source_row_to_expanded_dest_row_));
        allocator_->free((void**)(&expert_for_source_row_));
        allocator_->free((void**)(&fc2_result_));
    }
}

template<typename T>
bool ParallelGptDecoder<T>::isValidLayerParallelId(uint l)
{
    uint local_num_layer = (uint)(ceil(gpt_init_parameter_.num_layers_ * 1.0f / pipeline_para_.world_size_));
    return l < gpt_init_parameter_.num_layers_ && (l >= local_num_layer * pipeline_para_.rank_)
           && (l < local_num_layer * (pipeline_para_.rank_ + 1));
}

template<typename T>
bool ParallelGptDecoder<T>::isFirstLayerParallelId(uint l)
{
    uint local_num_layer = (uint)(ceil(gpt_init_parameter_.num_layers_ * 1.0f / pipeline_para_.world_size_));
    return l < gpt_init_parameter_.num_layers_ && (l == local_num_layer * pipeline_para_.rank_);
}

template<typename T>
bool ParallelGptDecoder<T>::isLastLayerParallelId(uint l)
{
    uint local_num_layer = (uint)(ceil(gpt_init_parameter_.num_layers_ * 1.0f / pipeline_para_.world_size_));
    return l < gpt_init_parameter_.num_layers_ && (l == local_num_layer * (pipeline_para_.rank_ + 1) - 1);
}

template<typename T>
int ParallelGptDecoder<T>::getFirstLayerParallelId()
{
    uint local_num_layer = (uint)(ceil(gpt_init_parameter_.num_layers_ * 1.0f / pipeline_para_.world_size_));
    return local_num_layer * pipeline_para_.rank_;
}

template<typename T>
ParallelGptDecoder<T>::~ParallelGptDecoder()
{
    freeBuffer();
}

template<typename T>
void ParallelGptDecoder<T>::forward(std::unordered_map<std::string, Tensor>*              output_tensors,
                                    const std::unordered_map<std::string, Tensor>*        input_tensors,
                                    const std::vector<ParallelGptDecoderLayerWeight<T>*>* gpt_decoder_layer_weight)
{
    // input tensors:
    //      decoder_input [local_batch_size, hidden_dimension],
    //      finished [local_batch_size],
    //      sequence_lengths [local_batch_size],
    //      input_lengths [local_batch_size]
    //      max_input_length [1] on cpu
    //      step [1] on cpu
    //      ite [1] on cpu
    //      cache_indirection [local_batch_size / beam_width, beam_width, memory_len]
    //          Here, local_batch_size contains the beam_width, so local_batch_size / beam_width
    //          is real local_batch_size. (optional.)
    //      masked_tokens [local_batch_size, memory_len]
    //      linear_bias_slopes [head_num], optional

    // output tensors:
    //      decoder_output [local_batch_size, hidden_dimension],
    //      key_cache [num_layer, batch_size, head_num, size_per_head // x, memory_len, x]
    //      value_cache [num_layer, batch_size, head_num, memory_len, size_per_head]

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    FT_CHECK(input_tensors->count("decoder_input"));
    FT_CHECK(input_tensors->count("finished"));
    FT_CHECK(input_tensors->count("sequence_lengths"));
    FT_CHECK(input_tensors->count("input_lengths"));
    FT_CHECK(input_tensors->count("lora_input_lengths"));
    FT_CHECK(input_tensors->count("max_input_length"));
    FT_CHECK(input_tensors->count("step"));
    FT_CHECK(input_tensors->count("ite"));
    FT_CHECK(input_tensors->count("lora_ids"));
    // FT_CHECK(input_tensors->count("masked_tokens"));
    FT_CHECK(output_tensors->count("decoder_output"));
    FT_CHECK(output_tensors->count("key_cache"));
    FT_CHECK(output_tensors->count("value_cache"));

    const size_t local_batch_size = input_tensors->at("decoder_input").shape()[0];
    const size_t hidden_units = gpt_init_parameter_.hidden_size_;
    bool reuse_buf = !gpt_init_parameter_.use_norm_input_residual_;
    bool pre_attn_ln = gpt_decoder_layer_weight->at(0)->pre_attn_layernorm_weights.gamma;
    allocateBuffer(local_batch_size, reuse_buf, pre_attn_ln);

    const DataType data_type = getTensorType<T>();

    const int ite = input_tensors->at("ite").getVal<int>();

    Tensor k_cache = output_tensors->at("key_cache");
    Tensor v_cache = output_tensors->at("value_cache");

    // The resize of the key cache buffer by
    //   (local_batch_size, local_head_num, size_per_head // x, max_seq_len, x) where x is constant.
    std::vector<size_t> self_k_cache_size(k_cache.shape().begin() + 2, k_cache.shape().end());
    self_k_cache_size.insert(self_k_cache_size.begin(), local_batch_size);

    // The resize of the value cache buffer by (local_batch_size, local_head_num, max_seq_len, size_per_head).
    std::vector<size_t> self_v_cache_size(v_cache.shape().begin() + 2, v_cache.shape().end());
    self_v_cache_size.insert(self_v_cache_size.begin(), local_batch_size);

    const auto activation_in_type  = gpt_init_parameter_.int8_mode_ == 2 ? TYPE_INT8 : data_type;
    const auto activation_out_type = data_type;

    int seq_len = 1;

    for (uint l = 0; l < gpt_init_parameter_.num_layers_; l++) {
        PUSH_RANGE(stream_, fmtstr("layer_%d", l));
        bool use_moe = std::find(gpt_init_parameter_.moe_layer_index_.begin(), gpt_init_parameter_.moe_layer_index_.end(), l) != gpt_init_parameter_.moe_layer_index_.end();
        if (isValidLayerParallelId(l) == false) {
            continue;
        }
        T* decoder_input = (l == 0) ? input_tensors->at("decoder_input").getPtr<T>() : decoder_layer_output_;
        T* decoder_output =
            (l == gpt_init_parameter_.num_layers_ - 1) ? output_tensors->at("decoder_output").getPtr<T>() : decoder_layer_output_;

        print_bsd(l, "decoder input", decoder_input, local_batch_size, seq_len, hidden_units);

        if (isFirstLayerParallelId(l) == true && pipeline_para_.rank_ != 0 && pipeline_para_.world_size_ > 1) {
            size_t data_size = local_batch_size * hidden_units / tensor_para_.world_size_;
            PUSH_RANGE(stream_, "input communication");
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

        PUSH_RANGE(stream_, "pre_mha_ln");
        ParallelGptDecoderLayerWeight<T>* layer_weight = gpt_decoder_layer_weight->at(l);

        norm_wrapper_->initDecoderLayerNorm(decoder_normed_input_,
                                                decoder_input,
                                                layer_weight->pre_layernorm_weights.gamma,
                                                layer_weight->pre_layernorm_weights.beta,
                                                gpt_init_parameter_.layernorm_eps_,
                                                local_batch_size,
                                                hidden_units,
                                                const_cast<float*>(layer_weight->self_attention_weights.query_weight.scale),
                                                nullptr,
                                                gpt_init_parameter_.int8_mode_,
                                                stream_);
        print_bsd(l, "pre ln", decoder_normed_input_, local_batch_size, seq_len, hidden_units);

        if (pre_attn_ln) {
            norm_wrapper_->preAttentionLayerNorm(attn_normed_input_,
                                                 decoder_input,
                                                 layer_weight->pre_attn_layernorm_weights.gamma,
                                                 layer_weight->pre_attn_layernorm_weights.beta,
                                                 gpt_init_parameter_.layernorm_eps_,
                                                 local_batch_size,
                                                 hidden_units,
                                                 nullptr,
                                                 nullptr,
                                                 gpt_init_parameter_.int8_mode_,
                                                 stream_);

            print_bsd(l, "pre attn ln", attn_normed_input_, local_batch_size, seq_len, hidden_units);
        }
        sync_check_cuda_error();
        POP_RANGE;

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
                 {local_batch_size, hidden_units},
                 input_query}},
            {"finished", input_tensors->at("finished")},
            {"sequence_lengths", input_tensors->at("sequence_lengths")},
            {"input_lengths", input_tensors->at("input_lengths")},
            {"lora_input_lengths", input_tensors->at("lora_input_lengths")},
            {"max_input_length", input_tensors->at("max_input_length")},
            {"step", input_tensors->at("step")},
            {"layer_id", Tensor{MEMORY_CPU, TYPE_INT32, {(size_t)1}, &l}},
	        {"lora_ids", input_tensors->at("lora_ids")}};
        if (input_tensors->count("masked_tokens")) {
            self_attention_input_tensors.insert("masked_tokens", input_tensors->at("masked_tokens"));
        }
        if (input_tensors->count("cache_indirection")) {
            self_attention_input_tensors.insert("cache_indirection", input_tensors->at("cache_indirection"));
        }
        if (input_tensors->count("linear_bias_slopes")) {
            self_attention_input_tensors.insert("linear_bias_slopes", input_tensors->at("linear_bias_slopes"));
        }

        if (input_tensors->find("d_prefix_prompt_lengths") != input_tensors->end()) {
            self_attention_input_tensors.insert(
                    "d_prefix_prompt_lengths", input_tensors->at("d_prefix_prompt_lengths")
            );
            self_attention_input_tensors.insert(
                    "max_prefix_prompt_length", input_tensors->at("max_prefix_prompt_length")
            );
        }

        size_t cache_offset = l - getFirstLayerParallelId();
        for (auto t = k_cache.shape().begin() + 1; t != k_cache.shape().end(); ++t) {
            cache_offset *= *t;
        };
        size_t ite_cache_offset = ite * local_batch_size;
        for (auto t = k_cache.shape().begin() + 2; t != k_cache.shape().end(); ++t) {
            ite_cache_offset *= *t;
        }
        cache_offset += ite_cache_offset;

        TensorMap self_attention_output_tensors{
            {"hidden_features",
             Tensor(MEMORY_GPU, activation_out_type, {local_batch_size, hidden_units}, self_attn_output_)},
            {"key_cache", Tensor(MEMORY_GPU, data_type, self_k_cache_size, k_cache.getPtrWithOffset<T>(cache_offset))},
            {"value_cache",
             Tensor(MEMORY_GPU, data_type, self_v_cache_size, v_cache.getPtrWithOffset<T>(cache_offset))}};

        if (gpt_init_parameter_.is_sparse_head_ && gpt_init_parameter_.layer_head_num_[l] == 0) {
            check_cuda_error(cudaMemcpyAsync(self_attn_output_,
                                             input_query,
                                             sizeof(T) * local_batch_size * hidden_units,
                                             cudaMemcpyDeviceToDevice,
                                             stream_));
        }
        else {
            self_attention_layer_->forward(
                &self_attention_output_tensors, &self_attention_input_tensors, &layer_weight->self_attention_weights);
        }

        print_bsd(l, "attn out", self_attn_output_, local_batch_size, seq_len, hidden_units);

        // the adapter after attention
        PUSH_RANGE(stream_, "post_mha_ln");

        T *input_residual = nullptr;
        if (!layer_weight->self_attn_layernorm_weights.gamma) {
            // falcon
            input_residual = decoder_input;
            std::swap(normed_self_attn_output_, decoder_normed_input_);
        } else {
            norm_wrapper_->attentionAddBiasResidualLayerNorm(
                    self_attn_output_,
                    normed_self_attn_output_,
                    self_attn_output_,
                    gpt_init_parameter_.use_norm_input_residual_ ? decoder_normed_input_ : decoder_input,
                    layer_weight->self_attn_layernorm_weights.gamma,
                    layer_weight->self_attn_layernorm_weights.beta,
                    layer_weight->self_attention_weights.attention_output_weight.bias,
                    gpt_init_parameter_.layernorm_eps_,
                    local_batch_size,
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

        print_bsd(l, "post_attn_ln", normed_self_attn_output_, local_batch_size, seq_len, hidden_units);

        T* ffn_output_ptr = decoder_normed_input_;
        TensorMap ffn_input_tensors(
            {{"ffn_input",
              Tensor{MEMORY_GPU,
                     activation_in_type,
                     {local_batch_size, hidden_units},
                     gpt_init_parameter_.layernorm_type_ == LayerNormType::pre_layernorm ? normed_self_attn_output_ :
                                                                       self_attn_output_}},
	        {"layer_id", Tensor{MEMORY_CPU, TYPE_INT32, {(size_t)1}, &l}},
	        {"lora_ids", input_tensors->at("lora_ids")},
            {"lora_input_lengths", input_tensors->at("lora_input_lengths")},
            {"batch_size", Tensor{MEMORY_CPU, TYPE_INT32, {(size_t)1}, &local_batch_size}}});
        TensorMap ffn_output_tensors;
        if (!use_moe) {
            ffn_output_tensors.insert(
                    "ffn_output",
                    Tensor{MEMORY_GPU, activation_out_type, {local_batch_size, hidden_units}, ffn_output_ptr});
        }

        ffn_layer_->forward(&ffn_output_tensors, &ffn_input_tensors, &layer_weight->ffn_weights);

        print_bsd(l, "ffn out decoder", ffn_output_ptr, local_batch_size, seq_len, hidden_units);

        // the adapter after ffn
        PUSH_RANGE(stream_, "post_ffn_ln");

        if (!use_moe) {
            norm_wrapper_->ffnAddBiasResidualLayerNorm(decoder_output,
                                                           gpt_init_parameter_.use_norm_attn_out_residual_ ? normed_self_attn_output_ : self_attn_output_,
                                                           ffn_output_ptr,
                                                           input_residual,
                                                           layer_weight->ffn_weights.output_weight.bias,
                                                           layer_weight->self_attn_layernorm_weights.gamma,
                                                           layer_weight->self_attn_layernorm_weights.beta,
                                                           gpt_init_parameter_.layernorm_eps_,
                                                           local_batch_size,
                                                           hidden_units,
                                                           nullptr,
                                                           nullptr,
                                                           stream_);
        }

        print_bsd(l, "decoder out", decoder_output, local_batch_size, seq_len, hidden_units);

        sync_check_cuda_error();
        POP_RANGE;

        PUSH_RANGE(stream_, "Nccl send");
        if (isLastLayerParallelId(l) == true && pipeline_para_.rank_ != pipeline_para_.world_size_ - 1
            && pipeline_para_.world_size_ > 1) {

            ftNcclSend(decoder_output
                       + local_batch_size * hidden_units / tensor_para_.world_size_ * tensor_para_.rank_,
                       local_batch_size * hidden_units / tensor_para_.world_size_,
                       pipeline_para_.rank_ + 1,
                       pipeline_para_,
                       stream_);
        }
        POP_RANGE;
        POP_RANGE;
    }

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
    final_check_error();
}

template class ParallelGptDecoder<float>;
template class ParallelGptDecoder<half>;
#ifdef ENABLE_BF16
template class ParallelGptDecoder<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
