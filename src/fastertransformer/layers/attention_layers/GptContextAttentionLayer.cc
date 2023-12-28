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

#include "src/fastertransformer/layers/attention_layers/GptContextAttentionLayer.h"
#include "src/fastertransformer/kernels/unfused_attention_kernels.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/utils/nvtx_utils.h"
#include "src/fastertransformer/kernels/kv_cache_utils.h"

namespace fastertransformer {

template<typename T>
void GptContextAttentionLayer<T>::preAllocate()
{
    if (max_batch_size_ > 0) {
        allocateBuffer(max_batch_size_, max_seq_len_, max_seq_len_, true);
    }
}

template<typename T>
void GptContextAttentionLayer<T>::forward(TensorMap*                output_tensors,
                                          TensorMap*                input_tensors,
                                          const AttentionWeight<T>* attention_weights)
{
    // input_tensors:
    //      input_query [token_num, hidden_dimension]
    //      attention_mask [batch_size, 1, seq_len, seq_len + max_prompt_length]
    //      attention_type [1]
    //      is_final_layer [1], bool on cpu
    //      layer_id [1], int on cpu
    //      padding_offset, int, [token_num] (optional)
    //      cu_seqlens, int, [batch_size] (optional)
    //      d_prefix_prompt_batch [global_batch_size], (optional)
    //          each element contains ptr with buffer shape[2, local_head_num_, prompt_length, size_per_head]
    //      d_prefix_prompt_lengths [batch_size], int (optional)
    //      linear_bias_slopes [head_num] (optional)

    // output_tensors:
    //      hidden_features [token_num, hidden_dimension]
    //      key_cache [batch, local_head_num, size_per_head // x, max_seq_len, x]
    //      value_cache [batch, local_head_num, max_seq_len, size_per_head]
    FT_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    FT_CHECK(output_tensors->at("key_cache").shape.size() == 4);
    FT_CHECK(output_tensors->at("value_cache").shape.size() == 4
             || output_tensors->at("value_cache").shape.size() == 3);

    const AttentionType attention_type = input_tensors->getVal<AttentionType>("attention_type");
    FT_CHECK_WITH_INFO(attention_type != AttentionType::FUSED_PADDED_MHA,
                       "Gpt Context FUSED_PADDED_MHA is not supported !");

    GptContextAttentionLayer<T>::Attention(output_tensors, input_tensors, attention_weights);

    FT_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

template<typename T>
void GptContextAttentionLayer<T>::Attention(TensorMap*                output_tensors,
                                            TensorMap*                input_tensors,
                                            const AttentionWeight<T>* attention_weights)
{
    // FT_CHECK(local_head_num_kv_ == 1);
    const int request_batch_size = input_tensors->at("attention_mask").shape[0];
    const int request_seq_len    = input_tensors->at("attention_mask").shape[2];
    const int* input_lengths     = input_tensors->getPtr<int>("input_lengths");
    const int* lora_input_lengths = input_tensors->getPtr<int>("lora_input_lengths", nullptr);
    const int max_prompt_length =
        input_tensors->at("attention_mask").shape[3] - input_tensors->at("attention_mask").shape[2];
    const int  layer_id                = input_tensors->getVal<int>("layer_id");
    const T**  d_prefix_prompt_batch   = input_tensors->getPtr<const T*>("d_prefix_prompt_batch", nullptr);
    const int* d_prefix_prompt_lengths = input_tensors->getPtr<int>("d_prefix_prompt_lengths", nullptr);
    const int* padding_offset          = input_tensors->getPtr<int>("padding_offset", nullptr);
    int*       cu_seqlens              = input_tensors->getPtr<int>("cu_seqlens", nullptr);
    T*         linear_bias_slopes      = input_tensors->getPtr<T>("linear_bias_slopes", nullptr);
    const int* block_index_map         = input_tensors->getPtr<int>("block_index_map", nullptr);

    const int* repeat_prompt_block_map = input_tensors->getPtr<int>("prefix_block_map", nullptr);
    const bool count_prefix_length     = input_tensors->getVal<bool>("count_prefix_length", false);
    const int  min_prefix_length       = input_tensors->getVal<int>("min_prefix_length", 0);

    T* attention_out   = output_tensors->at("hidden_features").getPtr<T>();
    T* attention_input = input_tensors->at("input_query").getPtr<T>();
    T* attention_mask  = input_tensors->at("attention_mask").getPtr<T>();

    const AttentionType attention_type = input_tensors->getVal<AttentionType>("attention_type");

    PUSH_RANGE(stream_, "attention_buffer_alloc");
    allocateBuffer(request_batch_size, request_seq_len, request_seq_len + max_prompt_length, attention_type != AttentionType::FUSED_MHA);
    POP_RANGE;
    sync_check_cuda_error();

    const bool is_final = input_tensors->at("is_final_layer").getVal<bool>();

    const int m = input_tensors->at("input_query").shape[0];

    const int local_hidden_units_rt = (is_sparse_head_ ? local_layer_head_num_[layer_id]: local_head_num_) * size_per_head_;
    const int local_hidden_units_kv_rt = (is_sparse_head_ ? local_layer_head_num_kv_[layer_id]: local_head_num_kv_) * size_per_head_;
    const int local_head_num = is_sparse_head_? local_layer_head_num_[layer_id] : local_head_num_;
    const int local_head_num_kv = is_sparse_head_? local_layer_head_num_kv_[layer_id] : local_head_num_kv_;

    // lora
    int* lora_ids = input_tensors->getPtr<int>("lora_ids", nullptr);


    PUSH_RANGE(stream_, "qkv_gemm");
    
#ifdef SPARSITY_ENABLED
    const int m_padded   = 8 * div_up(m, 8);
    bool      use_sparse = sparse_ && cublas_wrapper_->isUseSparse(1, local_hidden_units_rt + 2 * local_hidden_units_kv_rt, m_padded, hidden_units_);
#else
    constexpr bool use_sparse = false;
    const int m_padded   = 0;
#endif

    // QKV gemm: [m, hidden_dim] * [hidden_dim, qkv_dim] = [m, qkv_dim]
    gemm_runner_->Gemm( request_batch_size,
                        lora_input_lengths,
                        m,
                        local_hidden_units_rt + 2 * local_hidden_units_kv_rt,
                        hidden_units_,
                        attention_input,
                        &attention_weights->query_weight,
                        qkv_buf_,
                        lora_ids,
                        int8_mode_,
                        use_sparse,
                        mixed_gemm_workspace_,
                        mixed_gemm_ws_bytes_,
                        m_padded);
    POP_RANGE;

    int k_start = local_hidden_units_rt;
    int v_start = local_hidden_units_rt + local_hidden_units_kv_rt;

    print_bsd(layer_id, "q", qkv_buf_, request_batch_size, request_seq_len, local_hidden_units_rt + 2 * local_hidden_units_kv_rt,
              0, 4);
    print_bsd(layer_id, "k", qkv_buf_, request_batch_size, request_seq_len, local_hidden_units_rt + 2 * local_hidden_units_kv_rt,
              k_start, k_start + 4);
    print_bsd(layer_id, "v", qkv_buf_, request_batch_size, request_seq_len, local_hidden_units_rt + 2 * local_hidden_units_kv_rt,
              v_start, v_start + 4);
    const int max_seq_len = (int)(output_tensors->at("key_cache").shape[2]);  // max output seq length
    int max_blocks_per_batch = 0;
    if (block_index_map) {
        max_blocks_per_batch = (int)(input_tensors->at("block_index_map").shape[1]);
    }

    sync_check_cuda_error();
    PrefixPromptBatchWeightsParam<T>* param = new PrefixPromptBatchWeightsParam<T>();
    if (repeat_prompt_block_map && d_prefix_prompt_batch) {
        throw std::runtime_error("not support both prefix_prompt and repeat_prompt");
    }
    else if (d_prefix_prompt_batch) {
        param = new PrefixPromptBatchWeightsParam<T>{
            d_prefix_prompt_lengths,
            max_prompt_length,
            count_prefix_length,
            KVBlockArray(0,0,0,0),
            ContinuousCacheParam<T>{d_prefix_prompt_batch, (size_t)(layer_id * 2 * local_head_num_kv_ * size_per_head_)}
        };
    }

    PUSH_RANGE(stream_, "qkv_bias_add");
    if (padding_offset != nullptr) {
        // q_buf_2_, k_buf_2_ and v_buf_2_ are continuous
        cudaMemsetAsync(
            q_buf_2_, 0, request_batch_size * (request_seq_len + max_prompt_length) * (local_hidden_units_rt + 2 * local_hidden_units_kv_rt) * sizeof(T), stream_);
    }   
    invokeAddFusedQKVBiasTranspose(q_buf_2_,
                                   k_buf_2_,
                                   v_buf_2_,
                                   *param,  // prefix prompt
                                   qkv_buf_,
                                   nullptr, // position_ids
                                   attention_weights->query_weight.bias,
                                   padding_offset,
                                   cu_seqlens,
                                   request_batch_size,
                                   request_seq_len,
                                   m,
                                   local_head_num,
                                   local_head_num_kv,
                                   size_per_head_,
                                   rotary_embedding_dim_,
                                   rotary_embedding_style_,
                                   rotary_embedding_base_,
                                   dynamic_embedding_scalar_,
                                   dynamic_embedding_max_pos_,
                                   position_embeddings_scale_,
                                   base_scale_,
                                   logn_seq_len_,
                                   use_logn_attn_,
                                   attention_weights->query_weight.scale_out,
                                   int8_mode_,
                                   stream_);
    POP_RANGE;

    print_bhsd(layer_id, "q bias rotary", q_buf_2_, request_batch_size, local_head_num, request_seq_len, size_per_head_);
    print_bhsd(layer_id, "k bias rotary", k_buf_2_, request_batch_size, local_head_num_kv, request_seq_len + max_prompt_length, size_per_head_);
    print_bhsd(layer_id, "v bias rotary", v_buf_2_, request_batch_size, local_head_num_kv, request_seq_len + max_prompt_length, size_per_head_);

    sync_check_cuda_error();
    KVLinearBuffer kv_cache_buffer(request_batch_size, 1, output_tensors->at("key_cache").shape[2], local_head_num_kv_ * size_per_head_ * sizeof(T));
    kv_cache_buffer.k_data = reinterpret_cast<int8_t*>(output_tensors->getPtr<T>("key_cache"));
    kv_cache_buffer.v_data = reinterpret_cast<int8_t*>(output_tensors->getPtr<T>("value_cache"));
    const KvCacheDataType cache_type = KvCacheDataType::BASE;
    delete param;
    // Use batch major
    // put k/v_buf from shape [B, H, PL + L, Dh]
    // to cache [B, H, Dh/x, PL + L, x]  and [B, H, PL + L, Dh/x, x], PL denotes prompt length
    // length_base means some blocks is reused in kvcache and not need to copy
    PUSH_RANGE(stream_, "kv_cache");
    invokeTranspose4dBatchMajor(k_buf_2_,
                                v_buf_2_,
                                kv_cache_buffer,
                                request_batch_size,
                                max_prompt_length + request_seq_len,  // max input length + prefix prompt length
                                size_per_head_,
                                local_head_num_kv,
                                cache_type,
                                nullptr, // kvScaleOrigQuant
                                input_lengths,
                                d_prefix_prompt_lengths,
                                stream_);
    // IDEA : after this, k_cache = (batch_size, num_heads, Dh/x, prefix_prompt_len + L, x)
    // k_cache = (batch_size, num_heads, prefix_prompt_len + L, Dh)
    sync_check_cuda_error();

    // key cache size is from gpt context decoder op
    // print_kv_cache(layer_id, "key_cache", output_tensors->getPtr<T>("key_cache"), 1, 1, 8, 16, 34, 8);

    // TODO: fmha kernels doesn't support different seq lengths of q and kv
    if (attention_type == AttentionType::FUSED_MHA) {
        throw std::runtime_error("not support fused mha");
    }
    // NOTE: qkv buffer shape (batch_size, num_heads,L or prompt_len + L, Dh)

    POP_RANGE;
    const cudaDataType_t gemm_data_type      = getCudaDataType<T>();
    const int            attention_seq_len_1 = request_seq_len;                      // q length
    const int            attention_seq_len_2 = max_prompt_length + request_seq_len;  // kv length
    const T              qk_scale            = static_cast<T>(1.0f / sqrtf(size_per_head_ * 1.0f));
    if (attention_type != AttentionType::FUSED_MHA) {
        if (is_qk_buf_float_ == true && gemm_data_type != CUDA_R_32F) {
            PUSH_RANGE(stream_, "Q*K");
            cublas_wrapper_->stridedBatchedGemm(
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                attention_seq_len_2,                                          // m
                attention_seq_len_1 * (local_head_num / local_head_num_kv),                        // n
                size_per_head_,                                               // k
                1.0f,
                k_buf_2_,                                                     // A
                gemm_data_type,                                               // Atype
                size_per_head_,                                               // lda             k
                attention_seq_len_2 * size_per_head_,                         // strideA n * k
                q_buf_2_,                                                     // B
                gemm_data_type,                                               // Btype
                size_per_head_,                                               // ldb                  // k
                attention_seq_len_1 * size_per_head_ * (local_head_num / local_head_num_kv),       // strideB m * k
                0.0f,
                qk_buf_float_,                                                // C
                CUDA_R_32F,                                                   // Ctype
                attention_seq_len_2,                                          // ldc  n
                attention_seq_len_2 * attention_seq_len_1 * (local_head_num / local_head_num_kv),  // strideC
                request_batch_size * local_head_num_kv,                                           // global batch size
                CUDA_R_32F);

            sync_check_cuda_error();
            POP_RANGE;

            int max_seq_len = min(attention_seq_len_1, 1024); // attention_seq_len_1 may be too big
            print_bhss(layer_id, "qk", qk_buf_float_, request_batch_size, local_head_num, max_seq_len, attention_seq_len_2);

            PUSH_RANGE(stream_, "softmax");
            MaskedSoftmaxParam<T, float> param;
            param.attention_score    = qk_buf_;         // (batch_size, head_num, q_length, k_length)
            param.qk                 = qk_buf_float_;   // (batch_size, head_num, q_length, k_length)
            param.attention_mask     = attention_mask;  // (batch_size, q_length, k_length)
            param.batch_size         = request_batch_size;
            param.q_length           = attention_seq_len_1;
            param.k_length           = attention_seq_len_2;
            param.num_heads          = local_head_num;
            param.qk_scale           = qk_scale;
            param.linear_bias_slopes = const_cast<T*>(linear_bias_slopes);  // (head_num,), optional
            invokeMaskedSoftmax(param, stream_);
            
            print_bhss(layer_id, "softmax", qk_buf_, request_batch_size, local_head_num, max_seq_len, attention_seq_len_2);
            sync_check_cuda_error();
            POP_RANGE;
        }

        else {
            PUSH_RANGE(stream_, "Q*K");
            cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_T,
                                                CUBLAS_OP_N,
                                                attention_seq_len_2,
                                                attention_seq_len_1 * (local_head_num / local_head_num_kv),
                                                size_per_head_,
                                                k_buf_2_,
                                                size_per_head_,
                                                attention_seq_len_2 * size_per_head_,
                                                q_buf_2_,
                                                size_per_head_,
                                                attention_seq_len_1 * size_per_head_ * (local_head_num / local_head_num_kv),
                                                qk_buf_,
                                                attention_seq_len_2,
                                                attention_seq_len_2 * attention_seq_len_1 * (local_head_num / local_head_num_kv),
                                                request_batch_size * local_head_num_kv);
            print_bhss(layer_id, "qk", qk_buf_, request_batch_size, local_head_num, attention_seq_len_1, attention_seq_len_2);
            POP_RANGE;

            PUSH_RANGE(stream_, "softmax");
            MaskedSoftmaxParam<T, T> param;
            param.attention_score    = qk_buf_;         // (batch_size, head_num, q_length, k_length)
            param.qk                 = qk_buf_;         // (batch_size, head_num, q_length, k_length)
            param.attention_mask     = attention_mask;  // (batch_size, q_length, k_length)
            param.batch_size         = request_batch_size;
            param.q_length           = attention_seq_len_1;
            param.k_length           = attention_seq_len_2;
            param.num_heads          = local_head_num;
            param.qk_scale           = qk_scale;
            param.linear_bias_slopes = const_cast<T*>(linear_bias_slopes);  // (head_num,), optional
            invokeMaskedSoftmax(param, stream_);
            print_bhss(layer_id, "softmax", qk_buf_, request_batch_size, local_head_num, attention_seq_len_1, attention_seq_len_2);
            sync_check_cuda_error();
            POP_RANGE;
        }

        PUSH_RANGE(stream_, "QK*V");

        cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_N,
                                            CUBLAS_OP_N,
                                            size_per_head_,
                                            attention_seq_len_1 * (local_head_num / local_head_num_kv),
                                            attention_seq_len_2,
                                            v_buf_2_,
                                            size_per_head_,
                                            attention_seq_len_2 * size_per_head_,
                                            qk_buf_,
                                            attention_seq_len_2,
                                            attention_seq_len_1 * (local_head_num / local_head_num_kv) * attention_seq_len_2,
                                            qkv_buf_,
                                            size_per_head_,
                                            attention_seq_len_1 * (local_head_num / local_head_num_kv) * size_per_head_,
                                            request_batch_size * local_head_num_kv);
        POP_RANGE;

        print_bhsd(layer_id, "qkv_weighted", qkv_buf_, request_batch_size, local_head_num, attention_seq_len_1, size_per_head_);

        PUSH_RANGE(stream_, "transpose");
        // transpose (batch_size, num_heads, L, Dh) to (batch_size, L, num_heads * Dh)
        if (padding_offset == nullptr) {
            invokeTransposeQKV(qkv_buf_2_,
                                qkv_buf_,
                                request_batch_size,
                                attention_seq_len_1,
                                local_head_num,
                                size_per_head_,
                                attention_weights->attention_output_weight.scale,
                                int8_mode_,
                                stream_);
            sync_check_cuda_error();
        }
        else {
            invokeTransposeAttentionOutRemovePadding(qkv_buf_,
                                                        qkv_buf_2_,
                                                        m,
                                                        request_batch_size,
                                                        attention_seq_len_1,
                                                        local_head_num,
                                                        size_per_head_,
                                                        padding_offset,
                                                        attention_weights->attention_output_weight.scale,
                                                        int8_mode_,
                                                        stream_);
        }
        POP_RANGE;
    }
    sync_check_cuda_error();

    print_bshd(layer_id, "qkv_weighted_t", qkv_buf_2_, 1, m, local_head_num, size_per_head_);

    PUSH_RANGE(stream_, "ln");
    float layernorm_eps = 1E-5;
    T* qkv_buf_3_input = nullptr;
    if (attention_weights->attention_layernorm.gamma && attention_weights->attention_layernorm.beta) {
        invokeGeneralLayerNorm(qkv_buf_,
                                qkv_buf_2_,
                                attention_weights->attention_layernorm.gamma,
                                attention_weights->attention_layernorm.beta,
                                layernorm_eps,
                                m,
                                local_hidden_units_rt,
                                nullptr,
                                nullptr,
                                int8_mode_,
                                stream_);
        qkv_buf_3_input = qkv_buf_;
        sync_check_cuda_error();
        print_bsd(layer_id, "attn ln", qkv_buf_, request_batch_size, request_seq_len, local_hidden_units_rt);
    } else {
        qkv_buf_3_input = qkv_buf_2_;
    }
    POP_RANGE;

    print_bsd(layer_id, "attn before o", qkv_buf_3_input, request_batch_size, request_seq_len, local_hidden_units_rt);

    PUSH_RANGE(stream_, "proj_gemm");

#ifdef SPARSITY_ENABLED
    bool use_sparse = sparse_ && cublas_wrapper_->isUseSparse(1, hidden_units_, m_padded, local_hidden_units_rt);
#endif

    // QKV gemm: [m, hidden_dim] * [hidden_dim, qkv_dim] = [m, qkv_dim]
    gemm_runner_->Gemm( request_batch_size,
                        lora_input_lengths,
                        m,
                        hidden_units_,
                        local_hidden_units_rt,
                        qkv_buf_3_input,
                        &attention_weights->attention_output_weight,
                        attention_out,
                        lora_ids,
                        int8_mode_,
                        use_sparse,
                        mixed_gemm_workspace_,
                        mixed_gemm_ws_bytes_,
                        m_padded);
    
    POP_RANGE;

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
    sync_check_cuda_error();

    print_bsd(layer_id, "attn output", attention_out, 1, m, hidden_units_);
}

template<typename T>
GptContextAttentionLayer<T>::GptContextAttentionLayer(size_t               max_batch_size,
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
                                                      bool                 sparse,
                                                      bool                 is_sparse_head,
                                                      int                  int8_mode):
    BaseAttentionLayer<T>(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, sparse),
    max_batch_size_(max_batch_size),
    max_seq_len_(max_seq_len),
    head_num_(head_num),
    head_num_kv_(head_num_kv),
    size_per_head_(size_per_head),
    hidden_units_(head_num * size_per_head),
    local_head_num_(local_head_num),
    local_head_num_kv_(local_head_num_kv),
    local_hidden_units_(local_head_num_ * size_per_head),
    local_layer_head_num_(local_layer_head_num),
    local_layer_head_num_kv_(local_layer_head_num_kv),
    rotary_embedding_dim_(rotary_embedding_dim),
    rotary_embedding_style_(rotary_embedding_style),
    rotary_embedding_base_(rotary_embedding_base),
    dynamic_embedding_scalar_(dynamic_embedding_scalar),
    dynamic_embedding_max_pos_(dynamic_embedding_max_pos),
    position_embeddings_scale_(position_embeddings_scale),
    base_scale_(base_scale),
    use_logn_attn_(use_logn_attn),
    logn_seq_len_(logn_seq_len),
    is_qk_buf_float_(is_qk_buf_float),
    weight_only_int8_fc_runner_(int8_mode == 1 ? std::make_shared<CutlassFpAIntBGemmRunner<T, uint8_t>>() : nullptr),
    gemm_runner_(std::make_shared<GemmRunner<T>>(sparse, stream, allocator, cublas_wrapper, weight_only_int8_fc_runner_)),
    is_sparse_head_(is_sparse_head),
    int8_mode_(int8_mode)
{
    if (int8_mode_ == 2) {
        abort();
    }
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
}

template<typename T>
GptContextAttentionLayer<T>::GptContextAttentionLayer(GptContextAttentionLayer<T> const& attention_layer):
    BaseAttentionLayer<T>(attention_layer.stream_,
                          attention_layer.cublas_wrapper_,
                          attention_layer.allocator_,
                          attention_layer.is_free_buffer_after_forward_,
                          attention_layer.sparse_),
    max_batch_size_(attention_layer.max_batch_size_),
    max_seq_len_(attention_layer.max_seq_len_),
    head_num_(attention_layer.head_num_),
    head_num_kv_(attention_layer.head_num_kv_),
    size_per_head_(attention_layer.size_per_head_),
    hidden_units_(attention_layer.hidden_units_),
    local_head_num_(attention_layer.local_head_num_),
    local_head_num_kv_(attention_layer.local_head_num_kv_),
    local_hidden_units_(attention_layer.local_hidden_units_),
    local_layer_head_num_(attention_layer.local_layer_head_num_),
    local_layer_head_num_kv_(attention_layer.local_layer_head_num_kv_),
    rotary_embedding_dim_(attention_layer.rotary_embedding_dim_),
    rotary_embedding_style_(attention_layer.rotary_embedding_style_),
    rotary_embedding_base_(attention_layer.rotary_embedding_base_),
    dynamic_embedding_scalar_(attention_layer.dynamic_embedding_scalar_),
    dynamic_embedding_max_pos_(attention_layer.dynamic_embedding_max_pos_),
    position_embeddings_scale_(attention_layer.position_embeddings_scale_),
    base_scale_(attention_layer.base_scale_),
    use_logn_attn_(attention_layer.use_logn_attn_),
    logn_seq_len_(attention_layer.logn_seq_len_),
    is_qk_buf_float_(attention_layer.is_qk_buf_float_),
    weight_only_int8_fc_runner_(attention_layer.weight_only_int8_fc_runner_),
    gemm_runner_(attention_layer.gemm_runner_),
    is_sparse_head_(attention_layer.is_sparse_head_),
    int8_mode_(attention_layer.int8_mode_)
{
}

template<typename T>
GptContextAttentionLayer<T>::~GptContextAttentionLayer()
{
    cublas_wrapper_ = nullptr;
    freeBuffer();
}

template<typename T>
void GptContextAttentionLayer<T>::allocateBuffer()
{
    FT_CHECK(false);
}

template<typename T>
void GptContextAttentionLayer<T>::allocateBuffer(size_t batch_size, size_t token_num, size_t seq_len, bool allocate_qk_buf)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    // const auto type_size = int8_mode_ == 2 ? sizeof(int8_t) : sizeof(T);
    // NOTE (perkzz): use sizeof(T) here for cutlass int8 kernels.
    qkv_buf_ = (T*)allocator_->reMalloc(qkv_buf_, sizeof(T) * batch_size * token_num * (local_hidden_units_ + 2 * local_head_num_kv_ * size_per_head_), true);
    q_buf_2_ = (T*)allocator_->reMalloc(q_buf_2_, sizeof(T) * batch_size * seq_len * (local_hidden_units_ + 2 * local_head_num_kv_ * size_per_head_), false);
    k_buf_2_ = q_buf_2_ + batch_size * seq_len * local_head_num_ * size_per_head_;
    v_buf_2_ = k_buf_2_ + batch_size * seq_len * local_head_num_kv_ * size_per_head_;

    // save memory usage when using fmha
    if (allocate_qk_buf) {
        qk_buf_ = (T*)allocator_->reMalloc(qk_buf_, sizeof(T) * batch_size * local_head_num_ * token_num * seq_len, true);
    }
    else {
        allocator_->free((void**)(&qk_buf_));
    }
    qkv_buf_2_ = (T*)allocator_->reMalloc(qkv_buf_2_, sizeof(T) * batch_size * token_num * local_hidden_units_, false);

    if (is_qk_buf_float_ == true) {
        if (allocate_qk_buf) {
            qk_buf_float_ = (float*)allocator_->reMalloc(
                qk_buf_float_, sizeof(float) * batch_size * local_head_num_ * token_num * seq_len, true);
        }
        else {
            allocator_->free((void**)(&qk_buf_float_));
        }
    }

    if (int8_mode_ == 1) {
        // We use max_size for n and k since we reuse buffers for both FCs and want to allocate the max
        // possible memory that would be required by any of the individual gemms.
        const int max_size    = std::max(hidden_units_, local_hidden_units_ + 2 * local_head_num_kv_ * size_per_head_);
        mixed_gemm_ws_bytes_  = weight_only_int8_fc_runner_->getWorkspaceSize(batch_size * seq_len, max_size, max_size);
        mixed_gemm_workspace_ = (char*)allocator_->reMalloc(mixed_gemm_workspace_, mixed_gemm_ws_bytes_, false);
    }
    is_allocate_buffer_ = true;
}

template<typename T>
void GptContextAttentionLayer<T>::freeBuffer()
{
    if (is_allocate_buffer_) {
        FT_LOG_DEBUG(__PRETTY_FUNCTION__);
        allocator_->free((void**)(&qkv_buf_));
        allocator_->free((void**)(&q_buf_2_));
        allocator_->free((void**)(&qk_buf_));
        allocator_->free((void**)(&qkv_buf_2_));

        if (is_qk_buf_float_ == true) {
            allocator_->free((void**)(&qk_buf_float_));
        }

        allocator_->free((void**)(&mixed_gemm_workspace_));
        mixed_gemm_ws_bytes_ = 0;

        allocator_->free((void**)(&int8_gemm_workspace_));
        int8_gemm_ws_bytes_ = 0;

        is_allocate_buffer_ = false;
    }
}

template class GptContextAttentionLayer<float>;
template class GptContextAttentionLayer<half>;
#ifdef ENABLE_BF16
template class GptContextAttentionLayer<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
