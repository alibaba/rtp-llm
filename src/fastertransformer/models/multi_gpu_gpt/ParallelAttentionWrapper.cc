#include "src/fastertransformer/models/multi_gpu_gpt/ParallelAttentionWrapper.h"
#include "src/fastertransformer/kernels/decoder_masked_multihead_attention.h"
#include "src/fastertransformer/kernels/kv_cache_utils.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/kernels/unfused_attention_kernels.h"
#include "src/fastertransformer/cuda/nvtx/nvtx_utils.h"
#include "src/fastertransformer/cuda/cuda_utils.h"

#include <type_traits>
#include <cassert>

namespace fastertransformer {

template<typename T>
bool ParallelAttentionWrapper<T>::CheckUseFMHA() const {
    char* fmha_env        = std::getenv("ENABLE_FMHA");
    bool  fmha_enable     = (fmha_env == nullptr || std::string(fmha_env) != "OFF");
    char* block_cache_env = std::getenv("REUSE_CACHE");
    bool  not_prefix_prompt =
        params_.pre_seq_len_ == 0 && (block_cache_env == nullptr || std::string(block_cache_env) != "1");
    char* multi_task_prompt_env = std::getenv("MULTI_TASK_PROMPT");
    bool use_medusa = params_.use_medusa_;
    char* sp_model_env = std::getenv("SP_MODEL_TYPE");

    if (!fmha_enable){
        FT_LOG_WARNING("FMHA is not enbaled");
        return false;
    }
    if (params_.rotary_embedding_style_ == 2){
        FT_LOG_WARNING("FMHA is disabled for not support chat-GLM");
        return false;
    }
    // TODO - TRT FMHA is likely support prefix prompt, need to test
    if (!not_prefix_prompt){
        FT_LOG_WARNING("FMHA is disabled for prefix prompt");
        return false;
    }
    if (sp_model_env != nullptr){
        FT_LOG_WARNING("FMHA not support sp_model");
        return false;
    }
    if(std::is_same<T, float>::value){
        FT_LOG_WARNING("FMHA not support float");
        return false;
    }
    if (multi_task_prompt_env != nullptr) {
        FT_LOG_WARNING("FMHA not support multi_task_prompt");
        return false;
    }
    if (use_medusa) {
        FT_LOG_WARNING("FMHA not support medusa model");
        return false;
    }
    return true;
}

template<typename T>
bool ParallelAttentionWrapper<T>::UseOpenSourceFMHA() const {
    bool use_open_source_fmha = CheckUseFMHA();
    if (!(is_sm8x() || is_sm90())) {
        FT_LOG_WARNING("opensource FMHA is disabled for sm %d", get_sm());
        use_open_source_fmha = false;
    }
    char* fmha_env = std::getenv("ENABLE_OPENSOURCE_FMHA");
    if (fmha_env && std::string(fmha_env) == "OFF") {
        FT_LOG_WARNING("opensource FMHA is disabled for by env");
        use_open_source_fmha = false;
    }
    return use_open_source_fmha;
}


template<typename T>
bool ParallelAttentionWrapper<T>::UseTRTFMHA() const {
    bool use_trt_fmha = CheckUseFMHA();
    if (!(is_sm8x() || is_sm90() || is_sm70())) {
        FT_LOG_WARNING("TRT FMHA is disabled for sm %d", get_sm());
        use_trt_fmha = false;
    }
    if (params_.is_sparse_head_){
        FT_LOG_WARNING("TRT FMHA is disabled for sparse");
        use_trt_fmha = false;
    }
    char* fmha_env = std::getenv("ENABLE_TRT_FMHA");
    if (fmha_env && std::string(fmha_env) == "OFF") {
        FT_LOG_WARNING("TRT FMHA is disabled for by env");
        use_trt_fmha = false;
    }
    return use_trt_fmha;
}

template<typename T>
bool ParallelAttentionWrapper<T>::UseMultiBlockMode() const {
    bool use_multi_block_mode = false;

    char* multi_block_mode_env    = std::getenv("ENABLE_MULTI_BLOCK_MODE");
    bool  multi_block_mode_enable = (multi_block_mode_env == nullptr || std::string(multi_block_mode_env) != "OFF");
#if (CUDART_VERSION >= 11070)
    if (!multi_block_mode_enable) {
        FT_LOG_WARNING("MMHA multi_block_mode is disabled");
        return false;
    }
    if (sm_ == 80 || sm_ >= 89) {
        FT_LOG_INFO("MMHA multi_block_mode is enabled");
        use_multi_block_mode = true;
    }
#endif
    return use_multi_block_mode;
}

template<typename T>
void ParallelAttentionWrapper<T>::TRTFMHA(const ContextAttentionParams& params, cudaStream_t stream)
{
#if (CUDART_VERSION >= 12000)
    const int num_heads = params_.head_num_;
    const int num_kv_heads = params_.head_num_kv_;
    const int head_size = params_.size_per_head_;
    const int mTokensPerBlock = params_.seq_size_per_block_;


    const int local_hidden_units_qo = num_heads * head_size;
    const int local_hidden_units_kv = num_kv_heads * head_size;
    const float q_scaling = q_scaling_;

    const bool mPagedContextFMHA=false;

    KVBlockArray kv_cache_buffer;
    using BufferDataType = int64_t;
    int64_t*   host_kv_cache_block_ptrs = nullptr;
    kv_cache_buffer          = KVBlockArray(params.batch_size, params.max_blocks_per_sequence, mTokensPerBlock, 0);
    kv_cache_buffer.data     = reinterpret_cast<BufferDataType*>(params.block_pointers);
    host_kv_cache_block_ptrs = reinterpret_cast<int64_t*>(params.host_block_pointers);

    const size_t cu_seqlens_size = sizeof(int) * (params.batch_size + 1);
    // It is assumed that the number of tokens per paged kv block should be >= 128.
    const size_t blocks_per_context_sequence = div_up(params.input_seq_length, mTokensPerBlock);
    const size_t paged_kv_tma_desc_size =
        mPagedContextFMHA ? params.batch_size * 2 * tensorrt_llm::kernels::TMA_DESC_SIZE_IN_BYTE * blocks_per_context_sequence : 0;

    int* cu_q_seqlens = params.cu_seqlens;
    int* cu_kv_seqlens = params.cu_seqlens;
    // TODO
    void* paged_kv_tma_desc = nullptr;

    // in context phase, currently FMHA runner has two restrictions:
    // 1. only apply to self attention. If want fused multi-head cross attention, FMHCA kernels and runner is needed
    // 2. doesn't apply to MHA with relative attention bias, i.e. softmax(QK + bias) * V
    // We update mEnableContextFMHA in constructor to check these conditions
    const bool enablePagedKVContextFMHA = mPagedContextFMHA;
    //  It is not needed with packed QKV input.
    if (enablePagedKVContextFMHA) {
        // to enable chunked attention,
        // 1. make sure you call setup_paged_kv(batch_size, max_query_length, max_kv_length, ....)
        // 2. make sure you call run_paged_kv(q_ptr, kv_tma_desc_device_ptr, kv_cache_block_ptrs_on_host,
        //                                    kv_cache_buffer, cu_q_seqlens, cu_kv_seqlens, ...)
        //    - q_ptr: [B, S, H, D], which supports variable sequence length
        //    - kv_tma_desc_device_ptr: allocated on device based on the number of context kv blocks.
        //    - kv_cache_block_ptrs_on_host: tma descriptors need the paged kv cache device ptrs to be in host.
        //    - kv_cache_buffer: paged kv buffer
        //    - cu_q_seqlens: the cumulative query sequence lengths, needed for variable sequence length.
        //    - cu_kv_seqlens: the cumulative kv sequence lengths, needed for variable sequence length.

        // the token will pay attention to previous tokens while starting from max(0, rowIdx -
        // cyclic_kv_cache_length);
        PUSH_RANGE(stream_, "trt_fmha");
        mFMHARunner->setup_paged_kv(params.batch_size,
                                    params.input_seq_length,
                                    params.max_past_kv_len,
                                    blocks_per_context_sequence,
                                    mTokensPerBlock,
                                    params.cyclic_kv_cache_length,
                                    params.num_tokens,
                                    params.is_alibi,
                                    params.is_alibi_with_sacle,
                                    tensor_para_.world_size_,
                                    tensor_para_.rank_);
        mFMHARunner->run_paged_kv(q_buf_2_,
                                  paged_kv_tma_desc,
                                  host_kv_cache_block_ptrs,
                                  reinterpret_cast<KVBlockArray&>(kv_cache_buffer),
                                  cu_q_seqlens,
                                  cu_kv_seqlens,
                                  params.context_buf,
                                  stream);
        POP_RANGE;
        sync_check_cuda_error();
    }
    else {
        // the token will pay attention to previous tokens while starting from max(0, rowIdx -
        // cyclic_kv_cache_length);
        PUSH_RANGE(stream_, "trt_fmha");
        mFMHARunner->setup(params.batch_size,
                           params.input_seq_length,
                           params.cyclic_kv_cache_length,
                           params.num_tokens,
                           params.is_alibi,
                           params.is_alibi_with_sacle,
                           1,
                           0);
        mFMHARunner->run(const_cast<T*>(params.attention_input), cu_q_seqlens, params.context_buf, stream);
        POP_RANGE;
        sync_check_cuda_error();
    }
#endif
}

template<typename T>
void ParallelAttentionWrapper<T>::OpenSourceFMHA(
    T*           qkv,  // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    int*         cu_seqlens,
    const int    batch_size,
    const int    num_heads,
    const int    num_heads_kv,
    const int    head_size,
    const int    max_seqlen,
    const float  softmax_scale,
    T*           linear_bias_slopes,
    T*           out,
    cudaStream_t stream)
{
    FT_CHECK_WITH_INFO(head_size % 8 == 0, "FlashAttention only supports head size % 8 = 0");

    FT_CHECK_WITH_INFO(num_heads % num_heads_kv == 0, "Number of heads in key/value must divide number of heads in query");
    auto      round_multiple    = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_rounded = round_multiple(head_size, 32);
    const int seqlen_rounded    = round_multiple(max_seqlen, 128);
    memset(&flash_fwd_params_, 0, sizeof(flash_fwd_params_));
    flash_fwd_params_.is_bf16 = std::is_same_v<__nv_bfloat16, T>;

    // TODO(wangyin): pass hidden_units from params
    const int hidden_units          = num_heads * head_size;
    const int hidden_units_kv       = num_heads_kv * head_size;
    // Set the pointers and strides.
    flash_fwd_params_.q_ptr = qkv;
    flash_fwd_params_.k_ptr = qkv + hidden_units;
    flash_fwd_params_.v_ptr = qkv + hidden_units + hidden_units_kv;
    // All stride are in elements, not bytes.
    flash_fwd_params_.q_row_stride  = hidden_units + 2 * hidden_units_kv;
    flash_fwd_params_.k_row_stride  = hidden_units + 2 * hidden_units_kv;
    flash_fwd_params_.v_row_stride  = hidden_units + 2 * hidden_units_kv;
    flash_fwd_params_.q_head_stride = head_size;
    flash_fwd_params_.k_head_stride = head_size;
    flash_fwd_params_.v_head_stride = head_size;
    flash_fwd_params_.o_ptr         = out;
    flash_fwd_params_.o_row_stride  = hidden_units;
    flash_fwd_params_.o_head_stride = head_size;

    if (cu_seqlens == nullptr) {
        flash_fwd_params_.q_batch_stride = max_seqlen * (hidden_units + 2 * hidden_units_kv);
        flash_fwd_params_.k_batch_stride = max_seqlen * (hidden_units + 2 * hidden_units_kv);
        flash_fwd_params_.v_batch_stride = max_seqlen * (hidden_units + 2 * hidden_units_kv);
        flash_fwd_params_.o_batch_stride = max_seqlen * hidden_units;
    }

    flash_fwd_params_.cu_seqlens_q = static_cast<int*>(cu_seqlens);
    flash_fwd_params_.cu_seqlens_k = static_cast<int*>(cu_seqlens);

    // P = softmax(QK^T)
    flash_fwd_params_.p_ptr = nullptr;

    // Softmax sum
    flash_fwd_params_.softmax_lse_ptr = softmax_lse_;

    // Set the dimensions.
    flash_fwd_params_.b                = batch_size;
    flash_fwd_params_.h                = num_heads;
    flash_fwd_params_.h_k              = num_heads_kv;
    flash_fwd_params_.h_h_k_ratio      = num_heads / num_heads_kv;
    flash_fwd_params_.seqlen_q         = max_seqlen;
    flash_fwd_params_.seqlen_k         = max_seqlen;
    flash_fwd_params_.seqlen_q_rounded = seqlen_rounded;
    flash_fwd_params_.seqlen_k_rounded = seqlen_rounded;
    flash_fwd_params_.d                = head_size;
    flash_fwd_params_.d_rounded        = head_size_rounded;

    // Set the different scale values.
    flash_fwd_params_.scale_softmax      = softmax_scale;
    flash_fwd_params_.scale_softmax_log2 = softmax_scale * M_LOG2E;

    // Set this to probability of keeping an element to simplify things.
    float p_dropout             = 0.0f;
    flash_fwd_params_.p_dropout = 1.f - p_dropout;
    // Convert p from float to int so we don't have to convert the random uint to float to compare.
    // [Minor] We want to round down since when we do the comparison we use <= instead of <
    // params.p_dropout_in_uint = uint32_t(std::floor(params.p_dropout * 4294967295.0));
    // params.p_dropout_in_uint16_t = uint16_t(std::floor(params.p_dropout * 65535.0));
    flash_fwd_params_.p_dropout_in_uint8_t     = uint8_t(std::floor(flash_fwd_params_.p_dropout * 255.0));
    flash_fwd_params_.rp_dropout               = 1.f / flash_fwd_params_.p_dropout;
    flash_fwd_params_.scale_softmax_rp_dropout = flash_fwd_params_.rp_dropout * flash_fwd_params_.scale_softmax;
    TORCH_CHECK(p_dropout < 1.f);

    flash_fwd_params_.is_causal = params_.is_causal_;
    flash_fwd_params_.is_alibi  = false;
    if (linear_bias_slopes) {
        flash_fwd_params_.is_alibi           = true;
        flash_fwd_params_.linear_bias_slopes = linear_bias_slopes;
    }
    flash_fwd_params_.is_seqlens_k_cumulative = true;

    PUSH_RANGE(stream_, "fmha");
    run_mha_fwd(flash_fwd_params_, stream);
    POP_RANGE;
}

template<typename T>
void ParallelAttentionWrapper<T>::preAllocate()
{
    FT_LOG_WARNING("use fmha: %s", std::to_string(use_open_source_fmha_ || use_trt_fmha_).c_str());
    allocateBuffer(params_.max_generate_batch_size_ + params_.max_context_batch_size_ * params_.max_seq_len_,
                   params_.max_context_batch_size_,
                   params_.max_generate_batch_size_,
                   params_.max_seq_len_,
                   params_.max_seq_len_,
                   !(use_open_source_fmha_ || use_trt_fmha_),
                   multi_block_mode_);
}

template<typename T>
void ParallelAttentionWrapper<T>::forward(TensorMap*                output_tensors,
                                          TensorMap*                input_tensors,
                                          const AttentionWeight<T>* attention_weights)
{
    // input_tensors:
    //      input_query [token_num, hidden_dimension]
    //      attention_mask [batch_size, 1, seq_len, seq_len + max_prompt_length]
    //      is_final_layer [1], bool on cpu
    //      layer_id [1], int on cpu
    //      padding_offset, int, [token_num] (optional)
    //      cu_seqlens, int, [batch_size] (optional)
    //      d_prefix_prompt_batch [global_batch_size], (optional)
    //          each element contains ptr with buffer shape[2, local_head_num_, prompt_length, size_per_head]
    //      d_prefix_prompt_lengths [batch_size], int (optional)
    //      linear_bias_slopes [head_num] (optional)
    //      lora_ids [batch_size] (optional)

    // output_tensors:
    //      hidden_features [token_num, hidden_dimension]
    //      key_cache [batch, local_head_num, size_per_head // x, max_seq_len, x]
    //      value_cache [batch, local_head_num, max_seq_len, size_per_head]
    FT_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    Attention(output_tensors, input_tensors, attention_weights);

    // PUSH_RANGE(stream_, "all reduce sum");
    const size_t size          = output_tensors->at("hidden_features").size();
    T*           attention_out = output_tensors->getPtr<T>("hidden_features");
    if (tensor_para_.world_size_ > 1) {
        ftNcclAllReduceSum(attention_out, attention_out, size, tensor_para_, stream_);
        sync_check_cuda_error();
    }
    // POP_RANGE;

    FT_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

template<typename T>
void ParallelAttentionWrapper<T>::DenseGemm(const int                 h_token_num,
                                            const int                 layer_id,
                                            T*                        attention_out,
                                            const AttentionWeight<T>* attention_weights,
                                            int *                     lora_ids,
                                            int                       batch_size,
                                            const int*                lora_input_lengths)
{
    const int local_hidden_units_rt =
        (params_.is_sparse_head_ ? local_layer_head_num_[layer_id] : local_head_num_) * params_.size_per_head_;
    const int local_hidden_units_kv_rt =
        (params_.is_sparse_head_ ? local_layer_head_num_kv_[layer_id] : local_head_num_kv_) * params_.size_per_head_;
    T* qkv_buf_3_input = nullptr;
    if (attention_weights->attention_layernorm.gamma && attention_weights->attention_layernorm.beta) {
        invokeGeneralLayerNorm(qkv_buf_,
                               qkv_buf_2_,
                               attention_weights->attention_layernorm.gamma,
                               attention_weights->attention_layernorm.beta,
                               params_.layernorm_eps_,
                               h_token_num,
                               local_hidden_units_rt,
                               nullptr,
                               nullptr,
                               params_.quant_algo_->int8_mode_,
                               stream_);
        qkv_buf_3_input = qkv_buf_;
        sync_check_cuda_error();
        print_bsd(layer_id, "attn ln", qkv_buf_, h_token_num, 1, local_hidden_units_rt);
    }
    else {
        qkv_buf_3_input = qkv_buf_2_;
    }
    print_bsd(layer_id, "attn before o", qkv_buf_3_input, h_token_num, 1, local_hidden_units_rt);

    PUSH_RANGE(stream_, "proj_gemm");
#ifdef SPARSITY_ENABLED
    const int m_padded   = 8 * div_up(m, 8);
    bool      use_sparse = sparse_ && cublas_wrapper_->isUseSparse(1, hidden_units_, m_padded, local_hidden_units_rt);
#else
    constexpr bool use_sparse = false;
    const int      m_padded   = 0;
#endif

    // QKV gemm: [m, hidden_dim] * [hidden_dim, qkv_dim] = [m, qkv_dim]
    gemm_runner_->Gemm(h_token_num,
                       hidden_units_,
                       local_hidden_units_rt,
                       qkv_buf_3_input,
                       &attention_weights->attention_output_weight,
                       attention_out);
    
    // lora
    lora_gemm_->applyLoRA(h_token_num,
                          batch_size,
                          lora_input_lengths,
                          local_hidden_units_rt,
                          hidden_units_,
                          lora_ids,
                          attention_weights->attention_output_weight.lora_weights,
                          qkv_buf_3_input,
                          attention_out);

    POP_RANGE;
    // print_bsd(layer_id, "attn out", attention_out, 1, h_token_num, local_hidden_units_);
}

template<typename T>
void ParallelAttentionWrapper<T>::QKVGemm(const int                 h_token_num,
                                          const int                 layer_id,
                                          const T*                  attention_input,
                                          const AttentionWeight<T>* attention_weights,
                                          int *                     lora_ids,
                                          int                       batch_size,
                                          const int*                lora_input_lengths)
{
    const int local_head_num        = params_.is_sparse_head_ ? local_layer_head_num_[layer_id] : local_head_num_;
    const int local_head_num_kv     = params_.is_sparse_head_ ? local_layer_head_num_kv_[layer_id] : local_head_num_kv_;
    const int local_hidden_units_rt = local_head_num * params_.size_per_head_;
    const int local_hidden_units_kv_rt = local_head_num_kv * params_.size_per_head_;
    PUSH_RANGE(stream_, "qkv_gemm");

#ifdef SPARSITY_ENABLED
    const int m_padded   = 8 * div_up(m, 8);
    bool      use_sparse = sparse_
                      && cublas_wrapper_->isUseSparse(
                          1, local_hidden_units_rt + 2 * local_hidden_units_kv_rt, m_padded, hidden_units_);
#else
    constexpr bool use_sparse = false;
    const int      m_padded   = 0;
#endif

    // QKV gemm: [m, hidden_dim] * [hidden_dim, qkv_dim] = [m, qkv_dim]
    gemm_runner_->Gemm(h_token_num,
                       local_hidden_units_rt + 2 * local_hidden_units_kv_rt,
                       hidden_units_,
                       attention_input,
                       &attention_weights->query_weight,
                       qkv_buf_);
    
    // lora

    lora_gemm_->applyLoRA(h_token_num,
                          batch_size,
                          lora_input_lengths,
                          hidden_units_,
                          local_hidden_units_rt + 2 * local_hidden_units_kv_rt,
                          lora_ids,
                          attention_weights->query_weight.lora_weights,
                          attention_input,
                          qkv_buf_);

    int k_start = local_hidden_units_rt;
    int v_start = local_hidden_units_rt + local_hidden_units_kv_rt;
    print_bsd(layer_id, "parallel attention q", qkv_buf_, h_token_num, 1, local_hidden_units_rt + 2 * local_hidden_units_kv_rt, 0, 4);
    print_bsd(layer_id,
              "k",
              qkv_buf_,
              h_token_num,
              1,
              local_hidden_units_rt + 2 * local_hidden_units_kv_rt,
              k_start,
              k_start + 4);
    print_bsd(layer_id,
              "v",
              qkv_buf_,
              h_token_num,
              1,
              local_hidden_units_rt + 2 * local_hidden_units_kv_rt,
              v_start,
              v_start + 4);
    POP_RANGE;
}

template<typename T>
void ParallelAttentionWrapper<T>::SelfAttention(TensorMap*                output_tensors,
                                                TensorMap*                input_tensors,
                                                const AttentionWeight<T>* attention_weights)
{   
    if (input_tensors->isExist("use_kvcache") && !input_tensors->getVal<bool>("use_kvcache")) {
        throw std::runtime_error("use_kvcahe == false should not do self attention!");
    }
    const int  layer_id                = input_tensors->getVal<int>("layer_id");
    const int  generate_batch_size     = input_tensors->getVal<int>("generate_batch_size");
    const int* cache_indir             = input_tensors->getPtr<int>("cache_indirection", nullptr);
    const int  beam_width              = cache_indir ? input_tensors->at("cache_indirection").shape()[1] : 1;
    const T*   relative_attention_bias = input_tensors->getPtr<T>("relative_attention_bias", nullptr);
    const int  relative_attention_bias_stride =
        relative_attention_bias ? input_tensors->at("relative_attention_bias").shape()[3] : 0;
    const int64_t* block_pointers       = input_tensors->getPtr<int64_t>("block_pointers", nullptr);
    const int64_t* block_scale_pointers = input_tensors->getPtr<int64_t>("block_scale_pointers", nullptr);
    const int      max_blocks_per_batch = block_pointers ? input_tensors->at("block_pointers").shape()[3] : 0;
    const int      local_head_num       = params_.is_sparse_head_ ? local_layer_head_num_[layer_id] : local_head_num_;
    const int    local_head_num_kv = params_.is_sparse_head_ ? local_layer_head_num_kv_[layer_id] : local_head_num_kv_;
    KVBlockArray kv_block_array(generate_batch_size, max_blocks_per_batch, params_.seq_size_per_block_, 0);
    kv_block_array.data = const_cast<int64_t*>(block_pointers);
    if (params_.int8_kv_cache_) {
        kv_block_array.int8_mode = true;
        kv_block_array.scale     = const_cast<int64_t*>(block_scale_pointers);
    }
    fusedQKV_masked_attention_dispatch(
        qkv_buf_,
        attention_weights->query_weight.bias,
        relative_attention_bias,
        cache_indir,
        qkv_buf_2_,
        input_tensors->getPtr<bool>("finished", nullptr),
        input_tensors->getPtr<int>(
            "sequence_lengths"),  // NOTE: current seq len including padding (fixed after meeting the finished id)
        generate_batch_size,
        beam_width,
        local_head_num,
        local_head_num_kv,
        params_.size_per_head_,
        params_.rotary_embedding_dim_,
        params_.rotary_embedding_style_,
        params_.rotary_embedding_base_,
        params_.logn_seq_len_,
        params_.use_logn_attn_,
        params_.dynamic_embedding_scalar_,
        params_.dynamic_embedding_max_pos_,
        params_.position_embeddings_scale_,
        params_.base_scale_,
        input_tensors->getVal<int>("step"),  // memory_max_len
        input_tensors->getPtr<int>("d_prefix_prompt_lengths", nullptr),
        input_tensors->getVal<int>("max_prefix_prompt_length", 0),
        input_tensors->getVal<bool>("count_prefix_length", false),
        input_tensors->getPtr<int>("input_lengths", nullptr),
        input_tensors->getVal<int>("step"),
        q_scaling_,
        relative_attention_bias_stride,
        input_tensors->getPtr<T>("linear_bias_slopes", nullptr),
        input_tensors->getPtr<bool>("masked_tokens", nullptr),
        params_.quant_algo_->int8_mode_ == 2 ? attention_weights->query_weight.scale_out : nullptr,
        params_.quant_algo_->int8_mode_ == 2 ? attention_weights->attention_output_weight.scale : nullptr,
        params_.quant_algo_->int8_mode_,
        multi_block_mode_,
        max_seq_len_tile_,
        partial_out_,
        partial_sum_,
        partial_max_,
        block_counter_,
        kv_block_array,
        stream_);
    sync_check_cuda_error();
}

template<typename T>
void ParallelAttentionWrapper<T>::ContextAttention(TensorMap*                output_tensors,
                                                   TensorMap*                input_tensors,
                                                   const AttentionWeight<T>* attention_weights)
{
    const bool use_kvcache = !input_tensors->isExist("use_kvcache") || input_tensors->getVal<bool>("use_kvcache");
    const int  generate_batch_size      = input_tensors->getVal<int>("generate_batch_size");
    const int  h_token_num              = input_tensors->at("input_query").shape()[0];
    const int  context_h_token_num      = h_token_num - generate_batch_size;
    const int  layer_id                 = input_tensors->getVal<int>("layer_id");
    const int  context_batch_size       = input_tensors->getVal<int>("context_batch_size");
    const int  max_context_seq_length   = input_tensors->getVal<int>("max_context_seq_length");
    const int  min_prefix_length        = input_tensors->getVal<int>("min_prefix_length", 0);
    const T**  d_prefix_prompt_batch    = input_tensors->getPtr<const T*>("d_prefix_prompt_batch", nullptr);
    const T*   attention_mask           = input_tensors->getPtr<const T>("attention_mask", nullptr);
    const int* d_prefix_prompt_lengths_ = input_tensors->getPtr<int>("d_prefix_prompt_lengths", nullptr);
    const int* d_prefix_prompt_lengths =
        d_prefix_prompt_lengths_ ? d_prefix_prompt_lengths_ + generate_batch_size : d_prefix_prompt_lengths_;
    const int* padding_offset     = input_tensors->getPtr<int>("padding_offset", nullptr);
    // position_id shape: [h_token_num]
    const int* position_ids       = input_tensors->getPtr<int>("position_ids", nullptr);
    int*       cu_seqlens         = input_tensors->getPtr<int>("cu_seqlens", nullptr);
    T*         linear_bias_slopes = input_tensors->getPtr<T>("linear_bias_slopes", nullptr);
    // const int* block_index_map_     = input_tensors->getPtr<int>("block_index_map", nullptr);
    int64_t* block_pointers_       = input_tensors->getPtr<int64_t>("block_pointers", nullptr);
    int64_t* host_block_pointers_  = input_tensors->getPtr<int64_t>("host_block_pointers", nullptr);
    const int64_t* block_scale_pointers_ = input_tensors->getPtr<int64_t>("block_scale_pointers", nullptr);
    const int      max_blocks_per_batch  = block_pointers_ ? input_tensors->at("block_pointers").shape()[3] : 0;
    const int64_t* block_pointers =
        block_pointers_ ? block_pointers_ + generate_batch_size * 2 * max_blocks_per_batch : block_pointers_;
    const int64_t* block_scale_pointers = block_scale_pointers_ ?
                                              block_scale_pointers_ + generate_batch_size * 2 * max_blocks_per_batch :
                                              block_scale_pointers_;

    const int  max_prompt_length   = attention_mask == nullptr ? 0 :
                                                                 input_tensors->at("attention_mask").shape()[3]
                                                                  - input_tensors->at("attention_mask").shape()[2];
    const bool count_prefix_length = input_tensors->getVal<bool>("count_prefix_length", false);
    const int* input_lengths       = input_tensors->getPtr<int>("input_lengths");

    const int local_head_num        = params_.is_sparse_head_ ? local_layer_head_num_[layer_id] : local_head_num_;
    const int local_head_num_kv     = params_.is_sparse_head_ ? local_layer_head_num_kv_[layer_id] : local_head_num_kv_;
    const int local_hidden_units_rt = local_head_num * params_.size_per_head_;
    const int local_hidden_units_kv_rt = local_head_num_kv * params_.size_per_head_;

    T* qkv_buf   = qkv_buf_ + generate_batch_size * (local_hidden_units_rt + 2 * local_hidden_units_kv_rt);
    T* qkv_buf_2 = qkv_buf_2_ + generate_batch_size * local_hidden_units_rt;

    KVBlockArray kv_block_array(context_batch_size, max_blocks_per_batch, params_.seq_size_per_block_, 0);
    KvCacheDataType cache_type = KvCacheDataType::BASE;
    if (use_kvcache) {
        kv_block_array.data        = const_cast<int64_t*>(block_pointers);        
        if (params_.int8_kv_cache_) {
            kv_block_array.scale     = const_cast<int64_t*>(block_scale_pointers);
            kv_block_array.int8_mode = true;
            cache_type               = KvCacheDataType::INT8;
        }
    }

    PrefixPromptBatchWeightsParam<T>* param          = new PrefixPromptBatchWeightsParam<T>();

    if (d_prefix_prompt_lengths && d_prefix_prompt_batch) {
        throw std::runtime_error("not support both prefix_prompt and repeat_prompt");
    }
    else if (d_prefix_prompt_batch) {
        param = new PrefixPromptBatchWeightsParam<T>{
            d_prefix_prompt_lengths,
            max_prompt_length,
            count_prefix_length,
            kv_block_array,
            ContinuousCacheParam<T>{d_prefix_prompt_batch,
                                    (size_t)(layer_id * 2 * local_head_num_kv * params_.size_per_head_)}};
    }
    else if (d_prefix_prompt_lengths) {
        param = new PrefixPromptBatchWeightsParam<T>{
            d_prefix_prompt_lengths, max_prompt_length, count_prefix_length, kv_block_array, ContinuousCacheParam<T>{}};
    }

    if (padding_offset != nullptr) {
        // q_buf_2_, k_buf_2_ and v_buf_2_ are continuousd
        cudaMemsetAsync(q_buf_2_,
                        0,
                        context_batch_size * (max_context_seq_length + max_prompt_length)
                            * (local_hidden_units_rt + 2 * local_hidden_units_kv_rt) * sizeof(T),
                        stream_);
    }

    PUSH_RANGE(stream_, "qkv_bias_add");
    invokeAddFusedQKVBiasTranspose(q_buf_2_,
                                   k_buf_2_,
                                   v_buf_2_,
                                   *param,  // prefix prompt
                                   qkv_buf,
                                   position_ids,
                                   attention_weights->query_weight.bias,
                                   padding_offset,
                                   cu_seqlens,
                                   context_batch_size,
                                   max_context_seq_length,
                                   context_h_token_num,
                                   local_head_num,
                                   local_head_num_kv,
                                   params_.size_per_head_,
                                   params_.rotary_embedding_dim_,
                                   params_.rotary_embedding_style_,
                                   params_.rotary_embedding_base_,
                                   params_.dynamic_embedding_scalar_,
                                   params_.dynamic_embedding_max_pos_,
                                   params_.position_embeddings_scale_,
                                   params_.base_scale_,
                                   params_.logn_seq_len_,
                                   params_.use_logn_attn_,
                                   attention_weights->query_weight.scale_out,
                                   params_.quant_algo_->int8_mode_,
                                   stream_);
    POP_RANGE;
    print_bhsd(layer_id,
               "q bias rotary",
               q_buf_2_,
               context_batch_size,
               local_head_num,
               max_context_seq_length,
               params_.size_per_head_);
    print_bhsd(layer_id,
               "k bias rotary",
               k_buf_2_,
               context_batch_size,
               local_head_num_kv,
               max_context_seq_length + max_prompt_length,
               params_.size_per_head_);
    print_bhsd(layer_id,
               "v bias rotary",
               v_buf_2_,
               context_batch_size,
               local_head_num_kv,
               max_context_seq_length + max_prompt_length,
               params_.size_per_head_);

    sync_check_cuda_error();
    delete param;
    // Use batch major
    // put k/v_buf from shape [B, H, PL + L, Dh]
    // to cache [B, H, Dh/x, PL + L, x]  and [B, H, PL + L, Dh/x, x], PL denotes prompt length
    // length_base means some blocks is reused in kvcache and not need to copy

    PUSH_RANGE(stream_, "kv_cache");
    if (use_kvcache) {
        invokeTranspose4dBatchMajor(k_buf_2_,
                                    v_buf_2_,
                                    kv_block_array,
                                    context_batch_size,
                                    max_prompt_length + max_context_seq_length,  // max input length + prefix prompt length
                                    params_.size_per_head_,
                                    local_head_num_kv,
                                    cache_type,
                                    nullptr,  // kvScaleOrigQuant
                                    input_lengths + generate_batch_size,
                                    d_prefix_prompt_lengths,
                                    stream_);
    }
    POP_RANGE;
    // IDEA : after this, k_cache = (batch_size, num_heads, Dh/x, prefix_prompt_len + L, x)
    // k_cache = (batch_size, num_heads, prefix_prompt_len + L, Dh)
    sync_check_cuda_error();

    // NOTE: qkv buffer shape (batch_size, num_heads,L or prompt_len + L, Dh)
    const cudaDataType_t gemm_data_type      = getCudaDataType<T>();
    const int            attention_seq_len_1 = max_context_seq_length;                      // q length
    const int            attention_seq_len_2 = max_prompt_length + max_context_seq_length;  // kv length
    const T              qk_scale               = static_cast<T>(1.0f / sqrtf(params_.size_per_head_ * 1.0f));
    if (use_trt_fmha_) {
        ContextAttentionParams context_attention_params{qkv_buf,                 // attention_input
                                                        max_context_seq_length,  // max_context_q_len
                                                        max_context_seq_length,  // max_context_kv_len,
                                                        cu_seqlens,
                                                        max_context_seq_length,
                                                        qkv_buf_2,        // context_buf_,
                                                        block_pointers_,  // block_pointers,
                                                        host_block_pointers_,
                                                        context_batch_size,    // batch_size
                                                        context_h_token_num,   // localNbTokens,
                                                        max_blocks_per_batch,  // max_blocks_per_sequence
                                                        linear_bias_slopes != nullptr};
        TRTFMHA(context_attention_params, stream_);
    }
    else if (use_open_source_fmha_) {
        OpenSourceFMHA(qkv_buf,
                       cu_seqlens,
                       context_batch_size,
                       local_head_num,
                       local_head_num_kv,
                       params_.size_per_head_,
                       max_context_seq_length,
                       (1.0f / sqrtf(params_.size_per_head_ * 1.0f)),
                       linear_bias_slopes,
                       qkv_buf_2,
                       stream_);
        sync_check_cuda_error();
    }
    else {
        FT_CHECK_WITH_INFO(attention_mask != nullptr, "attention mask should not be nullptr when not use flash attention");
        if (is_qk_buf_float_ == true && gemm_data_type != CUDA_R_32F) {
            PUSH_RANGE(stream_, "Q*K");
            cublas_wrapper_->stridedBatchedGemm(
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                attention_seq_len_2,                                         // m
                attention_seq_len_1 * (local_head_num / local_head_num_kv),  // n
                params_.size_per_head_,                                      // k
                1.0f,
                k_buf_2_,                                      // A
                gemm_data_type,                                // Atype
                params_.size_per_head_,                        // lda             k
                attention_seq_len_2 * params_.size_per_head_,  // strideA n * k
                q_buf_2_,                                      // B
                gemm_data_type,                                // Btype
                params_.size_per_head_,                        // ldb                  // k
                attention_seq_len_1 * params_.size_per_head_ * (local_head_num / local_head_num_kv),  // strideB m * k
                0.0f,
                qk_buf_float_,                                                                     // C
                CUDA_R_32F,                                                                        // Ctype
                attention_seq_len_2,                                                               // ldc  n
                attention_seq_len_2 * attention_seq_len_1 * (local_head_num / local_head_num_kv),  // strideC
                context_batch_size * local_head_num_kv,                                            // global batch size
                CUDA_R_32F);

            sync_check_cuda_error();
            POP_RANGE;

            print_bhss(layer_id,
                       "qk",
                       qk_buf_float_,
                       context_batch_size,
                       local_head_num,
                       attention_seq_len_1,
                       attention_seq_len_2);

            PUSH_RANGE(stream_, "softmax");
            MaskedSoftmaxParam<T, float> param;
            param.attention_score    = qk_buf_;         // (batch_size, head_num, q_length, k_length)
            param.qk                 = qk_buf_float_;   // (batch_size, head_num, q_length, k_length)
            param.attention_mask     = attention_mask;  // (batch_size, q_length, k_length)
            param.batch_size         = context_batch_size;
            param.q_length           = attention_seq_len_1;
            param.k_length           = attention_seq_len_2;
            param.num_heads          = local_head_num;
            param.qk_scale           = qk_scale;
            param.linear_bias_slopes = const_cast<T*>(linear_bias_slopes);  // (head_num,), optional
            invokeMaskedSoftmax(param, stream_);
            print_bhss(layer_id,
                       "softmax",
                       qk_buf_,
                       context_batch_size,
                       local_head_num,
                       attention_seq_len_1,
                       attention_seq_len_2);
            sync_check_cuda_error();
            POP_RANGE;
        }

        else {
            PUSH_RANGE(stream_, "Q*K");
            cublas_wrapper_->stridedBatchedGemm(
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                attention_seq_len_2,
                attention_seq_len_1 * (local_head_num / local_head_num_kv),
                params_.size_per_head_,
                k_buf_2_,
                params_.size_per_head_,
                attention_seq_len_2 * params_.size_per_head_,
                q_buf_2_,
                params_.size_per_head_,
                attention_seq_len_1 * params_.size_per_head_ * (local_head_num / local_head_num_kv),
                qk_buf_,
                attention_seq_len_2,
                attention_seq_len_2 * attention_seq_len_1 * (local_head_num / local_head_num_kv),
                context_batch_size * local_head_num_kv);
            print_bhss(
                layer_id, "qk", qk_buf_, context_batch_size, local_head_num, attention_seq_len_1, attention_seq_len_2);
            POP_RANGE;

            PUSH_RANGE(stream_, "softmax");
            MaskedSoftmaxParam<T, T> param;
            param.attention_score    = qk_buf_;         // (batch_size, head_num, q_length, k_length)
            param.qk                 = qk_buf_;         // (batch_size, head_num, q_length, k_length)
            param.attention_mask     = attention_mask;  // (batch_size, q_length, k_length)
            param.batch_size         = context_batch_size;
            param.q_length           = attention_seq_len_1;
            param.k_length           = attention_seq_len_2;
            param.num_heads          = local_head_num;
            param.qk_scale           = qk_scale;
            param.linear_bias_slopes = const_cast<T*>(linear_bias_slopes);  // (head_num,), optional
            invokeMaskedSoftmax(param, stream_);
            print_bhss(layer_id,
                       "softmax",
                       qk_buf_,
                       context_batch_size,
                       local_head_num,
                       attention_seq_len_1,
                       attention_seq_len_2);
            sync_check_cuda_error();
            POP_RANGE;
        }

        PUSH_RANGE(stream_, "QK*V");
        cublas_wrapper_->stridedBatchedGemm(
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            params_.size_per_head_,
            attention_seq_len_1 * (local_head_num / local_head_num_kv),
            attention_seq_len_2,
            v_buf_2_,
            params_.size_per_head_,
            attention_seq_len_2 * params_.size_per_head_,
            qk_buf_,
            attention_seq_len_2,
            attention_seq_len_1 * (local_head_num / local_head_num_kv) * attention_seq_len_2,
            qkv_buf_3_,
            params_.size_per_head_,
            attention_seq_len_1 * (local_head_num / local_head_num_kv) * params_.size_per_head_,
            context_batch_size * local_head_num_kv);
        POP_RANGE;

        print_bhsd(layer_id, "qkv_weighted", qkv_buf_3_, context_batch_size, local_head_num, attention_seq_len_1, params_.size_per_head_);

        // transpose (batch_size, num_heads, L, Dh) to (batch_size, L, num_heads * Dh)
        PUSH_RANGE(stream_, "transpose");
        if (padding_offset == nullptr) {
            invokeTransposeQKV(qkv_buf_2,
                               qkv_buf_3_,
                               context_batch_size,
                               attention_seq_len_1,
                               local_head_num,
                               params_.size_per_head_,
                               attention_weights->attention_output_weight.scale,
                               params_.quant_algo_->int8_mode_,
                               stream_);
            sync_check_cuda_error();
        }
        else {
            invokeTransposeAttentionOutRemovePadding(qkv_buf_3_,
                                                     qkv_buf_2,
                                                     context_h_token_num,
                                                     context_batch_size,
                                                     attention_seq_len_1,
                                                     local_head_num,
                                                     params_.size_per_head_,
                                                     padding_offset,
                                                     attention_weights->attention_output_weight.scale,
                                                     params_.quant_algo_->int8_mode_,
                                                     stream_);
        }
        POP_RANGE;
    }
    sync_check_cuda_error();
    print_bshd(layer_id, "qkv_weighted_t", qkv_buf_2, 1, h_token_num, local_head_num, params_.size_per_head_);
}

template<typename T>
void ParallelAttentionWrapper<T>::Attention(TensorMap*                output_tensors,
                                            TensorMap*                input_tensors,
                                            const AttentionWeight<T>* attention_weights)
{

    const int           generate_batch_size = input_tensors->getVal<int>("generate_batch_size");
    const int           context_batch_size  = input_tensors->getVal<int>("context_batch_size");
    int                 max_context_seq_len = 0;
    int                 max_context_seq_len_with_prefix = 0;
    const int           layer_id            = input_tensors->getVal<int>("layer_id");
    const int           h_token_num         = input_tensors->at("input_query").shape()[0];
    // lora
    int* lora_ids = input_tensors->getPtr<int>("lora_ids", nullptr);
    int batch_size = context_batch_size + generate_batch_size;
    const int* lora_input_lengths = input_tensors->getPtr<int>("lora_input_lengths", nullptr);

    if (context_batch_size) {
        if (input_tensors->isExist("attention_mask")) {
            max_context_seq_len_with_prefix = input_tensors->at("attention_mask").shape()[3];
            max_context_seq_len             = input_tensors->at("attention_mask").shape()[2];
        } else {
            max_context_seq_len = input_tensors->getVal<int>("max_context_seq_length");
            max_context_seq_len_with_prefix = max_context_seq_len;
        }
    }
    PUSH_RANGE(stream_, "attention_buffer_alloc");
    allocateBuffer(h_token_num, context_batch_size, generate_batch_size, max_context_seq_len, max_context_seq_len_with_prefix, !(use_open_source_fmha_||use_trt_fmha_), multi_block_mode_);
    POP_RANGE;
    sync_check_cuda_error();

    QKVGemm(h_token_num, layer_id, input_tensors->at("input_query").getPtr<T>(), attention_weights,
            lora_ids, batch_size, lora_input_lengths);
    if (context_batch_size) {
        ContextAttention(output_tensors, input_tensors, attention_weights);
    }
    if (generate_batch_size) {
        SelfAttention(output_tensors, input_tensors, attention_weights);
    }
    DenseGemm(h_token_num, layer_id, output_tensors->at("hidden_features").getPtr<T>(), attention_weights,
              lora_ids, batch_size, lora_input_lengths);
    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
    sync_check_cuda_error();
}

template<typename T>
ParallelAttentionWrapper<T>::ParallelAttentionWrapper(const GptInitParameter& gpt_init_parameter,
                                                      NcclParam               tensor_para,
                                                      cudaStream_t            stream,
                                                      cublasMMWrapper*        cublas_wrapper,
                                                      tc::QuantAlgo           quant_algo,
                                                      IAllocator*             allocator,
                                                      bool                    is_free_buffer_after_forward,
                                                      bool                    is_qk_buf_float,
                                                      bool                    sparse):

    BaseAttentionLayer<T>(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, sparse),
    params_(gpt_init_parameter),
    hidden_units_(gpt_init_parameter.hidden_size_),
    local_head_num_(gpt_init_parameter.head_num_ / tensor_para.world_size_),
    local_head_num_kv_(
        gpt_init_parameter.head_num_kv_ == 1 ? 1 : gpt_init_parameter.head_num_kv_ / tensor_para.world_size_),
    local_hidden_units_(gpt_init_parameter.hidden_size_ / tensor_para.world_size_),
    is_qk_buf_float_(is_qk_buf_float),
    lora_gemm_(std::make_shared<LoraGemm<T>>(stream, allocator, cublas_wrapper)),
    gemm_runner_(std::make_shared<GemmRunner<T>>(stream, allocator, cublas_wrapper, quant_algo)),
    local_layer_head_num_(getLocalParameter(gpt_init_parameter.layer_head_num_, tensor_para.world_size_)),
    local_layer_head_num_kv_(getLocalParameter(gpt_init_parameter.layer_head_num_kv_, tensor_para.world_size_)),
    q_scaling_(1.0f),
    tensor_para_(tensor_para) {
    multi_block_mode_ = UseMultiBlockMode();

    if (params_.quant_algo_->int8_mode_ == 2) {
        abort();
    }
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    tensorrt_llm::kernels::Data_type data_type;
    if constexpr (std::is_same<T, half>::value) {
        data_type = tensorrt_llm::kernels::DATA_TYPE_FP16;
    }
#ifdef ENABLE_BF16
    if constexpr (std::is_same<T, __nv_bfloat16>::value) {
        data_type = tensorrt_llm::kernels::DATA_TYPE_BF16;
    }
#endif

#if (CUDART_VERSION >= 12000)
    use_trt_fmha_ = UseTRTFMHA();
    if (use_trt_fmha_) {
        // Load kernels for contiguous cache and paged kv cache at the same time.
        mFMHARunner.reset(new tensorrt_llm::kernels::FusedMHARunnerV2(
            data_type, local_head_num_, params_.size_per_head_, q_scaling_));
        if (mFMHARunner -> fmha_supported()) {
            FT_LOG_INFO("use TRT fmha");
            // Set flags: force_fp32_acc, is_s_padded, causal_mask, num_kv_heads.
            mFMHARunner->setup_flags(mFMHAForceFP32Acc, !mRemovePadding, params_.is_causal_, local_head_num_kv_);
        }
        else {
            FT_LOG_WARNING("FMHA is disabled for size_per_head %d", params_.size_per_head_);
            use_trt_fmha_ = false;
        }
    }
#endif
    use_open_source_fmha_ = UseOpenSourceFMHA();
    if (use_open_source_fmha_) {
        FT_LOG_INFO("use open source fmha");
    }
}

template<typename T>
bool ParallelAttentionWrapper<T>::UseFMHA()
{
    return use_open_source_fmha_ || use_trt_fmha_; 
}

template<typename T>
ParallelAttentionWrapper<T>::~ParallelAttentionWrapper()
{
    cublas_wrapper_ = nullptr;
    freeBuffer();
}

template<typename T>
void ParallelAttentionWrapper<T>::allocateBuffer()
{
    FT_CHECK(false);
}

template<typename T>
void ParallelAttentionWrapper<T>::allocateBuffer(
    size_t h_token_num, size_t context_batch_size, size_t generate_batch_size, size_t seq_len, size_t seq_len_with_prefix, bool allocate_qk_buf, bool multi_block_mode)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    // const auto type_size = int8_mode_ == 2 ? sizeof(int8_t) : sizeof(T);
    // NOTE (perkzz): use sizeof(T) here for cutlass int8 kernels.
    const auto qkv_hidden_size = local_head_num_ * params_.size_per_head_;
    const auto qkv_merged_size = qkv_hidden_size + 2 * local_head_num_kv_ * params_.size_per_head_;
    qkv_buf_   = (T*)allocator_->reMalloc(qkv_buf_,
                                        sizeof(T) * h_token_num * qkv_merged_size,
                                        true);
    q_buf_2_   = (T*)allocator_->reMalloc(q_buf_2_,
                                        sizeof(T) * context_batch_size * seq_len_with_prefix * qkv_merged_size,
                                        false);
    k_buf_2_   = q_buf_2_ + context_batch_size * seq_len_with_prefix * local_head_num_ * params_.size_per_head_;
    v_buf_2_   = k_buf_2_ + context_batch_size * seq_len_with_prefix * local_head_num_kv_ * params_.size_per_head_;
    qkv_buf_2_ = (T*)allocator_->reMalloc(qkv_buf_2_, sizeof(T) * h_token_num * qkv_hidden_size, false);

    // save memory usage when using fmha
    if (allocate_qk_buf) {
        qk_buf_ = (T*)allocator_->reMalloc(
            qk_buf_, sizeof(T) * context_batch_size * local_head_num_ * seq_len * seq_len_with_prefix, true);
        qkv_buf_3_ =
            (T*)allocator_->reMalloc(qkv_buf_3_, sizeof(T) * context_batch_size * seq_len * qkv_hidden_size, true);
    }
    else {
        softmax_lse_ = (float*)allocator_->reMalloc(
            softmax_lse_, sizeof(float) * context_batch_size * local_head_num_ * params_.max_seq_len_, true);
    }

    if (is_qk_buf_float_ == true) {
        if (allocate_qk_buf) {
            // allocator_->free((void**)(&qk_buf_));
            qk_buf_float_ = (float*)allocator_->reMalloc(qk_buf_float_,
                                                         sizeof(float) * context_batch_size * local_head_num_ * seq_len
                                                             * seq_len_with_prefix,
                                                         true);
            // qk_buf_ = (T *)qk_buf_float_;
        }
    }

    if(multi_block_mode_){
        const int threads_per_value = pow2roundup(params_.size_per_head_) * sizeof(T) / 16;
        max_seq_len_tile_ = 256 / threads_per_value; // for allocate partial output results memory. Regardless to THDS_PER_BLOCK
        partial_out_ = (T*)allocator_->reMalloc(partial_out_, sizeof(T) * max_seq_len_tile_ * generate_batch_size * params_.size_per_head_ * local_head_num_,false);
        partial_sum_ = (float*)allocator_->reMalloc(partial_sum_, sizeof(float) * max_seq_len_tile_ * generate_batch_size * local_head_num_, false);
        partial_max_ = (float*)allocator_->reMalloc(partial_max_, sizeof(float) * max_seq_len_tile_ * generate_batch_size * local_head_num_, false);
        block_counter_ = (int*)allocator_->reMalloc(block_counter_, sizeof(int) * generate_batch_size * local_head_num_, true);
    }
    is_allocate_buffer_ = true;
}

template<typename T>
void ParallelAttentionWrapper<T>::freeBuffer()
{
    if (is_allocate_buffer_) {
        FT_LOG_DEBUG(__PRETTY_FUNCTION__);
        allocator_->free((void**)(&qkv_buf_));
        allocator_->free((void**)(&q_buf_2_));
        allocator_->free((void**)(&qk_buf_));
        allocator_->free((void**)(&qkv_buf_3_));
        allocator_->free((void**)(&qkv_buf_2_));
        allocator_->free((void**)(&softmax_lse_));

        if (is_qk_buf_float_ == true) {
            allocator_->free((void**)(&qk_buf_float_));
        }

        if(multi_block_mode_ == true){
            allocator_->free((void**)(&partial_out_));
            allocator_->free((void**)(&partial_sum_));
            allocator_->free((void**)(&partial_max_));
            allocator_->free((void**)(&block_counter_));
        }

        allocator_->free((void**)(&mixed_gemm_workspace_));
        mixed_gemm_ws_bytes_ = 0;

        allocator_->free((void**)(&int8_gemm_workspace_));
        int8_gemm_ws_bytes_ = 0;

        is_allocate_buffer_ = false;
    }
}

template class ParallelAttentionWrapper<float>;
template class ParallelAttentionWrapper<half>;
#ifdef ENABLE_BF16
template class ParallelAttentionWrapper<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
