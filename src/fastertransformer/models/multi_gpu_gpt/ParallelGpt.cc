#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGpt.h"
#include "src/fastertransformer/kernels/bert_preprocess_kernels.h"
#include "src/fastertransformer/cuda/nvtx/nvtx_utils.h"
#include <algorithm>

namespace fastertransformer {

template<typename T>
void ParallelGpt<T>::initialize()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    quant_algo_                 = tc::QuantAlgo(params_.quant_algo_->int8_mode_,
                                params_.quant_algo_->int4_mode_,
                                params_.quant_algo_->has_zeros_,
                                params_.quant_algo_->weight_only_group_size_);
    parallel_attention_wrapper_ = new ParallelAttentionWrapper<T>(params_,
                                                                  tensor_para_,
                                                                  stream_,
                                                                  cublas_wrapper_,
                                                                  quant_algo_,
                                                                  allocator_,
                                                                  is_free_buffer_after_forward_,
                                                                  is_qk_buf_float_,
                                                                  sparse_);

    // max_seq_len + max_generate_batch_size because max_seq_len >> max_generate_batch_size
    ffn_layer_ = new TensorParallelFfnLayer<T>(params_.max_context_batch_size_,
                                               params_.max_seq_len_ + params_.max_generate_batch_size_,
                                               params_.hidden_size_,
                                               params_.expert_num_,  // expert_num
                                               params_.moe_k_,
                                               params_.inter_size_,
                                               params_.inter_padding_size_,
                                               params_.layer_inter_size_,
                                               params_.layer_inter_padding_size_,
                                               tensor_para_,
                                               stream_,
                                               cublas_wrapper_,
                                               quant_algo_,
                                               allocator_,
                                               true,
                                               is_free_buffer_after_forward_,
                                               sparse_,
                                               params_.is_sparse_head_,
                                               params_.activation_type_,
                                               params_.has_moe_norm_,
                                               params_.layernorm_eps_,
                                               custom_all_reduce_comm_,
                                               enable_custom_all_reduce_);

    norm_wrapper_.reset(
        new NormWrapper<T>(params_.layernorm_type_, params_.norm_type_, T(sqrt(2 * params_.num_layers_))));
}

template<typename T>
void ParallelGpt<T>::preAllocate()
{
    parallel_attention_wrapper_->preAllocate();
    ffn_layer_->preAllocate();
    allocateBuffer(
        params_.max_generate_batch_size_ + params_.max_context_batch_size_, params_.max_seq_len_, false, true);
}

template<typename T>
void ParallelGpt<T>::allocateBuffer()
{
    FT_CHECK(false);
}

template<typename T>
void ParallelGpt<T>::allocateBuffer(size_t total_batch_size, size_t h_token_num, bool reuse_buf, bool pre_attn_ln)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    size_t hidden_units   = params_.hidden_size_;
    decoder_normed_input_ = reinterpret_cast<T*>(
        allocator_->reMalloc(decoder_normed_input_, sizeof(T) * h_token_num * hidden_units, false));
    self_attn_output_ =
        reinterpret_cast<T*>(allocator_->reMalloc(self_attn_output_, sizeof(T) * h_token_num * hidden_units, false));
    if (!reuse_buf) {
        normed_self_attn_output_ = reinterpret_cast<T*>(
            allocator_->reMalloc(normed_self_attn_output_, sizeof(T) * h_token_num * hidden_units, false));
    }
    else {
        normed_self_attn_output_ = decoder_normed_input_;
    }
    if (pre_attn_ln) {
        attn_normed_input_ = reinterpret_cast<T*>(
            allocator_->reMalloc(attn_normed_input_, sizeof(T) * h_token_num * hidden_units, false));
    }
    // only allocate additionl buffers when has adapters
    decoder_layer_output_ = reinterpret_cast<T*>(
        allocator_->reMalloc(decoder_layer_output_, sizeof(T) * h_token_num * hidden_units, false));
    if (params_.quant_algo_->int8_mode_ == 2) {
        FT_LOG_ERROR("int8_mode == 2 not support");
        abort();
    }
    h_pinned_token_num_ptr_ = (size_t*)allocator_->reMalloc(h_pinned_token_num_ptr_, sizeof(size_t), true);
    padding_offset_ = reinterpret_cast<int*>(allocator_->reMalloc(padding_offset_, sizeof(int) * (h_token_num), false));
    cu_seqlens_ =
        reinterpret_cast<int*>(allocator_->reMalloc(cu_seqlens_, sizeof(int) * (total_batch_size + 1), false));
    context_lengths_ =
        reinterpret_cast<int*>(allocator_->reMalloc(context_lengths_, sizeof(int) * (total_batch_size), false));
    sequence_lengths_ =
        reinterpret_cast<int*>(allocator_->reMalloc(sequence_lengths_, sizeof(int) * (total_batch_size), false));
    prefix_lengths_ =
        reinterpret_cast<int*>(allocator_->reMalloc(prefix_lengths_, sizeof(int) * (total_batch_size), false));
    block_pointers_ =
        reinterpret_cast<int64_t*>(allocator_->reMalloc(block_pointers_, sizeof(int64_t) * (2 * params_.num_layers_ * total_batch_size * params_.max_seq_len_ / params_.seq_size_per_block_ + 32), true));
    if (params_.int8_kv_cache_) {
        block_scale_pointers_ =
            reinterpret_cast<int64_t*>(allocator_->reMalloc(block_scale_pointers_, sizeof(int64_t) * (2 * params_.num_layers_ * total_batch_size * params_.max_seq_len_ / params_.seq_size_per_block_ + 32), true));
    }

    // for moe
    expert_scales_ = reinterpret_cast<T*>(
        allocator_->reMalloc(expert_scales_, sizeof(T) * pad_to_multiple_of_16(params_.moe_k_ * h_token_num), false));
    expanded_source_row_to_expanded_dest_row_ = reinterpret_cast<int*>(allocator_->reMalloc(
        expanded_source_row_to_expanded_dest_row_, sizeof(int) * pad_to_multiple_of_16(params_.moe_k_ * h_token_num), false));
    expert_for_source_row_                    = reinterpret_cast<int*>(
        allocator_->reMalloc(expert_for_source_row_, sizeof(int) * pad_to_multiple_of_16(params_.moe_k_ * h_token_num), false));
    fc2_result_ = reinterpret_cast<T*>(
        allocator_->malloc(sizeof(T) * pad_to_multiple_of_16(params_.moe_k_ * h_token_num * hidden_units), false));

    is_allocate_buffer_ = true;
}

template<typename T>
void ParallelGpt<T>::freeBuffer()
{
    if (is_allocate_buffer_) {
        FT_LOG_DEBUG(__PRETTY_FUNCTION__);
        if (normed_self_attn_output_ != decoder_normed_input_) {
            allocator_->free((void**)(&normed_self_attn_output_));
        }
        if (attn_normed_input_) {
            allocator_->free((void**)(&attn_normed_input_));
        }
        allocator_->free((void**)(&decoder_normed_input_));
        allocator_->free((void**)(&self_attn_output_));
        allocator_->free((void**)(&decoder_layer_output_));
        allocator_->free((void**)(&h_pinned_token_num_ptr_));
        allocator_->free((void**)(&padding_offset_));
        allocator_->free((void**)(&cu_seqlens_));
        allocator_->free((void**)(&context_lengths_));
        allocator_->free((void**)(&sequence_lengths_));
        allocator_->free((void**)(&prefix_lengths_));
        allocator_->free((void**)(&block_pointers_));
        allocator_->free((void**)(&block_scale_pointers_));
        if (params_.quant_algo_->int8_mode_ == 2) {
            allocator_->free((void**)(&attention_query_dynamic_scale_));
            allocator_->free((void**)(&ffn_intermediate_dynamic_scale_));
        }
        allocator_->free((void**)(&expert_scales_));
        allocator_->free((void**)(&expanded_source_row_to_expanded_dest_row_));
        allocator_->free((void**)(&expert_for_source_row_));
        allocator_->free((void**)(&fc2_result_));
        is_allocate_buffer_ = false;
    }
}

template<typename T>
bool ParallelGpt<T>::isValidLayerParallelId(uint l)
{
    uint local_num_layer = (uint)(ceil(params_.num_layers_ * 1.0f / pipeline_para_.world_size_));
    return l < params_.num_layers_ && (l >= local_num_layer * pipeline_para_.rank_)
           && (l < local_num_layer * (pipeline_para_.rank_ + 1));
}

template<typename T>
bool ParallelGpt<T>::isFirstLayerParallelId(uint l)
{
    uint local_num_layer = (uint)(ceil(params_.num_layers_ * 1.0f / pipeline_para_.world_size_));
    return l < params_.num_layers_ && (l == local_num_layer * pipeline_para_.rank_);
}

template<typename T>
bool ParallelGpt<T>::isLastLayerParallelId(uint l)
{
    uint local_num_layer = (uint)(ceil(params_.num_layers_ * 1.0f / pipeline_para_.world_size_));
    return l < params_.num_layers_ && (l == local_num_layer * (pipeline_para_.rank_ + 1) - 1);
}

template<typename T>
int ParallelGpt<T>::getFirstLayerParallelId()
{
    uint local_num_layer = (uint)(ceil(params_.num_layers_ * 1.0f / pipeline_para_.world_size_));
    return local_num_layer * pipeline_para_.rank_;
}

template<typename T>
ParallelGpt<T>::ParallelGpt(const GptInitParameter&             gpt_init_parameter,
                            NcclParam                           tensor_para,
                            NcclParam                           pipeline_para,
                            cudaStream_t                        stream,
                            cublasMMWrapper*                    cublas_wrapper,
                            IAllocator*                         allocator,
                            bool                                is_free_buffer_after_forward,
                            bool                                is_qk_buf_float,
                            bool                                sparse,
                            std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm,
                            int                                 enable_custom_all_reduce):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, sparse),
    params_(gpt_init_parameter),
    tensor_para_(tensor_para),
    pipeline_para_(pipeline_para),
    custom_all_reduce_comm_(custom_all_reduce_comm),
    enable_custom_all_reduce_(enable_custom_all_reduce),
    is_qk_buf_float_(is_qk_buf_float)
{
    initialize();
}

template<typename T>
bool ParallelGpt<T>::UseFMHA()
{
    FT_CHECK_WITH_INFO(parallel_attention_wrapper_ != nullptr, "parallel_attention_wrapper_ should not be nullptr");
    return parallel_attention_wrapper_->UseFMHA();
}

template<typename T>
ParallelGpt<T>::~ParallelGpt()
{
    delete parallel_attention_wrapper_;
    delete ffn_layer_;
    freeBuffer();
}

template<typename T>
void ParallelGpt<T>::convert_to_block_pointers(TensorMap* output_tensors,
                                               const TensorMap* input_tensors,
                                               int total_batch_size)
{
    Tensor k_cache         = output_tensors->at("key_cache");
    Tensor v_cache         = output_tensors->at("value_cache");
    size_t kv_cache_offset = 1;
    
    for (auto t = k_cache.shape().begin() + 1; t != k_cache.shape().end(); ++t) {
        kv_cache_offset *= *t;
    };

    Tensor block_index_map      = input_tensors->at("block_index_map");
    uint   max_blocks_per_batch = (uint)(input_tensors->at("block_index_map").shape()[1]);
    int*   block_index          = block_index_map.getPtr<int>();
    block_pointers_vector_.clear();
    block_scale_pointers_vector_.clear();

    block_pointers_vector_.resize(params_.num_layers_ * total_batch_size * max_blocks_per_batch * 2);
    int size_of_type = sizeof(T);
    if (params_.int8_kv_cache_) {
        size_of_type = 1;
        block_scale_pointers_vector_.resize(params_.num_layers_ * total_batch_size * max_blocks_per_batch * 2);
    }

    const size_t index_offset = kv_cache_offset / k_cache.shape()[1] * size_of_type;
    for (uint l = 0; l < params_.num_layers_; l++) {
        const size_t cache_offset  = (l - getFirstLayerParallelId()) * kv_cache_offset;
        char*        layer_k_cache = reinterpret_cast<char*>(k_cache.getPtrWithOffset(cache_offset));
        char*        layer_v_cache = reinterpret_cast<char*>(v_cache.getPtrWithOffset(cache_offset));
        const size_t stride        = total_batch_size * 2 * max_blocks_per_batch;
        for (int i = 0; i < total_batch_size; ++i) {
            for (int j = 0; j < max_blocks_per_batch; ++j) {
                block_pointers_vector_[l * stride + i * 2 * max_blocks_per_batch + j] = int64_t(layer_k_cache + block_index[i * max_blocks_per_batch + j] * index_offset);
                block_pointers_vector_[l * stride + i * 2 * max_blocks_per_batch + max_blocks_per_batch + j] = int64_t(layer_v_cache + block_index[i * max_blocks_per_batch + j] * index_offset);
            }
        }
    }
    cudaMemcpyAsync(block_pointers_, block_pointers_vector_.data(), sizeof(int64_t) * params_.num_layers_ * total_batch_size * max_blocks_per_batch * 2, cudaMemcpyHostToDevice, stream_);
    if (params_.int8_kv_cache_) {
        Tensor k_cache_scale = output_tensors->at("key_cache_scale");
        Tensor v_cache_scale = output_tensors->at("value_cache_scale");
        for (uint l = 0; l < params_.num_layers_; l++) {
            const size_t cache_offset        = (l - getFirstLayerParallelId()) * kv_cache_offset;
            const size_t scale_cache_offset  = cache_offset / params_.size_per_head_;
            const size_t scale_index_offset  = kv_cache_offset / params_.size_per_head_ / k_cache.shape()[1];
            float*       layer_k_scale_cache = k_cache_scale.getPtrWithOffset<float>(scale_cache_offset);
            float*       layer_v_scale_cache = v_cache_scale.getPtrWithOffset<float>(scale_cache_offset);
            const size_t stride = total_batch_size * 2 * max_blocks_per_batch;
            for (int i = 0; i < total_batch_size; ++i) {
                for (int j = 0; j < max_blocks_per_batch; ++j) {
                    block_scale_pointers_vector_[l * stride + i * 2 * max_blocks_per_batch + j] = int64_t(layer_k_scale_cache + block_index[i * max_blocks_per_batch + j] * scale_index_offset);
                    block_scale_pointers_vector_[l * stride + i * 2 * max_blocks_per_batch + max_blocks_per_batch + j] = int64_t(layer_v_scale_cache + block_index[i * max_blocks_per_batch + j] * scale_index_offset);
                }
            }
        }
        cudaMemcpyAsync(block_scale_pointers_, block_scale_pointers_vector_.data(), sizeof(int64_t) * params_.num_layers_ * total_batch_size * max_blocks_per_batch * 2, cudaMemcpyHostToDevice, stream_);
    }
}

template<typename T>
void ParallelGpt<T>::forward(TensorMap*                                            output_tensors,
                             const TensorMap*                                      input_tensors,
                             const std::vector<ParallelGptDecoderLayerWeight<T>*>* gpt_decoder_layer_weight)
{
    // input tensors:
    //      decoder_input [batch_size + context_batch_size, hidden_dimension],
    //      attention_mask [context_batch_size, 1, seq_len, seq_len]
    //      input_lengths [batch_size + context_batch_size]
    //      sequence_lengths [batch_size]
    //      block_index_map [batch_size + context_batch_size, max_block_size]
    //      linear_bias_slopes [head_num], optional

    // output tensors:
    //      decoder_output [batch_size + context_batch_size, hidden_dimension]
    //      key_cache [num_layer, batch_size, head_num, params_.size_per_head_ // x, memory_len, x]
    //      value_cache [num_layer, batch_size, head_num, memory_len, params_.size_per_head_]

    FT_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    FT_CHECK(input_tensors->isExist("decoder_input"));
    FT_CHECK(input_tensors->isExist("input_lengths"));
    FT_CHECK(input_tensors->isExist("block_index_map"));
    FT_CHECK(output_tensors->isExist("decoder_output"));
    FT_CHECK(input_tensors->isExist("lora_ids"));
    FT_CHECK(input_tensors->isExist("lora_input_lengths"));


    Tensor decoder_input_tensor = input_tensors->at("decoder_input");
    size_t hidden_units         = params_.hidden_size_;
    FT_CHECK(decoder_input_tensor.shape()[1] == hidden_units);
    const size_t total_batch_size = input_tensors->at("input_lengths").shape()[0];
    size_t       batch_size       = 0;
    if (input_tensors->isExist("sequence_lengths")) {
        batch_size = input_tensors->at("sequence_lengths").shape()[0];
    }
    const size_t   h_token_num = decoder_input_tensor.shape()[0];
    const DataType data_type   = getTensorType<T>();
    const bool     use_kvcache = output_tensors->isExist("key_cache") && output_tensors->isExist("value_cache");

    PUSH_RANGE(stream_, "buffer allocation");
    bool reuse_buf   = !params_.use_norm_input_residual_;
    bool pre_attn_ln = gpt_decoder_layer_weight->at(0)->pre_attn_layernorm_weights.gamma;
    allocateBuffer(total_batch_size, h_token_num, reuse_buf, pre_attn_ln);
    POP_RANGE;

    const size_t context_batch_size = total_batch_size - batch_size;
    int*         input_lengths      = input_tensors->getPtr<int>("input_lengths");
    cudaMemcpyAsync(context_lengths_, input_lengths, sizeof(int) * total_batch_size, cudaMemcpyHostToDevice, stream_);
    size_t max_input_length = 0;
    size_t max_context_seq_length = 0;
    size_t step             = 0;
    if (context_batch_size) {
        max_context_seq_length = *std::max_element(input_lengths + batch_size, input_lengths + total_batch_size);
        if (input_tensors->isExist("attention_mask")) {
            FT_CHECK(input_tensors->at("attention_mask").shape()[0] == context_batch_size);
        }
    }
    if (batch_size) {
        int* sequence_lengths = input_tensors->getPtr<int>("sequence_lengths");
        cudaMemcpyAsync(sequence_lengths_, sequence_lengths, sizeof(int) * batch_size, cudaMemcpyHostToDevice, stream_);
        max_input_length = *std::max_element(input_lengths, input_lengths + batch_size);
        step             = *std::max_element(sequence_lengths, sequence_lengths + batch_size);
        step             = step + 1;
    }

    size_t min_prefix_length = 0;
    if (input_tensors->isExist("d_prefix_prompt_lengths")) {
        int *d_prefix_prompt_lengths = input_tensors->getPtr<int>("d_prefix_prompt_lengths");
        cudaMemcpyAsync(prefix_lengths_, d_prefix_prompt_lengths, sizeof(int) * total_batch_size,
                        cudaMemcpyHostToDevice, stream_);
        if (input_tensors->getVal<bool>("count_prefix_length")) {
            min_prefix_length =
                *std::min_element(d_prefix_prompt_lengths + batch_size, d_prefix_prompt_lengths + total_batch_size);
        }
    }

    size_t kv_cache_offset = 0;
    uint   max_blocks_per_batch = 0;
    size_t block_stride = 0;
    if (use_kvcache) {
        convert_to_block_pointers(output_tensors, input_tensors, total_batch_size);
        Tensor k_cache = output_tensors->at("key_cache");
        for (auto t = k_cache.shape().begin() + 1; t != k_cache.shape().end(); ++t) {
            kv_cache_offset *= *t;
        }
        max_blocks_per_batch = (uint)(input_tensors->at("block_index_map").shape()[1]);
        block_stride = total_batch_size * 2 * max_blocks_per_batch;
    }    

    const auto activation_in_type  = params_.quant_algo_->int8_mode_ == 2 ? TYPE_INT8 : data_type;
    const auto activation_out_type = data_type;

    size_t context_h_token_num = h_token_num - batch_size;
    if (context_batch_size) {
        PUSH_RANGE(stream_, "remove padding");
        invokeGetPaddingOffsetAndCuSeqLens(
            h_pinned_token_num_ptr_,
            padding_offset_,
            cu_seqlens_,
            context_lengths_ + batch_size,
            context_batch_size,
            max_context_seq_length,
            stream_);
        FT_CHECK_WITH_INFO(context_h_token_num>0, "input should not be empty");
        POP_RANGE;
    }
    PUSH_RANGE(stream_, "context_generation");
    for (uint l = 0; l < params_.num_layers_; l++) {
        PUSH_RANGE(stream_, fmtstr("layer_%u", l));
        bool use_moe = std::find(params_.moe_layer_index_.begin(), params_.moe_layer_index_.end(), l) != params_.moe_layer_index_.end();
        if (isValidLayerParallelId(l) == false) {
            POP_RANGE;  // escape for NVTX Range: layer_%u
            continue;
        }
        ParallelGptDecoderLayerWeight<T>* layer_weight = gpt_decoder_layer_weight->at(l);
        T* decoder_input  = (l == 0) ? decoder_input_tensor.getPtr<T>() : decoder_layer_output_;
        T* decoder_output = decoder_layer_output_;
        sync_check_cuda_error();

        print_bsd(l, "decoder input", decoder_input, 1, h_token_num, hidden_units);
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
        sync_check_cuda_error();

        PUSH_RANGE(stream_, "pre-mha layernorm");
        norm_wrapper_->initDecoderLayerNorm(decoder_normed_input_,
                                            decoder_input,
                                            layer_weight->pre_layernorm_weights.gamma,
                                            layer_weight->pre_layernorm_weights.beta,
                                            params_.layernorm_eps_,
                                            h_token_num,
                                            hidden_units,
                                            const_cast<float*>(layer_weight->self_attention_weights.query_weight.scale),
                                            nullptr,
                                            params_.quant_algo_->int8_mode_,
                                            stream_);
        print_bsd(l, "pre ln", decoder_normed_input_, 1, h_token_num, hidden_units);
        sync_check_cuda_error();

        if (pre_attn_ln) {
            norm_wrapper_->preAttentionLayerNorm(attn_normed_input_,
                                                 decoder_input,
                                                 layer_weight->pre_attn_layernorm_weights.gamma,
                                                 layer_weight->pre_attn_layernorm_weights.beta,
                                                 params_.layernorm_eps_,
                                                 h_token_num,
                                                 hidden_units,
                                                 nullptr,
                                                 nullptr,
                                                 params_.quant_algo_->int8_mode_,
                                                 stream_);
            print_bsd(l, "pre attn ln", attn_normed_input_, 1, h_token_num, hidden_units);
        }

        sync_check_cuda_error();
        POP_RANGE;
        
        size_t   cache_offset = (l - getFirstLayerParallelId()) * kv_cache_offset;
        const T* input_query  = nullptr;
        if (pre_attn_ln) {
            input_query = attn_normed_input_;
        }
        else if (params_.layernorm_type_ == LayerNormType::pre_layernorm) {
            input_query = decoder_normed_input_;
        }
        else {
            input_query = decoder_input;
        }
        TensorMap attention_input_tensors{
            {"input_query", Tensor{MEMORY_GPU, activation_in_type, {h_token_num, hidden_units}, input_query}},
            {"use_kvcache", Tensor{MEMORY_CPU, TYPE_BOOL, {(size_t)1}, &use_kvcache}},
            {"block_pointers",
             Tensor{MEMORY_GPU, TYPE_INT64, {total_batch_size, 1, 2, max_blocks_per_batch}, block_pointers_ + l * block_stride}},
            {"host_block_pointers", Tensor{MEMORY_CPU, TYPE_INT64, {total_batch_size, 1, 2, max_blocks_per_batch}, block_pointers_vector_.data() + l * block_stride}},
            {"block_scale_pointers",
             Tensor{MEMORY_GPU, TYPE_INT64, {total_batch_size, 1, 2, max_blocks_per_batch}, block_scale_pointers_ + l * block_stride}},
            {"layer_id", Tensor{MEMORY_CPU, TYPE_INT32, {(size_t)1}, &l}},
            {"generate_batch_size", Tensor{MEMORY_CPU, TYPE_INT32, {(size_t)1}, &batch_size}},
            {"context_batch_size", Tensor{MEMORY_CPU, TYPE_INT32, {(size_t)1}, &context_batch_size}},
            {"max_input_length", Tensor{MEMORY_CPU, TYPE_INT32, {(size_t)1}, &max_input_length}},
            {"max_context_seq_length", Tensor{MEMORY_CPU, TYPE_INT32, {(size_t)1}, &max_context_seq_length}},
            {"step", Tensor{MEMORY_CPU, TYPE_INT32, {(size_t)1}, &step}},
            {"sequence_lengths", Tensor{MEMORY_GPU, TYPE_INT32, {batch_size}, sequence_lengths_}},
            {"input_lengths", Tensor{MEMORY_GPU, TYPE_INT32, {total_batch_size}, context_lengths_}},
            {"lora_ids", input_tensors->at("lora_ids")},
            {"lora_input_lengths", input_tensors->at("lora_input_lengths")}};

        if (context_batch_size) {
            if (input_tensors->isExist("attention_mask")) {
                const T* attention_ptr    = input_tensors->at("attention_mask").getPtr<T>();
                auto     attention_tensor = input_tensors->at("attention_mask");
                attention_input_tensors.insert(
                    "attention_mask", 
                    Tensor{MEMORY_GPU,
                    data_type,
                    {context_batch_size, 1, attention_tensor.shape()[1], attention_tensor.shape()[2]},attention_ptr}
                );
            }
            attention_input_tensors.insert("padding_offset",
                                           Tensor{MEMORY_GPU, TYPE_INT32, {context_h_token_num}, padding_offset_});
            attention_input_tensors.insert(
                "cu_seqlens", Tensor{MEMORY_GPU, TYPE_INT32, {size_t(context_batch_size + 1)}, cu_seqlens_});
        }
        if (input_tensors->isExist("linear_bias_slopes")) {
            attention_input_tensors.insert("linear_bias_slopes", input_tensors->at("linear_bias_slopes"));
        }
        if (input_tensors->isExist("position_ids")) {
            attention_input_tensors.insert("position_ids", input_tensors->at("position_ids"));
        }
        if (input_tensors->isExist("d_prefix_prompt_lengths")) {
            attention_input_tensors.insert(
                    "max_prefix_prompt_length", input_tensors->at("max_prefix_prompt_length")
            );
            attention_input_tensors.insert("d_prefix_prompt_lengths",
                                           Tensor{MEMORY_GPU, TYPE_INT32, {total_batch_size}, prefix_lengths_});
            attention_input_tensors.insert("count_prefix_length", input_tensors->at("count_prefix_length"));
            attention_input_tensors.insert("min_prefix_length",
                                           Tensor{MEMORY_CPU, TYPE_INT32, {1}, &min_prefix_length});
        }
        TensorMap attention_output_tensors{
            {"hidden_features",
             Tensor(MEMORY_GPU, activation_out_type, {h_token_num, hidden_units}, self_attn_output_)}};

        if (params_.is_sparse_head_ && params_.layer_head_num_[l] == 0) {
            check_cuda_error(cudaMemcpyAsync(self_attn_output_,
                                             input_query,
                                             sizeof(T) * h_token_num * hidden_units,
                                             cudaMemcpyDeviceToDevice,
                                             stream_));
        }
        else {
            parallel_attention_wrapper_->forward(
                &attention_output_tensors, &attention_input_tensors, &layer_weight->self_attention_weights);
        }

        print_bsd(l, "attn out", self_attn_output_, 1, h_token_num, hidden_units);

        // the adapter after attention (only pre layernorm currently)
        PUSH_RANGE(stream_, "post_mha_ln");
        T* input_residual = nullptr;
        if (!layer_weight->self_attn_layernorm_weights.gamma) {
            // falcon7b
            // output = attn(norm1(in)) + mlp(norm1(in)) + in
            // falcon40b
            // output = attn(norm2(in)) + mlp(norm1(in)) + in
            input_residual = decoder_input;
            std::swap(normed_self_attn_output_, decoder_normed_input_);
        }
        else {
            norm_wrapper_->attentionAddBiasResidualLayerNorm(
                self_attn_output_,
                normed_self_attn_output_,
                self_attn_output_,
                params_.use_norm_input_residual_ ? decoder_normed_input_ : decoder_input,
                layer_weight->self_attn_layernorm_weights.gamma,
                layer_weight->self_attn_layernorm_weights.beta,
                layer_weight->self_attention_weights.attention_output_weight.bias,
                params_.layernorm_eps_,
                h_token_num,
                hidden_units,
                nullptr,
                nullptr,
                const_cast<float*>(layer_weight->ffn_weights.intermediate_weight.scale),
                nullptr,  // NOTE (perkzz): dynamic_quant_ ? ffn_intermediate_dynamic_scale_ : nullptr,
                params_.quant_algo_->int8_mode_,
                stream_);
        }
        sync_check_cuda_error();
        POP_RANGE;

        T* ffn_output_ptr =  params_.layernorm_type_ == LayerNormType::pre_layernorm ? decoder_normed_input_ : decoder_output;

        int ffn_batch_size_lora = batch_size + context_batch_size;
        const int* lora_input_lengths = input_tensors->getPtr<int>("lora_input_lengths", nullptr);;

        print_bsd(l, "before ffn", params_.layernorm_type_ == LayerNormType::pre_layernorm ? normed_self_attn_output_ : self_attn_output_, 1, h_token_num, hidden_units);

        TensorMap ffn_input_tensors(
            {{"ffn_input",
              Tensor{MEMORY_GPU,
                     activation_in_type,
                     {h_token_num, hidden_units},
                     params_.layernorm_type_ == LayerNormType::pre_layernorm ? normed_self_attn_output_ :
                                                                               self_attn_output_}},
             {"layer_id", Tensor{MEMORY_CPU, TYPE_INT32, {(size_t)1}, &l}},
             {"lora_ids", input_tensors->at("lora_ids")},
             {"lora_input_lengths", Tensor{MEMORY_GPU, TYPE_INT32, {total_batch_size}, lora_input_lengths}},
             {"batch_size", Tensor{MEMORY_CPU, TYPE_INT32, {(size_t)1}, &ffn_batch_size_lora}}});
        TensorMap ffn_output_tensors;
        size_t    moe_k = params_.moe_k_;
        ffn_output_tensors.insert("ffn_output",
                                  Tensor{MEMORY_GPU, activation_out_type, {h_token_num, hidden_units}, ffn_output_ptr});
        if (use_moe) {
            ffn_output_tensors.insert(
                "fc2_result",
                Tensor{MEMORY_GPU, activation_out_type, {moe_k * h_token_num, hidden_units}, fc2_result_});
            ffn_output_tensors.insert("expert_scales",
                                      Tensor{MEMORY_GPU, activation_out_type, {h_token_num, moe_k}, expert_scales_});
            ffn_output_tensors.insert(
                "expanded_source_row_to_expanded_dest_row",
                Tensor{MEMORY_GPU, TYPE_INT32, {h_token_num, moe_k}, expanded_source_row_to_expanded_dest_row_});
            ffn_output_tensors.insert("expert_for_source_row",
                                      Tensor{MEMORY_GPU, TYPE_INT32, {h_token_num, moe_k}, expert_for_source_row_});
        }
        ffn_layer_->forward(&ffn_output_tensors, &ffn_input_tensors, &layer_weight->ffn_weights);

        print_bsd(l, "post ffn", ffn_output_ptr, 1, h_token_num, hidden_units);

        // the adapter after ffn (only pre layernorm currently)
        PUSH_RANGE(stream_, "post ffn");

        norm_wrapper_->ffnAddBiasResidualLayerNorm(decoder_output,
                                                   params_.use_norm_attn_out_residual_ ? normed_self_attn_output_ :
                                                                                         self_attn_output_,
                                                   ffn_output_ptr,
                                                   input_residual,
                                                   layer_weight->ffn_weights.output_weight.bias,
                                                   layer_weight->posf_ffn_layernorm_weights.gamma,
                                                   layer_weight->posf_ffn_layernorm_weights.beta,
                                                   params_.layernorm_eps_,
                                                   h_token_num,
                                                   hidden_units,
                                                   nullptr,
                                                   nullptr,
                                                   stream_);

        sync_check_cuda_error();
        POP_RANGE;

        // PUSH_RANGE(stream_, "Nccl send");
        if (isLastLayerParallelId(l) == true && (pipeline_para_.rank_ != pipeline_para_.world_size_ - 1)) {
            const int data_size = h_token_num * hidden_units / tensor_para_.world_size_;
            ftNcclSend(decoder_output + data_size * tensor_para_.rank_,
                       data_size,
                       pipeline_para_.rank_ + 1,
                       pipeline_para_,
                       stream_);
        }
        // POP_RANGE;

        POP_RANGE;
    }
    POP_RANGE;

    // PUSH_RANGE(stream_, "Rebuild padding");
    T* base_ptr = output_tensors->at("decoder_output").getPtr<T>();
    cudaD2Dcpy(base_ptr, decoder_layer_output_, h_token_num * hidden_units);
    sync_check_cuda_error();
    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
    FT_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
    final_check_error();
}

template class ParallelGpt<float>;
template class ParallelGpt<half>;
#ifdef ENABLE_BF16
template class ParallelGpt<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
