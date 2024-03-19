#include "maga_transformer/cpp/components/ParallelModelWrapper.h"
#include "maga_transformer/cpp/common/cuda_resources.h"
#include "src/fastertransformer/kernels/bert_preprocess_kernels.h"
#include "src/fastertransformer/kernels/gen_relative_pos_bias.h"
#include "src/fastertransformer/kernels/gpt_kernels.h"
#include "src/fastertransformer/th_op/GptCommonInputs.h"
#include "src/fastertransformer/utils/compiler_config.h"
#include <algorithm>
#include <math.h>
#include <memory>

using namespace rtp_llm;
using namespace torch_ext;
using namespace std;
namespace th = torch;

namespace rtp_llm {

template<typename T>
void ParallelModelWrapperImpl<T>::initialize()
{
    auto& cuda_resources = CudaResourcesSingleton::getInstance();
    cublas_wrapper_      = std::move(cuda_resources.newCublasWrapper());

    if (std::is_same<T, half>::value) {
        cublas_wrapper_->setGemmConfig(CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F);
    }

    else if constexpr (std::is_same<T, __nv_bfloat16>::value && CompileConfig::enable_bf16) {
        cublas_wrapper_->setBF16GemmConfig();
    }

    else if (std::is_same<T, float>::value) {
        cublas_wrapper_->setFP32GemmConfig();
    }

    parallel_word_embedding_wrapper_.reset(new ParallelWordEmbeddingWrapper<T>(params_,
                                                                               tensor_para_,
                                                                               stream_,
                                                                               cublas_wrapper_.get(),
                                                                               allocator_,
                                                                               true,
                                                                               &weights_->embedding_table,
                                                                               &weights_->position_encoding_table));
    parallel_gpt_decoder_.reset(new ParallelGpt<T>(params_,
                                                   tensor_para_,
                                                   pipeline_para_,
                                                   stream_,
                                                   cublas_wrapper_.get(),
                                                   allocator_,
                                                   false,
                                                   true,
                                                   &weights_->gpt_layer_weights,
                                                   &weights_->pre_layernorm_weights,
                                                   &weights_->post_layernorm_weights,
                                                   false,
                                                   nullptr,
                                                   0));
    parallel_logits_wrapper_.reset(new ParallelLogitsWrapper<T>(
        params_, tensor_para_, stream_, cublas_wrapper_.get(), allocator_, true, &weights_->embedding_table));

    if (params_.use_attention_linear_bias_) {
        linear_bias_slopes_ =
            reinterpret_cast<T*>(allocator_->reMalloc(linear_bias_slopes_, sizeof(T) * params_.head_num_, true));
        invokeBuildAlibiSlopes(linear_bias_slopes_, params_.head_num_, stream_);
    }
}

template<typename T>
ParallelModelWrapperImpl<T>::ParallelModelWrapperImpl(
    const GptInitParameter&                                         gpt_init_parameter,
    const int                                                       tensor_para_size,
    const std::string&                                              master_ip,
    const int                                                       master_port,
    const std::unordered_map<std::string, ft::Tensor>&              global_weights,
    const std::vector<std::unordered_map<std::string, ft::Tensor>>& layer_weights,
    const std::vector<std::unordered_map<std::string, ft::Tensor>>& layer_int8_weights,
    const std::vector<std::unordered_map<std::string, ft::Tensor>>& layer_int8_scales):
    params_(gpt_init_parameter), data_type_(getTensorType<T>())
{
    // std::cout << "ss1:" << std::endl;

    auto& cuda_resources = CudaResourcesSingleton::getInstance();
    allocator_           = cuda_resources.cuda_allocator.get();
    stream_              = cuda_resources.stream;
    ftNcclInitialize(tensor_para_, pipeline_para_, tensor_para_size, 1, master_ip, master_port);
    // std::cout << "ss2:" << std::endl;
    weights_.reset(new GptWeights<T>(pipeline_para_.world_size_,
                                     pipeline_para_.rank_,
                                     params_.num_layers_,
                                     params_.int8_mode_,
                                     global_weights,
                                     layer_weights,
                                     layer_int8_weights,
                                     layer_int8_scales));
    initialize();
    // std::cout << "ss3:" << std::endl;
}

template<typename T>
ParallelModelWrapperImpl<T>::~ParallelModelWrapperImpl()
{
    ftNcclParamDestroy(tensor_para_);
    ftNcclParamDestroy(pipeline_para_);
    freeBuffer();
}

template<typename T>
void ParallelModelWrapperImpl<T>::allocateBuffer(size_t total_batch_size, size_t h_token_num)
{
    // block_pointers_ = reinterpret_cast<int64_t*>(allocator_->reMalloc(
    //     block_pointers_,
    //     sizeof(int64_t)
    //         * (2 * params_.num_layers_ * total_batch_size * params_.max_seq_len_ / params_.seq_size_per_block_ + 32),
    //     true));
    // if (params_.int8_kv_cache_) {
    //     block_scale_pointers_ = reinterpret_cast<int64_t*>(allocator_->reMalloc(
    //         block_scale_pointers_,
    //         sizeof(int64_t)
    //             * (2 * params_.num_layers_ * total_batch_size * params_.max_seq_len_ / params_.seq_size_per_block_
    //                + 32),
    //         true));
    // }
}

template<typename T>
void ParallelModelWrapperImpl<T>::freeBuffer()
{
    allocator_->free((void**)linear_bias_slopes_);
    // allocator_->free((void**)block_pointers_);
    // allocator_->free((void**)block_scale_pointers_);
}

// template<typename T>
// KVBlockArray ParallelModelWrapperImpl<T>::convert_to_kv_block_array(const th::Tensor& kv_cache_blocks,
//                                                         const th::Tensor& kv_cache_scales)
// {
//     assert(kv_cache_blocks.size(0) == params_.num_layers_);
//     uint batch_size           = kv_cache_blocks.size(1);
//     uint max_blocks_per_batch = kv_cache_blocks.size(2);
//     cudaMemcpyAsync(block_pointers_,
//                     kv_cache_blocks.data_ptr<int64_t>(),
//                     sizeof(int64_t) * params_.num_layers_ * batch_size * max_blocks_per_batch * 2,
//                     cudaMemcpyHostToDevice,
//                     stream_);
//     if (params_.int8_kv_cache_) {
//         assert(kv_cache_scales.size(0) == params_.num_layers_);
//         assert(kv_cache_scales.size(1) == batch_size);
//         assert(kv_cache_scales.size(2) == max_blocks_per_batch);
//         cudaMemcpyAsync(block_scale_pointers_,
//                         kv_cache_scales.data_ptr<int64_t>(),
//                         sizeof(int64_t) * params_.num_layers_ * batch_size * max_blocks_per_batch * 2,
//                         cudaMemcpyHostToDevice,
//                         stream_);
//     }
//     KVBlockArray kv_block_array(batch_size, max_blocks_per_batch, params_.seq_size_per_block_, 0);
//     kv_block_array.data  = block_pointers_;
//     kv_block_array.scale = block_scale_pointers_;
//     return kv_block_array;
// }

template<typename T>
Tensor ParallelModelWrapperImpl<T>::getPositionsId(const size_t      h_token_num,
                                                   const uint        generate_batch_size,
                                                   const uint        context_batch_size,
                                                   const th::Tensor& input_lengths,
                                                   const th::Tensor& sequence_lengths)
{
    if (!params_.has_positional_encoding_) {
        return Tensor();
    }
    Tensor position_ids(MEMORY_GPU, DataType::TYPE_INT32, {h_token_num}, allocator_, true);
    position_ids_.clear();
    position_ids_.reserve(h_token_num);
    for (auto i = 0; i < generate_batch_size; ++i) {
        position_ids_.push_back(*(sequence_lengths.data_ptr<int>() + i) - 1);
    }
    for (auto i = 0; i < context_batch_size; ++i) {
        for (auto j = 0; j < *(sequence_lengths.data_ptr<int>() + generate_batch_size + i); ++j) {
            position_ids_.push_back(j);
        }
    }
    assert(position_ids_.size() == h_token_num);
    cudaMemcpyAsync(
        position_ids.getPtr<int>(), position_ids_.data(), sizeof(int) * h_token_num, cudaMemcpyHostToDevice, stream_);
    return position_ids;
}

template<typename T>
Tensor ParallelModelWrapperImpl<T>::genAttentionMask(const Tensor& input_lengths,
                                                     const uint    context_batch_size,
                                                     const uint    max_context_seq_length)
{
    // TODO: support prefix
    Tensor attention_mask(
        MEMORY_GPU, data_type_, {context_batch_size, max_context_seq_length, max_context_seq_length}, allocator_, true);
    // std::cout << "aaaaa:" << attention_mask.getPtr<T>() << "|" << input_lengths.getPtr<int>() << "|"
    //           << context_batch_size << "|" << max_context_seq_length << std::endl;
    if (params_.rotary_embedding_style_ == 2) {
        invokeBuildGlmDecoderAttentionMask(attention_mask.getPtr<T>(),
                                           input_lengths.getPtr<int>(),
                                           context_batch_size,
                                           max_context_seq_length,
                                           stream_);
    }
    else {
        invokeBuildDecoderAttentionMask(attention_mask.getPtr<T>(),
                                        input_lengths.getPtr<int>(),
                                        nullptr,  // prefix_prompt_lengths
                                        context_batch_size,
                                        max_context_seq_length,
                                        0,  // max_prompt_length
                                        stream_);
    }
    return attention_mask;
}

template<typename T>
Tensor ParallelModelWrapperImpl<T>::calculate_loss(const Tensor& all_hidden_states)
{
    return all_hidden_states;
}

template<typename T>
std::unique_ptr<GptCommonInputs> ParallelModelWrapperImpl<T>::prepareGptCommonInputs(const ModelRequest& model_request)
{

    const size_t h_token_num      = model_request.combo_tokens.size(0);
    const uint   total_batch_size = model_request.generate_batch_size + model_request.context_batch_size;
    assert(params_.head_num_ % tensor_para_.world_size_ == 0);
    const int    local_head_num = params_.head_num_ / tensor_para_.world_size_;
    const size_t hidden_units   = params_.head_num_ * params_.size_per_head_;
    allocateBuffer(total_batch_size, h_token_num);
    // std::cout << "bb:" << std::endl;

    // assert(token_num == model_request.context_batch_size + model_request.generate_batch_size);
    const size_t context_h_token_num = h_token_num - model_request.generate_batch_size;
    assert(model_request.kv_cache_blocks.size(0) == params_.num_layers_);
    assert(model_request.kv_cache_blocks.size(1) == total_batch_size);
    const uint max_blocks_per_batch = model_request.kv_cache_blocks.size(3);

    Tensor kv_cache_blocks(MEMORY_GPU,
                           DataType::TYPE_INT64,
                           {(size_t)params_.num_layers_, total_batch_size, 2, max_blocks_per_batch},
                           allocator_,
                           true);
    Tensor kv_cache_scales(MEMORY_GPU,
                           DataType::TYPE_INT64,
                           {(size_t)params_.num_layers_, total_batch_size, 2, max_blocks_per_batch},
                           allocator_,
                           true);

    Tensor all_hidden_states(MEMORY_GPU, data_type_, {h_token_num, hidden_units}, allocator_, true);
    Tensor padding_offset(MEMORY_GPU, DataType::TYPE_INT32, {context_h_token_num}, allocator_, true);
    Tensor cu_seqlens(MEMORY_GPU, DataType::TYPE_INT32, {model_request.context_batch_size + 1}, allocator_, true);
    Tensor input_lengths(MEMORY_GPU, DataType::TYPE_INT32, {total_batch_size}, allocator_, true);
    Tensor sequence_lengths(MEMORY_GPU, DataType::TYPE_INT32, {model_request.generate_batch_size}, allocator_, true);
    Tensor prefix_prompt_lengths(MEMORY_GPU, DataType::TYPE_INT32, {total_batch_size}, allocator_, true);
    Tensor lora_ids(MEMORY_GPU, DataType::TYPE_INT32, {(size_t)model_request.lora_ids.size(0)}, allocator_, true);
    // std::cout << "cc:" << std::endl;

    cudaMemcpyAsync(input_lengths.getPtr<int>(),
                    model_request.input_lengths.data_ptr<int>(),
                    input_lengths.sizeBytes(),
                    cudaMemcpyHostToDevice,
                    stream_);
    cudaMemcpyAsync(sequence_lengths.getPtr<int>(),
                    model_request.sequence_lengths.data_ptr<int>(),
                    sequence_lengths.sizeBytes(),
                    cudaMemcpyHostToDevice,
                    stream_);
    cudaMemcpyAsync(kv_cache_blocks.getPtr<int>(),
                    model_request.kv_cache_blocks.data_ptr<int64_t>(),
                    kv_cache_blocks.sizeBytes(),
                    cudaMemcpyHostToDevice,
                    stream_);
    if (params_.int8_kv_cache_) {
        cudaMemcpyAsync(kv_cache_scales.getPtr<int64_t>(),
                        model_request.kv_cache_scales.data_ptr<int64_t>(),
                        kv_cache_scales.sizeBytes(),
                        cudaMemcpyHostToDevice,
                        stream_);
    }
    if (model_request.lora_ids.numel() > 0) {
        cudaMemcpyAsync(lora_ids.getPtr<int>(),
                        model_request.lora_ids.data_ptr<int>(),
                        sizeof(int) * model_request.lora_ids.size(0),
                        cudaMemcpyHostToDevice,
                        stream_);
    }
    if (model_request.prefix_lengths.numel() > 0) {
        assert(model_request.prefix_lengths.numel() == total_batch_size);
        cudaMemcpyAsync(prefix_prompt_lengths.getPtr<int>(),
                        model_request.prefix_lengths.data_ptr<int>(),
                        sizeof(int) * prefix_prompt_lengths.size(),
                        cudaMemcpyHostToDevice,
                        stream_);
    }

    // std::cout << "dd:" << model_request.generate_batch_size << total_batch_size << std::endl;
    std::unique_ptr<GptCommonInputs> args = make_unique<GptCommonInputs>();
    if (model_request.context_batch_size) {
        const int max_context_seq_length =
            *std::max_element(model_request.input_lengths.data_ptr<int>() + model_request.generate_batch_size,
                              model_request.input_lengths.data_ptr<int>() + total_batch_size);
        args->max_context_seq_length = max_context_seq_length;
        // std::cout << "dd1:" << int64_t(cu_seqlens.getPtr<int>()) << "|" << max_context_seq_length << std::endl;
        setPaddingOffsetAndCuSeqLens(args.get(),
                                     context_h_token_num,
                                     model_request.context_batch_size,
                                     max_context_seq_length,
                                     model_request.input_lengths.data_ptr<int>() + model_request.generate_batch_size);
        args->attention_mask =
            genAttentionMask(input_lengths, model_request.context_batch_size, max_context_seq_length);
    }
    args->generate_batch_size   = model_request.generate_batch_size;
    args->context_batch_size    = model_request.context_batch_size;
    args->input_lengths         = input_lengths;
    args->sequence_lengths      = sequence_lengths;
    args->prefix_prompt_lengths = prefix_prompt_lengths;
    args->count_prefix_length   = *(model_request.count_length.data_ptr<bool>());
    args->max_generate_seq_length =
        *std::max_element(model_request.sequence_lengths.data_ptr<int>(),
                          model_request.sequence_lengths.data_ptr<int>() + model_request.generate_batch_size);
    args->max_prefix_length =
        *std::max_element(model_request.prefix_lengths.data_ptr<int>(),
                          model_request.prefix_lengths.data_ptr<int>() + model_request.generate_batch_size);
    args->lora_ids        = lora_ids;
    args->kv_cache_blocks = kv_cache_blocks;
    args->kv_cache_scales = kv_cache_scales;
    if (params_.use_attention_linear_bias_) {
        args->linear_bias_slopes = Tensor{MEMORY_GPU,
                                          data_type_,
                                          {(size_t)local_head_num},
                                          linear_bias_slopes_ + tensor_para_.rank_ * local_head_num};
    }
    return args;
}

template<typename T>
void ParallelModelWrapperImpl<T>::setPaddingOffsetAndCuSeqLens(GptCommonInputs* inputs,
                                                               const uint       context_h_token_num,
                                                               const uint       context_batch_size,
                                                               const uint       max_context_seq_length,
                                                               const int*       input_lengths)
{
    Tensor padding_offset(MEMORY_GPU, DataType::TYPE_INT32, {context_h_token_num}, allocator_, true);
    Tensor cu_seqlens(MEMORY_GPU, DataType::TYPE_INT32, {context_batch_size + 1}, allocator_, true);
    std::unique_ptr<GptCommonInputs> gpt_common_inputs = make_unique<GptCommonInputs>();
    padding_offset_.resize(context_h_token_num);
    cu_seqlens_.resize(context_batch_size + 1);
    // int* input_lengths_ = model_request.input_lengths.data_ptr<int>() + model_request.generate_batch_size;
    int total_seq_len = 0;
    int cum_offset    = 0;
    int index         = 0;
    for (int i = 0; i < context_batch_size; ++i) {
        const int seq_len = input_lengths[i];
        cu_seqlens_[i]    = total_seq_len;
        for (int j = 0; j < seq_len; j++) {
            padding_offset_[index] = cum_offset;
            index++;
        }
        cum_offset += max_context_seq_length - seq_len;
        total_seq_len += seq_len;
    }
    cu_seqlens_[context_batch_size] = total_seq_len;
    cudaMemcpyAsync(padding_offset.getPtr<int>(),
                    padding_offset_.data(),
                    sizeof(int) * padding_offset.size(),
                    cudaMemcpyHostToDevice,
                    stream_);
    cudaMemcpyAsync(
        cu_seqlens.getPtr<int>(), cu_seqlens_.data(), sizeof(int) * cu_seqlens.size(), cudaMemcpyHostToDevice, stream_);
    inputs->padding_offset = padding_offset;
    inputs->cu_seqlens     = cu_seqlens;
}

template<typename T>
std::shared_ptr<ModelOutput> ParallelModelWrapperImpl<T>::forward(const ModelRequest& model_request)
{
    // std::cout << "aa:" << std::endl;

    const uint   total_batch_size = model_request.generate_batch_size + model_request.context_batch_size;
    const size_t h_token_num      = model_request.combo_tokens.size(0);
    assert(params_.head_num_ % tensor_para_.world_size_ == 0);
    const int    local_head_num = params_.head_num_ / tensor_para_.world_size_;
    const size_t hidden_units   = params_.head_num_ * params_.size_per_head_;
    allocateBuffer(total_batch_size, h_token_num);
    // std::cout << "bb:" << std::endl;

    // assert(token_num == model_request.context_batch_size + model_request.generate_batch_size);
    size_t context_h_token_num = h_token_num - model_request.generate_batch_size;
    Tensor all_hidden_states(MEMORY_GPU, data_type_, {h_token_num, hidden_units}, allocator_, true);
    Tensor last_hidden_states(MEMORY_GPU, data_type_, {total_batch_size, hidden_units}, allocator_, true);
    // Tensor logits_tmp(MEMORY_GPU, data_type_, {(size_t)total_batch_size, (size_t)params_.vocab_size_}, allocator_,
    // true);
    Tensor logits(
        MEMORY_GPU, DataType::TYPE_FP32, {(size_t)total_batch_size, (size_t)params_.vocab_size_}, allocator_, true);
    Tensor combo_tokens(MEMORY_GPU, DataType::TYPE_INT32, {h_token_num}, allocator_, true);

    cudaMemcpyAsync(combo_tokens.getPtr<int>(),
                    model_request.combo_tokens.data_ptr<int>(),
                    combo_tokens.sizeBytes(),
                    cudaMemcpyHostToDevice,
                    stream_);

    // word embedding
    parallel_word_embedding_wrapper_->forward(all_hidden_states,
                                              combo_tokens,
                                              getPositionsId(h_token_num,
                                                             model_request.generate_batch_size,
                                                             model_request.context_batch_size,
                                                             model_request.input_lengths,
                                                             model_request.sequence_lengths));
    auto args = prepareGptCommonInputs(model_request);
    // std::cout << "kk:" << std::endl;

    // gpt layer
    parallel_gpt_decoder_->forward(all_hidden_states, all_hidden_states, args.get());

    // last hidden states
    cudaMemcpyAsync(reinterpret_cast<T*>(all_hidden_states.getPtr<T>()),
                    reinterpret_cast<T*>(last_hidden_states.getPtr<T>()),
                    model_request.generate_batch_size * hidden_units,
                    cudaMemcpyDeviceToDevice,
                    stream_);
    // std::cout << "jj:" << std::endl;

    if (model_request.context_batch_size) {
        invokeLookupHiddenStateOfLastToken(
            last_hidden_states.getPtrWithOffset<T>(model_request.generate_batch_size * hidden_units),
            all_hidden_states.getPtrWithOffset<T>(model_request.generate_batch_size * hidden_units),
            args->cu_seqlens.template getPtrWithOffset<int>(1),
            0,
            model_request.context_batch_size,
            hidden_units,
            stream_);
    }

    // logits
    parallel_logits_wrapper_->forward(logits, last_hidden_states);
    // std::cout << "zz:" << std::endl;

    std::shared_ptr<ModelOutput> model_output = make_shared<ModelOutput>();
    model_output->logits                      = logits;
    if (model_request.return_hidden_state) {
        model_output->last_hidden_states = last_hidden_states;
    }
    if (model_request.calculate_loss) {
        model_output->loss = calculate_loss(all_hidden_states);
    }
    return model_output;
}

ParallelModelWrapper::ParallelModelWrapper(
    const GptInitParameter&                                         gpt_init_parameter,
    const int                                                       tensor_para_size,
    const std::string&                                              master_ip,
    const int                                                       master_port,
    const std::unordered_map<std::string, ft::Tensor>&              global_weights,
    const std::vector<std::unordered_map<std::string, ft::Tensor>>& layer_weights,
    const std::vector<std::unordered_map<std::string, ft::Tensor>>& layer_int8_weights,
    const std::vector<std::unordered_map<std::string, ft::Tensor>>& layer_int8_scales)
{
    DataType data_type = layer_weights[0].begin()->second.type;
    for (auto layer_weght : layer_weights) {
        for (auto weight : layer_weght) {
            FT_CHECK_WITH_INFO(weight.second.type == data_type, "input tensor not consistent");
        }
    }
    for (auto layer_weght : layer_int8_scales) {
        for (auto weight : layer_weght) {
            FT_CHECK_WITH_INFO(weight.second.type == data_type, "input tensor not consistent");
        }
    }
#define CREATE_INSTANCE(T_)                                                                                            \
    {                                                                                                                  \
        model_wrapper_ = new ParallelModelWrapperImpl<T_>(gpt_init_parameter,                                          \
                                                          tensor_para_size,                                            \
                                                          master_ip,                                                   \
                                                          master_port,                                                 \
                                                          global_weights,                                              \
                                                          layer_weights,                                               \
                                                          layer_int8_weights,                                          \
                                                          layer_int8_scales);                                          \
    }

    switch (data_type) {
        case DataType::TYPE_FP32:
            CREATE_INSTANCE(float);
            break;
        case DataType::TYPE_FP16:
            CREATE_INSTANCE(half);
            break;
        case DataType::TYPE_BF16:
            if constexpr (CompileConfig::enable_bf16) {
                CREATE_INSTANCE(__nv_bfloat16);
            }
            break;
        default:
            throw std::runtime_error("Wrong tensor type.");
    }
#undef CREATE_INSTANCE
}

std::shared_ptr<ModelOutput> ParallelModelWrapper::forward(const ModelRequest& model_request)
{
    return model_wrapper_->forward(model_request);
}

template class ParallelModelWrapperImpl<float>;
template class ParallelModelWrapperImpl<half>;
#ifdef ENABLE_BF16
template class ParallelModelWrapperImpl<__nv_bfloat16>;
#endif

}  // namespace rtp_llm
