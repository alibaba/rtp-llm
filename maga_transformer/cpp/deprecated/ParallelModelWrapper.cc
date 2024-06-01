#include "maga_transformer/cpp/deprecated/ParallelModelWrapper.h"
#include "maga_transformer/cpp/utils/StringUtil.h"
#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/core/allocator.h"
#include "src/fastertransformer/devices/DeviceBase.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "src/fastertransformer/kernels/bert_preprocess_kernels.h"
#include "src/fastertransformer/kernels/gen_relative_pos_bias.h"
#include "src/fastertransformer/kernels/gpt_kernels.h"
#include "src/fastertransformer/utils/compiler_config.h"

#include <algorithm>
#include <math.h>
#include <memory>

using namespace std;

namespace th = torch;
using namespace fastertransformer;

namespace rtp_llm {

template<typename T>
void ParallelModelWrapperImpl<T>::initialize() {
    parallel_word_embedding_wrapper_.reset(
        new ParallelWordEmbeddingWrapper<T>(params_,
                                            tensor_para_,
                                            stream_,
                                            cublas_wrapper_,
                                            allocator_,
                                            true,
                                            &global_weights_->embedding_table,
                                            &global_weights_->position_encoding_table,
                                            &global_weights_->token_type_embedding_table));
    parallel_gpt_decoder_.reset(new ParallelGpt<T>(
        params_, tensor_para_, pipeline_para_, stream_, cublas_wrapper_, allocator_, true, true, false, nullptr, 0));
    if (global_weights_->lm_head.kernel != nullptr) {
        parallel_logits_wrapper_.reset(new ParallelLogitsWrapper<T>(
            params_, tensor_para_, stream_, cublas_wrapper_, allocator_, true, &global_weights_->lm_head));
    }
    norm_wrapper_.reset(
        new NormWrapper<T>(params_.layernorm_type_, params_.norm_type_, T(sqrt(2 * params_.num_layers_))));

    for (int i = 0; i < static_cast<int>(params_.num_layers_); i++) {
        gpt_lora_layer_weights_.push_back(new ft::ParallelGptDecoderLoRALayerWeight<T>());
    }
}

template<typename T>
ParallelModelWrapperImpl<T>::ParallelModelWrapperImpl(
    const ft::GptInitParameter&                                             gpt_init_parameter,
    ft::NcclParam                                                           tensor_para,
    ft::NcclParam                                                           pipeline_para,
    const std::unordered_map<std::string, ft::ConstBufferPtr>&              global_weights,
    const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& layer_weights):
    params_(gpt_init_parameter),
    data_type_(ft::getTensorType<T>()),
    device_(dynamic_cast<ft::CudaDevice*>(ft::DeviceFactory::getDevice(ft::DeviceType::Cuda))),
    tensor_para_(tensor_para),
    pipeline_para_(pipeline_para)
{
    allocator_      = device_->getAllocator();
    cublas_wrapper_ = device_->cublasMMWrapperPtr();
    stream_         = device_->stream();

    std::vector<std::unordered_map<std::string, ft::Tensor>> layer_weights_;
    for (auto& weights : layer_weights) {
        std::unordered_map<std::string, ft::Tensor> weights_;
        for (auto& it : weights) {
            weights_.emplace(
                it.first,
                std::move(ft::Tensor(it.second->where(), it.second->type(), it.second->shape(), it.second->data())));
        }
        layer_weights_.emplace_back(std::move(weights_));
    }
    gpt_layer_weights_ =
        torch_ext::loadWeights<T>(pipeline_para_.world_size_,
                                  pipeline_para_.rank_,
                                  gpt_init_parameter.num_layers_,
                                  gpt_init_parameter.quant_algo_.toQuantAlgo(),
                                  layer_weights_,
                                  (const std::vector<ft::ParallelGptDecoderLoRALayerWeight<T>*>*)nullptr);
    global_weights_.reset(new GptGlobalWeights<T>(global_weights));

    initialize();
}

template<typename T>
ParallelModelWrapperImpl<T>::~ParallelModelWrapperImpl() {
    freeBuffer();
}

template<typename T>
void ParallelModelWrapperImpl<T>::allocateBuffer(size_t total_batch_size, size_t h_token_num, GptModelOutputs& model_output) {
    size_t hidden_units = params_.hidden_size_;
    const auto& dtype = getTensorType<T>();
    model_output.hidden_states = const_cast<ft::CudaDevice*>(device_)->allocateBuffer(
        {dtype, {(size_t)total_batch_size, (size_t)hidden_units}, ft::AllocationType::DEVICE},
        {});
    model_output.all_hidden_states = const_cast<ft::CudaDevice*>(device_)->allocateBuffer(
        {dtype, {(size_t)h_token_num, (size_t)hidden_units}, ft::AllocationType::DEVICE},
        {});
    last_hidden_states_ = (T*)model_output.hidden_states->data();
    all_hidden_states_  = (T*)model_output.all_hidden_states->data();
    combo_tokens_   = (int*)allocator_->reMalloc(combo_tokens_, sizeof(int) * h_token_num);
        combo_token_types_   = (int*)allocator_->reMalloc(combo_token_types_, sizeof(int) * h_token_num);
    combo_position_ids_   = (int*)allocator_->reMalloc(combo_position_ids_, sizeof(int) * h_token_num);
    padding_offset_ = reinterpret_cast<int*>(allocator_->reMalloc(padding_offset_, sizeof(int) * (h_token_num)));
    cu_seqlens_ =
        reinterpret_cast<int*>(allocator_->reMalloc(cu_seqlens_, sizeof(int) * (total_batch_size + 1)));
    input_lengths_ =
        reinterpret_cast<int*>(allocator_->reMalloc(input_lengths_, sizeof(int) * (total_batch_size)));
    sequence_lengths_ =
        reinterpret_cast<int*>(allocator_->reMalloc(sequence_lengths_, sizeof(int) * (total_batch_size)));
    prefix_lengths_ =
        reinterpret_cast<int*>(allocator_->reMalloc(prefix_lengths_, sizeof(int) * (total_batch_size)));
}

template<typename T>
void ParallelModelWrapperImpl<T>::freeBuffer() {
    allocator_->free((void**)all_hidden_states_);
    allocator_->free((void**)last_hidden_states_);
    allocator_->free((void**)combo_tokens_);
    allocator_->free((void**)padding_offset_);
    allocator_->free((void**)cu_seqlens_);
    allocator_->free((void**)input_lengths_);
    allocator_->free((void**)sequence_lengths_);
    allocator_->free((void**)prefix_lengths_);
    allocator_->free((void**)attention_mask_);
}

template<typename T>
void ParallelModelWrapperImpl<T>::setPaddingOffsetAndCuSeqLens(ft::Tensor& padding_offset,
                                                               ft::Tensor& cu_seqlens,
                                                               const uint  context_h_token_num,
                                                               const uint  context_batch_size,
                                                               const uint  max_context_seq_length,
                                                               const int*  input_lengths) {
    padding_offset_cpu_.resize(context_h_token_num);
    cu_seqlens_cpu_.resize(context_batch_size + 1);
    int total_seq_len = 0;
    int cum_offset    = 0;
    int index         = 0;
    for (int i = 0; i < context_batch_size; ++i) {
        const int seq_len  = input_lengths[i];
        cu_seqlens_cpu_[i] = total_seq_len;
        for (int j = 0; j < seq_len; j++) {
            padding_offset_cpu_[index] = cum_offset;
            index++;
        }
        cum_offset += max_context_seq_length - seq_len;
        total_seq_len += seq_len;
    }
    cu_seqlens_cpu_[context_batch_size] = total_seq_len;
    cudaMemcpyAsync(padding_offset.getPtr<int>(),
                    padding_offset_cpu_.data(),
                    sizeof(int) * padding_offset_cpu_.size(),
                    cudaMemcpyHostToDevice,
                    stream_);
    cudaMemcpyAsync(cu_seqlens.getPtr<int>(),
                    cu_seqlens_cpu_.data(),
                    sizeof(int) * cu_seqlens_cpu_.size(),
                    cudaMemcpyHostToDevice,
                    stream_);
}

template<typename T>
bool ParallelModelWrapperImpl<T>::useFMHA() {
    return parallel_gpt_decoder_->UseFMHA();
}
template<typename T>
void ParallelModelWrapperImpl<T>::createAttentionMask(size_t context_batch_size, size_t max_context_seq_length, int* input_lengths_host) {
    cudaMemcpyAsync(input_lengths_, input_lengths_host, sizeof(int) * context_batch_size, cudaMemcpyHostToDevice, stream_);
    attention_mask_  = (T*)allocator_->reMalloc(attention_mask_, sizeof(T) * context_batch_size * max_context_seq_length * max_context_seq_length);
    invokeBuildDecoderAttentionMask<T>(attention_mask_, input_lengths_, nullptr, context_batch_size, max_context_seq_length, 0, params_.is_causal_, stream_);
    sync_check_cuda_error();
}

template<typename T>
GptModelOutputs ParallelModelWrapperImpl<T>::forward(const ModelRequest& model_request) {
    const uint   total_batch_size = model_request.generate_batch_size + model_request.context_batch_size;
    const size_t h_token_num      = model_request.combo_tokens->shape()[0];
    GptModelOutputs model_output;
    allocateBuffer(total_batch_size, h_token_num, model_output);
    assert(params_.head_num_ % tensor_para_.world_size_ == 0);
    const int    local_head_num      = params_.head_num_ / tensor_para_.world_size_;
    const size_t hidden_units        = params_.head_num_ * params_.size_per_head_;
    size_t       context_h_token_num = h_token_num - model_request.generate_batch_size;
    ft::Tensor   all_hidden_states(ft::MEMORY_GPU, data_type_, {h_token_num, hidden_units}, all_hidden_states_);
    ft::Tensor   last_hidden_states(ft::MEMORY_GPU, data_type_, {total_batch_size, hidden_units}, last_hidden_states_);
    ft::Tensor   combo_tokens(ft::MEMORY_GPU, ft::DataType::TYPE_INT32, {h_token_num}, combo_tokens_);
    ft::Tensor   combo_token_type_ids(ft::MEMORY_GPU, ft::DataType::TYPE_INT32, {h_token_num}, combo_token_types_);
    ft::Tensor   combo_position_ids(ft::MEMORY_GPU, ft::DataType::TYPE_INT32, {h_token_num}, combo_position_ids_);

    ft::Tensor padding_offset(ft::MEMORY_GPU, ft::DataType::TYPE_INT32, {context_h_token_num}, padding_offset_);

    ft::Tensor cu_seqlens(
        ft::MEMORY_GPU, ft::DataType::TYPE_INT32, {(size_t)model_request.context_batch_size + 1}, cu_seqlens_);
    ft::Tensor input_lengths(
        ft::MEMORY_CPU, ft::DataType::TYPE_INT32, {total_batch_size}, model_request.input_lengths->data());
    ft::Tensor sequence_lengths(ft::MEMORY_CPU,
                                ft::DataType::TYPE_INT32,
                                {(size_t)model_request.generate_batch_size},
                                model_request.sequence_lengths->data());
    ft::Tensor lora_input_lengths(
        ft::MEMORY_CPU, ft::DataType::TYPE_INT32, {total_batch_size}, model_request.input_lengths->data());
    ft::Tensor lora_ids(ft::MEMORY_CPU, ft::DataType::TYPE_INT32, {0}, nullptr);

    model_output.logits                          = const_cast<ft::CudaDevice*>(device_)->allocateBuffer(
        {ft::DataType::TYPE_FP32, {(size_t)total_batch_size, (size_t)params_.vocab_size_}, ft::AllocationType::DEVICE},
        {});
    ft::Tensor logits(ft::MEMORY_GPU,
                      ft::DataType::TYPE_FP32,
                      {(size_t)total_batch_size, (size_t)params_.vocab_size_},
                      model_output.logits->data());

    cudaMemcpyAsync(combo_tokens.getPtr<int>(),
                    model_request.combo_tokens->data(),
                    combo_tokens.sizeBytes(),
                    cudaMemcpyHostToDevice,
                    stream_);
    if (model_request.combo_position_ids != nullptr) {
        cudaMemcpyAsync(combo_position_ids.getPtr<int>(),
                model_request.combo_position_ids->data(),
                combo_position_ids.sizeBytes(),
                cudaMemcpyHostToDevice,
                stream_);
    }
    if (model_request.combo_token_type_ids != nullptr) {
        cudaMemcpyAsync(combo_token_type_ids.getPtr<int>(),
                model_request.combo_token_type_ids->data(),
                combo_token_type_ids.sizeBytes(),
                cudaMemcpyHostToDevice,
                stream_);
    }

    size_t max_context_seq_length_ = 0;
    if (model_request.context_batch_size) {
        const int max_context_seq_length =
            *std::max_element((int*)model_request.input_lengths->data() + model_request.generate_batch_size,
                              (int*)model_request.input_lengths->data() + total_batch_size);
        max_context_seq_length_ = max_context_seq_length;
        setPaddingOffsetAndCuSeqLens(padding_offset,
                                     cu_seqlens,
                                     context_h_token_num,
                                     model_request.context_batch_size,
                                     max_context_seq_length,
                                     (int*)model_request.input_lengths->data() + model_request.generate_batch_size);
    }
    ft::Tensor attention_mask;
    if (!parallel_gpt_decoder_->UseFMHA() && model_request.attention_mask.get() == nullptr && model_request.context_batch_size > 0) {
        createAttentionMask(model_request.context_batch_size, max_context_seq_length_, model_request.input_lengths->data<int>() + model_request.generate_batch_size);
        attention_mask = ft::Tensor(
            ft::MEMORY_GPU,
            data_type_,
            {(size_t)model_request.context_batch_size, max_context_seq_length_, max_context_seq_length_},
        attention_mask_);
    } else if (model_request.attention_mask) {
        const auto& attention_mask_shape = model_request.attention_mask->shape();
        attention_mask = ft::Tensor(
            ft::MEMORY_GPU,
            data_type_,
            {attention_mask_shape[0], attention_mask_shape[1], attention_mask_shape[2]},
            model_request.attention_mask->data());
    } else {
        attention_mask = ft::Tensor(
        ft::MEMORY_GPU,
        data_type_,
        {(size_t)model_request.context_batch_size, max_context_seq_length_, max_context_seq_length_},
        nullptr);
    }

    if (attention_mask.data() != nullptr) {
        print_bsd(-1, "attention_mask", attention_mask.getPtr<T>(), model_request.context_batch_size, max_context_seq_length_, max_context_seq_length_);
    }

    ft::Tensor position_ids;
    // word embedding
    print_bsd(-1, "token", combo_tokens.getPtr<int>(), 1, 1, h_token_num);
    print_bsd(-1, "position", combo_position_ids.getPtr<int>(), 1, 1, h_token_num);
    print_bsd(-1, "type", combo_token_type_ids.getPtr<int>(), 1, 1, h_token_num);
    parallel_word_embedding_wrapper_->forward(all_hidden_states, combo_tokens, combo_token_type_ids, combo_position_ids);

    print_bsd(-1, "embedding", all_hidden_states.getPtr<T>(), 1, h_token_num, hidden_units);

    if (params_.has_pre_decoder_layernorm_) {
        norm_wrapper_->generalLayerNorm(all_hidden_states.getPtr<T>(),
                                            all_hidden_states.getPtr<T>(),
                                            global_weights_->pre_decoder_layernorm_weights.gamma,
                                            global_weights_->pre_decoder_layernorm_weights.beta,
                                            params_.layernorm_eps_,
                                            h_token_num,
                                            hidden_units,
                                            nullptr,  // scale
                                            nullptr,  // dyanmic scale
                                            reinterpret_cast<int8_t*>(last_hidden_states.getPtr<T>()),
                                            stream_);
    }
    sync_check_cuda_error();

    // gpt layer
    ft::TensorMap input_tensors({{"decoder_input", all_hidden_states},
                                 {"sequence_lengths", sequence_lengths},
                                 {"input_lengths", input_lengths},
                                 {"lora_ids", lora_ids},
                                 {"attention_mask", attention_mask},
                                 {"lora_input_lengths", lora_input_lengths}});
    ft::TensorMap output_tensors({{"decoder_output", all_hidden_states}});

    if (model_request.kv_cache_blocks != nullptr) {
        ft::Tensor block_pointers(
            ft::MEMORY_CPU,
            ft::DataType::TYPE_INT64,
            model_request.kv_cache_blocks->shape(),
            model_request.kv_cache_blocks->data());
        output_tensors.insert("block_pointers", block_pointers);
    }
    if (model_request.kv_cache_scales != nullptr) {
        ft::Tensor block_scale_pointers(
            ft::MEMORY_CPU,
            ft::DataType::TYPE_INT64,
            model_request.kv_cache_scales->shape(),
            model_request.kv_cache_scales->data());
        output_tensors.insert("block_scale_pointers", block_scale_pointers);
    }

    parallel_gpt_decoder_->forward(&output_tensors, &input_tensors, &gpt_layer_weights_);
    sync_check_cuda_error();
    // last hidden states
    cudaMemcpyAsync(reinterpret_cast<T*>(last_hidden_states.getPtr<T>()),
                    reinterpret_cast<T*>(all_hidden_states.getPtr<T>()),
                    model_request.generate_batch_size * hidden_units * sizeof(T),
                    cudaMemcpyDeviceToDevice,
                    stream_);
    sync_check_cuda_error();

    if (model_request.context_batch_size) {
        if (params_.is_causal_) {
            invokeLookupHiddenStateOfLastToken(
                last_hidden_states.getPtrWithOffset<T>(model_request.generate_batch_size * hidden_units),
                all_hidden_states.getPtrWithOffset<T>(model_request.generate_batch_size * hidden_units),
                cu_seqlens.template getPtrWithOffset<int>(1),
                model_request.context_batch_size,
                hidden_units,
                stream_);
        } else {
            invokeLookupHiddenStateOfFirstToken(
                last_hidden_states.getPtrWithOffset<T>(model_request.generate_batch_size * hidden_units),
                all_hidden_states.getPtrWithOffset<T>(model_request.generate_batch_size * hidden_units),
                cu_seqlens.template getPtrWithOffset<int>(1),
                model_request.context_batch_size,
                hidden_units,
                stream_);
        }
    }
    if (params_.has_post_decoder_layernorm_) {
        norm_wrapper_->initDecoderLayerNorm(last_hidden_states.getPtr<T>(),
                                            last_hidden_states.getPtr<T>(),
                                            global_weights_->post_layernorm_weights.gamma,
                                            global_weights_->post_layernorm_weights.beta,
                                            params_.layernorm_eps_,
                                            total_batch_size,
                                            hidden_units,
                                            nullptr,  // scale
                                            nullptr,  // dyanmic scale
                                            reinterpret_cast<int8_t*>(last_hidden_states.getPtr<T>()),
                                            stream_);
    }
    sync_check_cuda_error();

    // logits
    if (parallel_logits_wrapper_ != nullptr) {
        ft::Tensor logits(ft::MEMORY_GPU,
                    ft::DataType::TYPE_FP32,
                    {(size_t)total_batch_size, (size_t)params_.vocab_size_},
                    model_output.logits->data());

        parallel_logits_wrapper_->forward(logits, last_hidden_states);

        print_bsd(-1,
                "last_hidden_states",
                last_hidden_states.getPtr<T>(),
                total_batch_size,
                1,
                params_.hidden_size_);

        print_bsd(-1,
                "logits",
                logits.getPtr<float>(),
                total_batch_size,
                1,
                params_.vocab_size_);
    }
    sync_check_cuda_error();

    return model_output;
}

at::ScalarType getScalarType(const std::string& data_type) {
    at::ScalarType scalar_type;
    if (data_type == "fp16") {
        scalar_type = at::ScalarType::Half;

    } else if (data_type == "bf16") {
        scalar_type = at::ScalarType::BFloat16;

    } else if (data_type == "fp32") {
        scalar_type = at::ScalarType::Float;

    } else {
        FT_LOG_ERROR("datatype not implemented %s", data_type.c_str());
    }
    return scalar_type;
}

ParallelModelWrapper::ParallelModelWrapper(
    const GptInitParameter&                                                 gpt_init_parameter,
    const std::unordered_map<std::string, ft::ConstBufferPtr>&              global_weights,
    const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& layer_weights) {
    auto device = dynamic_cast<ft::CudaDevice*>(ft::DeviceFactory::getDevice(ft::DeviceType::Cuda));
    ft::NcclParam pipeline_para;
    auto tensor_para = device->getNcclParam();

#define CREATE_INSTANCE(T_)                                                                     \
    {                                                                                           \
        model_wrapper_ = new ParallelModelWrapperImpl<T_>(                                      \
                gpt_init_parameter, tensor_para, pipeline_para, global_weights, layer_weights); \
    }

    switch (getScalarType(gpt_init_parameter.data_type_)) {
    case at::ScalarType::Float:
        CREATE_INSTANCE(float);
        break;
    case at::ScalarType::Half:
        CREATE_INSTANCE(half);
        break;
    case at::ScalarType::BFloat16:
        if constexpr (CompileConfig::enable_bf16) {
            CREATE_INSTANCE(__nv_bfloat16);
        }
        break;
    default:
        throw std::runtime_error("Wrong tensor type.");
    }
#undef CREATE_INSTANCE
}

bool ParallelModelWrapper::useFMHA() {
    return model_wrapper_->useFMHA();
}

GptModelOutputs ParallelModelWrapper::forward(const ModelRequest& model_request) {
    return model_wrapper_->forward(model_request);
}

template class ParallelModelWrapperImpl<float>;
template class ParallelModelWrapperImpl<half>;
#ifdef ENABLE_BF16
template class ParallelModelWrapperImpl<__nv_bfloat16>;
#endif

}  // namespace rtp_llm
