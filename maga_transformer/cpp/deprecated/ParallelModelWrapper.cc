#include "maga_transformer/cpp/deprecated/ParallelModelWrapper.h"
#include "src/fastertransformer/core/Buffer.h"
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
namespace ft = fastertransformer;

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
                                            &global_weights_->position_encoding_table));
    parallel_gpt_decoder_.reset(new ParallelGpt<T>(
        params_, tensor_para_, pipeline_para_, stream_, cublas_wrapper_, allocator_, true, true, false, nullptr, 0));
    parallel_logits_wrapper_.reset(new ParallelLogitsWrapper<T>(
        params_, tensor_para_, stream_, cublas_wrapper_, allocator_, true, &global_weights_->embedding_table));
}

template<typename T>
ParallelModelWrapperImpl<T>::ParallelModelWrapperImpl(
    const GptInitParameter&                                                 gpt_init_parameter,
    const int                                                               tensor_para_size,
    const std::string&                                                      master_ip,
    const int                                                               master_port,
    const std::unordered_map<std::string, ft::ConstBufferPtr>&              global_weights,
    const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& layer_weights):
    params_(gpt_init_parameter),
    data_type_(ft::getTensorType<T>()),
    device_(dynamic_cast<CudaDevice*>(ft::DeviceFactory::getDevice(ft::DeviceType::Cuda))) {
    ftNcclInitialize(tensor_para_, pipeline_para_, tensor_para_size, 1, master_ip, master_port);
    allocator_      = device_->getAllocator();
    cublas_wrapper_ = device_->cublasMMWrapperPtr();
    stream_         = device_->stream();

    // std::cout << "ss1:" << std::endl;
    std::vector<std::unordered_map<std::string, ft::Tensor>> layer_weights_;
    for (auto& weights : layer_weights) {
        std::unordered_map<std::string, ft::Tensor> __weights;
        for (auto& it : weights) {
            __weights.emplace(
                it.first,
                std::move(ft::Tensor(it.second->where(), it.second->type(), it.second->shape(), it.second->data())));
        }
        layer_weights_.emplace_back(std::move(__weights));
    }
    gpt_layer_weights_ =
        torch_ext::loadWeights<T>(pipeline_para_.world_size_,
                                  pipeline_para_.rank_,
                                  gpt_init_parameter.num_layers_,
                                  gpt_init_parameter.quant_algo_,
                                  layer_weights_,
                                  (const std::vector<ft::ParallelGptDecoderLoRALayerWeight<T>*>*)nullptr);

    global_weights_.reset(new GptGlobalWeights<T>(global_weights));

    initialize();
}

template<typename T>
ParallelModelWrapperImpl<T>::~ParallelModelWrapperImpl() {
    ftNcclParamDestroy(tensor_para_);
    ftNcclParamDestroy(pipeline_para_);
    freeBuffer();
}

template<typename T>
void ParallelModelWrapperImpl<T>::allocateBuffer(size_t total_batch_size, size_t h_token_num) {
    size_t hidden_units = params_.hidden_size_;
    all_hidden_states_  = (T*)allocator_->reMalloc(all_hidden_states_, sizeof(T) * h_token_num * hidden_units, true);
    last_hidden_states_ =
        (T*)allocator_->reMalloc(last_hidden_states_, sizeof(T) * total_batch_size * hidden_units, true);
    combo_tokens_   = (int*)allocator_->reMalloc(combo_tokens_, sizeof(int) * h_token_num, true);
    padding_offset_ = reinterpret_cast<int*>(allocator_->reMalloc(padding_offset_, sizeof(int) * (h_token_num), false));
    cu_seqlens_ =
        reinterpret_cast<int*>(allocator_->reMalloc(cu_seqlens_, sizeof(int) * (total_batch_size + 1), false));
    input_lengths_ =
        reinterpret_cast<int*>(allocator_->reMalloc(input_lengths_, sizeof(int) * (total_batch_size), false));
    sequence_lengths_ =
        reinterpret_cast<int*>(allocator_->reMalloc(sequence_lengths_, sizeof(int) * (total_batch_size), false));
    prefix_lengths_ =
        reinterpret_cast<int*>(allocator_->reMalloc(prefix_lengths_, sizeof(int) * (total_batch_size), false));
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
std::unique_ptr<GptModelOutputs> ParallelModelWrapperImpl<T>::forward(const ModelRequest& model_request) {
    const uint   total_batch_size = model_request.generate_batch_size + model_request.context_batch_size;
    const size_t h_token_num      = model_request.combo_tokens->shape()[0];
    allocateBuffer(total_batch_size, h_token_num);
    assert(params_.head_num_ % tensor_para_.world_size_ == 0);
    const int    local_head_num      = params_.head_num_ / tensor_para_.world_size_;
    const size_t hidden_units        = params_.head_num_ * params_.size_per_head_;
    size_t       context_h_token_num = h_token_num - model_request.generate_batch_size;
    ft::Tensor   all_hidden_states(ft::MEMORY_GPU, data_type_, {h_token_num, hidden_units}, all_hidden_states_);
    ft::Tensor   last_hidden_states(ft::MEMORY_GPU, data_type_, {total_batch_size, hidden_units}, last_hidden_states_);
    ft::Tensor   combo_tokens(ft::MEMORY_GPU, ft::DataType::TYPE_INT32, {h_token_num}, combo_tokens_);

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

    ft::Tensor                       block_pointers(ft::MEMORY_CPU,
                              ft::DataType::TYPE_INT64,
                              model_request.kv_cache_blocks->shape(),
                              model_request.kv_cache_blocks->data());
    std::unique_ptr<GptModelOutputs> model_output = make_unique<GptModelOutputs>();
    model_output->logits                          = const_cast<ft::CudaDevice*>(device_)->allocateBuffer(
        {ft::DataType::TYPE_FP32, {(size_t)total_batch_size, (size_t)params_.vocab_size_}, ft::AllocationType::DEVICE},
        {});
    ft::Tensor logits(ft::MEMORY_GPU,
                      ft::DataType::TYPE_FP32,
                      {(size_t)total_batch_size, (size_t)params_.vocab_size_},
                      model_output->logits->data());

    cudaMemcpyAsync(combo_tokens.getPtr<int>(),
                    model_request.combo_tokens->data(),
                    combo_tokens.sizeBytes(),
                    cudaMemcpyHostToDevice,
                    stream_);
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
    ft::Tensor attention_mask(
        ft::MEMORY_GPU,
        ft::DataType::TYPE_FP16,
        {(size_t)model_request.context_batch_size, max_context_seq_length_, max_context_seq_length_},
        nullptr);

    ft::Tensor position_ids;
    // word embedding
    parallel_word_embedding_wrapper_->forward(all_hidden_states, combo_tokens, position_ids);

    // gpt layer
    ft::TensorMap input_tensors({{"decoder_input", all_hidden_states},
                                 {"sequence_lengths", sequence_lengths},
                                 {"input_lengths", input_lengths},
                                 {"lora_ids", lora_ids},
                                 {"attention_mask", attention_mask},
                                 {"lora_input_lengths", lora_input_lengths}});
    ft::TensorMap output_tensors({{"decoder_output", all_hidden_states}, {"block_pointers", block_pointers}});

    parallel_gpt_decoder_->forward(&output_tensors, &input_tensors, &gpt_layer_weights_);
    sync_check_cuda_error();
    // last hidden states
    cudaMemcpyAsync(reinterpret_cast<T*>(all_hidden_states.getPtr<T>()),
                    reinterpret_cast<T*>(last_hidden_states.getPtr<T>()),
                    model_request.generate_batch_size * hidden_units,
                    cudaMemcpyDeviceToDevice,
                    stream_);
    sync_check_cuda_error();

    if (model_request.context_batch_size) {
        invokeLookupHiddenStateOfLastToken(
            last_hidden_states.getPtrWithOffset<T>(model_request.generate_batch_size * hidden_units),
            all_hidden_states.getPtrWithOffset<T>(model_request.generate_batch_size * hidden_units),
            cu_seqlens.template getPtrWithOffset<int>(1),
            0,
            model_request.context_batch_size,
            hidden_units,
            stream_);
    }
    sync_check_cuda_error();

    // logits
    parallel_logits_wrapper_->forward(logits, last_hidden_states);
    return std::move(model_output);
}

ParallelModelWrapper::ParallelModelWrapper(
    const GptInitParameter&                                                 gpt_init_parameter,
    const int                                                               tensor_para_size,
    const std::string&                                                      master_ip,
    const int                                                               master_port,
    const std::unordered_map<std::string, ft::ConstBufferPtr>&              global_weights,
    const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& layer_weights) {
#define CREATE_INSTANCE(T_)                                                                                            \
    {                                                                                                                  \
        model_wrapper_ = new ParallelModelWrapperImpl<T_>(                                                             \
            gpt_init_parameter, tensor_para_size, master_ip, master_port, global_weights, layer_weights);              \
    }

    DataType data_type = DataType::TYPE_FP16;
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

std::unique_ptr<GptModelOutputs> ParallelModelWrapper::forward(const ModelRequest& model_request) {
    return model_wrapper_->forward(model_request);
}

template class ParallelModelWrapperImpl<float>;
template class ParallelModelWrapperImpl<half>;
#ifdef ENABLE_BF16
template class ParallelModelWrapperImpl<__nv_bfloat16>;
#endif

}  // namespace rtp_llm
