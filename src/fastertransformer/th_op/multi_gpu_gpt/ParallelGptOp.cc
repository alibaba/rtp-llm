#include "src/fastertransformer/th_op/multi_gpu_gpt/ParallelGptOp.h"
#include "src/fastertransformer/th_op/multi_gpu_gpt/Base.h"
#include <string>

namespace th = torch;
namespace ft = fastertransformer;

namespace torch_ext {

template<typename T>
FtGpt<T>::FtGpt(const GptInitParameter&       gpt_init_parameter,
                const int                     tensor_para_size,
                const int                     pipeline_para_size,
                const std::string&            master_ip,
                const int                     master_port,
                const std::vector<std::unordered_map<std::string, th::Tensor>> &weights):
    gpt_init_parameter_(gpt_init_parameter)
{
    check_cuda_error(cublasLtCreate(&cublaslt_handle_));
    cublas_algo_map_      = new ft::cublasAlgoMap(GEMM_CONFIG);
    cublas_wrapper_mutex_ = new std::mutex();

    ft::ftNcclInitialize(tensor_para_, pipeline_para_, tensor_para_size, pipeline_para_size, master_ip, master_port);

    for (int i = 0; i < static_cast<int>(gpt_init_parameter_.num_layers_); i++) {
        gpt_lora_layer_weights_.push_back(new ft::ParallelGptDecoderLoRALayerWeight<T>());
    }
    gpt_layer_weights_ = loadWeights<T>(pipeline_para_.world_size_,
                                        pipeline_para_.rank_,
                                        gpt_init_parameter_.num_layers_,
                                        gpt_init_parameter_.quant_algo_,
                                        weights,
                                        &gpt_lora_layer_weights_);
    
    auto stream        = at::cuda::getCurrentCUDAStream().stream();
    auto cublas_handle = at::cuda::getCurrentCUDABlasHandle();
    cublasSetStream(cublas_handle, stream);
    allocator_ = new ft::Allocator<ft::AllocatorType::TH>;
    allocator_->setStream(stream);

    cublas_wrapper_ = new ft::cublasMMWrapper(
        cublas_handle, cublaslt_handle_, stream, cublas_algo_map_, cublas_wrapper_mutex_, allocator_);

    if (std::is_same<T, half>::value) {
        cublas_wrapper_->setGemmConfig(CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F);
    }

    else if constexpr (std::is_same<T, __nv_bfloat16>::value && CompileConfig::enable_bf16) {
        cublas_wrapper_->setBF16GemmConfig();
    }

    else if (std::is_same<T, float>::value) {
        cublas_wrapper_->setFP32GemmConfig();
    }

    parallel_gpt_ = new ft::ParallelGpt<T>(gpt_init_parameter_,
                                           tensor_para_,
                                           pipeline_para_,
                                           stream,
                                           cublas_wrapper_,
                                           allocator_,
                                           true,
                                           true,
                                           false,
                                           nullptr,
                                           0);
    if (gpt_init_parameter_.pre_allocate_op_mem_) {
        parallel_gpt_->preAllocate();
    }
}

template<typename T>
FtGpt<T>::~FtGpt()
{
    ft::ftNcclParamDestroy(tensor_para_);
    ft::ftNcclParamDestroy(pipeline_para_);
    cublasLtDestroy(cublaslt_handle_);
    delete cublas_algo_map_;
    delete cublas_wrapper_mutex_;
    delete parallel_gpt_;
    delete cublas_wrapper_;
    delete allocator_;
}

template<typename T>
void FtGpt<T>::forward(th::Tensor&              decoder_output,
                       th::optional<th::Tensor> key_cache,
                       th::optional<th::Tensor> value_cache,
                       th::Tensor&              decoder_input,
                       th::Tensor&              input_lengths,
                       th::Tensor&              sequence_lengths,
                       th::Tensor&              block_index_map,
                       th::optional<th::Tensor> lora_ids,
                       th::optional<th::Tensor> attention_mask,
                       th::optional<th::Tensor> position_ids,
                       th::optional<th::Tensor> linear_bias_slopes,
                       th::optional<th::Tensor> prefix_prompt_lengths,
                       th::optional<th::Tensor> count_prefix_length,
                       th::optional<th::Tensor> max_prefix_length,
                       th::optional<th::Tensor> key_cache_scale,
                       th::optional<th::Tensor> value_cache_scale)
{
    auto stream        = at::cuda::getCurrentCUDAStream().stream();

    auto lora_input_lengths = input_lengths.clone().cpu().to(torch::kInt);
    int batch_size =  sequence_lengths.size(0);
    if (batch_size > 0 && lora_input_lengths.size(0) >= batch_size) {
        lora_input_lengths.slice(0, 0, batch_size) = 1;
    }

    ft::TensorMap input_tensors({{"decoder_input", convert_tensor<T>(decoder_input)},
                                 {"sequence_lengths", convert_tensor<int>(sequence_lengths)},
                                 {"input_lengths", convert_tensor<int>(input_lengths)},
                                 {"block_index_map", convert_tensor<int>(block_index_map)},
                                 {"lora_input_lengths", convert_tensor<int>(lora_input_lengths)}});
    if (attention_mask.has_value()) {
        input_tensors.insert("attention_mask", convert_tensor<T>(attention_mask.value()));
    }
    if (linear_bias_slopes.has_value()) {
        input_tensors.insert("linear_bias_slopes", convert_tensor<T>(linear_bias_slopes.value()));
    }
    if (count_prefix_length.has_value()) {        
        input_tensors.insert("count_prefix_length", convert_tensor<bool>(count_prefix_length.value()));
    }

    if (prefix_prompt_lengths.has_value()) {        
        input_tensors.insert("d_prefix_prompt_lengths", convert_tensor<int>(prefix_prompt_lengths.value()));
    }


    if (max_prefix_length.has_value()) {
        input_tensors.insert("max_prefix_prompt_length", convert_tensor<int>(max_prefix_length.value()));
    }

    ft::TensorMap output_tensors({{"decoder_output", convert_tensor<T>(decoder_output)}});
    if (gpt_init_parameter_.int8_kv_cache_ && key_cache.has_value() && value_cache.has_value()) {
        output_tensors.insert("key_cache", convert_tensor<int8_t>(key_cache.value()));
        output_tensors.insert("value_cache", convert_tensor<int8_t>(value_cache.value()));
    } else if (key_cache.has_value() && value_cache.has_value()) {
        output_tensors.insert("key_cache", convert_tensor<T>(key_cache.value()));
        output_tensors.insert("value_cache", convert_tensor<T>(value_cache.value()));
    }
    if (key_cache_scale.has_value()) {
        output_tensors.insert("key_cache_scale", convert_tensor<float>(key_cache_scale.value()));
        output_tensors.insert("value_cache_scale", convert_tensor<float>(value_cache_scale.value()));
    }
    // lora
    if (lora_ids.has_value()) {
        input_tensors.insert("lora_ids", convert_tensor<int>(lora_ids.value()));
    }

    // position_ids for rotary embedding
    if (position_ids.has_value()) {
        input_tensors.insert("position_ids", convert_tensor<int>(position_ids.value()));
    }
    
    parallel_gpt_->forward(&output_tensors, &input_tensors, &gpt_layer_weights_);
}

template<typename T>
void FtGpt<T>::addLoRA(const int                                                       lora_id,
                       const std::vector<std::unordered_map<std::string, th::Tensor>>& lora_a_weights,
                       const std::vector<std::unordered_map<std::string, th::Tensor>>& lora_b_weights)
{
    loadLoRAWeights<T>(
        gpt_init_parameter_.num_layers_, lora_id, lora_a_weights, lora_b_weights, gpt_lora_layer_weights_);
}
template<typename T>
void FtGpt<T>::removeLoRA(const int lora_id)
{
    removeLoRAWeights(lora_id, gpt_lora_layer_weights_);
}
template<typename T>
bool FtGpt<T>::UseFMHA()
{
    FT_CHECK_WITH_INFO(parallel_gpt_ != nullptr, "parallel_gpt_ should not be nullptr");
    return parallel_gpt_->UseFMHA();
}

ParallelGptOp::ParallelGptOp(c10::intrusive_ptr<GptInitParameter>                            gpt_init_parameter,
                             const int64_t                                                   tensor_para_size,
                             const int64_t                                                   pipeline_para_size,
                             const std::string                                               master_ip,
                             const int64_t                                                   master_port,
                             const std::vector<std::unordered_map<std::string, th::Tensor>>& weights):
    gpt_init_parameter_(*gpt_init_parameter),
    tensor_para_size_(tensor_para_size),
    pipeline_para_size_(pipeline_para_size),
    scalar_type_(getScalarType(gpt_init_parameter->data_type_))
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

#define CREATE_INSTANCE(T_)                                                                                            \
    gpt_ = new FtGpt<T_>(gpt_init_parameter_, tensor_para_size, pipeline_para_size, master_ip, master_port, weights);  \
    chunk_size_ = 16 / sizeof(T_)

    switch (scalar_type_) {
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

ParallelGptOp::~ParallelGptOp()
{
    delete gpt_;
}

th::Tensor ParallelGptOp::forward(th::Tensor               decoder_input,
                                  th::optional<th::Tensor> key_cache,
                                  th::optional<th::Tensor> value_cache,
                                  th::Tensor               input_lengths,
                                  th::Tensor               sequence_lengths,
                                  th::Tensor               block_index_map,
                                  th::optional<th::Tensor> lora_ids,
                                  th::optional<th::Tensor> attention_mask,
                                  th::optional<th::Tensor> position_ids,
                                  th::optional<th::Tensor> linear_bias_slopes,
                                  th::optional<th::Tensor> prefix_prompt_lengths,
                                  th::optional<th::Tensor> count_prefix_length,
                                  th::optional<th::Tensor> max_prefix_length,
                                  th::optional<th::Tensor> key_cache_scale,
                                  th::optional<th::Tensor> value_cache_scale)
{
    // Input Arguments:
    //     decoder_input [batch_size + context_batch_size, hidden_units], T
    //     attention_mask: [context_batch_size, 1, max_input_length, max_input_length], T
    //     input_lengths: [batch_size + context_batch_size], int
    //     sequence_lengths: [batch_size], int
    //     block_index_map [batch_size + context_batch_size, max_block_size]
    //     linear_bias_slopes_opt: [num_heads], optional
    // Output Arguments:
    //     decoder_output: [batch_size + context_batch_size, hidden_units]
    //     key_cache: [num_layers, batch_size * beam_width, local_num_heads, size_per_head / x, memory_length, x]
    //         x = 16 / sizeof(T), memory_length = max_input_length or max_input_length + gen_length
    //     value_cache: [num_layers, batch_size * beam_width, local_num_heads, memory_length, hidden_units]
    //         memory_length = max_input_length or max_input_length + gen_length

    CHECK_INPUT(decoder_input, scalar_type_);
    FT_CHECK_WITH_INFO(
        decoder_input.dim() == 2,
        ft::fmtstr("input_embeds is of shape (batch_size, hidden_size), "
                   "but got dim=%d shape=%s",
                   (int)decoder_input.dim(),
                   ft::vec2str(convert_shape(decoder_input)).c_str())
            .c_str());
    int batch_size       = sequence_lengths.size(0);
    int hidden_units     = decoder_input.size(1);
    int context_batch_size = input_lengths.size(0) - batch_size;

    // CHECK_INPUT(block_index_map, torch::kInt32);
    TORCH_CHECK(batch_size + context_batch_size > 0, "must input context decoder or decoder input");
    th::Tensor decoder_output =
        torch::zeros({(int64_t)(decoder_input.size(0)), (int64_t)hidden_units},
                     torch::dtype(scalar_type_).device(torch::kCUDA).requires_grad(false));
    // FT_LOG_WARNING("debug shape: %s %s %s", std::to_string(batch_size).c_str(), std::to_string(context_batch_size).c_str(), std::to_string(hidden_units).c_str());

    gpt_->forward(decoder_output,
                  key_cache,
                  value_cache,
                  decoder_input,
                  input_lengths,
                  sequence_lengths,
                  block_index_map,
                  lora_ids,
                  attention_mask,
                  position_ids,
                  linear_bias_slopes,
                  prefix_prompt_lengths,
                  count_prefix_length,
                  max_prefix_length,
                  key_cache_scale,
                  value_cache_scale);
    return decoder_output;
}

void ParallelGptOp::addLoRA(const int64_t                                                   lora_id,
                            const std::vector<std::unordered_map<std::string, th::Tensor>>& lora_a_weights,
                            const std::vector<std::unordered_map<std::string, th::Tensor>>& lora_b_weights)
{
    gpt_->addLoRA(lora_id, lora_a_weights, lora_b_weights);
}
void ParallelGptOp::removeLoRA(const int64_t lora_id)
{
    gpt_->removeLoRA(lora_id);
}
bool ParallelGptOp::UseFMHA()
{
    return gpt_->UseFMHA();
}

}  // namespace torch_ext

static auto fasterTransformerGptTHS =
#ifdef LEGACY_THS
    torch::jit::class_<torch_ext::ParallelGptOp>("FasterTransformerParallelGptOp")
#else
    torch::jit::class_<torch_ext::ParallelGptOp>("FasterTransformer", "ParallelGptOp")
#endif
        .def(torch::jit::init<c10::intrusive_ptr<GptInitParameter>,                         // gpt_init_parameter
                              int64_t,                  // tensor_para_size
                              int64_t,                  // pipeline_para_size
                              std::string,              // master_ip
                              int64_t,                  // master_port
                              std::vector<std::unordered_map<std::string, th::Tensor>>>()) 
        .def("forward", &torch_ext::ParallelGptOp::forward)
        .def("use_fmha", &torch_ext::ParallelGptOp::UseFMHA)
        .def("add_lora", &torch_ext::ParallelGptOp::addLoRA)
        .def("remove_lora", &torch_ext::ParallelGptOp::removeLoRA);
