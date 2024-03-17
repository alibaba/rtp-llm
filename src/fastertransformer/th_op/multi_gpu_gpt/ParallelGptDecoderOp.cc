/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/th_op/multi_gpu_gpt/ParallelGptDecoderOp.h"
#include "src/fastertransformer/th_op/multi_gpu_gpt/Base.h"

namespace th = torch;
namespace ft = fastertransformer;

namespace torch_ext {

template<typename T>
FtGptDecoder<T>::FtGptDecoder(const GptInitParameter&       gpt_init_parameter,
                              const int                     tensor_para_size,
                              const int                     pipeline_para_size,
                              const std::string&            master_ip,
                              const int                     master_port,
                              const std::vector<std::unordered_map<std::string, th::Tensor>>& weights):
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

}

template<typename T>
FtGptDecoder<T>::~FtGptDecoder()
{
    ft::ftNcclParamDestroy(tensor_para_);
    ft::ftNcclParamDestroy(pipeline_para_);
    cublasLtDestroy(cublaslt_handle_);
    delete cublas_algo_map_;
    delete cublas_wrapper_mutex_;
}

template<typename T>
void FtGptDecoder<T>::forward(const int64_t            max_input_length,
                              const int64_t            step,
                              const int64_t            ite,
                              th::Tensor&              input_embeds,
                              th::Tensor&              sequence_lengths,
                              th::Tensor&              finished,
                              th::Tensor&              input_lengths,
                              th::Tensor&              decoder_output,
                              th::Tensor&              key_cache,
                              th::Tensor&              value_cache,
                              th::Tensor&              lora_ids,
                              th::optional<th::Tensor>& masked_tokens,
                              th::optional<th::Tensor>& cache_indirection_opt,
                              th::optional<th::Tensor>& linear_bias_slopes_opt,
                              th::optional<th::Tensor>& prefix_lengths_opt,
                              th::optional<th::Tensor>& max_prefix_length_opt,
                              th::optional<th::Tensor>& block_index_map_opt)
{
    // Input Arguments:
    //     input_embeds: [local_batch_size * beam_width, hidden_units], T
    //     sequence_lengths: [local_batch_size * beam_width], int
    //     finished: [local_batch_size * beam_width], bool
    //     input_lengths: [local_batch_size * beam_width], int, optional
    //     masked_tokens [local_batch_size * beam_width, memory_length]
    //     decoder_output: [local_batch_size * beam_width, max_input_length, hidden_units]
    //     key_cache: [num_layers, batch_size * beam_width, local_num_heads, size_per_head / x, memory_length, x]
    //         x = 16 / sizeof(T)
    //     value_cache: [num_layers, batch_size * beam_width, local_num_heads, memory_length, hidden_units]
    //     cache_indirection [local_batch_size, beam_width, memory_length], int, optional.
    //     linear_bias_slopes_opt: [num_heads], optional

    auto stream        = at::cuda::getCurrentCUDAStream().stream();
    auto cublas_handle = at::cuda::getCurrentCUDABlasHandle();
    cublasSetStream(cublas_handle, stream);

    ft::Allocator<ft::AllocatorType::TH> allocator;
    allocator.setStream(stream);

    ft::cublasMMWrapper cublas_wrapper(
        cublas_handle, cublaslt_handle_, stream, cublas_algo_map_, cublas_wrapper_mutex_, &allocator);

    if (std::is_same<T, half>::value) {
        cublas_wrapper.setGemmConfig(CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F);
    }
    else if constexpr (std::is_same<T, __nv_bfloat16>::value && CompileConfig::enable_bf16) {
        cublas_wrapper.setBF16GemmConfig();
    }

    else if (std::is_same<T, float>::value) {
        cublas_wrapper.setFP32GemmConfig();
    }

    const int  _max_input_length = static_cast<int>(max_input_length);
    const int  _step             = static_cast<int>(step);
    const uint _ite              = static_cast<uint>(ite);

    ft::ParallelGptDecoder<T> gpt_decoder(0,
                                          gpt_init_parameter_,
                                          tensor_para_,
                                          pipeline_para_,
                                          stream,
                                          &cublas_wrapper,
                                          &allocator,
                                          false,
                                          false,
                                          nullptr,
                                          0);
    
    auto lora_input_lengths = torch::ones_like(input_lengths).clone().cpu().to(torch::kInt);

    std::unordered_map<std::string, ft::Tensor> input_tensors{
        {"decoder_input", convert_tensor<T>(input_embeds)},
        {"finished", convert_tensor<bool>(finished)},
        {"sequence_lengths", convert_tensor<int>(sequence_lengths)},
        {"input_lengths", convert_tensor<int>(input_lengths)},
        {"lora_input_lengths", convert_tensor<int>(lora_input_lengths)},
        {"max_input_length", ft::Tensor(ft::MEMORY_CPU, ft::TYPE_INT32, {1}, &_max_input_length)},
        {"step", ft::Tensor(ft::MEMORY_CPU, ft::TYPE_INT32, {1}, &_step)},
        {"ite", ft::Tensor(ft::MEMORY_CPU, ft::TYPE_INT32, {1}, &_ite)}};
    if (cache_indirection_opt.has_value()) {
        FT_CHECK_WITH_INFO(
            cache_indirection_opt.value().dim() == 3,
            ft::fmtstr("cache_indirection assumes to be of shape (batch_size, beam_width, memory_length), "
                       "but got %s",
                       ft::vec2str(convert_shape(cache_indirection_opt.value())).c_str()));
        input_tensors.insert({"cache_indirection", convert_tensor<int>(cache_indirection_opt.value())});
    }
    if (masked_tokens.has_value()) {
        input_tensors.insert({"masked_tokens", convert_tensor<bool>(masked_tokens.value())});
    }
    if (linear_bias_slopes_opt.has_value()) {
        input_tensors.insert({"linear_bias_slopes", convert_tensor<T>(linear_bias_slopes_opt.value())});
    }
    if (block_index_map_opt.has_value()) {
        input_tensors.insert({"block_index_map", convert_tensor<T>(block_index_map_opt.value())});
    }
    if (prefix_lengths_opt.has_value()) {
        FT_CHECK_WITH_INFO(max_prefix_length_opt.has_value(), "max_prefix_length_opt should not be empty!");

        input_tensors.insert({"d_prefix_prompt_lengths", convert_tensor<int>(prefix_lengths_opt.value())});
        input_tensors.insert({"max_prefix_prompt_length", convert_tensor<int>(max_prefix_length_opt.value())});
    }

    input_tensors.insert({"lora_ids", convert_tensor<int>(lora_ids)});

    std::unordered_map<std::string, ft::Tensor> output_tensors{{"decoder_output", convert_tensor<T>(decoder_output)},
                                                               {"key_cache", convert_tensor<T>(key_cache)},
                                                               {"value_cache", convert_tensor<T>(value_cache)}};

    gpt_decoder.forward(&output_tensors, &input_tensors, &gpt_layer_weights_);
}

template<typename T>
void FtGptDecoder<T>::addLoRA(const int                                                       lora_id,
                              const std::vector<std::unordered_map<std::string, th::Tensor>>& lora_a_weights,
                              const std::vector<std::unordered_map<std::string, th::Tensor>>& lora_b_weights)
{
    loadLoRAWeights<T>(
        gpt_init_parameter_.num_layers_, lora_id, lora_a_weights, lora_b_weights, gpt_lora_layer_weights_);
}
template<typename T>
void FtGptDecoder<T>::removeLoRA(const int lora_id)
{
    removeLoRAWeights(lora_id, gpt_lora_layer_weights_);
}

ParallelGptDecoderOp::ParallelGptDecoderOp(
    c10::intrusive_ptr<GptInitParameter>                            gpt_init_parameter,
    const int64_t                                                   tensor_para_size,
    const int64_t                                                   pipeline_para_size,
    std::string                                                     master_ip,
    const int64_t                                                   master_port,
    const std::vector<std::unordered_map<std::string, th::Tensor>>& weights):
    gpt_init_parameter_(*gpt_init_parameter), 
    scalar_type_(getScalarType(gpt_init_parameter->data_type_))
{
    for (size_t i = 0; i < weights.size(); i++) {
        // for (auto weight : weights[i]) {
        //     CHECK_INPUT(weight.second, scalar_type_);
        // }

        if (gpt_init_parameter_.quant_algo_->int8_mode_ == 1) {
            TORCH_CHECK(scalar_type_ != torch::kFloat32, "Int8 weight only quant does not work for FP32.");
            // for (auto quant_weight : quant_weights[i]) {
            //     CHECK_INPUT(quant_weight.second, torch::kInt8);
            // }

            // for (auto quant_scale : quant_scales[i]) {
            //     CHECK_INPUT(quant_scale.second, scalar_type_);
            // }
        }
    }

#define CREATE_INSTANCE(T_)                                                                                            \
    gpt_decoder_ = new FtGptDecoder<T_>(gpt_init_parameter_,                                                           \
                                        tensor_para_size,                                                              \
                                        pipeline_para_size,                                                            \
                                        master_ip,                                                                     \
                                        master_port,                                                                   \
                                        weights)                                                                       

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

ParallelGptDecoderOp::~ParallelGptDecoderOp()
{
    delete gpt_decoder_;
}

std::vector<th::Tensor> ParallelGptDecoderOp::forward(const int64_t            max_input_length,
                                                      const int64_t            step,
                                                      const int64_t            ite,
                                                      th::Tensor               input_embeds,
                                                      th::Tensor               sequence_lengths,
                                                      th::Tensor               finished,
                                                      th::Tensor               input_lengths,
                                                      th::Tensor               key_cache,
                                                      th::Tensor               value_cache,
                                                      th::Tensor               lora_ids,
                                                      th::optional<th::Tensor> masked_tokens,
                                                      th::optional<th::Tensor> cache_indirection_opt,
                                                      th::optional<th::Tensor> linear_bias_slopes_opt,
                                                      th::optional<th::Tensor> prefix_lengths_opt,
                                                      th::optional<th::Tensor> max_prefix_length_opt,
                                                      th::optional<th::Tensor> block_index_map_opt)
{
    CHECK_INPUT(input_embeds, scalar_type_);
    CHECK_INPUT(finished, torch::kBool);
    CHECK_INPUT(sequence_lengths, torch::kInt32);
    CHECK_INPUT(input_lengths, torch::kInt32);
    // CHECK_INPUT(masked_tokens, torch::kBool);
    CHECK_INPUT(key_cache, scalar_type_);
    CHECK_INPUT(value_cache, scalar_type_);
    CHECK_OPTIONAL_INPUT(prefix_lengths_opt, torch::kInt32);

    if (cache_indirection_opt.has_value()) {
        CHECK_INPUT(cache_indirection_opt.value(), torch::kInt32);
    }
    if (linear_bias_slopes_opt.has_value()) {
        CHECK_INPUT(linear_bias_slopes_opt.value(), scalar_type_);
    }

    th::Tensor decoder_output = torch::empty_like(input_embeds);

    gpt_decoder_->forward(max_input_length,
                          step,
                          ite,
                          input_embeds,
                          sequence_lengths,
                          finished,
                          input_lengths,
                          decoder_output,
                          key_cache,
                          value_cache,
                          lora_ids,
                          masked_tokens,
                          cache_indirection_opt,
                          linear_bias_slopes_opt,
                          prefix_lengths_opt,
                          max_prefix_length_opt,
                          block_index_map_opt);
    return std::vector<th::Tensor>{decoder_output};
}

void ParallelGptDecoderOp::addLoRA(const int64_t                                                   lora_id,
                                   const std::vector<std::unordered_map<std::string, th::Tensor>>& lora_a_weights,
                                   const std::vector<std::unordered_map<std::string, th::Tensor>>& lora_b_weights)
{
    gpt_decoder_->addLoRA(lora_id, lora_a_weights, lora_b_weights);
}
void ParallelGptDecoderOp::removeLoRA(const int64_t lora_id)
{
    gpt_decoder_->removeLoRA(lora_id);
}

}  // namespace torch_ext

static auto fasterTransformerGptDecoderTHS =
#ifdef LEGACY_THS
    torch::jit::class_<torch_ext::ParallelGptDecoderOp>("FasterTransformerParallelGptDecoderOp")
#else
    torch::jit::class_<torch_ext::ParallelGptDecoderOp>("FasterTransformer", "ParallelGptDecoderOp")
#endif
        .def(torch::jit::init<c10::intrusive_ptr<GptInitParameter>,                         // gpt_init_parameter
                              int64_t,                     // tensor_para_size
                              int64_t,                     // pipeline_para_size
                              std::string,                 // master_ip
                              int64_t,                     // master_port
                              std::vector<std::unordered_map<std::string, th::Tensor>>>())  // quant_pre_scales
        .def("forward", &torch_ext::ParallelGptDecoderOp::forward)
        .def("add_lora", &torch_ext::ParallelGptDecoderOp::addLoRA)
        .def("remove_lora", &torch_ext::ParallelGptDecoderOp::removeLoRA);
