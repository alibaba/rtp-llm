/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/th_op/multi_gpu_gpt/ParallelGptContextDecoderOp.h"
#include "src/fastertransformer/th_op/multi_gpu_gpt/Base.h"
#include "src/fastertransformer/cuda/cuda_utils.h"

namespace th = torch;
namespace ft = fastertransformer;

namespace torch_ext {

template<typename T>
FtGptContextDecoder<T>::FtGptContextDecoder(const GptInitParameter&       gpt_init_parameter,
                                            const int                     tensor_para_size,
                                            const int                     pipeline_para_size,
                                            const std::string&            master_ip,
                                            const int                     master_port,
                                            const std::vector<std::unordered_map<std::string, th::Tensor>>& weights,
                                            const bool                    remove_padding):
    gpt_init_parameter_(gpt_init_parameter),
    remove_padding_(remove_padding)
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
                                        gpt_init_parameter_.int8_mode_,
                                        gpt_init_parameter_.int4_mode_,
                                        weights,
                                        &gpt_lora_layer_weights_);
}

template<typename T>
FtGptContextDecoder<T>::~FtGptContextDecoder()
{
    ft::ftNcclParamDestroy(tensor_para_);
    ft::ftNcclParamDestroy(pipeline_para_);
    cublasLtDestroy(cublaslt_handle_);
    delete cublas_algo_map_;
    delete cublas_wrapper_mutex_;
}

template<typename T>
void FtGptContextDecoder<T>::forward(th::Tensor&               decoder_output,
                                     th::Tensor&               key_cache,
                                     th::Tensor&               value_cache,
                                     th::Tensor&               last_token_hidden_states,
                                     th::Tensor&               input_embeds,
                                     th::Tensor&               attention_mask,
                                     th::Tensor&               input_lengths,
                                     th::Tensor&               lora_ids,
                                     th::optional<th::Tensor>& compact_idx,
                                     th::optional<th::Tensor>& batch_to_compact_idx,
                                     th::optional<th::Tensor>& linear_bias_slopes,
                                     th::optional<th::Tensor>& prefix_prompt_opt,
                                     th::optional<th::Tensor>& prefix_lengths_opt,
                                     th::optional<th::Tensor>& block_index_map)
{
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

    ft::AttentionType attention_type = ft::getAttentionType<T>(gpt_init_parameter_.size_per_head_,
                                                               ft::getSMVersion(),
                                                               remove_padding_,  // remove_padding
                                                               0,                // gpt supports any-seq-length fmha
                                                               true,             // is_fuse
                                                               false,            // with_relative_position_bias
                                                               true);            // causal_mask

    ft::ParallelGptContextDecoder<T> gpt_context_decoder(0,
                                                         0,
                                                         gpt_init_parameter_,
                                                         tensor_para_,
                                                         pipeline_para_,
                                                         stream,
                                                         &cublas_wrapper,
                                                         &allocator,
                                                         false,
                                                         true,
                                                         attention_type,
                                                         false,
                                                         nullptr,
                                                         0);

    auto lora_input_lengths = input_lengths.clone().cpu().to(torch::kInt);

    ft::TensorMap input_tensors({{"decoder_input", convert_tensor<T>(input_embeds)},
                                 {"attention_mask", convert_tensor<T>(attention_mask)},
                                 {"input_lengths", convert_tensor<int>(input_lengths)},
                                 {"lora_input_lengths", convert_tensor<int>(lora_input_lengths)}});

    if (compact_idx.has_value() || batch_to_compact_idx.has_value()) {
        FT_CHECK_WITH_INFO(
            compact_idx.has_value() && batch_to_compact_idx.has_value(),
            "Please provide both compact_idx and batch_to_compact_idx to enable shared context feature.");
        input_tensors.insert("compact_idx", convert_tensor<int>(compact_idx.value()));
        input_tensors.insert("batch_to_compact_idx", convert_tensor<int>(batch_to_compact_idx.value()));
    }
    if (linear_bias_slopes.has_value()) {
        input_tensors.insert("linear_bias_slopes", convert_tensor<T>(linear_bias_slopes.value()));
    }
    if (block_index_map.has_value()) {
        input_tensors.insert({"block_index_map", convert_tensor<T>(block_index_map.value())});
    }
    if (prefix_prompt_opt.has_value()) {
        size_t batch_size = input_embeds.size(0);
        FT_CHECK_WITH_INFO(prefix_lengths_opt.has_value(), "prefix_length should not be empty!");
        std::vector<const T*> prefix_prompt_weight_batch_ptrs;
        // do broadcast for every batch_size
        if (prefix_prompt_opt.value().size(0) == 1) {
            for (auto i = 0; i < batch_size; i++) {
                prefix_prompt_weight_batch_ptrs.push_back(get_ptr<T>(prefix_prompt_opt.value()));
            }
        } else {
            FT_CHECK_WITH_INFO(prefix_prompt_opt.value().size(0) == batch_size, "prefix prompt length should equal to batch size");
            for (auto i = 0; i < batch_size; i++) {
                // batch_idx * head_num * max_propt_length * size_per_head
                auto bias = i * prefix_prompt_opt.value().size(1) * prefix_prompt_opt.value().size(2) * prefix_prompt_opt.value().size(3);
                prefix_prompt_weight_batch_ptrs.push_back(get_ptr<T>(prefix_prompt_opt.value()) + bias);
            }
        }

        const T** prompt_learning_weight_batch = (const T**)(allocator.reMalloc(prompt_learning_weight_batch, sizeof(T*) * batch_size, false));
        cudaMemcpyAsync(prompt_learning_weight_batch,
                        prefix_prompt_weight_batch_ptrs.data(),
                        sizeof(T*) * batch_size,
                        cudaMemcpyDefault,
                        stream);
        sync_check_cuda_error();

        input_tensors.insert({"d_prefix_prompt_batch", ft::Tensor{ft::MEMORY_GPU, ft::getTensorType<T>(), {batch_size}, prompt_learning_weight_batch}});
        input_tensors.insert({"d_prefix_prompt_lengths", convert_tensor<T>(prefix_lengths_opt.value())});
    }

    input_tensors.insert({"lora_ids", convert_tensor<int>(lora_ids)});

    ft::TensorMap output_tensors({{"decoder_output", convert_tensor<T>(decoder_output)},
                                  {"key_cache", convert_tensor<T>(key_cache)},
                                  {"value_cache", convert_tensor<T>(value_cache)},
                                  {"last_token_hidden_units", convert_tensor<T>(last_token_hidden_states)}});

    gpt_context_decoder.forward(&output_tensors, &input_tensors, &gpt_layer_weights_);
}

template<typename T>
void FtGptContextDecoder<T>::addLoRA(const int                                                       lora_id,
                                     const std::vector<std::unordered_map<std::string, th::Tensor>>& lora_a_weights,
                                     const std::vector<std::unordered_map<std::string, th::Tensor>>& lora_b_weights)
{
    loadLoRAWeights<T>(
        gpt_init_parameter_.num_layers_, lora_id, lora_a_weights, lora_b_weights, gpt_lora_layer_weights_);
}
template<typename T>
void FtGptContextDecoder<T>::removeLoRA(const int lora_id)
{
    removeLoRAWeights(lora_id, gpt_lora_layer_weights_);
}

ParallelGptContextDecoderOp::ParallelGptContextDecoderOp(
    c10::intrusive_ptr<GptInitParameter>                            gpt_init_parameter,
    const int64_t                                                   tensor_para_size,
    const int64_t                                                   pipeline_para_size,
    const std::string                                               master_ip,
    const int64_t                                                   master_port,
    const std::vector<std::unordered_map<std::string, th::Tensor>>& weights,
    const bool                                                      remove_padding):
    gpt_init_parameter_(*gpt_init_parameter),
    tensor_para_size_(tensor_para_size),
    pipeline_para_size_(pipeline_para_size),
    scalar_type_(getScalarType(gpt_init_parameter->data_type_))
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    for (size_t i = 0; i < weights.size(); i++) {
        if (scalar_type_ == torch::kInt8){
            FT_LOG_ERROR("scalar type int8");
        }

        if (gpt_init_parameter_.int8_mode_ == 1) {
            // TORCH_CHECK(scalar_type_ != torch::kFloat32, "Int8 weight only quant does not work for FP32.");
        }
    }

#define CREATE_INSTANCE(T_)                                                                                            \
    gpt_context_decoder_ = new FtGptContextDecoder<T_>(gpt_init_parameter_,                                            \
                                                       tensor_para_size,                                               \
                                                       pipeline_para_size,                                             \
                                                       master_ip,                                                      \
                                                       master_port,                                                    \
                                                       weights,                                                        \
                                                       remove_padding);                                                \
    chunk_size_          = 16 / sizeof(T_)

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

ParallelGptContextDecoderOp::~ParallelGptContextDecoderOp()
{
    delete gpt_context_decoder_;
}

std::vector<th::Tensor> ParallelGptContextDecoderOp::forward(th::Tensor               input_embeds,
                                                             th::Tensor               attention_mask,
                                                             th::Tensor               input_lengths,
                                                             th::Tensor               lora_names,
                                                             th::optional<int64_t>    memory_length_opt,
                                                             th::optional<th::Tensor> compact_idx_opt,
                                                             th::optional<th::Tensor> batch_to_compact_idx_opt,
                                                             th::optional<th::Tensor> linear_bias_slopes_opt,
                                                             th::optional<th::Tensor> prefix_prompt_opt,
                                                             th::optional<th::Tensor> prefix_lengths_opt,
                                                             th::optional<th::Tensor> key_cache,
                                                             th::optional<th::Tensor> value_cache,
                                                             th::optional<th::Tensor> block_index_map)
{
    // Input Arguments:
    //     input_embeds: [batch_size * beam_width, max_input_length, hidden_units], T
    //     attention_mask: [batch_size * beam_width, 1, max_input_length, max_input_length], T
    //     input_lengths: [batch_size * beam_width], int
    //     memory_length_opt: scalar, optional
    //     compact_idx_opt: [compact_batchxbeam], int, optional
    //     batch_to_compact_idx_opt: [batch_size * beam_width], int, optional
    //     linear_bias_slopes_opt: [num_heads], optional
    // Output Arguments:
    //     decoder_output: [batch_size * beam_width, max_input_length, hidden_units]
    //     key_cache: [num_layers, batch_size * beam_width, local_num_heads, size_per_head / x, memory_length, x]
    //         x = 16 / sizeof(T), memory_length = max_input_length or max_input_length + gen_length
    //     value_cache: [num_layers, batch_size * beam_width, local_num_heads, memory_length, hidden_units]
    //         memory_length = max_input_length or max_input_length + gen_length
    //     last_token_hidden_states: [batch_size * beam_width, hidden_units]

    CHECK_INPUT(input_embeds, scalar_type_);
    FT_CHECK_WITH_INFO(
        input_embeds.dim() == 3,
        ft::fmtstr("input_embeds is of shape (batch_size * beam_width, max_input_length, hidden_size), "
                   "but got dim=%d shape=%s",
                   (int)input_embeds.dim(),
                   ft::vec2str(convert_shape(input_embeds)).c_str())
            .c_str());
    CHECK_INPUT(attention_mask, scalar_type_);
    CHECK_INPUT(input_lengths, torch::kInt32);

    if (compact_idx_opt.has_value()) {
        CHECK_INPUT(compact_idx_opt.value(), torch::kInt32);
    }
    if (batch_to_compact_idx_opt.has_value()) {
        CHECK_INPUT(batch_to_compact_idx_opt.value(), torch::kInt32);
    }
    
    CHECK_OPTIONAL_INPUT(prefix_prompt_opt, scalar_type_);
    CHECK_OPTIONAL_INPUT(prefix_lengths_opt, torch::kInt32);
    
    int batch_size       = input_embeds.size(0);
    int max_input_length = input_embeds.size(1);
    int hidden_units     = input_embeds.size(2);

    th::Tensor decoder_output = torch::empty_like(input_embeds);
    th::Tensor last_token_hidden_states =
        torch::empty({(int64_t)batch_size, (int64_t)hidden_units},
                     torch::dtype(scalar_type_).device(torch::kCUDA).requires_grad(false));

    int mem_length = memory_length_opt.has_value() ? memory_length_opt.value() : max_input_length;

    int local_num_heads_kv = gpt_init_parameter_.head_num_kv_;
    if (local_num_heads_kv > 1) {
        local_num_heads_kv = gpt_init_parameter_.head_num_kv_ / tensor_para_size_;
    }
    th::Tensor key_cache_, value_cache_;
    if (key_cache.has_value()) {
        key_cache_ = key_cache.value();
    } else {
        key_cache_ = torch::zeros({static_cast<long int>(gpt_init_parameter_.num_layers_ / pipeline_para_size_),
                static_cast<long int>(batch_size),
                static_cast<long int>(local_num_heads_kv),
                static_cast<long int>(mem_length),
                static_cast<long int>(gpt_init_parameter_.size_per_head_)},
            torch::dtype(scalar_type_).device(torch::kCUDA).requires_grad(false));
    }

    if (ft::should_print()) {
        printf("key_cache shape = %s\n", ft::vec2str(convert_shape(key_cache_)).c_str());
    }

    if (value_cache.has_value()) {
        value_cache_ = value_cache.value();
    } else {
        value_cache_ = torch::zeros({static_cast<long int>(gpt_init_parameter_.num_layers_ / pipeline_para_size_),
                static_cast<long int>(batch_size),
                static_cast<long int>(local_num_heads_kv),
                static_cast<long int>(mem_length),
                static_cast<long int>(gpt_init_parameter_.size_per_head_)},
            torch::dtype(scalar_type_).device(torch::kCUDA).requires_grad(false));
    }

    if (ft::should_print()) {
        printf("value cache shape = %s\n", ft::vec2str(convert_shape(value_cache_)).c_str());
    }

    gpt_context_decoder_->forward(decoder_output,
                                  key_cache_,
                                  value_cache_,
                                  last_token_hidden_states,
                                  input_embeds,
                                  attention_mask,
                                  input_lengths,
                                  lora_names,
                                  compact_idx_opt,
                                  batch_to_compact_idx_opt,
                                  linear_bias_slopes_opt,
                                  prefix_prompt_opt,
                                  prefix_lengths_opt,
                                  block_index_map);

    return std::vector<th::Tensor>{decoder_output, key_cache_, value_cache_, last_token_hidden_states};
}



void ParallelGptContextDecoderOp::addLoRA(
    const int64_t                                                   lora_id,
    const std::vector<std::unordered_map<std::string, th::Tensor>>& lora_a_weights,
    const std::vector<std::unordered_map<std::string, th::Tensor>>& lora_b_weights)
{
    gpt_context_decoder_->addLoRA(lora_id, lora_a_weights, lora_b_weights);
}

void ParallelGptContextDecoderOp::removeLoRA(const int64_t lora_id)
{
    gpt_context_decoder_->removeLoRA(lora_id);
}


}  // namespace torch_ext

static auto fasterTransformerGptContextDecoderTHS =
#ifdef LEGACY_THS
    torch::jit::class_<torch_ext::ParallelGptContextDecoderOp>("FasterTransformerParallelGptContextDecoderOp")
#else
    torch::jit::class_<torch_ext::ParallelGptContextDecoderOp>("FasterTransformer", "ParallelGptContextDecoderOp")
#endif
        .def(torch::jit::init<c10::intrusive_ptr<GptInitParameter>,                      // gpt_init_parameter
                              int64_t,                  // tensor_para_size
                              int64_t,                  // pipeline_para_size
                              std::string,              // master_ip
                              int64_t,                  // master_port
                              std::vector<std::unordered_map<std::string, th::Tensor>>,  // weights
                              bool>())                  // remove_padding
        .def("forward", &torch_ext::ParallelGptContextDecoderOp::forward)
        .def("add_lora", &torch_ext::ParallelGptContextDecoderOp::addLoRA)
        .def("remove_lora", &torch_ext::ParallelGptContextDecoderOp::removeLoRA);
