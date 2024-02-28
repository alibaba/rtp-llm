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

#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGptDecoder.h"
#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGptDecoderLoRALayerWeight.h"
#include "src/fastertransformer/th_op/th_utils.h"

#include "src/fastertransformer/cuda/nccl/nccl_utils.h"
#include "src/fastertransformer/th_op/GptInitParameter.h"
#include "src/fastertransformer/utils/compiler_config.h"

namespace ft = fastertransformer;
namespace th = torch;
namespace torch_ext {

class IFtGptDecoder {
public:
    virtual ~IFtGptDecoder() {}

    virtual void forward(const int64_t            max_input_length,
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
                        th::optional<th::Tensor>& cache_indirection,
                        th::optional<th::Tensor>& linear_bias_slopes,
                        th::optional<th::Tensor>& prefix_lengths_opt,
                        th::optional<th::Tensor>& max_prefix_length_opt,
                        th::optional<th::Tensor>& block_index_map) = 0;
    virtual void addLoRA(const int                                                     lora_id,
                 const std::vector<std::unordered_map<std::string, th::Tensor>>& lora_a_weights,
                 const std::vector<std::unordered_map<std::string, th::Tensor>>& lora_b_weights) = 0;
    virtual void removeLoRA(const int lora_id) = 0;
};

template<typename T>
class FtGptDecoder: public IFtGptDecoder {

public:
    FtGptDecoder(const GptInitParameter&       gpt_init_parameter,
                 const int                     tensor_para_size,
                 const int                     pipeline_para_size,
                 const std::string&            master_ip,
                 const int                     master_port,
                 const std::vector<std::unordered_map<std::string, th::Tensor>>& weights);

    ~FtGptDecoder() override;

    void forward(const int64_t            max_input_length,
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
                 th::optional<th::Tensor>& cache_indirection,
                 th::optional<th::Tensor>& linear_bias_slopes,
                 th::optional<th::Tensor>& prefix_lengths_opt,
                 th::optional<th::Tensor>& max_prefix_length_opt,
                 th::optional<th::Tensor>& block_index_map) override;
    
    void addLoRA(const int                                                       lora_id,
                 const std::vector<std::unordered_map<std::string, th::Tensor>>& lora_a_weights,
                 const std::vector<std::unordered_map<std::string, th::Tensor>>& lora_b_weights) override;
    void removeLoRA(const int lora_id) override;

private:
    const GptInitParameter gpt_init_parameter_;

    ft::NcclParam tensor_para_;
    ft::NcclParam pipeline_para_;

    cublasLtHandle_t      cublaslt_handle_;
    std::mutex*           cublas_wrapper_mutex_;
    ft::cublasAlgoMap*    cublas_algo_map_;
    struct cudaDeviceProp prop_;

    std::vector<ft::ParallelGptDecoderLoRALayerWeight<T>*> gpt_lora_layer_weights_;
    std::vector<ft::ParallelGptDecoderLayerWeight<T>*> gpt_layer_weights_;
};

class ParallelGptDecoderOp: public th::jit::CustomClassHolder {
public:
    ParallelGptDecoderOp(c10::intrusive_ptr<GptInitParameter>                            gpt_init_parameter,
                         const int64_t                                                   tensor_para_size,
                         const int64_t                                                   pipeline_para_size,
                         std::string                                                     master_ip,
                         const int64_t                                                   master_port,
                         const std::vector<std::unordered_map<std::string, th::Tensor>>& weights);

    ~ParallelGptDecoderOp();

    std::vector<th::Tensor> forward(const int64_t            max_input_length,
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
                                    th::optional<th::Tensor> block_index_map_opt);
    
    void addLoRA(const int64_t                                                   lora_id,
                 const std::vector<std::unordered_map<std::string, th::Tensor>>& lora_a_weights,
                 const std::vector<std::unordered_map<std::string, th::Tensor>>& lora_b_weights);
    void removeLoRA(const int64_t lora_id);

private:
    GptInitParameter        gpt_init_parameter_;

    at::ScalarType          scalar_type_;
    IFtGptDecoder*          gpt_decoder_;
};

}  // namespace torch_ext
