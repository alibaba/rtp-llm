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

#pragma once

#include "src/fastertransformer/cutlass/interface.h"
#include "src/fastertransformer/kernels/activation_kernels.h"
#include "src/fastertransformer/kernels/matrix_vector_multiplication.h"
#include "src/fastertransformer/layers/BaseLayer.h"
#include "src/fastertransformer/layers/FfnWeight.h"
#include "src/fastertransformer/layers/GemmRunner.h"
#include "src/fastertransformer/layers/LoraGemm.h"
#include "src/fastertransformer/utils/activation_types.h"
#include "src/fastertransformer/cuda/cuda_utils.h"
#include "src/fastertransformer/cuda/memory_utils.h"
#include "src/fastertransformer/kernels/alpha_layernorm_kernels.h"
#include "src/fastertransformer/trt_plugins/mixtureOfExperts/mixtureOfExpertsPlugin.h"
#include "src/fastertransformer/utils/layernorm_types.h"
#include <stdint.h>
#include <vector>

namespace fastertransformer {

template<typename T>
class FfnLayer: public BaseLayer {
private:
    // buffer handling
    size_t max_token_num_ = 0;

    // meta data
    size_t expert_num_    = 0;
    size_t moe_k_         = 0;

    // calculated data
    size_t hidden_units_ = 0;

    std::shared_ptr<LoraGemm<T>>   lora_gemm_;
    std::shared_ptr<GemmRunner<T>> gemm_runner_;

    std::shared_ptr<tensorrt_llm::plugins::MixtureOfExpertsPlugin> moe_plugin_;

    void allocateBuffer() override;
    void freeBuffer() override;
    void allocateBuffer(size_t token_num, int moe_k = 0, bool use_moe = false);

protected:
    T*     inter_buf_        = nullptr;
    T*     inter_buf_2_      = nullptr;  // for gated activation
    T*     inter_buf_normed_ = nullptr;
    float* moe_gates_buf_    = nullptr;
    char*  moe_fc_workspace_ = nullptr;

    char*  mixed_gemm_workspace_ = nullptr;
    size_t mixed_gemm_ws_bytes_  = 0;

    size_t inter_size_         = 0;
    size_t inter_padding_size_ = 0;

    const bool                 is_sparse_head_ = false;
    const std::vector<int64_t> local_layer_inter_size_;
    const std::vector<int64_t> local_layer_inter_padding_size_;

    tc::QuantAlgo quant_algo_;

    ActivationType activation_type_ = ActivationType::InvalidType;
    bool has_moe_norm_ = false;

    bool use_gated_activation_ = false;

    float layernorm_eps_ = 0;

    ActivationType getActivationType() const {
        return activation_type_;
    };

    void genericActivation(int          layer_id,
                           int          m,
                           const T*     bias1,
                           const T*     bias2,
                           const int*   ia3_tasks,
                           const T*     ia3_weights,
                           const float* activation_in,
                           const float* activation_out,
                           const int*   padding_offset,
                           const int    seq_len);

public:
    FfnLayer(size_t               max_batch_size,
             size_t               max_seq_len,
             size_t               hidden_units,
             size_t               expert_num,
             size_t               moe_k,
             size_t               inter_size,
             size_t               inter_padding_size,
             std::vector<int64_t> local_layer_inter_size,
             std::vector<int64_t> local_layer_inter_padding_size,
             cudaStream_t         stream,
             cublasMMWrapper*     cublas_wrapper,
             tc::QuantAlgo        quant_algo,
             IAllocator*          allocator,
             bool                 is_free_buffer_after_forward,
             bool                 sparse,
             bool                 is_sparse_head,
             ActivationType       activation_type,
             bool                 has_moe_norm,
             float                layernorm_eps);

    virtual ~FfnLayer();

    virtual void forward(std::vector<fastertransformer::Tensor>*       output_tensors,
                         const std::vector<fastertransformer::Tensor>* input_tensors,
                         const FfnWeight<T>*                           ffn_weights);
    virtual void forward(TensorMap* output_tensors, TensorMap* input_tensors, const FfnWeight<T>* ffn_weights);
    virtual void preAllocate();
};

}  // namespace fastertransformer
