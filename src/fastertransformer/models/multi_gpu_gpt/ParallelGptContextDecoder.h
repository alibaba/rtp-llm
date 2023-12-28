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

#include <vector>

#include "src/fastertransformer/kernels/add_residual_kernels.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/layers/BaseLayer.h"
#include "src/fastertransformer/layers/TensorParallelFfnLayer.h"
#include "src/fastertransformer/layers/attention_layers/TensorParallelGptContextAttentionLayer.h"
#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGptDecoderLayerWeight.h"
#include "src/fastertransformer/models/multi_gpu_gpt/NormWrapper.h"
#include "src/fastertransformer/utils/Tensor.h"
#include "src/fastertransformer/utils/allocator.h"
#include "src/fastertransformer/utils/cublasMMWrapper.h"
#include "src/fastertransformer/utils/custom_ar_comm.h"
#include "src/fastertransformer/th_op/GptInitParameter.h"

namespace fastertransformer {

template<typename T>
class ParallelGptContextDecoder: public BaseLayer {
private:
    // buffer handling
    size_t max_batch_size_ = 0;
    size_t max_seq_len_    = 0;

    // meta data
    const GptInitParameter& gpt_init_parameter_;

    NcclParam tensor_para_;
    NcclParam pipeline_para_;

    std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm_;
    int                                 enable_custom_all_reduce_;
    AttentionType                       attention_type_;

    // NOTE (perkzz): dynamic_quant enabled
    bool   dynamic_quant_                  = true;
    float* attention_query_dynamic_scale_  = nullptr;
    float* ffn_intermediate_dynamic_scale_ = nullptr;

    bool is_qk_buf_float_;

    std::unique_ptr<BaseAttentionLayer<T>> self_attention_layer_;
    std::unique_ptr<FfnLayer<T>>           ffn_layer_;
    std::unique_ptr<NormWrapper<T>>        norm_wrapper_;

    void allocateBuffer() override;
    void allocateBuffer(size_t batch_size, size_t seq_len, bool use_shared_contexts, bool reuse_buf, bool pre_attn_norm);
    void freeBuffer() override;
    bool isValidLayerParallelId(uint l);
    void initialize();
    bool isFirstLayerParallelId(uint l);
    bool isLastLayerParallelId(uint l);
    int  getFirstLayerParallelId();

    T*      decoder_normed_input_    = nullptr;
    T*      attn_normed_input_       = nullptr;
    T*      self_attn_output_        = nullptr;
    T*      normed_self_attn_output_ = nullptr;
    T*      decoder_layer_output_    = nullptr;
    size_t* h_pinned_token_num_ptr_  = nullptr;
    int*    padding_offset_          = nullptr;
    int*    cu_seqlens_              = nullptr;

    T*   compact_decoder_features_ = nullptr;
    T*   compact_attention_mask_   = nullptr;
    int* compact_input_lengths_    = nullptr;
    T*   k_cache_layer_            = nullptr;
    T*   v_cache_layer_            = nullptr;

    T*   expert_scales_                            = nullptr;
    int* expanded_source_row_to_expanded_dest_row_ = nullptr;
    int* expert_for_source_row_                    = nullptr;
    T*   fc2_result_                               = nullptr;
    T*   adapter_fc2_result_                       = nullptr;

protected:
public:
    ParallelGptContextDecoder(size_t                              max_batch_size,
                              size_t                              max_seq_len,
                              const GptInitParameter&             gpt_init_parameter,
                              NcclParam                           tensor_para,
                              NcclParam                           pipeline_para,
                              cudaStream_t                        stream,
                              cublasMMWrapper*                    cublas_wrapper,
                              IAllocator*                         allocator,
                              bool                                is_free_buffer_after_forward,
                              bool                                is_qk_buf_float,
                              AttentionType                       attention_type           = AttentionType::UNFUSED_MHA,
                              bool                                sparse                   = false,
                              std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm   = nullptr,
                              int                                 enable_custom_all_reduce = 0);

    ParallelGptContextDecoder(ParallelGptContextDecoder<T> const& decoder);

    ~ParallelGptContextDecoder();

    void forward(TensorMap*                                            output_tensors,
                 const TensorMap*                                      input_tensors,
                 const std::vector<ParallelGptDecoderLayerWeight<T>*>* decoder_layer_weights);
};

}  // namespace fastertransformer
