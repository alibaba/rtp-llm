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

#include "src/fastertransformer/layers/FfnLayer.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/kernels/transpose_int8_kernels.h"
#include "src/fastertransformer/cuda/nvtx/nvtx_utils.h"

namespace fastertransformer {

template<typename T>
void FfnLayer<T>::preAllocate() {
    if (max_token_num_ > 0) {
        allocateBuffer(max_token_num_, 1, false);
    }
}

template<typename T>
void FfnLayer<T>::forward(std::vector<fastertransformer::Tensor>*       output_tensors,
                          const std::vector<fastertransformer::Tensor>* input_tensors,
                          const FfnWeight<T>*                           ffn_weights) {
    TensorMap input_tensor({{"ffn_input", input_tensors->at(0)}});
    TensorMap output_tensor({{"ffn_output", output_tensors->at(0)}});
    forward(&output_tensor, &input_tensor, ffn_weights);
}

template<typename T>
void FfnLayer<T>::forward(TensorMap* output_tensors, TensorMap* input_tensors, const FfnWeight<T>* ffn_weights) {
    // input tensors:
    //      ffn_input [token_num, hidden_dimension],
    //      ia3_tasks [batch_size] (optional)
    //      moe_k     [1], uint64 (optional)
    //      padding_offset [token_num] (optional)
    //      seq_len [1], int32, (optional), only used for ia3

    // output tensors:
    //      ffn_output [token_num, hidden_dimension] or [moe_k * token_num, hidden_dimension] if use_moe
    //      expert_scales [token_num, moe_k] (optional)
    //      expanded_source_row_to_expanded_dest_row [token_num, moe_k] (optional)
    //      expert_for_source_row [token_num, moe_k] (optional)

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    FT_CHECK(input_tensors->size() >= 1 && input_tensors->size() <= 9);
    FT_CHECK(output_tensors->size() >= 1 || output_tensors->size() <= 4);
    bool   use_moe = false;
    size_t moe_k   = 0;
    if (input_tensors->isExist("moe_k")) {
        use_moe = true;
        moe_k   = input_tensors->at("moe_k").getVal<size_t>();
    }

    bool use_ffn3 = ffn_weights->intermediate_weight2.int8_kernel != nullptr ||
                    ffn_weights->intermediate_weight2.kernel != nullptr;

    allocateBuffer(input_tensors->at("ffn_input").shape()[0], moe_k, use_moe, use_ffn3);

    const int m             = input_tensors->at("ffn_input").shape()[0];
    T*        output_tensor = output_tensors->at("ffn_output").getPtr<T>();
    const T*  input_tensor  = input_tensors->at("ffn_input").getPtr<const T>();
    const int layer_id      = input_tensors->getVal<int>("layer_id");

    const int  batch_size         = input_tensors->getVal<int>("batch_size", 1);
    const int* lora_input_lengths = input_tensors->getPtr<int>("lora_input_lengths", nullptr);
    // lora
    const int* lora_ids = input_tensors->getPtr<int>("lora_ids", nullptr);

    // TODO: INT8 and Sparsity are currently not implemented (geglu or reglu)
    bool use_gated_activation;
    if (int8_mode_ == 0) {
        use_gated_activation = use_gated_activation_ && ffn_weights->intermediate_weight2.kernel != nullptr;
    } else {
        use_gated_activation = use_gated_activation_ && ffn_weights->intermediate_weight2.int8_kernel != nullptr;
    }
    //  for moe output
    T*   expert_scales    = nullptr;
    int* permuted_rows    = nullptr;
    int* permuted_experts = nullptr;

    // moe outputs should exist or not together
    FT_CHECK((use_moe && output_tensors->isExist("expert_scales")
              && output_tensors->isExist("expanded_source_row_to_expanded_dest_row")
              && output_tensors->isExist("expert_for_source_row"))
             || (!use_moe && !output_tensors->isExist("expert_scales")
                 && !output_tensors->isExist("expanded_source_row_to_expanded_dest_row")
                 && !output_tensors->isExist("expert_for_source_row")));

    if (use_moe) {
        expert_scales    = output_tensors->at("expert_scales").getPtr<T>();
        permuted_rows    = output_tensors->at("expanded_source_row_to_expanded_dest_row").getPtr<int>();
        permuted_experts = output_tensors->at("expert_for_source_row").getPtr<int>();
    }

    // moe can't be used with use_gated_activation currently
    auto activation_type = getActivationType();

    const int* ia3_tasks = input_tensors->getPtr<const int>("ia3_tasks", nullptr);

    if (use_moe) {
        PUSH_RANGE(stream_, "FFN moe");
        FT_CHECK(ia3_tasks == nullptr);
        FT_CHECK(ffn_weights->gating_weight.kernel != nullptr);
        print_bsd(layer_id, "moe input", input_tensor, 1, m, hidden_units_);
        cublas_wrapper_->Gemm(CUBLAS_OP_N,
                              CUBLAS_OP_N,
                              expert_num_,
                              m,
                              hidden_units_,
                              ffn_weights->gating_weight.kernel,
                              expert_num_,
                              input_tensor,
                              hidden_units_,
                              moe_gates_buf_,
                              expert_num_);
        print_bsd(layer_id, "moe gate", moe_gates_buf_, 1, m, expert_num_);

        if (int8_mode_ == 0) {

            if (use_ffn3) {

                FT_CHECK(ffn_weights->intermediate_weight2.kernel != nullptr);
                moe_fc_runner_->run_moe_fc(input_tensor,
                                           moe_gates_buf_,
                                           ffn_weights->intermediate_weight.kernel,
                                           nullptr,
                                           ffn_weights->intermediate_weight.bias,
                                           activation_type,
                                           ffn_weights->output_weight.kernel,
                                           nullptr,
                                           ffn_weights->intermediate_weight2.kernel,
                                           nullptr,
                                           m,
                                           hidden_units_,
                                           inter_size_,
                                           expert_num_,
                                           moe_k,
                                           moe_fc_workspace_,
                                           output_tensor,
                                           nullptr,
                                           m,
                                           expert_scales,
                                           permuted_rows,
                                           permuted_experts,
                                           stream_);

            } else {
                moe_fc_runner_->run_moe_fc(input_tensor,
                                           moe_gates_buf_,
                                           ffn_weights->intermediate_weight.kernel,
                                           nullptr,
                                           ffn_weights->intermediate_weight.bias,
                                           activation_type,
                                           ffn_weights->output_weight.kernel,
                                           nullptr,
                                           m,
                                           hidden_units_,
                                           inter_size_,
                                           expert_num_,
                                           moe_k,
                                           moe_fc_workspace_,
                                           output_tensor,
                                           expert_scales,
                                           permuted_rows,
                                           permuted_experts,
                                           stream_);
            }
        } else if (int8_mode_ == 1) {
            FT_CHECK_WITH_INFO(moe_int8_weight_only_fc_runner_.get() != NULL,
                               "weight only runner was not initialized.");

            FT_CHECK(ffn_weights->intermediate_weight.int8_kernel != NULL
                     && ffn_weights->intermediate_weight.weight_only_quant_scale != NULL);

            FT_CHECK(ffn_weights->output_weight.int8_kernel != NULL
                     && ffn_weights->output_weight.weight_only_quant_scale != NULL);

            if (use_ffn3) {
                FT_CHECK(ffn_weights->intermediate_weight2.int8_kernel != NULL
                         && ffn_weights->intermediate_weight2.weight_only_quant_scale != NULL);
                moe_int8_weight_only_fc_runner_->run_moe_fc(
                    input_tensor,
                    moe_gates_buf_,
                    reinterpret_cast<const uint8_t*>(ffn_weights->intermediate_weight.int8_kernel),
                    ffn_weights->intermediate_weight.weight_only_quant_scale,
                    ffn_weights->intermediate_weight.bias,
                    activation_type,
                    reinterpret_cast<const uint8_t*>(ffn_weights->output_weight.int8_kernel),
                    ffn_weights->output_weight.weight_only_quant_scale,
                    reinterpret_cast<const uint8_t*>(ffn_weights->intermediate_weight2.int8_kernel),
                    ffn_weights->intermediate_weight2.weight_only_quant_scale,
                    m,
                    hidden_units_,
                    inter_size_,
                    expert_num_,
                    moe_k,
                    moe_fc_workspace_,
                    output_tensor,
                    nullptr,
                    m,
                    expert_scales,
                    permuted_rows,
                    permuted_experts,
                    stream_);
            } else {
                moe_int8_weight_only_fc_runner_->run_moe_fc(
                    input_tensor,
                    moe_gates_buf_,
                    reinterpret_cast<const uint8_t*>(ffn_weights->intermediate_weight.int8_kernel),
                    ffn_weights->intermediate_weight.weight_only_quant_scale,
                    ffn_weights->intermediate_weight.bias,
                    activation_type,
                    reinterpret_cast<const uint8_t*>(ffn_weights->output_weight.int8_kernel),
                    ffn_weights->output_weight.weight_only_quant_scale,
                    m,
                    hidden_units_,
                    inter_size_,
                    expert_num_,
                    moe_k,
                    moe_fc_workspace_,
                    output_tensor,
                    expert_scales,
                    permuted_rows,
                    permuted_experts,
                    stream_);
            }

        } else {
            FT_CHECK_WITH_INFO(false, "Invalid int8 mode for MoE");
        }

        sync_check_cuda_error();
        if (is_free_buffer_after_forward_ == true) {
            freeBuffer();
        }
        sync_check_cuda_error();
        POP_RANGE;
        return;
    }

    const int64_t inter_size = is_sparse_head_ ? local_layer_inter_size_[layer_id] : inter_size_;
    const int64_t inter_padding_size =
        is_sparse_head_ ? local_layer_inter_padding_size_[layer_id] : inter_padding_size_;

    PUSH_RANGE(stream_, "ffn_gemm_1");
    int m_tmp = input_tensors->at("ffn_input").shape()[0];
    if (m_tmp % 8 != 0) {
        m_tmp = (m_tmp / 8 + 1) * 8;
    }
#ifdef SPARSITY_ENABLED
    const int m_padded        = m_tmp;
    bool      use_sparse_gemm = sparse_ && cublas_wrapper_->isUseSparse(1, inter_size, m, hidden_units_);
#else
    constexpr bool use_sparse_gemm = false;
    constexpr int  m_padded        = 0;
#endif

    const bool fused_gemm_activation = int8_mode_ == 1 && ia3_tasks == nullptr && !use_gated_activation;
    if (fused_gemm_activation) {
        // launch fused GEMM + activation
        weight_only_int8_fc_runner_->gemm_bias_act(
            input_tensor,
            reinterpret_cast<const uint8_t*>(ffn_weights->intermediate_weight.int8_kernel),
            ffn_weights->intermediate_weight.weight_only_quant_scale,
            ffn_weights->intermediate_weight.bias,
            inter_buf_,
            m,
            inter_padding_size,
            hidden_units_,
            activation_type,
            mixed_gemm_workspace_,
            mixed_gemm_ws_bytes_,
            stream_);
        sync_check_cuda_error();
    } else {
        // gemm used inter_size, int8 use inter_padding_size
        const int cur_inter_size = int8_mode_ == 1 ? inter_padding_size : inter_size;
        gemm_runner_->Gemm(batch_size,
                           lora_input_lengths,
                           m,
                           cur_inter_size,
                           hidden_units_,
                           input_tensor,
                           &ffn_weights->intermediate_weight,
                           inter_buf_,
                           lora_ids,
                           int8_mode_,
                           use_sparse_gemm,
                           mixed_gemm_workspace_,
                           mixed_gemm_ws_bytes_,
                           m_padded);

        if (use_gated_activation) {
            gemm_runner_->Gemm(batch_size,
                               lora_input_lengths,
                               m,
                               cur_inter_size,
                               hidden_units_,
                               input_tensor,
                               &ffn_weights->intermediate_weight2,
                               inter_buf_2_,
                               lora_ids,
                               int8_mode_,
                               use_sparse_gemm,
                               mixed_gemm_workspace_,
                               mixed_gemm_ws_bytes_,
                               m_padded);
        }
    }
    POP_RANGE;  // End for NVTX Range: FFN gemm 1

    print_bsd(layer_id, "ffn1", inter_buf_, 1, m, inter_padding_size);
    if (use_gated_activation) {
        print_bsd(layer_id, "ffn2", inter_buf_2_, 1, m, inter_padding_size);
    }

    if (int8_mode_ != 1 || ia3_tasks != nullptr || use_gated_activation) {
        // if int8_mode == 1 && ia3_tasks == nullptr && we don't use gated activations, we use cutlass
        // to fuse GEMM + bias + activation, so we skip the activation function here. In all
        // other cases, we must apply the activation function separately.
        PUSH_RANGE(stream_, "add_bias_act");
        genericActivation(layer_id,
                          m,
                          ffn_weights->intermediate_weight.bias,
                          use_gated_activation ? ffn_weights->intermediate_weight2.bias : nullptr,
                          input_tensors->at("ia3_tasks", {MEMORY_GPU, TYPE_INT32, {}, nullptr}).getPtr<const int>(),
                          ffn_weights->ia3_weight.kernel,
                          int8_mode_ == 2 ? ffn_weights->intermediate_weight.scale_out : (float*)nullptr,
                          int8_mode_ == 2 ? ffn_weights->output_weight.scale : (float*)nullptr,
                          input_tensors->getPtr<int>("padding_offset", nullptr),
                          input_tensors->getVal<int>("seq_len", 1));
        POP_RANGE;
    }

    sync_check_cuda_error();

    print_bsd(layer_id, "ffn act", inter_buf_, 1, m, inter_padding_size);

    T* inter_buf_normed_output = nullptr;
    if (ffn_weights->dense_layernorm.gamma && ffn_weights->dense_layernorm.beta) {
        invokeGeneralLayerNormWithPadding(inter_buf_normed_,
                                          inter_buf_,
                                          ffn_weights->dense_layernorm.gamma,
                                          ffn_weights->dense_layernorm.beta,
                                          layernorm_eps_,
                                          m,
                                          inter_size,
                                          inter_padding_size,
                                          nullptr,
                                          nullptr,
                                          int8_mode_,
                                          stream_);

        inter_buf_normed_output = inter_buf_normed_;
        sync_check_cuda_error();
    } else {
        inter_buf_normed_output = inter_buf_;
    }
    print_bsd(layer_id, "ffn ln", inter_buf_normed_output, 1, m, inter_padding_size);

    PUSH_RANGE(stream_, "ffn_gemm_2");
    const int cur_inter_size = int8_mode_ == 1 ? inter_padding_size : inter_size;
    gemm_runner_->Gemm(batch_size,
                       lora_input_lengths,
                       m,
                       hidden_units_,
                       cur_inter_size,
                       inter_buf_normed_output,
                       &ffn_weights->output_weight,
                       output_tensor,
                       lora_ids,
                       int8_mode_,
                       use_sparse_gemm,
                       mixed_gemm_workspace_,
                       mixed_gemm_ws_bytes_,
                       m_padded);

    print_bsd(layer_id, "ffn out layer", output_tensor, 1, m, hidden_units_);

    sync_check_cuda_error();
    POP_RANGE;

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
    sync_check_cuda_error();
}

template<typename T>
FfnLayer<T>::FfnLayer(size_t               max_batch_size,
                      size_t               max_seq_len,
                      size_t               head_num,
                      size_t               size_per_head,
                      size_t               expert_num,
                      size_t               inter_size,
                      size_t               inter_padding_size,
                      std::vector<int64_t> local_layer_inter_size,
                      std::vector<int64_t> local_layer_inter_padding_size,
                      cudaStream_t         stream,
                      cublasMMWrapper*     cublas_wrapper,
                      IAllocator*          allocator,
                      bool                 is_free_buffer_after_forward,
                      bool                 sparse,
                      bool                 is_sparse_head,
                      int                  int8_mode,
                      ActivationType       activation_type,
                      float                layernorm_eps):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, sparse),
    max_token_num_(max_batch_size * max_seq_len),
    head_num_(head_num),
    size_per_head_(size_per_head),
    expert_num_(expert_num),
    hidden_units_(head_num * size_per_head),
    inter_size_(inter_size),
    inter_padding_size_(inter_padding_size),
    local_layer_inter_size_(local_layer_inter_size),
    local_layer_inter_padding_size_(local_layer_inter_padding_size),
    is_sparse_head_(is_sparse_head),
    int8_mode_(int8_mode),
    activation_type_(activation_type),
    layernorm_eps_(layernorm_eps) {
    use_gated_activation_ = activation_type_ == ActivationType::GeGLU
                            || activation_type_ == ActivationType::GeGluNoneApproximate
                            || activation_type_ == ActivationType::ReGLU || activation_type_ == ActivationType::SiGLU;
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (int8_mode_ == 1) {
        FT_CHECK_WITH_INFO(!(std::is_same<T, float>::value), "Weight only quant not supported for fp32.");
        weight_only_int8_fc_runner_     = std::make_shared<CutlassFpAIntBGemmRunner<T, uint8_t>>();
        moe_int8_weight_only_fc_runner_ = std::make_shared<CutlassMoeFCRunner<T, uint8_t>>();

    } else if (int8_mode_ == 2) {
        abort();
    }
    gemm_runner_ =
        std::make_shared<GemmRunner<T>>(sparse, stream, allocator, cublas_wrapper, weight_only_int8_fc_runner_);
    moe_fc_runner_ = std::make_shared<CutlassMoeFCRunner<T, T>>();
}

template<typename T>
FfnLayer<T>::FfnLayer(FfnLayer<T> const& ffn_layer):
    BaseLayer(ffn_layer.stream_,
              ffn_layer.cublas_wrapper_,
              ffn_layer.allocator_,
              ffn_layer.is_free_buffer_after_forward_,
              ffn_layer.cuda_device_prop_,
              ffn_layer.sparse_),
    max_token_num_(ffn_layer.max_token_num_),
    head_num_(ffn_layer.head_num_),
    size_per_head_(ffn_layer.size_per_head_),
    expert_num_(ffn_layer.expert_num_),
    hidden_units_(ffn_layer.hidden_units_),
    inter_size_(ffn_layer.inter_size_),
    inter_padding_size_(ffn_layer.inter_padding_size_),
    local_layer_inter_size_(ffn_layer.local_layer_inter_size_),
    local_layer_inter_padding_size_(ffn_layer.local_layer_inter_padding_size_),
    is_sparse_head_(ffn_layer.is_sparse_head_),
    int8_mode_(ffn_layer.int8_mode_),
    activation_type_(ffn_layer.activation_type_),
    layernorm_eps_(ffn_layer.layernorm_eps_),
    use_gated_activation_(ffn_layer.use_gated_activation_),
    weight_only_int8_fc_runner_(ffn_layer.weight_only_int8_fc_runner_),
    gemm_runner_(ffn_layer.gemm_runner_),
    moe_fc_runner_(ffn_layer.moe_fc_runner_),
    moe_int8_weight_only_fc_runner_(ffn_layer.moe_int8_weight_only_fc_runner_) {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
}

template<typename T>
FfnLayer<T>::~FfnLayer() {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    cublas_wrapper_ = nullptr;
    freeBuffer();
}

template<typename T>
void FfnLayer<T>::allocateBuffer() {
    FT_CHECK_WITH_INFO(false,
                       "FfnLayer::allocateBuffer() is deprecated. Use `allocateBuffer(size_t token_num, ...)` instead");
}

template<typename T>
void FfnLayer<T>::allocateBuffer(size_t token_num, int moe_k, bool use_moe, bool use_ffn3) {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (use_moe) {
        moe_gates_buf_ =
            (T*)allocator_->reMalloc(moe_gates_buf_, sizeof(T) * pad_to_multiple_of_16(token_num * expert_num_), false);
        size_t ws_size_moe = 0;
        if (int8_mode_ == 0) {
            FT_CHECK_WITH_INFO(moe_fc_runner_.get() != NULL, "moe runner was not initialized.");
            ws_size_moe = moe_fc_runner_->getWorkspaceSize(
                token_num, hidden_units_, inter_size_, expert_num_, moe_k, use_ffn3);
        } else if (int8_mode_ == 1) {
            FT_CHECK_WITH_INFO(moe_int8_weight_only_fc_runner_.get() != NULL,
                               "weight only moe runner was not initialized.");
            ws_size_moe = moe_int8_weight_only_fc_runner_->getWorkspaceSize(
                token_num, hidden_units_, inter_size_, expert_num_, moe_k, use_ffn3);
        }

        moe_fc_workspace_ = (char*)allocator_->reMalloc(moe_fc_workspace_, sizeof(char) * ws_size_moe, false);
    } else {
        const auto type_size = int8_mode_ == 2 ? sizeof(int8_t) : sizeof(T);
        inter_buf_ =
            (T*)allocator_->reMalloc(inter_buf_, type_size * token_num * inter_padding_size_ + token_num * 4, false);
        if (use_gated_activation_) {
            inter_buf_2_ = (T*)allocator_->reMalloc(
                inter_buf_2_, sizeof(T) * token_num * inter_padding_size_ + token_num * 4, false);
        }
        inter_buf_normed_ = (T*)(allocator_->reMalloc(
            inter_buf_normed_, sizeof(T) * token_num * inter_padding_size_ + token_num * 4, true));

        if (int8_mode_ == 1) {
            FT_CHECK_WITH_INFO(weight_only_int8_fc_runner_.get() != NULL, "weight only runner was not initialized.");
            // We use max_size for n and k since we reuse buffers for both FCs and want to allocate the max
            // possible memory that would be required by any of the individual gemms.
            const int max_size    = std::max(hidden_units_, inter_padding_size_);
            mixed_gemm_ws_bytes_  = weight_only_int8_fc_runner_->getWorkspaceSize(token_num, max_size, max_size);
            mixed_gemm_workspace_ = (char*)allocator_->reMalloc(mixed_gemm_workspace_, mixed_gemm_ws_bytes_, false);
        }
    }
    is_allocate_buffer_ = true;
}

template<typename T>
void FfnLayer<T>::freeBuffer() {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_) {
        allocator_->free((void**)(&inter_buf_));
        if (use_gated_activation_) {
            allocator_->free((void**)(&inter_buf_2_));
        }
        allocator_->free((void**)(&inter_buf_normed_));

        if (expert_num_ != 0) {
            allocator_->free((void**)(&moe_gates_buf_));
            allocator_->free((void**)(&moe_fc_workspace_));
        }

        if (mixed_gemm_workspace_) {
            allocator_->free((void**)(&mixed_gemm_workspace_));
            mixed_gemm_ws_bytes_ = 0;
        }

        is_allocate_buffer_ = false;
    }
}

#define INVOKE_GENERIC_ACT(ACT)                                                                                        \
    invokeGenericActivation<ACT>(inter_buf_,                                                                           \
                                 bias1,                                                                                \
                                 inter_buf_2_,                                                                         \
                                 bias2,                                                                                \
                                 ia3_tasks,                                                                            \
                                 ia3_weights,                                                                          \
                                 m,                                                                                    \
                                 inter_padding_size,                                                                   \
                                 int8_mode_,                                                                           \
                                 activation_in,                                                                        \
                                 activation_out,                                                                       \
                                 padding_offset,                                                                       \
                                 seq_len,                                                                              \
                                 stream_);

template<typename T>
void FfnLayer<T>::genericActivation(int          layer_id,
                                    int          m,
                                    const T*     bias1,
                                    const T*     bias2,
                                    const int*   ia3_tasks,
                                    const T*     ia3_weights,
                                    const float* activation_in,
                                    const float* activation_out,
                                    const int*   padding_offset,
                                    const int    seq_len) {
    if (ia3_tasks != nullptr) {
        FT_CHECK(seq_len > 0);
    }

    const int64_t inter_padding_size =
        is_sparse_head_ ? local_layer_inter_padding_size_[layer_id] : inter_padding_size_;

    // dispatch according to actual activation
    switch (getActivationType()) {
        case ActivationType::Gelu:
        case ActivationType::GeGLU:
            if (inter_buf_2_ == nullptr && int8_mode_ <= 1) {
                invokeAddBiasGeluV2(
                    inter_buf_, bias1, ia3_tasks, ia3_weights, padding_offset, seq_len, m, inter_padding_size, stream_);
            } else {
                INVOKE_GENERIC_ACT(GeluActivation);
            }
            break;
        case ActivationType::Relu:
        case ActivationType::ReGLU:
            INVOKE_GENERIC_ACT(ReluActivation);
            break;
        case ActivationType::Silu:
        case ActivationType::SiGLU:
            INVOKE_GENERIC_ACT(SiluActivation);
            break;
        case ActivationType::Identity:
            INVOKE_GENERIC_ACT(IdentityActivation);
            break;
        case ActivationType::GeluNoneApproximate:
        case ActivationType::GeGluNoneApproximate:
            INVOKE_GENERIC_ACT(GeluActivationNoneApproximate);
            break;
        default:
            FT_CHECK_WITH_INFO(false, "not support activation type");
            break;
    }
}

#undef INVOKE_GENERIC_ACT

template class FfnLayer<float>;
template class FfnLayer<half>;
#ifdef ENABLE_BF16
template class FfnLayer<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
