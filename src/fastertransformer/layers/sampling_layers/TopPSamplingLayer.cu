/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
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

#include <algorithm>
#include <float.h>

#include "src/fastertransformer/kernels/reduce_kernel_utils.cuh"
#include "src/fastertransformer/kernels/sampling_penalty_kernels.h"
#include "src/fastertransformer/kernels/sampling_topk_kernels.h"
#include "src/fastertransformer/kernels/sampling_topp_kernels.h"
#include "src/fastertransformer/layers/sampling_layers/TopPSamplingLayer.h"
#include "src/fastertransformer/utils/logger.h"
#include "src/fastertransformer/cuda/memory_utils.h"

namespace fastertransformer {

template<typename T>
void TopPSamplingLayer<T>::allocateBuffer()
{
    FT_CHECK(false);
}

template<typename T>
void TopPSamplingLayer<T>::allocateBuffer(size_t batch_size, Tensor top_k, Tensor top_p)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    BaseSamplingLayer<T>::allocateBuffer(batch_size, top_k, top_p);
    invokeTopPSampling<T>(nullptr,  // workspace
                          sampling_workspace_size_,
                          cub_temp_storage_size_,
                          nullptr,  // output_ids
                          nullptr,  // sequence_length
                          nullptr,  // finished_buffer
                          nullptr,  // cum_log_probs
                          nullptr,  // output_log_probs
                          nullptr,  // log_probs
                          topp_id_vals_buf_,
                          topp_offset_buf_,
                          begin_topp_offset_buf_,
                          curandstate_buf_,
                          batch_size,
                          vocab_size_padded_,
                          nullptr,
                          top_p.size() > 0 ? top_p.max<float>() : 0.0f,
                          stream_,
                          cuda_device_prop_,
                          skip_decode_buf_);
    sampling_workspace_ = allocator_->reMalloc(sampling_workspace_, sampling_workspace_size_);
    runtime_top_k_buf_ =
        reinterpret_cast<uint*>(allocator_->reMalloc(runtime_top_k_buf_, sizeof(uint) * batch_size));
    runtime_top_p_buf_ =
        reinterpret_cast<float*>(allocator_->reMalloc(runtime_top_p_buf_, sizeof(float) * batch_size));
    initial_top_p_buf_ =
        reinterpret_cast<float*>(allocator_->reMalloc(initial_top_p_buf_, sizeof(float) * batch_size));
    top_p_decay_buf_ =
        reinterpret_cast<float*>(allocator_->reMalloc(top_p_decay_buf_, sizeof(float) * batch_size));
    top_p_min_buf_ = reinterpret_cast<float*>(allocator_->reMalloc(top_p_min_buf_, sizeof(float) * batch_size));
    top_p_reset_ids_buf_ =
        reinterpret_cast<int32_t*>(allocator_->reMalloc(top_p_reset_ids_buf_, sizeof(int32_t) * batch_size));
    topp_id_vals_buf_ = reinterpret_cast<int*>(
        allocator_->reMalloc(topp_id_vals_buf_, sizeof(int) * batch_size * vocab_size_padded_));
    topp_offset_buf_ =
        reinterpret_cast<int*>(allocator_->reMalloc(topp_offset_buf_, sizeof(int) * (batch_size + 1)));
    begin_topp_offset_buf_ =
        reinterpret_cast<int*>(allocator_->reMalloc(begin_topp_offset_buf_, sizeof(int) * (batch_size + 1)));
    is_allocate_buffer_ = true;
}

template<typename T>
void TopPSamplingLayer<T>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_) {
        allocator_->free((void**)(&sampling_workspace_));
        allocator_->free((void**)(&topp_id_vals_buf_));
        allocator_->free((void**)(&topp_offset_buf_));
        allocator_->free((void**)(&begin_topp_offset_buf_));
        allocator_->free((void**)(&runtime_top_k_buf_));
        allocator_->free((void**)(&runtime_top_p_buf_));
        allocator_->free((void**)(&initial_top_p_buf_));
        allocator_->free((void**)(&top_p_decay_buf_));
        allocator_->free((void**)(&top_p_min_buf_));
        allocator_->free((void**)(&top_p_reset_ids_buf_));
    }
    BaseSamplingLayer<T>::freeBuffer();
    is_allocate_buffer_ = false;
}

template<typename T>
void TopPSamplingLayer<T>::setup(const size_t batch_size, const size_t beam_width, TensorMap* runtime_args)
{
    /**
    * @brief Set up the sampling layer for given runtime arguments.

    * runtime_args:
    *   \param  runtime_top_k [1] or [batch_size] on cpu, optional.
    *   \param  runtime_top_p [1] or [batch_size] on cpu, optional
    *   \param  temperature [1] or [batch_size] on cpu, optional
    *   \param  repetition_penalty [1] or [batch_size] on cpu, optional
    *   \param  top_p_decay [batch_size] on gpu, float, optional
    *   \param  top_p_min [batch_size] on gpu, float, optional
    *   \param  top_p_reset_ids [batch_size] on gpu, uint32, optional
    **/

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    BaseSamplingLayer<T>::setup(batch_size, beam_width, runtime_args);
    const Tensor runtime_top_p = runtime_args->isExist("runtime_top_p") ? runtime_args->at("runtime_top_p") : Tensor();
    const size_t runtime_top_p_size = runtime_top_p.size();
    if (runtime_top_p_size == 0) {
        std::fill_n(skip_decode_, batch_size, true);
        return;
    }

    uint         tmp_top_k          = 0;
    const Tensor runtime_top_k      = runtime_args->isExist("runtime_top_k") ?
                                          runtime_args->at("runtime_top_k") :
                                          Tensor(MEMORY_CPU, TYPE_UINT32, {1}, &tmp_top_k);
    const size_t runtime_top_k_size = runtime_top_k.size();

    uint  top_k = runtime_top_k.getVal<uint>();
    float top_p = runtime_top_p.getVal<float>();

    if (runtime_top_k_size > 1) {
        FT_CHECK(runtime_top_k.size() == batch_size);
        cudaH2Dcpy(runtime_top_k_buf_, runtime_top_k.getPtr<uint>(), batch_size);
    }
    if (runtime_top_p_size > 1) {
        FT_CHECK(runtime_top_p.size() == batch_size);
        cudaH2Dcpy(runtime_top_p_buf_, runtime_top_p.getPtr<float>(), batch_size);
    }

    dim3 block(std::min((int)batch_size, 256));
    dim3 grid(div_up((int)batch_size, (int)block.x));

    const float*    top_p_decay     = runtime_args->getPtr<float>("top_p_decay", nullptr);
    const float*    top_p_min       = runtime_args->getPtr<float>("top_p_min", nullptr);
    const uint32_t* top_p_reset_ids = runtime_args->getPtr<uint32_t>("top_p_reset_ids", nullptr);
    invokeSetupTopPRuntimeArgs(batch_size,
                                top_k,
                                runtime_top_k_buf_,
                                runtime_top_k_size,
                                top_p,
                                runtime_top_p_buf_,
                                runtime_top_p_size,
                                skip_decode_buf_,
                                initial_top_p_buf_,
                                top_p_decay_buf_,
                                top_p_decay,
                                top_p_min_buf_,
                                top_p_min,
                                top_p_reset_ids_buf_,
                                top_p_reset_ids,
                                stream_);
    sync_check_cuda_error();
    cudaAutoCpy(skip_decode_, skip_decode_buf_, batch_size, stream_);
    float* runtime_top_ps = new float[batch_size];
    cudaAutoCpy(runtime_top_ps, runtime_top_p_buf_, batch_size, stream_);
    runtime_max_top_p_ = *std::max_element(runtime_top_ps, runtime_top_ps + batch_size);
    delete[] runtime_top_ps;
}

template<typename T>
void TopPSamplingLayer<T>::runSampling(TensorMap* output_tensors, TensorMap* input_tensors)
{
    /**
    * input_tensors:
    *   \param  logits [local_batch_size, vocab_size_padded]
    *   \param  embedding_bias [vocab_size_padded], optional
    *   \param  step [1] on cpu
    *   \param  max_input_length [1] on cpu
    *   \param  input_lengths [local_batch_size], optional
    *   \param  ite [1] on cpu

    * output_tensors:
    *   \param  output_ids [max_seq_len, batch_size]
    *   \param  finished [local_batch_size], optional
    *   \param  sequence_length [local_batch_size], optional
    *   \param  cum_log_probs [batch_size], must be float*, optional
    *   \param  The cumultative log probability of generated tokens.
    *   \param  output_log_probs [local_batch_size], must be float*, optional
                    log probs at the current step.
    **/

    FT_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    FT_CHECK(input_tensors->size() >= 4);
    FT_CHECK(output_tensors->size() >= 1);

    const int batch_size       = output_tensors->at("output_ids").shape()[1];
    const int local_batch_size = input_tensors->at("logits").shape()[0];
    const int step             = input_tensors->at("step").getVal<int>();
    const int ite              = input_tensors->at("ite").getVal<int>();

    // in case of skip any, the logit value is already copied and processed.
    T* logits = !skip_any_ ? input_tensors->at("logits").getPtr<T>() : runtime_logits_buf_;

    invokeTopPInitialize(
        topp_id_vals_buf_, topp_offset_buf_, begin_topp_offset_buf_, local_batch_size, vocab_size_padded_, stream_);
    sync_check_cuda_error();

    invokeAddBiasSoftMax(logits,
                         (T*)(nullptr),
                         input_tensors->at("end_id").getPtr<int>(),
                         output_tensors->at("finished", Tensor{MEMORY_GPU, TYPE_INVALID, {}, nullptr}).getPtr<bool>(),
                         local_batch_size,
                         vocab_size_padded_,
                         vocab_size_,
                         stream_);
    sync_check_cuda_error();

    if (output_tensors->isExist("index_log_probs") and output_tensors->isExist("token_id_for_index_prob")) {
        invokeCopyLogits<T>(
            output_tensors->at("index_log_probs").getPtrWithOffset<float>((ite * local_batch_size)),
            output_tensors->at("token_id_for_index_prob").getPtrWithOffset<int>((ite * local_batch_size)),
            logits,
            skip_decode_buf_,
            local_batch_size,
            vocab_size_padded_,
            stream_
        );
    }
    sync_check_cuda_error();

    float* cum_log_probs =
        output_tensors->isExist("cum_log_probs") ? output_tensors->at("cum_log_probs").getPtr<float>() : nullptr;
    float* output_log_probs =
        output_tensors->isExist("output_log_probs") ? output_tensors->at("output_log_probs").getPtr<float>() : nullptr;

    invokeBatchTopPSampling<T>(
        sampling_workspace_,
        sampling_workspace_size_,
        cub_temp_storage_size_,
        output_tensors->at("output_ids").getPtrWithOffset<int>(step * batch_size + ite * local_batch_size),
        output_tensors->at("sequence_length", Tensor{MEMORY_GPU, TYPE_INVALID, {}, nullptr}).getPtr<int>(),
        output_tensors->at("finished", Tensor{MEMORY_GPU, TYPE_INVALID, {}, nullptr}).getPtr<bool>(),
        cum_log_probs,
        output_log_probs,
        logits,
        topp_id_vals_buf_,
        topp_offset_buf_,
        begin_topp_offset_buf_,
        curandstate_buf_ + ite * local_batch_size,
        local_batch_size,
        vocab_size_padded_,
        input_tensors->at("end_id").getPtr<int>(),
        runtime_max_top_p_,
        runtime_top_p_buf_ + ite * local_batch_size,
        stream_,
        cuda_device_prop_,
        skip_decode_buf_ + ite * local_batch_size);
    sync_check_cuda_error();

    invokeComputeToppDecay(
        runtime_top_p_buf_ + ite * local_batch_size,
        initial_top_p_buf_ + ite * local_batch_size,
        output_tensors->getPtrWithOffset<int>("output_ids", step * batch_size + ite * local_batch_size),
        top_p_decay_buf_ + ite * local_batch_size,
        top_p_min_buf_ + ite * local_batch_size,
        top_p_reset_ids_buf_ + ite * local_batch_size,
        local_batch_size,
        stream_);
    sync_check_cuda_error();
    FT_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

template<typename T>
TopPSamplingLayer<T>::TopPSamplingLayer(size_t             max_batch_size,
                                        size_t             vocab_size,
                                        size_t             vocab_size_padded,
                                        int                end_id,
                                        float              top_p,
                                        unsigned long long random_seed,
                                        float              temperature,
                                        float              len_penalty,
                                        float              repetition_penalty,
                                        cudaStream_t       stream,
                                        cublasMMWrapper*   cublas_wrapper,
                                        IAllocator*        allocator,
                                        bool               is_free_buffer_after_forward,
                                        cudaDeviceProp*    cuda_device_prop):
    BaseSamplingLayer<T>(max_batch_size,
                         vocab_size,
                         vocab_size_padded,
                         end_id,
                         0,
                         top_p,
                         random_seed,
                         temperature,
                         len_penalty,
                         repetition_penalty,
                         stream,
                         cublas_wrapper,
                         allocator,
                         is_free_buffer_after_forward,
                         cuda_device_prop)
{
}

template<typename T>
TopPSamplingLayer<T>::TopPSamplingLayer(TopPSamplingLayer<T> const& top_p_sampling_layer):
    BaseSamplingLayer<T>(top_p_sampling_layer)
{
}

template<typename T>
TopPSamplingLayer<T>::~TopPSamplingLayer()
{
    freeBuffer();
}

template class TopPSamplingLayer<float>;
template class TopPSamplingLayer<half>;

}  // namespace fastertransformer
