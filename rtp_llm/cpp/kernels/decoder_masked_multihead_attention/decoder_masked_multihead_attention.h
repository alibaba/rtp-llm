/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "rtp_llm/cpp/model_utils/quantization.h"

#include "rtp_llm/cpp/kernels/kv_cache/kv_cache_utils.h"
#include "rtp_llm/cpp/model_utils/RopeConfig.h"

#if USING_CUDA
#include "rtp_llm/cpp/cuda/cuda_fp8_utils.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#endif

#if USING_ROCM
#include "rtp_llm/cpp/rocm/cuda_shims.h"
#endif

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
namespace trt_common = tensorrt_llm::common;
namespace rtp_llm {

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Round up to next higher power of 2 (return x if it's already a power
/// of 2).
inline int pow2roundup(int x) {
    if (x < 0)
        return 0;
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x + 1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// The structure of parameters for the masked multihead attention kernel.
//
// We use the following terminology to describe the different dimensions.
//
// B:  Batch size (number of sequences),
// L:  Sequence length,
// D:  Hidden dimension,
// H:  Number of heads,
// Dh: Hidden dimension per head - Dh = D / H.

template<typename T>
struct Multihead_attention_params_base {

    // The output buffer. Dimensions B x D.
    T* out = nullptr;

    // The input Qs and the associated bias. Dimensions B x D and D, resp.
    const T *q = nullptr, *q_bias = nullptr;
    // The input Ks and the associated bias. Dimensions B x D and D, resp.
    const T *k = nullptr, *k_bias = nullptr;
    // The input Vs and the associated bias. Dimensions B x D and D, resp.
    const T *v = nullptr, *v_bias = nullptr;

    // The indirections to use for cache when beam sampling.
    const int* cache_indir = nullptr;

    // scales
    const float* query_weight_output_scale               = nullptr;
    const float* attention_qk_scale                      = nullptr;
    const float* attention_output_weight_input_scale_inv = nullptr;

    // Stride to handle the case when KQV is a single buffer
    int stride = 0;

    // The batch size.
    int batch_size = 0;
    // The beam width
    int beam_width = 0;
    // By default, max_kv_cache_length == cyclic_kv_cache_length
    // unless each layer has different cyclic kv cache length.
    // Max cache capacity (used to allocate KV cache)
    int max_kv_cache_length = 0;
    // Cyclic kv cache capacity (used to get the cyclic kv cache position for new tokens)
    int cyclic_kv_cache_length = 0;
    // The number of heads (H).
    int num_heads = 0;
    // Controls MHA/MQA/GQA
    int num_kv_heads = 0;
    // The hidden dimension per head (Dh).
    int hidden_size_per_head = 0;

    RopeConfig rope_config;

    // Position id of rotary embedding for CogVlm2
    const int* position_ids = nullptr;

    bool use_logn_attn = false;

    const int* prefix_prompt_lengths    = nullptr;
    int        max_prefix_prompt_length = 0;
    bool       count_prefix_length      = false;

    // The current timestep. TODO Check that do we only this param in cross attention?
    int timestep = 0;
    // The current timestep of each sentences (support different timestep for different sentences)

    // The 1.f / sqrt(Dh). Computed on the host.
    float inv_sqrt_dh = 0.0f;

    // If relative position embedding is used
    const T* relative_attention_bias        = nullptr;
    int      relative_attention_bias_stride = 0;
    int      max_distance                   = 0;

    // The slope per head of linear position bias to attention score (H).
    const float* linear_bias_slopes = nullptr;

    const T*   ia3_key_weights   = nullptr;
    const T*   ia3_value_weights = nullptr;
    const int* ia3_tasks         = nullptr;

    const float* qkv_scale_quant_orig           = nullptr;
    const float* attention_out_scale_orig_quant = nullptr;

    const float* kv_scale_orig_quant = nullptr;
    const float* kv_scale_quant_orig = nullptr;

    bool int8_kv_cache = false;
    bool fp8_kv_cache  = false;

    // Multi-block setups
    bool multi_block_mode = false;

    // Number of streaming processors on the device.
    // Tune block size to maximum occupancy.
    int multi_processor_count = 1;

    int timesteps_per_block = -1;
    int seq_len_tile        = -1;

    int max_seq_len_tile = -1;
    // The partial output buffer. Dimensions max_seq_len_tile x B x D. (for each timestep only seq_len_tile x B x D is
    // needed)
    T* partial_out = nullptr;
    // ThreadBlock sum. Dimensions max_seq_len_tile x 1. (for each timestep only seq_len_tile x 1 is needed)
    float* partial_sum = nullptr;
    // ThreadBlock max. Dimensions max_seq_len_tile x 1. (for each timestep only seq_len_tile x 1 is needed)
    float* partial_max = nullptr;
    // threadblock counter to identify the complete of partial attention computations
    int* block_counter = nullptr;

    const int* memory_length_per_sample = nullptr;

    bool enable_multi_block_mode() {
        return seq_len_tile > 1;
    }
};

template<typename T, bool USE_CROSS_ATTENTION = false>
struct Multihead_attention_params;

// self-attention params
template<typename T>
struct Multihead_attention_params<T, false>: public Multihead_attention_params_base<T> {
    static constexpr bool DO_CROSS_ATTENTION = false;

    int max_decoder_seq_len = 0;

    // allows to exit attention early
    bool* finished = nullptr;

    // required in case of masked attention with different length
    const int* length_per_sample = nullptr;

    // input lengths to identify the paddings (i.e. input seq < padding < new generated seq).
    const int* input_lengths = nullptr;
};
template<class T>
using Masked_multihead_attention_params = Multihead_attention_params<T, false>;

// cross-attention params
template<typename T>
struct Multihead_attention_params<T, true>: public Multihead_attention_params_base<T> {
    static constexpr bool DO_CROSS_ATTENTION = true;

    int max_decoder_seq_len = 0;

    // allows to exit attention early
    bool* finished = nullptr;

    // required in case of masked attention with different length
    const int* length_per_sample = nullptr;

    // input lengths to identify the paddings (i.e. input seq < padding < new generated seq).
    const int* input_lengths = nullptr;
};
template<class T>
using Cross_multihead_attention_params = Multihead_attention_params<T, true>;

template<typename T>
struct outputCrossAttentionParam {
    // max decoder output length
    int  max_decoder_seq_len        = 0;
    T*   cross_attention_out        = nullptr;
    bool is_return_cross_attentions = false;
};

template<typename T, typename KVCacheBuffer>
void fusedQKV_masked_attention_dispatch(const T*                    qkv_buf,
                                        const T*                    qkv_bias,
                                        const T*                    relative_attention_bias,
                                        const int*                  cache_indir,
                                        T*                          context_buf,
                                        const bool*                 finished,
                                        const int*                  sequence_lengths,
                                        const int                   inference_batch_size,
                                        const int                   beam_width,
                                        const int                   head_num,
                                        const int                   head_num_kv,
                                        const int                   size_per_head,
                                        const RopeConfig            rope_config,
                                        const bool                  use_logn_attn,
                                        const int*                  position_ids,
                                        const int                   memory_max_len,
                                        const int*                  prefix_prompt_lengths,
                                        const int                   max_prefix_prompt_length,
                                        const bool                  count_prefix_length,
                                        const int*                  input_lengths,
                                        const int                   step,
                                        const float                 q_scaling,
                                        const int                   relative_attention_bias_stride,
                                        const float*                linear_bias_slopes,
                                        const bool*                 masked_tokens,
                                        const float*                qkv_scale_out,
                                        const float*                attention_out_scale,
                                        const int                   int8_mode,
                                        const trt_common::QuantMode kv_cache_quant_mode,
                                        const bool                  multi_block_mode,
                                        int                         max_seq_tile,
                                        T*                          partial_out,
                                        float*                      partial_sum,
                                        float*                      partial_max,
                                        int*                        block_counter,
                                        const float                 softmax_extra_scale,
                                        KVCacheBuffer               kv_cache_buffer,
                                        cudaStream_t                stream);

////////////////////////////////////////////////////////////////////////////////////////////////////
}  // namespace rtp_llm
