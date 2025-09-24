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

#include "decoder_masked_multihead_attention.h"
#include "rtp_llm/cpp/kernels/kv_cache/kv_cache_utils.h"
#include "rtp_llm/cpp/utils/utils.h"

#if USING_CUDA
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#endif
#if USING_ROCM
#include "rtp_llm/cpp/rocm/hip_host_utils.h"
#endif

#include <assert.h>
#include <float.h>
#include <type_traits>

namespace rtp_llm {
#if USING_ROCM
using namespace rocm;
#endif

template<typename T,
         typename T_cache,
         typename KVCacheBuffer,
         typename KernelParamsType,
         int       Dh,
         bool      HAS_BEAMS,
         bool      DO_MULTI_BLOCK,
         RopeStyle ROPE_STYLE>
void mmha_launch_kernel_ex(KernelParamsType&    params,
                           const KVCacheBuffer& kv_cache_buffer,
                           const cudaStream_t&  stream,
                           int                  tlength);

#ifdef ENABLE_FP8
#define FT_TCACHE_SWITCH(NAME, ...)                                                                                    \
    [&] {                                                                                                              \
        if constexpr (!std::is_same<T, float>::value) {                                                                \
            if (params.int8_kv_cache) {                                                                                \
                typedef int8_t NAME;                                                                                   \
                return __VA_ARGS__();                                                                                  \
            } else if (params.fp8_kv_cache) {                                                                          \
                typedef __nv_fp8_e4m3 NAME;                                                                            \
                return __VA_ARGS__();                                                                                  \
            } else {                                                                                                   \
                typedef T NAME;                                                                                        \
                return __VA_ARGS__();                                                                                  \
            }                                                                                                          \
        } else {                                                                                                       \
            if (params.int8_kv_cache || params.fp8_kv_cache) {                                                         \
                RTP_LLM_FAIL("unsupported float type mmha with quant kvcache");                                        \
            }                                                                                                          \
            typedef T NAME;                                                                                            \
            return __VA_ARGS__();                                                                                      \
        }                                                                                                              \
    }()
#else
#define FT_TCACHE_SWITCH(NAME, ...)                                                                                    \
    [&] {                                                                                                              \
        if constexpr (!std::is_same<T, float>::value) {                                                                \
            if (params.int8_kv_cache) {                                                                                \
                typedef int8_t NAME;                                                                                   \
                return __VA_ARGS__();                                                                                  \
            } else {                                                                                                   \
                typedef T NAME;                                                                                        \
                return __VA_ARGS__();                                                                                  \
            }                                                                                                          \
        } else {                                                                                                       \
            if (params.int8_kv_cache) {                                                                                \
                RTP_LLM_FAIL("unsupported float type mmha with quant kvcache");                                        \
            }                                                                                                          \
            typedef T NAME;                                                                                            \
            return __VA_ARGS__();                                                                                      \
        }                                                                                                              \
    }()
#endif

#define FT_HEADN_SWITCH(COND, ...)                                                                                     \
    [&] {                                                                                                              \
        switch (COND) {                                                                                                \
            FT_SWITCH_ONE_CASE(Dh, 64, __VA_ARGS__)                                                                    \
            FT_SWITCH_ONE_CASE(Dh, 96, __VA_ARGS__)                                                                    \
            FT_SWITCH_ONE_CASE(Dh, 128, __VA_ARGS__)                                                                   \
            FT_SWITCH_ONE_CASE(Dh, 192, __VA_ARGS__)                                                                   \
            FT_SWITCH_ONE_CASE(Dh, 256, __VA_ARGS__)                                                                   \
            default:                                                                                                   \
                RTP_LLM_FAIL("unsupported head_size: %d", COND);                                                       \
        }                                                                                                              \
    }()

template<typename T, typename KVCacheBuffer, typename KernelParamsType>
void mmha_launch_kernel(KernelParamsType& params, const KVCacheBuffer& kv_cache_buffer, const cudaStream_t& stream) {
    const int tlength = params.timestep;
    FT_SWITCH(params.multi_block_mode, MULTI_BLOCK_MODE, [&] {
        FT_TCACHE_SWITCH(Tcache, [&] {
            FT_HEADN_SWITCH(params.hidden_size_per_head, [&] {
                FT_ROPE_SWITCH(params.rope_config.style, ROPE_STYLE, [&] {
                    mmha_launch_kernel_ex<T,
                                          Tcache,
                                          KVCacheBuffer,
                                          KernelParamsType,
                                          Dh,
                                          false,
                                          MULTI_BLOCK_MODE,
                                          ROPE_STYLE>(params, kv_cache_buffer, stream, tlength);
                });
            });
        });
    });
}

template<typename T>
struct SATypeConverter {
    using Type = T;
};

template<>
struct SATypeConverter<half> {
    using Type = uint16_t;
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
                                        int                         max_seq_len_tile,
                                        T*                          partial_out,
                                        float*                      partial_sum,
                                        float*                      partial_max,
                                        int*                        block_counter,
                                        const float                 softmax_extra_scale,
                                        KVCacheBuffer               kv_cache_buffer,
                                        cudaStream_t                stream) {
    using DataType = typename SATypeConverter<T>::Type;
    Masked_multihead_attention_params<DataType> params;
    memset(&params, 0, sizeof(params));
    int hidden_units = head_num * size_per_head;
    if (qkv_bias != nullptr) {
        params.q_bias = reinterpret_cast<const DataType*>(qkv_bias);
        params.k_bias = reinterpret_cast<const DataType*>(qkv_bias) + hidden_units;
        params.v_bias = reinterpret_cast<const DataType*>(qkv_bias) + hidden_units + size_per_head * head_num_kv;
    } else {
        params.q_bias = nullptr;
        params.k_bias = nullptr;
        params.v_bias = nullptr;
    }
    // Set the output buffer.
    params.out = reinterpret_cast<DataType*>(context_buf);

    // Set the input buffers.
    params.q = reinterpret_cast<const DataType*>(qkv_buf);
    if (int8_mode != 2) {
        params.k = reinterpret_cast<const DataType*>(qkv_buf) + hidden_units;
        params.v = reinterpret_cast<const DataType*>(qkv_buf) + hidden_units + size_per_head * head_num_kv;
    } else {
        params.k = reinterpret_cast<const DataType*>(reinterpret_cast<const int8_t*>(qkv_buf) + hidden_units);
        params.v = reinterpret_cast<const DataType*>(reinterpret_cast<const int8_t*>(qkv_buf) + hidden_units
                                                     + size_per_head * head_num_kv);
    }

    params.int8_kv_cache                  = kv_cache_quant_mode.hasInt8KvCache();
    params.fp8_kv_cache                   = kv_cache_quant_mode.hasFp8KvCache();
    params.attention_out_scale_orig_quant = attention_out_scale;
    // if (kv_cache_quant_mode.hasKvCacheQuant())
    // {
    //     params.kv_scale_orig_quant = input_params.kv_scale_orig_quant;
    //     params.kv_scale_quant_orig = input_params.kv_scale_quant_orig;
    // }

    // if (int8_mode == 2) {
    //     params.qkv_scale_out       = qkv_scale_out;
    //     params.attention_out_scale = attention_out_scale;
    // }
    params.stride                   = hidden_units + 2 * size_per_head * head_num_kv;
    params.finished                 = const_cast<bool*>(finished);
    params.cache_indir              = cache_indir;
    params.batch_size               = inference_batch_size;
    params.beam_width               = beam_width;
    params.max_kv_cache_length      = memory_max_len;
    params.cyclic_kv_cache_length   = memory_max_len + max_prefix_prompt_length;
    params.prefix_prompt_lengths    = prefix_prompt_lengths;
    params.max_prefix_prompt_length = max_prefix_prompt_length;
    params.count_prefix_length =
        count_prefix_length;  // should prefix_length counted in sequence_length, when rotary_embedding_style == 1

    params.length_per_sample = sequence_lengths;  // max_input_length + current output length
    // timestep for shared memory size calculation and rotary embedding computation
    params.timestep             = step + max_prefix_prompt_length - 1;
    params.num_heads            = head_num;
    params.num_kv_heads         = head_num_kv;
    params.hidden_size_per_head = size_per_head;
    params.rope_config          = rope_config;
    params.position_ids         = position_ids;
    // Note: keep norm factor (sqrt(K_dim)) when adopting megatron T5 structure (may adjust)
    params.inv_sqrt_dh = 1.F / (sqrtf((float)size_per_head) * q_scaling) * softmax_extra_scale;
    if (relative_attention_bias != nullptr) {
        params.relative_attention_bias = reinterpret_cast<const DataType*>(relative_attention_bias);
    }
    params.relative_attention_bias_stride = relative_attention_bias_stride;
    params.max_distance                   = 0;
    // The slope of linear position bias per head, e.g., ALiBi.
    if (linear_bias_slopes != nullptr) {
        params.linear_bias_slopes = linear_bias_slopes;
    }

    params.input_lengths = input_lengths;

    params.multi_block_mode = multi_block_mode;
    if (multi_block_mode) {
        params.max_seq_len_tile = max_seq_len_tile;
        params.partial_out      = reinterpret_cast<DataType*>(partial_out);
        params.partial_sum      = partial_sum;
        params.partial_max      = partial_max;
        params.block_counter    = block_counter;
    }

    // get device attributes
    params.multi_processor_count = getMultiProcessorCount();
    mmha_launch_kernel<DataType>(params, kv_cache_buffer, stream);
    check_cuda_error();
}

#define INSTANTIATE_FUSEDQKV_MASKED_ATTENTION_DISPATCH(T, KVCacheBuffer)                                               \
    template void fusedQKV_masked_attention_dispatch(const T*                    qkv_buf,                              \
                                                     const T*                    qkv_bias,                             \
                                                     const T*                    relative_attention_bias,              \
                                                     const int*                  cache_indir,                          \
                                                     T*                          context_buf,                          \
                                                     const bool*                 finished,                             \
                                                     const int*                  sequence_lengths,                     \
                                                     const int                   inference_batch_size,                 \
                                                     const int                   beam_width,                           \
                                                     const int                   head_num,                             \
                                                     const int                   head_num_kv,                          \
                                                     const int                   size_per_head,                        \
                                                     const RopeConfig            rope_config,                          \
                                                     const bool                  use_logn_attn,                        \
                                                     const int*                  position_ids,                         \
                                                     const int                   memory_max_len,                       \
                                                     const int*                  prefix_prompt_lengths,                \
                                                     const int                   max_prefix_prompt_length,             \
                                                     const bool                  count_prefix_length,                  \
                                                     const int*                  input_lengths,                        \
                                                     const int                   step,                                 \
                                                     const float                 q_scaling,                            \
                                                     const int                   relative_attention_bias_stride,       \
                                                     const float*                linear_bias_slopes,                   \
                                                     const bool*                 masked_tokens,                        \
                                                     const float*                qkv_scale_out,                        \
                                                     const float*                attention_out_scale,                  \
                                                     const int                   int8_mode,                            \
                                                     const trt_common::QuantMode kv_cache_quant_mode,                  \
                                                     const bool                  multi_block_mode,                     \
                                                     int                         max_seq_tile,                         \
                                                     T*                          partial_out,                          \
                                                     float*                      partial_sum,                          \
                                                     float*                      partial_max,                          \
                                                     int*                        block_counter,                        \
                                                     const float                 softmax_extra_scale,                  \
                                                     KVCacheBuffer               kv_cache_buffer,                      \
                                                     cudaStream_t                stream)

INSTANTIATE_FUSEDQKV_MASKED_ATTENTION_DISPATCH(float, KVBlockArray);
INSTANTIATE_FUSEDQKV_MASKED_ATTENTION_DISPATCH(half, KVBlockArray);
#ifdef ENABLE_BF16
INSTANTIATE_FUSEDQKV_MASKED_ATTENTION_DISPATCH(__nv_bfloat16, KVBlockArray);
#endif

#undef INSTANTIATE_FUSEDQKV_MASKED_ATTENTION_DISPATCH

////////////////////////////////////////////////////////////////////////////////////////////////////
}  // namespace rtp_llm
