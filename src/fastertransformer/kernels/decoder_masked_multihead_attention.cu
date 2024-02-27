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

#include "src/fastertransformer/kernels/decoder_masked_multihead_attention.h"
#include "src/fastertransformer/kernels/kv_cache_utils.h"
#include "src/fastertransformer/cuda/nvtx/nvtx_utils.h"
#include <assert.h>
#include <float.h>
#include <type_traits>

namespace fastertransformer {
////////////////////////////////////////////////////////////////////////////////////////////////////

// Forward declaration of the kernel launcher to avoid including decoderMaskedMultiheadAttentionLaunch.h
template<typename T, typename KVCacheBuffer, typename T_PARAMS, int Dh>
void mmha_launch_kernel(T_PARAMS& params, const KVCacheBuffer& kv_cache_buffer, const cudaStream_t& stream);

#define MMHA_LAUNCH_KERNEL(Dh)                                                                                         \
    mmha_launch_kernel<T, KVCacheBuffer, KERNEL_PARAMS_TYPE, Dh>(params, kv_cache_buffer, stream);                     \
    break;

template<typename T, typename KVCacheBuffer, typename KERNEL_PARAMS_TYPE>
void multihead_attention_(KERNEL_PARAMS_TYPE& params, const KVCacheBuffer& kv_cache_buffer, const cudaStream_t& stream)
{
    switch (params.hidden_size_per_head) {
        // case 32: mmha_launch_kernel<T, KVCacheBuffer, KERNEL_PARAMS_TYPE, 32>(params, kv_cache_buffer, stream);
        // break; case 48: mmha_launch_kernel<T, KVCacheBuffer, KERNEL_PARAMS_TYPE, 48>(params, kv_cache_buffer,
        // stream); break;
        case 64:
            mmha_launch_kernel<T, KVCacheBuffer, KERNEL_PARAMS_TYPE, 64>(params, kv_cache_buffer, stream);
            break;
        // case 80: mmha_launch_kernel<T, KVCacheBuffer, KERNEL_PARAMS_TYPE, 80>(params, kv_cache_buffer, stream);
        // break;
        case 96:
            mmha_launch_kernel<T, KVCacheBuffer, KERNEL_PARAMS_TYPE, 96>(params, kv_cache_buffer,stream);
            break;
        // case 112:
        //     mmha_launch_kernel<T, KVCacheBuffer, KERNEL_PARAMS_TYPE, 112>(params, kv_cache_buffer, stream);
        //     break;
        case 128:
            mmha_launch_kernel<T, KVCacheBuffer, KERNEL_PARAMS_TYPE, 128>(params, kv_cache_buffer, stream);
            break;
        // case 144:
        //     mmha_launch_kernel<T, KVCacheBuffer, KERNEL_PARAMS_TYPE, 144>(params, kv_cache_buffer, stream);
        //     break;
        // case 160:
        //     mmha_launch_kernel<T, KVCacheBuffer, KERNEL_PARAMS_TYPE, 160>(params, kv_cache_buffer, stream);
        //     break;
        // case 192:
        //     mmha_launch_kernel<T, KVCacheBuffer, KERNEL_PARAMS_TYPE, 192>(params, kv_cache_buffer, stream);
        //     break;
        // case 224:
        //     mmha_launch_kernel<T, KVCacheBuffer, KERNEL_PARAMS_TYPE, 224>(params, kv_cache_buffer, stream);
        //     break;
        case 256:
            mmha_launch_kernel<T, KVCacheBuffer, KERNEL_PARAMS_TYPE, 256>(params, kv_cache_buffer, stream);
            break;
        default:
            throw std::invalid_argument("unsupported head_size: " + std::to_string(params.hidden_size_per_head));
    }
}

#undef MMHA_LAUNCH_KERNEL

////////////////////////////////////////////////////////////////////////////////////////////////////

#define INSTANTIATE_MMHA_NORMAL_AND_PAGED(T, CROSS_ATTENTION)                                                          \
    void masked_multihead_attention(Multihead_attention_params<T, CROSS_ATTENTION>& params,                            \
                                    const KVBlockArray&                             kv_cache_buffer,                   \
                                    const cudaStream_t&                             stream)                            \
    {                                                                                                                  \
        multihead_attention_<T, KVBlockArray, Multihead_attention_params<T, CROSS_ATTENTION>>(                         \
            params, kv_cache_buffer, stream);                                                                          \
    }                                                                                                                  \
    void masked_multihead_attention(Multihead_attention_params<T, CROSS_ATTENTION>& params,                            \
                                    const KVLinearBuffer&                           kv_cache_buffer,                   \
                                    const cudaStream_t&                             stream)                            \
    {                                                                                                                  \
        multihead_attention_<T, KVLinearBuffer, Multihead_attention_params<T, CROSS_ATTENTION>>(                       \
            params, kv_cache_buffer, stream);                                                                          \
    }
INSTANTIATE_MMHA_NORMAL_AND_PAGED(float, true)
INSTANTIATE_MMHA_NORMAL_AND_PAGED(float, false)
INSTANTIATE_MMHA_NORMAL_AND_PAGED(uint16_t, true)
INSTANTIATE_MMHA_NORMAL_AND_PAGED(uint16_t, false)
#ifdef ENABLE_BF16
INSTANTIATE_MMHA_NORMAL_AND_PAGED(__nv_bfloat16, true)
INSTANTIATE_MMHA_NORMAL_AND_PAGED(__nv_bfloat16, false)
#endif
#undef INSTANTIATE_MMHA_NORMAL_AND_PAGED

template<typename T>
struct SATypeConverter {
    using Type = T;
};

template<>
struct SATypeConverter<half> {
    using Type = uint16_t;
};

template<typename T, typename KVCacheBuffer>
void fusedQKV_masked_attention_dispatch(const T*      qkv_buf,
                                        const T*      qkv_bias,
                                        const T*      relative_attention_bias,
                                        const int*    cache_indir,
                                        T*            context_buf,
                                        const bool*   finished,
                                        const int*    sequence_lengths,
                                        const int     inference_batch_size,
                                        const int     beam_width,
                                        const int     head_num,
                                        const int     head_num_kv,
                                        const int     size_per_head,
                                        const int     rotary_embedding_dim,
                                        const int     rotary_embedding_style,
                                        const int     rotary_embedding_base,
                                        const int     logn_seq_len,
                                        const bool    use_logn_attn,
                                        const float   dynamic_embedding_scalar,
                                        const int     dynamic_embedding_max_pos,
                                        const int     position_embeddings_scale,
                                        const int     base_scale,
                                        const int     memory_max_len,
                                        const int*    prefix_prompt_lengths,
                                        const int     max_prefix_prompt_length,
                                        const bool    count_prefix_length,
                                        const int*    input_lengths,
                                        const int     step,
                                        const float   q_scaling,
                                        const int     relative_attention_bias_stride,
                                        const T*      linear_bias_slopes,
                                        const bool*   masked_tokens,
                                        const float*  qkv_scale_out,
                                        const float*  attention_out_scale,
                                        const int     int8_mode,
                                        const bool    multi_block_mode,
                                        int           max_seq_len_tile,
                                        T*            partial_out,
                                        float*        partial_sum,
                                        float*        partial_max,
                                        int*          block_counter,
                                        KVCacheBuffer kv_cache_buffer,
                                        cudaStream_t  stream)
{
    using DataType = typename SATypeConverter<T>::Type;
    Masked_multihead_attention_params<DataType> params;
    memset(&params, 0, sizeof(params));
    int hidden_units = head_num * size_per_head;
    if (qkv_bias != nullptr) {
        params.q_bias = reinterpret_cast<const DataType*>(qkv_bias);
        params.k_bias = reinterpret_cast<const DataType*>(qkv_bias) + hidden_units;
        params.v_bias = reinterpret_cast<const DataType*>(qkv_bias) + hidden_units + size_per_head * head_num_kv;
    }
    else {
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
    }
    else {
        params.k = reinterpret_cast<const DataType*>(reinterpret_cast<const int8_t*>(qkv_buf) + hidden_units);
        params.v = reinterpret_cast<const DataType*>(reinterpret_cast<const int8_t*>(qkv_buf) + hidden_units
                                                     + size_per_head * head_num_kv);
    }

    params.int8_kv_cache = kv_cache_buffer.int8_mode;
    // params.fp8_kv_cache = input_params.kv_cache_quant_mode.hasFp8KvCache();
    // if (input_params.kv_cache_quant_mode.hasKvCacheQuant())
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
    params.timestep                       = step + max_prefix_prompt_length - 1;
    params.num_heads                      = head_num;
    params.num_kv_heads                   = head_num_kv;
    params.hidden_size_per_head           = size_per_head;
    params.rotary_embedding_dim           = rotary_embedding_dim;
    params.rotary_embedding_base          = rotary_embedding_base;
    params.rotary_embedding_scale         = dynamic_embedding_scalar;
    params.rotary_embedding_max_positions = dynamic_embedding_max_pos;
    params.rotary_embedding_style         = rotary_embedding_style;
    params.use_logn_attn                  = use_logn_attn;
    params.logn_seq_len                   = logn_seq_len;
    // Note: keep norm factor (sqrt(K_dim)) when adopting megatron T5 structure (may adjust)
    params.inv_sqrt_dh = 1.F / (sqrtf((float)size_per_head) * q_scaling);
    if (relative_attention_bias != nullptr) {
        params.relative_attention_bias = reinterpret_cast<const DataType*>(relative_attention_bias);
    }
    params.relative_attention_bias_stride = relative_attention_bias_stride;
    params.max_distance                   = 0;
    params.position_embeddings_scale      = position_embeddings_scale;
    params.base_scale                     = base_scale;
    // The slope of linear position bias per head, e.g., ALiBi.
    if (linear_bias_slopes != nullptr) {
        params.linear_bias_slopes = reinterpret_cast<const DataType*>(linear_bias_slopes);
    }

    params.input_lengths = input_lengths;

    params.multi_block_mode = multi_block_mode;
    if (multi_block_mode)
    {
        params.max_seq_len_tile = max_seq_len_tile;
        params.partial_out = reinterpret_cast<DataType*>(partial_out);
        params.partial_sum = partial_sum;
        params.partial_max = partial_max;
        params.block_counter = block_counter;
    }

    // get device attributes
    int device_id;
    cudaGetDevice(&device_id);
    cudaDeviceGetAttribute(&params.multi_processor_count, cudaDevAttrMultiProcessorCount, device_id);
    PUSH_RANGE(stream, "mmha");
    masked_multihead_attention(params, kv_cache_buffer, stream);
    POP_RANGE;
    sync_check_cuda_error();
}

#define INSTANTIATE_FUSEDQKV_MASKED_ATTENTION_DISPATCH(T, KVCacheBuffer)                                               \
    template void fusedQKV_masked_attention_dispatch(const T*      qkv_buf,                                            \
                                                     const T*      qkv_bias,                                           \
                                                     const T*      relative_attention_bias,                            \
                                                     const int*    cache_indir,                                        \
                                                     T*            context_buf,                                        \
                                                     const bool*   finished,                                           \
                                                     const int*    sequence_lengths,                                   \
                                                     const int     inference_batch_size,                               \
                                                     const int     beam_width,                                         \
                                                     const int     head_num,                                           \
                                                     const int     head_num_kv,                                        \
                                                     const int     size_per_head,                                      \
                                                     const int     rotary_embedding_dim,                               \
                                                     const int     rotary_embedding_style,                             \
                                                     const int     rotary_embedding_base,                              \
                                                     const int     logn_seq_len,                                       \
                                                     const bool    use_logn_attn,                                      \
                                                     const float   dynamic_embedding_scalar,                           \
                                                     const int     dynamic_embedding_max_pos,                          \
                                                     const int     position_embeddings_scale,                          \
                                                     const int     base_scale,                                         \
                                                     const int     memory_max_len,                                     \
                                                     const int*    prefix_prompt_lengths,                              \
                                                     const int     max_prefix_prompt_length,                           \
                                                     const bool    count_prefix_length,                                \
                                                     const int*    input_lengths,                                      \
                                                     const int     step,                                               \
                                                     const float   q_scaling,                                          \
                                                     const int     relative_attention_bias_stride,                     \
                                                     const T*      linear_bias_slopes,                                 \
                                                     const bool*   masked_tokens,                                      \
                                                     const float*  qkv_scale_out,                                      \
                                                     const float*  attention_out_scale,                                \
                                                     const int     int8_mode,                                          \
                                                     const bool    multi_block_mode,                                   \
                                                     int           max_seq_tile,                                       \
                                                     T*            partial_out,                                        \
                                                     float*        partial_sum,                                        \
                                                     float*        partial_max,                                        \
                                                     int*          block_counter,                                      \
                                                     KVCacheBuffer kv_cache_buffer,                                    \
                                                     cudaStream_t  stream)

INSTANTIATE_FUSEDQKV_MASKED_ATTENTION_DISPATCH(float, KVLinearBuffer);
INSTANTIATE_FUSEDQKV_MASKED_ATTENTION_DISPATCH(half, KVLinearBuffer);
INSTANTIATE_FUSEDQKV_MASKED_ATTENTION_DISPATCH(float, KVBlockArray);
INSTANTIATE_FUSEDQKV_MASKED_ATTENTION_DISPATCH(half, KVBlockArray);
#ifdef ENABLE_BF16
INSTANTIATE_FUSEDQKV_MASKED_ATTENTION_DISPATCH(__nv_bfloat16, KVLinearBuffer);
INSTANTIATE_FUSEDQKV_MASKED_ATTENTION_DISPATCH(__nv_bfloat16, KVBlockArray);
#endif

#undef INSTANTIATE_FUSEDQKV_MASKED_ATTENTION_DISPATCH

////////////////////////////////////////////////////////////////////////////////////////////////////
}  // namespace fastertransformer
