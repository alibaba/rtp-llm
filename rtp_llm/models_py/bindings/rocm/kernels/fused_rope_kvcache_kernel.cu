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

#include "rtp_llm/cpp/utils/utils.h"
#include "rtp_llm/cpp/utils/math_utils.h"
#include "rtp_llm/models_py/bindings/common/kernels/kv_cache/kv_cache_utils.h"
#include "rtp_llm/models_py/bindings/cuda/reduce_kernel_utils.cuh"
#include "rtp_llm/models_py/bindings/common/kernels/rotary_position_embedding.h"
#include "rtp_llm/models_py/bindings/rocm/kernels/fused_rope_kvcache_kernel.h"
#include "rtp_llm/models_py/bindings/cuda/cuda_type_utils.cuh"
#include "rtp_llm/models_py/bindings/rocm/cuda_shims.h"
#include <cstdlib>

namespace rtp_llm {

template<typename T>
struct Vec_t {
    static constexpr int size = 0;
};

template<>
struct Vec_t<float> {
    using Type                = float2;
    static constexpr int size = 2;
#ifdef ENABLE_FP8
    using QuantizedType = fp8_2_t;
#endif
};

template<>
struct Vec_t<half> {
    using Type                = uint32_t;
    static constexpr int size = 2;
#ifdef ENABLE_FP8
    using QuantizedType = fp8_2_t;
#endif
};

#ifdef ENABLE_BF16
template<>
struct Vec_t<__nv_bfloat16> {
    using Type                = __nv_bfloat162;
    static constexpr int size = 2;
#ifdef ENABLE_FP8
    using QuantizedType = fp8_2_t;
#endif
};
#endif

template<typename T>
struct Vec_t2 {
    static constexpr int size = 0;
};

template<>
struct Vec_t2<float> {
    using Type                = Float8_;
    static constexpr int size = 8;
#ifdef ENABLE_FP8
    using QuantizedType = fp8_8_t;
#endif
};

template<>
struct Vec_t2<half> {
    using Type                = uint4;
    static constexpr int size = 8;
#ifdef ENABLE_FP8
    using QuantizedType = fp8_8_t;
#endif
};

#ifdef ENABLE_BF16
template<>
struct Vec_t2<__nv_bfloat16> {
    using Type                = bf16_8_t;
    static constexpr int size = 8;
#ifdef ENABLE_FP8
    using QuantizedType = fp8_8_t;
#endif
};
#endif

// Multiple calls of reinterpret_cast.
template<typename type_in, typename type_out>
inline __device__ type_out* reinterpret_ptr(void* ptr, size_t offset) {
    return reinterpret_cast<type_out*>(reinterpret_cast<type_in*>(ptr) + offset);
}

inline __device__ void convert_to_fp8(__hip_fp8x2_e4m3_fnuz* v, const amd_bfloat162 u) {
    __hip_bfloat162_raw   raw_bf16  = *reinterpret_cast<const __hip_bfloat162_raw*>(&u);
    __hip_fp8x2_storage_t raw_fp8x2 = __hip_cvt_bfloat16raw2_to_fp8x2(raw_bf16, __HIP_SATFINITE, __HIP_E4M3_FNUZ);
    *v                              = *reinterpret_cast<__hip_fp8x2_e4m3_fnuz*>(&raw_fp8x2);
}

inline __device__ void convert_to_fp8(__hip_fp8x2_e4m3_fnuz* v, const float2 u) {
    __half2               h2        = __float22half2_rn(u);
    __half2_raw           raw_h2    = *reinterpret_cast<const __half2_raw*>(&h2);
    __hip_fp8x2_storage_t raw_fp8x2 = __hip_cvt_halfraw2_to_fp8x2(raw_h2, __HIP_SATFINITE, __HIP_E4M3_FNUZ);
    *v                              = *reinterpret_cast<const __hip_fp8x2_e4m3_fnuz*>(&raw_fp8x2);
}

inline __device__ void convert_to_fp8(__hip_fp8x2_e4m3_fnuz* v, const uint32_t u) {
    __half2_raw           raw_h2    = *reinterpret_cast<const __half2_raw*>(&u);
    __hip_fp8x2_storage_t raw_fp8x2 = __hip_cvt_halfraw2_to_fp8x2(raw_h2, __HIP_SATFINITE, __HIP_E4M3_FNUZ);
    *v                              = *reinterpret_cast<const __hip_fp8x2_e4m3_fnuz*>(&raw_fp8x2);
}

template<typename T, typename Tcache, bool PREFIX_PROMPT, bool USE_PAGED_FMHA, RopeStyle ROPE_STYLE>
__global__ void add_fusedQKV_bias_transpose_prefill_kernel_v1(T*                            q_buf,
                                                              T*                            k_buf,
                                                              T*                            v_buf,
                                                              PrefixPromptBatchWeightsParam param,
                                                              T*                            QKV,
                                                              void*                         QuantizedQKV,
                                                              const int*                    position_ids,
                                                              const T* __restrict qkv_bias,
                                                              const int*    padding_offset,
                                                              const int*    cu_seqlens,
                                                              const int     batch_size,
                                                              const int     seq_len,
                                                              const int     head_num,
                                                              const int     head_num_kv,
                                                              const int     size_per_head,
                                                              RopeConfig    rope_config,
                                                              const bool    use_logn_attn,
                                                              bool          store_qkv,
                                                              bool          store_q,
                                                              bool          store_kv,
                                                              bool          store_cache,
                                                              const float2* cos_sin_cache) {
    // This kernel add bias to QKV, which has shape [batch_size, seq_len, 3,
    // head_num, size_per_head], and QKV split to 3 split buffer q, k, v and
    // transpose them to [batch_size, head_num, seq_len, size_per_head]. For q and
    // k, also apply the rotary embedding.

    // When we pass prefix prompt, this kernel also concatenate the prefix prompt
    // and key/value along seq_len dimension like [prompt, key/value]. So, the
    // final shape of q is same ([batch_size, head_num, seq_len, size_per_head]),
    // but the shapes of key and values become [batch_size, head_num,
    // max_prefix_prompt_length + seq_len, size_per_head].

    // NOTE: QKV src shape (batch_size, seq_len, 3, head_num, size_per_head)
    //  QKV dst shape (3, batch_size, head_num, seq_len, size_per_head)
    extern __shared__ __align__(sizeof(float2)) char smem_[];  // align on largest vector type

    static constexpr bool ENABLE_8BITS_CACHE = sizeof(Tcache) == 1;

#ifdef ENABLE_FP8
    // Quantized output only supports fp8 currently.
    using QuantizedEltType = __nv_fp8_e4m3;
    using QuantizedVecType = typename Vec_t<T>::QuantizedType;
#endif
    constexpr int vec_size         = Vec_t<T>::size;
    using Vec_t                    = typename Vec_t<T>::Type;
    const int token_idx            = blockIdx.x;
    const int token_padding_offset = padding_offset == nullptr ? 0 : padding_offset[token_idx];
    const int tgt_token_idx        = token_idx + token_padding_offset;

    const int batch_idx = tgt_token_idx / seq_len;
    const int seq_idx   = tgt_token_idx % seq_len;

    const int head_idx      = blockIdx.y;
    const int tidx          = threadIdx.x;
    const int total_seq_len = param.max_prefix_prompt_length + seq_len;

    if (tidx * vec_size >= size_per_head) {
        return;
    }

    const int prefix_prompt_length = PREFIX_PROMPT ? param.d_prefix_prompt_lengths[batch_idx] : 0;
    const int hidden_idx           = head_idx * size_per_head + tidx * vec_size;
    const int n                    = head_num * size_per_head;
    const int kv_n                 = head_num_kv * size_per_head;  // MQA
    // the [0..seq_len) indices really handle KV [max_pp_len..seq_len+max_pp_len)
    // and Q [0..seq_len)
    // Note: if !PREFIX_PROMPT, max_pp_len = 0, so it's no-op
    const int dst_kv_seq_idx = seq_idx + prefix_prompt_length;

    // NOTE: q has seq len excluding prefix prompt
    // src QKV: [batch, time, 3, head, hidden]
    const int src_q_idx = token_idx * (n + 2 * kv_n) + hidden_idx;
    const int src_k_idx = token_idx * (n + 2 * kv_n) + hidden_idx + n;
    const int src_v_idx = token_idx * (n + 2 * kv_n) + hidden_idx + kv_n + n;

    Vec_t q, k, v;
    q = *reinterpret_cast<const Vec_t*>(&QKV[src_q_idx]);

    if (head_idx < head_num_kv) {
        k = *reinterpret_cast<const Vec_t*>(&QKV[src_k_idx]);
        v = *reinterpret_cast<const Vec_t*>(&QKV[src_v_idx]);
    }

    if (qkv_bias) {
        Vec_t q_bias, k_bias, v_bias;
        q_bias = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx]);
        q      = add(q, q_bias);

        if (head_idx < head_num_kv) {
            k_bias = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx + n]);
            v_bias = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx + n + kv_n]);
            k      = add(k, k_bias);
            v      = add(v, v_bias);
        }
    }
    int position_id = -1;
    if (rope_config.style == RopeStyle::Mrope) {
        int rope_dim = rope_config.mrope_dim1 + rope_config.mrope_dim2 + rope_config.mrope_dim3;
        int now_idx = tidx % rope_dim, now_dim = 0;
        if (now_idx >= rope_config.mrope_dim1 + rope_config.mrope_dim2) {
            now_dim = 2;
        } else if (now_idx >= rope_config.mrope_dim1) {
            now_dim = 1;
        }
        position_id = position_ids[token_idx * rope_config.index_factor + now_dim];
    } else if (position_ids) {
        position_id = position_ids[token_idx * rope_config.index_factor];
    }
    const int pre_len   = cu_seqlens[batch_idx];
    const int input_len = cu_seqlens[batch_idx + 1] - pre_len;
    context_rope<T, Vec_t, ROPE_STYLE>(rope_config,
                                       q,
                                       k,
                                       reinterpret_cast<T*>(smem_),
                                       tidx,
                                       seq_idx,
                                       position_id,
                                       seq_len,
                                       input_len,
                                       PREFIX_PROMPT,
                                       prefix_prompt_length,
                                       param.count_length,
                                       cos_sin_cache);

    if (use_logn_attn) {
        logn_attention(q, seq_idx, rope_config.max_pos);
    }

    __syncthreads();

    if (store_qkv) {
        *reinterpret_cast<Vec_t*>(&QKV[src_q_idx]) = q;
        if (head_idx < head_num_kv) {
#ifdef ENABLE_FP8
            if (QuantizedQKV != nullptr) {
                // use 1.0f scale currently for qkv input of FP8 FMHA.
                convert_to_fp8(
                    reinterpret_cast<QuantizedVecType*>(reinterpret_cast<QuantizedEltType*>(QuantizedQKV) + src_k_idx),
                    k);
                convert_to_fp8(
                    reinterpret_cast<QuantizedVecType*>(reinterpret_cast<QuantizedEltType*>(QuantizedQKV) + src_v_idx),
                    v);
            }
#endif
            *reinterpret_cast<Vec_t*>(&QKV[src_k_idx]) = k;
            *reinterpret_cast<Vec_t*>(&QKV[src_v_idx]) = v;
        }
#ifdef ENABLE_FP8
        if (QuantizedQKV != nullptr) {
            size_t dest_q_idx = batch_idx * size_per_head * seq_len * head_num + head_idx * size_per_head * seq_len
                                + seq_idx * size_per_head + tidx * vec_size;
            if constexpr (USE_PAGED_FMHA) {
                dest_q_idx =
                    (pre_len + seq_idx) * size_per_head * head_num + head_idx * size_per_head + tidx * vec_size;
            }
            *reinterpret_cast<Vec_t*>(&q_buf[dest_q_idx]) = q;
            QuantizedVecType* quantized_q_ptr =
                USE_PAGED_FMHA ? reinterpret_ptr<QuantizedEltType, QuantizedVecType>(q_buf, dest_q_idx) :
                                 reinterpret_ptr<QuantizedEltType, QuantizedVecType>(QuantizedQKV, src_q_idx);
            convert_to_fp8(quantized_q_ptr, q);
        }
#endif
    }

    if (store_q) {
        size_t dest_q_idx = batch_idx * size_per_head * seq_len * head_num + head_idx * size_per_head * seq_len
                            + seq_idx * size_per_head + tidx * vec_size;
        if constexpr (USE_PAGED_FMHA) {
            dest_q_idx = (pre_len + seq_idx) * size_per_head * head_num + head_idx * size_per_head + tidx * vec_size;
        }
        *reinterpret_cast<Vec_t*>(&q_buf[dest_q_idx]) = q;
    }

    if (store_kv) {
        const int dest_kv_idx = batch_idx * size_per_head * total_seq_len * head_num_kv
                                + head_idx * size_per_head * total_seq_len + dst_kv_seq_idx * size_per_head
                                + tidx * vec_size;

        if (head_idx < head_num_kv) {
            *reinterpret_cast<Vec_t*>(&k_buf[dest_kv_idx]) = k;
            *reinterpret_cast<Vec_t*>(&v_buf[dest_kv_idx]) = v;
        }
    }

    if (store_cache) {
        if (head_idx < head_num_kv) {
            KVBlockArray kv_block_array = param.kv_block_array;
            Tcache*      k_cache = reinterpret_cast<Tcache*>(kv_block_array.getKBlockPtr(batch_idx, dst_kv_seq_idx));
            Tcache*      v_cache = reinterpret_cast<Tcache*>(kv_block_array.getVBlockPtr(batch_idx, dst_kv_seq_idx));
            if constexpr (std::is_same<Tcache, __nv_fp8_e4m3>::value) {
                float* k_scale_ptr   = reinterpret_cast<float*>(kv_block_array.getKScalePtr(batch_idx, dst_kv_seq_idx));
                float* v_scale_ptr   = reinterpret_cast<float*>(kv_block_array.getVScalePtr(batch_idx, dst_kv_seq_idx));
                const int inScaleIdx = kv_block_array.getKVScaleLocalIdx(dst_kv_seq_idx, head_idx);

                __shared__ float s_max[2];
                s_max[0] = float(1 << (8 - 1));
                s_max[1] = float(1 << (8 - 1));

#pragma unroll
                for (int vec_i = 0; vec_i < vec_size; vec_i++) {
                    const int inKBlockIdx = kv_block_array.getKLocalIdx<KvCacheDataType::FP8>(
                        dst_kv_seq_idx, head_idx, size_per_head, tidx * vec_size + vec_i);

                    const int inVBlockIdx =
                        kv_block_array.getVLocalIdx(dst_kv_seq_idx, head_idx, size_per_head, tidx * vec_size + vec_i);

                    k_cache[inKBlockIdx] =
                        Tcache(float(reinterpret_cast<T*>(&k)[vec_i]) * (float(1 << (8 - 1)) / s_max[0]));
                    v_cache[inVBlockIdx] =
                        Tcache(float(reinterpret_cast<T*>(&v)[vec_i]) * (float(1 << (8 - 1)) / s_max[1]));
                }

                if (tidx == 0) {
                    *reinterpret_cast<float*>(&k_scale_ptr[inScaleIdx]) = s_max[0] / float(1 << (8 - 1));
                    *reinterpret_cast<float*>(&v_scale_ptr[inScaleIdx]) = s_max[1] / float(1 << (8 - 1));
                }
            } else {
#pragma unroll
                for (int vec_i = 0; vec_i < vec_size; vec_i++) {
                    const int inKBlockIdx = kv_block_array.getKLocalIdx<KvCacheDataType::BASE>(
                        dst_kv_seq_idx, head_idx, size_per_head, tidx * vec_size + vec_i);
                    k_cache[inKBlockIdx] = reinterpret_cast<T*>(&k)[vec_i];

                    const int inVBlockIdx =
                        kv_block_array.getVLocalIdx(dst_kv_seq_idx, head_idx, size_per_head, tidx * vec_size + vec_i);
                    v_cache[inVBlockIdx] = reinterpret_cast<T*>(&v)[vec_i];
                }
            }
        }
    }
}

template<typename T>
void invokeAddFusedQKVBiasTransposePrefillV1(T*                             q_buf,
                                             T*                             k_buf,
                                             T*                             v_buf,
                                             PrefixPromptBatchWeightsParam* param_ptr,
                                             T*                             QKV,
                                             void*                          QuantizedQKV,
                                             const int*                     position_ids,
                                             const T*                       qkv_bias,
                                             const int*                     padding_offset,
                                             const int*                     cu_seqlens,
                                             const int                      batch_size,
                                             const int                      seq_len,
                                             const int                      token_num,
                                             const int                      head_num,
                                             const int                      head_num_kv,
                                             const int                      size_per_head,
                                             const RopeConfig               rope_config,
                                             const bool                     use_logn_attn,
                                             const float*                   scale,
                                             const int                      int8_mode,
                                             const bool                     use_paged_fmha,
                                             const bool                     store_qkv,
                                             const bool                     store_q,
                                             const bool                     store_kv,
                                             const bool                     store_cache,
                                             const float2*                  cos_sin_cache,
                                             cudaStream_t                   stream) {
    auto&  param = *param_ptr;
    dim3   block((size_per_head / Vec_t<T>::size + 31) / 32 * 32);
    dim3   grid(token_num, head_num);
    size_t smem_size = rope_config.style == RopeStyle::No ? 0 : 2 * rope_config.dim * sizeof(T);

    FT_SWITCH(param.max_prefix_prompt_length != 0, PREFIX_PROMPT, [&] {
        FT_SWITCH(use_paged_fmha, USE_PAGED_FMHA, [&] {
            FT_SWITCH_KV_CACHE_TYPE_CASE(param.kv_block_array.cache_type, Tcache, [&] {
                FT_ROPE_SWITCH(rope_config.style, ROPE_STYLE, [&] {
                    add_fusedQKV_bias_transpose_prefill_kernel_v1<T, Tcache, PREFIX_PROMPT, USE_PAGED_FMHA, ROPE_STYLE>
                        <<<grid, block, smem_size, stream>>>(q_buf,
                                                             k_buf,
                                                             v_buf,
                                                             param,
                                                             QKV,
                                                             QuantizedQKV,
                                                             position_ids,
                                                             qkv_bias,
                                                             padding_offset,
                                                             cu_seqlens,
                                                             batch_size,
                                                             seq_len,
                                                             head_num,
                                                             head_num_kv,
                                                             size_per_head,
                                                             rope_config,
                                                             use_logn_attn,
                                                             store_qkv,
                                                             store_q,
                                                             store_kv,
                                                             store_cache,
                                                             cos_sin_cache);
                });
            });
        });
    });
}

template<typename T, typename Tcache, bool PREFIX_PROMPT, bool USE_PAGED_FMHA, bool PAD_QUERY, RopeStyle ROPE_STYLE>
__global__ void add_fusedQKV_bias_transpose_prefill_kernel(T*                            q_buf,
                                                           T*                            k_buf,
                                                           T*                            v_buf,
                                                           PrefixPromptBatchWeightsParam param,
                                                           T*                            QKV,
                                                           void*                         QuantizedQKV,
                                                           const int*                    position_ids,
                                                           const T* __restrict qkv_bias,
                                                           const int*    padding_offset,
                                                           const int*    cu_seqlens,
                                                           const int     batch_size,
                                                           const int     seq_len,
                                                           const int     head_num,
                                                           const int     head_num_kv,
                                                           const int     size_per_head,
                                                           RopeConfig    rope_config,
                                                           const bool    use_logn_attn,
                                                           bool          store_qkv,
                                                           bool          store_q,
                                                           bool          store_kv,
                                                           bool          store_cache,
                                                           const float2* cos_sin_cache) {
    // This kernel add bias to QKV, which has shape [batch_size, seq_len, 3,
    // head_num, size_per_head], and QKV split to 3 split buffer q, k, v and
    // transpose them to [batch_size, head_num, seq_len, size_per_head]. For q and
    // k, also apply the rotary embedding.

    // When we pass prefix prompt, this kernel also concatenate the prefix prompt
    // and key/value along seq_len dimension like [prompt, key/value]. So, the
    // final shape of q is same ([batch_size, head_num, seq_len, size_per_head]),
    // but the shapes of key and values become [batch_size, head_num,
    // max_prefix_prompt_length + seq_len, size_per_head].

    // NOTE: QKV src shape (batch_size, seq_len, 3, head_num, size_per_head)
    //  QKV dst shape (3, batch_size, head_num, seq_len, size_per_head)
    extern __shared__ __align__(sizeof(float2)) char smem_[];  // align on largest vector type

    static constexpr bool ENABLE_8BITS_CACHE = sizeof(Tcache) == 1;

    // Quantized output only supports fp8 currently.
    using QuantizedEltType = __hip_fp8_e4m3_fnuz;
    using QuantizedVecType = __hip_fp8x2_e4m3_fnuz;

    constexpr int vec_size         = Vec_t<T>::size;
    using Vec_t                    = typename Vec_t<T>::Type;
    const int token_idx            = blockIdx.x;
    const int token_padding_offset = padding_offset == nullptr ? 0 : padding_offset[token_idx];
    const int tgt_token_idx        = token_idx + token_padding_offset;

    const int batch_idx = tgt_token_idx / seq_len;
    const int seq_idx   = tgt_token_idx % seq_len;

    const int head_idx      = blockIdx.y;
    const int tidx          = threadIdx.x;
    const int total_seq_len = param.max_prefix_prompt_length + seq_len;

    if (tidx * vec_size >= size_per_head) {
        return;
    }

    const int prefix_prompt_length = PREFIX_PROMPT ? param.d_prefix_prompt_lengths[batch_idx] : 0;
    const int hidden_idx           = head_idx * size_per_head + tidx * vec_size;
    const int n                    = head_num * size_per_head;
    const int kv_n                 = head_num_kv * size_per_head;  // MQA
    // the [0..seq_len) indices really handle KV [max_pp_len..seq_len+max_pp_len)
    // and Q [0..seq_len)
    // Note: if !PREFIX_PROMPT, max_pp_len = 0, so it's no-op
    const int dst_kv_seq_idx = seq_idx + prefix_prompt_length;

    // NOTE: q has seq len excluding prefix prompt
    // src QKV: [batch, time, 3, head, hidden]
    const int src_q_idx = token_idx * (n + 2 * kv_n) + hidden_idx;
    const int src_k_idx = token_idx * (n + 2 * kv_n) + hidden_idx + n;
    const int src_v_idx = token_idx * (n + 2 * kv_n) + hidden_idx + kv_n + n;

    Vec_t q, k, v;
    q = *reinterpret_cast<const Vec_t*>(&QKV[src_q_idx]);

    if (head_idx < head_num_kv) {
        k = *reinterpret_cast<const Vec_t*>(&QKV[src_k_idx]);
        v = *reinterpret_cast<const Vec_t*>(&QKV[src_v_idx]);
    }

    if (qkv_bias) {
        Vec_t q_bias, k_bias, v_bias;
        q_bias = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx]);
        q      = add(q, q_bias);

        if (head_idx < head_num_kv) {
            k_bias = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx + n]);
            v_bias = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx + n + kv_n]);
            k      = add(k, k_bias);
            v      = add(v, v_bias);
        }
    }
    int position_id = -1;
    if (rope_config.style == RopeStyle::Mrope) {
        int rope_dim = rope_config.mrope_dim1 + rope_config.mrope_dim2 + rope_config.mrope_dim3;
        int now_idx = tidx % rope_dim, now_dim = 0;
        if (now_idx >= rope_config.mrope_dim1 + rope_config.mrope_dim2) {
            now_dim = 2;
        } else if (now_idx >= rope_config.mrope_dim1) {
            now_dim = 1;
        }
        position_id = position_ids[token_idx * rope_config.index_factor + now_dim];
    } else if (position_ids) {
        position_id = position_ids[token_idx * rope_config.index_factor];
    }
    const int pre_len   = cu_seqlens[batch_idx];
    const int input_len = cu_seqlens[batch_idx + 1] - pre_len;
    context_rope<T, Vec_t, ROPE_STYLE>(rope_config,
                                       q,
                                       k,
                                       reinterpret_cast<T*>(smem_),
                                       tidx,
                                       seq_idx,
                                       position_id,
                                       seq_len,
                                       input_len,
                                       PREFIX_PROMPT,
                                       prefix_prompt_length,
                                       param.count_length,
                                       cos_sin_cache);

    if (use_logn_attn) {
        logn_attention(q, seq_idx, rope_config.max_pos);
    }

    __syncthreads();

    if (store_qkv) {
        *reinterpret_cast<Vec_t*>(&QKV[src_q_idx]) = q;
        if (head_idx < head_num_kv) {
            if (QuantizedQKV != nullptr) {
                // use 1.0f scale currently for qkv input of FP8 FMHA.
                convert_to_fp8(
                    reinterpret_cast<QuantizedVecType*>(reinterpret_cast<QuantizedEltType*>(QuantizedQKV) + src_k_idx),
                    k);
                convert_to_fp8(
                    reinterpret_cast<QuantizedVecType*>(reinterpret_cast<QuantizedEltType*>(QuantizedQKV) + src_v_idx),
                    v);
            }
            *reinterpret_cast<Vec_t*>(&QKV[src_k_idx]) = k;
            *reinterpret_cast<Vec_t*>(&QKV[src_v_idx]) = v;
        }

        if (QuantizedQKV != nullptr) {
            size_t dest_q_idx = batch_idx * size_per_head * seq_len * head_num + head_idx * size_per_head * seq_len
                                + seq_idx * size_per_head + tidx * vec_size;
            if constexpr (USE_PAGED_FMHA) {
                dest_q_idx =
                    (pre_len + seq_idx) * size_per_head * head_num + head_idx * size_per_head + tidx * vec_size;
            }
            *reinterpret_cast<Vec_t*>(&q_buf[dest_q_idx]) = q;
            QuantizedVecType* quantized_q_ptr =
                USE_PAGED_FMHA ? reinterpret_ptr<QuantizedEltType, QuantizedVecType>(q_buf, dest_q_idx) :
                                 reinterpret_ptr<QuantizedEltType, QuantizedVecType>(QuantizedQKV, src_q_idx);
            convert_to_fp8(quantized_q_ptr, q);
        }
    }

    if (store_q) {
        size_t dest_q_idx = batch_idx * size_per_head * seq_len * head_num + head_idx * size_per_head * seq_len
                            + seq_idx * size_per_head + tidx * vec_size;
        if constexpr (USE_PAGED_FMHA) {
            if constexpr (PAD_QUERY) {
                dest_q_idx = (batch_idx * seq_len + (seq_len - input_len) + seq_idx) * size_per_head * head_num
                             + head_idx * size_per_head + tidx * vec_size;
            } else {
                dest_q_idx =
                    (pre_len + seq_idx) * size_per_head * head_num + head_idx * size_per_head + tidx * vec_size;
            }
        }
        *reinterpret_cast<Vec_t*>(&q_buf[dest_q_idx]) = q;
        if (QuantizedQKV != nullptr) {
            QuantizedVecType* quantized_q_ptr =
                USE_PAGED_FMHA ? reinterpret_ptr<QuantizedEltType, QuantizedVecType>(QuantizedQKV, dest_q_idx) :
                                 reinterpret_ptr<QuantizedEltType, QuantizedVecType>(q_buf, dest_q_idx);
            convert_to_fp8(quantized_q_ptr, q);
        }
    }

    if (store_kv) {
        const int dest_kv_idx = batch_idx * size_per_head * total_seq_len * head_num_kv
                                + head_idx * size_per_head * total_seq_len + dst_kv_seq_idx * size_per_head
                                + tidx * vec_size;

        if (head_idx < head_num_kv) {
            *reinterpret_cast<Vec_t*>(&k_buf[dest_kv_idx]) = k;
            *reinterpret_cast<Vec_t*>(&v_buf[dest_kv_idx]) = v;
        }
    }

    if (store_cache) {
        if (head_idx < head_num_kv) {
            KVBlockArray kv_block_array = param.kv_block_array;
            Tcache*      k_cache = reinterpret_cast<Tcache*>(kv_block_array.getKBlockPtr(batch_idx, dst_kv_seq_idx));
            Tcache*      v_cache = reinterpret_cast<Tcache*>(kv_block_array.getVBlockPtr(batch_idx, dst_kv_seq_idx));
            if constexpr (std::is_same<Tcache, __nv_fp8_e4m3>::value) {
                float* k_scale_ptr   = reinterpret_cast<float*>(kv_block_array.getKScalePtr(batch_idx, dst_kv_seq_idx));
                float* v_scale_ptr   = reinterpret_cast<float*>(kv_block_array.getVScalePtr(batch_idx, dst_kv_seq_idx));
                const int inScaleIdx = kv_block_array.getKVScaleLocalIdx(dst_kv_seq_idx, head_idx);

                __shared__ float s_max[2];
                s_max[0] = float(1 << (8 - 1));
                s_max[1] = float(1 << (8 - 1));

#pragma unroll
                for (int vec_i = 0; vec_i < vec_size; vec_i++) {
                    const int inKBlockIdx = kv_block_array.getKLocalIdx<KvCacheDataType::FP8>(
                        dst_kv_seq_idx, head_idx, size_per_head, tidx * vec_size + vec_i);

                    const int inVBlockIdx = kv_block_array.getVLocalIdx<KvCacheDataType::FP8>(
                        dst_kv_seq_idx, head_idx, size_per_head, tidx * vec_size + vec_i);

                    // convert_to_fp8(reinterpret_cast<__nv_fp8_e4m3*>(k_cache) + inKBlockIdx,
                    // float(reinterpret_cast<T*>(&k)[vec_i]) * float(1 << (8 - 1)) / s_max[0]);
                    k_cache[inKBlockIdx] =
                        Tcache(float(reinterpret_cast<T*>(&k)[vec_i]) * (float(1 << (8 - 1)) / s_max[0]));
                    v_cache[inVBlockIdx] =
                        Tcache(float(reinterpret_cast<T*>(&v)[vec_i]) * (float(1 << (8 - 1)) / s_max[1]));
                }

                if (tidx == 0) {
                    *reinterpret_cast<float*>(&k_scale_ptr[inScaleIdx]) = s_max[0] / float(1 << (8 - 1));
                    *reinterpret_cast<float*>(&v_scale_ptr[inScaleIdx]) = s_max[1] / float(1 << (8 - 1));
                }
            } else {
#pragma unroll
                for (int vec_i = 0; vec_i < vec_size; vec_i++) {
                    const int inKBlockIdx = kv_block_array.getKLocalIdx<KvCacheDataType::BASE>(
                        dst_kv_seq_idx, head_idx, size_per_head, tidx * vec_size + vec_i);
                    k_cache[inKBlockIdx] = reinterpret_cast<T*>(&k)[vec_i];

                    const int inVBlockIdx = kv_block_array.getVLocalIdx<KvCacheDataType::BASE>(
                        dst_kv_seq_idx, head_idx, size_per_head, tidx * vec_size + vec_i);

                    v_cache[inVBlockIdx] = reinterpret_cast<T*>(&v)[vec_i];
                }
            }
        }
    }
}

template<typename T>
void invokeAddFusedQKVBiasTransposePrefill(T*                             q_buf,
                                           T*                             k_buf,
                                           T*                             v_buf,
                                           PrefixPromptBatchWeightsParam* param_ptr,
                                           T*                             QKV,
                                           void*                          QuantizedQKV,
                                           const int*                     position_ids,
                                           const T*                       qkv_bias,
                                           const int*                     padding_offset,
                                           const int*                     cu_seqlens,
                                           const int                      batch_size,
                                           const int                      seq_len,
                                           const int                      token_num,
                                           const int                      head_num,
                                           const int                      head_num_kv,
                                           const int                      size_per_head,
                                           const RopeConfig               rope_config,
                                           const bool                     use_logn_attn,
                                           const float*                   scale,
                                           const int                      int8_mode,
                                           const bool                     use_paged_fmha,
                                           const bool                     store_qkv,
                                           const bool                     store_q,
                                           const bool                     store_kv,
                                           const bool                     store_cache,
                                           const float2*                  cos_sin_cache,
                                           const bool                     pad_query,
                                           cudaStream_t                   stream) {
    auto&  param = *param_ptr;
    dim3   block((size_per_head / Vec_t<T>::size + 31) / 32 * 32);
    dim3   grid(token_num, head_num);
    size_t smem_size = rope_config.style == RopeStyle::No ? 0 : 2 * rope_config.dim * sizeof(T);

    FT_SWITCH(param.max_prefix_prompt_length != 0, PREFIX_PROMPT, [&] {
        FT_SWITCH(use_paged_fmha, USE_PAGED_FMHA, [&] {
            FT_SWITCH(pad_query && use_paged_fmha, PAD_QUERY, [&] {
                FT_SWITCH_KV_CACHE_TYPE_CASE(param.kv_block_array.cache_type, Tcache, [&] {
                    FT_ROPE_SWITCH(rope_config.style, ROPE_STYLE, [&] {
                        add_fusedQKV_bias_transpose_prefill_kernel<T,
                                                                   Tcache,
                                                                   PREFIX_PROMPT,
                                                                   USE_PAGED_FMHA,
                                                                   PAD_QUERY,
                                                                   ROPE_STYLE>
                            <<<grid, block, smem_size, stream>>>(q_buf,
                                                                 k_buf,
                                                                 v_buf,
                                                                 param,
                                                                 QKV,
                                                                 QuantizedQKV,
                                                                 position_ids,
                                                                 qkv_bias,
                                                                 padding_offset,
                                                                 cu_seqlens,
                                                                 batch_size,
                                                                 seq_len,
                                                                 head_num,
                                                                 head_num_kv,
                                                                 size_per_head,
                                                                 rope_config,
                                                                 use_logn_attn,
                                                                 store_qkv,
                                                                 store_q,
                                                                 store_kv,
                                                                 store_cache,
                                                                 cos_sin_cache);
                    });
                });
            });
        });
    });
}

template<typename T, typename Tcache, bool PREFIX_PROMPT, bool USE_PAGED_FMHA, RopeStyle ROPE_STYLE>
__global__ void add_fusedQKV_bias_transpose_decode_kernel_v1(T*                            q_buf,
                                                             T*                            k_buf,
                                                             T*                            v_buf,
                                                             PrefixPromptBatchWeightsParam param,
                                                             const int*                    input_lengths,
                                                             T*                            QKV,
                                                             void*                         QuantizedQKV,
                                                             const int*                    position_ids,
                                                             const T* __restrict qkv_bias,
                                                             const int*    padding_offset,
                                                             const int*    cu_seqlens,
                                                             const int*    sequence_lengths,
                                                             const int     batch_size,
                                                             const int     seq_len,
                                                             const int     head_num,
                                                             const int     head_num_kv,
                                                             const int     size_per_head,
                                                             RopeConfig    rope_config,
                                                             const bool    use_logn_attn,
                                                             bool          store_qkv,
                                                             bool          store_q,
                                                             bool          store_kv,
                                                             bool          store_cache,
                                                             const float2* cos_sin_cache) {
    extern __shared__ __align__(sizeof(float2)) char smem_[];

    constexpr int vec_size         = Vec_t<T>::size;
    using Vec_t                    = typename Vec_t<T>::Type;
    const int token_idx            = blockIdx.x;
    const int token_padding_offset = padding_offset == nullptr ? 0 : padding_offset[token_idx];
    const int tgt_token_idx        = token_idx + token_padding_offset;

    const int batch_idx = tgt_token_idx / seq_len;
    const int seq_idx   = tgt_token_idx % seq_len;

    const int head_idx = blockIdx.y;
    const int tidx     = threadIdx.x;

    if (tidx * vec_size >= size_per_head) {
        return;
    }

    const int prefix_prompt_length = PREFIX_PROMPT ? param.d_prefix_prompt_lengths[batch_idx] : 0;
    const int sequence_length      = sequence_lengths[batch_idx];
    const int tlength              = sequence_length + param.max_prefix_prompt_length;
    const int hidden_idx           = head_idx * size_per_head + tidx * vec_size;
    const int n                    = head_num * size_per_head;
    const int kv_n                 = head_num_kv * size_per_head;  // MQA
    // the [0..seq_len) indices really handle KV [max_pp_len..seq_len+max_pp_len)
    // and Q [0..seq_len)
    // Note: if !PREFIX_PROMPT, max_pp_len = 0, so it's no-op
    const int dst_kv_seq_idx = seq_idx + tlength;

    // NOTE: q has seq len excluding prefix prompt
    // src QKV: [batch, time, 3, head, hidden]
    const int src_q_idx = token_idx * (n + 2 * kv_n) + hidden_idx;
    const int src_k_idx = token_idx * (n + 2 * kv_n) + hidden_idx + n;
    const int src_v_idx = token_idx * (n + 2 * kv_n) + hidden_idx + kv_n + n;

    Vec_t q, k, v;
    q = *reinterpret_cast<const Vec_t*>(&QKV[src_q_idx]);

    if (head_idx < head_num_kv) {
        k = *reinterpret_cast<const Vec_t*>(&QKV[src_k_idx]);
        v = *reinterpret_cast<const Vec_t*>(&QKV[src_v_idx]);
    }

    if (qkv_bias) {
        Vec_t q_bias, k_bias, v_bias;
        q_bias = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx]);
        q      = add(q, q_bias);

        if (head_idx < head_num_kv) {
            k_bias = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx + n]);
            v_bias = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx + n + kv_n]);
            k      = add(k, k_bias);
            v      = add(v, v_bias);
        }
    }

    // refer to the implementation of hipify decode attention
    const auto batch_beam_idx = blockIdx.y;
    const int  position_id    = position_ids == nullptr ? -1 : position_ids[token_idx * rope_config.index_factor];

    const int input_len = (input_lengths == nullptr) ? 0 : input_lengths[batch_beam_idx];
    const int timestep  = tlength;
    attention_rope<T, Vec_t, ROPE_STYLE>(rope_config,
                                         q,
                                         k,
                                         reinterpret_cast<T*>(smem_),
                                         tidx,
                                         tlength,
                                         tlength,  // timestep,
                                         sequence_length,
                                         position_id,
                                         input_len,
                                         prefix_prompt_length,
                                         true /*count_prefix_length*/,
                                         true /*HANDLE_KV*/,
                                         cos_sin_cache);

    if (use_logn_attn) {
        logn_attention(q, tlength, rope_config.max_pos);
    }

    __syncthreads();

    if (store_q) {
        size_t dest_q_idx = batch_idx * size_per_head * seq_len * head_num + head_idx * size_per_head * seq_len
                            + seq_idx * size_per_head + tidx * vec_size;
        *reinterpret_cast<Vec_t*>(&q_buf[dest_q_idx]) = q;
    }

    if (store_cache) {
        if (head_idx < head_num_kv) {
            KVBlockArray kv_block_array = param.kv_block_array;
            Tcache*      k_cache = reinterpret_cast<Tcache*>(kv_block_array.getKBlockPtr(batch_idx, dst_kv_seq_idx));
            Tcache*      v_cache = reinterpret_cast<Tcache*>(kv_block_array.getVBlockPtr(batch_idx, dst_kv_seq_idx));
            if constexpr (std::is_same<Tcache, __nv_fp8_e4m3>::value) {
                float* k_scale_ptr   = reinterpret_cast<float*>(kv_block_array.getKScalePtr(batch_idx, dst_kv_seq_idx));
                float* v_scale_ptr   = reinterpret_cast<float*>(kv_block_array.getVScalePtr(batch_idx, dst_kv_seq_idx));
                const int inScaleIdx = kv_block_array.getKVScaleLocalIdx(dst_kv_seq_idx, head_idx);

                __shared__ float s_max[2];
                s_max[0] = float(1 << (8 - 1));
                s_max[1] = float(1 << (8 - 1));
#pragma unroll
                for (int vec_i = 0; vec_i < vec_size; vec_i++) {
                    const int inKBlockIdx = kv_block_array.getKLocalIdx<KvCacheDataType::FP8>(
                        dst_kv_seq_idx, head_idx, size_per_head, tidx * vec_size + vec_i);

                    const int inVBlockIdx =
                        kv_block_array.getVLocalIdx(dst_kv_seq_idx, head_idx, size_per_head, tidx * vec_size + vec_i);

                    k_cache[inKBlockIdx] =
                        Tcache(float(reinterpret_cast<T*>(&k)[vec_i]) * (float(1 << (8 - 1)) / s_max[0]));
                    v_cache[inVBlockIdx] =
                        Tcache(float(reinterpret_cast<T*>(&v)[vec_i]) * (float(1 << (8 - 1)) / s_max[1]));
                }
                if (tidx == 0) {
                    *reinterpret_cast<float*>(&k_scale_ptr[inScaleIdx]) = s_max[0] / float(1 << (8 - 1));
                    *reinterpret_cast<float*>(&v_scale_ptr[inScaleIdx]) = s_max[1] / float(1 << (8 - 1));
                }
            } else {
#pragma unroll
                for (int vec_i = 0; vec_i < vec_size; vec_i++) {
                    const int inKBlockIdx = kv_block_array.getKLocalIdx<KvCacheDataType::BASE>(
                        dst_kv_seq_idx, head_idx, size_per_head, tidx * vec_size + vec_i);
                    k_cache[inKBlockIdx] = reinterpret_cast<T*>(&k)[vec_i];

                    const int inVBlockIdx =
                        kv_block_array.getVLocalIdx(dst_kv_seq_idx, head_idx, size_per_head, tidx * vec_size + vec_i);
                    v_cache[inVBlockIdx] = reinterpret_cast<T*>(&v)[vec_i];
                }
            }
        }
    }
}

template<typename T, typename Tcache, bool PREFIX_PROMPT, bool USE_PAGED_FMHA, RopeStyle ROPE_STYLE>
__global__ void add_fusedQKV_bias_transpose_decode_kernel(T*                            q_buf,
                                                          T*                            k_buf,
                                                          T*                            v_buf,
                                                          PrefixPromptBatchWeightsParam param,
                                                          const int*                    input_lengths,
                                                          T*                            QKV,
                                                          void*                         QuantizedQKV,
                                                          const int*                    position_ids,
                                                          const T* __restrict qkv_bias,
                                                          const int*    padding_offset,
                                                          const int*    cu_seqlens,
                                                          const int*    sequence_lengths,
                                                          const int     batch_size,
                                                          const int     seq_len,
                                                          const int     head_num,
                                                          const int     head_num_kv,
                                                          const int     size_per_head,
                                                          RopeConfig    rope_config,
                                                          const bool    use_logn_attn,
                                                          bool          store_qkv,
                                                          bool          store_q,
                                                          bool          store_kv,
                                                          bool          store_cache,
                                                          const float2* cos_sin_cache) {
    extern __shared__ __align__(sizeof(float2)) char smem_[];

    constexpr int vec_size         = Vec_t<T>::size;
    using Vec_t                    = typename Vec_t<T>::Type;
    const int token_idx            = blockIdx.x;
    const int token_padding_offset = padding_offset == nullptr ? 0 : padding_offset[token_idx];
    const int tgt_token_idx        = token_idx + token_padding_offset;

    const int             batch_idx          = tgt_token_idx / seq_len;
    const int             seq_idx            = tgt_token_idx % seq_len;
    static constexpr bool ENABLE_8BITS_CACHE = sizeof(Tcache) == 1;

    const int head_idx = blockIdx.y;
    const int tidx     = threadIdx.x;

    if (tidx * vec_size >= size_per_head) {
        return;
    }

    const int prefix_prompt_length = PREFIX_PROMPT ? param.d_prefix_prompt_lengths[batch_idx] : 0;
    const int sequence_length      = sequence_lengths[batch_idx];
    const int tlength              = sequence_length + param.max_prefix_prompt_length;
    const int hidden_idx           = head_idx * size_per_head + tidx * vec_size;
    const int n                    = head_num * size_per_head;
    const int kv_n                 = head_num_kv * size_per_head;  // MQA
    // the [0..seq_len) indices really handle KV [max_pp_len..seq_len+max_pp_len)
    // and Q [0..seq_len)
    // Note: if !PREFIX_PROMPT, max_pp_len = 0, so it's no-op
    const int dst_kv_seq_idx = seq_idx + tlength;

    // NOTE: q has seq len excluding prefix prompt
    // src QKV: [batch, time, 3, head, hidden]
    const int src_q_idx = token_idx * (n + 2 * kv_n) + hidden_idx;
    const int src_k_idx = token_idx * (n + 2 * kv_n) + hidden_idx + n;
    const int src_v_idx = token_idx * (n + 2 * kv_n) + hidden_idx + kv_n + n;

    Vec_t q, k, v;
    q = *reinterpret_cast<const Vec_t*>(&QKV[src_q_idx]);

    if (head_idx < head_num_kv) {
        k = *reinterpret_cast<const Vec_t*>(&QKV[src_k_idx]);
        v = *reinterpret_cast<const Vec_t*>(&QKV[src_v_idx]);
    }

    if (qkv_bias) {
        Vec_t q_bias, k_bias, v_bias;
        q_bias = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx]);
        q      = add(q, q_bias);

        if (head_idx < head_num_kv) {
            k_bias = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx + n]);
            v_bias = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx + n + kv_n]);
            k      = add(k, k_bias);
            v      = add(v, v_bias);
        }
    }

    // refer to the implementation of hipify decode attention
    const auto batch_beam_idx = blockIdx.y;
    const int  position_id    = position_ids == nullptr ? -1 : position_ids[token_idx * rope_config.index_factor];

    const int input_len = (input_lengths == nullptr) ? 0 : input_lengths[batch_beam_idx];
    const int timestep  = tlength;
    attention_rope<T, Vec_t, ROPE_STYLE>(rope_config,
                                         q,
                                         k,
                                         reinterpret_cast<T*>(smem_),
                                         tidx,
                                         tlength,
                                         tlength,  // timestep,
                                         sequence_length,
                                         position_id,
                                         input_len,
                                         prefix_prompt_length,
                                         true /*count_prefix_length*/,
                                         true /*HANDLE_KV*/,
                                         cos_sin_cache);

    if (use_logn_attn) {
        logn_attention(q, tlength, rope_config.max_pos);
    }

    __syncthreads();

    using QuantizedEltType = __hip_fp8_e4m3_fnuz;
    using QuantizedVecType = __hip_fp8x2_e4m3_fnuz;

    if (store_q) {
        size_t dest_q_idx = batch_idx * size_per_head * seq_len * head_num + head_idx * size_per_head * seq_len
                            + seq_idx * size_per_head + tidx * vec_size;
        if (QuantizedQKV != nullptr) {
            QuantizedVecType* quantized_q_ptr = reinterpret_ptr<QuantizedEltType, QuantizedVecType>(q_buf, dest_q_idx);
            convert_to_fp8(quantized_q_ptr, q);
        } else {
            *reinterpret_cast<Vec_t*>(&q_buf[dest_q_idx]) = q;
        }
    }
    if (store_cache) {
        if (head_idx < head_num_kv) {
            KVBlockArray kv_block_array = param.kv_block_array;
            Tcache*      k_cache = reinterpret_cast<Tcache*>(kv_block_array.getKBlockPtr(batch_idx, dst_kv_seq_idx));
            Tcache*      v_cache = reinterpret_cast<Tcache*>(kv_block_array.getVBlockPtr(batch_idx, dst_kv_seq_idx));
            if constexpr (std::is_same<Tcache, __nv_fp8_e4m3>::value) {
                float* k_scale_ptr   = reinterpret_cast<float*>(kv_block_array.getKScalePtr(batch_idx, dst_kv_seq_idx));
                float* v_scale_ptr   = reinterpret_cast<float*>(kv_block_array.getVScalePtr(batch_idx, dst_kv_seq_idx));
                const int inScaleIdx = kv_block_array.getKVScaleLocalIdx(dst_kv_seq_idx, head_idx);

                __shared__ float s_max[2];
                s_max[0] = float(1 << (8 - 1));
                s_max[1] = float(1 << (8 - 1));
#pragma unroll
                for (int vec_i = 0; vec_i < vec_size; vec_i++) {
                    const int inKBlockIdx = kv_block_array.getKLocalIdx<KvCacheDataType::FP8>(
                        dst_kv_seq_idx, head_idx, size_per_head, tidx * vec_size + vec_i);

                    const int inVBlockIdx = kv_block_array.getVLocalIdx<KvCacheDataType::FP8>(
                        dst_kv_seq_idx, head_idx, size_per_head, tidx * vec_size + vec_i);

                    k_cache[inKBlockIdx] =
                        Tcache(float(reinterpret_cast<T*>(&k)[vec_i]) * (float(1 << (8 - 1)) / s_max[0]));
                    v_cache[inVBlockIdx] =
                        Tcache(float(reinterpret_cast<T*>(&v)[vec_i]) * (float(1 << (8 - 1)) / s_max[1]));
                }

                if (tidx == 0) {
                    *reinterpret_cast<float*>(&k_scale_ptr[inScaleIdx]) = s_max[0] / float(1 << (8 - 1));
                    *reinterpret_cast<float*>(&v_scale_ptr[inScaleIdx]) = s_max[1] / float(1 << (8 - 1));
                }
            } else {
#pragma unroll
                for (int vec_i = 0; vec_i < vec_size; vec_i++) {
                    const int inKBlockIdx = kv_block_array.getKLocalIdx<KvCacheDataType::BASE>(
                        dst_kv_seq_idx, head_idx, size_per_head, tidx * vec_size + vec_i);
                    k_cache[inKBlockIdx] = reinterpret_cast<T*>(&k)[vec_i];

                    const int inVBlockIdx = kv_block_array.getVLocalIdx<KvCacheDataType::BASE>(
                        dst_kv_seq_idx, head_idx, size_per_head, tidx * vec_size + vec_i);

                    v_cache[inVBlockIdx] = reinterpret_cast<T*>(&v)[vec_i];
                }
            }
        }
    }
}

template<typename T>
void invokeAddFusedQKVBiasTransposeDecodeV1(T*                             q_buf,
                                            T*                             k_buf,
                                            T*                             v_buf,
                                            PrefixPromptBatchWeightsParam* param_ptr,
                                            const int*                     input_lengths,
                                            T*                             QKV,
                                            void*                          QuantizedQKV,
                                            const int*                     position_ids,
                                            const T*                       qkv_bias,
                                            const int*                     padding_offset,
                                            const int*                     cu_seqlens,
                                            const int*                     sequence_lengths,
                                            const int                      batch_size,
                                            const int                      seq_len,
                                            const int                      token_num,
                                            const int                      head_num,
                                            const int                      head_num_kv,
                                            const int                      size_per_head,
                                            const RopeConfig               rope_config,
                                            const bool                     use_logn_attn,
                                            const float*                   scale,
                                            const int                      int8_mode,
                                            const bool                     use_paged_fmha,
                                            const bool                     store_qkv,
                                            const bool                     store_q,
                                            const bool                     store_kv,
                                            const bool                     store_cache,
                                            const float2*                  cos_sin_cache,
                                            cudaStream_t                   stream) {
    auto&  param = *param_ptr;
    dim3   block((size_per_head / Vec_t<T>::size + 31) / 32 * 32);
    dim3   grid(token_num, head_num);
    size_t smem_size = rope_config.style == RopeStyle::No ? 0 : 2 * rope_config.dim * sizeof(T);

    FT_SWITCH(param.max_prefix_prompt_length != 0, PREFIX_PROMPT, [&] {
        FT_SWITCH(use_paged_fmha, USE_PAGED_FMHA, [&] {
            FT_SWITCH_KV_CACHE_TYPE_CASE(param.kv_block_array.cache_type, Tcache, [&] {
                FT_ROPE_SWITCH(rope_config.style, ROPE_STYLE, [&] {
                    add_fusedQKV_bias_transpose_decode_kernel_v1<T, Tcache, PREFIX_PROMPT, USE_PAGED_FMHA, ROPE_STYLE>
                        <<<grid, block, smem_size, stream>>>(q_buf,
                                                             k_buf,
                                                             v_buf,
                                                             param,
                                                             input_lengths,
                                                             QKV,
                                                             QuantizedQKV,
                                                             position_ids,
                                                             qkv_bias,
                                                             padding_offset,
                                                             cu_seqlens,
                                                             sequence_lengths,
                                                             batch_size,
                                                             seq_len,
                                                             head_num,
                                                             head_num_kv,
                                                             size_per_head,
                                                             rope_config,
                                                             use_logn_attn,
                                                             store_qkv,
                                                             store_q,
                                                             store_kv,
                                                             store_cache,
                                                             cos_sin_cache);
                });
            });
        });
    });
}

template<typename T>
void invokeAddFusedQKVBiasTransposeDecode(T*                             q_buf,
                                          T*                             k_buf,
                                          T*                             v_buf,
                                          PrefixPromptBatchWeightsParam* param_ptr,
                                          const int*                     input_lengths,
                                          T*                             QKV,
                                          void*                          QuantizedQKV,
                                          const int*                     position_ids,
                                          const T*                       qkv_bias,
                                          const int*                     padding_offset,
                                          const int*                     cu_seqlens,
                                          const int*                     sequence_lengths,
                                          const int                      batch_size,
                                          const int                      seq_len,
                                          const int                      token_num,
                                          const int                      head_num,
                                          const int                      head_num_kv,
                                          const int                      size_per_head,
                                          const RopeConfig               rope_config,
                                          const bool                     use_logn_attn,
                                          const float*                   scale,
                                          const int                      int8_mode,
                                          const bool                     use_paged_fmha,
                                          const bool                     store_qkv,
                                          const bool                     store_q,
                                          const bool                     store_kv,
                                          const bool                     store_cache,
                                          const float2*                  cos_sin_cache,
                                          cudaStream_t                   stream) {
    auto&  param = *param_ptr;
    dim3   block((size_per_head / Vec_t<T>::size + 31) / 32 * 32);
    dim3   grid(token_num, head_num);
    size_t smem_size = rope_config.style == RopeStyle::No ? 0 : 2 * rope_config.dim * sizeof(T);

    FT_SWITCH(param.max_prefix_prompt_length != 0, PREFIX_PROMPT, [&] {
        FT_SWITCH(use_paged_fmha, USE_PAGED_FMHA, [&] {
            FT_SWITCH_KV_CACHE_TYPE_CASE(param.kv_block_array.cache_type, Tcache, [&] {
                FT_ROPE_SWITCH(rope_config.style, ROPE_STYLE, [&] {
                    add_fusedQKV_bias_transpose_decode_kernel<T, Tcache, PREFIX_PROMPT, USE_PAGED_FMHA, ROPE_STYLE>
                        <<<grid, block, smem_size, stream>>>(q_buf,
                                                             k_buf,
                                                             v_buf,
                                                             param,
                                                             input_lengths,
                                                             QKV,
                                                             QuantizedQKV,
                                                             position_ids,
                                                             qkv_bias,
                                                             padding_offset,
                                                             cu_seqlens,
                                                             sequence_lengths,
                                                             batch_size,
                                                             seq_len,
                                                             head_num,
                                                             head_num_kv,
                                                             size_per_head,
                                                             rope_config,
                                                             use_logn_attn,
                                                             store_qkv,
                                                             store_q,
                                                             store_kv,
                                                             store_cache,
                                                             cos_sin_cache);
                });
            });
        });
    });
}

#define INSTANTIATEADDFUSEDQKVBIASTRANSPOSEPREFILLV1(T)                                                                \
    template void invokeAddFusedQKVBiasTransposePrefillV1(T*                             q_buf,                        \
                                                          T*                             k_buf,                        \
                                                          T*                             v_buf,                        \
                                                          PrefixPromptBatchWeightsParam* param,                        \
                                                          T*                             QKV,                          \
                                                          void*                          QuantizedQKV,                 \
                                                          const int*                     position_ids,                 \
                                                          const T*                       qkv_bias,                     \
                                                          const int*                     padding_offset,               \
                                                          const int*                     cu_seqlens,                   \
                                                          const int                      batch_size,                   \
                                                          const int                      seq_len,                      \
                                                          const int                      token_num,                    \
                                                          const int                      head_num,                     \
                                                          const int                      head_num_kv,                  \
                                                          const int                      size_per_head,                \
                                                          const RopeConfig               rope_config,                  \
                                                          const bool                     use_logn_attn,                \
                                                          const float*                   scale,                        \
                                                          const int                      int8_mode,                    \
                                                          const bool                     use_paged_fmha,               \
                                                          const bool                     store_qkv,                    \
                                                          const bool                     store_q,                      \
                                                          const bool                     store_kv,                     \
                                                          const bool                     store_cache,                  \
                                                          const float2*                  cos_sin_cache,                \
                                                          cudaStream_t                   stream)
INSTANTIATEADDFUSEDQKVBIASTRANSPOSEPREFILLV1(float);
INSTANTIATEADDFUSEDQKVBIASTRANSPOSEPREFILLV1(half);
#ifdef ENABLE_BF16
INSTANTIATEADDFUSEDQKVBIASTRANSPOSEPREFILLV1(__nv_bfloat16);
#endif
#undef INSTANTIATEADDFUSEDQKVBIASTRANSPOSEPREFILLV1

#define INSTANTIATEADDFUSEDQKVBIASTRANSPOSEPREFILL(T)                                                                  \
    template void invokeAddFusedQKVBiasTransposePrefill(T*                             q_buf,                          \
                                                        T*                             k_buf,                          \
                                                        T*                             v_buf,                          \
                                                        PrefixPromptBatchWeightsParam* param,                          \
                                                        T*                             QKV,                            \
                                                        void*                          QuantizedQKV,                   \
                                                        const int*                     position_ids,                   \
                                                        const T*                       qkv_bias,                       \
                                                        const int*                     padding_offset,                 \
                                                        const int*                     cu_seqlens,                     \
                                                        const int                      batch_size,                     \
                                                        const int                      seq_len,                        \
                                                        const int                      token_num,                      \
                                                        const int                      head_num,                       \
                                                        const int                      head_num_kv,                    \
                                                        const int                      size_per_head,                  \
                                                        const RopeConfig               rope_config,                    \
                                                        const bool                     use_logn_attn,                  \
                                                        const float*                   scale,                          \
                                                        const int                      int8_mode,                      \
                                                        const bool                     use_paged_fmha,                 \
                                                        const bool                     store_qkv,                      \
                                                        const bool                     store_q,                        \
                                                        const bool                     store_kv,                       \
                                                        const bool                     store_cache,                    \
                                                        const float2*                  cos_sin_cache,                  \
                                                        const bool                     pad_query,                      \
                                                        cudaStream_t                   stream)
INSTANTIATEADDFUSEDQKVBIASTRANSPOSEPREFILL(float);
INSTANTIATEADDFUSEDQKVBIASTRANSPOSEPREFILL(half);
#ifdef ENABLE_BF16
INSTANTIATEADDFUSEDQKVBIASTRANSPOSEPREFILL(__nv_bfloat16);
#endif
#undef INSTANTIATEADDFUSEDQKVBIASTRANSPOSEPREFILL

#define INSTANTIATEADDFUSEDQKVBIASTRANSPOSEDECODEV1(T)                                                                 \
    template void invokeAddFusedQKVBiasTransposeDecodeV1(T*                             q_buf,                         \
                                                         T*                             k_buf,                         \
                                                         T*                             v_buf,                         \
                                                         PrefixPromptBatchWeightsParam* param,                         \
                                                         const int*                     input_lengths,                 \
                                                         T*                             QKV,                           \
                                                         void*                          QuantizedQKV,                  \
                                                         const int*                     position_ids,                  \
                                                         const T*                       qkv_bias,                      \
                                                         const int*                     padding_offset,                \
                                                         const int*                     cu_seqlens,                    \
                                                         const int*                     sequence_lengths,              \
                                                         const int                      batch_size,                    \
                                                         const int                      seq_len,                       \
                                                         const int                      token_num,                     \
                                                         const int                      head_num,                      \
                                                         const int                      head_num_kv,                   \
                                                         const int                      size_per_head,                 \
                                                         const RopeConfig               rope_config,                   \
                                                         const bool                     use_logn_attn,                 \
                                                         const float*                   scale,                         \
                                                         const int                      int8_mode,                     \
                                                         const bool                     use_paged_fmha,                \
                                                         const bool                     store_qkv,                     \
                                                         const bool                     store_q,                       \
                                                         const bool                     store_kv,                      \
                                                         const bool                     store_cache,                   \
                                                         const float2*                  cos_sin_cache,                 \
                                                         cudaStream_t                   stream)
INSTANTIATEADDFUSEDQKVBIASTRANSPOSEDECODEV1(float);
INSTANTIATEADDFUSEDQKVBIASTRANSPOSEDECODEV1(half);
#ifdef ENABLE_BF16
INSTANTIATEADDFUSEDQKVBIASTRANSPOSEDECODEV1(__nv_bfloat16);
#endif
#undef INSTANTIATEADDFUSEDQKVBIASTRANSPOSEDECODEV1

#define INSTANTIATEADDFUSEDQKVBIASTRANSPOSEDECODE(T)                                                                   \
    template void invokeAddFusedQKVBiasTransposeDecode(T*                             q_buf,                           \
                                                       T*                             k_buf,                           \
                                                       T*                             v_buf,                           \
                                                       PrefixPromptBatchWeightsParam* param,                           \
                                                       const int*                     input_lengths,                   \
                                                       T*                             QKV,                             \
                                                       void*                          QuantizedQKV,                    \
                                                       const int*                     position_ids,                    \
                                                       const T*                       qkv_bias,                        \
                                                       const int*                     padding_offset,                  \
                                                       const int*                     cu_seqlens,                      \
                                                       const int*                     sequence_lengths,                \
                                                       const int                      batch_size,                      \
                                                       const int                      seq_len,                         \
                                                       const int                      token_num,                       \
                                                       const int                      head_num,                        \
                                                       const int                      head_num_kv,                     \
                                                       const int                      size_per_head,                   \
                                                       const RopeConfig               rope_config,                     \
                                                       const bool                     use_logn_attn,                   \
                                                       const float*                   scale,                           \
                                                       const int                      int8_mode,                       \
                                                       const bool                     use_paged_fmha,                  \
                                                       const bool                     store_qkv,                       \
                                                       const bool                     store_q,                         \
                                                       const bool                     store_kv,                        \
                                                       const bool                     store_cache,                     \
                                                       const float2*                  cos_sin_cache,                   \
                                                       cudaStream_t                   stream)
INSTANTIATEADDFUSEDQKVBIASTRANSPOSEDECODE(float);
INSTANTIATEADDFUSEDQKVBIASTRANSPOSEDECODE(half);
#ifdef ENABLE_BF16
INSTANTIATEADDFUSEDQKVBIASTRANSPOSEDECODE(__nv_bfloat16);
#endif
#undef INSTANTIATEADDFUSEDQKVBIASTRANSPOSEDECODE

}  // namespace rtp_llm
