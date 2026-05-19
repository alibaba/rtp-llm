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

__device__ __forceinline__ int get_mrope_position_dim(const RopeConfig& rope_config, const int tidx) {
    const int rope_dim = rope_config.mrope_dim1 + rope_config.mrope_dim2 + rope_config.mrope_dim3;
    if (rope_dim <= 0) {
        return 0;
    }

    const int now_idx = tidx % rope_dim;
    if (rope_config.mrope_interleaved) {
        if (now_idx < rope_config.mrope_dim2 * 3 && now_idx % 3 == 1) {
            return 1;
        }
        if (now_idx < rope_config.mrope_dim3 * 3 && now_idx % 3 == 2) {
            return 2;
        }
        return 0;
    }

    if (now_idx >= rope_config.mrope_dim1 + rope_config.mrope_dim2) {
        return 2;
    }
    if (now_idx >= rope_config.mrope_dim1) {
        return 1;
    }
    return 0;
}

__device__ __forceinline__ int
get_rope_position_id(const RopeConfig& rope_config, const int* position_ids, const int token_idx, const int tidx) {
    if (!position_ids) {
        return -1;
    }
    if (rope_config.style == RopeStyle::Mrope) {
        const int now_dim = get_mrope_position_dim(rope_config, tidx);
        return position_ids[token_idx * rope_config.index_factor + now_dim];
    }
    return position_ids[token_idx * rope_config.index_factor];
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
    int       position_id = get_rope_position_id(rope_config, position_ids, token_idx, tidx);
    const int pre_len     = cu_seqlens[batch_idx];
    const int input_len   = cu_seqlens[batch_idx + 1] - pre_len;
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
            // Paged FP8: write quantized Q into QuantizedQKV (the FP8 Q buffer).
            // Non-paged FP8: write quantized Q into q_buf (in-place over BF16 data).
            // Must match ASM kernel convention.
            QuantizedVecType* quantized_q_ptr =
                USE_PAGED_FMHA ? reinterpret_ptr<QuantizedEltType, QuantizedVecType>(QuantizedQKV, dest_q_idx) :
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
        // FP8 Q output: when QuantizedQKV is provided, write FP8 Q into
        // QuantizedQKV (paged) or q_buf (non-paged), matching ASM kernel convention.
        if (QuantizedQKV != nullptr) {
            using QuantizedEltType = __hip_fp8_e4m3_fnuz;
            using QuantizedVecType = __hip_fp8x2_e4m3_fnuz;
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
            // FP8 KV cache uses vectorized layout (getKLocalIdx<FP8>/getVLocalIdx<FP8>)
            // with per-head scaling. INT8 KV cache uses the same BASE layout as
            // non-quantized cache but with int8 quantization — it must NOT enter
            // this branch. Guard on the actual FP8 type to avoid misrouting INT8.
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

                    const int inVBlockIdx =
                        kv_block_array.getVLocalIdx(dst_kv_seq_idx, head_idx, size_per_head, tidx * vec_size + vec_i);
                    v_cache[inVBlockIdx] = reinterpret_cast<T*>(&v)[vec_i];
                }
            }
        }
    }
}

// =====================================================================================
// v3 optimized kernel: vec=8 + 4 tokens/block + smem partner exchange + __hadd2.
// One block per token-chunk iterates over ALL heads inside, sharing cos/sin across
// heads (cos/sin only depends on token+rope_d, not on head). Grid: (token_chunks, 1).
// Hard-coded for the production hot path (BF16 + Tcache=BF16, packed K/V, paged FMHA
// Q layout, no prefix, RopeStyle::Base, store_q+store_kv (+optional store_cache),
// head_dim%8==0). Other configurations fall back to the original kernel below.
//
// V_VEC_LAYOUT: production paged-cache V layouts differ between the V1-path
// kernel (NonAsm invoker) and the non-v1 kernel (ASM invoker). The non-v1
// kernel calls getVLocalIdx<KvCacheDataType::BASE> producing layout
// [numHeads, mTokensPerBlock/vs, dimsPerHead, vs], while the V1 kernel calls
// the non-templated getVLocalIdx producing [numHeads, dimsPerHead, mTokensPerBlock].
// V3 must match its caller's layout or the FMHA reader will see garbage V.
// =====================================================================================
template<bool V_VEC_LAYOUT>
__global__ void add_fusedQKV_bias_transpose_prefill_v3_all_heads(__nv_bfloat16*       q_buf,
                                                                 __nv_bfloat16*       k_buf,
                                                                 __nv_bfloat16*       v_buf,
                                                                 const __nv_bfloat16* QKV,
                                                                 const __nv_bfloat16* __restrict qkv_bias,
                                                                 const int* __restrict padding_offset,
                                                                 const float2* __restrict cos_sin_cache,
                                                                 PrefixPromptBatchWeightsParam param,
                                                                 int                           token_num,
                                                                 int                           head_num,
                                                                 int                           head_num_kv,
                                                                 int                           head_dim,
                                                                 int                           seq_len,
                                                                 int                           rot_dim,
                                                                 float                         rope_base,
                                                                 float                         rope_scale,
                                                                 bool                          store_kv,
                                                                 bool                          store_cache) {
    constexpr int                   VEC   = 8;
    constexpr int                   K_TOK = 4;
    extern __shared__ __nv_bfloat16 smem[];  // [K_TOK][head_dim] BF16

    const int tok_local = threadIdx.y;
    const int token_idx = blockIdx.x * K_TOK + tok_local;
    // Tail-block tok_local values where token_idx >= token_num must NOT early-
    // return: the per-head loops below contain unconditional __syncthreads(),
    // and dropping a subset of threads from the block deadlocks the rest (HIP
    // requires every active thread to participate). Instead, mark inactive,
    // clamp index math to safe_token_idx so loads/index calcs stay in-range,
    // and skip only the global stores.
    const bool active         = (token_idx < token_num);
    const int  safe_token_idx = active ? token_idx : 0;

    const int tid = threadIdx.x;
    const int d   = tid * VEC;

    // Partial rotary: only [0, rot_dim) is rotated; [rot_dim, head_dim) is
    // passthrough (no RoPE, no partner exchange). Each thread owns VEC=8
    // consecutive dims; the V3 dispatch guard requires rot_dim % VEC == 0
    // so an 8-dim thread is fully in one region.
    const bool in_rope = (d < rot_dim);
    const int  half_r  = rot_dim / 2;
    const bool is_lo   = in_rope && (d < half_r);
    const int  d_part  = is_lo ? (d + half_r) : (d - half_r);
    const int  rope_d  = is_lo ? d : d_part;

    const int token_padding_offset = padding_offset ? padding_offset[safe_token_idx] : 0;
    const int tgt_token_idx        = safe_token_idx + token_padding_offset;
    const int batch_idx            = tgt_token_idx / seq_len;
    const int dst_kv_seq_idx       = tgt_token_idx % seq_len;
    // V3 dispatch guard enforces prefix==0, so batch-local seq_idx is the
    // absolute RoPE position — no external position_ids needed.
    const int pos_id = dst_kv_seq_idx;

    const int n       = head_num * head_dim;
    const int kv_n    = head_num_kv * head_dim;
    const int hidden  = n + 2 * kv_n;
    const int row_off = safe_token_idx * hidden;

    auto           load8    = [](const __nv_bfloat16* p) { return *reinterpret_cast<const int4*>(p); };
    auto           store8   = [](__nv_bfloat16* p, int4 v) { *reinterpret_cast<int4*>(p) = v; };
    __nv_bfloat16* smem_row = smem + tok_local * head_dim;

    // ---- inv_freq (per thread, shared across all heads) ----
    // Only rotary-region threads need cos/sin; passthrough threads skip.
    // Match V1 baseline precision (rotary_position_embedding.h): use libdevice
    // powf (not __powf) and double-precision sincos on ROCm. Fast intrinsics
    // (__powf ~6 ULP, __sincosf ~2-3 ULP) drift enough across many layers/heads
    // to flip greedy decoding at marginal logit positions (observed regression
    // in Eagle speculative decoding on Qwen2-14B).
    float inv_freq_lo[VEC / 2], inv_freq_hi[VEC / 2];
    if (in_rope && !cos_sin_cache) {
        const float inv_rot_dim = 1.0f / float(rot_dim);
#pragma unroll
        for (int i = 0; i < VEC / 2; ++i) {
            inv_freq_lo[i] = powf(rope_base, -float(2 * (rope_d + 2 * i)) * inv_rot_dim);
            inv_freq_hi[i] = powf(rope_base, -float(2 * (rope_d + 2 * i + 1)) * inv_rot_dim);
        }
    }

    // ---- cs_lo/cs_hi (per token, shared across all heads of this block) ----
    float2 cs_lo[VEC / 2], cs_hi[VEC / 2];
    if (in_rope) {
#pragma unroll
        for (int i = 0; i < VEC / 2; ++i) {
            if (cos_sin_cache) {
                cs_lo[i] = cos_sin_cache[pos_id * half_r + rope_d + 2 * i];
                cs_hi[i] = cos_sin_cache[pos_id * half_r + rope_d + 2 * i + 1];
            } else {
                // Compute angle in float, but evaluate sincos in double to
                // match V1 (rotary_position_embedding.h:340 declares
                // `double sin_i, cos_i;` for ROCm before calling sincos).
                float  angle0 = float(pos_id) * inv_freq_lo[i] / rope_scale;
                float  angle1 = float(pos_id) * inv_freq_hi[i] / rope_scale;
                double s0, c0, s1, c1;
                sincos((double)angle0, &s0, &c0);
                sincos((double)angle1, &s1, &c1);
                cs_lo[i].x = (float)c0;
                cs_lo[i].y = (float)s0;
                cs_hi[i].x = (float)c1;
                cs_hi[i].y = (float)s1;
            }
        }
    }

    // ---- Process all Q heads ----
    for (int h = 0; h < head_num; ++h) {
        const int q_off  = h * head_dim;
        int4      q_pack = load8(&QKV[row_off + q_off + d]);
        auto*     q2     = reinterpret_cast<__nv_bfloat162*>(&q_pack);
        if (qkv_bias) {
            int4  qb_pack = load8(&qkv_bias[q_off + d]);
            auto* qb2     = reinterpret_cast<__nv_bfloat162*>(&qb_pack);
#pragma unroll
            for (int i = 0; i < VEC / 2; ++i)
                q2[i] = __hadd2(q2[i], qb2[i]);
        }
        // Partner exchange via smem. Only rotary-region threads (in_rope=true)
        // need to read their partner; passthrough threads still write so the
        // sync below has uniform participation across the block.
        store8(&smem_row[d], q_pack);
        __syncthreads();
        int4 q_out_pack = q_pack;  // passthrough default
        if (in_rope) {
            int4  q_partner_pack = load8(&smem_row[d_part]);
            auto* qp2            = reinterpret_cast<__nv_bfloat162*>(&q_partner_pack);
            auto* qo2            = reinterpret_cast<__nv_bfloat162*>(&q_out_pack);
#pragma unroll
            for (int i = 0; i < VEC / 2; ++i) {
                float2 cs0 = cs_lo[i], cs1 = cs_hi[i];
                float  s0 = __bfloat162float(__low2bfloat16(q2[i]));
                float  s1 = __bfloat162float(__high2bfloat16(q2[i]));
                float  p0 = __bfloat162float(__low2bfloat16(qp2[i]));
                float  p1 = __bfloat162float(__high2bfloat16(qp2[i]));
                float  o0 = is_lo ? (s0 * cs0.x - p0 * cs0.y) : (s0 * cs0.x + p0 * cs0.y);
                float  o1 = is_lo ? (s1 * cs1.x - p1 * cs1.y) : (s1 * cs1.x + p1 * cs1.y);
                qo2[i]    = __floats2bfloat162_rn(o0, o1);
            }
        }
        if (active) {
            store8(&q_buf[(size_t)token_idx * n + q_off + d], q_out_pack);
        }
        __syncthreads();  // before next head reuses smem
    }

    // ---- Process all K, V heads ----
    KVBlockArray   kv_block_array;
    __nv_bfloat16 *k_cache = nullptr, *v_cache = nullptr;
    if (store_cache && active) {
        kv_block_array = param.kv_block_array;
        k_cache        = reinterpret_cast<__nv_bfloat16*>(kv_block_array.getKBlockPtr(batch_idx, dst_kv_seq_idx));
        v_cache        = reinterpret_cast<__nv_bfloat16*>(kv_block_array.getVBlockPtr(batch_idx, dst_kv_seq_idx));
    }
    for (int h = 0; h < head_num_kv; ++h) {
        const int k_off = n + h * head_dim;
        const int v_off = n + kv_n + h * head_dim;

        // K
        int4  k_pack = load8(&QKV[row_off + k_off + d]);
        auto* k2     = reinterpret_cast<__nv_bfloat162*>(&k_pack);
        if (qkv_bias) {
            int4  kb_pack = load8(&qkv_bias[k_off + d]);
            auto* kb2     = reinterpret_cast<__nv_bfloat162*>(&kb_pack);
#pragma unroll
            for (int i = 0; i < VEC / 2; ++i)
                k2[i] = __hadd2(k2[i], kb2[i]);
        }
        store8(&smem_row[d], k_pack);
        __syncthreads();
        int4 k_out_pack = k_pack;  // passthrough default
        if (in_rope) {
            int4  k_partner_pack = load8(&smem_row[d_part]);
            auto* kp2            = reinterpret_cast<__nv_bfloat162*>(&k_partner_pack);
            auto* ko2            = reinterpret_cast<__nv_bfloat162*>(&k_out_pack);
#pragma unroll
            for (int i = 0; i < VEC / 2; ++i) {
                float2 cs0 = cs_lo[i], cs1 = cs_hi[i];
                float  s0 = __bfloat162float(__low2bfloat16(k2[i]));
                float  s1 = __bfloat162float(__high2bfloat16(k2[i]));
                float  p0 = __bfloat162float(__low2bfloat16(kp2[i]));
                float  p1 = __bfloat162float(__high2bfloat16(kp2[i]));
                float  o0 = is_lo ? (s0 * cs0.x - p0 * cs0.y) : (s0 * cs0.x + p0 * cs0.y);
                float  o1 = is_lo ? (s1 * cs1.x - p1 * cs1.y) : (s1 * cs1.x + p1 * cs1.y);
                ko2[i]    = __floats2bfloat162_rn(o0, o1);
            }
        }
        if (active && store_kv) {
            store8(&k_buf[(size_t)token_idx * kv_n + h * head_dim + d], k_out_pack);
        }
        __syncthreads();  // before next iteration uses smem

        // V (no RoPE)
        int4  v_pack = load8(&QKV[row_off + v_off + d]);
        auto* v2     = reinterpret_cast<__nv_bfloat162*>(&v_pack);
        if (qkv_bias) {
            int4  vb_pack = load8(&qkv_bias[v_off + d]);
            auto* vb2     = reinterpret_cast<__nv_bfloat162*>(&vb_pack);
#pragma unroll
            for (int i = 0; i < VEC / 2; ++i)
                v2[i] = __hadd2(v2[i], vb2[i]);
        }
        if (active && store_kv) {
            store8(&v_buf[(size_t)token_idx * kv_n + h * head_dim + d], v_pack);
        }

        // Cache write
        if (store_cache && active) {
            const int inK = kv_block_array.getKLocalIdx<KvCacheDataType::BASE>(dst_kv_seq_idx, h, head_dim, d);
            *reinterpret_cast<int4*>(&k_cache[inK]) = k_out_pack;
            const __nv_bfloat16* v_src              = reinterpret_cast<const __nv_bfloat16*>(&v_pack);
#pragma unroll
            for (int vi = 0; vi < VEC; ++vi) {
                const int inV =
                    V_VEC_LAYOUT ?
                        kv_block_array.getVLocalIdx<KvCacheDataType::BASE>(dst_kv_seq_idx, h, head_dim, d + vi) :
                        kv_block_array.getVLocalIdx(dst_kv_seq_idx, h, head_dim, d + vi);
                v_cache[inV] = v_src[vi];
            }
        }
    }
}

// Compile-time dispatcher: only BF16 specialization actually launches v3.
template<typename T>
struct V3OptKernelDispatch {
    static bool try_launch(T*,
                           T*,
                           T*,
                           const T*,
                           const T*,
                           const int*,
                           const float2*,
                           PrefixPromptBatchWeightsParam&,
                           int,
                           int,
                           int,
                           int,
                           int,
                           int,
                           float,
                           float,
                           bool,
                           bool,
                           bool,
                           cudaStream_t) {
        return false;
    }
};
template<>
struct V3OptKernelDispatch<__nv_bfloat16> {
    static bool try_launch(__nv_bfloat16*                 q_buf,
                           __nv_bfloat16*                 k_buf,
                           __nv_bfloat16*                 v_buf,
                           const __nv_bfloat16*           QKV,
                           const __nv_bfloat16*           qkv_bias,
                           const int*                     padding_offset,
                           const float2*                  cos_sin_cache,
                           PrefixPromptBatchWeightsParam& param,
                           int                            token_num,
                           int                            head_num,
                           int                            head_num_kv,
                           int                            head_dim,
                           int                            seq_len,
                           int                            rot_dim,
                           float                          rope_base,
                           float                          rope_scale,
                           bool                           store_kv,
                           bool                           store_cache,
                           bool                           v_vec_layout,
                           cudaStream_t                   stream) {
        constexpr int VEC = 8, K_TOK = 4;
        if (head_dim % VEC != 0)
            return false;
        if (head_dim % 2 != 0)
            return false;
        // Partial rotary requires rot_dim aligned to VEC*2 so each thread is
        // either fully in the rotary region or fully passthrough, and the
        // partner index (rope_d ± half_r) stays inside the rotary region.
        if (rot_dim % (VEC * 2) != 0)
            return false;
        if (rot_dim > head_dim)
            return false;
        dim3 block(head_dim / VEC, K_TOK);
        // v3: one block per token-chunk, iterates ALL heads inside
        dim3   grid((token_num + K_TOK - 1) / K_TOK, 1);
        size_t smem = K_TOK * head_dim * sizeof(__nv_bfloat16);
        if (v_vec_layout) {
            add_fusedQKV_bias_transpose_prefill_v3_all_heads<true><<<grid, block, smem, stream>>>(q_buf,
                                                                                                  k_buf,
                                                                                                  v_buf,
                                                                                                  QKV,
                                                                                                  qkv_bias,
                                                                                                  padding_offset,
                                                                                                  cos_sin_cache,
                                                                                                  param,
                                                                                                  token_num,
                                                                                                  head_num,
                                                                                                  head_num_kv,
                                                                                                  head_dim,
                                                                                                  seq_len,
                                                                                                  rot_dim,
                                                                                                  rope_base,
                                                                                                  rope_scale,
                                                                                                  store_kv,
                                                                                                  store_cache);
        } else {
            add_fusedQKV_bias_transpose_prefill_v3_all_heads<false><<<grid, block, smem, stream>>>(q_buf,
                                                                                                   k_buf,
                                                                                                   v_buf,
                                                                                                   QKV,
                                                                                                   qkv_bias,
                                                                                                   padding_offset,
                                                                                                   cos_sin_cache,
                                                                                                   param,
                                                                                                   token_num,
                                                                                                   head_num,
                                                                                                   head_num_kv,
                                                                                                   head_dim,
                                                                                                   seq_len,
                                                                                                   rot_dim,
                                                                                                   rope_base,
                                                                                                   rope_scale,
                                                                                                   store_kv,
                                                                                                   store_cache);
        }
        return true;
    }
};

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
    auto& param = *param_ptr;

    // ---- v3 fast path (default ON) ----
    // Hand-tuned BF16 kernel: vec=8 + 4 tokens/block + smem partner exchange + cos/sin
    // reuse across heads. ~5.77x kernel speedup (1087 -> 188 us/call) on
    // Qwen3.5-9B prefill 15k TP=2 (verified on both ASM and NonAsm dispatch paths).
    //
    // Activation requires the production hot-path config: bf16 + paged_fmha +
    // no prefix + store_q + store_cache + no store_qkv + no store_kv +
    // RopeStyle::Base + Tcache=BASE + no FP8 quant + no logn_attn. Anything else
    // falls through to the original kernel below.
    //
    // store_kv is disallowed here because the in-tree fallback writes packed K/V
    // in BHSD ([batch, head_kv, seq, dim]) layout, while V3 uses THD ([token,
    // head_kv, dim]). Production NonAsm uses store_kv=false (K/V only flow into
    // paged cache), so V3 fires by writing only to the paged cache.
    if (use_paged_fmha && param.max_prefix_prompt_length == 0 && store_q && !store_qkv && !store_kv && store_cache
        && rope_config.style == RopeStyle::Base
        && rope_config.dim <= size_per_head  // partial rotary supported (rot_dim<=head_dim)
        && rope_config.scale == 1.0f         // V3 scale path lacks precision tests; fall back to V1
        && !use_logn_attn && QuantizedQKV == nullptr
        && qkv_bias == nullptr  // bias path is implemented but untested; fall back
        && param.kv_block_array.cache_type == KvCacheDataType::BASE) {
        // V1 invoker (NonAsm path) writes V cache in non-templated layout
        // [numHeads, dimsPerHead, mTokensPerBlock]. Pass v_vec_layout=false so
        // V3's V cache write matches what the NonAsm CK FMHA reader expects.
        if (V3OptKernelDispatch<T>::try_launch(q_buf,
                                               k_buf,
                                               v_buf,
                                               QKV,
                                               qkv_bias,
                                               padding_offset,
                                               cos_sin_cache,
                                               param,
                                               token_num,
                                               head_num,
                                               head_num_kv,
                                               size_per_head,
                                               seq_len,
                                               rope_config.dim,
                                               rope_config.base,
                                               rope_config.scale,
                                               store_kv,
                                               store_cache,
                                               /*v_vec_layout=*/false,
                                               stream)) {
            return;
        }
    }
    // ---- end v3 fast path ----

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
    int       position_id = get_rope_position_id(rope_config, position_ids, token_idx, tidx);
    const int pre_len     = cu_seqlens[batch_idx];
    const int input_len   = cu_seqlens[batch_idx + 1] - pre_len;
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
    auto& param = *param_ptr;

    // ---- v3 fast path (default ON, ASM-side invoker) ----
    // Mirrors the V3 dispatch in invokeAddFusedQKVBiasTransposePrefillV1. Required
    // for production setups where USE_ASM_PA=1 routes through this invoker, which
    // otherwise launches the original add_fusedQKV_bias_transpose_prefill_kernel —
    // the hottest kernel in BF16 prefill.
    //
    // Activation requires the production hot-path config: bf16 + paged_fmha +
    // !pad_query (V3 writes packed-token Q layout) + no prefix + store_q +
    // store_cache + no store_qkv + no store_kv + RopeStyle::Base + Tcache=BASE
    // + no FP8 quant + no logn_attn. Everything else falls through to the
    // original kernel below.
    //
    // store_kv is excluded because the in-tree ASM kernel writes packed K/V in
    // BHSD layout while V3 uses THD; in production store_kv=false anyway, so
    // V3 only writes to the paged cache.
    if (use_paged_fmha && !pad_query && param.max_prefix_prompt_length == 0 && store_q && !store_qkv && !store_kv
        && store_cache && rope_config.style == RopeStyle::Base
        && rope_config.dim <= size_per_head  // partial rotary supported
        && rope_config.scale == 1.0f && !use_logn_attn && QuantizedQKV == nullptr
        && qkv_bias == nullptr  // bias path is implemented but untested; fall back
        && param.kv_block_array.cache_type == KvCacheDataType::BASE) {
        // ASM invoker writes V cache via getVLocalIdx<KvCacheDataType::BASE>
        // (vectorized [numHeads, mTokensPerBlock/vs, dimsPerHead, vs] layout).
        // Pass v_vec_layout=true so V3 produces the layout the ASM-flash FMHA
        // reader expects — using the V1-style flat layout here corrupts V cache
        // and yields garbage attention output (root cause of the precision bug).
        if (V3OptKernelDispatch<T>::try_launch(q_buf,
                                               k_buf,
                                               v_buf,
                                               QKV,
                                               qkv_bias,
                                               padding_offset,
                                               cos_sin_cache,
                                               param,
                                               token_num,
                                               head_num,
                                               head_num_kv,
                                               size_per_head,
                                               seq_len,
                                               rope_config.dim,
                                               rope_config.base,
                                               rope_config.scale,
                                               store_kv,
                                               store_cache,
                                               /*v_vec_layout=*/true,
                                               stream)) {
            return;
        }
    }
    // ---- end v3 fast path ----

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
    int        position_id    = get_rope_position_id(rope_config, position_ids, token_idx, tidx);

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
    int        position_id    = get_rope_position_id(rope_config, position_ids, token_idx, tidx);

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
        // Always write BF16 Q into q_buf.
        *reinterpret_cast<Vec_t*>(&q_buf[dest_q_idx]) = q;
        // When FP8 Q buffer is provided, also write FP8 Q into QuantizedQKV.
        if (QuantizedQKV != nullptr) {
            QuantizedVecType* quantized_q_ptr =
                reinterpret_ptr<QuantizedEltType, QuantizedVecType>(QuantizedQKV, dest_q_idx);
            convert_to_fp8(quantized_q_ptr, q);
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
