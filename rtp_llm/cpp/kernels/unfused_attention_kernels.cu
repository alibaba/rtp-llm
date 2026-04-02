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
#include "rtp_llm/cpp/kernels/kv_cache/kv_cache_utils.h"
#include "rtp_llm/cpp/cuda/reduce_kernel_utils.cuh"
#include "rtp_llm/cpp/kernels/rotary_position_embedding.h"
#include "rtp_llm/cpp/kernels/unfused_attention_kernels.h"
#include "rtp_llm/cpp/cuda/cuda_type_utils.cuh"
#if USING_CUDA
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#endif
#if USING_ROCM
#include "rtp_llm/cpp/rocm/cuda_shims.h"
#endif
#include <cstdlib>

namespace rtp_llm {

__device__ float convert_to_float(int val) {
    return float(val);
}

template<typename T>
__global__ void debug_kernel2(T* data, int start_row, int start_col, int m, int n, int row_len, int info_id) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("debug_kernel2 start: %d\n", info_id);
        for (int i = start_row; i < start_row + m; i++) {
            for (int j = start_col; j < start_col + n; j++) {
                int   index = i * row_len + j;
                float value = convert_to_float(data[index]);
                printf("%f ", value);
            }
            printf("\n");
        }
        printf("debug_kernel2 end: %d\n", info_id);
    }
}

template<typename T>
void invoke_debug_kernel2(
    T* data, int start_row, int start_col, int m, int n, int row_len, int info_id, cudaStream_t stream) {
    debug_kernel2<<<1, 1, 0, stream>>>(data, start_row, start_col, m, n, row_len, info_id);
}

#define INSTANTIATEDEBUGKERNEL2(T)                                                                                     \
    template void invoke_debug_kernel2(                                                                                \
        T* data, int start_row, int start_col, int m, int n, int row_len, int info_id, cudaStream_t stream)
INSTANTIATEDEBUGKERNEL2(float);
INSTANTIATEDEBUGKERNEL2(half);
INSTANTIATEDEBUGKERNEL2(int);
#ifdef ENABLE_BF16
INSTANTIATEDEBUGKERNEL2(__nv_bfloat16);
#endif
#undef INSTANTIATEDEBUGKERNEL2

// Bandwidth-bound kernel by reading cos/sin coefficients from global memory (pre-computed and saved as weights).

#if USING_CUDA
template<typename T,
         typename Tcache,
         bool      PREFIX_PROMPT,
         bool      USE_PAGED_FMHA,
         RopeStyle ROPE_STYLE,
         int       HEAD_Q_BLOCK_NUM,
         int       HEAD_K_BLOCK_NUM,
         int       HEAD_V_BLOCK_NUM>
__global__ void add_fusedQKV_bias_transpose_non_int8_with_rope_cache_kernel(T* q_no_transpose_buf,
                                                                            T* q_buf,
                                                                            T* k_buf,
                                                                            T* v_buf,
                                                                            PrefixPromptBatchWeightsParam param,
                                                                            T*                            QKV,
                                                                            void*                         QuantizedQKV,
                                                                            const int*                    position_ids,
                                                                            const T* __restrict qkv_bias,
                                                                            const int*   padding_offset,
                                                                            const int*   cu_seqlens,
                                                                            const float* rope_cache,
                                                                            const int    threads_per_token,
                                                                            const int    tokens_per_block,
                                                                            const int    batch_size,
                                                                            const int    seq_len,
                                                                            const int    head_num,
                                                                            const int    head_num_kv,
                                                                            const int    size_per_head,
                                                                            RopeConfig   rope_config,
                                                                            const bool   use_logn_attn,
                                                                            bool         store_qkv,
                                                                            bool         store_q_no_transpose,
                                                                            bool         store_q,
                                                                            bool         store_kv,
                                                                            bool         store_cache) {
    // This kernel add bias to QKV, which has shape [batch_size, seq_len, 3, head_num, size_per_head], and
    // QKV split to 3 split buffer q, k, v and transpose them to [batch_size, head_num, seq_len, size_per_head].
    // For q and k, also apply the rotary embedding.

    // When we pass prefix prompt, this kernel also concatenate the prefix prompt and key/value along
    // seq_len dimension like [prompt, key/value].
    // So, the final shape of q is same ([batch_size, head_num, seq_len, size_per_head]), but
    // the shapes of key and values become [batch_size, head_num, max_prefix_prompt_length + seq_len, size_per_head].

    // NOTE: QKV src shape (batch_size, seq_len, 3, head_num, size_per_head)
    //  QKV dst shape (3, batch_size, head_num, seq_len, size_per_head)
    static constexpr bool ENABLE_8BITS_CACHE = sizeof(Tcache) == 1;

#ifdef ENABLE_FP8
    // Quantized output only supports fp8 currently.
    using QuantizedEltType = __nv_fp8_e4m3;
    using QuantizedVecType = typename Vec_t2<T>::QuantizedType;
#endif
    constexpr int vec_size = Vec_t2<T>::size;
    using vec_t2           = typename Vec_t2<T>::Type;

    const int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) {
        return;
    }

    const int tidx    = threadIdx.x;
    const int lane_id = tidx % threads_per_token;
    if (lane_id * vec_size >= size_per_head) {
        return;
    }

    const int pre_len         = cu_seqlens[batch_idx];
    const int curr_seq_len    = cu_seqlens[batch_idx + 1] - pre_len;
    const int block_token_idx = tidx / threads_per_token;
    const int curr_token_idx  = blockIdx.y * tokens_per_block + block_token_idx;
    if (curr_token_idx >= curr_seq_len) {
        return;
    }

    const int bidz       = blockIdx.z;
    const int max_q_bidz = head_num / HEAD_Q_BLOCK_NUM;
    const int max_k_bidz = max_q_bidz + head_num_kv / HEAD_K_BLOCK_NUM;
    const int max_v_bidz = max_k_bidz + head_num_kv / HEAD_V_BLOCK_NUM;
    if (bidz >= max_v_bidz) {
        return;
    }

    const int token_idx            = pre_len + curr_token_idx;
    const int token_padding_offset = padding_offset == nullptr ? 0 : padding_offset[token_idx];
    const int tgt_token_idx        = token_idx + token_padding_offset;
    const int seq_idx              = tgt_token_idx % seq_len;

    const int total_seq_len        = param.max_prefix_prompt_length + seq_len;
    const int prefix_prompt_length = PREFIX_PROMPT ? param.d_prefix_prompt_lengths[batch_idx] : 0;
    const int n                    = head_num * size_per_head;
    const int kv_n                 = head_num_kv * size_per_head;  // MQA

    // NOTE: q has seq len excluding prefix prompt
    // src QKV: [batch, time, 3, head, hidden]
    int     position_id = position_ids ? position_ids[token_idx * rope_config.index_factor] :
                                         (PREFIX_PROMPT && param.count_length ? seq_idx + prefix_prompt_length : seq_idx);
    Float8_ coef1;
    Float8_ coef2;
    if (bidz < max_k_bidz) {
        const int idx                        = position_id * rope_config.dim + lane_id * 16;
        *reinterpret_cast<float4*>(&coef1.x) = *(reinterpret_cast<const float4*>(&rope_cache[idx]));
        *reinterpret_cast<float4*>(&coef1.z) = *(reinterpret_cast<const float4*>(&rope_cache[idx + 4]));
        *reinterpret_cast<float4*>(&coef2.x) = *(reinterpret_cast<const float4*>(&rope_cache[idx + 8]));
        *reinterpret_cast<float4*>(&coef2.z) = *(reinterpret_cast<const float4*>(&rope_cache[idx + 12]));
    }

    if (bidz < max_q_bidz) {
        vec_t2 q[2];
        vec_t2 q_bias[2];
        vec_t2 q_permuted[2];
        vec_t2 q_permuted_bias[2];
        int    q_load_idx  = 0;
        int    q_store_idx = 0;
        int    q_idx_off   = 1;

        int    head_idx       = bidz * HEAD_Q_BLOCK_NUM;
        size_t hidden_idx     = head_idx * size_per_head + lane_id * vec_size;
        size_t src_q_idx      = token_idx * (n + 2 * kv_n) + hidden_idx;
        size_t src_q_last_idx = 0;

        size_t hidden_permuted_idx     = hidden_idx + rope_config.dim / 2;
        size_t src_q_permuted_idx      = src_q_idx + rope_config.dim / 2;
        size_t src_q_permuted_last_idx = 0;

        size_t dest_qkv_idx = 0;
        if constexpr (USE_PAGED_FMHA) {
            dest_qkv_idx =
                (pre_len + seq_idx) * size_per_head * head_num + head_idx * size_per_head + lane_id * vec_size;
        } else {
            dest_qkv_idx = batch_idx * size_per_head * seq_len * head_num + head_idx * size_per_head * seq_len
                           + seq_idx * size_per_head + lane_id * vec_size;
        }
        size_t dest_qkv_permuted_idx = dest_qkv_idx + rope_config.dim / 2;

        size_t dest_q_no_transpose_idx =
            (pre_len + seq_idx) * head_num * size_per_head + head_idx * size_per_head + lane_id * vec_size;
        size_t dest_q_permuted_no_transpose_idx = dest_q_no_transpose_idx + rope_config.dim / 2;

        size_t dest_q_idx = 0;
        if constexpr (USE_PAGED_FMHA) {
            dest_q_idx = (pre_len + seq_idx) * size_per_head * head_num + head_idx * size_per_head + lane_id * vec_size;
        } else {
            dest_q_idx = batch_idx * size_per_head * seq_len * head_num + head_idx * size_per_head * seq_len
                         + seq_idx * size_per_head + lane_id * vec_size;
        }
        size_t dest_q_permuted_idx = dest_q_idx + rope_config.dim / 2;

        *reinterpret_cast<int4*>(&q[q_load_idx])          = *reinterpret_cast<int4*>(&QKV[src_q_idx]);
        *reinterpret_cast<int4*>(&q_permuted[q_load_idx]) = *reinterpret_cast<int4*>(&QKV[src_q_permuted_idx]);

        if (qkv_bias) {
            *reinterpret_cast<int4*>(&q_bias[q_load_idx]) = *reinterpret_cast<const int4*>(&qkv_bias[hidden_idx]);
            *reinterpret_cast<int4*>(&q_permuted_bias[q_load_idx]) =
                *reinterpret_cast<const int4*>(&qkv_bias[hidden_permuted_idx]);
            q[q_load_idx]          = add(q[q_load_idx], q_bias[q_load_idx]);
            q_permuted[q_load_idx] = add(q_permuted[q_load_idx], q_permuted_bias[q_load_idx]);
        }

#pragma unroll
        for (int h = 1; h < HEAD_Q_BLOCK_NUM; ++h) {
            q_load_idx ^= q_idx_off;

            ++head_idx;
            hidden_idx += size_per_head;
            src_q_last_idx = src_q_idx;
            src_q_idx += size_per_head;
            hidden_permuted_idx += size_per_head;
            src_q_permuted_last_idx = src_q_permuted_idx;
            src_q_permuted_idx += size_per_head;

            *reinterpret_cast<int4*>(&q[q_load_idx])          = *reinterpret_cast<int4*>(&QKV[src_q_idx]);
            *reinterpret_cast<int4*>(&q_permuted[q_load_idx]) = *reinterpret_cast<int4*>(&QKV[src_q_permuted_idx]);

            if (qkv_bias) {
                *reinterpret_cast<int4*>(&q_bias[q_load_idx]) = *reinterpret_cast<const int4*>(&qkv_bias[hidden_idx]);
                *reinterpret_cast<int4*>(&q_permuted_bias[q_load_idx]) =
                    *reinterpret_cast<const int4*>(&qkv_bias[hidden_permuted_idx]);
                q[q_load_idx]          = add(q[q_load_idx], q_bias[q_load_idx]);
                q_permuted[q_load_idx] = add(q_permuted[q_load_idx], q_permuted_bias[q_load_idx]);
            }

            apply_rope_with_cache<vec_t2, T, Float8_, ROPE_STYLE>(
                q[q_store_idx], q_permuted[q_store_idx], coef1, coef2);

            if (use_logn_attn) {
                logn_attention(q[q_store_idx], seq_idx, rope_config.max_pos);
                logn_attention(q_permuted[q_store_idx], seq_idx, rope_config.max_pos);
            }

            if (store_qkv) {
                *reinterpret_cast<int4*>(&QKV[src_q_last_idx]) = *reinterpret_cast<int4*>(&q[q_store_idx]);
                *reinterpret_cast<int4*>(&QKV[src_q_permuted_last_idx]) =
                    *reinterpret_cast<int4*>(&q_permuted[q_store_idx]);
#ifdef ENABLE_FP8
                if (QuantizedQKV != nullptr) {
                    *reinterpret_cast<int4*>(&q_buf[dest_qkv_idx]) = *reinterpret_cast<int4*>(&q[q_store_idx]);
                    *reinterpret_cast<int4*>(&q_buf[dest_qkv_permuted_idx]) =
                        *reinterpret_cast<int4*>(&q_permuted[q_store_idx]);
                    QuantizedVecType* quantized_q_ptr =
                        USE_PAGED_FMHA ?
                            reinterpret_ptr<QuantizedEltType, QuantizedVecType>(q_buf, dest_qkv_idx) :
                            reinterpret_ptr<QuantizedEltType, QuantizedVecType>(QuantizedQKV, src_q_last_idx);
                    QuantizedVecType* quantized_q_permuted_ptr =
                        USE_PAGED_FMHA ?
                            reinterpret_ptr<QuantizedEltType, QuantizedVecType>(q_buf, dest_qkv_permuted_idx) :
                            reinterpret_ptr<QuantizedEltType, QuantizedVecType>(QuantizedQKV, src_q_permuted_last_idx);
                    convert_to_fp8(quantized_q_ptr, q[q_store_idx]);
                    convert_to_fp8(quantized_q_permuted_ptr, q_permuted[q_store_idx]);

                    if constexpr (USE_PAGED_FMHA) {
                        dest_qkv_idx += size_per_head;
                        dest_qkv_permuted_idx += size_per_head;
                    } else {
                        dest_qkv_idx += (size_per_head * seq_len);
                        dest_qkv_permuted_idx += (size_per_head * seq_len);
                    }
                }
#endif
            }

            if (store_q_no_transpose) {
                *reinterpret_cast<int4*>(&q_no_transpose_buf[dest_q_no_transpose_idx]) =
                    *reinterpret_cast<int4*>(&q[q_store_idx]);
                *reinterpret_cast<int4*>(&q_no_transpose_buf[dest_q_permuted_no_transpose_idx]) =
                    *reinterpret_cast<int4*>(&q_permuted[q_store_idx]);
                dest_q_no_transpose_idx += size_per_head;
                dest_q_permuted_no_transpose_idx += size_per_head;
            }

            if (store_q) {
#ifdef ENABLE_FP8
                if (QuantizedQKV != nullptr) {
                    // fp8 paged fmha
                    QuantizedVecType* quantized_q_ptr =
                        USE_PAGED_FMHA ?
                            reinterpret_ptr<QuantizedEltType, QuantizedVecType>(q_buf, dest_q_idx) :
                            reinterpret_ptr<QuantizedEltType, QuantizedVecType>(QuantizedQKV, src_q_last_idx);
                    QuantizedVecType* quantized_q_permuted_ptr =
                        USE_PAGED_FMHA ?
                            reinterpret_ptr<QuantizedEltType, QuantizedVecType>(q_buf, dest_q_permuted_idx) :
                            reinterpret_ptr<QuantizedEltType, QuantizedVecType>(QuantizedQKV, src_q_permuted_last_idx);
                    convert_to_fp8(quantized_q_ptr, q[q_store_idx]);
                    convert_to_fp8(quantized_q_permuted_ptr, q_permuted[q_store_idx]);
                } else {
                    // paged fmha
                    *reinterpret_cast<int4*>(&q_buf[dest_q_idx]) = *reinterpret_cast<int4*>(&q[q_store_idx]);
                    *reinterpret_cast<int4*>(&q_buf[dest_q_permuted_idx]) =
                        *reinterpret_cast<int4*>(&q_permuted[q_store_idx]);
                }
#else
                // paged fmha
                *reinterpret_cast<int4*>(&q_buf[dest_q_idx]) = *reinterpret_cast<int4*>(&q[q_store_idx]);
                *reinterpret_cast<int4*>(&q_buf[dest_q_permuted_idx]) =
                    *reinterpret_cast<int4*>(&q_permuted[q_store_idx]);
#endif

                if constexpr (USE_PAGED_FMHA) {
                    dest_q_idx += size_per_head;
                    dest_q_permuted_idx += size_per_head;
                } else {
                    dest_q_idx += (size_per_head * seq_len);
                    dest_q_permuted_idx += (size_per_head * seq_len);
                }
            }

            q_store_idx ^= q_idx_off;
        }

        apply_rope_with_cache<vec_t2, T, Float8_, ROPE_STYLE>(q[q_store_idx], q_permuted[q_store_idx], coef1, coef2);

        if (use_logn_attn) {
            logn_attention(q[q_store_idx], seq_idx, rope_config.max_pos);
            logn_attention(q_permuted[q_store_idx], seq_idx, rope_config.max_pos);
        }

        if (store_qkv) {
            *reinterpret_cast<int4*>(&QKV[src_q_idx])          = *reinterpret_cast<int4*>(&q[q_store_idx]);
            *reinterpret_cast<int4*>(&QKV[src_q_permuted_idx]) = *reinterpret_cast<int4*>(&q_permuted[q_store_idx]);
#ifdef ENABLE_FP8
            if (QuantizedQKV != nullptr) {
                *reinterpret_cast<int4*>(&q_buf[dest_qkv_idx]) = *reinterpret_cast<int4*>(&q[q_store_idx]);
                *reinterpret_cast<int4*>(&q_buf[dest_qkv_permuted_idx]) =
                    *reinterpret_cast<int4*>(&q_permuted[q_store_idx]);
                QuantizedVecType* quantized_q_ptr =
                    USE_PAGED_FMHA ? reinterpret_ptr<QuantizedEltType, QuantizedVecType>(q_buf, dest_qkv_idx) :
                                     reinterpret_ptr<QuantizedEltType, QuantizedVecType>(QuantizedQKV, src_q_idx);
                QuantizedVecType* quantized_q_permuted_ptr =
                    USE_PAGED_FMHA ?
                        reinterpret_ptr<QuantizedEltType, QuantizedVecType>(q_buf, dest_qkv_permuted_idx) :
                        reinterpret_ptr<QuantizedEltType, QuantizedVecType>(QuantizedQKV, src_q_permuted_idx);
                convert_to_fp8(quantized_q_ptr, q[q_store_idx]);
                convert_to_fp8(quantized_q_permuted_ptr, q_permuted[q_store_idx]);
            }
#endif
        }

        if (store_q_no_transpose) {
            *reinterpret_cast<int4*>(&q_no_transpose_buf[dest_q_no_transpose_idx]) =
                *reinterpret_cast<int4*>(&q[q_store_idx]);
            *reinterpret_cast<int4*>(&q_no_transpose_buf[dest_q_permuted_no_transpose_idx]) =
                *reinterpret_cast<int4*>(&q_permuted[q_store_idx]);
        }

        if (store_q) {
#ifdef ENABLE_FP8
            if (QuantizedQKV != nullptr) {
                // fp8 paged fmha
                QuantizedVecType* quantized_q_ptr =
                    USE_PAGED_FMHA ? reinterpret_ptr<QuantizedEltType, QuantizedVecType>(q_buf, dest_q_idx) :
                                     reinterpret_ptr<QuantizedEltType, QuantizedVecType>(QuantizedQKV, src_q_idx);
                QuantizedVecType* quantized_q_permuted_ptr =
                    USE_PAGED_FMHA ?
                        reinterpret_ptr<QuantizedEltType, QuantizedVecType>(q_buf, dest_q_permuted_idx) :
                        reinterpret_ptr<QuantizedEltType, QuantizedVecType>(QuantizedQKV, src_q_permuted_idx);
                convert_to_fp8(quantized_q_ptr, q[q_store_idx]);
                convert_to_fp8(quantized_q_permuted_ptr, q_permuted[q_store_idx]);
            } else {
                // paged fmha
                *reinterpret_cast<int4*>(&q_buf[dest_q_idx]) = *reinterpret_cast<int4*>(&q[q_store_idx]);
                *reinterpret_cast<int4*>(&q_buf[dest_q_permuted_idx]) =
                    *reinterpret_cast<int4*>(&q_permuted[q_store_idx]);
            }
#else
            // paged fmha
            *reinterpret_cast<int4*>(&q_buf[dest_q_idx])          = *reinterpret_cast<int4*>(&q[q_store_idx]);
            *reinterpret_cast<int4*>(&q_buf[dest_q_permuted_idx]) = *reinterpret_cast<int4*>(&q_permuted[q_store_idx]);
#endif
        }
    } else if (bidz < max_k_bidz) {
        vec_t2 k[2];
        vec_t2 k_bias[2];
        vec_t2 k_permuted[2];
        vec_t2 k_permuted_bias[2];
        int    k_load_idx  = 0;
        int    k_store_idx = 0;
        int    k_idx_off   = 1;

        int    head_idx       = (bidz - max_q_bidz) * HEAD_K_BLOCK_NUM;
        size_t hidden_idx     = head_idx * size_per_head + lane_id * vec_size;
        size_t src_k_idx      = token_idx * (n + 2 * kv_n) + hidden_idx + n;
        size_t src_k_last_idx = 0;

        size_t hidden_permuted_idx     = hidden_idx + rope_config.dim / 2;
        size_t src_k_permuted_idx      = src_k_idx + rope_config.dim / 2;
        size_t src_k_permuted_last_idx = 0;

        const int dst_kv_seq_idx = seq_idx + prefix_prompt_length;
        size_t    dest_kv_idx    = batch_idx * size_per_head * total_seq_len * head_num_kv
                             + head_idx * size_per_head * total_seq_len + dst_kv_seq_idx * size_per_head
                             + lane_id * vec_size;
        size_t dest_kv_permuted_idx = dest_kv_idx + rope_config.dim / 2;

        KVBlockArray kv_block_array      = param.kv_block_array;
        const float  scale               = 1.f;
        Tcache*      k_cache             = nullptr;
        int          inBlockIdx          = -1;
        int          inBlockIdx_permuted = -1;
        float*       k_scale_ptr         = nullptr;
        int          inScaleIdx          = -1;
        if (store_cache) {
            k_cache    = reinterpret_cast<Tcache*>(kv_block_array.getKBlockPtr(batch_idx, dst_kv_seq_idx));
            inBlockIdx = kv_block_array.getKVLocalIdx(dst_kv_seq_idx, head_idx, size_per_head, lane_id * vec_size);
            inBlockIdx_permuted = kv_block_array.getKVLocalIdx(
                dst_kv_seq_idx, head_idx, size_per_head, lane_id * vec_size + rope_config.dim / 2);
            if constexpr (ENABLE_8BITS_CACHE) {
                k_scale_ptr = reinterpret_cast<float*>(kv_block_array.getKScalePtr(batch_idx, dst_kv_seq_idx));
                inScaleIdx  = kv_block_array.getKVScaleLocalIdx(dst_kv_seq_idx, head_idx);
            }
        }

        *reinterpret_cast<int4*>(&k[k_load_idx])          = *reinterpret_cast<int4*>(&QKV[src_k_idx]);
        *reinterpret_cast<int4*>(&k_permuted[k_load_idx]) = *reinterpret_cast<int4*>(&QKV[src_k_permuted_idx]);

        if (qkv_bias) {
            *reinterpret_cast<int4*>(&k_bias[k_load_idx]) = *reinterpret_cast<const int4*>(&qkv_bias[hidden_idx + n]);
            *reinterpret_cast<int4*>(&k_permuted_bias[k_load_idx]) =
                *reinterpret_cast<const int4*>(&qkv_bias[hidden_permuted_idx + n]);
            k[k_load_idx]          = add(k[k_load_idx], k_bias[k_load_idx]);
            k_permuted[k_load_idx] = add(k_permuted[k_load_idx], k_permuted_bias[k_load_idx]);
        }

#pragma unroll
        for (int h = 1; h < HEAD_K_BLOCK_NUM; ++h) {
            k_load_idx ^= k_idx_off;

            ++head_idx;
            hidden_idx += size_per_head;
            src_k_last_idx = src_k_idx;
            src_k_idx += size_per_head;
            hidden_permuted_idx += size_per_head;
            src_k_permuted_last_idx = src_k_permuted_idx;
            src_k_permuted_idx += size_per_head;

            *reinterpret_cast<int4*>(&k[k_load_idx])          = *reinterpret_cast<int4*>(&QKV[src_k_idx]);
            *reinterpret_cast<int4*>(&k_permuted[k_load_idx]) = *reinterpret_cast<int4*>(&QKV[src_k_permuted_idx]);

            if (qkv_bias) {
                *reinterpret_cast<int4*>(&k_bias[k_load_idx]) =
                    *reinterpret_cast<const int4*>(&qkv_bias[hidden_idx + n]);
                *reinterpret_cast<int4*>(&k_permuted_bias[k_load_idx]) =
                    *reinterpret_cast<const int4*>(&qkv_bias[hidden_permuted_idx + n]);
                k[k_load_idx]          = add(k[k_load_idx], k_bias[k_load_idx]);
                k_permuted[k_load_idx] = add(k_permuted[k_load_idx], k_permuted_bias[k_load_idx]);
            }

            apply_rope_with_cache<vec_t2, T, Float8_, ROPE_STYLE>(
                k[k_store_idx], k_permuted[k_store_idx], coef1, coef2);

            if (store_qkv) {
#ifdef ENABLE_FP8
                if (QuantizedQKV != nullptr) {
                    // use 1.0f scale currently for qkv input of FP8 FMHA.
                    convert_to_fp8(reinterpret_cast<QuantizedVecType*>(reinterpret_cast<QuantizedEltType*>(QuantizedQKV)
                                                                       + src_k_last_idx),
                                   k[k_store_idx]);
                    convert_to_fp8(reinterpret_cast<QuantizedVecType*>(reinterpret_cast<QuantizedEltType*>(QuantizedQKV)
                                                                       + src_k_permuted_last_idx),
                                   k_permuted[k_store_idx]);
                }
#endif
                *reinterpret_cast<int4*>(&QKV[src_k_last_idx]) = *reinterpret_cast<int4*>(&k[k_store_idx]);
                *reinterpret_cast<int4*>(&QKV[src_k_permuted_last_idx]) =
                    *reinterpret_cast<int4*>(&k_permuted[k_store_idx]);
            }

            if (store_kv) {
                *reinterpret_cast<int4*>(&k_buf[dest_kv_idx]) = *reinterpret_cast<int4*>(&k[k_store_idx]);
                *reinterpret_cast<int4*>(&k_buf[dest_kv_permuted_idx]) =
                    *reinterpret_cast<int4*>(&k_permuted[k_store_idx]);
                dest_kv_idx += (size_per_head * total_seq_len);
                dest_kv_permuted_idx += (size_per_head * total_seq_len);
            }

            if (store_cache) {
                if constexpr (ENABLE_8BITS_CACHE) {
                    store_8bits_kv_cache_vec(k_cache, k[k_store_idx], inBlockIdx, scale);
                    store_8bits_kv_cache_vec(k_cache, k_permuted[k_store_idx], inBlockIdx_permuted, scale);
                    if (lane_id == 0) {
                        *reinterpret_cast<float*>(&k_scale_ptr[inScaleIdx]) = scale;
                    }
                } else {
                    *reinterpret_cast<int4*>(&k_cache[inBlockIdx]) = *reinterpret_cast<int4*>(&k[k_store_idx]);
                    *reinterpret_cast<int4*>(&k_cache[inBlockIdx_permuted]) =
                        *reinterpret_cast<int4*>(&k_permuted[k_store_idx]);
                }

                inBlockIdx = kv_block_array.getKVLocalIdx(dst_kv_seq_idx, head_idx, size_per_head, lane_id * vec_size);
                inBlockIdx_permuted = kv_block_array.getKVLocalIdx(
                    dst_kv_seq_idx, head_idx, size_per_head, lane_id * vec_size + rope_config.dim / 2);
                if constexpr (ENABLE_8BITS_CACHE) {
                    inScaleIdx = kv_block_array.getKVScaleLocalIdx(dst_kv_seq_idx, head_idx);
                }
            }

            k_store_idx ^= k_idx_off;
        }

        apply_rope_with_cache<vec_t2, T, Float8_, ROPE_STYLE>(k[k_store_idx], k_permuted[k_store_idx], coef1, coef2);

        if (store_qkv) {
#ifdef ENABLE_FP8
            if (QuantizedQKV != nullptr) {
                // use 1.0f scale currently for qkv input of FP8 FMHA.
                convert_to_fp8(
                    reinterpret_cast<QuantizedVecType*>(reinterpret_cast<QuantizedEltType*>(QuantizedQKV) + src_k_idx),
                    k[k_store_idx]);
                convert_to_fp8(reinterpret_cast<QuantizedVecType*>(reinterpret_cast<QuantizedEltType*>(QuantizedQKV)
                                                                   + src_k_permuted_idx),
                               k_permuted[k_store_idx]);
            }
#endif
            *reinterpret_cast<int4*>(&QKV[src_k_idx])          = *reinterpret_cast<int4*>(&k[k_store_idx]);
            *reinterpret_cast<int4*>(&QKV[src_k_permuted_idx]) = *reinterpret_cast<int4*>(&k_permuted[k_store_idx]);
        }

        if (store_kv) {
            *reinterpret_cast<int4*>(&k_buf[dest_kv_idx])          = *reinterpret_cast<int4*>(&k[k_store_idx]);
            *reinterpret_cast<int4*>(&k_buf[dest_kv_permuted_idx]) = *reinterpret_cast<int4*>(&k_permuted[k_store_idx]);
        }

        if (store_cache) {
            if constexpr (ENABLE_8BITS_CACHE) {
                store_8bits_kv_cache_vec(k_cache, k[k_store_idx], inBlockIdx, scale);
                store_8bits_kv_cache_vec(k_cache, k_permuted[k_store_idx], inBlockIdx_permuted, scale);
                if (lane_id == 0) {
                    *reinterpret_cast<float*>(&k_scale_ptr[inScaleIdx]) = scale;
                }
            } else {
                *reinterpret_cast<int4*>(&k_cache[inBlockIdx]) = *reinterpret_cast<int4*>(&k[k_store_idx]);
                *reinterpret_cast<int4*>(&k_cache[inBlockIdx_permuted]) =
                    *reinterpret_cast<int4*>(&k_permuted[k_store_idx]);
            }
        }
    } else {
        vec_t2 v[2];
        vec_t2 v_permuted[2];
        vec_t2 v_bias[2];
        vec_t2 v_permuted_bias[2];
        int    v_load_idx  = 0;
        int    v_store_idx = 0;
        int    v_idx_off   = 1;

        int    head_idx       = (bidz - max_k_bidz) * HEAD_V_BLOCK_NUM;
        size_t hidden_idx     = head_idx * size_per_head + lane_id * vec_size;
        size_t src_v_idx      = token_idx * (n + 2 * kv_n) + hidden_idx + n + kv_n;
        size_t src_v_last_idx = 0;

        size_t hidden_permuted_idx     = hidden_idx + rope_config.dim / 2;
        size_t src_v_permuted_idx      = src_v_idx + rope_config.dim / 2;
        size_t src_v_permuted_last_idx = 0;

        const int dst_kv_seq_idx = seq_idx + prefix_prompt_length;
        size_t    dest_kv_idx    = batch_idx * size_per_head * total_seq_len * head_num_kv
                             + head_idx * size_per_head * total_seq_len + dst_kv_seq_idx * size_per_head
                             + lane_id * vec_size;
        size_t dest_kv_permuted_idx = dest_kv_idx + rope_config.dim / 2;

        KVBlockArray kv_block_array      = param.kv_block_array;
        const float  scale               = 1.f;
        Tcache*      v_cache             = nullptr;
        int          inBlockIdx          = -1;
        int          inBlockIdx_permuted = -1;
        float*       v_scale_ptr         = nullptr;
        int          inScaleIdx          = -1;
        if (store_cache) {
            v_cache    = reinterpret_cast<Tcache*>(kv_block_array.getVBlockPtr(batch_idx, dst_kv_seq_idx));
            inBlockIdx = kv_block_array.getKVLocalIdx(dst_kv_seq_idx, head_idx, size_per_head, lane_id * vec_size);
            inBlockIdx_permuted = kv_block_array.getKVLocalIdx(
                dst_kv_seq_idx, head_idx, size_per_head, lane_id * vec_size + rope_config.dim / 2);
            if constexpr (ENABLE_8BITS_CACHE) {
                v_scale_ptr = reinterpret_cast<float*>(kv_block_array.getVScalePtr(batch_idx, dst_kv_seq_idx));
                inScaleIdx  = kv_block_array.getKVScaleLocalIdx(dst_kv_seq_idx, head_idx);
            }
        }

        *reinterpret_cast<int4*>(&v[v_load_idx])          = *reinterpret_cast<int4*>(&QKV[src_v_idx]);
        *reinterpret_cast<int4*>(&v_permuted[v_load_idx]) = *reinterpret_cast<int4*>(&QKV[src_v_permuted_idx]);

        if (qkv_bias) {
            *reinterpret_cast<int4*>(&v_bias[v_load_idx]) =
                *reinterpret_cast<const int4*>(&qkv_bias[hidden_idx + n + kv_n]);
            *reinterpret_cast<int4*>(&v_permuted_bias[v_load_idx]) =
                *reinterpret_cast<const int4*>(&qkv_bias[hidden_permuted_idx + n + kv_n]);
            v[v_load_idx]          = add(v[v_load_idx], v_bias[v_load_idx]);
            v_permuted[v_load_idx] = add(v_permuted[v_load_idx], v_permuted_bias[v_load_idx]);
        }

#pragma unroll
        for (int h = 1; h < HEAD_V_BLOCK_NUM; ++h) {
            v_load_idx ^= v_idx_off;

            ++head_idx;
            hidden_idx += size_per_head;
            src_v_last_idx = src_v_idx;
            src_v_idx += size_per_head;
            hidden_permuted_idx += size_per_head;
            src_v_permuted_last_idx = src_v_permuted_idx;
            src_v_permuted_idx += size_per_head;

            *reinterpret_cast<int4*>(&v[v_load_idx])          = *reinterpret_cast<int4*>(&QKV[src_v_idx]);
            *reinterpret_cast<int4*>(&v_permuted[v_load_idx]) = *reinterpret_cast<int4*>(&QKV[src_v_permuted_idx]);

            if (qkv_bias) {
                *reinterpret_cast<int4*>(&v_bias[v_load_idx]) =
                    *reinterpret_cast<const int4*>(&qkv_bias[hidden_idx + n + kv_n]);
                *reinterpret_cast<int4*>(&v_permuted_bias[v_load_idx]) =
                    *reinterpret_cast<const int4*>(&qkv_bias[hidden_permuted_idx + n + kv_n]);
                v[v_load_idx]          = add(v[v_load_idx], v_bias[v_load_idx]);
                v_permuted[v_load_idx] = add(v_permuted[v_load_idx], v_permuted_bias[v_load_idx]);
            }

            if (store_qkv) {
#ifdef ENABLE_FP8
                if (QuantizedQKV != nullptr) {
                    // use 1.0f scale currently for qkv input of FP8 FMHA.
                    convert_to_fp8(reinterpret_cast<QuantizedVecType*>(reinterpret_cast<QuantizedEltType*>(QuantizedQKV)
                                                                       + src_v_last_idx),
                                   v[v_store_idx]);
                    convert_to_fp8(reinterpret_cast<QuantizedVecType*>(reinterpret_cast<QuantizedEltType*>(QuantizedQKV)
                                                                       + src_v_permuted_last_idx),
                                   v_permuted[v_store_idx]);
                }
#endif
                *reinterpret_cast<int4*>(&QKV[src_v_last_idx]) = *reinterpret_cast<int4*>(&v[v_store_idx]);
                *reinterpret_cast<int4*>(&QKV[src_v_permuted_last_idx]) =
                    *reinterpret_cast<int4*>(&v_permuted[v_store_idx]);
            }

            if (store_kv) {
                *reinterpret_cast<int4*>(&v_buf[dest_kv_idx]) = *reinterpret_cast<int4*>(&v[v_store_idx]);
                *reinterpret_cast<int4*>(&v_buf[dest_kv_permuted_idx]) =
                    *reinterpret_cast<int4*>(&v_permuted[v_store_idx]);
                dest_kv_idx += (size_per_head * total_seq_len);
                dest_kv_permuted_idx += (size_per_head * total_seq_len);
            }

            if (store_cache) {
                if constexpr (ENABLE_8BITS_CACHE) {
                    store_8bits_kv_cache_vec(v_cache, v[v_store_idx], inBlockIdx, scale);
                    store_8bits_kv_cache_vec(v_cache, v_permuted[v_store_idx], inBlockIdx_permuted, scale);
                    if (lane_id == 0) {
                        *reinterpret_cast<float*>(&v_scale_ptr[inScaleIdx]) = scale;
                    }
                } else {
                    *reinterpret_cast<int4*>(&v_cache[inBlockIdx]) = *reinterpret_cast<int4*>(&v[v_store_idx]);
                    *reinterpret_cast<int4*>(&v_cache[inBlockIdx_permuted]) =
                        *reinterpret_cast<int4*>(&v_permuted[v_store_idx]);
                }

                inBlockIdx = kv_block_array.getKVLocalIdx(dst_kv_seq_idx, head_idx, size_per_head, lane_id * vec_size);
                inBlockIdx_permuted = kv_block_array.getKVLocalIdx(
                    dst_kv_seq_idx, head_idx, size_per_head, lane_id * vec_size + rope_config.dim / 2);
                if constexpr (ENABLE_8BITS_CACHE) {
                    inScaleIdx = kv_block_array.getKVScaleLocalIdx(dst_kv_seq_idx, head_idx);
                }
            }

            v_store_idx ^= v_idx_off;
        }

        if (store_qkv) {
#ifdef ENABLE_FP8
            if (QuantizedQKV != nullptr) {
                // use 1.0f scale currently for qkv input of FP8 FMHA.
                convert_to_fp8(
                    reinterpret_cast<QuantizedVecType*>(reinterpret_cast<QuantizedEltType*>(QuantizedQKV) + src_v_idx),
                    v[v_store_idx]);
                convert_to_fp8(reinterpret_cast<QuantizedVecType*>(reinterpret_cast<QuantizedEltType*>(QuantizedQKV)
                                                                   + src_v_permuted_idx),
                               v_permuted[v_store_idx]);
            }
#endif
            *reinterpret_cast<int4*>(&QKV[src_v_idx])          = *reinterpret_cast<int4*>(&v[v_store_idx]);
            *reinterpret_cast<int4*>(&QKV[src_v_permuted_idx]) = *reinterpret_cast<int4*>(&v_permuted[v_store_idx]);
        }

        if (store_kv) {
            *reinterpret_cast<int4*>(&v_buf[dest_kv_idx])          = *reinterpret_cast<int4*>(&v[v_store_idx]);
            *reinterpret_cast<int4*>(&v_buf[dest_kv_permuted_idx]) = *reinterpret_cast<int4*>(&v_permuted[v_store_idx]);
        }

        if (store_cache) {
            if constexpr (ENABLE_8BITS_CACHE) {
                store_8bits_kv_cache_vec(v_cache, v[v_store_idx], inBlockIdx, scale);
                store_8bits_kv_cache_vec(v_cache, v_permuted[v_store_idx], inBlockIdx_permuted, scale);
                if (lane_id == 0) {
                    *reinterpret_cast<float*>(&v_scale_ptr[inScaleIdx]) = scale;
                }
            } else {
                *reinterpret_cast<int4*>(&v_cache[inBlockIdx]) = *reinterpret_cast<int4*>(&v[v_store_idx]);
                *reinterpret_cast<int4*>(&v_cache[inBlockIdx_permuted]) =
                    *reinterpret_cast<int4*>(&v_permuted[v_store_idx]);
            }
        }
    }
}
#endif

template<typename T, typename Tcache, bool PREFIX_PROMPT, bool USE_PAGED_FMHA, RopeStyle ROPE_STYLE>
__global__ void add_fusedQKV_bias_transpose_with_rope_cache_kernel(T*                            q_no_transpose_buf,
                                                                   T*                            q_buf,
                                                                   T*                            k_buf,
                                                                   T*                            v_buf,
                                                                   PrefixPromptBatchWeightsParam param,
                                                                   T*                            QKV,
                                                                   void*                         QuantizedQKV,
                                                                   const int*                    position_ids,
                                                                   const T* __restrict qkv_bias,
                                                                   const int*   padding_offset,
                                                                   const int*   cu_seqlens,
                                                                   const float* rope_cache,
                                                                   const int    batch_size,
                                                                   const int    seq_len,
                                                                   const int    head_num,
                                                                   const int    head_num_kv,
                                                                   const int    size_per_head,
                                                                   RopeConfig   rope_config,
                                                                   const bool   use_logn_attn,
                                                                   bool         store_qkv,
                                                                   bool         store_q_no_transpose,
                                                                   bool         store_q,
                                                                   bool         store_kv,
                                                                   bool         store_cache) {
    // This kernel add bias to QKV, which has shape [batch_size, seq_len, 3, head_num, size_per_head], and
    // QKV split to 3 split buffer q, k, v and transpose them to [batch_size, head_num, seq_len, size_per_head].
    // For q and k, also apply the rotary embedding.

    // When we pass prefix prompt, this kernel also concatenate the prefix prompt and key/value along
    // seq_len dimension like [prompt, key/value].
    // So, the final shape of q is same ([batch_size, head_num, seq_len, size_per_head]), but
    // the shapes of key and values become [batch_size, head_num, max_prefix_prompt_length + seq_len, size_per_head].

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

    const int bidy          = blockIdx.y;
    const int tidx          = threadIdx.x;
    const int total_seq_len = param.max_prefix_prompt_length + seq_len;

    if (tidx * vec_size >= size_per_head) {
        return;
    }

    const int prefix_prompt_length = PREFIX_PROMPT ? param.d_prefix_prompt_lengths[batch_idx] : 0;
    const int n                    = head_num * size_per_head;
    const int kv_n                 = head_num_kv * size_per_head;  // MQA

    // NOTE: q has seq len excluding prefix prompt
    // src QKV: [batch, time, 3, head, hidden]
    int    position_id = position_ids ? position_ids[token_idx * rope_config.index_factor] :
                                        (PREFIX_PROMPT && param.count_length ? seq_idx + prefix_prompt_length : seq_idx);
    bool   work        = false;
    float2 coef;
    if (bidy < head_num + head_num_kv) {
        constexpr int rope_size = vector_size<T, Vec_t>::size;
        const int     rope_idx  = tidx * rope_size;
        work                    = (rope_idx >= 0 && rope_idx < rope_config.dim);
        if (work) {
            coef =
                *(reinterpret_cast<float2*>(const_cast<float*>(&rope_cache[position_id * rope_config.dim + tidx * 2])));
        }
    }

    if (bidy < head_num) {
        const int    head_idx   = bidy;
        const size_t hidden_idx = head_idx * size_per_head + tidx * vec_size;
        const size_t src_q_idx  = token_idx * (n + 2 * kv_n) + hidden_idx;
        const int    pre_len    = cu_seqlens[batch_idx];

        Vec_t q = *reinterpret_cast<const Vec_t*>(&QKV[src_q_idx]);

        if (qkv_bias) {
            Vec_t q_bias = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx]);
            q            = add(q, q_bias);
        }

        apply_rope_with_cache<Vec_t, T, float2, ROPE_STYLE>(
            q, reinterpret_cast<T*>(smem_), tidx, rope_config.dim, coef, work);

        if (use_logn_attn) {
            logn_attention(q, seq_idx, rope_config.max_pos);
        }

        if (store_qkv) {
            *reinterpret_cast<Vec_t*>(&QKV[src_q_idx]) = q;
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

        if (store_q_no_transpose) {
            size_t dest_q_no_transpose_idx =
                (pre_len + seq_idx) * head_num * size_per_head + head_idx * size_per_head + tidx * vec_size;

            *reinterpret_cast<Vec_t*>(&q_no_transpose_buf[dest_q_no_transpose_idx]) = q;
        }

        if (store_q) {
            size_t dest_q_idx = batch_idx * size_per_head * seq_len * head_num + head_idx * size_per_head * seq_len
                                + seq_idx * size_per_head + tidx * vec_size;
            if constexpr (USE_PAGED_FMHA) {
                dest_q_idx =
                    (pre_len + seq_idx) * size_per_head * head_num + head_idx * size_per_head + tidx * vec_size;
            }

#ifdef ENABLE_FP8
            if (QuantizedQKV != nullptr) {
                // fp8 paged fmha
                QuantizedVecType* quantized_q_ptr =
                    USE_PAGED_FMHA ? reinterpret_ptr<QuantizedEltType, QuantizedVecType>(q_buf, dest_q_idx) :
                                     reinterpret_ptr<QuantizedEltType, QuantizedVecType>(QuantizedQKV, src_q_idx);
                convert_to_fp8(quantized_q_ptr, q);
            } else {
                // paged fmha
                *reinterpret_cast<Vec_t*>(&q_buf[dest_q_idx]) = q;
            }
#else
            // paged fmha
            *reinterpret_cast<Vec_t*>(&q_buf[dest_q_idx]) = q;
#endif
        }
    } else if (bidy < head_num + head_num_kv) {
        const int    head_idx       = bidy - head_num;
        const size_t hidden_idx     = head_idx * size_per_head + tidx * vec_size;
        const size_t src_k_idx      = token_idx * (n + 2 * kv_n) + hidden_idx + n;
        const int    dst_kv_seq_idx = seq_idx + prefix_prompt_length;

        Vec_t k = *reinterpret_cast<const Vec_t*>(&QKV[src_k_idx]);

        if (qkv_bias) {
            Vec_t k_bias = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx + n]);
            k            = add(k, k_bias);
        }

        apply_rope_with_cache<Vec_t, T, float2, ROPE_STYLE>(
            k, reinterpret_cast<T*>(smem_), tidx, rope_config.dim, coef, work);

        if (store_qkv) {
#ifdef ENABLE_FP8
            if (QuantizedQKV != nullptr) {
                // use 1.0f scale currently for qkv input of FP8 FMHA.
                convert_to_fp8(
                    reinterpret_cast<QuantizedVecType*>(reinterpret_cast<QuantizedEltType*>(QuantizedQKV) + src_k_idx),
                    k);
            }
#endif
            *reinterpret_cast<Vec_t*>(&QKV[src_k_idx]) = k;
        }

        if (store_kv) {
            const size_t dest_kv_idx = batch_idx * size_per_head * total_seq_len * head_num_kv
                                       + head_idx * size_per_head * total_seq_len + dst_kv_seq_idx * size_per_head
                                       + tidx * vec_size;
            *reinterpret_cast<Vec_t*>(&k_buf[dest_kv_idx]) = k;
        }

        if (store_cache) {
            KVBlockArray kv_block_array = param.kv_block_array;
            Tcache*      k_cache = reinterpret_cast<Tcache*>(kv_block_array.getKBlockPtr(batch_idx, dst_kv_seq_idx));
            const int    inBlockIdx =
                kv_block_array.getKVLocalIdx(dst_kv_seq_idx, head_idx, size_per_head, tidx * vec_size);
            if constexpr (ENABLE_8BITS_CACHE) {
                float* k_scale_ptr = reinterpret_cast<float*>(kv_block_array.getKScalePtr(batch_idx, dst_kv_seq_idx));
                const int        inScaleIdx = kv_block_array.getKVScaleLocalIdx(dst_kv_seq_idx, head_idx);
                __shared__ float s_max;
                if constexpr (std::is_same<Tcache, int8_t>::value) {
                    float local_max;
                    local_max = vector_abs_max(k);
                    blockReduceMaxV2<float, 1>(&local_max);
                    if (tidx == 0) {
                        s_max = local_max;
                    }
                } else {
                    s_max = float(1 << (8 - 1));
                }
                __syncthreads();

                store_8bits_kv_cache_vec(k_cache, k, inBlockIdx, float(1 << (8 - 1)) / s_max);
                if (tidx == 0) {
                    *reinterpret_cast<float*>(&k_scale_ptr[inScaleIdx]) = s_max / float(1 << (8 - 1));
                }
            } else {
                *reinterpret_cast<Vec_t*>(&k_cache[inBlockIdx]) = k;
            }
        }
    } else {
        const int    head_idx       = bidy - head_num - head_num_kv;
        const size_t hidden_idx     = head_idx * size_per_head + tidx * vec_size;
        const size_t src_v_idx      = token_idx * (n + 2 * kv_n) + hidden_idx + n + kv_n;
        const int    dst_kv_seq_idx = seq_idx + prefix_prompt_length;

        Vec_t v = *reinterpret_cast<const Vec_t*>(&QKV[src_v_idx]);

        if (qkv_bias) {
            Vec_t v_bias = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx + n + kv_n]);
            v            = add(v, v_bias);
        }

        __syncthreads();

        if (store_qkv) {
#ifdef ENABLE_FP8
            if (QuantizedQKV != nullptr) {
                // use 1.0f scale currently for qkv input of FP8 FMHA.
                convert_to_fp8(
                    reinterpret_cast<QuantizedVecType*>(reinterpret_cast<QuantizedEltType*>(QuantizedQKV) + src_v_idx),
                    v);
            }
#endif
            *reinterpret_cast<Vec_t*>(&QKV[src_v_idx]) = v;
        }

        if (store_kv) {
            const size_t dest_kv_idx = batch_idx * size_per_head * total_seq_len * head_num_kv
                                       + head_idx * size_per_head * total_seq_len + dst_kv_seq_idx * size_per_head
                                       + tidx * vec_size;
            *reinterpret_cast<Vec_t*>(&v_buf[dest_kv_idx]) = v;
        }

        if (store_cache) {
            KVBlockArray kv_block_array = param.kv_block_array;
            Tcache*      v_cache = reinterpret_cast<Tcache*>(kv_block_array.getVBlockPtr(batch_idx, dst_kv_seq_idx));
            const int    inBlockIdx =
                kv_block_array.getKVLocalIdx(dst_kv_seq_idx, head_idx, size_per_head, tidx * vec_size);
            if constexpr (ENABLE_8BITS_CACHE) {
                float* v_scale_ptr = reinterpret_cast<float*>(kv_block_array.getVScalePtr(batch_idx, dst_kv_seq_idx));
                const int        inScaleIdx = kv_block_array.getKVScaleLocalIdx(dst_kv_seq_idx, head_idx);
                __shared__ float s_max;
                if constexpr (std::is_same<Tcache, int8_t>::value) {
                    float local_max;
                    local_max = vector_abs_max(v);
                    blockReduceMaxV2<float, 1>(&local_max);
                    if (tidx == 0) {
                        s_max = local_max;
                    }
                } else {
                    s_max = float(1 << (8 - 1));
                }
                __syncthreads();

                store_8bits_kv_cache_vec(v_cache, v, inBlockIdx, float(1 << (8 - 1)) / s_max);
                if (tidx == 0) {
                    *reinterpret_cast<float*>(&v_scale_ptr[inScaleIdx]) = s_max / float(1 << (8 - 1));
                }
            } else {
                *reinterpret_cast<Vec_t*>(&v_cache[inBlockIdx]) = v;
            }
        }
    }
}

template<typename T, typename Tcache, bool PREFIX_PROMPT, bool USE_PAGED_FMHA, RopeStyle ROPE_STYLE>
__global__ void add_fusedQKV_bias_transpose_kernel(T*                            q_no_transpose_buf,
                                                   T*                            q_buf,
                                                   T*                            k_buf,
                                                   T*                            v_buf,
                                                   PrefixPromptBatchWeightsParam param,
                                                   T*                            QKV,
                                                   void*                         QuantizedQKV,
                                                   const int*                    position_ids,
                                                   const T* __restrict qkv_bias,
                                                   const int* padding_offset,
                                                   const int* cu_seqlens,
                                                   const int  batch_size,
                                                   const int  seq_len,
                                                   const int  head_num,
                                                   const int  head_num_kv,
                                                   const int  size_per_head,
                                                   RopeConfig rope_config,
                                                   const bool use_logn_attn,
                                                   bool       store_qkv,
                                                   bool       store_q_no_transpose,
                                                   bool       store_q,
                                                   bool       store_kv,
                                                   bool       store_cache) {
    // This kernel add bias to QKV, which has shape [batch_size, seq_len, 3, head_num, size_per_head], and
    // QKV split to 3 split buffer q, k, v and transpose them to [batch_size, head_num, seq_len, size_per_head].
    // For q and k, also apply the rotary embedding.

    // When we pass prefix prompt, this kernel also concatenate the prefix prompt and key/value along
    // seq_len dimension like [prompt, key/value].
    // So, the final shape of q is same ([batch_size, head_num, seq_len, size_per_head]), but
    // the shapes of key and values become [batch_size, head_num, max_prefix_prompt_length + seq_len, size_per_head].

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
                                       param.count_length);

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

    if (store_q_no_transpose) {
        size_t dest_q_no_transpose_idx =
            (pre_len + seq_idx) * head_num * size_per_head + head_idx * size_per_head + tidx * vec_size;

        *reinterpret_cast<Vec_t*>(&q_no_transpose_buf[dest_q_no_transpose_idx]) = q;
    }

    if (store_q) {
        size_t dest_q_idx = batch_idx * size_per_head * seq_len * head_num + head_idx * size_per_head * seq_len
                            + seq_idx * size_per_head + tidx * vec_size;
        if constexpr (USE_PAGED_FMHA) {
            dest_q_idx = (pre_len + seq_idx) * size_per_head * head_num + head_idx * size_per_head + tidx * vec_size;
        }

#ifdef ENABLE_FP8
        if (QuantizedQKV != nullptr) {
            // fp8 paged fmha
            QuantizedVecType* quantized_q_ptr =
                USE_PAGED_FMHA ? reinterpret_ptr<QuantizedEltType, QuantizedVecType>(q_buf, dest_q_idx) :
                                 reinterpret_ptr<QuantizedEltType, QuantizedVecType>(QuantizedQKV, src_q_idx);
            convert_to_fp8(quantized_q_ptr, q);
        } else {
            // paged fmha
            *reinterpret_cast<Vec_t*>(&q_buf[dest_q_idx]) = q;
        }
#else
        // paged fmha
        *reinterpret_cast<Vec_t*>(&q_buf[dest_q_idx]) = q;
#endif
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
            if constexpr (ENABLE_8BITS_CACHE) {
                float* k_scale_ptr = reinterpret_cast<float*>(kv_block_array.getKScalePtr(batch_idx, dst_kv_seq_idx));
                float* v_scale_ptr = reinterpret_cast<float*>(kv_block_array.getVScalePtr(batch_idx, dst_kv_seq_idx));
                const int inBlockIdx =
                    kv_block_array.getKVLocalIdx(dst_kv_seq_idx, head_idx, size_per_head, tidx * vec_size);
                const int        inScaleIdx = kv_block_array.getKVScaleLocalIdx(dst_kv_seq_idx, head_idx);
                __shared__ float s_max[2];
                if constexpr (std::is_same<Tcache, int8_t>::value) {
                    float local_max[2];
                    local_max[0] = vector_abs_max(k);
                    local_max[1] = vector_abs_max(v);
                    blockReduceMaxV2<float, 2>(local_max);
                    if (threadIdx.x == 0) {
                        s_max[0] = local_max[0];
                        s_max[1] = local_max[1];
                    }
                } else {
                    s_max[0] = float(1 << (8 - 1));
                    s_max[1] = float(1 << (8 - 1));
                }
                __syncthreads();

                store_8bits_kv_cache_vec(k_cache, k, inBlockIdx, float(1 << (8 - 1)) / s_max[0]);
                store_8bits_kv_cache_vec(v_cache, v, inBlockIdx, float(1 << (8 - 1)) / s_max[1]);
                if (tidx == 0) {
                    *reinterpret_cast<float*>(&k_scale_ptr[inScaleIdx]) = s_max[0] / float(1 << (8 - 1));
                    *reinterpret_cast<float*>(&v_scale_ptr[inScaleIdx]) = s_max[1] / float(1 << (8 - 1));
                }
            } else {
                const int inBlockIdx =
                    kv_block_array.getKVLocalIdx(dst_kv_seq_idx, head_idx, size_per_head, tidx * vec_size);
                *reinterpret_cast<Vec_t*>(&k_cache[inBlockIdx]) = k;
                *reinterpret_cast<Vec_t*>(&v_cache[inBlockIdx]) = v;
            }
        }
    }
}

#define ADD_FUSEDQKV_BIAS_TRANSPOSE_NON_INT8_WITH_ROPE_CACHE(head_q_block_num, head_k_block_num, head_v_block_num)     \
    constexpr int thread_num        = 128;                                                                             \
    const int     threads_per_token = size_per_head / Vec_t2<T>::size / 2;                                             \
    const int     tokens_per_block  = thread_num / threads_per_token;                                                  \
    dim3          block(thread_num);                                                                                   \
    dim3          grid(batch_size,                                                                                     \
              ceil_div<int>(seq_len, tokens_per_block),                                                       \
              head_num / head_q_block_num + head_num_kv / head_k_block_num + head_num_kv / head_v_block_num); \
    FT_SWITCH(param_ptr->max_prefix_prompt_length != 0, PREFIX_PROMPT, [&] {                                           \
        FT_SWITCH(use_paged_fmha, USE_PAGED_FMHA, [&] {                                                                \
            FT_SWITCH_KV_CACHE_TYPE_NON_INT8_CASE(param_ptr->kv_block_array.cache_type, Tcache, [&] {                  \
                FT_ROPE_WITH_CACHE_SWITCH(rope_config.style, ROPE_STYLE, [&] {                                         \
                    add_fusedQKV_bias_transpose_non_int8_with_rope_cache_kernel<T,                                     \
                                                                                Tcache,                                \
                                                                                PREFIX_PROMPT,                         \
                                                                                USE_PAGED_FMHA,                        \
                                                                                ROPE_STYLE,                            \
                                                                                head_q_block_num,                      \
                                                                                head_k_block_num,                      \
                                                                                head_v_block_num>                      \
                        <<<grid, block, 0, stream>>>(q_no_transpose_buf,                                               \
                                                     q_buf,                                                            \
                                                     k_buf,                                                            \
                                                     v_buf,                                                            \
                                                     *param_ptr,                                                       \
                                                     QKV,                                                              \
                                                     QuantizedQKV,                                                     \
                                                     position_ids,                                                     \
                                                     qkv_bias,                                                         \
                                                     padding_offset,                                                   \
                                                     cu_seqlens,                                                       \
                                                     rope_cache,                                                       \
                                                     threads_per_token,                                                \
                                                     tokens_per_block,                                                 \
                                                     batch_size,                                                       \
                                                     seq_len,                                                          \
                                                     head_num,                                                         \
                                                     head_num_kv,                                                      \
                                                     size_per_head,                                                    \
                                                     rope_config,                                                      \
                                                     use_logn_attn,                                                    \
                                                     store_qkv,                                                        \
                                                     store_q_no_transpose,                                             \
                                                     store_q,                                                          \
                                                     store_kv,                                                         \
                                                     store_cache);                                                     \
                });                                                                                                    \
            });                                                                                                        \
        });                                                                                                            \
    })

template<typename T>
void invokeAddFusedQKVBiasTranspose(T*                             q_no_transpose_buf,
                                    T*                             q_buf,
                                    T*                             k_buf,
                                    T*                             v_buf,
                                    PrefixPromptBatchWeightsParam* param_ptr,
                                    T*                             QKV,
                                    void*                          QuantizedQKV,
                                    const int*                     position_ids,
                                    const T*                       qkv_bias,
                                    const int*                     padding_offset,
                                    const int*                     cu_seqlens,
                                    const bool                     use_rope_cache,
                                    const float*                   rope_cache,
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
                                    const bool                     store_q_no_transpose,
                                    const bool                     store_q,
                                    const bool                     store_kv,
                                    const bool                     store_cache,
                                    cudaStream_t                   stream) {
    if (use_rope_cache && rope_cache) {
#if USING_CUDA
        if (head_num % 8 == 0 && head_num_kv % 4 == 0 && param_ptr->kv_block_array.cache_type != KvCacheDataType::INT8
            && size_per_head == rope_config.dim) {
            ADD_FUSEDQKV_BIAS_TRANSPOSE_NON_INT8_WITH_ROPE_CACHE(8, 4, 4);
        } else {
#endif
            dim3         block((size_per_head / Vec_t<T>::size + 31) / 32 * 32);
            dim3         grid(token_num, head_num + head_num_kv * 2);
            const size_t smem_size = rope_config.style == RopeStyle::No ? 0 : 2 * rope_config.dim * sizeof(T);

            FT_SWITCH(param_ptr->max_prefix_prompt_length != 0, PREFIX_PROMPT, [&] {
                FT_SWITCH(use_paged_fmha, USE_PAGED_FMHA, [&] {
                    FT_SWITCH_KV_CACHE_TYPE_CASE(param_ptr->kv_block_array.cache_type, Tcache, [&] {
                        FT_ROPE_WITH_CACHE_SWITCH(rope_config.style, ROPE_STYLE, [&] {
                            add_fusedQKV_bias_transpose_with_rope_cache_kernel<T,
                                                                               Tcache,
                                                                               PREFIX_PROMPT,
                                                                               USE_PAGED_FMHA,
                                                                               ROPE_STYLE>
                                <<<grid, block, smem_size, stream>>>(q_no_transpose_buf,
                                                                     q_buf,
                                                                     k_buf,
                                                                     v_buf,
                                                                     *param_ptr,
                                                                     QKV,
                                                                     QuantizedQKV,
                                                                     position_ids,
                                                                     qkv_bias,
                                                                     padding_offset,
                                                                     cu_seqlens,
                                                                     rope_cache,
                                                                     batch_size,
                                                                     seq_len,
                                                                     head_num,
                                                                     head_num_kv,
                                                                     size_per_head,
                                                                     rope_config,
                                                                     use_logn_attn,
                                                                     store_qkv,
                                                                     store_q_no_transpose,
                                                                     store_q,
                                                                     store_kv,
                                                                     store_cache);
                        });
                    });
                });
            });
#if USING_CUDA
        }
#endif
    } else {
        dim3         block((size_per_head / Vec_t<T>::size + 31) / 32 * 32);
        dim3         grid(token_num, head_num);
        const size_t smem_size = rope_config.style == RopeStyle::No ? 0 : 2 * rope_config.dim * sizeof(T);

        FT_SWITCH(param_ptr->max_prefix_prompt_length != 0, PREFIX_PROMPT, [&] {
            FT_SWITCH(use_paged_fmha, USE_PAGED_FMHA, [&] {
                FT_SWITCH_KV_CACHE_TYPE_CASE(param_ptr->kv_block_array.cache_type, Tcache, [&] {
                    FT_ROPE_SWITCH(rope_config.style, ROPE_STYLE, [&] {
                        add_fusedQKV_bias_transpose_kernel<T, Tcache, PREFIX_PROMPT, USE_PAGED_FMHA, ROPE_STYLE>
                            <<<grid, block, smem_size, stream>>>(q_no_transpose_buf,
                                                                 q_buf,
                                                                 k_buf,
                                                                 v_buf,
                                                                 *param_ptr,
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
                                                                 store_q_no_transpose,
                                                                 store_q,
                                                                 store_kv,
                                                                 store_cache);
                    });
                });
            });
        });
    }
#if USING_CUDA
    check_cuda_value(cudaPeekAtLastError());
    check_cuda_error();
#endif
}

template<typename T, typename Tcache, RopeStyle ROPE_STYLE>
__global__ void decode_add_fusedQKV_bias_transpose_with_rope_cache_kernel(T*           q_buf,
                                                                          T*           k_buf,
                                                                          T*           v_buf,
                                                                          KVBlockArray kv_block_array,
                                                                          T*           QKV,
                                                                          const int*   position_ids,
                                                                          const T* __restrict qkv_bias,
                                                                          const float* rope_cache,
                                                                          const int    batch_size,
                                                                          const int    head_num,
                                                                          const int    head_num_kv,
                                                                          const int    size_per_head,
                                                                          RopeConfig   rope_config,
                                                                          const bool   use_logn_attn,
                                                                          bool         store_q,
                                                                          bool         store_kv,
                                                                          bool         store_cache) {
    // This kernel add bias to QKV, which has shape [batch_size, seq_len, 3, head_num, size_per_head], and
    // QKV split to 3 split buffer q, k, v and transpose them to [batch_size, head_num, seq_len, size_per_head].
    // For q and k, also apply the rotary embedding.

    // When we pass prefix prompt, this kernel also concatenate the prefix prompt and key/value along
    // seq_len dimension like [prompt, key/value].
    // So, the final shape of q is same ([batch_size, head_num, seq_len, size_per_head]), but
    // the shapes of key and values become [batch_size, head_num, max_prefix_prompt_length + seq_len, size_per_head].

    // NOTE: QKV src shape (batch_size, seq_len, 3, head_num, size_per_head)
    //  QKV dst shape (3, batch_size, head_num, seq_len, size_per_head)
    extern __shared__ __align__(sizeof(float2)) char smem_[];  // align on largest vector type

    static constexpr bool ENABLE_8BITS_CACHE = sizeof(Tcache) == 1;

#ifdef ENABLE_FP8
    // Quantized output only supports fp8 currently.
    using QuantizedEltType = __nv_fp8_e4m3;
    using QuantizedVecType = typename Vec_t<T>::QuantizedType;
#endif
    constexpr int vec_size      = Vec_t<T>::size;
    using Vec_t                 = typename Vec_t<T>::Type;
    const int     batch_idx     = blockIdx.x;
    constexpr int seq_len       = 1;
    const int     token_idx     = batch_idx;
    constexpr int seq_idx       = 0;
    const int     tidx          = threadIdx.x;
    const int     bidy          = blockIdx.y;
    const int     total_seq_len = seq_len;

    if (tidx * vec_size >= size_per_head) {
        return;
    }

    const int n    = head_num * size_per_head;
    const int kv_n = head_num_kv * size_per_head;  // MQA

    int    position_id = position_ids[token_idx * rope_config.index_factor];
    bool   work        = false;
    float2 coef;
    if (bidy < head_num + head_num_kv) {
        constexpr int rope_size = vector_size<T, Vec_t>::size;
        const int     rope_idx  = tidx * rope_size;
        work                    = (rope_idx >= 0 && rope_idx < rope_config.dim);
        if (work) {
            coef =
                *(reinterpret_cast<float2*>(const_cast<float*>(&rope_cache[position_id * rope_config.dim + tidx * 2])));
        }
    }

    if (bidy < head_num) {
        const int    head_idx   = bidy;
        const size_t hidden_idx = head_idx * size_per_head + tidx * vec_size;
        const size_t src_q_idx  = token_idx * (n + 2 * kv_n) + hidden_idx;

        Vec_t q = *reinterpret_cast<const Vec_t*>(&QKV[src_q_idx]);

        if (qkv_bias) {
            Vec_t q_bias = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx]);
            q            = add(q, q_bias);
        }

        apply_rope_with_cache<Vec_t, T, float2, ROPE_STYLE>(
            q, reinterpret_cast<T*>(smem_), tidx, rope_config.dim, coef, work);

        if (use_logn_attn) {
            logn_attention(q, seq_idx, rope_config.max_pos);
        }

        if (store_q) {
            size_t dest_q_idx = batch_idx * size_per_head * seq_len * head_num + head_idx * size_per_head * seq_len
                                + seq_idx * size_per_head + tidx * vec_size;
            *reinterpret_cast<Vec_t*>(&q_buf[dest_q_idx]) = q;
        }
    } else if (bidy < head_num + head_num_kv) {
        const int    head_idx   = bidy - head_num;
        const size_t hidden_idx = head_idx * size_per_head + tidx * vec_size;
        const size_t src_k_idx  = token_idx * (n + 2 * kv_n) + hidden_idx + n;

        Vec_t k = *reinterpret_cast<const Vec_t*>(&QKV[src_k_idx]);

        if (qkv_bias) {
            Vec_t k_bias = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx + n]);
            k            = add(k, k_bias);
        }

        apply_rope_with_cache<Vec_t, T, float2, ROPE_STYLE>(
            k, reinterpret_cast<T*>(smem_), tidx, rope_config.dim, coef, work);

        if (store_kv) {
            const int    dst_kv_seq_idx = seq_idx;
            const size_t dest_kv_idx    = batch_idx * size_per_head * total_seq_len * head_num_kv
                                       + head_idx * size_per_head * total_seq_len + dst_kv_seq_idx * size_per_head
                                       + tidx * vec_size;
            *reinterpret_cast<Vec_t*>(&k_buf[dest_kv_idx]) = k;
        }

        if (store_cache) {
            const int dst_kv_seq_idx = seq_idx + position_id;
            Tcache*   k_cache = reinterpret_cast<Tcache*>(kv_block_array.getKBlockPtr(batch_idx, dst_kv_seq_idx));
            const int inBlockIdx =
                kv_block_array.getKVLocalIdx(dst_kv_seq_idx, head_idx, size_per_head, tidx * vec_size);
            if constexpr (ENABLE_8BITS_CACHE) {
                float* k_scale_ptr = reinterpret_cast<float*>(kv_block_array.getKScalePtr(batch_idx, dst_kv_seq_idx));
                const int        inScaleIdx = kv_block_array.getKVScaleLocalIdx(dst_kv_seq_idx, head_idx);
                __shared__ float s_max;
                if constexpr (std::is_same<Tcache, int8_t>::value) {
                    float local_max;
                    local_max = vector_abs_max(k);
                    blockReduceMaxV2<float, 1>(&local_max);
                    if (tidx == 0) {
                        s_max = local_max;
                    }
                } else {
                    s_max = float(1 << (8 - 1));
                }
                __syncthreads();

                store_8bits_kv_cache_vec(k_cache, k, inBlockIdx, float(1 << (8 - 1)) / s_max);
                if (tidx == 0) {
                    *reinterpret_cast<float*>(&k_scale_ptr[inScaleIdx]) = s_max / float(1 << (8 - 1));
                }
            } else {
                *reinterpret_cast<Vec_t*>(&k_cache[inBlockIdx]) = k;
            }
        }
    } else {
        const int    head_idx   = bidy - head_num - head_num_kv;
        const size_t hidden_idx = head_idx * size_per_head + tidx * vec_size;
        const size_t src_v_idx  = token_idx * (n + 2 * kv_n) + hidden_idx + n + kv_n;

        Vec_t v = *reinterpret_cast<const Vec_t*>(&QKV[src_v_idx]);

        if (qkv_bias) {
            Vec_t v_bias = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx + n + kv_n]);
            v            = add(v, v_bias);
        }

        __syncthreads();

        if (store_kv) {
            const int    dst_kv_seq_idx = seq_idx;
            const size_t dest_kv_idx    = batch_idx * size_per_head * total_seq_len * head_num_kv
                                       + head_idx * size_per_head * total_seq_len + dst_kv_seq_idx * size_per_head
                                       + tidx * vec_size;
            *reinterpret_cast<Vec_t*>(&v_buf[dest_kv_idx]) = v;
        }

        if (store_cache) {
            const int dst_kv_seq_idx = seq_idx + position_id;
            Tcache*   v_cache = reinterpret_cast<Tcache*>(kv_block_array.getVBlockPtr(batch_idx, dst_kv_seq_idx));
            const int inBlockIdx =
                kv_block_array.getKVLocalIdx(dst_kv_seq_idx, head_idx, size_per_head, tidx * vec_size);
            if constexpr (ENABLE_8BITS_CACHE) {
                float* v_scale_ptr = reinterpret_cast<float*>(kv_block_array.getVScalePtr(batch_idx, dst_kv_seq_idx));
                const int        inScaleIdx = kv_block_array.getKVScaleLocalIdx(dst_kv_seq_idx, head_idx);
                __shared__ float s_max;
                if constexpr (std::is_same<Tcache, int8_t>::value) {
                    float local_max;
                    local_max = vector_abs_max(v);
                    blockReduceMaxV2<float, 1>(&local_max);
                    if (tidx == 0) {
                        s_max = local_max;
                    }
                } else {
                    s_max = float(1 << (8 - 1));
                }
                __syncthreads();

                store_8bits_kv_cache_vec(v_cache, v, inBlockIdx, float(1 << (8 - 1)) / s_max);
                if (tidx == 0) {
                    *reinterpret_cast<float*>(&v_scale_ptr[inScaleIdx]) = s_max / float(1 << (8 - 1));
                }
            } else {
                *reinterpret_cast<Vec_t*>(&v_cache[inBlockIdx]) = v;
            }
        }
    }
}

template<typename T, typename Tcache, RopeStyle ROPE_STYLE>
__global__ void decode_add_fusedQKV_bias_transpose_kernel(T*           q_buf,
                                                          T*           k_buf,
                                                          T*           v_buf,
                                                          KVBlockArray kv_block_array,
                                                          T*           QKV,
                                                          const int*   position_ids,
                                                          const T* __restrict qkv_bias,
                                                          const int  batch_size,
                                                          const int  head_num,
                                                          const int  head_num_kv,
                                                          const int  size_per_head,
                                                          RopeConfig rope_config,
                                                          const bool use_logn_attn,
                                                          bool       store_q,
                                                          bool       store_kv,
                                                          bool       store_cache) {
    // This kernel add bias to QKV, which has shape [batch_size, seq_len, 3, head_num, size_per_head], and
    // QKV split to 3 split buffer q, k, v and transpose them to [batch_size, head_num, seq_len, size_per_head].
    // For q and k, also apply the rotary embedding.

    // When we pass prefix prompt, this kernel also concatenate the prefix prompt and key/value along
    // seq_len dimension like [prompt, key/value].
    // So, the final shape of q is same ([batch_size, head_num, seq_len, size_per_head]), but
    // the shapes of key and values become [batch_size, head_num, max_prefix_prompt_length + seq_len, size_per_head].

    // NOTE: QKV src shape (batch_size, seq_len, 3, head_num, size_per_head)
    //  QKV dst shape (3, batch_size, head_num, seq_len, size_per_head)
    extern __shared__ __align__(sizeof(float2)) char smem_[];  // align on largest vector type

    static constexpr bool ENABLE_8BITS_CACHE = sizeof(Tcache) == 1;

#ifdef ENABLE_FP8
    // Quantized output only supports fp8 currently.
    using QuantizedEltType = __nv_fp8_e4m3;
    using QuantizedVecType = typename Vec_t<T>::QuantizedType;
#endif
    constexpr int vec_size      = Vec_t<T>::size;
    using Vec_t                 = typename Vec_t<T>::Type;
    const int     batch_idx     = blockIdx.x;
    constexpr int seq_len       = 1;
    const int     token_idx     = batch_idx;
    constexpr int seq_idx       = 0;
    const int     tidx          = threadIdx.x;
    const int     bidy          = blockIdx.y;
    const int     total_seq_len = seq_len;

    if (tidx * vec_size >= size_per_head) {
        return;
    }

    const int n    = head_num * size_per_head;
    const int kv_n = head_num_kv * size_per_head;  // MQA

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

    if (bidy < head_num) {
        const int    head_idx   = bidy;
        const size_t hidden_idx = head_idx * size_per_head + tidx * vec_size;
        const size_t src_q_idx  = token_idx * (n + 2 * kv_n) + hidden_idx;

        Vec_t q = *reinterpret_cast<const Vec_t*>(&QKV[src_q_idx]);

        if (qkv_bias) {
            Vec_t q_bias = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx]);
            q            = add(q, q_bias);
        }

        apply_rope<T, Vec_t, ROPE_STYLE>(rope_config, q, reinterpret_cast<T*>(smem_), tidx, position_id, seq_len);

        if (use_logn_attn) {
            logn_attention(q, seq_idx, rope_config.max_pos);
        }

        __syncthreads();

        if (store_q) {
            size_t dest_q_idx = batch_idx * size_per_head * seq_len * head_num + head_idx * size_per_head * seq_len
                                + seq_idx * size_per_head + tidx * vec_size;
            *reinterpret_cast<Vec_t*>(&q_buf[dest_q_idx]) = q;
        }
    } else if (bidy < head_num + head_num_kv) {
        const int    head_idx   = bidy - head_num;
        const size_t hidden_idx = head_idx * size_per_head + tidx * vec_size;
        const size_t src_k_idx  = token_idx * (n + 2 * kv_n) + hidden_idx + n;

        Vec_t k = *reinterpret_cast<const Vec_t*>(&QKV[src_k_idx]);

        if (qkv_bias) {
            Vec_t k_bias = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx + n]);
            k            = add(k, k_bias);
        }

        apply_rope<T, Vec_t, ROPE_STYLE>(rope_config, k, reinterpret_cast<T*>(smem_), tidx, position_id, seq_len);

        __syncthreads();

        if (store_kv) {
            const int    dst_kv_seq_idx = seq_idx;
            const size_t dest_kv_idx    = batch_idx * size_per_head * total_seq_len * head_num_kv
                                       + head_idx * size_per_head * total_seq_len + dst_kv_seq_idx * size_per_head
                                       + tidx * vec_size;
            *reinterpret_cast<Vec_t*>(&k_buf[dest_kv_idx]) = k;
        }

        if (store_cache) {
            const int dst_kv_seq_idx = seq_idx + position_id;
            Tcache*   k_cache = reinterpret_cast<Tcache*>(kv_block_array.getKBlockPtr(batch_idx, dst_kv_seq_idx));
            const int inBlockIdx =
                kv_block_array.getKVLocalIdx(dst_kv_seq_idx, head_idx, size_per_head, tidx * vec_size);
            if constexpr (ENABLE_8BITS_CACHE) {
                float* k_scale_ptr = reinterpret_cast<float*>(kv_block_array.getKScalePtr(batch_idx, dst_kv_seq_idx));
                const int        inScaleIdx = kv_block_array.getKVScaleLocalIdx(dst_kv_seq_idx, head_idx);
                __shared__ float s_max;
                if constexpr (std::is_same<Tcache, int8_t>::value) {
                    float local_max;
                    local_max = vector_abs_max(k);
                    blockReduceMaxV2<float, 1>(&local_max);
                    if (tidx == 0) {
                        s_max = local_max;
                    }
                } else {
                    s_max = float(1 << (8 - 1));
                }
                __syncthreads();

                store_8bits_kv_cache_vec(k_cache, k, inBlockIdx, float(1 << (8 - 1)) / s_max);
                if (tidx == 0) {
                    *reinterpret_cast<float*>(&k_scale_ptr[inScaleIdx]) = s_max / float(1 << (8 - 1));
                }
            } else {
                *reinterpret_cast<Vec_t*>(&k_cache[inBlockIdx]) = k;
            }
        }
    } else {
        const int    head_idx   = bidy - head_num - head_num_kv;
        const size_t hidden_idx = head_idx * size_per_head + tidx * vec_size;
        const size_t src_v_idx  = token_idx * (n + 2 * kv_n) + hidden_idx + n + kv_n;

        Vec_t v = *reinterpret_cast<const Vec_t*>(&QKV[src_v_idx]);

        if (qkv_bias) {
            Vec_t v_bias = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx + n + kv_n]);
            v            = add(v, v_bias);
        }

        __syncthreads();

        if (store_kv) {
            const int    dst_kv_seq_idx = seq_idx;
            const size_t dest_kv_idx    = batch_idx * size_per_head * total_seq_len * head_num_kv
                                       + head_idx * size_per_head * total_seq_len + dst_kv_seq_idx * size_per_head
                                       + tidx * vec_size;
            *reinterpret_cast<Vec_t*>(&v_buf[dest_kv_idx]) = v;
        }

        if (store_cache) {
            const int dst_kv_seq_idx = seq_idx + position_id;
            Tcache*   v_cache = reinterpret_cast<Tcache*>(kv_block_array.getVBlockPtr(batch_idx, dst_kv_seq_idx));
            const int inBlockIdx =
                kv_block_array.getKVLocalIdx(dst_kv_seq_idx, head_idx, size_per_head, tidx * vec_size);
            if constexpr (ENABLE_8BITS_CACHE) {
                float* v_scale_ptr = reinterpret_cast<float*>(kv_block_array.getVScalePtr(batch_idx, dst_kv_seq_idx));
                const int        inScaleIdx = kv_block_array.getKVScaleLocalIdx(dst_kv_seq_idx, head_idx);
                __shared__ float s_max;
                if constexpr (std::is_same<Tcache, int8_t>::value) {
                    float local_max;
                    local_max = vector_abs_max(v);
                    blockReduceMaxV2<float, 1>(&local_max);
                    if (tidx == 0) {
                        s_max = local_max;
                    }
                } else {
                    s_max = float(1 << (8 - 1));
                }
                __syncthreads();

                store_8bits_kv_cache_vec(v_cache, v, inBlockIdx, float(1 << (8 - 1)) / s_max);
                if (tidx == 0) {
                    *reinterpret_cast<float*>(&v_scale_ptr[inScaleIdx]) = s_max / float(1 << (8 - 1));
                }
            } else {
                *reinterpret_cast<Vec_t*>(&v_cache[inBlockIdx]) = v;
            }
        }
    }
}

template<typename T,
         typename Tcache,
         RopeStyle ROPE_STYLE,
         int       HEAD_Q_BLOCK_NUM,
         int       HEAD_K_BLOCK_NUM,
         int       HEAD_V_BLOCK_NUM>
__global__ void decode_add_fusedQKV_bias_transpose_non_int8_with_rope_cache_kernel(T*           q_buf,
                                                                                   T*           k_buf,
                                                                                   T*           v_buf,
                                                                                   KVBlockArray kv_block_array,
                                                                                   T*           QKV,
                                                                                   const int*   position_ids,
                                                                                   const T* __restrict qkv_bias,
                                                                                   const float* rope_cache,
                                                                                   const int    batch_size,
                                                                                   const int    head_num,
                                                                                   const int    head_num_kv,
                                                                                   const int    size_per_head,
                                                                                   RopeConfig   rope_config,
                                                                                   const bool   use_logn_attn,
                                                                                   bool         store_q,
                                                                                   bool         store_kv,
                                                                                   bool         store_cache) {
    // This kernel add bias to QKV, which has shape [batch_size, seq_len, 3, head_num, size_per_head], and
    // QKV split to 3 split buffer q, k, v and transpose them to [batch_size, head_num, seq_len, size_per_head].
    // For q and k, also apply the rotary embedding.

    // When we pass prefix prompt, this kernel also concatenate the prefix prompt and key/value along
    // seq_len dimension like [prompt, key/value].
    // So, the final shape of q is same ([batch_size, head_num, seq_len, size_per_head]), but
    // the shapes of key and values become [batch_size, head_num, max_prefix_prompt_length + seq_len, size_per_head].

    // NOTE: QKV src shape (batch_size, seq_len, 3, head_num, size_per_head)
    //  QKV dst shape (3, batch_size, head_num, seq_len, size_per_head)
    extern __shared__ __align__(sizeof(float2)) char smem_[];  // align on largest vector type

    static constexpr bool ENABLE_8BITS_CACHE = sizeof(Tcache) == 1;

#ifdef ENABLE_FP8
    // Quantized output only supports fp8 currently.
    using QuantizedEltType = __nv_fp8_e4m3;
    using QuantizedVecType = typename Vec_t<T>::QuantizedType;
#endif
    constexpr int vec_size      = Vec_t<T>::size;
    using Vec_t                 = typename Vec_t<T>::Type;
    const int     batch_idx     = blockIdx.x;
    constexpr int seq_len       = 1;
    const int     token_idx     = batch_idx;
    constexpr int seq_idx       = 0;
    const int     tidx          = threadIdx.x;
    const int     bidy          = blockIdx.y;
    const int     max_q_bidy    = head_num / HEAD_Q_BLOCK_NUM;
    const int     max_k_bidy    = max_q_bidy + head_num_kv / HEAD_K_BLOCK_NUM;
    const int     max_v_bidy    = max_k_bidy + head_num_kv / HEAD_V_BLOCK_NUM;
    const int     total_seq_len = seq_len;
    if (bidy >= max_v_bidy) {
        return;
    }

    if (tidx * vec_size >= size_per_head) {
        return;
    }

    const int n    = head_num * size_per_head;
    const int kv_n = head_num_kv * size_per_head;  // MQA

    int    position_id = position_ids[token_idx * rope_config.index_factor];
    bool   work        = false;
    float2 coef;
    if (bidy < max_k_bidy) {
        constexpr int rope_size = vector_size<T, Vec_t>::size;
        const int     rope_idx  = tidx * rope_size;
        work                    = (rope_idx >= 0 && rope_idx < rope_config.dim);
        if (work) {
            coef =
                *(reinterpret_cast<float2*>(const_cast<float*>(&rope_cache[position_id * rope_config.dim + tidx * 2])));
        }
    }

    if (bidy < max_q_bidy) {
        Vec_t q[2];
        Vec_t q_bias[2];
        int   q_load_idx  = 0;
        int   q_store_idx = 0;
        int   q_idx_off   = 1;

        int    head_idx   = bidy * HEAD_Q_BLOCK_NUM;
        size_t hidden_idx = head_idx * size_per_head + tidx * vec_size;
        size_t src_q_idx  = token_idx * (n + 2 * kv_n) + hidden_idx;

        size_t dest_q_idx = batch_idx * size_per_head * seq_len * head_num + head_idx * size_per_head * seq_len
                            + seq_idx * size_per_head + tidx * vec_size;

        q[q_load_idx] = *reinterpret_cast<const Vec_t*>(&QKV[src_q_idx]);

        if (qkv_bias) {
            q_bias[q_load_idx] = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx]);
            q[q_load_idx]      = add(q[q_load_idx], q_bias[q_load_idx]);
        }

#pragma unroll
        for (int h = 1; h < HEAD_Q_BLOCK_NUM; ++h) {
            q_load_idx ^= q_idx_off;

            ++head_idx;
            hidden_idx += size_per_head;
            src_q_idx += size_per_head;

            q[q_load_idx] = *reinterpret_cast<const Vec_t*>(&QKV[src_q_idx]);

            if (qkv_bias) {
                q_bias[q_load_idx] = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx]);
                q[q_load_idx]      = add(q[q_load_idx], q_bias[q_load_idx]);
            }

            apply_rope_with_cache<Vec_t, T, float2, ROPE_STYLE>(
                q[q_store_idx], reinterpret_cast<T*>(smem_), tidx, rope_config.dim, coef, work);

            if (use_logn_attn) {
                logn_attention(q[q_store_idx], seq_idx, rope_config.max_pos);
            }

            if (store_q) {
                *reinterpret_cast<Vec_t*>(&q_buf[dest_q_idx]) = q[q_store_idx];
                dest_q_idx += (size_per_head * seq_len);
            }

            q_store_idx ^= q_idx_off;
        }

        apply_rope_with_cache<Vec_t, T, float2, ROPE_STYLE>(
            q[q_store_idx], reinterpret_cast<T*>(smem_), tidx, rope_config.dim, coef, work);

        if (use_logn_attn) {
            logn_attention(q[q_store_idx], seq_idx, rope_config.max_pos);
        }

        if (store_q) {
            *reinterpret_cast<Vec_t*>(&q_buf[dest_q_idx]) = q[q_store_idx];
        }
    } else if (bidy < max_k_bidy) {
        Vec_t k[2];
        Vec_t k_bias[2];
        int   k_load_idx  = 0;
        int   k_store_idx = 0;
        int   k_idx_off   = 1;

        int    head_idx   = (bidy - max_q_bidy) * HEAD_K_BLOCK_NUM;
        size_t hidden_idx = head_idx * size_per_head + tidx * vec_size;
        size_t src_k_idx  = token_idx * (n + 2 * kv_n) + hidden_idx + n;

        const int dst_kv_seq_idx = seq_idx;
        size_t    dest_kv_idx    = batch_idx * size_per_head * total_seq_len * head_num_kv
                             + head_idx * size_per_head * total_seq_len + dst_kv_seq_idx * size_per_head
                             + tidx * vec_size;

        const int   dst_kv_pos_idx = seq_idx + position_id;
        const float scale          = 1.f;
        Tcache*     k_cache        = nullptr;
        int         inBlockIdx     = -1;
        float*      k_scale_ptr    = nullptr;
        int         inScaleIdx     = -1;
        if (store_cache) {
            k_cache    = reinterpret_cast<Tcache*>(kv_block_array.getKBlockPtr(batch_idx, dst_kv_pos_idx));
            inBlockIdx = kv_block_array.getKVLocalIdx(dst_kv_pos_idx, head_idx, size_per_head, tidx * vec_size);
            if constexpr (ENABLE_8BITS_CACHE) {
                k_scale_ptr = reinterpret_cast<float*>(kv_block_array.getKScalePtr(batch_idx, dst_kv_pos_idx));
                inScaleIdx  = kv_block_array.getKVScaleLocalIdx(dst_kv_pos_idx, head_idx);
            }
        }

        k[k_load_idx] = *reinterpret_cast<const Vec_t*>(&QKV[src_k_idx]);

        if (qkv_bias) {
            k_bias[k_load_idx] = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx + n]);
            k[k_load_idx]      = add(k[k_load_idx], k_bias[k_load_idx]);
        }

#pragma unroll
        for (int h = 1; h < HEAD_K_BLOCK_NUM; ++h) {
            k_load_idx ^= k_idx_off;

            ++head_idx;
            hidden_idx += size_per_head;
            src_k_idx += size_per_head;

            k[k_load_idx] = *reinterpret_cast<const Vec_t*>(&QKV[src_k_idx]);

            if (qkv_bias) {
                k_bias[k_load_idx] = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx + n]);
                k[k_load_idx]      = add(k[k_load_idx], k_bias[k_load_idx]);
            }

            apply_rope_with_cache<Vec_t, T, float2, ROPE_STYLE>(
                k[k_store_idx], reinterpret_cast<T*>(smem_), tidx, rope_config.dim, coef, work);

            if (store_kv) {
                *reinterpret_cast<Vec_t*>(&k_buf[dest_kv_idx]) = k[k_store_idx];
                dest_kv_idx += (size_per_head * total_seq_len);
            }

            if (store_cache) {
                if constexpr (ENABLE_8BITS_CACHE) {
                    store_8bits_kv_cache_vec(k_cache, k[k_store_idx], inBlockIdx, scale);
                    if (tidx == 0) {
                        *reinterpret_cast<float*>(&k_scale_ptr[inScaleIdx]) = scale;
                    }
                } else {
                    *reinterpret_cast<Vec_t*>(&k_cache[inBlockIdx]) = k[k_store_idx];
                }

                inBlockIdx = kv_block_array.getKVLocalIdx(dst_kv_pos_idx, head_idx, size_per_head, tidx * vec_size);
                if constexpr (ENABLE_8BITS_CACHE) {
                    inScaleIdx = kv_block_array.getKVScaleLocalIdx(dst_kv_pos_idx, head_idx);
                }
            }

            k_store_idx ^= k_idx_off;
        }

        apply_rope_with_cache<Vec_t, T, float2, ROPE_STYLE>(
            k[k_store_idx], reinterpret_cast<T*>(smem_), tidx, rope_config.dim, coef, work);

        if (store_kv) {
            *reinterpret_cast<Vec_t*>(&k_buf[dest_kv_idx]) = k[k_store_idx];
        }

        if (store_cache) {
            if constexpr (ENABLE_8BITS_CACHE) {
                store_8bits_kv_cache_vec(k_cache, k[k_store_idx], inBlockIdx, scale);
                if (tidx == 0) {
                    *reinterpret_cast<float*>(&k_scale_ptr[inScaleIdx]) = scale;
                }
            } else {
                *reinterpret_cast<Vec_t*>(&k_cache[inBlockIdx]) = k[k_store_idx];
            }
        }
    } else {
        Vec_t v[2];
        Vec_t v_bias[2];
        int   v_load_idx  = 0;
        int   v_store_idx = 0;
        int   v_idx_off   = 1;

        int    head_idx   = (bidy - max_k_bidy) * HEAD_V_BLOCK_NUM;
        size_t hidden_idx = head_idx * size_per_head + tidx * vec_size;
        size_t src_v_idx  = token_idx * (n + 2 * kv_n) + hidden_idx + n + kv_n;

        const int dst_kv_seq_idx = seq_idx;
        size_t    dest_kv_idx    = batch_idx * size_per_head * total_seq_len * head_num_kv
                             + head_idx * size_per_head * total_seq_len + dst_kv_seq_idx * size_per_head
                             + tidx * vec_size;

        const int   dst_kv_pos_idx = seq_idx + position_id;
        const float scale          = 1.f;
        Tcache*     v_cache        = nullptr;
        int         inBlockIdx     = -1;
        float*      v_scale_ptr    = nullptr;
        int         inScaleIdx     = -1;
        if (store_cache) {
            v_cache    = reinterpret_cast<Tcache*>(kv_block_array.getVBlockPtr(batch_idx, dst_kv_pos_idx));
            inBlockIdx = kv_block_array.getKVLocalIdx(dst_kv_pos_idx, head_idx, size_per_head, tidx * vec_size);
            if constexpr (ENABLE_8BITS_CACHE) {
                v_scale_ptr = reinterpret_cast<float*>(kv_block_array.getVScalePtr(batch_idx, dst_kv_pos_idx));
                inScaleIdx  = kv_block_array.getKVScaleLocalIdx(dst_kv_pos_idx, head_idx);
            }
        }

        v[v_load_idx] = *reinterpret_cast<const Vec_t*>(&QKV[src_v_idx]);

        if (qkv_bias) {
            v_bias[v_load_idx] = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx + n + kv_n]);
            v[v_load_idx]      = add(v[v_load_idx], v_bias[v_load_idx]);
        }

        __syncthreads();

#pragma unroll
        for (int h = 1; h < HEAD_V_BLOCK_NUM; ++h) {
            v_load_idx ^= v_idx_off;

            ++head_idx;
            hidden_idx += size_per_head;
            src_v_idx += size_per_head;

            v[v_load_idx] = *reinterpret_cast<const Vec_t*>(&QKV[src_v_idx]);

            if (qkv_bias) {
                v_bias[v_load_idx] = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx + n + kv_n]);
                v[v_load_idx]      = add(v[v_load_idx], v_bias[v_load_idx]);
            }

            if (store_kv) {
                *reinterpret_cast<Vec_t*>(&v_buf[dest_kv_idx]) = v[v_store_idx];
                dest_kv_idx += (size_per_head * total_seq_len);
            }

            if (store_cache) {
                if constexpr (ENABLE_8BITS_CACHE) {
                    store_8bits_kv_cache_vec(v_cache, v[v_store_idx], inBlockIdx, scale);
                    if (tidx == 0) {
                        *reinterpret_cast<float*>(&v_scale_ptr[inScaleIdx]) = scale;
                    }
                } else {
                    *reinterpret_cast<Vec_t*>(&v_cache[inBlockIdx]) = v[v_store_idx];
                }

                inBlockIdx = kv_block_array.getKVLocalIdx(dst_kv_pos_idx, head_idx, size_per_head, tidx * vec_size);
                if constexpr (ENABLE_8BITS_CACHE) {
                    inScaleIdx = kv_block_array.getKVScaleLocalIdx(dst_kv_pos_idx, head_idx);
                }
            }

            v_store_idx ^= v_idx_off;

            __syncthreads();
        }

        if (store_kv) {
            *reinterpret_cast<Vec_t*>(&v_buf[dest_kv_idx]) = v[v_store_idx];
        }

        if (store_cache) {
            if constexpr (ENABLE_8BITS_CACHE) {
                store_8bits_kv_cache_vec(v_cache, v[v_store_idx], inBlockIdx, scale);
                if (tidx == 0) {
                    *reinterpret_cast<float*>(&v_scale_ptr[inScaleIdx]) = scale;
                }
            } else {
                *reinterpret_cast<Vec_t*>(&v_cache[inBlockIdx]) = v[v_store_idx];
            }
        }
    }
}

template<typename T,
         typename Tcache,
         RopeStyle ROPE_STYLE,
         int       HEAD_Q_BLOCK_NUM,
         int       HEAD_K_BLOCK_NUM,
         int       HEAD_V_BLOCK_NUM>
__global__ void decode_add_fusedQKV_bias_transpose_non_int8_kernel(T*           q_buf,
                                                                   T*           k_buf,
                                                                   T*           v_buf,
                                                                   KVBlockArray kv_block_array,
                                                                   T*           QKV,
                                                                   const int*   position_ids,
                                                                   const T* __restrict qkv_bias,
                                                                   const int  batch_size,
                                                                   const int  head_num,
                                                                   const int  head_num_kv,
                                                                   const int  size_per_head,
                                                                   RopeConfig rope_config,
                                                                   const bool use_logn_attn,
                                                                   bool       store_q,
                                                                   bool       store_kv,
                                                                   bool       store_cache) {
    // This kernel add bias to QKV, which has shape [batch_size, seq_len, 3, head_num, size_per_head], and
    // QKV split to 3 split buffer q, k, v and transpose them to [batch_size, head_num, seq_len, size_per_head].
    // For q and k, also apply the rotary embedding.

    // When we pass prefix prompt, this kernel also concatenate the prefix prompt and key/value along
    // seq_len dimension like [prompt, key/value].
    // So, the final shape of q is same ([batch_size, head_num, seq_len, size_per_head]), but
    // the shapes of key and values become [batch_size, head_num, max_prefix_prompt_length + seq_len, size_per_head].

    // NOTE: QKV src shape (batch_size, seq_len, 3, head_num, size_per_head)
    //  QKV dst shape (3, batch_size, head_num, seq_len, size_per_head)
    extern __shared__ __align__(sizeof(float2)) char smem_[];  // align on largest vector type

    static constexpr bool ENABLE_8BITS_CACHE = sizeof(Tcache) == 1;

#ifdef ENABLE_FP8
    // Quantized output only supports fp8 currently.
    using QuantizedEltType = __nv_fp8_e4m3;
    using QuantizedVecType = typename Vec_t<T>::QuantizedType;
#endif
    constexpr int vec_size      = Vec_t<T>::size;
    using Vec_t                 = typename Vec_t<T>::Type;
    const int     batch_idx     = blockIdx.x;
    constexpr int seq_len       = 1;
    const int     token_idx     = batch_idx;
    constexpr int seq_idx       = 0;
    const int     tidx          = threadIdx.x;
    const int     bidy          = blockIdx.y;
    const int     max_q_bidy    = head_num / HEAD_Q_BLOCK_NUM;
    const int     max_k_bidy    = max_q_bidy + head_num_kv / HEAD_K_BLOCK_NUM;
    const int     max_v_bidy    = max_k_bidy + head_num_kv / HEAD_V_BLOCK_NUM;
    const int     total_seq_len = seq_len;
    if (bidy >= max_v_bidy) {
        return;
    }

    if (tidx * vec_size >= size_per_head) {
        return;
    }

    const int n    = head_num * size_per_head;
    const int kv_n = head_num_kv * size_per_head;  // MQA

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

    if (bidy < max_q_bidy) {
        Vec_t q[2];
        Vec_t q_bias[2];
        int   q_load_idx  = 0;
        int   q_store_idx = 0;
        int   q_idx_off   = 1;

        int    head_idx   = bidy * HEAD_Q_BLOCK_NUM;
        size_t hidden_idx = head_idx * size_per_head + tidx * vec_size;
        size_t src_q_idx  = token_idx * (n + 2 * kv_n) + hidden_idx;

        size_t dest_q_idx = batch_idx * size_per_head * seq_len * head_num + head_idx * size_per_head * seq_len
                            + seq_idx * size_per_head + tidx * vec_size;

        q[q_load_idx] = *reinterpret_cast<const Vec_t*>(&QKV[src_q_idx]);

        if (qkv_bias) {
            q_bias[q_load_idx] = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx]);
            q[q_load_idx]      = add(q[q_load_idx], q_bias[q_load_idx]);
        }

#pragma unroll
        for (int h = 1; h < HEAD_Q_BLOCK_NUM; ++h) {
            q_load_idx ^= q_idx_off;

            ++head_idx;
            hidden_idx += size_per_head;
            src_q_idx += size_per_head;

            q[q_load_idx] = *reinterpret_cast<const Vec_t*>(&QKV[src_q_idx]);

            if (qkv_bias) {
                q_bias[q_load_idx] = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx]);
                q[q_load_idx]      = add(q[q_load_idx], q_bias[q_load_idx]);
            }

            apply_rope<T, Vec_t, ROPE_STYLE>(
                rope_config, q[q_store_idx], reinterpret_cast<T*>(smem_), tidx, position_id, seq_len);

            if (use_logn_attn) {
                logn_attention(q[q_store_idx], seq_idx, rope_config.max_pos);
            }

            __syncthreads();

            if (store_q) {
                *reinterpret_cast<Vec_t*>(&q_buf[dest_q_idx]) = q[q_store_idx];
                dest_q_idx += (size_per_head * seq_len);
            }

            q_store_idx ^= q_idx_off;
        }

        apply_rope<T, Vec_t, ROPE_STYLE>(
            rope_config, q[q_store_idx], reinterpret_cast<T*>(smem_), tidx, position_id, seq_len);

        if (use_logn_attn) {
            logn_attention(q[q_store_idx], seq_idx, rope_config.max_pos);
        }

        __syncthreads();

        if (store_q) {
            *reinterpret_cast<Vec_t*>(&q_buf[dest_q_idx]) = q[q_store_idx];
        }
    } else if (bidy < max_k_bidy) {
        Vec_t k[2];
        Vec_t k_bias[2];
        int   k_load_idx  = 0;
        int   k_store_idx = 0;
        int   k_idx_off   = 1;

        int    head_idx   = (bidy - max_q_bidy) * HEAD_K_BLOCK_NUM;
        size_t hidden_idx = head_idx * size_per_head + tidx * vec_size;
        size_t src_k_idx  = token_idx * (n + 2 * kv_n) + hidden_idx + n;

        const int dst_kv_seq_idx = seq_idx;
        size_t    dest_kv_idx    = batch_idx * size_per_head * total_seq_len * head_num_kv
                             + head_idx * size_per_head * total_seq_len + dst_kv_seq_idx * size_per_head
                             + tidx * vec_size;

        const int   dst_kv_pos_idx = seq_idx + position_id;
        Tcache*     k_cache        = reinterpret_cast<Tcache*>(kv_block_array.getKBlockPtr(batch_idx, dst_kv_pos_idx));
        int         inBlockIdx = kv_block_array.getKVLocalIdx(dst_kv_pos_idx, head_idx, size_per_head, tidx * vec_size);
        float*      k_scale_ptr = reinterpret_cast<float*>(kv_block_array.getKScalePtr(batch_idx, dst_kv_pos_idx));
        int         inScaleIdx  = kv_block_array.getKVScaleLocalIdx(dst_kv_pos_idx, head_idx);
        const float scale       = 1.f;

        k[k_load_idx] = *reinterpret_cast<const Vec_t*>(&QKV[src_k_idx]);

        if (qkv_bias) {
            k_bias[k_load_idx] = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx + n]);
            k[k_load_idx]      = add(k[k_load_idx], k_bias[k_load_idx]);
        }

#pragma unroll
        for (int h = 1; h < HEAD_K_BLOCK_NUM; ++h) {
            k_load_idx ^= k_idx_off;

            ++head_idx;
            hidden_idx += size_per_head;
            src_k_idx += size_per_head;

            k[k_load_idx] = *reinterpret_cast<const Vec_t*>(&QKV[src_k_idx]);

            if (qkv_bias) {
                k_bias[k_load_idx] = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx + n]);
                k[k_load_idx]      = add(k[k_load_idx], k_bias[k_load_idx]);
            }

            apply_rope<T, Vec_t, ROPE_STYLE>(
                rope_config, k[k_store_idx], reinterpret_cast<T*>(smem_), tidx, position_id, seq_len);

            __syncthreads();

            if (store_kv) {
                *reinterpret_cast<Vec_t*>(&k_buf[dest_kv_idx]) = k[k_store_idx];
                dest_kv_idx += (size_per_head * total_seq_len);
            }

            if (store_cache) {
                if constexpr (ENABLE_8BITS_CACHE) {
                    store_8bits_kv_cache_vec(k_cache, k[k_store_idx], inBlockIdx, scale);
                    if (tidx == 0) {
                        *reinterpret_cast<float*>(&k_scale_ptr[inScaleIdx]) = scale;
                    }
                } else {
                    *reinterpret_cast<Vec_t*>(&k_cache[inBlockIdx]) = k[k_store_idx];
                }

                inBlockIdx = kv_block_array.getKVLocalIdx(dst_kv_pos_idx, head_idx, size_per_head, tidx * vec_size);
                inScaleIdx = kv_block_array.getKVScaleLocalIdx(dst_kv_pos_idx, head_idx);
            }

            k_store_idx ^= k_idx_off;
        }

        apply_rope<T, Vec_t, ROPE_STYLE>(
            rope_config, k[k_store_idx], reinterpret_cast<T*>(smem_), tidx, position_id, seq_len);

        __syncthreads();

        if (store_kv) {
            *reinterpret_cast<Vec_t*>(&k_buf[dest_kv_idx]) = k[k_store_idx];
        }

        if (store_cache) {
            if constexpr (ENABLE_8BITS_CACHE) {
                store_8bits_kv_cache_vec(k_cache, k[k_store_idx], inBlockIdx, scale);
                if (tidx == 0) {
                    *reinterpret_cast<float*>(&k_scale_ptr[inScaleIdx]) = scale;
                }
            } else {
                *reinterpret_cast<Vec_t*>(&k_cache[inBlockIdx]) = k[k_store_idx];
            }
        }
    } else {
        Vec_t v[2];
        Vec_t v_bias[2];
        int   v_load_idx  = 0;
        int   v_store_idx = 0;
        int   v_idx_off   = 1;

        int    head_idx   = (bidy - max_k_bidy) * HEAD_V_BLOCK_NUM;
        size_t hidden_idx = head_idx * size_per_head + tidx * vec_size;
        size_t src_v_idx  = token_idx * (n + 2 * kv_n) + hidden_idx + n + kv_n;

        const int dst_kv_seq_idx = seq_idx;
        size_t    dest_kv_idx    = batch_idx * size_per_head * total_seq_len * head_num_kv
                             + head_idx * size_per_head * total_seq_len + dst_kv_seq_idx * size_per_head
                             + tidx * vec_size;

        const int   dst_kv_pos_idx = seq_idx + position_id;
        Tcache*     v_cache        = reinterpret_cast<Tcache*>(kv_block_array.getVBlockPtr(batch_idx, dst_kv_pos_idx));
        int         inBlockIdx = kv_block_array.getKVLocalIdx(dst_kv_pos_idx, head_idx, size_per_head, tidx * vec_size);
        float*      v_scale_ptr = reinterpret_cast<float*>(kv_block_array.getVScalePtr(batch_idx, dst_kv_pos_idx));
        int         inScaleIdx  = kv_block_array.getKVScaleLocalIdx(dst_kv_pos_idx, head_idx);
        const float scale       = 1.f;

        v[v_load_idx] = *reinterpret_cast<const Vec_t*>(&QKV[src_v_idx]);

        if (qkv_bias) {
            v_bias[v_load_idx] = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx + n + kv_n]);
            v[v_load_idx]      = add(v[v_load_idx], v_bias[v_load_idx]);
        }

        __syncthreads();

#pragma unroll
        for (int h = 1; h < HEAD_V_BLOCK_NUM; ++h) {
            v_load_idx ^= v_idx_off;

            ++head_idx;
            hidden_idx += size_per_head;
            src_v_idx += size_per_head;

            v[v_load_idx] = *reinterpret_cast<const Vec_t*>(&QKV[src_v_idx]);

            if (qkv_bias) {
                v_bias[v_load_idx] = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx + n + kv_n]);
                v[v_load_idx]      = add(v[v_load_idx], v_bias[v_load_idx]);
            }

            if (store_kv) {
                *reinterpret_cast<Vec_t*>(&v_buf[dest_kv_idx]) = v[v_store_idx];
                dest_kv_idx += (size_per_head * total_seq_len);
            }

            if (store_cache) {
                if constexpr (ENABLE_8BITS_CACHE) {
                    store_8bits_kv_cache_vec(v_cache, v[v_store_idx], inBlockIdx, scale);
                    if (tidx == 0) {
                        *reinterpret_cast<float*>(&v_scale_ptr[inScaleIdx]) = scale;
                    }
                } else {
                    *reinterpret_cast<Vec_t*>(&v_cache[inBlockIdx]) = v[v_store_idx];
                }

                inBlockIdx = kv_block_array.getKVLocalIdx(dst_kv_pos_idx, head_idx, size_per_head, tidx * vec_size);
                inScaleIdx = kv_block_array.getKVScaleLocalIdx(dst_kv_pos_idx, head_idx);
            }

            v_store_idx ^= v_idx_off;

            __syncthreads();
        }

        if (store_kv) {
            *reinterpret_cast<Vec_t*>(&v_buf[dest_kv_idx]) = v[v_store_idx];
        }

        if (store_cache) {
            if constexpr (ENABLE_8BITS_CACHE) {
                store_8bits_kv_cache_vec(v_cache, v[v_store_idx], inBlockIdx, scale);
                if (tidx == 0) {
                    *reinterpret_cast<float*>(&v_scale_ptr[inScaleIdx]) = scale;
                }
            } else {
                *reinterpret_cast<Vec_t*>(&v_cache[inBlockIdx]) = v[v_store_idx];
            }
        }
    }
}

template<typename T>
void invokeDecodeAddFusedQKVBiasTranspose(T*               q_buf,
                                          T*               k_buf,
                                          T*               v_buf,
                                          KVBlockArray     kv_block_array,
                                          T*               QKV,
                                          const int*       position_ids,
                                          const T*         qkv_bias,
                                          const bool       use_rope_cache,
                                          const float*     rope_cache,
                                          const int        batch_size,
                                          const int        head_num,
                                          const int        head_num_kv,
                                          const int        size_per_head,
                                          const RopeConfig rope_config,
                                          const bool       use_logn_attn,
                                          const bool       store_q,
                                          const bool       store_kv,
                                          const bool       store_cache,
                                          cudaStream_t     stream) {
    const size_t smem_size = rope_config.style == RopeStyle::No ? 0 : 2 * rope_config.dim * sizeof(T);
    dim3         block((size_per_head / Vec_t<T>::size + 31) / 32 * 32);

    if (use_rope_cache && rope_cache) {
        constexpr int head_q_block_num = 4;
        constexpr int head_k_block_num = 4;
        constexpr int head_v_block_num = 4;
        if (batch_size <= 16 || head_num % head_q_block_num != 0 || head_num_kv % head_k_block_num != 0
            || head_num_kv % head_v_block_num != 0 || kv_block_array.cache_type == KvCacheDataType::INT8) {
            dim3 grid(batch_size, head_num + head_num_kv * 2);

            FT_SWITCH_KV_CACHE_TYPE_CASE(kv_block_array.cache_type, Tcache, [&] {
                FT_ROPE_WITH_CACHE_SWITCH(rope_config.style, ROPE_STYLE, [&] {
                    decode_add_fusedQKV_bias_transpose_with_rope_cache_kernel<T, Tcache, ROPE_STYLE>
                        <<<grid, block, smem_size, stream>>>(q_buf,
                                                             k_buf,
                                                             v_buf,
                                                             kv_block_array,
                                                             QKV,
                                                             position_ids,
                                                             qkv_bias,
                                                             rope_cache,
                                                             batch_size,
                                                             head_num,
                                                             head_num_kv,
                                                             size_per_head,
                                                             rope_config,
                                                             use_logn_attn,
                                                             store_q,
                                                             store_kv,
                                                             store_cache);
                });
            });
        } else {
            dim3 grid(batch_size,
                      head_num / head_q_block_num + head_num_kv / head_k_block_num + head_num_kv / head_v_block_num);

            FT_SWITCH_KV_CACHE_TYPE_NON_INT8_CASE(kv_block_array.cache_type, Tcache, [&] {
                FT_ROPE_WITH_CACHE_SWITCH(rope_config.style, ROPE_STYLE, [&] {
                    decode_add_fusedQKV_bias_transpose_non_int8_with_rope_cache_kernel<T,
                                                                                       Tcache,
                                                                                       ROPE_STYLE,
                                                                                       head_q_block_num,
                                                                                       head_k_block_num,
                                                                                       head_v_block_num>
                        <<<grid, block, smem_size, stream>>>(q_buf,
                                                             k_buf,
                                                             v_buf,
                                                             kv_block_array,
                                                             QKV,
                                                             position_ids,
                                                             qkv_bias,
                                                             rope_cache,
                                                             batch_size,
                                                             head_num,
                                                             head_num_kv,
                                                             size_per_head,
                                                             rope_config,
                                                             use_logn_attn,
                                                             store_q,
                                                             store_kv,
                                                             store_cache);
                });
            });
        }
    } else {
        constexpr int head_q_block_num = 2;
        constexpr int head_k_block_num = 2;
        constexpr int head_v_block_num = 4;
        if (batch_size <= 16 || head_num % head_q_block_num != 0 || head_num_kv % head_k_block_num != 0
            || head_num_kv % head_v_block_num != 0 || kv_block_array.cache_type == KvCacheDataType::INT8) {
            dim3 grid(batch_size, head_num + head_num_kv * 2);

            FT_SWITCH_KV_CACHE_TYPE_CASE(kv_block_array.cache_type, Tcache, [&] {
                FT_ROPE_SWITCH(rope_config.style, ROPE_STYLE, [&] {
                    decode_add_fusedQKV_bias_transpose_kernel<T, Tcache, ROPE_STYLE>
                        <<<grid, block, smem_size, stream>>>(q_buf,
                                                             k_buf,
                                                             v_buf,
                                                             kv_block_array,
                                                             QKV,
                                                             position_ids,
                                                             qkv_bias,
                                                             batch_size,
                                                             head_num,
                                                             head_num_kv,
                                                             size_per_head,
                                                             rope_config,
                                                             use_logn_attn,
                                                             store_q,
                                                             store_kv,
                                                             store_cache);
                });
            });
        } else {
            dim3 grid(batch_size,
                      head_num / head_q_block_num + head_num_kv / head_k_block_num + head_num_kv / head_v_block_num);

            FT_SWITCH_KV_CACHE_TYPE_NON_INT8_CASE(kv_block_array.cache_type, Tcache, [&] {
                FT_ROPE_SWITCH(rope_config.style, ROPE_STYLE, [&] {
                    decode_add_fusedQKV_bias_transpose_non_int8_kernel<T,
                                                                       Tcache,
                                                                       ROPE_STYLE,
                                                                       head_q_block_num,
                                                                       head_k_block_num,
                                                                       head_v_block_num>
                        <<<grid, block, smem_size, stream>>>(q_buf,
                                                             k_buf,
                                                             v_buf,
                                                             kv_block_array,
                                                             QKV,
                                                             position_ids,
                                                             qkv_bias,
                                                             batch_size,
                                                             head_num,
                                                             head_num_kv,
                                                             size_per_head,
                                                             rope_config,
                                                             use_logn_attn,
                                                             store_q,
                                                             store_kv,
                                                             store_cache);
                });
            });
        }
    }
#if USING_CUDA
    check_cuda_value(cudaPeekAtLastError());
    check_cuda_error();
#endif
}

#if USING_ROCM
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
#endif

}  // namespace rtp_llm
