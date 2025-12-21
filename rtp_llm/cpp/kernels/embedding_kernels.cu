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

#include <assert.h>
#include <type_traits>
#include "rtp_llm/cpp/cuda/cuda_type_utils.cuh"
#include "rtp_llm/cpp/cuda/cuda_fp8_utils.h"
#if USING_CUDA
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#elif (CUDART_VERSION >= 11050)
#include <cub/cub.cuh>
#else
#include "3rdparty/cub/cub.cuh"
#endif
#endif
#include "rtp_llm/cpp/kernels/embedding_kernels.h"

namespace rtp_llm {

template<typename T, bool USE_POS_EMB, bool USE_TYPE_ID_EMB, bool USE_MASK>
__global__ void embedding_lookup_kernel(T*            from_tensor,
                                        const T*      embedding_table,
                                        double        input_embedding_scalar,
                                        const T*      pos_table,
                                        const T*      type_table,
                                        const int*    input_ids,
                                        const int*    input_pos,
                                        const int*    input_type,
                                        const int*    input_mask,
                                        const int     token_num,
                                        const int64_t hidden_units) {
    for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < (int64_t)(token_num * hidden_units);
         index += blockDim.x * gridDim.x) {
        const int64_t token_index = index / hidden_units;
        const int64_t col_index   = index % hidden_units;
        const int     input_id    = input_ids[token_index];
        T             embedding   = (T)0.0f;
        T             pos_embed   = (T)0.0f;
        T             type_embed  = (T)0.0f;

        if constexpr (USE_POS_EMB) {
            assert(pos_table != nullptr);
            pos_embed = pos_table[input_pos[token_index] * hidden_units + col_index];
        }
        if constexpr (USE_TYPE_ID_EMB) {
            assert(type_table != nullptr);
            type_embed = type_table[input_type[token_index] * hidden_units + col_index];
        }
        if constexpr (USE_MASK) {
            assert(input_mask != nullptr);
            if (input_mask[token_index] == 0) {
                from_tensor[index] = pos_embed + type_embed;
                continue;
            }
        }

        embedding = embedding_table[input_id * hidden_units + col_index];

        // embedding *= input_embedding_scalar;
        if constexpr (std::is_same<T, __nv_bfloat16>::value) {
            embedding *= __double2bfloat16(input_embedding_scalar);
        } else if constexpr (std::is_same<T, __half>::value) {
            embedding *= static_cast<T>(input_embedding_scalar);
        } else {
            embedding *= input_embedding_scalar;
        }

        from_tensor[index] = embedding + pos_embed + type_embed;
    }
}

#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])

template<typename VectorType, typename T, bool USE_POS_EMB, bool USE_TYPE_ID_EMB, bool USE_MASK>
__global__ void embedding_lookup_kernel_vec(T*            from_tensor,
                                            const T*      embedding_table,
                                            double        input_embedding_scalar,
                                            const T*      pos_table,
                                            const T*      type_table,
                                            const int*    input_ids,
                                            const int*    input_pos,
                                            const int*    input_type,
                                            const int*    input_mask,
                                            const int     token_num,
                                            const int64_t hidden_units) {
    const int64_t vector_size          = sizeof(VectorType) / sizeof(T);
    const int64_t aligned_hidden_units = hidden_units / vector_size;

    for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < (int64_t)(token_num * aligned_hidden_units);
         index += blockDim.x * gridDim.x) {
        const int64_t token_index = index / aligned_hidden_units;
        const int64_t col_index   = index % aligned_hidden_units;
        const int     input_id    = input_ids[token_index];

        VectorType embedding_vec = reinterpret_cast<const VectorType*>(
            &(embedding_table[input_id * hidden_units + col_index * vector_size]))[0];
        VectorType pos_embed_vec  = {.0f, .0f, .0f, .0f};
        VectorType type_embed_vec = {.0f, .0f, .0f, .0f};

        if constexpr (USE_POS_EMB) {
            assert(pos_table != nullptr);
            pos_embed_vec = LDST128BITS(pos_table[input_pos[token_index] * hidden_units + col_index * vector_size]);
        }
        if constexpr (USE_TYPE_ID_EMB) {
            assert(type_table != nullptr);
            type_embed_vec = LDST128BITS(type_table[input_pos[token_index] * hidden_units + col_index * vector_size]);
        }
        if constexpr (USE_MASK) {
            assert(input_mask != nullptr);
            if (input_mask[token_index] == 0) {
#pragma unroll
                for (int i = 0; i < vector_size; ++i) {
                    from_tensor[index * vector_size + i] =
                        reinterpret_cast<T*>(&pos_embed_vec)[i] + reinterpret_cast<T*>(&type_embed_vec)[i];
                }
                continue;
            }
        }

#pragma unroll
        for (int i = 0; i < vector_size; ++i) {
            if constexpr (std::is_same<T, __nv_bfloat16>::value) {
                reinterpret_cast<T*>(&embedding_vec)[i] *= __double2bfloat16(input_embedding_scalar);
            } else {
                reinterpret_cast<T*>(&embedding_vec)[i] *= static_cast<T>(input_embedding_scalar);
            }
            reinterpret_cast<T*>(&embedding_vec)[i] +=
                reinterpret_cast<T*>(&pos_embed_vec)[i] + reinterpret_cast<T*>(&type_embed_vec)[i];
        }

        LDST128BITS(from_tensor[index * vector_size]) = embedding_vec;
    }
}

#define INVOKE_WORD_EMBED_LOOKUP_VEC(USE_POS, USE_YPE, USE_MASK)                                                       \
    embedding_lookup_kernel_vec<float4, T, USE_POS, USE_YPE, USE_MASK>                                                 \
        <<<grid, block, 0, stream>>>(from_tensor,                                                                      \
                                     embedding_table,                                                                  \
                                     input_embedding_scalar,                                                           \
                                     pos_table,                                                                        \
                                     type_table,                                                                       \
                                     input_ids,                                                                        \
                                     input_pos,                                                                        \
                                     input_type,                                                                       \
                                     input_mask,                                                                       \
                                     token_num,                                                                        \
                                     hidden_units);

#define INVOKE_WORD_EMBED_LOOKUP(USE_POS, USE_YPE, USE_MASK)                                                           \
    embedding_lookup_kernel<T, USE_POS, USE_YPE, USE_MASK><<<grid, block, 0, stream>>>(from_tensor,                    \
                                                                                       embedding_table,                \
                                                                                       input_embedding_scalar,         \
                                                                                       pos_table,                      \
                                                                                       type_table,                     \
                                                                                       input_ids,                      \
                                                                                       input_pos,                      \
                                                                                       input_type,                     \
                                                                                       input_mask,                     \
                                                                                       token_num,                      \
                                                                                       hidden_units);

template<typename T>
void invokeEmbeddingLookupVec(T*           from_tensor,
                              const T*     embedding_table,
                              double       input_embedding_scalar,
                              const T*     pos_table,
                              const T*     type_table,
                              const int*   input_ids,
                              const int*   input_pos,
                              const int*   input_type,
                              const int*   input_mask,
                              const int    token_num,
                              const int    hidden_units,
                              cudaStream_t stream) {
    using VectorType          = float4;
    const int64_t vector_size = sizeof(VectorType) / sizeof(T);
    assert(hidden_units % vector_size == 0);
    assert(!pos_table && !type_table && !input_mask);
    dim3 grid(std::min(token_num, 65536));
    dim3 block(std::min(int(hidden_units / vector_size), 1024));
    INVOKE_WORD_EMBED_LOOKUP_VEC(false, false, false);
#if USING_CUDA
    check_cuda_value(cudaPeekAtLastError());
    check_cuda_error();
#endif
}

template<typename T>
void invokeEmbeddingLookup(T*           from_tensor,
                           const T*     embedding_table,
                           double       input_embedding_scalar,
                           const T*     pos_table,
                           const T*     type_table,
                           const int*   input_ids,
                           const int*   input_pos,
                           const int*   input_type,
                           const int*   input_mask,
                           const int    token_num,
                           const int    hidden_units,
                           cudaStream_t stream) {
    dim3 grid(std::min(token_num, 65536));
    dim3 block(std::min(hidden_units, 1024));

    printf("token_num = %lu\n", token_num);
    printf("hidden_units = %lu\n", hidden_units);

    if (!pos_table) {
        if (!type_table) {
            if (!input_mask) {
                INVOKE_WORD_EMBED_LOOKUP(false, false, false);
            } else {
                INVOKE_WORD_EMBED_LOOKUP(false, false, true);
            }
        } else {
            if (!input_mask) {
                INVOKE_WORD_EMBED_LOOKUP(false, true, false);
            } else {
                INVOKE_WORD_EMBED_LOOKUP(false, true, true);
            }
        }
    } else {
        if (!type_table) {
            if (!input_mask) {
                INVOKE_WORD_EMBED_LOOKUP(true, false, false);
            } else {
                INVOKE_WORD_EMBED_LOOKUP(true, false, true);
            }
        } else {
            if (!input_mask) {
                INVOKE_WORD_EMBED_LOOKUP(true, true, false);
            } else {
                INVOKE_WORD_EMBED_LOOKUP(true, true, true);
            }
        }
    }
#if USING_CUDA
    check_cuda_value(cudaPeekAtLastError());
    check_cuda_error();
#endif
}
#undef INVOKE_WORD_EMBED_LOOKUP

template void invokeEmbeddingLookup(float*       from_tensor,
                                    const float* embedding_table,
                                    double       input_embedding_scalar,
                                    const float* pos_table,
                                    const float* type_table,
                                    const int*   input_ids,
                                    const int*   input_pos,
                                    const int*   input_type,
                                    const int*   input_mask,
                                    const int    token_num,
                                    const int    hidden_units,
                                    cudaStream_t stream);

template void invokeEmbeddingLookup(half*        from_tensor,
                                    const half*  embedding_table,
                                    double       input_embedding_scalar,
                                    const half*  pos_table,
                                    const half*  type_table,
                                    const int*   input_ids,
                                    const int*   input_pos,
                                    const int*   input_type,
                                    const int*   input_mask,
                                    const int    token_num,
                                    const int    hidden_units,
                                    cudaStream_t stream);
#ifdef ENABLE_BF16
template void invokeEmbeddingLookup(__nv_bfloat16*       from_tensor,
                                    const __nv_bfloat16* embedding_table,
                                    double               input_embedding_scalar,
                                    const __nv_bfloat16* pos_table,
                                    const __nv_bfloat16* type_table,
                                    const int*           input_ids,
                                    const int*           input_pos,
                                    const int*           input_type,
                                    const int*           input_mask,
                                    const int            token_num,
                                    const int            hidden_units,
                                    cudaStream_t         stream);
#endif

template void invokeEmbeddingLookupVec(float*       from_tensor,
                                       const float* embedding_table,
                                       double       input_embedding_scalar,
                                       const float* pos_table,
                                       const float* type_table,
                                       const int*   input_ids,
                                       const int*   input_pos,
                                       const int*   input_type,
                                       const int*   input_mask,
                                       const int    token_num,
                                       const int    hidden_units,
                                       cudaStream_t stream);

template void invokeEmbeddingLookupVec(half*        from_tensor,
                                       const half*  embedding_table,
                                       double       input_embedding_scalar,
                                       const half*  pos_table,
                                       const half*  type_table,
                                       const int*   input_ids,
                                       const int*   input_pos,
                                       const int*   input_type,
                                       const int*   input_mask,
                                       const int    token_num,
                                       const int    hidden_units,
                                       cudaStream_t stream);

#ifdef ENABLE_BF16
template void invokeEmbeddingLookupVec(__nv_bfloat16*       from_tensor,
                                       const __nv_bfloat16* embedding_table,
                                       double               input_embedding_scalar,
                                       const __nv_bfloat16* pos_table,
                                       const __nv_bfloat16* type_table,
                                       const int*           input_ids,
                                       const int*           input_pos,
                                       const int*           input_type,
                                       const int*           input_mask,
                                       const int            token_num,
                                       const int            hidden_units,
                                       cudaStream_t         stream);
#endif

}  // namespace rtp_llm