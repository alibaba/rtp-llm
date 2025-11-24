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
#include "rtp_llm/cpp/kernels/kv_cache_kernels.h"

namespace rtp_llm {

// input: offset [b, m](kv_cache_block_id):
// ┌─────────────────┐
// │ batch0: [0,1,2] │
// │ batch1: [3,4,5] │
// │ batch2: [6,7,8] │
// └─────────────────┘
// flatten to (k,v block offset) -> k block id and v block id:
// output: offset_addr [b, 2, m](kv_block_array.data, kv_block_array.mPrimaryPool is the real kv cache data address):
// ┌─────────────────────────────────────────---┐
// │ batch0_K: [0,1,2]  batch0_V: [0+Δ,1+Δ,2+Δ] │
// │ batch1_K: [3,4,5]  batch1_V: [3+Δ,4+Δ,5+Δ] │
// │ batch2_K: [6,7,8]  batch2_V: [6+Δ,7+Δ,8+Δ] │
// └─────────────────────────────────────────---┘
// Δ is kv_block_offset (layer_num * block_num_per_layer).

__global__ void ConvertOffsetToBlockArrayData(int32_t*   offset_addr,
                                              const int* offset,  // [b, m]
                                              int        batch_size,
                                              int        max_block_num) {
    const int batch_stride = 2 * max_block_num;
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < batch_size * max_block_num;
         index += blockDim.x * gridDim.x) {
        const int     batch_index                     = index / max_block_num;
        const int     col_index                       = index % max_block_num;
        const int32_t block_offset                    = (int32_t)offset[batch_index * max_block_num + col_index];
        const int32_t block_addr_index                = (int32_t)batch_index * batch_stride + col_index;
        offset_addr[block_addr_index]                 = block_offset * 2;
        offset_addr[block_addr_index + max_block_num] = block_offset * 2 + 1;
    }
}

void invokeConvertOffsetToBlockArrayData(int32_t*     offset_addr,  // [b, 2, m]
                                         const int*   offset,       // [b, m]
                                         int          batch_size,
                                         int          max_block_num,
                                         cudaStream_t stream) {
    dim3 grid(min(batch_size, 65536));
    dim3 block(min(max_block_num, 1024));
    ConvertOffsetToBlockArrayData<<<grid, block, 0, stream>>>(offset_addr,  // [b, 2, m]
                                                              offset,       // [b, m]
                                                              batch_size,
                                                              max_block_num);
#if USING_CUDA
    check_cuda_value(cudaPeekAtLastError());
    check_cuda_error();
#endif
}

template<typename T>
__global__ void ReuseKVCacheIndexedBatchedKernel(T*             final_compressed_kv,
                                                 T*             final_k_pe,
                                                 const T*       compressed_kv,
                                                 const T*       k_pe,
                                                 const T*       kv_cache_base,
                                                 const int32_t* reuse_cache_page_indice,
                                                 const int32_t* batch_reuse_info_vec,
                                                 const int32_t* qo_indptr,
                                                 int            num_batches,
                                                 int            total_final_len,
                                                 int            compressed_kv_dim,
                                                 int            k_pe_dim,
                                                 int            tokens_per_block,
                                                 int            kv_dim) {

    // 优化1: 使用 shared memory 缓存 batch 信息（对于小 batch 数量）
    __shared__ int32_t s_batch_info[64 * 4];  // 最多支持 64 个 batch
    __shared__ int32_t s_qo_indptr[65];       // 最多支持 64 个 batch + 1

    // 关键修复：在检查 tid >= total_final_len 之前，先完成 shared memory 的加载
    // 因为 shared memory 的加载只需要 num_batches + 1 个线程（最多 65 个），
    // 不依赖于所有线程都执行，所以应该在所有线程检查之前完成
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // 边界检查：如果 num_batches = 0，直接返回（不应该发生，但添加保护）
    if (num_batches <= 0) {
        return;
    }

    // 边界检查：tokens_per_block 不能为 0
    if (tokens_per_block <= 0) {
        return;
    }

    // 每个 block 都初始化自己的 shared memory（如果 num_batches <= 64）
    if (num_batches <= 64) {
        // 初始化 shared memory（确保所有元素都被初始化）
        // 注意：shared memory 在声明时不会自动初始化为 0，需要显式初始化
        if (threadIdx.x < 65) {
            s_qo_indptr[threadIdx.x] = 0;  // 初始化为 0
        }
        __syncthreads();

        // 加载 batch_info（需要 num_batches 个线程，但限制在 64 以内）
        if (threadIdx.x < num_batches && threadIdx.x < 64) {
            const int32_t* batch_info = batch_reuse_info_vec + threadIdx.x * 4;
#pragma unroll
            for (int i = 0; i < 4; i++) {
                s_batch_info[threadIdx.x * 4 + i] = batch_info[i];
            }
        }

        // 加载 qo_indptr，需要 num_batches + 1 个元素（索引 0 到 num_batches）
        // 关键修复：不检查 tid，只检查 threadIdx.x，确保所有需要的线程都加载
        // 即使 tid >= total_final_len，这些线程也需要加载 shared memory 供其他线程使用
        if (threadIdx.x <= num_batches) {
            s_qo_indptr[threadIdx.x] = qo_indptr[threadIdx.x];
        }
    }
    __syncthreads();

    // 现在检查 tid >= total_final_len，在 shared memory 加载完成之后
    if (tid >= total_final_len) {
        return;
    }

    // 优化2: 使用二分搜索或优化的线性搜索
    int batch_idx    = -1;  // 初始化为 -1 表示未找到
    int final_offset = 0;

    // 对于小 batch 数量，使用 shared memory 中的数据进行搜索
    if (num_batches <= 64) {
        for (int i = 0; i < num_batches; i++) {
            const int reuse_len = s_batch_info[i * 4 + 1];
            // qo_indptr 应该有 num_batches + 1 个元素，所以 i + 1 是安全的（由调用方保证）
            // 但为了防御性编程，我们仍然检查（虽然理论上不应该发生）
            const int batch_q_len     = s_qo_indptr[i + 1] - s_qo_indptr[i];
            const int batch_final_len = reuse_len + batch_q_len;

            if (tid < final_offset + batch_final_len) {
                batch_idx = i;
                break;
            }
            final_offset += batch_final_len;
        }
    } else {
        // 对于大 batch 数量，使用全局内存（原始方法）
        for (int i = 0; i < num_batches; i++) {
            const int32_t* batch_info = batch_reuse_info_vec + i * 4;
            const int      reuse_len  = batch_info[1];
            // qo_indptr 应该有 num_batches + 1 个元素，所以 i + 1 是安全的（由调用方保证）
            const int batch_q_len     = qo_indptr[i + 1] - qo_indptr[i];
            const int batch_final_len = reuse_len + batch_q_len;

            if (tid < final_offset + batch_final_len) {
                batch_idx = i;
                break;
            }
            final_offset += batch_final_len;
        }
    }

    // 边界检查：如果未找到对应的 batch，直接返回
    if (batch_idx < 0 || batch_idx >= num_batches) {
        return;
    }

    // 获取当前batch的信息
    const int reuse_len =
        (num_batches <= 64) ? s_batch_info[batch_idx * 4 + 1] : batch_reuse_info_vec[batch_idx * 4 + 1];
    const int block_start_idx =
        (num_batches <= 64) ? s_batch_info[batch_idx * 4 + 2] : batch_reuse_info_vec[batch_idx * 4 + 2];
    const int local_idx            = tid - final_offset;
    const int compressed_kv_offset = (num_batches <= 64) ? s_qo_indptr[batch_idx] : qo_indptr[batch_idx];

    // 优化3: 向量化内存访问
    using VecType = typename std::conditional<
        sizeof(T) == 2,
        typename std::conditional<std::is_same<T, __half>::value, half2, __nv_bfloat162>::type,
        typename std::conditional<sizeof(T) == 4, float2, T>::type>::type;

    constexpr int  vec_size = sizeof(VecType) / sizeof(T);
    constexpr bool use_vec  = (sizeof(T) <= 4) && (vec_size > 1);

    // 边界检查：确保 reuse_len 非负
    if (local_idx < reuse_len && reuse_len > 0) {
        // 从reuse cache复制
        const int reuse_local_idx = local_idx;
        const int block_idx       = reuse_local_idx / tokens_per_block;
        const int token_in_block  = reuse_local_idx % tokens_per_block;
        // 边界检查：确保 block_start_idx + block_idx 不会越界
        // 注意：这里假设 block_start_idx 和 block_idx 都是有效的
        const int reuse_cache_index = block_start_idx + block_idx;
        // 注意：这里没有 reuse_cache_page_indice 的长度信息，所以无法检查越界
        // 如果越界会导致未定义行为，但添加日志可以帮助调试
        const int cache_block_idx = reuse_cache_page_indice[reuse_cache_index];

        const T* cache_block = kv_cache_base + cache_block_idx * tokens_per_block * kv_dim;
        const T* cache_token = cache_block + token_in_block * kv_dim;

        // 向量化复制 compressed_kv 部分
        T* dst_compressed = final_compressed_kv + tid * compressed_kv_dim;
        if (use_vec && compressed_kv_dim % vec_size == 0
            && reinterpret_cast<uintptr_t>(dst_compressed) % sizeof(VecType) == 0
            && reinterpret_cast<uintptr_t>(cache_token) % sizeof(VecType) == 0) {
            const int      vec_count = compressed_kv_dim / vec_size;
            VecType*       dst_vec   = reinterpret_cast<VecType*>(dst_compressed);
            const VecType* src_vec   = reinterpret_cast<const VecType*>(cache_token);
            // 修复：完整复制所有数据，对于小循环使用 unroll，大循环使用普通循环
            if (vec_count <= 32) {
#pragma unroll
                for (int i = 0; i < vec_count; i++) {
                    dst_vec[i] = src_vec[i];
                }
            } else {
                for (int i = 0; i < vec_count; i++) {
                    dst_vec[i] = src_vec[i];
                }
            }
        } else {
#pragma unroll 4
            for (int i = 0; i < compressed_kv_dim; i++) {
                dst_compressed[i] = cache_token[i];
            }
        }

        // 向量化复制 k_pe 部分
        T*       dst_k_pe = final_k_pe + tid * k_pe_dim;
        const T* src_k_pe = cache_token + compressed_kv_dim;
        if (use_vec && k_pe_dim % vec_size == 0 && reinterpret_cast<uintptr_t>(dst_k_pe) % sizeof(VecType) == 0
            && reinterpret_cast<uintptr_t>(src_k_pe) % sizeof(VecType) == 0) {
            const int      vec_count = k_pe_dim / vec_size;
            VecType*       dst_vec   = reinterpret_cast<VecType*>(dst_k_pe);
            const VecType* src_vec   = reinterpret_cast<const VecType*>(src_k_pe);
            // 修复：完整复制所有数据
            if (vec_count <= 32) {
#pragma unroll
                for (int i = 0; i < vec_count; i++) {
                    dst_vec[i] = src_vec[i];
                }
            } else {
                for (int i = 0; i < vec_count; i++) {
                    dst_vec[i] = src_vec[i];
                }
            }
        } else {
#pragma unroll 4
            for (int i = 0; i < k_pe_dim; i++) {
                dst_k_pe[i] = src_k_pe[i];
            }
        }
    } else {
        // 从compressed_kv/k_pe复制
        // 边界检查：确保 local_idx >= reuse_len（防御性检查）
        if (local_idx < reuse_len) {
            return;  // 不应该发生，但添加防御性检查
        }
        const int compressed_kv_local_idx  = local_idx - reuse_len;
        const int compressed_kv_global_idx = compressed_kv_offset + compressed_kv_local_idx;

        // 向量化复制 compressed_kv
        T*       dst_compressed = final_compressed_kv + tid * compressed_kv_dim;
        const T* src_compressed = compressed_kv + compressed_kv_global_idx * compressed_kv_dim;
        if (use_vec && compressed_kv_dim % vec_size == 0
            && reinterpret_cast<uintptr_t>(dst_compressed) % sizeof(VecType) == 0
            && reinterpret_cast<uintptr_t>(src_compressed) % sizeof(VecType) == 0) {
            const int      vec_count = compressed_kv_dim / vec_size;
            VecType*       dst_vec   = reinterpret_cast<VecType*>(dst_compressed);
            const VecType* src_vec   = reinterpret_cast<const VecType*>(src_compressed);
            // 修复：完整复制所有数据
            if (vec_count <= 32) {
#pragma unroll
                for (int i = 0; i < vec_count; i++) {
                    dst_vec[i] = src_vec[i];
                }
            } else {
                for (int i = 0; i < vec_count; i++) {
                    dst_vec[i] = src_vec[i];
                }
            }
        } else {
#pragma unroll 4
            for (int i = 0; i < compressed_kv_dim; i++) {
                dst_compressed[i] = src_compressed[i];
            }
        }

        // 向量化复制 k_pe
        T*       dst_k_pe = final_k_pe + tid * k_pe_dim;
        const T* src_k_pe = k_pe + compressed_kv_global_idx * k_pe_dim;
        if (use_vec && k_pe_dim % vec_size == 0 && reinterpret_cast<uintptr_t>(dst_k_pe) % sizeof(VecType) == 0
            && reinterpret_cast<uintptr_t>(src_k_pe) % sizeof(VecType) == 0) {
            const int      vec_count = k_pe_dim / vec_size;
            VecType*       dst_vec   = reinterpret_cast<VecType*>(dst_k_pe);
            const VecType* src_vec   = reinterpret_cast<const VecType*>(src_k_pe);
            // 修复：完整复制所有数据
            if (vec_count <= 32) {
#pragma unroll
                for (int i = 0; i < vec_count; i++) {
                    dst_vec[i] = src_vec[i];
                }
            } else {
                for (int i = 0; i < vec_count; i++) {
                    dst_vec[i] = src_vec[i];
                }
            }
        } else {
#pragma unroll 4
            for (int i = 0; i < k_pe_dim; i++) {
                dst_k_pe[i] = src_k_pe[i];
            }
        }
    }
}

template<typename T>
void invokeReuseKVCacheIndexedBatched(T*             final_compressed_kv,
                                      T*             final_k_pe,
                                      const T*       compressed_kv,
                                      const T*       k_pe,
                                      const T*       kv_cache_base,
                                      const int32_t* reuse_cache_page_indice,
                                      const int32_t* batch_reuse_info_vec,
                                      const int32_t* qo_indptr,
                                      int            num_batches,
                                      int            total_final_len,
                                      int            compressed_kv_dim,
                                      int            k_pe_dim,
                                      int            tokens_per_block,
                                      int            kv_dim,
                                      cudaStream_t   stream) {

    if (total_final_len == 0) {
        return;
    }

    // 优化4: 动态调整 block size
    int       block_size = 256;
    const int total_dim  = compressed_kv_dim + k_pe_dim;
    if (total_dim >= 256) {
        block_size = 128;  // 大维度使用较小 block size
    } else if (total_dim <= 64) {
        block_size = 512;  // 小维度可以使用更大 block size
    }

    const int grid_size = (total_final_len + block_size - 1) / block_size;

    ReuseKVCacheIndexedBatchedKernel<<<grid_size, block_size, 0, stream>>>(final_compressed_kv,
                                                                           final_k_pe,
                                                                           compressed_kv,
                                                                           k_pe,
                                                                           kv_cache_base,
                                                                           reuse_cache_page_indice,
                                                                           batch_reuse_info_vec,
                                                                           qo_indptr,
                                                                           num_batches,
                                                                           total_final_len,
                                                                           compressed_kv_dim,
                                                                           k_pe_dim,
                                                                           tokens_per_block,
                                                                           kv_dim);

#if USING_CUDA
    check_cuda_value(cudaPeekAtLastError());
    check_cuda_error();
#endif
}

// Explicit template instantiation
#define INSTANTIATE_REUSE_KV_CACHE_INDEXED_BATCHED(T)                                                                  \
    template void invokeReuseKVCacheIndexedBatched<T>(T*,                                                              \
                                                      T*,                                                              \
                                                      const T*,                                                        \
                                                      const T*,                                                        \
                                                      const T*,                                                        \
                                                      const int32_t*,                                                  \
                                                      const int32_t*,                                                  \
                                                      const int32_t*,                                                  \
                                                      int,                                                             \
                                                      int,                                                             \
                                                      int,                                                             \
                                                      int,                                                             \
                                                      int,                                                             \
                                                      int,                                                             \
                                                      cudaStream_t);

#if USING_CUDA
INSTANTIATE_REUSE_KV_CACHE_INDEXED_BATCHED(__nv_bfloat16)
#endif

#undef INSTANTIATE_REUSE_KV_CACHE_INDEXED_BATCHED

}  // namespace rtp_llm