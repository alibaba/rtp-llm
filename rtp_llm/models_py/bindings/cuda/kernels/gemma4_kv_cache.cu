#include "rtp_llm/models_py/bindings/cuda/kernels/gemma4_kv_cache.h"

#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

namespace rtp_llm {
namespace {

constexpr int kBf16PerVector = sizeof(uint4) / sizeof(__nv_bfloat16);

__global__ void gemma4GatherPagedKvBf16Kernel(const __nv_bfloat16* k_cache,
                                              const __nv_bfloat16* v_cache,
                                              const int32_t* __restrict__ page_indices,
                                              uint4* __restrict__ keys,
                                              uint4* __restrict__ values,
                                              bool* __restrict__ valid,
                                              int64_t vector_count,
                                              int32_t vectors_per_head,
                                              int32_t num_heads,
                                              int32_t page_size,
                                              int32_t first_offset,
                                              int64_t num_pages,
                                              int64_t page_indices_size,
                                              int64_t cache_page_stride,
                                              int64_t cache_head_stride,
                                              int64_t cache_token_stride) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= vector_count) {
        return;
    }
    const int32_t vector_in_head   = static_cast<int32_t>(index % vectors_per_head);
    const int32_t head             = static_cast<int32_t>((index / vectors_per_head) % num_heads);
    const int64_t token            = index / (static_cast<int64_t>(vectors_per_head) * num_heads);
    const int64_t page_offset      = static_cast<int64_t>(first_offset) + token;
    const int64_t page_slot        = page_offset / page_size;
    const int32_t position_in_page = static_cast<int32_t>(page_offset % page_size);
    const int32_t page             = page_slot < page_indices_size ? page_indices[page_slot] : -1;
    const bool    page_valid       = page >= 0 && page < num_pages;
    if (head == 0 && vector_in_head == 0) {
        valid[token] = page_valid;
    }
    if (!page_valid) {
        keys[index]   = make_uint4(0, 0, 0, 0);
        values[index] = make_uint4(0, 0, 0, 0);
        return;
    }
    const int64_t cache_offset = static_cast<int64_t>(page) * cache_page_stride
                                 + static_cast<int64_t>(head) * cache_head_stride
                                 + static_cast<int64_t>(position_in_page) * cache_token_stride
                                 + static_cast<int64_t>(vector_in_head) * kBf16PerVector;
    keys[index]   = *reinterpret_cast<const uint4*>(k_cache + cache_offset);
    values[index] = *reinterpret_cast<const uint4*>(v_cache + cache_offset);
}

__global__ void gemma4AppendSwaKvCacheBf16Kernel(const uint4* __restrict__ key,
                                                 const uint4* __restrict__ value,
                                                 const int32_t* __restrict__ batch_indices,
                                                 const int32_t* __restrict__ positions,
                                                 __nv_bfloat16* k_cache,
                                                 __nv_bfloat16* v_cache,
                                                 const int32_t* __restrict__ page_indices,
                                                 const int32_t* __restrict__ page_indptr,
                                                 int64_t vector_count,
                                                 int32_t vectors_per_head,
                                                 int32_t num_heads,
                                                 int32_t page_size,
                                                 int64_t num_pages,
                                                 int64_t page_indices_size,
                                                 int64_t page_indptr_size,
                                                 int64_t cache_page_stride,
                                                 int64_t cache_head_stride,
                                                 int64_t cache_token_stride) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= vector_count) {
        return;
    }
    const int32_t vector_in_head = static_cast<int32_t>(index % vectors_per_head);
    const int32_t head           = static_cast<int32_t>((index / vectors_per_head) % num_heads);
    const int64_t token          = index / (static_cast<int64_t>(vectors_per_head) * num_heads);
    const int32_t batch          = batch_indices[token];
    const int32_t position       = positions[token];
    if (batch < 0 || static_cast<int64_t>(batch) + 1 >= page_indptr_size || position < 0) {
        return;
    }
    const int64_t slot = static_cast<int64_t>(page_indptr[batch]) + position / page_size;
    if (slot < 0 || slot >= page_indices_size) {
        return;
    }
    const int32_t page = page_indices[slot];
    if (page < 0 || page >= num_pages) {
        return;
    }
    const int32_t position_in_page = position % page_size;
    const int64_t cache_offset     = static_cast<int64_t>(page) * cache_page_stride
                                 + static_cast<int64_t>(head) * cache_head_stride
                                 + static_cast<int64_t>(position_in_page) * cache_token_stride
                                 + static_cast<int64_t>(vector_in_head) * kBf16PerVector;
    *reinterpret_cast<uint4*>(k_cache + cache_offset) = key[index];
    *reinterpret_cast<uint4*>(v_cache + cache_offset) = value[index];
}

}  // namespace

void invokeGemma4GatherPagedKvBf16(const __nv_bfloat16* k_cache,
                                   const __nv_bfloat16* v_cache,
                                   const int32_t*       page_indices,
                                   __nv_bfloat16*       keys,
                                   __nv_bfloat16*       values,
                                   bool*                valid,
                                   int64_t              token_count,
                                   int32_t              first_offset,
                                   int32_t              num_heads,
                                   int32_t              head_dim,
                                   int32_t              page_size,
                                   int64_t              num_pages,
                                   int64_t              page_indices_size,
                                   int64_t              cache_page_stride,
                                   int64_t              cache_head_stride,
                                   int64_t              cache_token_stride,
                                   cudaStream_t         stream) {
    if (token_count == 0) {
        return;
    }
    constexpr int threads          = 256;
    const int32_t vectors_per_head = head_dim / kBf16PerVector;
    const int64_t vector_count     = token_count * num_heads * vectors_per_head;
    const int     blocks           = static_cast<int>((vector_count + threads - 1) / threads);
    gemma4GatherPagedKvBf16Kernel<<<blocks, threads, 0, stream>>>(k_cache,
                                                                  v_cache,
                                                                  page_indices,
                                                                  reinterpret_cast<uint4*>(keys),
                                                                  reinterpret_cast<uint4*>(values),
                                                                  valid,
                                                                  vector_count,
                                                                  vectors_per_head,
                                                                  num_heads,
                                                                  page_size,
                                                                  first_offset,
                                                                  num_pages,
                                                                  page_indices_size,
                                                                  cache_page_stride,
                                                                  cache_head_stride,
                                                                  cache_token_stride);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void invokeGemma4AppendSwaKvCacheBf16(const __nv_bfloat16* key,
                                      const __nv_bfloat16* value,
                                      const int32_t*       batch_indices,
                                      const int32_t*       positions,
                                      __nv_bfloat16*       k_cache,
                                      __nv_bfloat16*       v_cache,
                                      const int32_t*       page_indices,
                                      const int32_t*       page_indptr,
                                      int64_t              tokens,
                                      int32_t              num_heads,
                                      int32_t              head_dim,
                                      int32_t              page_size,
                                      int64_t              num_pages,
                                      int64_t              page_indices_size,
                                      int64_t              page_indptr_size,
                                      int64_t              cache_page_stride,
                                      int64_t              cache_head_stride,
                                      int64_t              cache_token_stride,
                                      cudaStream_t         stream) {
    if (tokens == 0) {
        return;
    }
    constexpr int threads          = 256;
    const int32_t vectors_per_head = head_dim / kBf16PerVector;
    const int64_t vector_count     = tokens * num_heads * vectors_per_head;
    const int     blocks           = static_cast<int>((vector_count + threads - 1) / threads);
    gemma4AppendSwaKvCacheBf16Kernel<<<blocks, threads, 0, stream>>>(reinterpret_cast<const uint4*>(key),
                                                                     reinterpret_cast<const uint4*>(value),
                                                                     batch_indices,
                                                                     positions,
                                                                     k_cache,
                                                                     v_cache,
                                                                     page_indices,
                                                                     page_indptr,
                                                                     vector_count,
                                                                     vectors_per_head,
                                                                     num_heads,
                                                                     page_size,
                                                                     num_pages,
                                                                     page_indices_size,
                                                                     page_indptr_size,
                                                                     cache_page_stride,
                                                                     cache_head_stride,
                                                                     cache_token_stride);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace rtp_llm
