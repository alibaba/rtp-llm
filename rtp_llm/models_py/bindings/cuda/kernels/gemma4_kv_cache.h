#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstdint>

namespace rtp_llm {

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
                                   cudaStream_t         stream);

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
                                      cudaStream_t         stream);

}  // namespace rtp_llm
