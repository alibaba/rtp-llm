#pragma once

#include <ATen/ATen.h>
#include <cuda_runtime_api.h>

namespace rtp_llm {

// Fill (paged_kv_last_page_len, decode_page_indptr, page_indice, batch_indice,
// positions) directly on the GPU for the MHA paged-attention path. Inputs are
// device tensors; outputs must be pre-allocated device tensors with sufficient
// capacity. The kernel intentionally fills only the fields the MHA wrappers
// (FlashInfer plan/append + MHA RoPE) actually consume; MLA-only fields
// (reuse_cache, batch_reuse_info, qo_indptr, ...) are not touched.
//
// Required sizes (caller responsibility):
//   paged_kv_last_page_len   >= batch_size
//   decode_page_indptr       >= batch_size + 1
//   page_indice              >= batch_size * max_blocks_per_bs (loose upper bound)
//   batch_indice             >= sum(input_lengths) for prefill, or batch_size for decode
//   positions                >= same as batch_indice
//
// One CTA, batch_size threads — designed for the small batches typical of
// decode/prefill in this engine (<= 1024).
void invokeMhaPagedAttnPlan(const at::Tensor& input_lengths,
                            const at::Tensor& sequence_lengths,
                            const at::Tensor& prefix_lengths,
                            const at::Tensor& kv_cache_block_id,
                            int               seq_size_per_block,
                            at::Tensor&       paged_kv_last_page_len,
                            at::Tensor&       decode_page_indptr,
                            at::Tensor&       page_indice,
                            at::Tensor&       batch_indice,
                            at::Tensor&       positions,
                            cudaStream_t      stream);

// Sliding-window variant. A windowed cache group keeps a full-length block table addressed
// by absolute page, but materializes only the active tail and leaves NULL_BLOCK_IDX (-1) in
// the older slots. FlashInfer dereferences page ids unconditionally, so a negative one
// reads before the pool; this variant redirects those slots to reserved block 0. Every NULL
// slot lies wholly outside the window, so with window_left set whatever block 0 holds is
// masked out of the softmax.
//
// Same contract as invokeMhaPagedAttnPlan otherwise. Do not use it for full-attention
// groups: there a negative page id is a bug, and silently rewriting it would hide it.
void invokeSwaMhaPagedAttnPlan(const at::Tensor& input_lengths,
                               const at::Tensor& sequence_lengths,
                               const at::Tensor& prefix_lengths,
                               const at::Tensor& kv_cache_block_id,
                               int               seq_size_per_block,
                               at::Tensor&       paged_kv_last_page_len,
                               at::Tensor&       decode_page_indptr,
                               at::Tensor&       page_indice,
                               at::Tensor&       batch_indice,
                               at::Tensor&       positions,
                               cudaStream_t      stream);

}  // namespace rtp_llm
