#pragma once

#include <torch/all.h>

namespace torch_ext {

// Per-row TopK for the prefill phase of the DeepSeek-V4 indexer.
//
// Vendored from vLLM (`csrc/sampler.cu::top_k_per_row_prefill`).  Hybrid
// strategy:
//   * the first ``kSortingAlgorithmThreshold = 12288`` rows are processed
//     with insertion-sort blocks (small final pass)
//   * any remaining rows are processed with radix-sort blocks
//
// Contract (matches the vLLM op signature):
//   logits      : [num_rows, max_T] float32; only ``[row_starts[r],
//                 row_ends[r])`` is read along the column dim
//   row_starts  : [num_rows]        int32   — inclusive row begin
//   row_ends    : [num_rows]        int32   — exclusive row end
//   indices_out : [num_rows, top_k] int32   — written; positions past the
//                 row's valid count are ``-1`` padded.  Indices are
//                 relative to ``row_starts[r]`` (i.e. 0..rowEnd-rowStart-1).
//   num_rows    : == logits.size(0)
//   stride0     : logits.stride(0)
//   stride1     : logits.stride(1)
//   top_k       : K
//
// CUDA-only.
void dsv4_top_k_per_row_prefill(const torch::Tensor& logits,
                                const torch::Tensor& row_starts,
                                const torch::Tensor& row_ends,
                                torch::Tensor&       indices_out,
                                int64_t              num_rows,
                                int64_t              stride0,
                                int64_t              stride1,
                                int64_t              top_k);

}  // namespace torch_ext
