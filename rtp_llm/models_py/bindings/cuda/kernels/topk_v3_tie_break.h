#pragma once

#include <torch/all.h>

namespace torch_ext {

// Stable per-row radix-select TopK for GLM5 indexer prefill.
//
// Each row reads scores[row, row_starts[row]:row_ends[row]]. Returned indices
// are relative to row_starts[row]. Equal threshold scores are resolved by the
// smallest relative index. The implementation uses exactly one CTA per row
// and dispatches only to register-resident or streaming implementations.
void topk_v3_tie_break(const torch::Tensor& scores,
                       const torch::Tensor& row_starts,
                       const torch::Tensor& row_ends,
                       torch::Tensor&       output,
                       int64_t              k,
                       int64_t              max_seq_len);

}  // namespace torch_ext
