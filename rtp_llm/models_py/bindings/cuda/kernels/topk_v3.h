#pragma once

#include <torch/all.h>

namespace torch_ext {

// Persistent radix-select TopK for the DeepSeek V4 sparse-attention indexer.
//
// Adapted from the exact SGLang radix-select implementation in commit
// e7b190c165b2edaa92bbc. It includes the FP32 boundary fallback required when
// the coarse threshold bin overflows the fixed candidate buffer.
//
// Contract:
//   logits   : [num_rows, stride] float32/bfloat16, contiguous in stride
//   lengths  : [num_rows]         int32   — per-row valid count, clamped on
//                                            device to [0, max_seq_len]; entries
//                                            past the clamped length are -1
//   output   : [num_rows, k]      int32   — written
//   workspace: contiguous CUDA uint8 buffer; retained for ABI compatibility
//              with dsv4_persistent_topk (the SGLang kernel does not use it)
//   k        : 512, 1024, or 2048 (compile-time dispatched)
//   max_seq_len: max valid sequence length represented by this launch; selects
//                the register, streaming, or CTA-cluster implementation
//
// Notes:
//   * Only CUDA — ROCm path raises.
//   * ``num_rows`` corresponds to ``B * S`` after flattening the leading
//     dims of the indexer ``score [B, S, T_max]`` tensor.
//   * BF16 values are promoted exactly to FP32 in registers; selection matches
//     an explicit logits.float() with no full-sized FP32 temporary.
void topk_v3(const torch::Tensor& logits,
             const torch::Tensor& lengths,
             torch::Tensor&       output,
             torch::Tensor&       workspace,
             int64_t              k,
             int64_t              max_seq_len);

// Same selection, then replace selected NaN/+inf/-inf entries with -1. Does
// not refill filtered slots; this includes rows with length <= k.
void dsv41_topk_v3_finite(const torch::Tensor& logits,
                          const torch::Tensor& lengths,
                          torch::Tensor&       output,
                          torch::Tensor&       workspace,
                          int64_t              k,
                          int64_t              max_seq_len);

}  // namespace torch_ext
