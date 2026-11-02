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
//   logits   : [num_rows, stride] float32, contiguous in stride
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
void topk_v3(const torch::Tensor& logits,
             const torch::Tensor& lengths,
             torch::Tensor& output,
             torch::Tensor& workspace,
             int64_t k,
             int64_t max_seq_len);

// Repair a native float TopK's boundary ties using true global positions.
// Scores have already been sanitized to -inf for every invalid candidate.
// Valid global positions must be unique within each row. Output is unordered;
// the caller canonicalizes only the bounded K candidates, never the history.
void dsa_repair_topk_candidates(const torch::Tensor& scores,
                                const torch::Tensor& positions,
                                const torch::Tensor& raw_indices,
                                torch::Tensor& output_scores,
                                torch::Tensor& output_positions);

// Unordered score/global-position pairs. Bounds exclude unread MQA columns;
// holes, non-finite values, signed zero and ties are handled inside selection.
void dsa_select_topk_candidates(torch::Tensor scores, torch::Tensor positions,
    c10::optional<torch::Tensor> starts, c10::optional<torch::Tensor> ends,
    torch::Tensor output_scores, torch::Tensor output_positions, int64_t topk);

}  // namespace torch_ext
