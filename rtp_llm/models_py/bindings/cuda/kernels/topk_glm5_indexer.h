#pragma once

#include <torch/all.h>

namespace torch_ext {

// Persistent radix-select TopK for the GLM5 sparse-attention indexer.
//
// Adapted from the exact SGLang radix-select implementation in commit
// e7b190c165b2edaa92bbc. It includes the FP32 boundary fallback required when
// the coarse threshold bin overflows the fixed candidate buffer.
//
// Contract:
//   logits   : [num_rows, stride] float32, contiguous in stride
//   lengths  : [num_rows]         int32   — per-row valid count;
//                                            entries past lengths[r]
//                                            are written as -1 in output
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
void topk_glm5_indexer(const torch::Tensor& logits,
                       const torch::Tensor& lengths,
                       torch::Tensor& output,
                       torch::Tensor& workspace,
                       int64_t k,
                       int64_t max_seq_len);

// Exact SGLang TopK continuation for paged MQA that already produced one
// 1024-bin ordered-FP16 histogram per row. It skips SGLang's histogram-building
// logits pass, but retains the original 1024-thread collection and exact FP32
// tie fallback.
void topk_glm5_indexer_from_histogram(
    const torch::Tensor& logits,
    const torch::Tensor& lengths,
    const torch::Tensor& histograms,
    torch::Tensor& output,
    torch::Tensor& workspace,
    int64_t k,
    int64_t max_seq_len);

}  // namespace torch_ext
