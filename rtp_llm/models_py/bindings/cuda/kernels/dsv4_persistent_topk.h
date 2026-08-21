#pragma once

#include <torch/all.h>

namespace torch_ext {

// Persistent radix-select TopK for the DeepSeek-V4 sparse-attention indexer.
//
// Vendored from vLLM (`csrc/persistent_topk.cuh` + `csrc/topk.cu`,
// commit ``b55d830``).  Replaces ``torch.topk`` on the indexer decode hot
// path: at K=512 / T_max=2048 torch.topk burns ~30us; this kernel runs in
// ~5-10us and writes -1 padding past per-row ``lengths`` directly into
// the output buffer (fold of the previous fill_/copy_/masked_fill_ chain).
//
// Contract:
//   logits   : [num_rows, stride] float32, contiguous in stride
//   lengths  : [num_rows]         int32   — per-row valid count;
//                                            entries past lengths[r]
//                                            are written as -1 in output
//   output   : [num_rows, k]      int32   — written
//   workspace: uint8 buffer, ≥ ``RADIX_TOPK_WORKSPACE_SIZE = 1MB``
//   k        : 512, 1024, or 2048 (compile-time dispatched)
//   max_seq_len: max possible stride across rows; controls cooperative
//                launch threshold inside the kernel
//
// Notes:
//   * Only CUDA — ROCm path raises.
//   * ``num_rows`` corresponds to ``B * S`` after flattening the leading
//     dims of the indexer ``score [B, S, T_max]`` tensor.
void dsv4_persistent_topk(const torch::Tensor& logits,
                          const torch::Tensor& lengths,
                          torch::Tensor&       output,
                          torch::Tensor&       workspace,
                          int64_t              k,
                          int64_t              max_seq_len);

}  // namespace torch_ext
