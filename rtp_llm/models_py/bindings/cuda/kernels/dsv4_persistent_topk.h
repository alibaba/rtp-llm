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
//   max_seq_len: host-known upper bound on the per-row ``lengths`` values
//                (must satisfy max_seq_len >= max(lengths)). Drives the
//                launch geometry — CTAs per row group, chunk size, shared
//                memory, grid — and the cooperative-launch threshold; row
//                addressing always uses the real ``logits.stride(0)``.
//                Callers without a host-known bound (varlen edge) pass the
//                full row width, which reproduces the legacy full-width
//                launch. Decode callers bucket the real max row length up
//                to a power of two so the grid is chosen from a small fixed
//                ladder and stays CUDA-graph-capture stable.
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
