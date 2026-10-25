#pragma once

#include <torch/extension.h>

namespace rtp_llm {

// TP2/4/8/16, rank-contiguous row shards. Symmetric storage, multicast mappings and
// zeroed protocol state are prepared collectively before use and retained
// through CUDA Graph replay. Calls sharing state must be serialized in the
// same order on every rank, with identical shapes and launch configurations.
// Input/output/workspace/protocol state must not overlap. Multicast addresses
// must alias the corresponding local tensor on all TP ranks.
//
// Staging uses uint8[2 * TP * slot_bytes] and int32[SM_count] phase counters.
// BF16 staging may change +zero to -zero; all other payload bits are preserved.
// Direct uses uint8[SM_count, 128] semaphores and preserves all payload bits.
// FP8 output scales use int32[ceil(K/512), align(global_rows,4)].
void custom_all_gather_staging(const torch::Tensor& input,
                               torch::Tensor&       output,
                               torch::Tensor&       workspace,
                               torch::Tensor&       counters,
                               int64_t              workspace_mc_ptr,
                               int64_t              rank,
                               int64_t              blocks,
                               int64_t              threads);

void custom_all_gather_direct(const torch::Tensor& input,
                              torch::Tensor&       output,
                              torch::Tensor&       semaphores,
                              int64_t              output_mc_ptr,
                              int64_t              semaphore_mc_ptr,
                              int64_t              rank,
                              int64_t              blocks,
                              int64_t              threads);

// FP8 payloads are raw uint8[local_rows, K], with packed scale words in
// int32[ceil(K/512), align(local_rows, 4)]. Outputs are uint8[8*local_rows, K]
// and int32[ceil(K/512), 8*local_rows], with input scale padding discarded.
// Staging requires slot_bytes % 80 == 0 and at least 20 bytes per 16-byte
// payload/scale vector. Metadata restores every reserved payload bit.
void custom_all_gather_fp8_staging(const torch::Tensor& values,
                                   const torch::Tensor& scales,
                                   torch::Tensor&       output_values,
                                   torch::Tensor&       output_scales,
                                   torch::Tensor&       workspace,
                                   torch::Tensor&       counters,
                                   int64_t              workspace_mc_ptr,
                                   int64_t              rank,
                                   int64_t              blocks,
                                   int64_t              threads);

void custom_all_gather_fp8_direct(const torch::Tensor& values,
                                  const torch::Tensor& scales,
                                  torch::Tensor&       output_values,
                                  torch::Tensor&       output_scales,
                                  torch::Tensor&       semaphores,
                                  int64_t              output_values_mc_ptr,
                                  int64_t              output_scales_mc_ptr,
                                  int64_t              semaphore_mc_ptr,
                                  int64_t              rank,
                                  int64_t              blocks,
                                  int64_t              threads);

}  // namespace rtp_llm
