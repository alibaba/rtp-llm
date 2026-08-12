#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

namespace rtp_llm {

void invokeMtpTargetVerifyPrepare(const torch::Tensor& sequence_lengths,
                                  torch::Tensor&       input_lengths,
                                  torch::Tensor&       prefix_lengths,
                                  torch::Tensor&       sequence_lengths_plus_1,
                                  torch::Tensor&       lm_output_indexes,
                                  int32_t              tokens_per_batch,
                                  cudaStream_t         stream);

void invokeMtpSpecDecodeMetadataPrepare(torch::Tensor& input_lengths,
                                        torch::Tensor& lm_output_indexes,
                                        int32_t        tokens_per_batch,
                                        cudaStream_t   stream);

void invokeMtpSpecDecodeTokensMetadataPrepare(const std::vector<torch::Tensor>& token_columns,
                                              torch::Tensor&                    spec_tokens,
                                              torch::Tensor&                    input_lengths,
                                              torch::Tensor&                    lm_output_indexes,
                                              int32_t                           tokens_per_batch,
                                              cudaStream_t                      stream);

// Fused kernel for dispatchDecodeAsync per-stream state publishing.
// Computes: next_seq_len[i] = prev_seq_len[i] + accept_len[i]  (int32)
//           hidden_idx[i]   = accept_len[i] - 1                 (int64)
// All inputs/outputs must be contiguous CUDA tensors with numel >= batch_size.
void invokeMtpDispatchStatePrepare(const torch::Tensor& accept_len,
                                   const torch::Tensor& prev_seq_len,
                                   torch::Tensor&       next_seq_len,
                                   torch::Tensor&       hidden_idx,
                                   int64_t              batch_size,
                                   cudaStream_t         stream);

inline constexpr int64_t MTP_LINEAR_BLOCK_PATCH_WIDTH = 4;

// Round N: capture the final values produced by specUpdate's two ordered
// LINEAR swaps. before/after use [batch, group, 4], and positions uses
// [batch, 4]. source_slots explicitly records the folded permutation so
// duplicate values such as NULL_BLOCK_IDX remain unambiguous.
void invokeMtpLinearKvCacheBlockPatchBuild(const torch::Tensor& block_ids,
                                           const torch::Tensor& group_types,
                                           const torch::Tensor& valid_block_counts,
                                           const torch::Tensor& prev_seq_len,
                                           const torch::Tensor& accept_len,
                                           torch::Tensor&       positions,
                                           torch::Tensor&       source_slots,
                                           torch::Tensor&       before_values,
                                           torch::Tensor&       after_values,
                                           torch::Tensor&       patch_valid,
                                           int32_t              seq_size_per_block,
                                           cudaStream_t         stream);

// Round N+1: repair a fresh host snapshot with the saved final values. The
// common pre/post cases are idempotent assignments. If allocator work changed
// a touched slot, the saved permutation is applied to the fresh values instead
// of restoring an obsolete block ID.
void invokeMtpLinearKvCacheBlockPatchApply(torch::Tensor&       block_ids,
                                           const torch::Tensor& group_types,
                                           const torch::Tensor& valid_block_counts,
                                           const torch::Tensor& positions,
                                           const torch::Tensor& source_slots,
                                           const torch::Tensor& before_values,
                                           const torch::Tensor& after_values,
                                           const torch::Tensor& patch_valid,
                                           const torch::Tensor& pending_patches,
                                           cudaStream_t         stream);

}  // namespace rtp_llm
