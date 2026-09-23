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

// Expand MiniMax-M3 MSA request-row metadata into the token-row layout used by
// paged multi-token attention. Returns, in order:
//   physical_block_table [batch * tokens_per_batch, max_blocks]
//   positions            [batch * tokens_per_batch]
//   sequence_lengths     [batch * tokens_per_batch]
//   valid_token_mask     [batch * tokens_per_batch]
// Padded request rows are identified by input_lengths == 0. The operation is
// asynchronous on the current PyTorch CUDA stream and performs no host readback.
std::vector<torch::Tensor> mtpMsaTargetVerifyAddressingPrepare(const torch::Tensor& request_block_table,
                                                               const torch::Tensor& prefix_lengths,
                                                               const torch::Tensor& input_lengths,
                                                               int64_t              tokens_per_batch);

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

}  // namespace rtp_llm
