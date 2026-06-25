#pragma once

#include <torch/extension.h>
#if USING_ROCM
#include <hip/hip_runtime.h>
#elif USING_CUDA
#include <cuda_runtime.h>
#endif
#include <vector>

namespace rtp_llm {

#if USING_CUDA
using MtpPrepareStream = cudaStream_t;
#elif USING_ROCM
using MtpPrepareStream = hipStream_t;
#endif

void invokeMtpTargetVerifyPrepare(const torch::Tensor& sequence_lengths,
                                  torch::Tensor&       input_lengths,
                                  torch::Tensor&       prefix_lengths,
                                  torch::Tensor&       sequence_lengths_plus_1,
                                  torch::Tensor&       lm_output_indexes,
                                  int32_t              tokens_per_batch,
                                  MtpPrepareStream     stream);

void invokeMtpSpecDecodeMetadataPrepare(torch::Tensor&   input_lengths,
                                        torch::Tensor&   lm_output_indexes,
                                        int32_t          tokens_per_batch,
                                        MtpPrepareStream stream);

void invokeMtpSpecDecodeTokensMetadataPrepare(const std::vector<torch::Tensor>& token_columns,
                                              torch::Tensor&                    spec_tokens,
                                              torch::Tensor&                    input_lengths,
                                              torch::Tensor&                    lm_output_indexes,
                                              int32_t                           tokens_per_batch,
                                              MtpPrepareStream                  stream);

// Fused kernel for dispatchDecodeAsync per-stream state publishing.
// Computes: next_seq_len[i] = prev_seq_len[i] + accept_len[i]  (int32)
//           hidden_idx[i]   = accept_len[i] - 1                 (int64)
// All inputs/outputs must be contiguous device tensors with numel >= batch_size.
void invokeMtpDispatchStatePrepare(const torch::Tensor& accept_len,
                                   const torch::Tensor& prev_seq_len,
                                   torch::Tensor&       next_seq_len,
                                   torch::Tensor&       hidden_idx,
                                   int64_t              batch_size,
                                   MtpPrepareStream     stream);

}  // namespace rtp_llm
