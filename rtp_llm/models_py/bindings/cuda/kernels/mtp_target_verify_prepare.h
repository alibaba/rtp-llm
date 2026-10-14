#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

namespace rtp_llm {

// Read each request's last accepted token directly from its device-state views.
// Descriptors travel in bounded kernel arguments; no pointer-table H2D staging.
// Returns false without changing output for layouts/dtypes handled by the
// existing ATen path. Callers must keep inputs alive and order their producer
// before this stream, including allocator recordStream for cross-stream use.
bool tryInvokeMtpAcceptedTokenGather(const std::vector<torch::Tensor>& accept_tokens,
                                     const std::vector<torch::Tensor>& accept_lengths,
                                     torch::Tensor&                    output,
                                     cudaStream_t                      stream);

void invokeMtpTargetVerifyPrepare(const torch::Tensor& sequence_lengths,
                                  torch::Tensor&       input_lengths,
                                  torch::Tensor&       prefix_lengths,
                                  torch::Tensor&       sequence_lengths_plus_1,
                                  torch::Tensor&       lm_output_indexes,
                                  int32_t              tokens_per_batch,
                                  cudaStream_t         stream,
                                  const torch::Tensor* position_bases = nullptr,
                                  torch::Tensor*       position_ids   = nullptr);

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

// Shift each packed Prefill request left by one token and append the sampled
// token from its final sample column. All tensors are contiguous CUDA int32.
void invokeMtpPrefillShiftAppend(const torch::Tensor& combo_tokens_in,
                                 const torch::Tensor& input_lengths,
                                 const torch::Tensor& batch_offsets,
                                 const torch::Tensor& new_all_token_ids,
                                 torch::Tensor&       combo_tokens_out,
                                 int32_t              token_stride,
                                 cudaStream_t         stream);

// For default one-dimensional positions, the sampled token keeps the final
// input position. Other position styles must use their stream policy.
void invokeMtpPrefillShiftAppend(const torch::Tensor& combo_tokens_in,
                                 const torch::Tensor& input_lengths,
                                 const torch::Tensor& batch_offsets,
                                 const torch::Tensor& new_all_token_ids,
                                 torch::Tensor&       combo_tokens_out,
                                 const torch::Tensor& position_ids_in,
                                 torch::Tensor&       position_ids_out,
                                 int32_t              token_stride,
                                 cudaStream_t         stream);

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
