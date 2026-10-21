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

// Pack verify inputs without materializing their first column. Each accepted
// row is CUDA int32 [width] or [1, width], and its CUDA int32 length is a scalar.
// Lengths must be in [1, width]. token_columns[0] is ignored; the remaining
// columns are contiguous CUDA int32 [batch]. Outputs must be preallocated,
// contiguous CUDA int32 with batch * tokens_per_batch, batch, and
// batch * tokens_per_batch elements respectively. Unsupported inputs return
// false before any launch or output modification. Sources must outlive stream use.
bool tryInvokeMtpSpecDecodeTokensMetadataPrepareFromState(const std::vector<torch::Tensor>& accepted_rows,
                                                          const std::vector<torch::Tensor>& accepted_length_rows,
                                                          const std::vector<torch::Tensor>& token_columns,
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
