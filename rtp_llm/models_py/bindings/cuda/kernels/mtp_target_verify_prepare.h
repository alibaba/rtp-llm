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

// For each batch b with input_lengths_d[b] tokens packed at offset cumsum(input_lengths_d)[b-1]
// in combo_tokens_in:
//   * shift combo_tokens_in[offset .. offset+input_length-1] left by 1 (drop first token)
//   * write new_all_token_ids[b, token_stride-1] at combo_tokens_out[offset+input_length-1]
// All inputs/outputs are int32 CUDA tensors. combo_tokens_out may alias combo_tokens_in;
// the kernel writes each position from a single thread per (batch, position) pair so
// in-place shift is safe.
void invokeMtpPrefillShiftAppend(const torch::Tensor& combo_tokens_in,
                                 const torch::Tensor& input_lengths,
                                 const torch::Tensor& batch_offsets,
                                 const torch::Tensor& new_all_token_ids,
                                 torch::Tensor&       combo_tokens_out,
                                 int32_t              token_stride,
                                 cudaStream_t         stream);

}  // namespace rtp_llm
