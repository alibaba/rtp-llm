#pragma once
#include <vector>

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

namespace rtp_llm {

// Publish MTP next sequence length, selected token and BF16 hidden row without
// materializing int64 gather indices. Undefined previous_lengths keeps an
// already prepared next_lengths tensor unchanged.
bool tryMtpPublishState(const torch::Tensor& accept_lengths,
                        const torch::Tensor& accepted_tokens,
                        const torch::Tensor& previous_lengths,
                        const torch::Tensor& hidden_states,
                        int32_t              tokens_per_batch,
                        torch::Tensor&       next_lengths,
                        torch::Tensor&       selected_tokens,
                        torch::Tensor&       selected_hidden,
                        cudaStream_t         stream,
                        const std::vector<const int32_t*>& previous_rows = {});

// Keep asynchronous metric snapshots independent of mutable acceptance storage.
// Greedy point-mass MTP only. Returns false without launching for unsupported layouts.
bool tryMtpGreedyFinalize(const torch::Tensor& draft,
                          const torch::Tensor& target,
                          torch::Tensor& tokens,
                          torch::Tensor& lengths,
                          torch::Tensor& rows,
                          torch::Tensor& total,
                          cudaStream_t stream);

bool tryMtpAcceptMetrics(const torch::Tensor& lengths, torch::Tensor& rows, torch::Tensor& total, cudaStream_t stream);

// Rejection sampling uses -1 for unused positions; the MTP public output uses 0.
bool tryMtpCleanRejectedTokens(torch::Tensor& tokens, cudaStream_t stream);

// Preserve argmax of the rounded probabilities (including first-index ties
// and NaNs), generate the point-mass distribution and map the token ID.
bool tryMtpGreedyPack(const torch::Tensor& probabilities,
                      const torch::Tensor& mapping,
                      torch::Tensor&       output_probabilities,
                      torch::Tensor&       output_tokens,
                      torch::Tensor&       output_tokens_int32,
                      const torch::Tensor& sequence_lengths,
                      const torch::Tensor& hidden_to_zero,
                      bool                 zero_tokens,
                      torch::Tensor&       next_sequence_lengths,
                      cudaStream_t         stream,
                      bool                 compact_point_mass = false);

// Initialize newly allocated MTP token mirrors and an optional fake-stream
// indexer seed. Undefined outputs are skipped.
void invokeMtpStreamInit(torch::Tensor& seed,
                         torch::Tensor& target_token,
                         torch::Tensor& propose_token,
                         int64_t        sequence_length,
                         cudaStream_t   stream);

// Generate post-rejection output indices and, optionally, the last accepted
// token in one pass. An undefined selected_tokens skips token selection.
void invokeMtpPostRejectionPrepare(const torch::Tensor& accept_len,
                                   const torch::Tensor& tokens,
                                   torch::Tensor&       indices,
                                   torch::Tensor&       selected_tokens,
                                   int32_t              tokens_per_batch,
                                   cudaStream_t         stream,
                                   bool                 initialize_fake = false);

void invokeMtpSelectedTokens(const torch::Tensor& tokens,
                             const torch::Tensor& indices,
                             torch::Tensor&       output,
                             cudaStream_t         stream);

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

// Computes the two FP32 terms needed by stable log-softmax without materializing
// an FP32 [rows, vocab] tensor: row_max and log(sum(exp(logit - row_max))).
// Keeping the terms separate avoids losing a small normalization correction in
// row_max + correction before the selected logit is subtracted.
void invokeMtpRowLogSoftmaxStats(const torch::Tensor& logits,
                                 torch::Tensor&       row_max,
                                 torch::Tensor&       row_shifted_logsumexp,
                                 int64_t              real_vocab_size,
                                 cudaStream_t         stream);

// Computes target-model logprobs for selected dense rows without gathering a
// [selected_rows, vocab] tensor. source_row_indices maps each compact output
// row to a row in logits and to the corresponding flattened emitted token ID.
// logits may have a padded row stride, but its width must already be cropped to
// the real vocabulary. Top-K values are returned in descending order.
void invokeMtpSelectedRowLogProbs(const torch::Tensor& logits,
                                  const torch::Tensor& source_row_indices,
                                  const torch::Tensor& emitted_token_ids,
                                  torch::Tensor&       token_logprobs,
                                  torch::Tensor&       top_logprob_token_ids,
                                  torch::Tensor&       top_logprobs,
                                  int64_t              top_k,
                                  cudaStream_t         stream);

// REBASE CONFLICT CONTEXT(518707c73): keep new base dispatch-state publishing
// kernel and add source branch prefill shift/append kernel to avoid sync-heavy
// CPU token manipulation.
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
