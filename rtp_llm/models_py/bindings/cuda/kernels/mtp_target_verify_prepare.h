#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

namespace rtp_llm {

struct MtpDSparkVerifyInputs {
    torch::Tensor tokens, input_lengths, lm_indexes, history, positions, proposals;
};

// Assemble target tokens, Engram windows and metadata in a single launch.
// Undefined tokens signal that the caller should retain its Torch fallback.
MtpDSparkVerifyInputs tryMtpDSparkVerifyPrepare(const torch::Tensor& anchors,
                                                const torch::Tensor& proposals,
                                                const torch::Tensor& history,
                                                const torch::Tensor& position_bases,
                                                cudaStream_t         stream,
                                                bool                 copy_proposals = false);

// Assemble both page tables directly from device snapshots and pinned host
// suffixes. Returns false without launching for unsupported layouts/devices.
bool tryMtpGatherPageSnapshots(const torch::Tensor&              physical_host,
                               const torch::Tensor&              kernel_host,
                               const std::vector<torch::Tensor>& physical_snapshots,
                               const std::vector<torch::Tensor>& kernel_snapshots,
                               torch::Tensor&                    physical_output,
                               torch::Tensor&                    kernel_output,
                               cudaStream_t                      stream);

// Fuses flip/cat/index construction/gather for the four-token Engram history.
// Returns undefined for layouts/dtypes that should use the Torch fallback.
torch::Tensor
tryMtpAdvanceEngramTokenWindows(const torch::Tensor& anchor, const torch::Tensor& tokens, const torch::Tensor& lengths);

// A row may independently use a device anchor, a device sequence length, or
// host scalar fallbacks. The caller owns all pointed-to int32 CUDA storage and
// orders its producers before this launch. token_stride is measured in elements.
struct MtpRoundHeadRow {
    const int32_t* tokens         = nullptr;
    const int32_t* accept_len     = nullptr;
    const int32_t* sequence       = nullptr;
    int64_t        token_stride   = 1;
    int32_t        token_count    = 0;
    int32_t        anchor         = 0;
    int32_t        sequence_value = 0;
    int32_t        sequence_bias  = 0;
};

// Writes anchors [B], committed_ends [B], and every combo_tokens [B, W] slot.
// Device sequence values receive sequence_bias; host sequence_value is final.
// Row descriptors travel in launch parameters: no descriptor H2D or sync.
void invokeMtpDSparkRoundHeadPrepare(const std::vector<MtpRoundHeadRow>& rows,
                                     torch::Tensor&                      anchors,
                                     torch::Tensor&                      committed_ends,
                                     torch::Tensor&                      combo_tokens,
                                     int32_t                             tokens_per_batch,
                                     int32_t                             mask_token,
                                     cudaStream_t                        stream);

struct MtpSequenceRow {
    const int32_t* sequence_length      = nullptr;
    int32_t        host_sequence_length = 0;
};

// Also materializes prev_seq_len, which is consumed by linear-cache advance.
void invokeMtpDispatchStatePrepareRows(const std::vector<MtpSequenceRow>& rows,
                                       const torch::Tensor&               accept_len,
                                       torch::Tensor&                     prev_seq_len,
                                       torch::Tensor&                     next_seq_len,
                                       torch::Tensor&                     hidden_idx,
                                       cudaStream_t                       stream,
                                       const torch::Tensor&               engram_anchor   = {},
                                       const torch::Tensor&               accepted_tokens = {},
                                       const torch::Tensor&               engram_output   = {});

// Each source row contains F contiguous CUDA int32 values, owned by the caller.
void invokeMtpPositionGather(const std::vector<const int32_t*>& rows, torch::Tensor& bases, cudaStream_t stream);

// Expands [B, F] bases to flat [B * W * F], adding the step to each factor.
// bases may have a non-contiguous row stride; its factor dimension is contiguous.
void invokeMtpPositionExpand(const torch::Tensor& bases, torch::Tensor& output, int64_t width, cudaStream_t stream);

// Materializes the speculative sampler token matrix in one launch.
//
// Every verify row of a request shares the same token history; only the `gamma`
// columns starting at `base_lengths[b]` differ, holding that request's draft
// proposal. Building the matrix row by row on the host copies the history
// `rows` times, so this expands one history row per request instead:
//   output[b, r, c] = c < base_lengths[b]         ? history[b, c]
//                   : c < base_lengths[b] + gamma ? draft[b, c - base_lengths[b]]
//                   :                               0
// history is CUDA int32 [B, width]; draft is CUDA int32 [B, gamma];
// base_lengths is CUDA int64 [B]; output is contiguous CUDA int32
// [B, rows, width] with width == history.size(1).
void invokeMtpSpecSamplerTokenMatrix(const torch::Tensor& history,
                                     const torch::Tensor& draft,
                                     const torch::Tensor& base_lengths,
                                     torch::Tensor&       output,
                                     int64_t              gamma,
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

}  // namespace rtp_llm
