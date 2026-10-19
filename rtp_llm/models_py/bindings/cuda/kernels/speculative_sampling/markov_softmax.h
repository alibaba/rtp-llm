#pragma once

#include <cstddef>
#include <cstdint>

#ifdef USING_ROCM
#include "rtp_llm/models_py/bindings/rocm/cuda_shims.h"
#endif

#if USING_CUDA
#include <cuda_runtime.h>
#endif

namespace rtp_llm {

// Dtype of the dense Markov-bias matrix fed to invokeMarkovSoftmaxStep. The
// bias is the raw torch::mm output of the DSpARK Markov head and keeps the
// checkpoint weight dtype; the kernel casts to FP32 elementwise, matching the
// previous `.to(torch::kFloat32)` host-side cast bit-for-bit.
enum class MarkovBiasDType : int {
    kFloat32  = 0,
    kBFloat16 = 1,
    kFloat16  = 2,
};

// Number of column splits invokeMarkovSoftmaxStep will use for the given
// shape (one row is split across this many CTAs so small batches still fill
// the GPU).
int markovSoftmaxStepSplits(int batch_size, int vocab_size);

// Bytes of CUDA scratch that invokeMarkovSoftmaxStep requires in `partials`
// for the given shape (per-(row, split) online-softmax accumulators).
size_t markovSoftmaxStepPartialsBytes(int batch_size, int vocab_size);

// One DSpARK draft step's distribution, fully fused:
//   q[row, v] = softmax_v((logits[row, v] + bias[row, v]) / temperature[row])
//
// Replaces the per-step host chain cast -> add -> div_ -> softmax -> copy_
// with two row-split kernels that read the FP32 logits view (row-strided: a
// select() of the [B, gamma, vocab] proposal view) and the native-dtype
// contiguous bias [batch_size, vocab_size], and write q to BOTH:
//   - q_all_probs: the (row-strided) all_probabilities.select(1, step) slice
//     consumed by rejection sampling, and
//   - q_contig: a contiguous [batch_size, vocab_size] scratch consumed by
//     FlashInfer SamplingFromProb (may be nullptr when no sampling follows).
// Kernel 1 folds each (row, split) chunk into an online max/sum accumulator;
// kernel 2 combines the per-row accumulators and writes both q outputs.
//
// Per-element math mirrors ATen exactly: FP32 add of the exact FP32 cast of
// the bias, FP32 division by the row temperature, expf(x - row_max), and a
// final FP32 divide by the row sum. Only the (unordered, online) reduction
// structure differs from ATen's cunn_SoftMaxForward, i.e. q matches
// torch::softmax of the materialized chain to ~1e-7 relative.
cudaError_t invokeMarkovSoftmaxStep(const float*    logits,
                                    int64_t         logits_row_stride,
                                    const void*     markov_bias,
                                    MarkovBiasDType bias_dtype,
                                    const float*    temperature,
                                    float*          q_all_probs,
                                    int64_t         q_all_probs_row_stride,
                                    float*          q_contig,
                                    int             batch_size,
                                    int             vocab_size,
                                    float*          partials,
                                    cudaStream_t    stream);

// Bytes of CUDA scratch that invokeMarkovArgmaxStep requires in `partials`
// for the given shape (per-(row, split) max/argmax accumulators).
size_t markovArgmaxStepPartialsBytes(int batch_size, int vocab_size);

// One DSpARK draft step's token, fully fused and softmax-free:
//   token[row] = argmin{ v : x[row, v] == max_v x[row, :] },
//   x[row, v]  = (logits[row, v] + fp32(bias[row, v])) / temperature[row]
//
// All-greedy verification uses exact target-token matching. Deterministic
// first-argmax proposals can therefore replace dense-q sampling, even though
// tiny-temperature softmax may retain mass at ties or close logits. Draft
// choices and RNG consumption can change; target verification still applies.
//
// Kernel 1 reduces each (row, split) chunk to a (max, first-argmax) partial;
// kernel 2 combines the per-row partials and writes tokens_out[row] (int32).
// `partials` needs markovArgmaxStepPartialsBytes(batch_size, vocab_size).
// vocab_size must stay below 2^24 so the argmax index is exact through its
// float partial slot; callers gate larger vocabs to the dense softmax path.
cudaError_t invokeMarkovArgmaxStep(const float*    logits,
                                   int64_t         logits_row_stride,
                                   const void*     markov_bias,
                                   MarkovBiasDType bias_dtype,
                                   const float*    temperature,
                                   int32_t*        tokens_out,
                                   int             batch_size,
                                   int             vocab_size,
                                   float*          partials,
                                   cudaStream_t    stream);

}  // namespace rtp_llm
