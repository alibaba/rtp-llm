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

// Number of column splits invokeGreedyTop1Renorm will use for the given shape
// (one row is split across this many CTAs so small batches still fill the GPU).
int greedyTop1RenormSplits(int batch_size, int vocab_size);

// Bytes of CUDA scratch that invokeGreedyTop1Renorm requires in `partials`
// for the given shape (per-(row, split) max/argmax accumulators).
size_t greedyTop1RenormPartialsBytes(int batch_size, int vocab_size);

// Fused greedy (top_k == 1) target-side sampling distribution:
//   samples_out[row]     = first argmax_v logits[row, v]
//   onehot_out[row, v]   = (v == samples_out[row]) ? 1.0f : 0.0f
//
// This is exactly what the replaced chain
//   probs = softmax(logits) -> argmax(probs) -> flashinfer
//   top_k_renorm_probs(probs, k=1) -> onehot
// produces for tie-free rows: softmax is strictly monotonic so
// argmax(softmax(logits)) == argmax(logits), and the k=1 renormalized
// distribution is the one-hot at that argmax. Ties resolve to the first
// (lowest-index) maximum, matching torch::argmax.
//
// logits is READ-ONLY: unlike the replaced chain, no softmax probabilities
// are written back into the logits buffer. Callers must only use this op when
// the logits buffer is provably private to the sampling round.
//
// Kernel 1 reduces each (row, split) chunk to (max, first-argmax) partials;
// kernel 2 combines the per-row partials and writes the one-hot row plus the
// sampled token. Layouts: logits/onehot_out are contiguous [batch, vocab]
// FP32; samples_out is [batch] INT32; partials is a FP32 scratch of
// greedyTop1RenormPartialsBytes() bytes.
//
// write_onehot=false selects the token-only point-mass mode: kernel 2 skips
// the [batch, vocab] zero-fill + one-hot scatter entirely and only writes
// samples_out (onehot_out may then be nullptr). Callers may only use this
// mode when every consumer of the renormalized distribution treats it as a
// point mass at samples_out (e.g. the speculative rejection kernel's
// target_probs_point_mass synthesis), so the materialized one-hot matrix
// would be a dead 2-pass store of information samples_out already carries.
cudaError_t invokeGreedyTop1Renorm(const float* logits,
                                   float*       onehot_out,
                                   int32_t*     samples_out,
                                   int          batch_size,
                                   int          vocab_size,
                                   float*       partials,
                                   cudaStream_t stream,
                                   bool         write_onehot = true);

}  // namespace rtp_llm
