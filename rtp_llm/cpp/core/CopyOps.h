#pragma once

#include <cstddef>
#include <torch/torch.h>

#include "rtp_llm/cpp/core/CopyTypes.h"

namespace rtp_llm {

// ===================================================================
// Copy ops
// ===================================================================

// Non-CUDA/ROCm builds are not a supported engine runtime. Their implementation
// is a compile/link compatibility fallback for host-only tools: only host copies
// are accepted, while accelerator-only operations fail explicitly.

// Synchronous copy honoring CopyParams::overlapped (CUDA: optional overlap stream).
void runtimeCopy(const CopyParams& params);

// Batched copy with D2D fast-path on CUDA; H2D/D2H/H2H fall back to per-buffer runtimeCopy.
void runtimeBatchCopy(const BatchCopyParams& params);

// CUDA in-place mask: set logits[b,v] to -INFINITY where mask[b,v] != 0.
// Unsupported backends throw OpException.
void runtimeMaskLogits(torch::Tensor& logits, const torch::Tensor& mask);

// Applies an int32 bit-packed allow mask to logits. row_indices optionally
// maps compact mask rows to rows in logits; out-of-range rows are skipped.
void runtimeApplyPackedMaskLogits(const torch::Tensor& logits,
                                  const torch::Tensor& packed_allow_mask,
                                  const torch::Tensor& row_indices,
                                  size_t               vocab_size);
void runtimeApplyPackedMaskLogits(const torch::Tensor& logits,
                                  const torch::Tensor& packed_allow_mask,
                                  size_t               vocab_size);

// Non-blocking copy on a dedicated stream (CUDA), for pinned-host scratch buffers.
void runtimeNoBlockCopy(const CopyParams& params);

// Multi-tensor copy used by cache connectors. CUDA additionally supports the
// split-KV staging/scatter path described by MultiCopyParams.
void runtimeNoBlockCopy(const MultiCopyParams& params);

// CUDA 12.8+ batches regular host/device pointer copies into one runtime call.
bool runtimeBatchedMemoryCopy(const BatchedMemoryCopyParams& params);

// Uses pinned-host and device staging buffers plus one SM gather/scatter kernel.
bool runtimeStagedMemoryCopy(const StagedMemoryCopyParams& params, StagedMemoryCopyScratch* scratch = nullptr);
void releaseStagedMemoryCopyScratch(StagedMemoryCopyScratch& scratch);

// Warm split-KV kernels before latency-sensitive cache traffic starts.
void runtimeWarmupNoBlockCopy();

void runtimeMultiMergeCopy(const MultiMergeCopyParams& params);

// Fused single-stream multi-buffer copies (kernel-level).
void fusedCopy(const FusedD2DCopyParams& params);
void fusedStridedCopy(const FusedStridedCopyParams& params);

}  // namespace rtp_llm
