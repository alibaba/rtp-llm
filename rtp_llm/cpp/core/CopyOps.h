#pragma once

#include <torch/torch.h>

#include "rtp_llm/models_py/bindings/common/kernels/fuse_copy_util.h"
#include "rtp_llm/models_py/bindings/core/OpData.h"

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

// In-place mask: zero out logits[b,v] where mask[v] != 0.
void runtimeMaskLogits(torch::Tensor& logits, const torch::Tensor& mask);

// Non-blocking copy on a dedicated stream (CUDA), for pinned-host scratch buffers.
void execNoBlockCopy(const CopyParams& params);

// Forwarders kept for callers that still use the exec* names.
void execBatchCopy(const BatchCopyParams& params);
void execMultiMergeCopy(const MultiMergeCopyParams& params);

// Fused single-stream multi-buffer copies (kernel-level).
void fusedCopy(const FusedD2DCopyParams& params);
void fusedStridedCopy(const FusedStridedCopyParams& params);

}  // namespace rtp_llm
