#pragma once

#include <torch/torch.h>
#include <cuda_runtime.h>
#include <vector>

namespace rtp_llm {

// Try to execute split-KV scatter/gather copy using SM copy kernels.
//
// src/dst: tensors grouped per block, 2*layer_num tensors per block
//          (kv_layer0, scale_layer0, kv_layer1, scale_layer1, ...)
//   H2D: src[i] host-pinned, dst[i] device
//   D2H: src[i] device,      dst[i] host-pinned
//
// Returns true on success, false if preconditions not met (caller falls back to plain memcpy).
bool trySplitKvMultiCopy(const std::vector<torch::Tensor>& src,
                         const std::vector<torch::Tensor>& dst,
                         int                               layer_num,
                         int64_t                           kv_cache_stride_bytes,
                         int64_t                           kv_scale_stride_bytes,
                         cudaStream_t                      stream);

// JIT-warm the split-KV scatter/gather kernels on the given stream.
bool warmupSplitKvCopyKernels(cudaStream_t stream);

// Free thread-local device buffers used by trySplitKvMultiCopy.
// Must be called on the same thread that calls trySplitKvMultiCopy,
// before that thread exits (e.g. end of NormalEngine::loop).
void releaseSplitKvCopyState();

}  // namespace rtp_llm
