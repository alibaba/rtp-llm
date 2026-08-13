#pragma once

#include <torch/torch.h>
#include <vector>

namespace rtp_llm {

struct MultiCopyParams {
    std::vector<torch::Tensor> multi_dst;
    std::vector<torch::Tensor> multi_src;

    // Split-KV scatter/gather path (CUDA only, uses SM copy kernels).
    // When split_kv_layer_num > 0, the copy fuses per-block H2D staging + D2D scatter.
    int    split_kv_layer_num          = 0;
    size_t split_kv_cache_stride_bytes = 0;
    size_t split_kv_scale_stride_bytes = 0;
};

// Multi-tensor copy with a device-specific implementation. On asynchronous
// implementations, return or exception propagation guarantees that all work
// submitted by this call is terminal; failure to prove that condition aborts.
void execNoBlockCopy(const MultiCopyParams& params);

// Warmup split-KV copy kernels. No-op on implementations without that path.
// Must be called after cudaSetDevice + setCurrentCUDAStream.
void warmupNoBlockCopy();

}  // namespace rtp_llm
