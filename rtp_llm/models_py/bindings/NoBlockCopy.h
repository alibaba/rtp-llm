#pragma once

#include <cstddef>
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

struct BatchedMemoryCopyTile {
    void*       dst   = nullptr;
    const void* src   = nullptr;
    size_t      bytes = 0;
};

struct BatchedMemoryCopyParams {
    std::vector<BatchedMemoryCopyTile> tiles;
    int                                device_index = -1;
};

enum class StagedMemoryCopyDirection {
    H2D = 0,
    D2H = 1,
};

struct StagedMemoryCopyTile {
    void*  gpu         = nullptr;
    size_t host_offset = 0;
    size_t bytes       = 0;
};

struct StagedMemoryCopyHostSegment {
    void*  host        = nullptr;
    size_t host_offset = 0;
    size_t bytes       = 0;
};

struct StagedMemoryCopyParams {
    void*                                    host_base  = nullptr;
    size_t                                   host_bytes = 0;
    std::vector<StagedMemoryCopyHostSegment> host_segments;
    std::vector<StagedMemoryCopyTile>        tiles;
    int                                      device_index = -1;
    StagedMemoryCopyDirection                direction    = StagedMemoryCopyDirection::H2D;
};

struct StagedMemoryCopyScratch {
    void*  host_staging       = nullptr;
    size_t host_capacity      = 0;
    void*  device_staging     = nullptr;
    size_t device_capacity    = 0;
    void*  device_ptrs        = nullptr;
    void*  device_offsets     = nullptr;
    void*  device_sizes       = nullptr;
    size_t meta_capacity      = 0;
    int    device_index       = -1;
};

// Multi-tensor non-blocking copy with device-specific implementation.
// CUDA: uses a dedicated stream + optional split-KV SM scatter path.
// ROCm: plain tensor copy_.
// Other devices: not supported (will abort).
void execNoBlockCopy(const MultiCopyParams& params);

// One CUDA runtime call copy executor for regular host/device pointers.
// CUDA 12.8+ uses cudaMemcpyBatchAsync to avoid per-tile cudaMemcpyAsync launches.
bool execBatchedMemoryCopy(const BatchedMemoryCopyParams& params);

// Stages compact host payload in GPU memory, then uses one SM gather/scatter kernel.
// host_segments may describe non-contiguous host blocks; they are packed/unpacked on CPU.
// scratch is optional; passing one lets callers reuse pinned host staging and device metadata buffers.
// H2D: compact host payload -> GPU staging -> tile.gpu by tile.host_offset.
// D2H: tile.gpu -> GPU staging by tile.host_offset -> compact host payload.
bool execStagedMemoryCopy(const StagedMemoryCopyParams& params, StagedMemoryCopyScratch* scratch = nullptr);
void releaseStagedMemoryCopyScratch(StagedMemoryCopyScratch& scratch);

// Warmup split-KV copy kernels. No-op on non-CUDA / PPU devices.
// Must be called after cudaSetDevice + setCurrentCUDAStream.
void warmupNoBlockCopy();

}  // namespace rtp_llm
