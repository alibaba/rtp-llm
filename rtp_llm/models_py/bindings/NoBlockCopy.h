#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
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

struct BatchedMemoryCopy3DTile {
    const void* src{nullptr};
    void*       dst{nullptr};
    size_t      bytes{0};
    int         global_layer_id{-1};
    int         component_index{-1};
    int         block_index{-1};
};

struct BatchedMemoryCopy3DRun {
    const void* src{nullptr};
    void*       dst{nullptr};
    size_t      width_bytes{0};
    size_t      src_layer_pitch_bytes{0};
    size_t      dst_layer_pitch_bytes{0};
    size_t      depth{0};
};

struct BatchedMemoryCopy3DParams {
    std::vector<BatchedMemoryCopy3DRun> runs;
    int                                 device_index{-1};
    bool                                source_is_cuda{false};
};

// Converts logical host/device tiles into maximal regular per-block/per-component runs.
// Invalid or overlapping layouts are rejected; gaps and pitch changes start a new run.
inline bool buildBatchedMemoryCopy3DRuns(const std::vector<BatchedMemoryCopy3DTile>& tiles,
                                         std::vector<BatchedMemoryCopy3DRun>&        runs,
                                         std::string*                                reason = nullptr) {
    auto fail = [&](const char* message) {
        runs.clear();
        if (reason != nullptr) {
            *reason = message;
        }
        return false;
    };
    runs.clear();
    if (tiles.empty()) {
        return true;
    }
    auto beginAddr = [](const void* ptr) { return reinterpret_cast<uintptr_t>(ptr); };
    std::vector<std::pair<uintptr_t, uintptr_t>> source_ranges;
    std::vector<std::pair<uintptr_t, uintptr_t>> destination_ranges;
    source_ranges.reserve(tiles.size());
    destination_ranges.reserve(tiles.size());
    for (const auto& tile : tiles) {
        if (tile.src == nullptr || tile.dst == nullptr || tile.bytes == 0 || tile.global_layer_id < 0
            || tile.component_index < 0 || tile.block_index < 0) {
            return fail("invalid_tile");
        }
        const uintptr_t src = beginAddr(tile.src);
        const uintptr_t dst = beginAddr(tile.dst);
        if (src + tile.bytes < src || dst + tile.bytes < dst) {
            return fail("address_overflow");
        }
        source_ranges.emplace_back(src, src + tile.bytes);
        destination_ranges.emplace_back(dst, dst + tile.bytes);
    }
    auto has_overlap = [](auto& ranges) {
        std::sort(ranges.begin(), ranges.end());
        for (size_t i = 1; i < ranges.size(); ++i) {
            if (ranges[i].first < ranges[i - 1].second) {
                return true;
            }
        }
        return false;
    };
    if (has_overlap(source_ranges) || has_overlap(destination_ranges)) {
        return fail("overlapping_tiles");
    }

    std::map<std::pair<int, int>, std::vector<BatchedMemoryCopy3DTile>> groups;
    for (const auto& tile : tiles) {
        groups[{tile.block_index, tile.component_index}].push_back(tile);
    }
    for (auto& [_, group] : groups) {
        std::sort(group.begin(), group.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.global_layer_id < rhs.global_layer_id;
        });
        size_t start = 0;
        while (start < group.size()) {
            size_t depth = 1;
            size_t src_pitch = group[start].bytes;
            size_t dst_pitch = group[start].bytes;
            if (start + 1 < group.size() && group[start + 1].global_layer_id == group[start].global_layer_id + 1
                && group[start + 1].bytes == group[start].bytes) {
                const uintptr_t src0 = beginAddr(group[start].src);
                const uintptr_t dst0 = beginAddr(group[start].dst);
                const uintptr_t src1 = beginAddr(group[start + 1].src);
                const uintptr_t dst1 = beginAddr(group[start + 1].dst);
                if (src1 > src0 && dst1 > dst0 && src1 - src0 >= group[start].bytes
                    && dst1 - dst0 >= group[start].bytes && (src1 - src0) % 32 == 0
                    && (dst1 - dst0) % 32 == 0) {
                    src_pitch = src1 - src0;
                    dst_pitch = dst1 - dst0;
                    depth = 2;
                    while (start + depth < group.size()) {
                        const auto& next = group[start + depth];
                        if (next.global_layer_id != group[start].global_layer_id + static_cast<int>(depth)
                            || next.bytes != group[start].bytes
                            || beginAddr(next.src) != src0 + depth * src_pitch
                            || beginAddr(next.dst) != dst0 + depth * dst_pitch) {
                            break;
                        }
                        ++depth;
                    }
                }
            }
            runs.push_back(BatchedMemoryCopy3DRun{group[start].src,
                                                  group[start].dst,
                                                  group[start].bytes,
                                                  src_pitch,
                                                  dst_pitch,
                                                  depth});
            start += depth;
        }
    }
    return true;
}

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

// CUDA 13+ pointer-to-pointer 3D batch executor. Synchronizes its dedicated stream before returning.
bool exec3DBatchedMemoryCopy(const BatchedMemoryCopy3DParams& params);

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
