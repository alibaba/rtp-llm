#include "rtp_llm/cpp/cache/block_tree_cache/copy_engine/DeviceHostCopyStrategy.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include <torch/torch.h>

#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/models_py/bindings/NoBlockCopy.h"

namespace rtp_llm {

StrategyResult GenericMultiCopyDeviceHostCopyStrategy::tryExecute(const DeviceHostCopyPlan& plan,
                                                                  const DeviceHostCopyOptions& /*options*/) {
    if (plan.copy_tiles.empty()) {
        return StrategyResult::done();
    }

    std::vector<torch::Tensor> dst_buffers;
    std::vector<torch::Tensor> src_buffers;

    auto byte_tensor = [](void* addr, size_t bytes, torch::Device device) {
        return torch::from_blob(
            addr, {static_cast<int64_t>(bytes)}, torch::TensorOptions().dtype(torch::kUInt8).device(device));
    };

    for (const auto& tile : plan.copy_tiles) {
        if (tile.bytes == 0) {
            continue;
        }
        auto cpu_device = torch::Device(torch::kCPU);
        auto cuda_device =
            tile.device_index >= 0 ? torch::Device(torch::kCUDA, tile.device_index) : torch::Device(torch::kCUDA);
        if (plan.device_to_host) {
            dst_buffers.push_back(byte_tensor(tile.host_addr, tile.bytes, cpu_device));
            src_buffers.push_back(byte_tensor(tile.device_addr, tile.bytes, cuda_device));
        } else {
            dst_buffers.push_back(byte_tensor(tile.device_addr, tile.bytes, cuda_device));
            src_buffers.push_back(byte_tensor(tile.host_addr, tile.bytes, cpu_device));
        }
    }

    if (!dst_buffers.empty()) {
        MultiCopyParams mc{dst_buffers, src_buffers};
        execNoBlockCopy(mc);
    }
    return StrategyResult::done();
}

StrategyResult CudaBatchDeviceHostCopyStrategy::tryExecute(const DeviceHostCopyPlan&    plan,
                                                           const DeviceHostCopyOptions& options) {
    if (!options.cuda_batch_copy_enabled) {
        return StrategyResult::notApplicable();
    }
    if (plan.copy_tiles.empty()) {
        return StrategyResult::done();
    }

    int device_index = -1;
    for (const auto& tile : plan.copy_tiles) {
        if (tile.device_index < 0) {
            return StrategyResult::notApplicable();
        }
        if (device_index < 0) {
            device_index = tile.device_index;
        } else if (tile.device_index != device_index) {
            return StrategyResult::notApplicable();
        }
    }

    BatchedMemoryCopyParams params;
    params.device_index = device_index;
    params.tiles.reserve(plan.copy_tiles.size());

    for (const auto& tile : plan.copy_tiles) {
        if (tile.bytes == 0) {
            continue;
        }
        BatchedMemoryCopyTile batch_tile;
        if (plan.device_to_host) {
            batch_tile.dst = tile.host_addr;
            batch_tile.src = tile.device_addr;
        } else {
            batch_tile.dst = tile.device_addr;
            batch_tile.src = tile.host_addr;
        }
        batch_tile.bytes = tile.bytes;
        params.tiles.push_back(batch_tile);
    }

    if (params.tiles.empty()) {
        return StrategyResult::done();
    }

    // The bool API cannot distinguish "unsupported CUDART" from "CUDA runtime error".
    // Conservatively fall back to generic until a proper attempt-status API is added.
    bool ok = execBatchedMemoryCopy(params);
    if (!ok) {
        return StrategyResult::notApplicable();
    }
    return StrategyResult::done();
}

static constexpr size_t kStagedAlignment = 16;

static size_t alignUp(size_t value, size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

StagedSmDeviceHostCopyStrategy::~StagedSmDeviceHostCopyStrategy() {
    for (auto& [_, scratch] : scratch_by_device_) {
        if (scratch) {
            releaseStagedMemoryCopyScratch(*scratch);
        }
    }
}

StrategyResult StagedSmDeviceHostCopyStrategy::tryExecute(const DeviceHostCopyPlan&    plan,
                                                          const DeviceHostCopyOptions& options) {
    if (!options.staged_sm_copy_enabled) {
        return StrategyResult::notApplicable();
    }

    if (plan.copy_tiles.empty()) {
        return StrategyResult::done();
    }

    if (plan.copy_tiles.size() < options.staged_sm_min_tile_count) {
        return StrategyResult::notApplicable();
    }

    size_t total_bytes  = 0;
    int    device_index = -1;
    for (const auto& tile : plan.copy_tiles) {
        if (tile.device_index < 0) {
            return StrategyResult::notApplicable();
        }
        if (device_index < 0) {
            device_index = tile.device_index;
        } else if (tile.device_index != device_index) {
            return StrategyResult::notApplicable();
        }
        total_bytes += tile.bytes;
    }

    if (total_bytes < options.staged_sm_min_bytes) {
        return StrategyResult::notApplicable();
    }

    // Build staged params with compact host segments
    StagedMemoryCopyParams staged_params;
    staged_params.host_base    = plan.host.base;
    staged_params.device_index = device_index;
    staged_params.direction    = plan.device_to_host ? StagedMemoryCopyDirection::D2H : StagedMemoryCopyDirection::H2D;

    size_t current_staging_offset = 0;
    staged_params.tiles.reserve(plan.copy_tiles.size());
    staged_params.host_segments.reserve(plan.copy_tiles.size());

    for (const auto& tile : plan.copy_tiles) {
        if (tile.bytes == 0) {
            continue;
        }
        size_t staging_offset = alignUp(current_staging_offset, kStagedAlignment);

        StagedMemoryCopyTile staged_tile;
        staged_tile.gpu         = tile.device_addr;
        staged_tile.host_offset = staging_offset;
        staged_tile.bytes       = tile.bytes;
        staged_params.tiles.push_back(staged_tile);

        StagedMemoryCopyHostSegment segment;
        segment.host        = tile.host_addr;
        segment.host_offset = staging_offset;
        segment.bytes       = tile.bytes;

        // Merge with previous segment if contiguous in both host and staging space
        if (!staged_params.host_segments.empty()) {
            auto& prev     = staged_params.host_segments.back();
            auto* prev_end = static_cast<uint8_t*>(prev.host) + prev.bytes;
            if (prev_end == tile.host_addr && prev.host_offset + prev.bytes == staging_offset) {
                prev.bytes += tile.bytes;
                current_staging_offset = staging_offset + tile.bytes;
                continue;
            }
        }
        staged_params.host_segments.push_back(segment);
        current_staging_offset = staging_offset + tile.bytes;
    }
    staged_params.host_bytes = current_staging_offset;

    if (staged_params.tiles.empty()) {
        return StrategyResult::done();
    }

    // Hold lock during the entire execStagedMemoryCopy call to prevent concurrent
    // staged copies on the same device from corrupting shared staging buffers.
    std::lock_guard<std::mutex> lock(scratch_mutex_);
    auto&                       entry = scratch_by_device_[device_index];
    if (!entry) {
        entry               = std::make_unique<StagedMemoryCopyScratch>();
        entry->device_index = device_index;
    }

    bool ok = execStagedMemoryCopy(staged_params, entry.get());
    if (!ok) {
        return StrategyResult::failed(CopyStatus::DEVICE_IO_ERROR);
    }
    return StrategyResult::done();
}

}  // namespace rtp_llm
