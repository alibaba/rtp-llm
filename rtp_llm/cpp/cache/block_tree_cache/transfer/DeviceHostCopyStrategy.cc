#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceHostCopyStrategy.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>

#include <torch/torch.h>

#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/models_py/bindings/NoBlockCopy.h"

namespace rtp_llm {

StrategyResult GenericMultiCopyDeviceHostCopyStrategy::tryExecute(const DeviceHostCopyPlan& plan,
                                                                  const DeviceHostCopyOptions& /*options*/,
                                                                  const DeviceHostCopyExecutionContext& context) {
    std::vector<torch::Tensor> dst_buffers;
    std::vector<torch::Tensor> src_buffers;

    auto byte_tensor = [](void* addr, size_t bytes, torch::Device device) {
        return torch::from_blob(
            addr, {static_cast<int64_t>(bytes)}, torch::TensorOptions().dtype(torch::kUInt8).device(device));
    };

    for (const auto& tile : plan.copy_tiles) {
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

    MultiCopyParams mc{dst_buffers, src_buffers};
    try {
        execNoBlockCopy(mc, context);
    } catch (const std::exception& error) {
        RTP_LLM_LOG_WARNING("generic copy failed group_set=%zu device=%d direction=%s stream=%p: %s",
                            plan.group_set_id, context.deviceIndex(), plan.device_to_host ? "D2H" : "H2D",
                            reinterpret_cast<void*>(context.stream()), error.what());
        return StrategyResult::failed(TransferStatus::DEVICE_IO_ERROR);
    }
    return StrategyResult::done();
}

StrategyResult CudaBatchDeviceHostCopyStrategy::tryExecute(const DeviceHostCopyPlan&    plan,
                                                           const DeviceHostCopyOptions& options,
                                                           const DeviceHostCopyExecutionContext& context) {
    if (!options.cuda_batch_copy_enabled) {
        return StrategyResult::notApplicable();
    }

    const int device_index = plan.copy_tiles.front().device_index;
    if (device_index < 0) {
        return StrategyResult::notApplicable();
    }

    BatchedMemoryCopyParams params;
    params.device_index = device_index;
    params.direction = plan.device_to_host ? DeviceHostCopyDirection::D2H : DeviceHostCopyDirection::H2D;
    params.tiles.reserve(plan.copy_tiles.size());

    for (const auto& tile : plan.copy_tiles) {
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

    const auto status = execBatchedMemoryCopy(params, context);
    if (status == BatchedMemoryCopyStatus::NOT_SUPPORTED) {
        return StrategyResult::notApplicable();
    }
    if (status == BatchedMemoryCopyStatus::EXECUTION_FAILED) {
        RTP_LLM_LOG_WARNING("batch copy failed group_set=%zu device=%d direction=%s stream=%p",
                            plan.group_set_id, context.deviceIndex(), plan.device_to_host ? "D2H" : "H2D",
                            reinterpret_cast<void*>(context.stream()));
        return StrategyResult::failed(TransferStatus::DEVICE_IO_ERROR);
    }
    return StrategyResult::done();
}

static constexpr size_t kStagedAlignment = 16;

StrategyResult StagedSmDeviceHostCopyStrategy::tryExecute(const DeviceHostCopyPlan&    plan,
                                                          const DeviceHostCopyOptions& options,
                                                          const DeviceHostCopyExecutionContext& context) {
    if (!options.staged_sm_copy_enabled || plan.copy_tiles.empty()) {
        return StrategyResult::notApplicable();
    }

    if (plan.copy_tiles.size() < options.staged_sm_min_tile_count) {
        return StrategyResult::notApplicable();
    }

    const int device_index = plan.copy_tiles.front().device_index;
    if (!pool_.allowsDevice(device_index)) {
        return StrategyResult::notApplicable();
    }

    const auto limits = pool_.limits();
    if (plan.copy_tiles.size() > limits.max_tiles_per_device
        || plan.copy_tiles.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
        return StrategyResult::notApplicable();
    }

    size_t total_bytes = 0;
    for (const auto& tile : plan.copy_tiles) {
        if (tile.device_index != device_index || tile.bytes > limits.max_staging_bytes_per_device - total_bytes) {
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
        if (current_staging_offset > limits.max_staging_bytes_per_device
            || current_staging_offset > SIZE_MAX - (kStagedAlignment - 1)) {
            return StrategyResult::notApplicable();
        }
        size_t staging_offset = (current_staging_offset + kStagedAlignment - 1) & ~(kStagedAlignment - 1);
        if (staging_offset > limits.max_staging_bytes_per_device
            || tile.bytes > limits.max_staging_bytes_per_device - staging_offset) {
            return StrategyResult::notApplicable();
        }

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

    auto acquired = pool_.tryAcquire();
    if (acquired.status == StagedCopyScratchPool::AcquireStatus::DISABLED) {
        return StrategyResult::failed(TransferStatus::DEVICE_IO_ERROR);
    }
    if (acquired.status == StagedCopyScratchPool::AcquireStatus::EXHAUSTED) {
        return StrategyResult::notApplicable();
    }
    auto& lease = *acquired.lease;
    const auto status = execStagedMemoryCopy(staged_params, lease.scratchFor(device_index), context);
    if (status == StagedMemoryCopyStatus::SUCCESS) {
        return StrategyResult::done();
    }
    if (status == StagedMemoryCopyStatus::NOT_SUPPORTED || status == StagedMemoryCopyStatus::RESOURCE_EXHAUSTED) {
        return StrategyResult::notApplicable();
    }
    if (status == StagedMemoryCopyStatus::UNSAFE) {
        lease.quarantine();
    }
    RTP_LLM_LOG_WARNING("staged copy failed group_set=%zu device=%d direction=%s stream=%p status=%d",
                        plan.group_set_id, context.deviceIndex(), plan.device_to_host ? "D2H" : "H2D",
                        reinterpret_cast<void*>(context.stream()), static_cast<int>(status));
    return StrategyResult::failed(TransferStatus::DEVICE_IO_ERROR);
}

}  // namespace rtp_llm
