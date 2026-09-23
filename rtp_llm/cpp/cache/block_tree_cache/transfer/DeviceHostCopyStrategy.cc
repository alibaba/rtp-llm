#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceHostCopyStrategy.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>

#include <torch/torch.h>

#include "rtp_llm/cpp/utils/CudacoreDiagnostics.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/models_py/bindings/NoBlockCopy.h"

namespace rtp_llm {

namespace {

// A fatal device error was already recorded: the instance must not accept new
// transfers, and no further GPU work must be submitted for them.
void rejectTransfersAfterFatalIncident() {
    if (fatalCudacoreIncidentActive()) {
        // Followers must honor the original collection window too. Returning
        // early could let an upper-layer RTP_LLM_FAIL abort the process first.
        failCopyWithDiagnostics("execution", __FILE__, __LINE__, "fatal CUDA incident already active");
    }
}

void annotateTransferContext(const DeviceHostCopyPlan& plan) {
    // A device plan may merge multiple descriptors. Its group and host fields
    // then describe only the first descriptor, so they must not label the
    // entire failing batch in the incident manifest.
    if (plan.mixed_descriptors) {
        return;
    }
    annotateFatalCudaTransferContext(
        plan.group_set_id, plan.device_to_host, reinterpret_cast<uintptr_t>(plan.host.base), plan.host.payload_bytes);
}

}  // namespace

bool validateDeviceHostCopyPlan(const DeviceHostCopyPlan& plan) {
    if (plan.copy_tiles.empty()) {
        RTP_LLM_LOG_WARNING("empty device-host copy plan");
        return false;
    }
    const int device      = plan.copy_tiles.front().device_index;
    size_t    total_bytes = 0;
    for (size_t i = 0; i < plan.copy_tiles.size(); ++i) {
        const auto& tile  = plan.copy_tiles[i];
        const auto  host  = reinterpret_cast<uintptr_t>(tile.host_addr);
        const auto  gpu   = reinterpret_cast<uintptr_t>(tile.device_addr);
        bool        valid = host != 0 && gpu != 0 && tile.bytes != 0 && tile.device_index == device
                     && tile.bytes <= UINTPTR_MAX - host && tile.bytes <= UINTPTR_MAX - gpu
                     && tile.bytes <= std::numeric_limits<size_t>::max() - total_bytes;
        if (!plan.origins.empty()) {
            if (tile.origin_index >= plan.origins.size()) {
                valid = false;
            } else {
                const auto& origin = plan.origins[tile.origin_index];
                const auto  base   = reinterpret_cast<uintptr_t>(origin.host.base);
                valid =
                    valid && origin.host.payload_bytes <= origin.host.capacity_bytes
                    && origin.host.capacity_bytes <= UINTPTR_MAX - base && tile.host_offset <= origin.host.payload_bytes
                    && tile.bytes <= origin.host.payload_bytes - tile.host_offset && host == base + tile.host_offset
                    && tile.layer_offset <= origin.layer_stride && tile.bytes <= origin.layer_stride - tile.layer_offset
                    && tile.bytes <= tile.device_buffer_bytes
                    && origin.device_pool_bytes <= UINTPTR_MAX - origin.device_pool_base
                    && gpu >= origin.device_pool_base && gpu - origin.device_pool_base <= origin.device_pool_bytes
                    && tile.device_buffer_bytes <= origin.device_pool_bytes - (gpu - origin.device_pool_base);
            }
        }
        if (!valid) {
            RTP_LLM_LOG_WARNING("invalid copy plan tile=%zu host=%p gpu=%p bytes=%zu device=%d expected_device=%d "
                                "origin=%zu host_offset=%zu layer_offset=%zu",
                                i,
                                tile.host_addr,
                                tile.device_addr,
                                tile.bytes,
                                tile.device_index,
                                device,
                                tile.origin_index,
                                tile.host_offset,
                                tile.layer_offset);
            return false;
        }
        total_bytes += tile.bytes;
    }
    return true;
}

std::string deviceHostCopyPlanJson(const DeviceHostCopyPlan& plan) {
    std::ostringstream out;
    auto               range = [&](uintptr_t base, size_t bytes) {
        out << "{\"start\":" << base << ",\"len\":" << bytes << ",\"end\":";
        if (bytes <= UINTPTR_MAX - base) {
            out << base + bytes;
        } else {
            out << "null";
        }
        out << '}';
    };
    auto lifetime =
        [&](const BlockDiagnosticSnapshot& before, const std::shared_ptr<IBlockPool>& pool, BlockIdxType block) {
            if (!pool) {
                out << "null";
                return;
            }
            const auto after    = pool->diagnosticSnapshot(block);
            auto       snapshot = [&](const BlockDiagnosticSnapshot& value) {
                out << "{\"query_ok\":" << (value.query_ok ? "true" : "false")
                    << ",\"valid\":" << (value.valid ? "true" : "false")
                    << ",\"allocated\":" << (value.allocated ? "true" : "false")
                    << ",\"allocation_generation\":" << value.allocation_generation
                    << ",\"external_reference\":" << (value.external_reference ? "true" : "false")
                    << ",\"tree_references\":" << value.tree_references << ",\"references_by_type\":[";
                for (size_t i = 0; i < value.references_by_type.size(); ++i) {
                    if (i) {
                        out << ',';
                    }
                    out << value.references_by_type[i];
                }
                out << "]}";
            };
            out << "{\"reference_order\":[\"CACHE\",\"LOAD\",\"EVICTION\",\"STORE\"],\"at_plan\":";
            snapshot(before);
            out << ",\"at_error\":";
            snapshot(after);
            out << ",\"reallocated_since_plan\":";
            if (before.valid && after.valid) {
                out << (before.allocation_generation != after.allocation_generation ? "true" : "false");
            } else {
                out << "null";
            }
            out << '}';
        };
    out << "{\"direction\":\"" << (plan.device_to_host ? "D2H" : "H2D")
        << "\",\"mixed_descriptors\":" << (plan.mixed_descriptors ? "true" : "false")
        << ",\"range_convention\":\"[start,end)\",\"origins\":[";
    for (size_t i = 0; i < plan.origins.size(); ++i) {
        const auto& o = plan.origins[i];
        if (i) {
            out << ',';
        }
        out << "{\"index\":" << i << ",\"descriptor_index\":" << o.descriptor_index
            << ",\"group_set_id\":" << o.group_set_id << ",\"member_group_id\":" << o.member_group_id
            << ",\"topology_group_id\":" << o.topology_group_id << ",\"path_index\":" << o.path_index
            << ",\"node_identity\":" << o.node << ",\"source_tier\":\"" << tierName(o.source_tier)
            << "\",\"target_tier\":\"" << tierName(o.target_tier) << "\",\"device_block\":" << o.device_block
            << ",\"other_block\":" << o.other_block << ",\"layer_stride\":" << o.layer_stride
            << ",\"kv_bytes\":" << o.kv_bytes << ",\"scale_bytes\":" << o.scale_bytes << ",\"mem_cache_payload\":";
        range(reinterpret_cast<uintptr_t>(o.host.base), o.host.payload_bytes);
        out << ",\"mem_cache_capacity\":";
        range(reinterpret_cast<uintptr_t>(o.host.base), o.host.capacity_bytes);
        out << ",\"gpu_kvcache_pool\":";
        range(o.device_pool_base, o.device_pool_bytes);
        out << ",\"mem_cache_pool\":";
        if (o.host_pool_base) {
            range(o.host_pool_base, o.host_pool_bytes);
        } else {
            out << "null";
        }
        out << ",\"mem_cache_pool_stride\":" << o.host_pool_stride;
        out << ",\"device_block_lifetime\":";
        lifetime(o.device_at_plan, o.device_pool, o.device_block);
        out << ",\"host_block_lifetime\":";
        lifetime(o.host_at_plan, o.host_pool, o.other_block);
        out << '}';
    }
    out << "],\"tile_count\":" << plan.copy_tiles.size() << ",\"tiles_truncated\":false,\"tiles\":[";
    for (size_t i = 0; i < plan.copy_tiles.size(); ++i) {
        const auto& t = plan.copy_tiles[i];
        if (i) {
            out << ',';
        }
        out << "{\"index\":" << i << ",\"origin\":" << t.origin_index << ",\"device_index\":" << t.device_index
            << ",\"member_group_id\":" << t.member_group_id << ",\"local_layer_index\":" << t.local_layer_index
            << ",\"buffer_index\":" << t.buffer_index << ",\"device_buffer_bytes\":" << t.device_buffer_bytes
            << ",\"host_offset\":" << t.host_offset << ",\"within_layer_offset\":" << t.layer_offset
            << ",\"gpu_kvcache\":";
        range(reinterpret_cast<uintptr_t>(t.device_addr), t.bytes);
        out << ",\"gpu_buffer_capacity\":";
        range(reinterpret_cast<uintptr_t>(t.device_addr), t.device_buffer_bytes);
        out << ",\"mem_cache\":";
        range(reinterpret_cast<uintptr_t>(t.host_addr), t.bytes);
        out << ",\"src\":" << reinterpret_cast<uintptr_t>(plan.device_to_host ? t.device_addr : t.host_addr)
            << ",\"dst\":" << reinterpret_cast<uintptr_t>(plan.device_to_host ? t.host_addr : t.device_addr)
            << ",\"len\":" << t.bytes << '}';
    }
    // Cold-path overlap analysis. Source overlap can be intentional; these are
    // observations, not claims about causality or the original Xid.
    auto overlaps = [&](bool destination) {
        std::vector<std::pair<uintptr_t, size_t>> sorted;
        sorted.reserve(plan.copy_tiles.size());
        for (size_t i = 0; i < plan.copy_tiles.size(); ++i) {
            const auto& t = plan.copy_tiles[i];
            sorted.emplace_back(
                reinterpret_cast<uintptr_t>(destination == plan.device_to_host ? t.host_addr : t.device_addr), i);
        }
        std::sort(sorted.begin(), sorted.end());
        out << '[';
        uintptr_t furthest_end   = 0;
        size_t    furthest_index = 0;
        bool      first          = true;
        for (const auto& item : sorted) {
            const size_t bytes = plan.copy_tiles[item.second].bytes;
            if (bytes > UINTPTR_MAX - item.first) {
                continue;
            }
            if (item.first < furthest_end) {
                if (!first) {
                    out << ',';
                }
                out << '[' << furthest_index << ',' << item.second << ']';
                first = false;
            }
            if (item.first + bytes > furthest_end) {
                furthest_end   = item.first + bytes;
                furthest_index = item.second;
            }
        }
        out << ']';
    };
    out << "],\"source_overlap_examples\":";
    overlaps(false);
    out << ",\"destination_overlap_examples\":";
    overlaps(true);
    out << '}';
    return out.str();
}

namespace {
std::string copyPlanEvidence(const void* plan) {
    return deviceHostCopyPlanJson(*static_cast<const DeviceHostCopyPlan*>(plan));
}
}  // namespace

StrategyResult GenericMultiCopyDeviceHostCopyStrategy::tryExecute(const DeviceHostCopyPlan& plan,
                                                                  const DeviceHostCopyOptions& /*options*/) {
    rejectTransfersAfterFatalIncident();
    CudacoreCopyScope evidence(&plan, copyPlanEvidence, "generic");
    if (!validateDeviceHostCopyPlan(plan)) {
        failCopyWithDiagnostics("invariant", __FILE__, __LINE__, "invalid device-host copy plan");
    }
    bool execution_attempted = false;
    try {
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
        execution_attempted = true;
        execNoBlockCopy(mc);
    } catch (const std::exception& error) {
        // CUDAGuard / PyTorch can throw before our numeric CUDA checks run.
        // Capture while this thread still owns the complete copy plan.
        if (isFatalCudaException(error)) {
            (void)recordFirstFatalCudaError(
                buildCudaExceptionRecord(error, FatalCudaErrorSite::DeviceHostCopy, __FILE__, __LINE__));
        }
        annotateTransferContext(plan);
        if (execution_attempted || isFatalCudaException(error)) {
            failCopyWithDiagnostics("execution", __FILE__, __LINE__, error.what());
        }
        throw;
    } catch (...) {
        annotateTransferContext(plan);
        if (execution_attempted) {
            failCopyWithDiagnostics("execution", __FILE__, __LINE__, "unknown generic copy exception");
        }
        throw;
    }
    return StrategyResult::done();
}

StrategyResult CudaBatchDeviceHostCopyStrategy::tryExecute(const DeviceHostCopyPlan&    plan,
                                                           const DeviceHostCopyOptions& options) {
    if (!options.cuda_batch_copy_enabled) {
        return StrategyResult::notApplicable();
    }

    CudacoreCopyScope evidence(&plan, copyPlanEvidence, "cuda_batch");
    if (!validateDeviceHostCopyPlan(plan)) {
        failCopyWithDiagnostics("invariant", __FILE__, __LINE__, "invalid device-host copy plan");
    }
    const int device_index = plan.copy_tiles.front().device_index;
    if (device_index < 0) {
        return StrategyResult::notApplicable();
    }

    rejectTransfersAfterFatalIncident();

    BatchedMemoryCopyParams params;
    params.device_index = device_index;
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

    BatchedMemoryCopyStatus status;
    try {
        status = execBatchedMemoryCopy(params);
    } catch (const std::exception& error) {
        // Device selection / stream initialization can fail before submission.
        if (isFatalCudaException(error)) {
            (void)recordFirstFatalCudaError(
                buildCudaExceptionRecord(error, FatalCudaErrorSite::DeviceHostCopy, __FILE__, __LINE__));
        }
        annotateTransferContext(plan);
        failCopyWithDiagnostics("execution", __FILE__, __LINE__, error.what());
    }
    if (status == BatchedMemoryCopyStatus::NOT_SUPPORTED) {
        return StrategyResult::notApplicable();
    }
    if (status == BatchedMemoryCopyStatus::EXECUTION_FAILED) {
        // Attach the transfer identity to the first fatal error record raised by
        // the copy executor, if this failure was classified as fatal.
        annotateTransferContext(plan);
        failCopyWithDiagnostics("execution", __FILE__, __LINE__, "CUDA batch submission/completion failed");
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

StagedMemoryCopyStatus StagedSmDeviceHostCopyStrategy::executeStagedCopy(const StagedMemoryCopyParams& params,
                                                                         StagedMemoryCopyScratch*      scratch) {
    return execStagedMemoryCopy(params, scratch);
}

StrategyResult StagedSmDeviceHostCopyStrategy::tryExecute(const DeviceHostCopyPlan&    plan,
                                                          const DeviceHostCopyOptions& options) {
    if (!options.staged_sm_copy_enabled) {
        return StrategyResult::notApplicable();
    }

    if (plan.copy_tiles.size() < options.staged_sm_min_tile_count) {
        return StrategyResult::notApplicable();
    }

    CudacoreCopyScope evidence(&plan, copyPlanEvidence, "staged_sm");
    if (!validateDeviceHostCopyPlan(plan)) {
        failCopyWithDiagnostics("invariant", __FILE__, __LINE__, "invalid device-host copy plan");
    }
    const int device_index = plan.copy_tiles.front().device_index;
    if (device_index < 0) {
        return StrategyResult::notApplicable();
    }

    size_t total_bytes = 0;
    for (const auto& tile : plan.copy_tiles) {
        total_bytes += tile.bytes;
    }

    if (total_bytes < options.staged_sm_min_bytes) {
        return StrategyResult::notApplicable();
    }

    rejectTransfersAfterFatalIncident();

    // Build staged params with compact host segments
    StagedMemoryCopyParams staged_params;
    staged_params.host_base    = plan.host.base;
    staged_params.device_index = device_index;
    staged_params.direction    = plan.device_to_host ? StagedMemoryCopyDirection::D2H : StagedMemoryCopyDirection::H2D;

    size_t current_staging_offset = 0;
    staged_params.tiles.reserve(plan.copy_tiles.size());
    staged_params.host_segments.reserve(plan.copy_tiles.size());

    for (const auto& tile : plan.copy_tiles) {
        if (current_staging_offset > std::numeric_limits<size_t>::max() - (kStagedAlignment - 1)) {
            failCopyWithDiagnostics("invariant", __FILE__, __LINE__, "staging offset/length overflow");
        }
        size_t staging_offset = alignUp(current_staging_offset, kStagedAlignment);
        if (tile.bytes > std::numeric_limits<size_t>::max() - staging_offset) {
            failCopyWithDiagnostics("invariant", __FILE__, __LINE__, "staging offset/length overflow");
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

    std::lock_guard<std::mutex> lock(scratch_mutex_);
    auto&                       entry = scratch_by_device_[device_index];
    if (!entry) {
        entry               = std::make_unique<StagedMemoryCopyScratch>();
        entry->device_index = device_index;
    }

    StagedMemoryCopyStatus status;
    try {
        status = executeStagedCopy(staged_params, entry.get());
    } catch (const std::exception& error) {
        if (isFatalCudaException(error)) {
            (void)recordFirstFatalCudaError(
                buildCudaExceptionRecord(error, FatalCudaErrorSite::DeviceHostCopy, __FILE__, __LINE__));
        }
        annotateTransferContext(plan);
        failCopyWithDiagnostics("execution", __FILE__, __LINE__, error.what());
    }
    switch (status) {
        case StagedMemoryCopyStatus::SUCCESS:
            return StrategyResult::done();
        case StagedMemoryCopyStatus::NOT_SUPPORTED:
        case StagedMemoryCopyStatus::RESOURCE_EXHAUSTED:
            // Only these outcomes guarantee no copy work was submitted.
            return StrategyResult::notApplicable();
        case StagedMemoryCopyStatus::INVALID_ARGUMENT:
            failCopyWithDiagnostics("invariant", __FILE__, __LINE__, "invalid staged copy parameters");
        case StagedMemoryCopyStatus::EXECUTION_FAILED:
            annotateTransferContext(plan);
            failCopyWithDiagnostics("execution", __FILE__, __LINE__, "staged copy execution failed");
    }
    failCopyWithDiagnostics("invariant", __FILE__, __LINE__, "unknown staged copy status");
}

}  // namespace rtp_llm
