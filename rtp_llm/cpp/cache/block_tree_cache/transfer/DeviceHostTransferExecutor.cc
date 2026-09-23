#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceHostTransferExecutor.h"

#include <algorithm>
#include <cstdlib>
#include <map>
#include <limits>
#include <string_view>
#include <sstream>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DeviceBlockPool.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/CudacoreDiagnostics.h"
#include "rtp_llm/models_py/bindings/NoBlockCopy.h"

namespace rtp_llm {

namespace {

struct PlanningEvidence {
    const std::vector<HostBufferView>&       hosts;
    const std::vector<TransferDescriptor>&   descriptors;
    const std::map<int, DeviceHostCopyPlan>& plans;
};

std::string planningEvidenceJson(const void* data) {
    const auto&        e = *static_cast<const PlanningEvidence*>(data);
    std::ostringstream out;
    out << "{\"phase\":\"generate_plan\",\"inputs\":[";
    for (size_t i = 0; i < e.descriptors.size(); ++i) {
        if (i)
            out << ',';
        const auto& d = e.descriptors[i];
        const auto& h = e.hosts[i];
        out << "{\"descriptor_index\":" << i << ",\"group_set_id\":" << d.group_set_id
            << ",\"host_base\":" << reinterpret_cast<uintptr_t>(h.base) << ",\"host_payload_bytes\":" << h.payload_bytes
            << ",\"host_capacity_bytes\":" << h.capacity_bytes << ",\"source_tier\":" << static_cast<int>(d.source_tier)
            << ",\"target_tier\":" << static_cast<int>(d.target_tier);
        auto blocks = [&](const char* name, const auto& values) {
            out << ",\"" << name << "\":[";
            for (size_t j = 0; j < values.size(); ++j) {
                if (j)
                    out << ',';
                out << values[j];
            }
            out << ']';
        };
        blocks("source_blocks", d.source_blocks);
        blocks("target_blocks", d.target_blocks);
        out << '}';
    }
    out << "],\"partial_plans\":[";
    bool first = true;
    for (const auto& entry : e.plans) {
        if (!first)
            out << ',';
        first = false;
        out << deviceHostCopyPlanJson(entry.second);
    }
    return out.str() + "]}";
}

std::string_view configurableStrategyName(const DeviceHostCopyStrategy& strategy) {
    if (dynamic_cast<const CudaBatchDeviceHostCopyStrategy*>(&strategy) != nullptr) {
        return "cuda_batch";
    }
    if (dynamic_cast<const StagedSmDeviceHostCopyStrategy*>(&strategy) != nullptr) {
        return "sm";
    }
    if (dynamic_cast<const GenericMultiCopyDeviceHostCopyStrategy*>(&strategy) != nullptr) {
        return "generic";
    }
    return {};
}

}  // namespace

DeviceHostTransferExecutor::DeviceHostTransferExecutor(BlockTreeTaskPool&    transfer_task_pool,
                                                       size_t                max_descriptors_per_batch,
                                                       DeviceHostCopyOptions options,
                                                       std::shared_ptr<BlockTreeCacheMetricsReporter> metrics_reporter):
    TransferExecutor(transfer_task_pool, max_descriptors_per_batch, std::move(metrics_reporter)),
    options_(std::move(options)) {
    strategies_.push_back(std::make_unique<CudaBatchDeviceHostCopyStrategy>());
    strategies_.push_back(std::make_unique<StagedSmDeviceHostCopyStrategy>());
    strategies_.push_back(std::make_unique<GenericMultiCopyDeviceHostCopyStrategy>());

    // Promote only a configurable existing strategy. Additional strategies retain
    // their default relative order unless one of these names is selected.
    const char* preferred = std::getenv("BLOCK_TREE_DEVICE_HOST_COPY_PRIORITY");
    if (preferred != nullptr && preferred[0] != '\0') {
        const std::string_view requested(preferred);
        const auto selected = std::find_if(strategies_.begin(), strategies_.end(), [&](const auto& strategy) {
            return configurableStrategyName(*strategy) == requested;
        });
        if (selected == strategies_.end()) {
            RTP_LLM_LOG_WARNING("invalid BLOCK_TREE_DEVICE_HOST_COPY_PRIORITY='%s'; keeping default copy order",
                                preferred);
        } else {
            std::rotate(strategies_.begin(), selected, selected + 1);
        }
    }
}

TransferStatus DeviceHostTransferExecutor::executeBatch(const std::vector<HostBufferView>&     hosts,
                                                        const std::vector<TransferDescriptor>& descriptors,
                                                        const std::vector<const GroupSet*>&    group_sets) {
    auto [status, plans] = generatePlan(hosts, descriptors, group_sets);
    if (status != TransferStatus::OK) {
        return status;
    }
    for (const auto& plan : plans) {
        bool handled = false;
        for (auto& strategy : strategies_) {
            auto result = strategy->tryExecute(plan, options_);
            if (result.status == StrategyStatus::DONE) {
                handled = true;
                break;
            }
            if (result.status == StrategyStatus::FAILED) {
                return result.copy_status;
            }
        }
        if (!handled) {
            RTP_LLM_LOG_WARNING("no strategy handled copy plan group_set=%zu", plan.group_set_id);
            return TransferStatus::DEVICE_IO_ERROR;
        }
    }
    return TransferStatus::OK;
}

std::pair<TransferStatus, std::vector<DeviceHostCopyPlan>>
DeviceHostTransferExecutor::generatePlan(const std::vector<HostBufferView>&     hosts,
                                         const std::vector<TransferDescriptor>& descriptors,
                                         const std::vector<const GroupSet*>&    group_sets) const {
    if (descriptors.empty() || hosts.size() != descriptors.size() || group_sets.size() != descriptors.size()) {
        RTP_LLM_LOG_WARNING("invalid device-host batch dimensions: hosts=%zu descriptors=%zu groups=%zu",
                            hosts.size(),
                            descriptors.size(),
                            group_sets.size());
        return {TransferStatus::INVALID_ARGS, {}};
    }
    const bool                        device_to_host = descriptors.front().target_tier != Tier::DEVICE;
    std::map<int, DeviceHostCopyPlan> plans_by_device;
    const PlanningEvidence            evidence{hosts, descriptors, plans_by_device};
    const auto                        fail = [&](const std::string& message, int line) {
        CudacoreCopyScope scope(&evidence, planningEvidenceJson, "generate_plan");
        failCopyWithDiagnostics("invariant", __FILE__, line, message);
    };
    for (size_t descriptor_index = 0; descriptor_index < descriptors.size(); ++descriptor_index) {
        const auto& descriptor = descriptors[descriptor_index];
        if (group_sets[descriptor_index] == nullptr || !descriptor.isExecutable()
            || (descriptor.source_tier != Tier::DEVICE && descriptor.target_tier != Tier::DEVICE)
            || (descriptor.target_tier != Tier::DEVICE) != device_to_host) {
            RTP_LLM_LOG_WARNING(
                "invalid device-host descriptor index=%zu: %s", descriptor_index, descriptor.debugString().c_str());
            return {TransferStatus::INVALID_ARGS, {}};
        }
        const auto&  group_set           = *group_sets[descriptor_index];
        const auto&  host                = hosts[descriptor_index];
        const size_t required_host_bytes = group_set.payloadBytes();
        if (!isValidHostBufferView(host, required_host_bytes, required_host_bytes)) {
            fail(fmtstr(
                     "invalid device-host batch item index=%zu group=%zu host=%p payload=%zu capacity=%zu required=%zu",
                     descriptor_index,
                     descriptor.group_set_id,
                     host.base,
                     host.payload_bytes,
                     host.capacity_bytes,
                     required_host_bytes),
                 __LINE__);
        }

        const std::vector<BlockIdxType>& device_blocks = descriptor.blocksAt(Tier::DEVICE);
        const auto&                      device_pools  = group_set.devicePools();
        if (device_blocks.size() != group_set.groupIds().size() || device_pools.size() != device_blocks.size()
            || descriptor.group_set_id != group_set.groupSetId()) {
            fail(fmtstr("invalid device-host member count or group id: %s", descriptor.debugString().c_str()),
                 __LINE__);
        }
        size_t host_offset = 0;
        for (size_t member_group_id = 0; member_group_id < group_set.groupIds().size(); ++member_group_id) {
            const auto& group_base = group_set.groupAt(member_group_id);
            const auto  device_snapshot =
                device_pools[member_group_id] ?
                     device_pools[member_group_id]->diagnosticSnapshot(device_blocks[member_group_id]) :
                     BlockDiagnosticSnapshot{};
            if (!device_snapshot.valid) {
                fail(fmtstr("invalid device block: descriptor=%zu member=%zu block=%d",
                            descriptor_index,
                            member_group_id,
                            device_blocks[member_group_id]),
                     __LINE__);
            }
            auto& device_pool = *device_pools[member_group_id];
            auto& plan        = plans_by_device[device_pool.deviceIndex()];
            if (plan.origins.empty()) {
                plan.device_to_host         = device_to_host;
                plan.group_set_id           = descriptor.group_set_id;
                plan.host                   = host;
                plan.first_descriptor_index = descriptor_index;
            } else if (plan.first_descriptor_index != descriptor_index) {
                plan.mixed_descriptors = true;
            }
            if (group_base.kv_scale_stride_bytes
                > std::numeric_limits<size_t>::max() - group_base.kv_block_stride_bytes) {
                fail(fmtstr("device-host layer stride overflow: descriptor=%zu member=%zu kv_bytes=%zu scale_bytes=%zu",
                            descriptor_index,
                            member_group_id,
                            group_base.kv_block_stride_bytes,
                            group_base.kv_scale_stride_bytes),
                     __LINE__);
            }
            DeviceHostCopyOrigin origin;
            origin.descriptor_index  = descriptor_index;
            origin.group_set_id      = descriptor.group_set_id;
            origin.member_group_id   = member_group_id;
            origin.topology_group_id = group_set.groupIds()[member_group_id];
            origin.path_index        = descriptor.path_index;
            origin.node              = reinterpret_cast<uintptr_t>(descriptor.node);
            origin.source_tier       = descriptor.source_tier;
            origin.target_tier       = descriptor.target_tier;
            origin.device_block      = device_blocks[member_group_id];
            origin.other_block = device_to_host ? descriptor.target_blocks.front() : descriptor.source_blocks.front();
            origin.host        = host;
            origin.device_pool = device_pools[member_group_id];
            origin.device_at_plan = device_snapshot;
            if ((descriptor.source_tier == Tier::HOST || descriptor.target_tier == Tier::HOST)
                && group_set.hostPool()) {
                origin.host_pool_base   = reinterpret_cast<uintptr_t>(group_set.hostPool()->getBaseAddress());
                origin.host_pool_bytes  = group_set.hostPool()->getTotalSizeBytes();
                origin.host_pool_stride = group_set.hostPool()->strideBytes();
                origin.host_pool        = group_set.hostPool();
                origin.host_at_plan     = origin.host_pool->diagnosticSnapshot(origin.other_block);
            }
            origin.device_pool_base   = reinterpret_cast<uintptr_t>(device_pool.getBaseAddress());
            origin.device_pool_bytes  = device_pool.getTotalSizeBytes();
            origin.kv_bytes           = group_base.kv_block_stride_bytes;
            origin.scale_bytes        = group_base.kv_scale_stride_bytes;
            origin.layer_stride       = origin.kv_bytes + origin.scale_bytes;
            const size_t origin_index = plan.origins.size();
            plan.origins.push_back(origin);
            for (size_t local_layer_index = 0; local_layer_index < group_base.layer_ids.size(); ++local_layer_index) {
                const size_t kv_bytes    = group_base.kv_block_stride_bytes;
                const size_t scale_bytes = group_base.kv_scale_stride_bytes;
                const size_t layer_bytes = kv_bytes + scale_bytes;
                if (host_offset > host.payload_bytes || layer_bytes > host.payload_bytes - host_offset
                    || reinterpret_cast<uintptr_t>(host.base) > UINTPTR_MAX - host.capacity_bytes) {
                    fail(fmtstr("device-host layer exceeds host view: descriptor=%zu member=%zu layer=%zu "
                                "offset=%zu layer_bytes=%zu payload=%zu capacity=%zu base=%p",
                                descriptor_index,
                                member_group_id,
                                local_layer_index,
                                host_offset,
                                layer_bytes,
                                host.payload_bytes,
                                host.capacity_bytes,
                                host.base),
                         __LINE__);
                }
                const auto buffers     = device_pool.convertIndexToBuffer(static_cast<int>(local_layer_index),
                                                                      device_blocks[member_group_id]);
                const auto append_tile = [&](size_t buffer_index, size_t logical_bytes, size_t layer_offset) {
                    if (logical_bytes == 0) {
                        return;
                    }
                    if (buffer_index >= buffers.size()) {
                        fail(fmtstr("missing device-host buffer: descriptor=%zu member=%zu layer=%zu buffer=%zu",
                                    descriptor_index,
                                    member_group_id,
                                    local_layer_index,
                                    buffer_index),
                             __LINE__);
                    }
                    const auto& buffer   = buffers[buffer_index];
                    const auto  gpu_addr = reinterpret_cast<uintptr_t>(buffer.addr);
                    // A group may copy only a prefix of a physical buffer. Check the
                    // copied span against the layer/host, and the physical view
                    // against its pool. Never require copying padding as payload.
                    // Subtraction-based checks cannot be bypassed by size_t overflow.
                    if (buffer.addr == nullptr || logical_bytes > buffer.size_bytes || layer_offset > layer_bytes
                        || logical_bytes > layer_bytes - layer_offset || layer_offset > host.payload_bytes - host_offset
                        || logical_bytes > host.payload_bytes - host_offset - layer_offset
                        || buffer.is_cuda != (device_pool.deviceIndex() >= 0)
                        || (buffer.is_cuda && buffer.device_index != device_pool.deviceIndex())
                        || origin.device_pool_base > UINTPTR_MAX - origin.device_pool_bytes
                        || gpu_addr < origin.device_pool_base
                        || gpu_addr - origin.device_pool_base > origin.device_pool_bytes
                        || buffer.size_bytes > origin.device_pool_bytes - (gpu_addr - origin.device_pool_base)) {
                        fail(fmtstr("invalid device-host tile: descriptor=%zu member=%zu layer=%zu buffer=%zu "
                                    "host=%p host_offset=%zu within_layer=%zu layer_stride=%zu payload=%zu "
                                    "gpu=%p actual_bytes=%zu logical_bytes=%zu buffer_device=%d plan_device=%d "
                                    "gpu_pool=%p gpu_pool_bytes=%zu",
                                    descriptor_index,
                                    member_group_id,
                                    local_layer_index,
                                    buffer_index,
                                    host.base,
                                    host_offset,
                                    layer_offset,
                                    layer_bytes,
                                    host.payload_bytes,
                                    buffer.addr,
                                    buffer.size_bytes,
                                    logical_bytes,
                                    buffer.device_index,
                                    device_pool.deviceIndex(),
                                    device_pool.getBaseAddress(),
                                    origin.device_pool_bytes),
                             __LINE__);
                    }
                    plan.copy_tiles.push_back(
                        DeviceHostCopyTile{static_cast<uint8_t*>(host.base) + host_offset + layer_offset,
                                           buffer.addr,
                                           host_offset + layer_offset,
                                           logical_bytes,
                                           device_pool.deviceIndex(),
                                           member_group_id,
                                           local_layer_index,
                                           origin_index,
                                           buffer_index,
                                           buffer.size_bytes,
                                           layer_offset});
                };
                append_tile(0, kv_bytes, 0);
                append_tile(1, scale_bytes, kv_bytes);
                host_offset += layer_bytes;
            }
        }
        if (host_offset != required_host_bytes) {
            fail(fmtstr("device-host layout payload mismatch: descriptor=%zu actual=%zu required=%zu",
                        descriptor_index,
                        host_offset,
                        required_host_bytes),
                 __LINE__);
        }
    }

    if (plans_by_device.empty()) {
        fail(fmtstr("%s batch generated no copy tile", device_to_host ? "D2H" : "H2D"), __LINE__);
    }

    std::vector<DeviceHostCopyPlan> plans;
    plans.reserve(plans_by_device.size());
    for (auto& [_, plan] : plans_by_device) {
        plans.push_back(std::move(plan));
    }
    return {TransferStatus::OK, std::move(plans)};
}

}  // namespace rtp_llm
