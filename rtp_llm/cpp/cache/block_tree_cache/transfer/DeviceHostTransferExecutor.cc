#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceHostTransferExecutor.h"

#include <algorithm>
#include <cstdlib>
#include <map>
#include <string_view>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DeviceBlockPool.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/models_py/bindings/NoBlockCopy.h"

namespace rtp_llm {

namespace {

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
                                                       StagedCopyScratchPool& scratch_pool,
                                                       size_t                max_descriptors_per_batch,
                                                       DeviceHostCopyOptions options,
                                                       std::shared_ptr<BlockTreeCacheMetricsReporter> metrics_reporter,
                                                       std::vector<int> allowed_devices):
    TransferExecutor(transfer_task_pool, max_descriptors_per_batch, std::move(metrics_reporter)),
    options_(std::move(options)) {
    for (int device_index : allowed_devices) {
        streams_by_device_.emplace(device_index, acquireDeviceHostCopyStreams(device_index));
    }
    strategies_.push_back(std::make_unique<CudaBatchDeviceHostCopyStrategy>());
    strategies_.push_back(std::make_unique<StagedSmDeviceHostCopyStrategy>(scratch_pool));
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
    std::vector<DeviceHostCopyExecutionContext> contexts;
    contexts.reserve(plans.size());
    for (const auto& plan : plans) {
        if (plan.copy_tiles.empty()) {
            return TransferStatus::INVALID_ARGS;
        }
        const int device_index = plan.copy_tiles.front().device_index;
        const auto stream_it = streams_by_device_.find(device_index);
        if (stream_it == streams_by_device_.end()) {
            RTP_LLM_LOG_WARNING("copy plan uses unconfigured device=%d group_set=%zu", device_index, plan.group_set_id);
            return TransferStatus::INVALID_ARGS;
        }
        for (const auto& tile : plan.copy_tiles) {
            if (tile.device_index != device_index) {
                return TransferStatus::INVALID_ARGS;
            }
        }
        const DeviceHostCopyExecutionContext context{
            stream_it->second, plan.device_to_host ? DeviceHostCopyDirection::D2H : DeviceHostCopyDirection::H2D};
        if (context.owner->unsafe.load(std::memory_order_acquire)) {
            RTP_LLM_LOG_WARNING("copy stream unsafe device=%d direction=%s stream=%p group_set=%zu",
                                device_index, plan.device_to_host ? "D2H" : "H2D",
                                reinterpret_cast<void*>(context.stream()), plan.group_set_id);
            return TransferStatus::DEVICE_IO_ERROR;
        }
        contexts.push_back(context);
    }
    for (size_t plan_index = 0; plan_index < plans.size(); ++plan_index) {
        const auto& plan = plans[plan_index];
        const auto& context = contexts[plan_index];
        bool handled = false;
        for (auto& strategy : strategies_) {
            auto result = strategy->tryExecute(plan, options_, context);
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
    const bool                        device_to_host = descriptors.front().target_tier != Tier::DEVICE;
    std::map<int, DeviceHostCopyPlan> plans_by_device;
    for (size_t descriptor_index = 0; descriptor_index < descriptors.size(); ++descriptor_index) {
        const auto&  descriptor          = descriptors[descriptor_index];
        const auto&  group_set           = *group_sets[descriptor_index];
        const auto&  host                = hosts[descriptor_index];
        const size_t required_host_bytes = group_set.payloadBytes();
        if (!isValidHostBufferView(host, required_host_bytes, required_host_bytes)) {
            RTP_LLM_LOG_WARNING(
                "invalid device-host batch item index=%zu group=%zu", descriptor_index, descriptor.group_set_id);
            return {host.base == nullptr ? TransferStatus::DEVICE_IO_ERROR : TransferStatus::INVALID_ARGS, {}};
        }

        const std::vector<BlockIdxType>& device_blocks = descriptor.blocksAt(Tier::DEVICE);
        const auto&                      device_pools  = group_set.devicePools();
        size_t                           host_offset   = 0;
        for (size_t member_group_id = 0; member_group_id < group_set.groupIds().size(); ++member_group_id) {
            const auto& group_base  = group_set.groupAt(member_group_id);
            auto&       device_pool = *device_pools[member_group_id];
            for (size_t local_layer_index = 0; local_layer_index < group_base.layer_ids.size(); ++local_layer_index) {
                const size_t kv_bytes        = group_base.kv_block_stride_bytes;
                const size_t scale_bytes     = group_base.kv_scale_stride_bytes;
                const size_t layer_bytes     = kv_bytes + scale_bytes;
                auto*        layer_host_addr = static_cast<uint8_t*>(host.base) + host_offset;
                const auto   buffers         = device_pool.convertIndexToBuffer(static_cast<int>(local_layer_index),
                                                                      device_blocks[member_group_id]);
                const auto   append_tile     = [&](size_t buffer_index, size_t logical_bytes, size_t layer_offset) {
                    if (logical_bytes == 0) {
                        return;
                    }
                    auto& plan = plans_by_device[device_pool.deviceIndex()];
                    if (plan.copy_tiles.empty()) {
                        plan.device_to_host = device_to_host;
                        plan.group_set_id   = descriptor.group_set_id;
                        plan.host           = host;
                    }
                    plan.copy_tiles.push_back(DeviceHostCopyTile{layer_host_addr + layer_offset,
                                                                 buffers[buffer_index].addr,
                                                                 host_offset + layer_offset,
                                                                 logical_bytes,
                                                                 device_pool.deviceIndex(),
                                                                 member_group_id,
                                                                 local_layer_index});
                };
                append_tile(0, kv_bytes, 0);
                append_tile(1, scale_bytes, kv_bytes);
                host_offset += layer_bytes;
            }
        }
    }

    if (plans_by_device.empty()) {
        RTP_LLM_LOG_WARNING("%s batch generated no copy tile", device_to_host ? "D2H" : "H2D");
        return {TransferStatus::INVALID_ARGS, {}};
    }

    std::vector<DeviceHostCopyPlan> plans;
    plans.reserve(plans_by_device.size());
    for (auto& [_, plan] : plans_by_device) {
        plans.push_back(std::move(plan));
    }
    return {TransferStatus::OK, std::move(plans)};
}

}  // namespace rtp_llm
