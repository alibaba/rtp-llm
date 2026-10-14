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
    if (dynamic_cast<const Cuda3DBatchDeviceHostCopyStrategy*>(&strategy) != nullptr) {
        return "cuda_3d_batch";
    }
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
    strategies_.push_back(std::make_unique<Cuda3DBatchDeviceHostCopyStrategy>());
    strategies_.push_back(std::make_unique<CudaBatchDeviceHostCopyStrategy>());
    strategies_.push_back(std::make_unique<StagedSmDeviceHostCopyStrategy>());
    strategies_.push_back(std::make_unique<GenericMultiCopyDeviceHostCopyStrategy>());
    if (const char* selected = std::getenv("BLOCK_TREE_DEVICE_HOST_COPY_PRIORITY"); selected && *selected) {
        const auto found = std::find_if(strategies_.begin(), strategies_.end(), [selected](const auto& strategy) {
            return configurableStrategyName(*strategy) == selected;
        });
        if (found != strategies_.end()) {
            std::rotate(strategies_.begin(), found, found + 1);
        }
    }
}

TransferStatus DeviceHostTransferExecutor::executeBatch(const std::vector<HostBufferView>&     hosts,
                                                        const std::vector<TransferDescriptor>& descriptors,
                                                        const std::vector<const GroupSet*>&    group_sets) {
    if (descriptors.empty() || hosts.size() != descriptors.size() || group_sets.size() != descriptors.size()) {
        return TransferStatus::INVALID_ARGS;
    }
    const Tier source = descriptors.front().source_tier;
    const Tier target = descriptors.front().target_tier;
    for (size_t i = 0; i < descriptors.size(); ++i) {
        if (descriptors[i].source_tier != source || descriptors[i].target_tier != target || group_sets[i] == nullptr
            || descriptors[i].blocksAt(Tier::DEVICE).size() != group_sets[i]->devicePools().size()) {
            return TransferStatus::INVALID_ARGS;
        }
    }
    std::vector<DeviceHostCopyPlan> plans;
    std::vector<bool>               handled;
    bool                            planned = false;
    bool                            any_handled = false;
    for (auto& strategy : strategies_) {
        if (auto* copy3d = dynamic_cast<Cuda3DBatchDeviceHostCopyStrategy*>(strategy.get())) {
            // A raw-batch copy must not repeat any plan already completed by an earlier strategy.
            if (any_handled) {
                continue;
            }
            const auto result = copy3d->tryExecute(hosts, descriptors, group_sets);
            if (result.status == StrategyStatus::DONE) {
                return TransferStatus::OK;
            }
            if (result.status == StrategyStatus::FAILED) {
                return result.copy_status;
            }
            continue;
        }
        if (!planned) {
            auto generated = generatePlan(hosts, descriptors, group_sets);
            if (generated.first != TransferStatus::OK) {
                return generated.first;
            }
            plans   = std::move(generated.second);
            handled.assign(plans.size(), false);
            planned = true;
        }
        for (size_t index = 0; index < plans.size(); ++index) {
            if (handled[index]) {
                continue;
            }
            const auto result = strategy->tryExecute(plans[index], options_);
            if (result.status == StrategyStatus::DONE) {
                handled[index] = true;
                any_handled    = true;
            } else if (result.status == StrategyStatus::FAILED) {
                return result.copy_status;
            }
        }
        if (std::all_of(handled.begin(), handled.end(), [](bool done) { return done; })) {
            return TransferStatus::OK;
        }
    }
    if (planned) {
        for (size_t index = 0; index < plans.size(); ++index) {
            if (!handled[index]) {
                RTP_LLM_LOG_WARNING("no strategy handled copy plan group_set=%zu", plans[index].group_set_id);
            }
        }
    }
    return TransferStatus::DEVICE_IO_ERROR;
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
        for (size_t member_group_id = 0; member_group_id < group_set.groupTags().size(); ++member_group_id) {
            const auto& group_tag   = group_set.groupTags()[member_group_id];
            const auto& group_base  = group_set.group(group_tag);
            const auto  layer_ids   = group_set.topologyPtr()->layerIdsForGroup(group_tag);
            auto&       device_pool = *device_pools[member_group_id];
            for (size_t local_layer_index = 0; local_layer_index < layer_ids.size(); ++local_layer_index) {
                auto*      layer_host_addr = static_cast<uint8_t*>(host.base) + host_offset;
                const auto buffers         = device_pool.convertIndexToBuffer(static_cast<int>(local_layer_index),
                                                                      device_blocks[member_group_id]);
                const auto append_tile     = [&](size_t buffer_index, size_t logical_bytes, size_t layer_offset) {
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
                size_t layer_offset = 0;
                if (group_set.usesPhysicalPayloadGeometry()) {
                    for (size_t buffer_index = 0; buffer_index < buffers.size(); ++buffer_index) {
                        append_tile(buffer_index, buffers[buffer_index].size_bytes, layer_offset);
                        layer_offset += buffers[buffer_index].size_bytes;
                    }
                } else {
                    const size_t kv_bytes    = group_base.kvBlockStrideBytes();
                    const size_t scale_bytes = group_base.kvScaleStrideBytes();
                    if (buffers.size() != (scale_bytes == 0 ? 1u : 2u) || buffers[0].size_bytes != kv_bytes
                        || (scale_bytes != 0 && buffers[1].size_bytes != scale_bytes)) {
                        return {TransferStatus::INVALID_ARGS, {}};
                    }
                    append_tile(0, kv_bytes, 0);
                    append_tile(1, scale_bytes, kv_bytes);
                    layer_offset = kv_bytes + scale_bytes;
                }
                host_offset += layer_offset;
            }
        }
        RTP_LLM_CHECK_WITH_INFO(host_offset == required_host_bytes,
                                "device-host payload geometry mismatch: planned=%zu required=%zu",
                                host_offset,
                                required_host_bytes);
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
