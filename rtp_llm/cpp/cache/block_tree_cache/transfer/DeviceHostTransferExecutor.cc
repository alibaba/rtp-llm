#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceHostTransferExecutor.h"

#include <map>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DeviceBlockPool.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/models_py/bindings/NoBlockCopy.h"

namespace rtp_llm {

DeviceHostTransferExecutor::DeviceHostTransferExecutor(DeviceHostCopyOptions options): options_(std::move(options)) {
    strategies_.push_back(std::make_unique<StagedSmDeviceHostCopyStrategy>());
    strategies_.push_back(std::make_unique<CudaBatchDeviceHostCopyStrategy>());
    strategies_.push_back(std::make_unique<GenericMultiCopyDeviceHostCopyStrategy>());
}

DeviceHostTransferExecutor::~DeviceHostTransferExecutor() = default;

TransferStatus DeviceHostTransferExecutor::deviceToHost(const TransferDescriptor& desc,
                                                        const GroupSet&           group_set,
                                                        HostBufferView            host) {
    return deviceToHost({desc}, {&group_set}, {host});
}

TransferStatus DeviceHostTransferExecutor::hostToDevice(HostBufferView            host,
                                                        const TransferDescriptor& desc,
                                                        const GroupSet&           group_set) {
    return hostToDevice({host}, {desc}, {&group_set});
}

TransferStatus DeviceHostTransferExecutor::deviceToHost(const std::vector<TransferDescriptor>& descriptors,
                                                        const std::vector<const GroupSet*>&    group_sets,
                                                        const std::vector<HostBufferView>&     hosts) {
    return lowerAndExecute(descriptors, group_sets, /*device_to_host=*/true, hosts);
}

TransferStatus DeviceHostTransferExecutor::hostToDevice(const std::vector<HostBufferView>&     hosts,
                                                        const std::vector<TransferDescriptor>& descriptors,
                                                        const std::vector<const GroupSet*>&    group_sets) {
    return lowerAndExecute(descriptors, group_sets, /*device_to_host=*/false, hosts);
}

TransferStatus DeviceHostTransferExecutor::lowerAndExecute(const std::vector<TransferDescriptor>& descriptors,
                                                           const std::vector<const GroupSet*>&    group_sets,
                                                           bool                                   device_to_host,
                                                           const std::vector<HostBufferView>&     hosts) {
    DeviceHostCopyPlan plan;
    plan.device_to_host = device_to_host;
    plan.group_set_id   = descriptors.front().group_set_id;
    plan.host           = hosts.front();

    const auto log_failure_candidates = [&](const char* phase, TransferStatus status) {
        RTP_LLM_LOG_WARNING("%s batch failed phase=%s status=%d descriptors=%zu",
                            device_to_host ? "D2H" : "H2D",
                            phase,
                            static_cast<int>(status),
                            descriptors.size());
        for (size_t descriptor_index = 0; descriptor_index < descriptors.size(); ++descriptor_index) {
            RTP_LLM_LOG_WARNING("%s batch failure candidate descriptor=%zu %s",
                                device_to_host ? "D2H" : "H2D",
                                descriptor_index,
                                descriptors[descriptor_index].debugString().c_str());
        }
    };

    int  first_device_index = -1;
    bool single_device      = true;
    for (size_t descriptor_index = 0; descriptor_index < descriptors.size(); ++descriptor_index) {
        auto descriptor_plan = lowerPlan(
            descriptors[descriptor_index], *group_sets[descriptor_index], device_to_host, hosts[descriptor_index]);

        for (const auto& tile : descriptor_plan.copy_tiles) {
            if (first_device_index < 0) {
                first_device_index = tile.device_index;
            } else if (tile.device_index != first_device_index) {
                single_device = false;
            }
            plan.copy_tiles.push_back(tile);
        }
    }
    plan.single_device = single_device;

    auto device_plans = splitByDevice(plan);
    for (const auto& device_plan : device_plans) {
        const auto status = executeStrategies(device_plan);
        if (status != TransferStatus::OK) {
            log_failure_candidates("execute", status);
            return status;
        }
    }
    return TransferStatus::OK;
}

DeviceHostCopyPlan DeviceHostTransferExecutor::lowerPlan(const TransferDescriptor& desc,
                                                         const GroupSet&           group_set,
                                                         bool                      device_to_host,
                                                         HostBufferView            host) const {
    DeviceHostCopyPlan plan;
    plan.device_to_host = device_to_host;
    plan.group_set_id   = desc.group_set_id;
    const size_t required_host_bytes = group_set.payloadBytes();
    plan.host.base          = host.base;
    plan.host.payload_bytes = required_host_bytes;

    const std::vector<BlockIdxType>& device_blocks = desc.blocksAt(Tier::DEVICE);
    const auto&                      device_pools  = group_set.devicePools();

    size_t host_offset = 0;
    for (size_t member_group_id = 0; member_group_id < group_set.groupIds().size(); ++member_group_id) {
        const auto& group_base        = group_set.groupAt(member_group_id);
        const auto  device_block      = device_blocks[member_group_id];
        auto&       device_pool       = *device_pools[member_group_id];
        const int   pool_device_index = device_pool.deviceIndex();

        for (size_t local_layer_index = 0; local_layer_index < group_base.layer_ids.size(); ++local_layer_index) {
            const size_t kv_bytes        = group_base.kv_block_stride_bytes;
            const size_t scale_bytes     = group_base.kv_scale_stride_bytes;
            const size_t layer_bytes     = kv_bytes + scale_bytes;
            auto*        layer_host_addr = static_cast<uint8_t*>(host.base) + host_offset;

            const auto buffers = device_pool.convertIndexToBuffer(static_cast<int>(local_layer_index), device_block);
            const auto append_tile = [&](size_t buffer_index, size_t logical_bytes, size_t layer_offset) {
                if (logical_bytes == 0) {
                    return;
                }
                DeviceHostCopyTile tile;
                tile.host_addr         = layer_host_addr + layer_offset;
                tile.device_addr       = buffers[buffer_index].addr;
                tile.host_offset       = host_offset + layer_offset;
                tile.bytes             = logical_bytes;
                tile.device_index      = pool_device_index;
                tile.member_group_id   = member_group_id;
                tile.local_layer_index = local_layer_index;
                plan.copy_tiles.push_back(tile);
            };
            append_tile(0, kv_bytes, 0);
            append_tile(1, scale_bytes, kv_bytes);
            host_offset += layer_bytes;
        }
    }
    return plan;
}

TransferStatus DeviceHostTransferExecutor::executeStrategies(const DeviceHostCopyPlan& plan) {
    for (auto& strategy : strategies_) {
        auto result = strategy->tryExecute(plan, options_);
        switch (result.status) {
            case StrategyStatus::DONE:
                return TransferStatus::OK;
            case StrategyStatus::FAILED:
                return result.copy_status;
            case StrategyStatus::NOT_APPLICABLE:
                continue;
        }
    }
    RTP_LLM_LOG_WARNING("no strategy handled copy plan group_set=%zu", plan.group_set_id);
    return TransferStatus::DEVICE_IO_ERROR;
}

std::vector<DeviceHostCopyPlan> DeviceHostTransferExecutor::splitByDevice(const DeviceHostCopyPlan& plan) {
    if (plan.copy_tiles.empty() || plan.single_device) {
        return {plan};
    }

    std::map<int, DeviceHostCopyPlan> by_device;
    for (const auto& tile : plan.copy_tiles) {
        auto& sub = by_device[tile.device_index];
        if (sub.copy_tiles.empty()) {
            sub.device_to_host = plan.device_to_host;
            sub.single_device  = true;
            sub.group_set_id   = plan.group_set_id;
            sub.host           = plan.host;
        }
        sub.copy_tiles.push_back(tile);
    }

    std::vector<DeviceHostCopyPlan> result;
    result.reserve(by_device.size());
    for (auto& [_, sub_plan] : by_device) {
        result.push_back(std::move(sub_plan));
    }
    return result;
}

}  // namespace rtp_llm
