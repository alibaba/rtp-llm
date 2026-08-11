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

TransferStatus DeviceHostTransferExecutor::execute(HostBufferView            host,
                                                   const TransferDescriptor& desc,
                                                   const GroupSet&           group_set) {
    auto [status, plans] = generatePlan(desc, group_set, host);
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
DeviceHostTransferExecutor::generatePlan(const TransferDescriptor& desc,
                                         const GroupSet&           group_set,
                                         HostBufferView            host) const {
    const size_t required_host_bytes = group_set.payloadBytes();
    if (!isValidHostBufferView(host, required_host_bytes, required_host_bytes)) {
        RTP_LLM_LOG_WARNING("invalid host buffer group=%zu payload=%zu capacity=%zu required=%zu",
                            desc.group_set_id,
                            host.payload_bytes,
                            host.capacity_bytes,
                            required_host_bytes);
        const auto status = host.base == nullptr ? TransferStatus::DEVICE_IO_ERROR : TransferStatus::INVALID_ARGS;
        return {status, {}};
    }

    const bool device_to_host = desc.target_tier != Tier::DEVICE;
    const std::vector<BlockIdxType>& device_blocks = desc.blocksAt(Tier::DEVICE);
    const auto&                       device_pools  = group_set.devicePools();
    std::map<int, DeviceHostCopyPlan> plans_by_device;
    size_t host_offset = 0;
    for (size_t member_group_id = 0; member_group_id < group_set.groupIds().size(); ++member_group_id) {
        const auto& group_base        = group_set.groupAt(member_group_id);
        auto&       device_pool       = *device_pools[member_group_id];
        for (size_t local_layer_index = 0; local_layer_index < group_base.layer_ids.size(); ++local_layer_index) {
            const size_t kv_bytes        = group_base.kv_block_stride_bytes;
            const size_t scale_bytes     = group_base.kv_scale_stride_bytes;
            const size_t layer_bytes     = kv_bytes + scale_bytes;
            auto*        layer_host_addr = static_cast<uint8_t*>(host.base) + host_offset;
            const auto buffers = device_pool.convertIndexToBuffer(static_cast<int>(local_layer_index), device_blocks[member_group_id]);
            const auto append_tile = [&](size_t buffer_index, size_t logical_bytes, size_t layer_offset) {
                if (logical_bytes == 0) {
                    return;
                }
                auto& plan = plans_by_device[device_pool.deviceIndex()];
                if (plan.copy_tiles.empty()) {
                    plan.device_to_host     = device_to_host;
                    plan.group_set_id       = desc.group_set_id;
                    plan.host.base          = host.base;
                    plan.host.payload_bytes = required_host_bytes;
                }
                DeviceHostCopyTile tile;
                tile.host_addr         = layer_host_addr + layer_offset;
                tile.device_addr       = buffers[buffer_index].addr;
                tile.host_offset       = host_offset + layer_offset;
                tile.bytes             = logical_bytes;
                tile.device_index      = device_pool.deviceIndex();
                tile.member_group_id   = member_group_id;
                tile.local_layer_index = local_layer_index;
                plan.copy_tiles.push_back(tile);
            };
            append_tile(0, kv_bytes, 0);
            append_tile(1, scale_bytes, kv_bytes);
            host_offset += layer_bytes;
        }
    }

    if (plans_by_device.empty()) {
        RTP_LLM_LOG_WARNING("%s copy plan generated no copy tile group_set=%zu",
                            device_to_host ? "D2H" : "H2D",
                            desc.group_set_id);
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
