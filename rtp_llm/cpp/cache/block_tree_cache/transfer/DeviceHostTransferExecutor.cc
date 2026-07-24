#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceHostTransferExecutor.h"

#include <cstring>
#include <map>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/DeviceBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/host/HostBlockPool.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/models_py/bindings/NoBlockCopy.h"

namespace rtp_llm {

DeviceHostTransferExecutor::DeviceHostTransferExecutor(DeviceHostCopyOptions options): options_(std::move(options)) {
    strategies_.push_back(std::make_unique<StagedSmDeviceHostCopyStrategy>());
    strategies_.push_back(std::make_unique<CudaBatchDeviceHostCopyStrategy>());
    strategies_.push_back(std::make_unique<GenericMultiCopyDeviceHostCopyStrategy>());
}

DeviceHostTransferExecutor::~DeviceHostTransferExecutor() = default;

TransferStatus DeviceHostTransferExecutor::execute(const TransferDescriptor& desc, const GroupSet& group) {
    bool device_to_host = (desc.source_tier == Tier::DEVICE && desc.target_tier == Tier::HOST);
    return lowerAndExecute(desc, group, device_to_host);
}

TransferStatus DeviceHostTransferExecutor::lowerAndExecute(const TransferDescriptor& desc,
                                                           const GroupSet&           group,
                                                           bool                      device_to_host) {
    TransferStatus lower_status = TransferStatus::OK;
    auto           plan         = lowerPlan(desc, group, device_to_host, lower_status);
    if (lower_status != TransferStatus::OK) {
        return lower_status;
    }

    if (plan.copy_tiles.empty()) {
        RTP_LLM_LOG_WARNING("%s copy plan has no copyable device block group=%zu",
                            device_to_host ? "D2H" : "H2D",
                            desc.group_set_id);
        return TransferStatus::INVALID_ARGS;
    }

    for (const auto& zero_tile : plan.zero_tiles) {
        std::memset(zero_tile.host_addr, 0, zero_tile.bytes);
    }

    auto device_plans = splitByDevice(plan);
    for (const auto& device_plan : device_plans) {
        auto status = executeStrategies(device_plan);
        if (status != TransferStatus::OK) {
            return status;
        }
    }
    return TransferStatus::OK;
}

DeviceHostCopyPlan DeviceHostTransferExecutor::lowerPlan(const TransferDescriptor& desc,
                                                         const GroupSet&           group,
                                                         bool                      device_to_host,
                                                         TransferStatus&           out_status) const {
    DeviceHostCopyPlan plan;
    plan.device_to_host     = device_to_host;
    plan.group_set_id = desc.group_set_id;
    out_status              = TransferStatus::OK;

    const auto host_block = desc.host_block;
    auto&      host_pool  = *group.hostPool();

    void* host_base = host_pool.blockBuffer(host_block).addr;
    if (!host_base) {
        RTP_LLM_LOG_WARNING("null host address for block %d", host_block);
        out_status = TransferStatus::DEVICE_IO_ERROR;
        return plan;
    }

    const size_t required_host_bytes = group.payloadBytes();

    plan.host.base          = host_base;
    plan.host.payload_bytes = required_host_bytes;

    const auto& device_blocks = desc.device_blocks;
    const auto& device_pools  = group.devicePools();

    int  first_device_index = -1;
    bool single_device      = true;

    size_t host_offset = 0;
    for (size_t local_group_index = 0; local_group_index < group.groupIds().size(); ++local_group_index) {
        const auto& group_base = group.groupAt(local_group_index);
        const auto  device_block = device_blocks[local_group_index];
        const bool  has_device_block = !isNullBlockIdx(device_block);
        auto&       device_pool = *device_pools[local_group_index];
        const int   pool_device_index = device_pool.deviceIndex();

        if (has_device_block) {
            if (first_device_index < 0) {
                first_device_index = pool_device_index;
            } else if (pool_device_index != first_device_index) {
                single_device = false;
            }
        }

        for (size_t local_layer_index = 0; local_layer_index < group_base.layer_ids.size(); ++local_layer_index) {
            const size_t kv_bytes    = group_base.kv_block_stride_bytes;
            const size_t scale_bytes = group_base.kv_scale_stride_bytes;
            const size_t layer_bytes = kv_bytes + scale_bytes;
            auto*        layer_host_addr = static_cast<uint8_t*>(host_base) + host_offset;

            if (!has_device_block) {
                if (device_to_host) {
                    plan.zero_tiles.push_back(HostZeroTile{layer_host_addr, layer_bytes});
                }
                host_offset += layer_bytes;
                continue;
            }

            const auto buffers =
                device_pool.convertIndexToBuffer(static_cast<int>(local_layer_index), device_block);
            const auto append_tile = [&](size_t buffer_index, size_t logical_bytes, size_t layer_offset) {
                if (logical_bytes == 0) {
                    return true;
                }
                if (buffer_index >= buffers.size() || buffers[buffer_index].addr == nullptr
                    || buffers[buffer_index].size_bytes < logical_bytes) {
                    RTP_LLM_LOG_WARNING(
                        "physical buffer cannot cover logical payload group_set=%zu local_group=%zu "
                        "group_id=%zu local_layer=%zu buffer=%zu physical=%zu logical=%zu block=%d",
                        desc.group_set_id,
                        local_group_index,
                        group.groupIds()[local_group_index],
                        local_layer_index,
                        buffer_index,
                        buffer_index < buffers.size() ? buffers[buffer_index].size_bytes : 0,
                        logical_bytes,
                        device_block);
                    return false;
                }
                DeviceHostCopyTile tile;
                tile.host_addr         = layer_host_addr + layer_offset;
                tile.device_addr       = buffers[buffer_index].addr;
                tile.host_offset       = host_offset + layer_offset;
                tile.bytes             = logical_bytes;
                tile.device_index      = pool_device_index;
                tile.local_group_index = local_group_index;
                tile.local_layer_index = local_layer_index;
                plan.copy_tiles.push_back(tile);
                return true;
            };
            if (!append_tile(0, kv_bytes, 0) || !append_tile(1, scale_bytes, kv_bytes)) {
                out_status = TransferStatus::INVALID_ARGS;
                return plan;
            }
            host_offset += layer_bytes;
        }
    }

    if (host_offset != required_host_bytes) {
        RTP_LLM_LOG_WARNING("logical payload drift group_set=%zu lowered=%zu expected=%zu",
                            desc.group_set_id,
                            host_offset,
                            required_host_bytes);
        out_status = TransferStatus::INVALID_ARGS;
        return plan;
    }

    plan.single_device = single_device;
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
    RTP_LLM_LOG_WARNING("no strategy handled copy plan group=%zu", plan.group_set_id);
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
            sub.device_to_host     = plan.device_to_host;
            sub.single_device      = true;
            sub.group_set_id = plan.group_set_id;
            sub.host               = plan.host;
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
