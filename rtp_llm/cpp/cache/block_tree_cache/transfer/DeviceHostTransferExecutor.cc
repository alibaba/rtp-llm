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

TransferStatus DeviceHostTransferExecutor::execute(const TransferDescriptor&     desc,
                                               const ComponentGroup&         group,
                                               const std::vector<Component>& components) {
    bool device_to_host = (desc.source_tier == Tier::DEVICE && desc.target_tier == Tier::HOST);
    return lowerAndExecute(desc, group, components, device_to_host);
}

TransferStatus DeviceHostTransferExecutor::lowerAndExecute(const TransferDescriptor&     desc,
                                                       const ComponentGroup&         group,
                                                       const std::vector<Component>& components,
                                                       bool                          device_to_host) {
    TransferStatus lower_status = TransferStatus::OK;
    auto       plan         = lowerPlan(desc, group, components, device_to_host, lower_status);
    if (lower_status != TransferStatus::OK) {
        return lower_status;
    }

    if (plan.copy_tiles.empty()) {
        RTP_LLM_LOG_WARNING("%s copy plan has no copyable device block group=%d",
                            device_to_host ? "D2H" : "H2D",
                            desc.component_group_id);
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

DeviceHostCopyPlan DeviceHostTransferExecutor::lowerPlan(const TransferDescriptor&     desc,
                                                         const ComponentGroup&         group,
                                                         const std::vector<Component>& components,
                                                         bool                          device_to_host,
                                                         TransferStatus&               out_status) const {
    DeviceHostCopyPlan plan;
    plan.device_to_host     = device_to_host;
    plan.component_group_id = desc.component_group_id;
    out_status              = TransferStatus::OK;

    const auto host_block = desc.host_block;
    auto&      host_pool  = *group.hostPool();

    void* host_base = host_pool.blockBuffer(host_block).addr;
    if (!host_base) {
        RTP_LLM_LOG_WARNING("null host address for block %d", host_block);
        out_status = TransferStatus::DEVICE_IO_ERROR;
        return plan;
    }

    const auto&  layout              = group.layout();
    const size_t required_host_bytes = layout.payloadBytes();

    plan.host.base          = host_base;
    plan.host.payload_bytes = required_host_bytes;

    const auto& device_blocks = desc.device_blocks;
    const auto& device_pools  = group.devicePools();

    int  first_device_index = -1;
    bool single_device      = true;

    for (const auto& slice : layout.slices()) {
        const int        component_index = group.componentIndices()[slice.component_idx];
        const Component& component       = components[static_cast<size_t>(component_index)];

        const auto   device_block     = device_blocks[slice.component_idx];
        const bool   has_device_block = !isNullBlockIdx(device_block);
        const size_t expected_bytes   = component.layerBytes(slice.layer_idx);
        auto*        slot_host_addr   = static_cast<uint8_t*>(host_base) + slice.offset_bytes;
        if (!has_device_block) {
            if (device_to_host) {
                plan.zero_tiles.push_back(HostZeroTile{slot_host_addr, expected_bytes});
            }
            continue;
        }

        auto&     device_pool       = *device_pools[slice.component_idx];
        const int pool_device_index = device_pool.deviceIndex();
        if (first_device_index < 0) {
            first_device_index = pool_device_index;
        } else if (pool_device_index != first_device_index) {
            single_device = false;
        }

        // Component layer order matches pool slots; model layer ids do not.
        auto   buffers           = device_pool.convertIndexToBuffer(static_cast<int>(slice.layer_idx), device_block);
        size_t slot_device_bytes = 0;
        for (const auto& buffer : buffers) {
            if (!buffer.addr || buffer.size_bytes == 0) {
                RTP_LLM_LOG_WARNING("null device buffer component=%d layer=%zu block=%d",
                                    component_index,
                                    slice.layer_idx,
                                    device_block);
                out_status = TransferStatus::DEVICE_IO_ERROR;
                return plan;
            }
            DeviceHostCopyTile tile;
            tile.host_addr       = slot_host_addr + slot_device_bytes;
            tile.device_addr     = buffer.addr;
            tile.host_offset     = slice.offset_bytes + slot_device_bytes;
            tile.bytes           = buffer.size_bytes;
            tile.device_index    = pool_device_index;
            tile.component_index = component_index;
            tile.layer_id        = static_cast<int>(slice.layer_idx);
            plan.copy_tiles.push_back(tile);
            slot_device_bytes += buffer.size_bytes;
        }

        if (slot_device_bytes != expected_bytes) {
            RTP_LLM_LOG_WARNING("device bytes %zu != layer bytes %zu component=%d layer=%zu block=%d",
                                slot_device_bytes,
                                expected_bytes,
                                component_index,
                                slice.layer_idx,
                                device_block);
            out_status = TransferStatus::INVALID_ARGS;
            return plan;
        }
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
    RTP_LLM_LOG_WARNING("no strategy handled copy plan group=%d", plan.component_group_id);
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
            sub.component_group_id = plan.component_group_id;
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
