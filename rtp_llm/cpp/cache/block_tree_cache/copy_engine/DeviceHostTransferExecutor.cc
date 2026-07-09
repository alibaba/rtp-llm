#include "rtp_llm/cpp/cache/block_tree_cache/copy_engine/DeviceHostTransferExecutor.h"

#include <cstring>
#include <map>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/DeviceBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/host/HostBlockPool.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/models_py/bindings/NoBlockCopy.h"

namespace rtp_llm {

DeviceHostTransferExecutor::DeviceHostTransferExecutor(DeviceHostCopyOptions options):
    options_(std::move(options)) {
    strategies_.push_back(std::make_unique<StagedSmDeviceHostCopyStrategy>());
    strategies_.push_back(std::make_unique<CudaBatchDeviceHostCopyStrategy>());
    strategies_.push_back(std::make_unique<GenericMultiCopyDeviceHostCopyStrategy>());
}

DeviceHostTransferExecutor::~DeviceHostTransferExecutor() = default;

CopyStatus DeviceHostTransferExecutor::execute(const TransferDescriptor& desc,
                                                const ResolvedGroupLayout& layout) {
    bool device_to_host = (desc.source_tier == Tier::DEVICE && desc.target_tier == Tier::HOST);
    return lowerAndExecute(desc, layout, device_to_host);
}

CopyStatus DeviceHostTransferExecutor::lowerAndExecute(const TransferDescriptor&  desc,
                                                       const ResolvedGroupLayout& layout,
                                                       bool                       device_to_host) {
    CopyStatus lower_status = CopyStatus::OK;
    auto       plan         = lowerPlan(desc, layout, device_to_host, lower_status);
    if (lower_status != CopyStatus::OK) {
        return lower_status;
    }

    if (plan.copy_tiles.empty()) {
        if (!plan.zero_tiles.empty()) {
            std::memset(plan.host.base, 0, plan.host.payload_bytes);
        }
        return CopyStatus::OK;
    }

    for (const auto& zero_tile : plan.zero_tiles) {
        std::memset(zero_tile.host_addr, 0, zero_tile.bytes);
    }

    auto device_plans = splitByDevice(plan);
    for (const auto& device_plan : device_plans) {
        auto status = executeStrategies(device_plan);
        if (status != CopyStatus::OK) {
            return status;
        }
    }
    return CopyStatus::OK;
}

DeviceHostCopyPlan DeviceHostTransferExecutor::lowerPlan(const TransferDescriptor&  desc,
                                                         const ResolvedGroupLayout& layout,
                                                         bool                       device_to_host,
                                                         CopyStatus&                out_status) const {
    DeviceHostCopyPlan plan;
    plan.device_to_host     = device_to_host;
    plan.component_group_id = desc.component_group_id;
    out_status              = CopyStatus::OK;

    const auto  host_block = desc.host_block;
    auto&       host_pool  = *layout.host_pool;

    if (!host_pool.isAllocated(host_block)) {
        RTP_LLM_LOG_WARNING("invalid or unallocated host block %d", host_block);
        out_status = CopyStatus::INVALID_ARGS;
        return plan;
    }

    void* host_base = host_pool.blockBuffer(host_block).addr;
    if (!host_base) {
        RTP_LLM_LOG_WARNING("null host address for block %d", host_block);
        out_status = CopyStatus::DEVICE_IO_ERROR;
        return plan;
    }

    const size_t required_host_bytes = layout.layout_bytes;
    if (required_host_bytes != host_pool.payloadBytes()) {
        RTP_LLM_LOG_WARNING("component layout bytes %zu != host payload bytes %zu",
                            required_host_bytes,
                            host_pool.payloadBytes());
        out_status = CopyStatus::INVALID_ARGS;
        return plan;
    }

    plan.host.base          = host_base;
    plan.host.payload_bytes = required_host_bytes;

    const auto& device_blocks = desc.device_blocks;

    size_t host_offset          = 0;
    int    first_device_index   = -1;
    bool   single_device        = true;

    for (size_t component_idx = 0; component_idx < layout.components.size(); ++component_idx) {
        const auto& component    = layout.components[component_idx];
        const auto  device_block = device_blocks[component_idx];

        const bool has_device_block = !isNullBlockIdx(device_block);
        auto&      device_pool      = *component.device_pool;
        if (has_device_block && !device_pool.isAllocated(device_block)) {
            RTP_LLM_LOG_WARNING("invalid or unallocated device block %d", device_block);
            out_status = CopyStatus::INVALID_ARGS;
            return plan;
        }

        const int pool_device_index = device_pool.deviceIndex();
        if (has_device_block) {
            if (first_device_index < 0) {
                first_device_index = pool_device_index;
            } else if (pool_device_index != first_device_index) {
                single_device = false;
            }
        }
        for (const auto& slot : component.layer_slots) {
            if (slot.stride_bytes == 0) {
                RTP_LLM_LOG_WARNING("zero-sized layer slot layer=%d", slot.layer_id);
                out_status = CopyStatus::INVALID_ARGS;
                return plan;
            }

            auto* slot_host_addr = static_cast<uint8_t*>(host_base) + host_offset;
            if (!has_device_block) {
                if (device_to_host) {
                    plan.zero_tiles.push_back(HostZeroTile{slot_host_addr, slot.stride_bytes});
                }
                host_offset += slot.stride_bytes;
                continue;
            }

            auto   buffers           = device_pool.blockBuffers(slot.layer_id, device_block);
            size_t slot_device_bytes = 0;
            for (const auto& buffer : buffers) {
                if (!buffer.addr || buffer.bytes == 0) {
                    RTP_LLM_LOG_WARNING("null device buffer layer=%d block=%d", slot.layer_id, device_block);
                    out_status = CopyStatus::DEVICE_IO_ERROR;
                    return plan;
                }
                DeviceHostCopyTile tile;
                tile.host_addr       = slot_host_addr + slot_device_bytes;
                tile.device_addr     = buffer.addr;
                tile.host_offset     = host_offset + slot_device_bytes;
                tile.bytes           = buffer.bytes;
                tile.device_index    = pool_device_index;
                tile.component_index = component.component_index;
                tile.layer_id        = slot.layer_id;
                plan.copy_tiles.push_back(tile);
                slot_device_bytes += buffer.bytes;
            }

            if (slot_device_bytes != slot.stride_bytes) {
                RTP_LLM_LOG_WARNING("device bytes %zu != slot stride %zu layer=%d block=%d",
                                    slot_device_bytes,
                                    slot.stride_bytes,
                                    slot.layer_id,
                                    device_block);
                out_status = CopyStatus::INVALID_ARGS;
                return plan;
            }
            host_offset += slot.stride_bytes;
        }
    }

    plan.single_device = single_device;
    return plan;
}

CopyStatus DeviceHostTransferExecutor::executeStrategies(const DeviceHostCopyPlan& plan) {
    for (auto& strategy : strategies_) {
        auto result = strategy->tryExecute(plan, options_);
        switch (result.status) {
            case StrategyStatus::DONE:
                return CopyStatus::OK;
            case StrategyStatus::FAILED:
                return result.copy_status;
            case StrategyStatus::NOT_APPLICABLE:
                continue;
        }
    }
    RTP_LLM_LOG_WARNING("no strategy handled copy plan group=%d", plan.component_group_id);
    return CopyStatus::DEVICE_IO_ERROR;
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
