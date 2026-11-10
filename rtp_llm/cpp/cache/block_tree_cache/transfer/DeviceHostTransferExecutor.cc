#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceHostTransferExecutor.h"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <map>
#include <string_view>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DeviceBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/CrcTransferService.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceHostCopyGeometry.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
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
                                                       size_t                max_descriptors_per_batch,
                                                       DeviceHostCopyOptions options,
                                                       std::shared_ptr<BlockTreeCacheMetricsReporter> metrics_reporter,
                                                       std::shared_ptr<CrcTransferService>            crc_service,
                                                       BlockTreeTaskPool*                             load_task_pool):
    TransferExecutor(transfer_task_pool, max_descriptors_per_batch, std::move(metrics_reporter), load_task_pool),
    crc_service_(std::move(crc_service)),
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

TransferResult DeviceHostTransferExecutor::executeBatch(const std::vector<HostBufferView>&     hosts,
                                                        const std::vector<TransferDescriptor>& descriptors,
                                                        const std::vector<const GroupSet*>&    group_sets) {
    if (hosts.empty() || hosts.size() != descriptors.size() || hosts.size() != group_sets.size()
        || std::any_of(group_sets.begin(), group_sets.end(), [](const GroupSet* group) { return group == nullptr; })) {
        return TransferStatus::INVALID_ARGS;
    }
    const bool enable_crc = group_sets.front()->crcEnabled();
    if (std::any_of(group_sets.begin(), group_sets.end(), [enable_crc](const GroupSet* group) {
            return group->crcEnabled() != enable_crc;
        })) {
        return TransferStatus::INVALID_ARGS;
    }
    if (enable_crc) {
        return crc_service_ ? crc_service_->copy(hosts, descriptors, group_sets) : TransferStatus::INVALID_ARGS;
    }
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
        for (const auto& tile : plan.copy_tiles) {
            if (tile.device_index != device_index) {
                return TransferStatus::INVALID_ARGS;
            }
        }
        if (device_index < 0) {
            contexts.push_back({std::make_shared<DeviceHostCopyStreams>(device_index, 0, 0),
                                plan.device_to_host ? DeviceHostCopyDirection::D2H : DeviceHostCopyDirection::H2D});
            continue;
        }
        std::call_once(copy_streams_once_,
                       [this, device_index] { copy_streams_ = acquireDeviceHostCopyStreams(device_index); });
        if (copy_streams_->device_index != device_index) {
            RTP_LLM_LOG_WARNING("copy plan device=%d differs from executor device=%d group_set=%zu",
                                device_index,
                                copy_streams_->device_index,
                                plan.group_set_id);
            return TransferStatus::INVALID_ARGS;
        }
        const DeviceHostCopyExecutionContext context{
            copy_streams_, plan.device_to_host ? DeviceHostCopyDirection::D2H : DeviceHostCopyDirection::H2D};
        contexts.push_back(context);
    }
    for (size_t plan_index = 0; plan_index < plans.size(); ++plan_index) {
        const auto& plan    = plans[plan_index];
        const auto& context = contexts[plan_index];
        bool        handled = false;
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
    RTP_LLM_CHECK_WITH_INFO(!descriptors.empty() && hosts.size() == descriptors.size()
                                && group_sets.size() == descriptors.size(),
                            "invalid device-host batch dimensions");
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
        for (const auto& tile : getDeviceHostCopyTiles(group_set, device_blocks)) {
            const auto& pool     = *group_set.devicePools()[tile.member_group_id];
            const auto  addr     = reinterpret_cast<uintptr_t>(tile.buffer.addr);
            const auto  base     = reinterpret_cast<uintptr_t>(pool.getBaseAddress());
            const auto  capacity = pool.getTotalSizeBytes();
            RTP_LLM_CHECK_WITH_INFO(tile.host_offset <= host.payload_bytes
                                        && tile.buffer.size_bytes <= host.payload_bytes - tile.host_offset,
                                    "device-host tile exceeds host payload");
            RTP_LLM_CHECK_WITH_INFO(tile.buffer.is_cuda == (pool.deviceIndex() >= 0)
                                        && (!tile.buffer.is_cuda || tile.buffer.device_index == pool.deviceIndex())
                                        && base <= UINTPTR_MAX - capacity && addr >= base && addr - base <= capacity
                                        && tile.buffer.size_bytes <= capacity - (addr - base),
                                    "invalid device-host buffer");
            const int device = pool.deviceIndex();
            auto&     plan   = plans_by_device[device];
            if (plan.copy_tiles.empty()) {
                plan.device_to_host = device_to_host;
                plan.group_set_id   = descriptor.group_set_id;
                plan.host           = host;
            }
            plan.copy_tiles.push_back(DeviceHostCopyTile{static_cast<uint8_t*>(host.base) + tile.host_offset,
                                                         tile.buffer.addr,
                                                         tile.buffer.size_bytes,
                                                         device,
                                                         tile.member_group_id,
                                                         tile.local_layer_index});
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
