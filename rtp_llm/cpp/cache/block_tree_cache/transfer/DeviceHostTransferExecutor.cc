#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceHostTransferExecutor.h"

#include <algorithm>
#include <map>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DeviceBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/CrcTransferService.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceHostCopyGeometry.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/models_py/bindings/NoBlockCopy.h"

namespace rtp_llm {

DeviceHostTransferExecutor::DeviceHostTransferExecutor(BlockTreeTaskPool&    transfer_task_pool,
                                                       size_t                max_descriptors_per_batch,
                                                       DeviceHostCopyOptions options,
                                                       std::shared_ptr<BlockTreeCacheMetricsReporter> metrics_reporter,
                                                       std::shared_ptr<CrcTransferService>            crc_service):
    TransferExecutor(transfer_task_pool, max_descriptors_per_batch, std::move(metrics_reporter)),
    crc_service_(std::move(crc_service)),
    options_(std::move(options)) {
    strategies_.push_back(std::make_unique<CudaBatchDeviceHostCopyStrategy>());
    strategies_.push_back(std::make_unique<StagedSmDeviceHostCopyStrategy>());
    strategies_.push_back(std::make_unique<GenericMultiCopyDeviceHostCopyStrategy>());
}

TransferStatus DeviceHostTransferExecutor::executeBatch(const std::vector<HostBufferView>&     hosts,
                                                        const std::vector<TransferDescriptor>& descriptors,
                                                        const std::vector<const GroupSet*>&    group_sets) {
    if (hosts.empty() || hosts.size() != descriptors.size() || hosts.size() != group_sets.size()
        || std::any_of(group_sets.begin(), group_sets.end(), [](const GroupSet* group) { return group == nullptr; })) {
        return TransferStatus::INVALID_ARGS;
    }
    const size_t protected_count =
        std::count_if(group_sets.begin(), group_sets.end(), [](const GroupSet* group) { return group->crcEnabled(); });
    if (protected_count == group_sets.size()) {
        return crc_service_ ? crc_service_->copy(hosts, descriptors, group_sets) : TransferStatus::INVALID_ARGS;
    }
    if (protected_count != 0) {
        std::vector<HostBufferView>     protected_hosts, plain_hosts;
        std::vector<TransferDescriptor> protected_descriptors, plain_descriptors;
        std::vector<const GroupSet*>    protected_groups, plain_groups;
        for (size_t index = 0; index < group_sets.size(); ++index) {
            const bool is_protected = group_sets[index]->crcEnabled();
            (is_protected ? protected_hosts : plain_hosts).push_back(hosts[index]);
            (is_protected ? protected_descriptors : plain_descriptors).push_back(descriptors[index]);
            (is_protected ? protected_groups : plain_groups).push_back(group_sets[index]);
        }
        // Check all protected records first. A bad CRC must not scatter even
        // the unprotected portion of a mixed host/GPU restore batch.
        const auto status = crc_service_ ?
                                crc_service_->copy(protected_hosts, protected_descriptors, protected_groups) :
                                TransferStatus::INVALID_ARGS;
        return status == TransferStatus::OK ? executeBatch(plain_hosts, plain_descriptors, plain_groups) : status;
    }
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
        try {
            visitDeviceHostCopyTiles(
                group_set, device_blocks, [&](const BlockInfo& buffer, size_t offset, size_t member, size_t layer) {
                    const int device = group_set.devicePools()[member]->deviceIndex();
                    auto&     plan   = plans_by_device[device];
                    if (plan.copy_tiles.empty()) {
                        plan.device_to_host = device_to_host;
                        plan.group_set_id   = descriptor.group_set_id;
                        plan.host           = host;
                    }
                    plan.copy_tiles.push_back(DeviceHostCopyTile{static_cast<uint8_t*>(host.base) + offset,
                                                                 buffer.addr,
                                                                 offset,
                                                                 buffer.size_bytes,
                                                                 device,
                                                                 member,
                                                                 layer});
                });
        } catch (const std::exception& error) {
            RTP_LLM_LOG_WARNING("invalid device-host backing geometry: %s", error.what());
            return {TransferStatus::INVALID_ARGS, {}};
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
