#include "rtp_llm/cpp/cache/block_tree_cache/transfer/legacy/LegacyTransferEngine.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/CrcCopyExecutor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include <algorithm>

namespace rtp_llm {
LegacyTransferEngine::LegacyTransferEngine(std::vector<GroupSetPtr> groups,
                                           bool                     enable_disk_cache,
                                           DeviceHostCopyOptions    device_host_options,
                                           size_t                   device_disk_staging_block_count,
                                           size_t                   max_descriptors_per_batch,
                                           size_t                   transfer_worker_count,
                                           size_t                   transfer_queue_max_size,
                                           std::shared_ptr<BlockTreeCacheMetricsReporter> metrics_reporter,
                                           int64_t                                        world_rank):
    group_sets_(std::move(groups)) {
    std::shared_ptr<CrcCopyExecutor> load_crc_executor, evict_crc_executor;
    if (std::any_of(group_sets_.begin(), group_sets_.end(), [enable_disk_cache](const GroupSetPtr& group) {
            return group->crcEnabled() && !group->devicePools().empty()
                   && (group->hostPool() || (enable_disk_cache && group->diskPool()));
        })) {
        const auto crc_groups = crcTransferGroups(group_sets_, enable_disk_cache);
        load_crc_executor     = std::make_shared<CrcCopyExecutor>(
            crc_groups, max_descriptors_per_batch, transfer_worker_count, world_rank, nullptr, false);
        evict_crc_executor = std::make_shared<CrcCopyExecutor>(
            crc_groups, max_descriptors_per_batch, transfer_worker_count, world_rank, nullptr, false);
    }
    load_transfer_task_pool_ =
        std::make_unique<BlockTreeTaskPool>(transfer_worker_count, transfer_queue_max_size, "btc_load");
    transfer_task_pool_ =
        std::make_unique<BlockTreeTaskPool>(transfer_worker_count, transfer_queue_max_size, "btc_evict");
    RTP_LLM_CHECK(load_transfer_task_pool_->start());
    RTP_LLM_CHECK(transfer_task_pool_->start());
    device_host_executor_ = std::make_unique<DeviceHostTransferExecutor>(*transfer_task_pool_,
                                                                         max_descriptors_per_batch,
                                                                         std::move(device_host_options),
                                                                         metrics_reporter,
                                                                         std::move(load_crc_executor),
                                                                         std::move(evict_crc_executor),
                                                                         load_transfer_task_pool_.get(),
                                                                         world_rank);
    host_disk_executor_   = std::make_unique<HostDiskTransferExecutor>(
        *transfer_task_pool_, max_descriptors_per_batch, metrics_reporter, load_transfer_task_pool_.get(), world_rank);
    if (enable_disk_cache && !group_sets_.empty()) {
        device_disk_executor_ = std::make_unique<DeviceDiskTransferExecutor>(*device_host_executor_,
                                                                             *host_disk_executor_,
                                                                             group_sets_,
                                                                             device_disk_staging_block_count,
                                                                             *transfer_task_pool_,
                                                                             max_descriptors_per_batch,
                                                                             std::move(metrics_reporter),
                                                                             load_transfer_task_pool_.get(),
                                                                             world_rank);
    }
}
LegacyTransferEngine::~LegacyTransferEngine() {
    drain();
}
std::shared_ptr<TransferAsyncContext> LegacyTransferEngine::execute(TransferTask                 task,
                                                                    std::vector<const GroupSet*> groups) {
    block_tree_cache_detail::ScopeRollback unsubmitted([proof = task.completionProof()] {
        if (proof) proof->report(true);
    });
    const auto& descriptors = task.descriptors();
    const auto  source      = descriptors.front().source_tier;
    const auto  target      = descriptors.front().target_tier;
    if (source == Tier::DISK && target == Tier::DEVICE) {
        unsubmitted.dismiss();
        return device_disk_executor_->execute(std::move(task), groups);
    }
    std::vector<HostBufferView> hosts;
    hosts.reserve(descriptors.size());
    for (size_t i = 0; i < descriptors.size(); ++i) {
        const auto buffer = groups[i]->hostPool()->blockBuffer(descriptors[i].singleBlockAt(Tier::HOST));
        hosts.push_back({buffer.addr, buffer.payload_bytes, buffer.stride_bytes});
    }
    unsubmitted.dismiss();
    if (source == Tier::DEVICE || target == Tier::DEVICE)
        return device_host_executor_->execute(std::move(task), std::move(hosts), std::move(groups));
    return host_disk_executor_->execute(std::move(task), std::move(hosts), std::move(groups));
}

void LegacyTransferEngine::stopAdmission() {
    load_transfer_task_pool_->stopAdmission();
    transfer_task_pool_->stopAdmission();
}
void LegacyTransferEngine::waitForIdle() {
    load_transfer_task_pool_->waitForIdle();
    transfer_task_pool_->waitForIdle();
}
void LegacyTransferEngine::drain() {
    stopAdmission();
    cancelPendingStagingTransfers();
    waitForIdle();
}
void LegacyTransferEngine::cancelPendingStagingTransfers() {
    if (device_disk_executor_)
        device_disk_executor_->cancelPendingTransfers();
}
BlockTreeQueueSizes LegacyTransferEngine::queueSizes() const {
    const auto load       = load_transfer_task_pool_->queueSizes();
    const auto background = transfer_task_pool_->queueSizes();
    return {
        load.load + background.load, load.background + background.background, load.completion + background.completion};
}

}  // namespace rtp_llm
