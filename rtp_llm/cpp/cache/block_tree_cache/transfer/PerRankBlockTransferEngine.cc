#include "rtp_llm/cpp/cache/block_tree_cache/transfer/PerRankBlockTransferEngine.h"

#include <memory>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DeviceBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DiskBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/HostBlockPool.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {

namespace {

std::shared_ptr<AsyncContext> invalidTransferContext() {
    return std::make_shared<CompletedAsyncContext>(
        ErrorInfo(ErrorCode::INVALID_PARAMS, "invalid block transfer request"));
}

}  // namespace

PerRankBlockTransferEngine::PerRankBlockTransferEngine(std::vector<GroupSetPtr> group_sets,
                                                       bool                     enable_disk_cache,
                                                       DeviceHostCopyOptions    device_host_options,
                                                       size_t                   device_disk_staging_block_count,
                                                       size_t                   max_descriptors_per_batch,
                                                       size_t                   transfer_worker_count,
                                                       size_t transfer_queue_max_size,
                                                       std::shared_ptr<BlockTreeCacheMetricsReporter> metrics_reporter):
    group_sets_(std::move(group_sets)), transfer_worker_count_(transfer_worker_count) {
    RTP_LLM_CHECK(max_descriptors_per_batch > 0);
    RTP_LLM_CHECK(transfer_worker_count > 0);
    transfer_task_pool_ =
        std::make_unique<BlockTreeTaskPool>(transfer_worker_count, transfer_queue_max_size, "BlockTransferEngine");
    RTP_LLM_CHECK(transfer_task_pool_->start());
    device_host_executor_ = std::make_unique<DeviceHostTransferExecutor>(
        *transfer_task_pool_, max_descriptors_per_batch, std::move(device_host_options), metrics_reporter);
    host_disk_executor_ = std::make_unique<HostDiskTransferExecutor>(
        *transfer_task_pool_, max_descriptors_per_batch, metrics_reporter);
    if (enable_disk_cache) {
        device_disk_executor_ = std::make_unique<DeviceDiskTransferExecutor>(*device_host_executor_,
                                                                             *host_disk_executor_,
                                                                             group_sets_,
                                                                             device_disk_staging_block_count,
                                                                             *transfer_task_pool_,
                                                                             max_descriptors_per_batch,
                                                                             std::move(metrics_reporter));
    }
}

PerRankBlockTransferEngine::~PerRankBlockTransferEngine() {
    transfer_task_pool_->stopAdmission();
    cancelPendingStagingTransfers();
    transfer_task_pool_->waitForIdle();
    transfer_task_pool_.reset();
}

void PerRankBlockTransferEngine::cancelPendingStagingTransfers() {
    if (device_disk_executor_ != nullptr) {
        device_disk_executor_->cancelPendingTransfers();
    }
}

BlockTreeQueueSizes PerRankBlockTransferEngine::queueSizes() const {
    return transfer_task_pool_->queueSizes();
}

std::shared_ptr<AsyncContext> PerRankBlockTransferEngine::execute(TransferTask task) {
    const auto& descriptors = task.descriptors();
    if (descriptors.empty()) {
        return invalidTransferContext();
    }
    const Tier                   source = descriptors.front().source_tier;
    const Tier                   target = descriptors.front().target_tier;
    std::vector<const GroupSet*> group_sets;
    std::vector<HostBufferView>  hosts;
    group_sets.reserve(descriptors.size());
    hosts.reserve(descriptors.size());
    for (const auto& descriptor : descriptors) {
        if (descriptor.source_tier != source || descriptor.target_tier != target) {
            return invalidTransferContext();
        }
        const auto* group_set = group_sets_[descriptor.group_set_id].get();
        group_sets.push_back(group_set);
        if (source == Tier::HOST || target == Tier::HOST) {
            hosts.push_back(resolveHostView(*group_set, descriptor.singleBlockAt(Tier::HOST)));
        }
    }

    if (source == Tier::DISK && target == Tier::DEVICE) {
        if (device_disk_executor_ == nullptr) {
            return invalidTransferContext();
        }
        return device_disk_executor_->executeDiskToDevice(std::move(task), group_sets);
    }
    if (source == Tier::DEVICE && target == Tier::DISK) {
        if (device_disk_executor_ == nullptr) {
            return invalidTransferContext();
        }
        return device_disk_executor_->executeDeviceToDisk(std::move(task), *group_sets.front());
    }
    if ((source == Tier::DEVICE && target == Tier::HOST) || (source == Tier::HOST && target == Tier::DEVICE)) {
        return device_host_executor_->execute(std::move(task), std::move(hosts), std::move(group_sets));
    }
    if ((source == Tier::HOST && target == Tier::DISK) || (source == Tier::DISK && target == Tier::HOST)) {
        return host_disk_executor_->execute(std::move(task), std::move(hosts), std::move(group_sets));
    }
    return invalidTransferContext();
}

HostBufferView PerRankBlockTransferEngine::resolveHostView(const GroupSet& group_set, BlockIdxType host_block) {
    const HostBlockBuffer buffer = group_set.hostPool()->blockBuffer(host_block);
    return HostBufferView{buffer.addr, buffer.payload_bytes, buffer.stride_bytes};
}

}  // namespace rtp_llm
