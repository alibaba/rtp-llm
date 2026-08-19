#include "rtp_llm/cpp/cache/block_tree_cache/transfer/PerRankBlockTransferEngine.h"

#include <algorithm>
#include <exception>
#include <memory>
#include <string>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DeviceBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DiskBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/HostBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferBatchAsyncContext.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {

constexpr size_t kTransferWorkerCount = 1;
constexpr size_t kTransferQueueSize   = 10000;

ErrorInfo transferStatusToErrorInfo(TransferStatus status) {
    switch (status) {
        case TransferStatus::OK:
            return ErrorInfo::OkStatus();
        case TransferStatus::INVALID_ARGS:
            return ErrorInfo(ErrorCode::INVALID_PARAMS, "invalid block transfer request");
        case TransferStatus::DEVICE_IO_ERROR:
            return ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "device block transfer failed");
        case TransferStatus::DISK_IO_ERROR:
            return ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "disk block transfer failed");
        case TransferStatus::RESOURCE_EXHAUSTED:
            return ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "device-disk staging pool exhausted");
    }
    return ErrorInfo(ErrorCode::UNKNOWN_ERROR, "unknown block transfer status");
}

}  // namespace

PerRankBlockTransferEngine::PerRankBlockTransferEngine(std::vector<GroupSetPtr> group_sets,
                                                       DeviceHostCopyOptions    device_host_options,
                                                       size_t                   device_disk_staging_block_count,
                                                       size_t                   max_descriptors_per_batch):
    group_sets_(std::move(group_sets)),
    device_host_executor_(std::make_unique<DeviceHostTransferExecutor>(std::move(device_host_options))),
    host_disk_executor_(std::make_unique<HostDiskTransferExecutor>()),
    max_descriptors_per_batch_(max_descriptors_per_batch) {
    RTP_LLM_CHECK(max_descriptors_per_batch_ > 0);
    device_to_host_task_pool_ =
        std::make_unique<BlockTreeTaskPool>(kTransferWorkerCount, kTransferQueueSize, "BlockD2HTransfer");
    host_to_device_task_pool_ =
        std::make_unique<BlockTreeTaskPool>(kTransferWorkerCount, kTransferQueueSize, "BlockH2DTransfer");
    host_to_disk_task_pool_ =
        std::make_unique<BlockTreeTaskPool>(kTransferWorkerCount, kTransferQueueSize, "BlockH2DiskTransfer");
    disk_to_host_task_pool_ =
        std::make_unique<BlockTreeTaskPool>(kTransferWorkerCount, kTransferQueueSize, "BlockDisk2HTransfer");
    RTP_LLM_CHECK(device_to_host_task_pool_->start());
    RTP_LLM_CHECK(host_to_device_task_pool_->start());
    RTP_LLM_CHECK(host_to_disk_task_pool_->start());
    RTP_LLM_CHECK(disk_to_host_task_pool_->start());

    const bool any_disk_pool = std::any_of(group_sets_.begin(), group_sets_.end(), [](const GroupSetPtr& group_set) {
        return group_set->diskPool() != nullptr;
    });
    if (any_disk_pool) {
        device_disk_executor_ = std::make_unique<DeviceDiskTransferExecutor>(
            *device_host_executor_, *host_disk_executor_, group_sets_, device_disk_staging_block_count);
    }
}

std::shared_ptr<AsyncContext>
PerRankBlockTransferEngine::submit(const std::vector<TransferDescriptor>& descriptors) {
    if (descriptors.empty()) {
        return std::make_shared<CompletedAsyncContext>(transferStatusToErrorInfo(TransferStatus::INVALID_ARGS));
    }

    const Tier source = descriptors.front().source_tier;
    const Tier target = descriptors.front().target_tier;
    std::vector<const GroupSet*> group_sets;
    std::vector<HostBufferView>  hosts;
    group_sets.reserve(descriptors.size());
    hosts.reserve(descriptors.size());
    for (const auto& descriptor : descriptors) {
        if (descriptor.source_tier != source || descriptor.target_tier != target) {
            return std::make_shared<CompletedAsyncContext>(transferStatusToErrorInfo(TransferStatus::INVALID_ARGS));
        }
        const auto* group_set = group_sets_[descriptor.group_set_id].get();
        group_sets.push_back(group_set);
        if (source == Tier::HOST || target == Tier::HOST) {
            hosts.push_back(resolveHostView(*group_set, descriptor.singleBlockAt(Tier::HOST)));
        }
    }

    if (source == Tier::DISK && target == Tier::DEVICE) {
        return device_disk_executor_->execute(descriptors, group_sets);
    }

    if (source == Tier::DEVICE && target == Tier::DISK) {
        if (descriptors.size() != 1) {
            return std::make_shared<CompletedAsyncContext>(transferStatusToErrorInfo(TransferStatus::INVALID_ARGS));
        }
        return std::make_shared<CompletedAsyncContext>(
            transferStatusToErrorInfo(device_disk_executor_->execute(descriptors.front(), *group_sets.front())));
    }

    BlockTreeTaskPool* task_pool = taskPoolForDirection(source, target);
    if (task_pool == nullptr) {
        return std::make_shared<CompletedAsyncContext>(transferStatusToErrorInfo(TransferStatus::INVALID_ARGS));
    }

    auto context = std::make_shared<TransferBatchAsyncContext>();
    const bool accepted = task_pool->submit([this, descriptors, group_sets, hosts, context] {
        try {
            for (size_t begin = 0; begin < descriptors.size(); begin += max_descriptors_per_batch_) {
                const size_t end = std::min(begin + max_descriptors_per_batch_, descriptors.size());
                const std::vector<HostBufferView> sub_hosts(hosts.begin() + begin, hosts.begin() + end);
                const std::vector<TransferDescriptor> sub_descriptors(descriptors.begin() + begin,
                                                                      descriptors.begin() + end);
                const std::vector<const GroupSet*> sub_group_sets(group_sets.begin() + begin,
                                                                  group_sets.begin() + end);
                const TransferStatus status = execute(sub_hosts, sub_descriptors, sub_group_sets);
                if (status != TransferStatus::OK) {
                    for (size_t index = begin; index < end; ++index) {
                        RTP_LLM_LOG_WARNING("transfer batch item failed, index=%zu %s",
                                            index,
                                            descriptors[index].debugString().c_str());
                    }
                    const auto error = transferStatusToErrorInfo(status);
                    context->complete(ErrorInfo(error.code(),
                                                error.ToString() + ", descriptor_range=[" + std::to_string(begin)
                                                    + "," + std::to_string(end) + ")"));
                    return;
                }
            }
            context->complete(ErrorInfo::OkStatus());
        } catch (const std::exception& error) {
            context->complete(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, error.what()));
        } catch (...) {
            context->complete(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "unknown transfer executor exception"));
        }
    });
    if (!accepted) {
        context->complete(
            ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "RESOURCE_EXHAUSTED: transfer queue is full or stopped"));
    }
    return context;
}

TransferStatus
PerRankBlockTransferEngine::execute(const std::vector<HostBufferView>&       hosts,
                                    const std::vector<TransferDescriptor>& descriptors,
                                    const std::vector<const GroupSet*>&    group_sets) const {
    const Tier source = descriptors.front().source_tier;
    const Tier target = descriptors.front().target_tier;
    if ((source == Tier::DEVICE && target == Tier::HOST) || (source == Tier::HOST && target == Tier::DEVICE)) {
        return device_host_executor_->execute(hosts, descriptors, group_sets);
    }
    if ((source == Tier::HOST && target == Tier::DISK) || (source == Tier::DISK && target == Tier::HOST)) {
        return host_disk_executor_->execute(hosts, descriptors, group_sets);
    }
    return TransferStatus::INVALID_ARGS;
}

BlockTreeTaskPool* PerRankBlockTransferEngine::taskPoolForDirection(Tier source, Tier target) const {
    if (source == Tier::DEVICE && target == Tier::HOST) {
        return device_to_host_task_pool_.get();
    }
    if (source == Tier::HOST && target == Tier::DEVICE) {
        return host_to_device_task_pool_.get();
    }
    if (source == Tier::HOST && target == Tier::DISK) {
        return host_to_disk_task_pool_.get();
    }
    if (source == Tier::DISK && target == Tier::HOST) {
        return disk_to_host_task_pool_.get();
    }
    return nullptr;
}

HostBufferView PerRankBlockTransferEngine::resolveHostView(const GroupSet& group_set, BlockIdxType host_block) {
    const HostBlockBuffer buffer = group_set.hostPool()->blockBuffer(host_block);
    return HostBufferView{buffer.addr, buffer.payload_bytes, buffer.stride_bytes};
}

}  // namespace rtp_llm
