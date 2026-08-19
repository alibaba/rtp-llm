#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceDiskTransferExecutor.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <string>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceHostTransferExecutor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/HostDiskTransferExecutor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferBatchAsyncContext.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {

constexpr std::chrono::milliseconds kStagingAcquireTimeout{1000};
constexpr size_t                    kLaneWorkerCount = 1;
constexpr size_t                    kLaneQueueSize   = 10000;

size_t alignedStride(size_t payload_bytes) {
    const size_t alignment = HostStagingBlockPool::kAlignment;
    return ((payload_bytes + alignment - 1) / alignment) * alignment;
}

ErrorInfo transferError(TransferStatus status, size_t begin, size_t end) {
    const ErrorCode code = status == TransferStatus::RESOURCE_EXHAUSTED ? ErrorCode::DEADLINE_EXCEEDED :
                                                                          ErrorCode::EXECUTION_EXCEPTION;
    const char* message = status == TransferStatus::RESOURCE_EXHAUSTED ? "staging slice wait timed out" :
                                                                        "disk-to-device transfer failed";
    return ErrorInfo(code,
                     std::string(message) + ", descriptor_range=[" + std::to_string(begin) + ","
                         + std::to_string(end) + ")");
}

void logBatchFailure(const std::vector<TransferDescriptor>& descriptors, size_t begin, const char* stage) {
    for (size_t index = 0; index < descriptors.size(); ++index) {
        RTP_LLM_LOG_WARNING("disk-to-device %s failed, index=%zu %s",
                            stage,
                            begin + index,
                            descriptors[index].debugString().c_str());
    }
}

}  // namespace

DeviceDiskTransferExecutor::DeviceDiskTransferExecutor(DeviceHostTransferExecutor&     device_host_executor,
                                                       HostDiskTransferExecutor&       host_disk_executor,
                                                       const std::vector<GroupSetPtr>& group_sets,
                                                       size_t                          staging_block_count):
    device_host_executor_(device_host_executor), host_disk_executor_(host_disk_executor) {
    RTP_LLM_CHECK(staging_block_count >= 2 && staging_block_count % 2 == 0);

    size_t max_stride  = 0;
    size_t full_stride = 0;
    size_t swa_stride  = 0;
    for (const auto& group_set : group_sets) {
        const size_t stride = alignedStride(group_set->payloadBytes());
        max_stride          = std::max(max_stride, stride);
        if (group_set->groupType() == CacheGroupType::FULL) {
            full_stride = std::max(full_stride, stride);
        } else if (group_set->groupType() == CacheGroupType::SWA) {
            swa_stride = std::max(swa_stride, stride);
        }
    }
    if (full_stride == 0) {
        full_stride = max_stride;
    }
    if (swa_stride == 0) {
        swa_stride = max_stride;
    }

    const size_t lane_bytes = staging_block_count * max_stride / 2;
    full_batch_capacity_    = lane_bytes / full_stride;
    swa_batch_capacity_     = lane_bytes / swa_stride;
    RTP_LLM_CHECK(full_batch_capacity_ > 0 && swa_batch_capacity_ > 0);

    full_staging_pool_ = std::make_unique<HostStagingBlockPool>(full_batch_capacity_, full_stride);
    swa_staging_pool_  = std::make_unique<HostStagingBlockPool>(swa_batch_capacity_, swa_stride);
    full_task_pool_ = std::make_unique<BlockTreeTaskPool>(kLaneWorkerCount, kLaneQueueSize, "BlockDisk2DeviceFull");
    swa_task_pool_  = std::make_unique<BlockTreeTaskPool>(kLaneWorkerCount, kLaneQueueSize, "BlockDisk2DeviceSwa");
    RTP_LLM_CHECK(full_task_pool_->start());
    RTP_LLM_CHECK(swa_task_pool_->start());
}

DeviceDiskTransferExecutor::~DeviceDiskTransferExecutor() {
    full_task_pool_->shutdown();
    swa_task_pool_->shutdown();
}

std::shared_ptr<AsyncContext>
DeviceDiskTransferExecutor::execute(const std::vector<TransferDescriptor>& descriptors,
                                    const std::vector<const GroupSet*>&    group_sets) {
    auto context = std::make_shared<TransferBatchAsyncContext>();
    const CacheGroupType group_type = group_sets.front()->groupType();
    HostStagingBlockPool* pool      = stagingPool(group_type);
    BlockTreeTaskPool* task_pool    = taskPool(group_type);
    const size_t capacity           = batchCapacity(group_type);
    if (pool == nullptr || task_pool == nullptr || capacity == 0) {
        context->complete(ErrorInfo(ErrorCode::INVALID_PARAMS, "unsupported disk-to-device cache group type"));
        return context;
    }

    const bool accepted = task_pool->submit([this, descriptors, group_sets, context, pool, capacity] {
        try {
            for (size_t begin = 0; begin < descriptors.size(); begin += capacity) {
                const size_t end = std::min(begin + capacity, descriptors.size());
                std::vector<HostStagingBlockPool::HostStagingBlockLease> leases;
                std::vector<HostBufferView>                              hosts;
                leases.reserve(end - begin);
                hosts.reserve(end - begin);
                for (size_t index = begin; index < end; ++index) {
                    auto lease = pool->mallocWithBackoff(kStagingAcquireTimeout);
                    if (!lease.has_value()) {
                        context->complete(transferError(TransferStatus::RESOURCE_EXHAUSTED, begin, end));
                        return;
                    }
                    hosts.push_back(lease->blockBuffer(group_sets[index]->payloadBytes()));
                    leases.push_back(std::move(*lease));
                }

                const std::vector<TransferDescriptor> sub_descriptors(descriptors.begin() + begin,
                                                                      descriptors.begin() + end);
                const std::vector<const GroupSet*> sub_group_sets(group_sets.begin() + begin,
                                                                  group_sets.begin() + end);
                TransferStatus status = host_disk_executor_.execute(hosts, sub_descriptors, sub_group_sets);
                if (status != TransferStatus::OK) {
                    logBatchFailure(sub_descriptors, begin, "disk-to-staging");
                    context->complete(transferError(status, begin, end));
                    return;
                }
                status = device_host_executor_.execute(hosts, sub_descriptors, sub_group_sets);
                if (status != TransferStatus::OK) {
                    logBatchFailure(sub_descriptors, begin, "staging-to-device");
                    context->complete(transferError(status, begin, end));
                    return;
                }
            }
            context->complete(ErrorInfo::OkStatus());
        } catch (const std::exception& error) {
            context->complete(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, error.what()));
        } catch (...) {
            context->complete(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "unknown disk-to-device exception"));
        }
    });
    if (!accepted) {
        context->complete(
            ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "RESOURCE_EXHAUSTED: disk-to-device queue is full or stopped"));
    }
    return context;
}

TransferStatus DeviceDiskTransferExecutor::execute(const TransferDescriptor& descriptor,
                                                   const GroupSet&           group_set) {
    HostStagingBlockPool* pool = stagingPool(group_set.groupType());
    if (pool == nullptr) {
        return TransferStatus::INVALID_ARGS;
    }
    auto lease = pool->mallocWithBackoff(kStagingAcquireTimeout);
    if (!lease.has_value()) {
        return TransferStatus::RESOURCE_EXHAUSTED;
    }

    const std::vector<HostBufferView> hosts{lease->blockBuffer(group_set.payloadBytes())};
    const std::vector<TransferDescriptor> descriptors{descriptor};
    const std::vector<const GroupSet*> group_sets{&group_set};
    const TransferStatus stage_one = device_host_executor_.execute(hosts, descriptors, group_sets);
    if (stage_one != TransferStatus::OK) {
        return stage_one;
    }
    return host_disk_executor_.execute(hosts, descriptors, group_sets);
}

HostStagingBlockPool* DeviceDiskTransferExecutor::stagingPool(CacheGroupType group_type) const {
    if (group_type == CacheGroupType::FULL) {
        return full_staging_pool_.get();
    }
    if (group_type == CacheGroupType::SWA) {
        return swa_staging_pool_.get();
    }
    return nullptr;
}

BlockTreeTaskPool* DeviceDiskTransferExecutor::taskPool(CacheGroupType group_type) const {
    if (group_type == CacheGroupType::FULL) {
        return full_task_pool_.get();
    }
    if (group_type == CacheGroupType::SWA) {
        return swa_task_pool_.get();
    }
    return nullptr;
}

size_t DeviceDiskTransferExecutor::batchCapacity(CacheGroupType group_type) const {
    if (group_type == CacheGroupType::FULL) {
        return full_batch_capacity_;
    }
    if (group_type == CacheGroupType::SWA) {
        return swa_batch_capacity_;
    }
    return 0;
}

}  // namespace rtp_llm
