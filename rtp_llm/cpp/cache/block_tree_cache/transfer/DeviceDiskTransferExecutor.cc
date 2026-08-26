#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceDiskTransferExecutor.h"

#include <algorithm>
#include <exception>
#include <string>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceHostTransferExecutor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/HostDiskTransferExecutor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferBatchAsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferStageState.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {

size_t alignedStride(size_t payload_bytes) {
    const size_t alignment = HostStagingBlockPool::kAlignment;
    return ((payload_bytes + alignment - 1) / alignment) * alignment;
}

ErrorInfo deviceToDiskError(TransferStatus status, const char* stage) {
    const ErrorCode code = status == TransferStatus::INVALID_ARGS ? ErrorCode::INVALID_PARAMS :
                                                                        ErrorCode::EXECUTION_EXCEPTION;
    return ErrorInfo(code, std::string("device-to-disk ") + stage + " failed");
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
                                                       size_t                          staging_block_count,
                                                       BlockTreeTaskPool&              transfer_task_pool):
    device_host_executor_(device_host_executor),
    host_disk_executor_(host_disk_executor),
    transfer_task_pool_(transfer_task_pool) {
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
}

DeviceDiskTransferExecutor::~DeviceDiskTransferExecutor() {
    cancelPendingTransfers();
}

void DeviceDiskTransferExecutor::cancelPendingTransfers() {
    full_staging_pool_->cancelAllBatchWaiters();
    swa_staging_pool_->cancelAllBatchWaiters();
}

std::shared_ptr<AsyncContext>
DeviceDiskTransferExecutor::execute(const std::vector<TransferDescriptor>& descriptors,
                                    const std::vector<const GroupSet*>&    group_sets) {
    auto context = std::make_shared<TransferBatchAsyncContext>();
    if (descriptors.empty() || descriptors.size() != group_sets.size()) {
        context->complete(ErrorInfo(ErrorCode::INVALID_PARAMS, "invalid disk-to-device batch"));
        return context;
    }
    const CacheGroupType group_type = group_sets.front()->groupType();
    if (std::any_of(group_sets.begin(), group_sets.end(), [group_type](const GroupSet* group_set) {
            return group_set == nullptr || group_set->groupType() != group_type;
        })) {
        context->complete(ErrorInfo(ErrorCode::INVALID_PARAMS, "mixed disk-to-device cache group types"));
        return context;
    }
    HostStagingBlockPool* pool      = stagingPool(group_type);
    const size_t capacity           = batchCapacity(group_type);
    if (pool == nullptr || capacity == 0) {
        context->complete(ErrorInfo(ErrorCode::INVALID_PARAMS, "unsupported disk-to-device cache group type"));
        return context;
    }

    auto stage_state = std::make_shared<TransferStageState>(
        [context](ErrorInfo error) { context->complete(std::move(error)); });
    for (size_t begin = 0; begin < descriptors.size(); begin += capacity) {
        const size_t end = std::min(begin + capacity, descriptors.size());
        std::vector<TransferDescriptor> sub_descriptors(descriptors.begin() + begin, descriptors.begin() + end);
        std::vector<const GroupSet*> sub_group_sets(group_sets.begin() + begin, group_sets.begin() + end);
        stage_state->addBatch();
        pool->requestBatch(end - begin,
                           [this,
                            stage_state,
                            sub_descriptors = std::move(sub_descriptors),
                            sub_group_sets = std::move(sub_group_sets),
                            begin,
                            end](std::optional<HostStagingBlockPool::HostStagingBlockBatch> leases) mutable {
            if (!leases.has_value()) {
                stage_state->completeBatch(ErrorInfo(
                    ErrorCode::EXECUTION_EXCEPTION,
                    "disk-to-device staging admission cancelled, descriptor_range=[" + std::to_string(begin) + ","
                        + std::to_string(end) + ")"));
                return;
            }

            auto batch_leases = std::make_shared<HostStagingBlockPool::HostStagingBlockBatch>(std::move(*leases));
            const bool accepted = transfer_task_pool_.submit(
                [this,
                 stage_state,
                 sub_descriptors = std::move(sub_descriptors),
                 sub_group_sets = std::move(sub_group_sets),
                 batch_leases = std::move(batch_leases),
                 begin,
                 end] {
                try {
                    std::vector<HostBufferView> hosts;
                    hosts.reserve(batch_leases->size());
                    for (size_t index = 0; index < batch_leases->size(); ++index) {
                        hosts.push_back((*batch_leases)[index].blockBuffer(sub_group_sets[index]->payloadBytes()));
                    }

                    TransferStatus status = host_disk_executor_.execute(hosts, sub_descriptors, sub_group_sets);
                    if (status != TransferStatus::OK) {
                        logBatchFailure(sub_descriptors, begin, "disk-to-staging");
                        stage_state->completeBatch(transferError(status, begin, end));
                        return;
                    }
                    status = device_host_executor_.execute(hosts, sub_descriptors, sub_group_sets);
                    if (status != TransferStatus::OK) {
                        logBatchFailure(sub_descriptors, begin, "staging-to-device");
                        stage_state->completeBatch(transferError(status, begin, end));
                        return;
                    }
                    stage_state->completeBatch(ErrorInfo::OkStatus());
                } catch (const std::exception& error) {
                    stage_state->completeBatch(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, error.what()));
                } catch (...) {
                    stage_state->completeBatch(
                        ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "unknown disk-to-device exception"));
                }
            });
            if (!accepted) {
                stage_state->completeBatch(ErrorInfo(
                    ErrorCode::EXECUTION_EXCEPTION,
                    "RESOURCE_EXHAUSTED: disk-to-device queue is full or stopped, descriptor_range=["
                        + std::to_string(begin) + "," + std::to_string(end) + ")"));
            }
        });
    }
    stage_state->finishSubmitting();
    return context;
}

std::shared_ptr<AsyncContext>
DeviceDiskTransferExecutor::executeDeviceToDisk(const TransferDescriptor& descriptor, const GroupSet& group_set) {
    auto context = std::make_shared<TransferBatchAsyncContext>();
    HostStagingBlockPool* pool = stagingPool(group_set.groupType());
    if (pool == nullptr) {
        context->complete(ErrorInfo(ErrorCode::INVALID_PARAMS, "unsupported device-to-disk cache group type"));
        return context;
    }
    const GroupSet* group_set_ptr = &group_set;
    pool->requestBatch(1, [this, context, descriptor, group_set_ptr](
                              std::optional<HostStagingBlockPool::HostStagingBlockBatch> leases) {
        if (!leases.has_value()) {
            context->complete(
                ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "device-to-disk staging admission cancelled"));
            return;
        }

        auto batch_leases = std::make_shared<HostStagingBlockPool::HostStagingBlockBatch>(std::move(*leases));
        const bool accepted = transfer_task_pool_.submit([this, context, descriptor, group_set_ptr, batch_leases] {
            try {
                const std::vector<HostBufferView> hosts{
                    batch_leases->front().blockBuffer(group_set_ptr->payloadBytes())};
                const std::vector<TransferDescriptor> descriptors{descriptor};
                const std::vector<const GroupSet*> group_sets{group_set_ptr};
                TransferStatus status = device_host_executor_.execute(hosts, descriptors, group_sets);
                if (status != TransferStatus::OK) {
                    context->complete(deviceToDiskError(status, "device-to-staging"));
                    return;
                }
                status = host_disk_executor_.execute(hosts, descriptors, group_sets);
                if (status != TransferStatus::OK) {
                    context->complete(deviceToDiskError(status, "staging-to-disk"));
                    return;
                }
                context->complete(ErrorInfo::OkStatus());
            } catch (const std::exception& error) {
                context->complete(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, error.what()));
            } catch (...) {
                context->complete(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "unknown device-to-disk exception"));
            }
        });
        if (!accepted) {
            context->complete(ErrorInfo(
                ErrorCode::EXECUTION_EXCEPTION, "RESOURCE_EXHAUSTED: device-to-disk queue is full or stopped"));
        }
    });
    return context;
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
