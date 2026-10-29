#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceDiskTransferExecutor.h"

#include <algorithm>
#include <exception>
#include <string>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheMetricsReporter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceHostTransferExecutor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/HostDiskTransferExecutor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferBatchAsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferStageState.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferStageTimer.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {

namespace {

size_t alignedStride(size_t payload_bytes) {
    const size_t alignment = HostStagingBlockPool::kAlignment;
    if (payload_bytes > std::numeric_limits<size_t>::max() - (alignment - 1)) {
        throw std::overflow_error("disk staging stride overflow");
    }
    return ((payload_bytes + alignment - 1) / alignment) * alignment;
}

TransferResult deviceToDiskError(TransferResult result, const char* stage) {
    result.status = ErrorInfo(result.status.code(),
                              std::string("device-to-disk ") + stage + " failed: " + result.status.ToString());
    return result;
}

TransferResult transferError(TransferResult result, size_t begin, size_t end, const char* stage) {
    result.status =
        ErrorInfo(result.status.code(),
                  std::string("disk-to-device transfer failed, stage=") + stage + ", descriptor_range=["
                      + std::to_string(begin) + "," + std::to_string(end) + "): " + result.status.ToString());
    return result;
}

void logBatchFailure(const std::vector<TransferDescriptor>& descriptors, size_t begin, const char* stage) {
    for (size_t index = 0; index < descriptors.size(); ++index) {
        RTP_LLM_LOG_WARNING(
            "disk-to-device %s failed, index=%zu %s", stage, begin + index, descriptors[index].debugString().c_str());
    }
}

}  // namespace

DeviceDiskTransferExecutor::DeviceDiskTransferExecutor(DeviceHostTransferExecutor&     device_host_executor,
                                                       HostDiskTransferExecutor&       host_disk_executor,
                                                       const std::vector<GroupSetPtr>& group_sets,
                                                       size_t                          staging_block_count,
                                                       BlockTreeTaskPool&              transfer_task_pool,
                                                       size_t                          max_descriptors_per_batch,
                                                       std::shared_ptr<BlockTreeCacheMetricsReporter> metrics_reporter,
                                                       BlockTreeTaskPool*                             load_task_pool,
                                                       int64_t                                        world_rank):
    device_host_executor_(device_host_executor),
    host_disk_executor_(host_disk_executor),
    transfer_task_pool_(transfer_task_pool),
    load_task_pool_(load_task_pool == nullptr ? &transfer_task_pool : load_task_pool),
    max_descriptors_per_batch_(max_descriptors_per_batch),
    metrics_reporter_(std::move(metrics_reporter)),
    world_rank_(world_rank) {
    RTP_LLM_CHECK(staging_block_count >= 2 && staging_block_count % 2 == 0);
    RTP_LLM_CHECK(max_descriptors_per_batch_ > 0);

    size_t max_stride  = 0;
    size_t full_stride = 0;
    size_t swa_stride  = 0;
    for (const auto& group_set : group_sets) {
        const size_t stride = alignedStride(group_set->storageBytes());
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

    RTP_LLM_CHECK(max_stride > 0 && staging_block_count <= std::numeric_limits<size_t>::max() / max_stride);
    const size_t lane_bytes = staging_block_count * max_stride / 2;
    full_batch_capacity_    = lane_bytes / full_stride;
    swa_batch_capacity_     = lane_bytes / swa_stride;
    RTP_LLM_CHECK(full_batch_capacity_ > 0 && swa_batch_capacity_ > 0);

    full_staging_pool_ = std::make_unique<HostStagingBlockPool>(full_batch_capacity_, full_stride);
    swa_staging_pool_  = std::make_unique<HostStagingBlockPool>(swa_batch_capacity_, swa_stride);
}

void DeviceDiskTransferExecutor::cancelPendingTransfers() {
    full_staging_pool_->cancelAllBatchWaiters();
    swa_staging_pool_->cancelAllBatchWaiters();
}

std::shared_ptr<TransferAsyncContext>
DeviceDiskTransferExecutor::executeDiskToDevice(TransferTask task, const std::vector<const GroupSet*>& group_sets) {
    auto        context     = std::make_shared<TransferBatchAsyncContext>(task.descriptors().size());
    const auto& descriptors = task.descriptors();
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
    HostStagingBlockPool* pool     = stagingPool(group_type);
    const size_t          capacity = std::min(batchCapacity(group_type), max_descriptors_per_batch_);
    if (pool == nullptr || capacity == 0) {
        context->complete(ErrorInfo(ErrorCode::INVALID_PARAMS, "unsupported disk-to-device cache group type"));
        return context;
    }

    auto stage_state = std::make_shared<TransferStageState>(
        [context](TransferResult result) { context->complete(std::move(result)); }, descriptors.size());
    for (size_t begin = 0; begin < descriptors.size(); begin += capacity) {
        const size_t                    end = std::min(begin + capacity, descriptors.size());
        std::vector<TransferDescriptor> sub_descriptors(descriptors.begin() + begin, descriptors.begin() + end);
        std::vector<const GroupSet*>    sub_group_sets(group_sets.begin() + begin, group_sets.begin() + end);
        auto                            batch_task = task.subtask(std::move(sub_descriptors));
        const auto                      deadline   = batch_task.deadline();
        stage_state->addBatch();
        std::vector<size_t> indices(end - begin);
        std::iota(indices.begin(), indices.end(), begin);
        const auto request_id  = batch_task.requestId().value_or(-1);
        const auto transfer_id = batch_task.traceId();
        auto       batch_timer = TransferStageTimer::create(
            metrics_reporter_, "worker_batch_total", request_id, begin, end, world_rank_, "worker_batch", transfer_id);
        auto staging_timer = TransferStageTimer::create(
            metrics_reporter_, "staging_wait", request_id, begin, end, world_rank_, "worker_batch", transfer_id);
        pool->requestBatch(
            end - begin,
            deadline,
            [this,
             stage_state,
             indices        = std::move(indices),
             batch_task     = std::move(batch_task),
             sub_group_sets = std::move(sub_group_sets),
             begin,
             end,
             request_id,
             transfer_id,
             batch_timer,
             staging_timer,
             deadline](std::optional<HostStagingBlockPool::HostStagingBlockBatch> leases) mutable {
                staging_timer.finish(leases.has_value());
                if (!leases.has_value()) {
                    batch_timer.finish(false);
                    const bool timed_out = batch_task.expired();
                    stage_state->completeBatch(
                        TransferResult(
                            ErrorInfo(timed_out ? ErrorCode::DEADLINE_EXCEEDED : ErrorCode::EXECUTION_EXCEPTION,
                                      std::string(timed_out ?
                                                      "disk-to-device staging admission timed out, descriptor_range=[" :
                                                      "disk-to-device staging admission cancelled, descriptor_range=[")
                                          + std::to_string(begin) + "," + std::to_string(end) + ")"),
                            CacheCopyError::NOT_EXECUTED),
                        indices);
                    return;
                }

                auto batch_leases = std::make_shared<HostStagingBlockPool::HostStagingBlockBatch>(std::move(*leases));
                const int64_t queue_begin = currentTimeUs();
                auto          queue_timer = TransferStageTimer::create(metrics_reporter_,
                                                                       "transfer_queue_wait",
                                                                       request_id,
                                                                       begin,
                                                                       end,
                                                                       world_rank_,
                                                                       "worker_batch",
                                                                       transfer_id);
                auto on_timeout = [this, stage_state, indices, begin, end, queue_begin, queue_timer, batch_timer]() {
                    queue_timer.finish(false);
                    batch_timer.finish(false);
                    if (metrics_reporter_ != nullptr) {
                        metrics_reporter_->reportQueueWaitMetric(
                            false, "transfer", nullptr, Tier::DISK, Tier::DEVICE, currentTimeUs() - queue_begin);
                    }
                    stage_state->completeBatch(
                        TransferResult(ErrorInfo(ErrorCode::DEADLINE_EXCEEDED,
                                                 "disk-to-device expired in TE worker queue, descriptor_range=["
                                                     + std::to_string(begin) + "," + std::to_string(end) + ")"),
                                       CacheCopyError::NOT_EXECUTED),
                        indices);
                };
                const bool accepted = load_task_pool_->submit(
                    BlockTreeTaskClass::LOAD,
                    [this,
                     stage_state,
                     indices,
                     batch_task     = std::move(batch_task),
                     sub_group_sets = std::move(sub_group_sets),
                     batch_leases   = std::move(batch_leases),
                     begin,
                     end,
                     request_id,
                     transfer_id,
                     queue_timer,
                     batch_timer,
                     queue_begin] {
                        queue_timer.finish(true);
                        const auto& sub_descriptors = batch_task.descriptors();
                        if (metrics_reporter_ != nullptr) {
                            metrics_reporter_->reportQueueWaitMetric(
                                false, "transfer", nullptr, Tier::DISK, Tier::DEVICE, currentTimeUs() - queue_begin);
                        }
                        try {
                            std::vector<HostBufferView> hosts;
                            hosts.reserve(batch_leases->size());
                            for (size_t index = 0; index < batch_leases->size(); ++index) {
                                hosts.push_back(
                                    (*batch_leases)[index].blockBuffer(sub_group_sets[index]->payloadBytes()));
                            }

                            TransferStageTimer disk_timer(metrics_reporter_,
                                                          "disk_read",
                                                          request_id,
                                                          begin,
                                                          end,
                                                          world_rank_,
                                                          "worker_batch",
                                                          transfer_id);
                            auto result = host_disk_executor_.executeBatch(hosts, sub_descriptors, sub_group_sets);
                            disk_timer.finish(result.ok());
                            if (!result.ok()) {
                                batch_timer.finish(false);
                                logBatchFailure(sub_descriptors, begin, "disk-to-staging");
                                stage_state->completeBatch(
                                    transferError(std::move(result), begin, end, "disk-to-staging"), indices);
                                return;
                            }
                            TransferStageTimer h2d_timer(metrics_reporter_,
                                                         "h2d",
                                                         request_id,
                                                         begin,
                                                         end,
                                                         world_rank_,
                                                         "worker_batch",
                                                         transfer_id);
                            result = device_host_executor_.executeBatch(hosts, sub_descriptors, sub_group_sets);
                            h2d_timer.finish(result.ok());
                            if (!result.ok()) {
                                batch_timer.finish(false);
                                logBatchFailure(sub_descriptors, begin, "staging-to-device");
                                stage_state->completeBatch(
                                    transferError(std::move(result), begin, end, "staging-to-device"), indices);
                                return;
                            }
                            batch_timer.finish(true);
                            stage_state->completeBatch(std::move(result), indices);
                        } catch (const std::exception& error) {
                            batch_timer.finish(false);
                            stage_state->completeBatch(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, error.what()),
                                                       indices);
                        } catch (...) {
                            batch_timer.finish(false);
                            stage_state->completeBatch(
                                ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "unknown disk-to-device exception"), indices);
                        }
                    },
                    deadline,
                    std::move(on_timeout));
                if (!accepted) {
                    queue_timer.finish(false);
                    batch_timer.finish(false);
                    stage_state->completeBatch(
                        TransferResult(
                            ErrorInfo(ErrorCode::EXECUTION_EXCEPTION,
                                      "RESOURCE_EXHAUSTED: disk-to-device queue is full or stopped, descriptor_range=["
                                          + std::to_string(begin) + "," + std::to_string(end) + ")"),
                            CacheCopyError::NOT_EXECUTED),
                        indices);
                }
            });
    }
    stage_state->finishSubmitting();
    return context;
}

std::shared_ptr<TransferAsyncContext> DeviceDiskTransferExecutor::executeDeviceToDisk(TransferTask    task,
                                                                                      const GroupSet& group_set) {
    auto context = std::make_shared<TransferBatchAsyncContext>(task.descriptors().size());
    if (task.descriptors().size() != 1) {
        context->complete(ErrorInfo(ErrorCode::INVALID_PARAMS, "invalid device-to-disk batch"));
        return context;
    }
    HostStagingBlockPool* pool = stagingPool(group_set.groupType());
    if (pool == nullptr) {
        context->complete(ErrorInfo(ErrorCode::INVALID_PARAMS, "unsupported device-to-disk cache group type"));
        return context;
    }
    const GroupSet* group_set_ptr = &group_set;
    const auto      deadline      = task.deadline();
    const auto      identity      = CacheStageContext::local(Tier::DEVICE, Tier::DISK);
    const auto      request_id    = task.requestId().value_or(-1);
    const auto      transfer_id   = task.traceId();
    auto            total         = TransferStageTimer::create(
        metrics_reporter_, "worker_batch_total", request_id, 0, 1, world_rank_, "worker_batch", transfer_id, identity);
    auto staging = TransferStageTimer::create(
        metrics_reporter_, "staging_wait", request_id, 0, 1, world_rank_, "worker_batch", transfer_id, identity);
    const auto complete = [context, total](TransferResult result) {
        total.finish(result.ok());
        context->complete(std::move(result));
    };
    pool->requestBatch(
        1,
        deadline,
        [this, complete, task = std::move(task), group_set_ptr, deadline, identity, request_id, transfer_id, staging](
            std::optional<HostStagingBlockPool::HostStagingBlockBatch> leases) {
            staging.finish(leases.has_value());
            if (!leases.has_value()) {
                const bool timed_out = task.expired();
                complete(
                    TransferResult(ErrorInfo(timed_out ? ErrorCode::DEADLINE_EXCEEDED : ErrorCode::EXECUTION_EXCEPTION,
                                             timed_out ? "device-to-disk staging admission timed out" :
                                                         "device-to-disk staging admission cancelled"),
                                   CacheCopyError::NOT_EXECUTED));
                return;
            }

            auto batch_leases = std::make_shared<HostStagingBlockPool::HostStagingBlockBatch>(std::move(*leases));
            const int64_t queue_begin = currentTimeUs();
            auto          queue       = TransferStageTimer::create(metrics_reporter_,
                                                                   "transfer_queue_wait",
                                                                   request_id,
                                                                   0,
                                                                   1,
                                                                   world_rank_,
                                                                   "worker_batch",
                                                                   transfer_id,
                                                                   identity);
            auto          on_timeout  = [this, complete, queue_begin, queue]() {
                queue.finish(false);
                if (metrics_reporter_ != nullptr) {
                    metrics_reporter_->reportQueueWaitMetric(
                        false, "transfer", nullptr, Tier::DEVICE, Tier::DISK, currentTimeUs() - queue_begin);
                }
                complete(
                    TransferResult(ErrorInfo(ErrorCode::DEADLINE_EXCEEDED, "device-to-disk expired in TE worker queue"),
                                   CacheCopyError::NOT_EXECUTED));
            };
            const bool accepted = transfer_task_pool_.submit(
                BlockTreeTaskClass::BACKGROUND,
                [this,
                 complete,
                 task = std::move(task),
                 group_set_ptr,
                 batch_leases,
                 queue_begin,
                 identity,
                 request_id,
                 transfer_id,
                 queue] {
                    queue.finish(true);
                    if (metrics_reporter_ != nullptr) {
                        metrics_reporter_->reportQueueWaitMetric(
                            false, "transfer", nullptr, Tier::DEVICE, Tier::DISK, currentTimeUs() - queue_begin);
                    }
                    try {
                        const auto&                       descriptor = task.descriptors().front();
                        const std::vector<HostBufferView> hosts{
                            batch_leases->front().blockBuffer(group_set_ptr->payloadBytes())};
                        const std::vector<TransferDescriptor> descriptors{descriptor};
                        const std::vector<const GroupSet*>    group_sets{group_set_ptr};
                        TransferStageTimer                    copy(metrics_reporter_,
                                                                   "d2h",
                                                                   request_id,
                                                                   0,
                                                                   1,
                                                                   world_rank_,
                                                                   "worker_batch",
                                                                   transfer_id,
                                                                   identity);
                        auto result = device_host_executor_.executeBatch(hosts, descriptors, group_sets);
                        copy.finish(result.ok());
                        if (!result.ok()) {
                            complete(deviceToDiskError(std::move(result), "device-to-staging"));
                            return;
                        }
                        TransferStageTimer disk(metrics_reporter_,
                                                "disk_write",
                                                request_id,
                                                0,
                                                1,
                                                world_rank_,
                                                "worker_batch",
                                                transfer_id,
                                                identity);
                        result = host_disk_executor_.executeBatch(hosts, descriptors, group_sets);
                        disk.finish(result.ok());
                        if (!result.ok()) {
                            complete(deviceToDiskError(std::move(result), "staging-to-disk"));
                            return;
                        }
                        complete(std::move(result));
                    } catch (const std::exception& error) {
                        complete(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, error.what()));
                    } catch (...) {
                        complete(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "unknown device-to-disk exception"));
                    }
                },
                deadline,
                std::move(on_timeout));
            if (!accepted) {
                queue.finish(false);
                complete(TransferResult(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION,
                                                  "RESOURCE_EXHAUSTED: device-to-disk queue is full or stopped"),
                                        CacheCopyError::NOT_EXECUTED));
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
