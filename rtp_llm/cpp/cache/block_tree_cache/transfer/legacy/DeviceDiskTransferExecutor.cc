#include "rtp_llm/cpp/cache/block_tree_cache/transfer/legacy/DeviceDiskTransferExecutor.h"

#include <algorithm>
#include <exception>
#include <string>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheMetricsReporter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/legacy/DeviceHostTransferExecutor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/legacy/HostDiskTransferExecutor.h"
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

TransferResult
transferError(TransferResult result, size_t begin, size_t end, const char* transfer_name, const char* stage) {
    result.status =
        ErrorInfo(result.status.code(),
                  std::string(transfer_name) + " transfer failed, stage=" + stage + ", descriptor_range=["
                      + std::to_string(begin) + "," + std::to_string(end) + "): " + result.status.ToString());
    return result;
}

void logBatchFailure(const std::vector<TransferDescriptor>& descriptors,
                     size_t                                 begin,
                     const char*                            transfer_name,
                     const char*                            stage) {
    for (size_t index = 0; index < descriptors.size(); ++index) {
        RTP_LLM_LOG_WARNING("%s %s failed, index=%zu %s",
                            transfer_name,
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
                                                       BlockTreeTaskPool&              transfer_task_pool,
                                                       size_t                          max_descriptors_per_batch,
                                                       std::shared_ptr<BlockTreeCacheMetricsReporter> metrics_reporter,
                                                       BlockTreeTaskPool*                             load_task_pool,
                                                       int64_t                                        world_rank):
    device_host_executor_(device_host_executor),
    host_disk_executor_(host_disk_executor),
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
DeviceDiskTransferExecutor::execute(TransferTask task, const std::vector<const GroupSet*>& group_sets) {
    auto        context     = std::make_shared<TransferBatchAsyncContext>(task.descriptors().size());
    const auto&           descriptors   = task.descriptors();
    const Tier source = descriptors.front().source_tier;
    const Tier            target        = descriptors.front().target_tier;
    const char*          transfer_name = "disk-to-device";
    const auto           identity      = CacheStageContext::local(source, target);
    const CacheGroupType  group_type    = group_sets.front()->groupType();
    HostStagingBlockPool* pool     = stagingPool(group_type);
    const size_t          capacity = std::min(batchCapacity(group_type), max_descriptors_per_batch_);
    if (pool == nullptr || capacity == 0) {
        context->complete(ErrorInfo(ErrorCode::INVALID_PARAMS, "unsupported device-disk cache group type"));
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
        const auto request_id    = batch_task.requestId().value_or(-1);
        const auto transfer_id = batch_task.traceId();
        auto       batch_timer   = TransferStageTimer::create(metrics_reporter_,
                                                      "worker_batch_total",
                                                      request_id,
                                                      begin,
                                                      end,
                                                      world_rank_,
                                                      "worker_batch",
                                                      transfer_id,
                                                      identity);
        auto       staging_timer = TransferStageTimer::create(metrics_reporter_,
                                                        "staging_wait",
                                                        request_id,
                                                        begin,
                                                        end,
                                                        world_rank_,
                                                        "worker_batch",
                                                        transfer_id,
                                                        identity,
                                                        AsyncProfileKind::Wait);
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
             deadline,
             transfer_name,
             source,
             target,
             identity](std::optional<HostStagingBlockPool::HostStagingBlockBatch> leases) mutable {
                staging_timer.finish(leases.has_value());
                if (!leases.has_value()) {
                    batch_timer.finish(false);
                    const bool timed_out = batch_task.expired();
                    stage_state->completeBatch(
                        TransferResult(
                            ErrorInfo(timed_out ? ErrorCode::DEADLINE_EXCEEDED : ErrorCode::EXECUTION_EXCEPTION,
                                      std::string(transfer_name)
                                          + (timed_out ? " staging admission timed out, descriptor_range=[" :
                                                         " staging admission cancelled, descriptor_range=[")
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
                                                              transfer_id,
                                                              identity,
                                                              AsyncProfileKind::Wait);
                auto          on_timeout  = [this,
                                   stage_state,
                                   indices,
                                   begin,
                                   end,
                                   queue_begin,
                                   queue_timer,
                                   batch_timer,
                                   transfer_name,
                                   source,
                                   target]() {
                    queue_timer.finish(false);
                    batch_timer.finish(false);
                    if (metrics_reporter_ != nullptr) {
                        metrics_reporter_->reportQueueWaitMetric(
                            false, "transfer", nullptr, source, target, currentTimeUs() - queue_begin);
                    }
                    stage_state->completeBatch(
                        TransferResult(ErrorInfo(ErrorCode::DEADLINE_EXCEEDED,
                                                 std::string(transfer_name)
                                                     + " expired in TE worker queue, descriptor_range=["
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
                     queue_begin,
                     transfer_name,
                     source,
                     target,
                     identity] {
                        queue_timer.finish(true);
                        const auto& sub_descriptors = batch_task.descriptors();
                        if (metrics_reporter_ != nullptr) {
                            metrics_reporter_->reportQueueWaitMetric(
                                false, "transfer", nullptr, source, target, currentTimeUs() - queue_begin);
                        }
                        try {
                            std::vector<HostBufferView> hosts;
                            hosts.reserve(batch_leases->size());
                            for (size_t index = 0; index < batch_leases->size(); ++index) {
                                hosts.push_back(
                                    (*batch_leases)[index].blockBuffer(sub_group_sets[index]->payloadBytes()));
                            }

                            TransferResult result;
                            for (const bool disk_stage : {true, false}) {
                                const char*        stage = disk_stage ? "disk_read" : "h2d";
                                TransferStageTimer timer(metrics_reporter_,
                                                         stage,
                                                         request_id,
                                                         begin,
                                                         end,
                                                         world_rank_,
                                                         "worker_batch",
                                                         transfer_id,
                                                         identity);
                                result = disk_stage ?
                                             host_disk_executor_.executeBatch(hosts, sub_descriptors, sub_group_sets) :
                                             device_host_executor_.executeBatch(hosts, sub_descriptors, sub_group_sets);
                                timer.finish(result.ok());
                                if (!result.ok()) {
                                    batch_timer.finish(false);
                                    const char* failed_stage = disk_stage ? "disk-to-staging" : "staging-to-device";
                                    logBatchFailure(sub_descriptors, begin, transfer_name, failed_stage);
                                    stage_state->completeBatch(
                                        transferError(std::move(result), begin, end, transfer_name, failed_stage),
                                        indices);
                                    return;
                                }
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
                                ErrorInfo(ErrorCode::EXECUTION_EXCEPTION,
                                          std::string("unknown ") + transfer_name + " exception"),
                                indices);
                        }
                    },
                    deadline,
                    std::move(on_timeout));
                if (!accepted) {
                    queue_timer.finish(false);
                    batch_timer.finish(false);
                    stage_state->completeBatch(
                        TransferResult(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION,
                                                 std::string("RESOURCE_EXHAUSTED: ") + transfer_name
                                                     + " queue is full or stopped, descriptor_range=["
                                                     + std::to_string(begin) + "," + std::to_string(end) + ")"),
                                       CacheCopyError::NOT_EXECUTED),
                        indices);
                }
            });
    }
    stage_state->finishSubmitting();
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
