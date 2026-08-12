#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceDiskTransferExecutor.h"

#include <algorithm>
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
constexpr size_t                    kPipelineWorkerCount = 1;
constexpr size_t                    kPipelineQueueSize   = 1000;

ErrorInfo transferError(TransferStatus status, size_t begin, size_t end) {
    const ErrorCode code = status == TransferStatus::RESOURCE_EXHAUSTED ? ErrorCode::DEADLINE_EXCEEDED :
                                                                          ErrorCode::EXECUTION_EXCEPTION;
    const char* message = status == TransferStatus::RESOURCE_EXHAUSTED ? "staging slice wait timed out" :
                                                                        "disk-to-device transfer failed";
    return ErrorInfo(code,
                     std::string(message) + ", descriptor_range=[" + std::to_string(begin) + ","
                         + std::to_string(end) + ")");
}

void logSliceFailure(const std::vector<TransferDescriptor>& descriptors,
                     size_t                                 begin,
                     const char*                            stage) {
    for (size_t index = 0; index < descriptors.size(); ++index) {
        RTP_LLM_LOG_WARNING("disk-to-device %s failed, index=%zu %s",
                            stage,
                            begin + index,
                            descriptors[index].debugString().c_str());
    }
}

}  // namespace

struct DeviceDiskTransferExecutor::BatchState {
    BatchState(std::vector<TransferDescriptor> descriptors,
               std::vector<const GroupSet*>    group_sets,
               std::shared_ptr<TransferBatchAsyncContext> context):
        descriptors(std::move(descriptors)), group_sets(std::move(group_sets)), context(std::move(context)) {}

    void fail(ErrorInfo error) {
        std::lock_guard<std::mutex> lock(mutex);
        if (result.ok()) {
            result = std::move(error);
        }
    }

    void sliceSubmitted() {
        std::lock_guard<std::mutex> lock(mutex);
        ++pending_slices;
    }

    void sliceFinished() {
        finish(true);
    }

    void admissionFinished() {
        finish(false);
    }

    void finish(bool slice_finished) {
        ErrorInfo completion;
        bool      complete = false;
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (slice_finished) {
                --pending_slices;
            } else {
                admission_done = true;
            }
            complete   = admission_done && pending_slices == 0;
            completion = result;
        }
        if (complete) {
            context->complete(completion);
        }
    }

    std::vector<TransferDescriptor>            descriptors;
    std::vector<const GroupSet*>               group_sets;
    std::shared_ptr<TransferBatchAsyncContext> context;
    size_t                                     next_descriptor{0};
    std::mutex                                 mutex;
    ErrorInfo                                  result{ErrorInfo::OkStatus()};
    size_t                                     pending_slices{0};
    bool                                       admission_done{false};
};

struct DeviceDiskTransferExecutor::Slice {
    std::shared_ptr<BatchState>                                  batch;
    size_t                                                       begin{0};
    size_t                                                       end{0};
    std::vector<HostStagingBlockPool::HostStagingBlockLease>     leases;
    std::vector<HostBufferView>                                  hosts;
    std::vector<TransferDescriptor>                              descriptors;
    std::vector<const GroupSet*>                                 group_sets;
};

struct DeviceDiskTransferExecutor::PoolWork {
    size_t                              pool_index{0};
    std::vector<std::shared_ptr<Slice>> slices;
};

DeviceDiskTransferExecutor::DeviceDiskTransferExecutor(DeviceHostTransferExecutor&     device_host_executor,
                                                       HostDiskTransferExecutor&       host_disk_executor,
                                                       const std::vector<GroupSetPtr>& group_sets,
                                                       size_t                          staging_block_count):
    device_host_executor_(device_host_executor), host_disk_executor_(host_disk_executor) {
    RTP_LLM_CHECK(staging_block_count >= 2 && staging_block_count % 2 == 0);
    size_t max_payload_bytes = 0;
    for (const auto& group_set : group_sets) {
        max_payload_bytes = std::max(max_payload_bytes, group_set->payloadBytes());
    }
    const size_t alignment   = HostStagingBlockPool::kAlignment;
    const size_t stride_bytes = ((max_payload_bytes + alignment - 1) / alignment) * alignment;
    staging_pool_capacity_    = staging_block_count / 2;
    for (auto& pool : staging_pools_) {
        pool = std::make_unique<HostStagingBlockPool>(staging_pool_capacity_, stride_bytes);
    }
    pool_states_.fill(PoolState::FREE);
    disk_to_staging_task_pool_ =
        std::make_unique<BlockTreeTaskPool>(kPipelineWorkerCount, kPipelineQueueSize, "BlockDisk2Staging");
    staging_to_device_task_pool_ =
        std::make_unique<BlockTreeTaskPool>(kPipelineWorkerCount, kPipelineQueueSize, "BlockStaging2Device");
    RTP_LLM_CHECK(disk_to_staging_task_pool_->start());
    RTP_LLM_CHECK(staging_to_device_task_pool_->start());
}

DeviceDiskTransferExecutor::~DeviceDiskTransferExecutor() {
    disk_to_staging_task_pool_->shutdown();
    staging_to_device_task_pool_->shutdown();
}

std::shared_ptr<AsyncContext>
DeviceDiskTransferExecutor::execute(const std::vector<TransferDescriptor>& descriptors,
                                    const std::vector<const GroupSet*>&    group_sets,
                                    std::shared_ptr<void>                  completion_guard) {
    auto context = std::make_shared<TransferBatchAsyncContext>(std::move(completion_guard));
    auto batch   = std::make_shared<BatchState>(descriptors, group_sets, context);
    bool schedule = false;
    {
        std::lock_guard<std::mutex> lock(pending_mutex_);
        if (pending_batches_.size() >= kPipelineQueueSize) {
            context->complete(
                ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "RESOURCE_EXHAUSTED: disk-to-device queue is full"));
            return context;
        }
        pending_batches_.push_back(batch);
        if (!stage_one_scheduled_) {
            stage_one_scheduled_ = true;
            schedule             = true;
        }
    }
    if (schedule && !disk_to_staging_task_pool_->submit([this] { drainStageOne(); })) {
        std::deque<std::shared_ptr<BatchState>> rejected;
        {
            std::lock_guard<std::mutex> lock(pending_mutex_);
            stage_one_scheduled_ = false;
            rejected.swap(pending_batches_);
        }
        for (const auto& rejected_batch : rejected) {
            rejected_batch->fail(ErrorInfo(
                ErrorCode::EXECUTION_EXCEPTION, "RESOURCE_EXHAUSTED: disk-to-staging queue is full or stopped"));
            rejected_batch->admissionFinished();
        }
    }
    return context;
}

TransferStatus DeviceDiskTransferExecutor::execute(const TransferDescriptor& descriptor,
                                                   const GroupSet&           group_set) {
    const auto pool_index = acquirePool(kStagingAcquireTimeout);
    if (!pool_index.has_value()) {
        return TransferStatus::RESOURCE_EXHAUSTED;
    }
    auto pool_guard = std::shared_ptr<void>(nullptr, [this, pool_index](void*) { releasePool(*pool_index); });
    auto lease      = staging_pools_[*pool_index]->malloc();
    RTP_LLM_CHECK(lease.has_value());

    const HostBufferView                 host = lease->blockBuffer(group_set.payloadBytes());
    const std::vector<HostBufferView>    hosts{host};
    const std::vector<TransferDescriptor> descriptors{descriptor};
    const std::vector<const GroupSet*>    group_sets{&group_set};
    const TransferStatus stage_one = device_host_executor_.execute(hosts, descriptors, group_sets);
    if (stage_one != TransferStatus::OK) {
        return stage_one;
    }
    return host_disk_executor_.execute(hosts, descriptors, group_sets);
}

void DeviceDiskTransferExecutor::drainStageOne() {
    while (true) {
        {
            std::lock_guard<std::mutex> lock(pending_mutex_);
            if (pending_batches_.empty()) {
                stage_one_scheduled_ = false;
                return;
            }
        }

        const auto pool_index = acquirePool(kStagingAcquireTimeout);
        if (!pool_index.has_value()) {
            std::shared_ptr<BatchState> batch;
            {
                std::lock_guard<std::mutex> lock(pending_mutex_);
                batch = pending_batches_.front();
                pending_batches_.pop_front();
            }
            batch->fail(transferError(TransferStatus::RESOURCE_EXHAUSTED,
                                      batch->next_descriptor,
                                      std::min(batch->next_descriptor + staging_pool_capacity_,
                                               batch->descriptors.size())));
            batch->admissionFinished();
            continue;
        }

        auto   pool_work       = std::make_shared<PoolWork>();
        pool_work->pool_index  = *pool_index;
        size_t remaining_slots = staging_pool_capacity_;
        while (remaining_slots > 0) {
            std::shared_ptr<BatchState> batch;
            {
                std::lock_guard<std::mutex> lock(pending_mutex_);
                if (pending_batches_.empty()) {
                    break;
                }
                batch = pending_batches_.front();
            }

            const size_t begin = batch->next_descriptor;
            const size_t end   = std::min(begin + remaining_slots, batch->descriptors.size());
            auto slice         = std::make_shared<Slice>();
            slice->batch       = batch;
            slice->begin       = begin;
            slice->end         = end;
            slice->descriptors.assign(batch->descriptors.begin() + begin, batch->descriptors.begin() + end);
            slice->group_sets.assign(batch->group_sets.begin() + begin, batch->group_sets.begin() + end);
            slice->leases.reserve(end - begin);
            slice->hosts.reserve(end - begin);
            for (size_t index = begin; index < end; ++index) {
                auto lease = staging_pools_[*pool_index]->malloc();
                RTP_LLM_CHECK(lease.has_value());
                slice->hosts.push_back(lease->blockBuffer(batch->group_sets[index]->payloadBytes()));
                slice->leases.push_back(std::move(*lease));
            }

            TransferStatus stage_one = TransferStatus::OK;
            for (size_t part_begin = 0; part_begin < slice->descriptors.size();) {
                const auto* disk_pool = slice->group_sets[part_begin]->diskPool().get();
                size_t      part_end  = part_begin + 1;
                while (part_end < slice->descriptors.size()
                       && slice->group_sets[part_end]->diskPool().get() == disk_pool) {
                    ++part_end;
                }
                const std::vector<HostBufferView> part_hosts(slice->hosts.begin() + part_begin,
                                                             slice->hosts.begin() + part_end);
                const std::vector<TransferDescriptor> part_descriptors(
                    slice->descriptors.begin() + part_begin, slice->descriptors.begin() + part_end);
                const std::vector<const GroupSet*> part_group_sets(slice->group_sets.begin() + part_begin,
                                                                   slice->group_sets.begin() + part_end);
                stage_one = host_disk_executor_.execute(part_hosts, part_descriptors, part_group_sets);
                if (stage_one != TransferStatus::OK) {
                    break;
                }
                part_begin = part_end;
            }

            const bool batch_finished = end == batch->descriptors.size();
            {
                std::lock_guard<std::mutex> lock(pending_mutex_);
                batch->next_descriptor = stage_one == TransferStatus::OK ? end : batch->descriptors.size();
                if (batch_finished || stage_one != TransferStatus::OK) {
                    pending_batches_.pop_front();
                }
            }
            if (stage_one != TransferStatus::OK) {
                logSliceFailure(slice->descriptors, begin, "stage1");
                batch->fail(transferError(stage_one, begin, end));
                slice->leases.clear();
                batch->admissionFinished();
                continue;
            }

            batch->sliceSubmitted();
            pool_work->slices.push_back(std::move(slice));
            remaining_slots -= end - begin;
            if (batch_finished) {
                batch->admissionFinished();
            }
        }

        if (pool_work->slices.empty()) {
            releasePool(*pool_index);
            continue;
        }

        setPoolState(*pool_index, PoolState::STAGE2_READY);
        const bool accepted = staging_to_device_task_pool_->submit([this, pool_work] {
            setPoolState(pool_work->pool_index, PoolState::STAGE2_IN_FLIGHT);
            for (const auto& slice : pool_work->slices) {
                try {
                    const TransferStatus status =
                        device_host_executor_.execute(slice->hosts, slice->descriptors, slice->group_sets);
                    if (status != TransferStatus::OK) {
                        logSliceFailure(slice->descriptors, slice->begin, "stage2");
                        slice->batch->fail(transferError(status, slice->begin, slice->end));
                    }
                } catch (const std::exception& error) {
                    logSliceFailure(slice->descriptors, slice->begin, "stage2");
                    slice->batch->fail(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, error.what()));
                } catch (...) {
                    logSliceFailure(slice->descriptors, slice->begin, "stage2");
                    slice->batch->fail(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "unknown stage-two exception"));
                }
                slice->leases.clear();
                slice->batch->sliceFinished();
            }
            releasePool(pool_work->pool_index);
        });
        if (!accepted) {
            for (const auto& slice : pool_work->slices) {
                logSliceFailure(slice->descriptors, slice->begin, "stage2 enqueue");
                slice->batch->fail(ErrorInfo(
                    ErrorCode::EXECUTION_EXCEPTION, "RESOURCE_EXHAUSTED: staging-to-device queue is full or stopped"));
                slice->leases.clear();
                slice->batch->sliceFinished();
            }
            releasePool(*pool_index);
        }
    }
}

std::optional<size_t> DeviceDiskTransferExecutor::acquirePool(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(pool_mutex_);
    const bool available = pool_cv_.wait_for(lock, timeout, [this] {
        return pool_states_[0] == PoolState::FREE || pool_states_[1] == PoolState::FREE;
    });
    if (!available) {
        return std::nullopt;
    }
    for (size_t offset = 0; offset < pool_states_.size(); ++offset) {
        const size_t index = (next_pool_index_ + offset) % pool_states_.size();
        if (pool_states_[index] == PoolState::FREE) {
            pool_states_[index] = PoolState::STAGE1_IN_FLIGHT;
            next_pool_index_     = (index + 1) % pool_states_.size();
            return index;
        }
    }
    return std::nullopt;
}

void DeviceDiskTransferExecutor::setPoolState(size_t pool_index, PoolState state) {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    pool_states_[pool_index] = state;
}

void DeviceDiskTransferExecutor::releasePool(size_t pool_index) {
    {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        pool_states_[pool_index] = PoolState::FREE;
    }
    pool_cv_.notify_one();
}

}  // namespace rtp_llm
