#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceDiskTransferExecutor.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <exception>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <utility>

#include <torch/torch.h>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceHostTransferExecutor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/HostDiskTransferExecutor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferBatchAsyncContext.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {

constexpr size_t kStagingAlignment = 4096;

constexpr std::chrono::milliseconds kStagingAcquireTimeout{1000};
constexpr size_t                    kPipelineWorkerCount   = 1;
constexpr size_t                    kPipelineQueueCapacity = 1000;

size_t alignUp(size_t value, size_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

}  // namespace

struct DeviceDiskTransferExecutor::PipelineBatchState {
    PipelineBatchState(size_t logical_descriptor_count, std::shared_ptr<void> completion_guard):
        context(std::make_shared<TransferBatchAsyncContext>(std::move(completion_guard))),
        logical_descriptors(logical_descriptor_count) {}

    bool failed() const {
        std::lock_guard<std::mutex> lock(mutex);
        return !error.ok();
    }

    void fail(ErrorInfo failure) {
        std::lock_guard<std::mutex> lock(mutex);
        if (error.ok()) {
            error = std::move(failure);
        }
    }

    void stageTwoScheduled() {
        std::lock_guard<std::mutex> lock(mutex);
        ++pending_stage_two;
    }

    void stageOneFinished() {
        std::lock_guard<std::mutex> lock(mutex);
        stage_one_finished = true;
        maybeCompleteLocked();
    }

    void stageTwoFinished() {
        std::lock_guard<std::mutex> lock(mutex);
        RTP_LLM_CHECK(pending_stage_two > 0);
        --pending_stage_two;
        maybeCompleteLocked();
    }

    void maybeCompleteLocked() {
        if (stage_one_finished && pending_stage_two == 0) {
            context->complete(error);
        }
    }

    std::shared_ptr<TransferBatchAsyncContext> context;
    size_t                                     logical_descriptors{0};
    mutable std::mutex                         mutex;
    ErrorInfo                                  error{ErrorInfo::OkStatus()};
    size_t                                     pending_stage_two{0};
    bool                                       stage_one_finished{false};
};

struct DeviceDiskTransferExecutor::PipelineSlice {
    size_t                                       pool_index{0};
    size_t                                       begin{0};
    size_t                                       end{0};
    std::vector<TransientHostStagingPool::Lease> leases;
    std::vector<HostBufferView>                  hosts;
    std::vector<TransferDescriptor>              descriptors;
    std::vector<const GroupSet*>                 group_sets;
    std::shared_ptr<PipelineBatchState>          batch;
};

DeviceDiskTransferExecutor::TransientHostStagingPool::TransientHostStagingPool(size_t block_count,
                                                                               size_t stride_bytes,
                                                                               bool   try_pin_memory):
    block_count_(block_count), stride_bytes_(stride_bytes) {
    RTP_LLM_CHECK_WITH_INFO(block_count_ > 0, "staging block_count must be > 0");
    RTP_LLM_CHECK_WITH_INFO(stride_bytes_ > 0, "staging stride_bytes must be > 0");
    RTP_LLM_CHECK_WITH_INFO(block_count_ <= (std::numeric_limits<size_t>::max() - kStagingAlignment) / stride_bytes_,
                            "staging backing size overflow: blocks=%zu stride=%zu",
                            block_count_,
                            stride_bytes_);

    const size_t total_bytes = block_count_ * stride_bytes_;
    RTP_LLM_CHECK_WITH_INFO(total_bytes + kStagingAlignment <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
                            "staging backing exceeds int64 tensor dim: total_bytes=%zu",
                            total_bytes);
    auto cpu = torch::empty({static_cast<int64_t>(total_bytes + kStagingAlignment)},
                            torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
    if (try_pin_memory) {
        try {
            backing_ = cpu.pin_memory();
        } catch (const std::exception& e) {
            RTP_LLM_LOG_WARNING("pin staging memory failed, fallback to pageable CPU memory, error=%s", e.what());
        }
    }
    if (!backing_.defined() || !backing_.is_pinned()) {
        backing_ = cpu;
    }
    RTP_LLM_CHECK(backing_.defined() && backing_.is_contiguous());
    const auto raw_base = reinterpret_cast<uintptr_t>(backing_.data_ptr<uint8_t>());
    base_ptr_           = reinterpret_cast<uint8_t*>(alignUp(raw_base, kStagingAlignment));

    free_id_list_.reserve(block_count_);
    for (size_t block_id = 0; block_id < block_count_; ++block_id) {
        free_id_list_.push_back(block_id);
    }
    RTP_LLM_LOG_INFO("device-disk staging pool ready: blocks=%zu stride=%zu total_bytes=%zu pinned=%d",
                     block_count_,
                     stride_bytes_,
                     total_bytes,
                     static_cast<int>(backing_.is_pinned()));
}

std::optional<DeviceDiskTransferExecutor::TransientHostStagingPool::Lease>
DeviceDiskTransferExecutor::TransientHostStagingPool::tryAcquire() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (free_id_list_.empty()) {
        return std::nullopt;
    }
    const size_t block_id = free_id_list_.back();
    free_id_list_.pop_back();
    return Lease(this, block_id);
}

void DeviceDiskTransferExecutor::TransientHostStagingPool::release(size_t block_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    free_id_list_.push_back(block_id);
}

HostBufferView DeviceDiskTransferExecutor::TransientHostStagingPool::viewFor(size_t block_id,
                                                                             size_t payload_bytes) const {
    RTP_LLM_CHECK(block_id < block_count_);
    RTP_LLM_CHECK_WITH_INFO(
        payload_bytes <= stride_bytes_, "staging payload %zu exceeds stride %zu", payload_bytes, stride_bytes_);
    void* base = base_ptr_ + block_id * stride_bytes_;
    return HostBufferView{base, payload_bytes, stride_bytes_};
}

DeviceDiskTransferExecutor::DeviceDiskTransferExecutor(DeviceHostTransferExecutor&     device_host_executor,
                                                       HostDiskTransferExecutor&       host_disk_executor,
                                                       const std::vector<GroupSetPtr>& group_sets,
                                                       size_t                          staging_block_count):
    device_host_executor_(device_host_executor), host_disk_executor_(host_disk_executor) {
    size_t max_payload_bytes = 0;
    for (const auto& group_set : group_sets) {
        if (group_set != nullptr) {
            max_payload_bytes = std::max(max_payload_bytes, group_set->payloadBytes());
        }
    }
    RTP_LLM_CHECK_WITH_INFO(max_payload_bytes > 0, "device-disk staging requires a non-zero group payload");
    const size_t stride_bytes  = alignUp(max_payload_bytes, kStagingAlignment);
    active_staging_pool_count_ = staging_block_count >= 2 && staging_block_count % 2 == 0 ? 2 : 1;
    staging_pool_capacity_     = staging_block_count / active_staging_pool_count_;
    for (size_t pool_index = 0; pool_index < active_staging_pool_count_; ++pool_index) {
        staging_pools_[pool_index] = std::make_unique<TransientHostStagingPool>(staging_pool_capacity_, stride_bytes);
        staging_pool_states_[pool_index] = StagingPoolState::FREE;
    }
    disk_to_staging_task_pool_ =
        std::make_unique<BlockTreeTaskPool>(kPipelineWorkerCount, kPipelineQueueCapacity, "BlockDiskToStaging");
    staging_to_device_task_pool_ =
        std::make_unique<BlockTreeTaskPool>(kPipelineWorkerCount, kPipelineQueueCapacity, "BlockStagingToDevice");
    RTP_LLM_CHECK_WITH_INFO(disk_to_staging_task_pool_->start(), "failed to start Disk->staging task queue");
    RTP_LLM_CHECK_WITH_INFO(staging_to_device_task_pool_->start(), "failed to start staging->Device task queue");
}

DeviceDiskTransferExecutor::~DeviceDiskTransferExecutor() = default;

TransferStatus DeviceDiskTransferExecutor::execute(const TransferDescriptor& desc, const GroupSet& group) {
    const bool device_to_disk = desc.source_tier == Tier::DEVICE && desc.target_tier == Tier::DISK;
    const auto pool_index = acquireStagingPool(kStagingAcquireTimeout);
    if (!pool_index.has_value()) {
        return TransferStatus::RESOURCE_EXHAUSTED;
    }
    auto lease = staging_pools_[*pool_index]->tryAcquire();
    if (!lease.has_value()) {
        releaseStagingPool(*pool_index);
        return TransferStatus::RESOURCE_EXHAUSTED;
    }
    setStagingPoolState(*pool_index, StagingPoolState::STAGE1_IN_FLIGHT);
    const HostBufferView host = lease->view(group.payloadBytes());

    TransferStatus result = TransferStatus::OK;
    try {
        if (device_to_disk) {
            const TransferStatus stage_one = device_host_executor_.deviceToHost(desc, group, host);
            if (stage_one != TransferStatus::OK) {
                result = stage_one;
            } else {
                result = host_disk_executor_.hostToDisk({host}, {desc}, {&group});
            }
        } else {
            const TransferStatus stage_one = host_disk_executor_.diskToHost({desc}, {&group}, {host});
            if (stage_one != TransferStatus::OK) {
                result = stage_one;
            } else {
                result = device_host_executor_.hostToDevice(host, desc, group);
            }
        }
    } catch (...) {
        lease.reset();
        releaseStagingPool(*pool_index);
        throw;
    }
    lease.reset();
    releaseStagingPool(*pool_index);
    return result;
}

std::optional<size_t> DeviceDiskTransferExecutor::acquireStagingPool(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(staging_state_mutex_);
    const auto                   has_free_pool = [this] {
        for (size_t pool_index = 0; pool_index < active_staging_pool_count_; ++pool_index) {
            if (staging_pool_states_[pool_index] == StagingPoolState::FREE) {
                return true;
            }
        }
        return false;
    };
    if (!staging_state_cv_.wait_for(lock, timeout, has_free_pool)) {
        return std::nullopt;
    }
    for (size_t offset = 0; offset < active_staging_pool_count_; ++offset) {
        const size_t pool_index = (next_staging_pool_index_ + offset) % active_staging_pool_count_;
        if (staging_pool_states_[pool_index] == StagingPoolState::FREE) {
            staging_pool_states_[pool_index] = StagingPoolState::STAGE1_FILLING;
            next_staging_pool_index_         = (pool_index + 1) % active_staging_pool_count_;
            return pool_index;
        }
    }
    return std::nullopt;
}

void DeviceDiskTransferExecutor::setStagingPoolState(size_t pool_index, StagingPoolState state) {
    std::lock_guard<std::mutex> lock(staging_state_mutex_);
    RTP_LLM_CHECK(pool_index < active_staging_pool_count_);
    staging_pool_states_[pool_index] = state;
}

void DeviceDiskTransferExecutor::releaseStagingPool(size_t pool_index) {
    {
        std::lock_guard<std::mutex> lock(staging_state_mutex_);
        RTP_LLM_CHECK(pool_index < active_staging_pool_count_);
        staging_pool_states_[pool_index] = StagingPoolState::FREE;
    }
    staging_state_cv_.notify_all();
}

std::shared_ptr<AsyncContext>
DeviceDiskTransferExecutor::diskToDevice(const std::vector<TransferDescriptor>& descriptors,
                                         const std::vector<const GroupSet*>&    group_sets,
                                         std::shared_ptr<void>                  completion_guard) {
    auto       batch    = std::make_shared<PipelineBatchState>(descriptors.size(), std::move(completion_guard));
    const bool accepted = disk_to_staging_task_pool_->trySubmit([this, descriptors, group_sets, batch] {
        for (size_t begin = 0; begin < descriptors.size() && !batch->failed(); begin += staging_pool_capacity_) {
            const size_t end        = std::min(begin + staging_pool_capacity_, descriptors.size());
            const auto   pool_index = acquireStagingPool(kStagingAcquireTimeout);
            if (!pool_index.has_value()) {
                batch->fail(ErrorInfo(ErrorCode::DEADLINE_EXCEEDED,
                                      "DEADLINE_EXCEEDED: timed out waiting for a free staging slice"));
                break;
            }

            auto slice        = std::make_shared<PipelineSlice>();
            slice->pool_index = *pool_index;
            slice->begin      = begin;
            slice->end        = end;
            slice->batch      = batch;
            slice->descriptors.assign(descriptors.begin() + begin, descriptors.begin() + end);
            slice->group_sets.assign(group_sets.begin() + begin, group_sets.begin() + end);
            slice->leases.reserve(end - begin);
            slice->hosts.reserve(end - begin);
            bool acquired_all = true;
            for (size_t index = begin; index < end; ++index) {
                auto lease = staging_pools_[*pool_index]->tryAcquire();
                if (!lease.has_value()) {
                    acquired_all = false;
                    break;
                }
                slice->hosts.push_back(lease->view(group_sets[index]->payloadBytes()));
                slice->leases.push_back(std::move(*lease));
            }
            if (!acquired_all) {
                slice->leases.clear();
                releaseStagingPool(*pool_index);
                batch->fail(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "staging pool lease accounting failure"));
                break;
            }

            setStagingPoolState(*pool_index, StagingPoolState::STAGE1_IN_FLIGHT);
            TransferStatus stage_one = TransferStatus::DISK_IO_ERROR;
            try {
                stage_one        = TransferStatus::OK;
                size_t run_begin = 0;
                while (run_begin < slice->descriptors.size()) {
                    const auto* disk_pool = slice->group_sets[run_begin]->diskPool().get();
                    size_t      run_end   = run_begin + 1;
                    while (run_end < slice->descriptors.size()
                           && slice->group_sets[run_end]->diskPool().get() == disk_pool) {
                        ++run_end;
                    }
                    const std::vector<TransferDescriptor> run_descriptors(slice->descriptors.begin() + run_begin,
                                                                          slice->descriptors.begin() + run_end);
                    const std::vector<const GroupSet*>    run_group_sets(slice->group_sets.begin() + run_begin,
                                                                      slice->group_sets.begin() + run_end);
                    const std::vector<HostBufferView>     run_hosts(slice->hosts.begin() + run_begin,
                                                                slice->hosts.begin() + run_end);
                    stage_one = host_disk_executor_.diskToHost(run_descriptors, run_group_sets, run_hosts);
                    if (stage_one != TransferStatus::OK) {
                        break;
                    }
                    run_begin = run_end;
                }
            } catch (const std::exception& error) {
                slice->leases.clear();
                releaseStagingPool(*pool_index);
                batch->fail(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION,
                                      "Disk->Device stage1 threw, descriptor_range=[" + std::to_string(begin) + ","
                                          + std::to_string(end) + "): " + error.what()));
                break;
            } catch (...) {
                slice->leases.clear();
                releaseStagingPool(*pool_index);
                batch->fail(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION,
                                      "Disk->Device stage1 threw, descriptor_range=[" + std::to_string(begin) + ","
                                          + std::to_string(end) + ")"));
                break;
            }
            if (stage_one != TransferStatus::OK) {
                slice->leases.clear();
                releaseStagingPool(*pool_index);
                batch->fail(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION,
                                      "Disk->Device stage1 failed, descriptor_range=[" + std::to_string(begin) + ","
                                          + std::to_string(end)
                                          + "), logical_descriptors=" + std::to_string(descriptors.size())));
                break;
            }

            setStagingPoolState(*pool_index, StagingPoolState::STAGE2_READY);
            batch->stageTwoScheduled();
            const bool stage_two_accepted = staging_to_device_task_pool_->trySubmit([this, slice] {
                setStagingPoolState(slice->pool_index, StagingPoolState::STAGE2_IN_FLIGHT);
                try {
                    if (!slice->batch->failed()) {
                        const TransferStatus stage_two =
                            device_host_executor_.hostToDevice(slice->hosts, slice->descriptors, slice->group_sets);
                        if (stage_two != TransferStatus::OK) {
                            slice->batch->fail(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION,
                                                         "Disk->Device stage2 failed, descriptor_range=["
                                                             + std::to_string(slice->begin) + ","
                                                             + std::to_string(slice->end) + "), logical_descriptors="
                                                             + std::to_string(slice->batch->logical_descriptors)));
                        }
                    }
                } catch (const std::exception& error) {
                    slice->batch->fail(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION,
                                                 "Disk->Device stage2 threw, descriptor_range=["
                                                     + std::to_string(slice->begin) + "," + std::to_string(slice->end)
                                                     + "): " + error.what()));
                } catch (...) {
                    slice->batch->fail(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION,
                                                 "Disk->Device stage2 threw, descriptor_range=["
                                                     + std::to_string(slice->begin) + "," + std::to_string(slice->end)
                                                     + ")"));
                }
                slice->leases.clear();
                releaseStagingPool(slice->pool_index);
                slice->batch->stageTwoFinished();
            });
            if (!stage_two_accepted) {
                batch->fail(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION,
                                      "RESOURCE_EXHAUSTED: staging->Device task queue is full or stopped"));
                slice->leases.clear();
                releaseStagingPool(*pool_index);
                batch->stageTwoFinished();
                break;
            }
        }
        batch->stageOneFinished();
    });
    if (!accepted) {
        batch->fail(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION,
                              "RESOURCE_EXHAUSTED: Disk->staging task queue is full or stopped"));
        batch->stageOneFinished();
    }
    return batch->context;
}

}  // namespace rtp_llm
