#include "rtp_llm/cpp/cache/block_tree_cache/transfer/PerRankBlockTransferEngine.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceHostCopyGeometry.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DiskTransferUtils.h"

#include <exception>
#include <iterator>
#include <memory>
#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"
#include <algorithm>
#include <numeric>
#include <limits>
#include <stdexcept>
#include <array>
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/CrcCopyExecutor.h"
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheLayout.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DeviceBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DiskBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/HostBlockPool.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferBatchAsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferStageState.h"

namespace rtp_llm {

namespace {

std::shared_ptr<TransferAsyncContext> invalidTransferContext(const TransferTask& task) {
    TransferResult result(ErrorInfo(ErrorCode::INVALID_PARAMS, "invalid block transfer request"));
    result.buffer_safe = true;
    return std::make_shared<CompletedTransferAsyncContext>(
        std::move(result), task.descriptors().size(), task.completionProof());
}

// Each load owner has private FULL and SWA/LINEAR pools with equal byte budgets.
// Its reference block count is a share of the rank-wide SWA block budget.
std::array<std::shared_ptr<HostStagingBlockPool>, 2> makeLoadStaging(const std::vector<GroupSetPtr>& groups,
                                                                     size_t                          block_count,
                                                                     size_t                          batch_limit,
                                                                     bool                            enabled,
                                                                     int64_t                         world_rank,
                                                                     size_t                          worker_index) {
    std::array<std::shared_ptr<HostStagingBlockPool>, 2> result;
    if (!enabled)
        return result;
    size_t                max_stride = 0;
    std::array<size_t, 2> strides{};
    for (const auto& group : groups) {
        if (!group || !group->diskPool() || group->devicePools().empty()
            || (!group->crcEnabled() && !isPlainDeviceTransferGroup(*group)))
            continue;
        const auto required = std::max(group->storageBytes(), group->diskPool()->strideBytes());
        if (required > std::numeric_limits<size_t>::max() - 4095)
            throw std::overflow_error("async disk staging stride overflow");
        const size_t stride = ((required + 4095) / 4096) * 4096;
        const size_t lane   = diskLoadStagingLane(group->groupType());
        strides[lane]       = std::max(strides[lane], stride);
        max_stride          = std::max(max_stride, stride);
    }
    if (!max_stride)
        return result;
    const size_t reference_stride = strides[1] ? strides[1] : max_stride;
    if (!block_count)
        throw std::invalid_argument("async staging block count must be >= load worker count");
    if (block_count > std::numeric_limits<size_t>::max() / reference_stride / 2)
        throw std::overflow_error("async disk staging size overflow");
    const size_t lane_bytes = block_count * reference_stride;
    for (size_t lane = 0; lane < result.size(); ++lane) {
        if (!strides[lane])
            strides[lane] = reference_stride;
        const size_t capacity = lane_bytes / strides[lane];
        if (capacity < batch_limit) {
            RTP_LLM_LOG_ERROR("async staging pool too small: rank=%ld worker=%zu class=%s capacity=%zu batch_limit=%zu",
                              world_rank,
                              worker_index,
                              lane == 0 ? "FULL" : "SWA/LINEAR",
                              capacity,
                              batch_limit);
            throw std::invalid_argument("async staging pool capacity must be >= max_descriptors_per_batch");
        }
        result[lane] = std::make_shared<HostStagingBlockPool>(capacity, strides[lane]);
    }
    RTP_LLM_LOG_INFO(
        "async staging private per load owner: rank=%ld worker=%zu swa_reference_blocks=%zu reference_stride=%zu "
        "full_blocks=%zu full_stride=%zu swa_blocks=%zu swa_stride=%zu "
        "data_bytes=%zu allocated_bytes=%zu",
        world_rank,
        worker_index,
        block_count,
        reference_stride,
        result[0]->capacity(),
        strides[0],
        result[1]->capacity(),
        strides[1],
        result[0]->capacity() * strides[0] + result[1]->capacity() * strides[1],
        result[0]->backingBytes() + result[1]->backingBytes());
    return result;
}

bool validDescriptor(const TransferDescriptor& descriptor, const GroupSet& group) {
    if (!descriptor.isExecutable())
        return false;
    for (const auto tier : {descriptor.source_tier, descriptor.target_tier}) {
        const auto& blocks = descriptor.blocksAt(tier);
        if (tier == Tier::DEVICE) {
            if (blocks.size() != group.devicePools().size())
                return false;
            for (size_t i = 0; i < blocks.size(); ++i)
                if (!group.devicePools()[i]->validBlock(blocks[i]))
                    return false;
        } else {
            const auto pool = tier == Tier::HOST ? std::static_pointer_cast<IBlockPool>(group.hostPool()) :
                                                   std::static_pointer_cast<IBlockPool>(group.diskPool());
            if (!pool || blocks.size() != 1 || !pool->validBlock(blocks[0]))
                return false;
        }
    }
    return true;
}

}  // namespace

PerRankBlockTransferEngine::PerRankBlockTransferEngine(std::vector<GroupSetPtr> group_sets,
                                                       bool                     enable_disk_cache,
                                                       size_t                   device_disk_staging_block_count,
                                                       size_t                   max_descriptors_per_batch,
                                                       size_t                   transfer_worker_count,
                                                       size_t                   transfer_queue_max_size,
                                                       std::shared_ptr<BlockTreeCacheMetricsReporter> metrics_reporter,
                                                       int64_t                                        world_rank,
                                                       size_t               btc_crc_slot_count,
                                                       DiskIOExecutorConfig disk_io_config):
    group_sets_(std::move(group_sets)),
    max_descriptors_per_batch_(max_descriptors_per_batch),
    enable_disk_cache_(enable_disk_cache),
    transfer_worker_count_(transfer_worker_count),
    crc_slot_count_(btc_crc_slot_count) {
    RTP_LLM_CHECK(max_descriptors_per_batch > 0);
    RTP_LLM_CHECK(transfer_worker_count > 0 && transfer_worker_count <= std::numeric_limits<size_t>::max() / 2);
    RTP_LLM_CHECK(btc_crc_slot_count > 0 && btc_crc_slot_count <= std::numeric_limits<size_t>::max() / 2);
    RTP_LLM_LOG_INFO("transfer_lifecycle event=worker_init rank=%ld engine=%p group_set=%p",
                     world_rank,
                     static_cast<const void*>(this),
                     group_sets_.empty() ? nullptr : static_cast<const void*>(group_sets_.front().get()));
    RTP_LLM_CHECK(transfer_worker_count <= std::numeric_limits<size_t>::max() / (2 * btc_crc_slot_count));
    if (transfer_queue_max_size < transfer_worker_count
        || transfer_queue_max_size > std::numeric_limits<size_t>::max() / max_descriptors_per_batch)
        throw std::invalid_argument("WorkLoop transfer requires Q >= worker count and nonoverflowing Q*M");

    std::vector<GroupSetPtr> crc_groups;
    for (const auto& group : group_sets_) {
        if (group && group->crcEnabled() && !group->devicePools().empty()
            && (group->hostPool() || (enable_disk_cache && group->diskPool())))
            crc_groups.push_back(group);
    }
    if (enable_disk_cache_ && std::any_of(group_sets_.begin(), group_sets_.end(), [](const auto& group) {
            return group && group->diskPool();
        })) {
        // Disk reads use LOAD and writes use EVICT. Q covers all owners/files;
        // each resident batch has at most one disk stage in flight.
        if (!disk_io_config.setQueueSizesFromBatchCapacity(
                transfer_queue_max_size, transfer_queue_max_size, max_descriptors_per_batch_))
            throw std::invalid_argument("async disk I/O queue capacity overflow");
        if (!disk_io_config.valid())
            throw std::invalid_argument("invalid engine-wide disk I/O budgets");
        disk_io_executor_ = std::make_unique<DiskIOExecutor>(disk_io_config);
        RTP_LLM_LOG_INFO("disk I/O queue capacities: read=%zu write=%zu resident_batches_per_direction=%zu "
                         "batch_limit=%zu read_threads=%zu write_threads=%zu",
                         disk_io_config.read_queue_size,
                         disk_io_config.write_queue_size,
                         transfer_queue_max_size,
                         max_descriptors_per_batch_,
                         disk_io_config.read_thread_count,
                         disk_io_config.write_thread_count);
    }
    const bool needs_crc = !crc_groups.empty();
    const auto make_crc  = [&](CrcCopyDirection direction) -> std::unique_ptr<CrcCopyExecutor> {
        return needs_crc ? std::make_unique<CrcCopyExecutor>(
                               crc_groups, direction, max_descriptors_per_batch_, crc_slot_count_, world_rank) :
                            nullptr;
    };
    load_loops_.reserve(transfer_worker_count);
    evict_loops_.reserve(transfer_worker_count);
    for (size_t worker = 0; worker < transfer_worker_count; ++worker) {
        const auto queue_capacity = transfer_queue_max_size / transfer_worker_count
                                    + (worker < transfer_queue_max_size % transfer_worker_count ? 1 : 0);
        const size_t staging_blocks = device_disk_staging_block_count / transfer_worker_count
                                      + (worker < device_disk_staging_block_count % transfer_worker_count ? 1 : 0);
        auto staging = makeLoadStaging(
            group_sets_, staging_blocks, max_descriptors_per_batch_, enable_disk_cache, world_rank, worker);
        load_loops_.push_back(std::make_unique<TransferWorkLoop>(group_sets_,
                                                                 make_crc(CrcCopyDirection::LOAD),
                                                                 disk_io_executor_.get(),
                                                                 TransferOperation::LOAD,
                                                                 max_descriptors_per_batch_,
                                                                 queue_capacity,
                                                                 std::move(staging[0]),
                                                                 metrics_reporter,
                                                                 world_rank,
                                                                 worker,
                                                                 std::move(staging[1])));
        evict_loops_.push_back(std::make_unique<TransferWorkLoop>(group_sets_,
                                                                  make_crc(CrcCopyDirection::EVICT),
                                                                  disk_io_executor_.get(),
                                                                  TransferOperation::EVICT,
                                                                  max_descriptors_per_batch_,
                                                                  queue_capacity,
                                                                  nullptr,
                                                                  metrics_reporter,
                                                                  world_rank,
                                                                  worker));
    }
    // Warm both disk pools before owners or business producers can submit.
    if (disk_io_executor_)
        disk_io_executor_->start();
    for (auto& loop : load_loops_)
        loop->start();
    for (auto& loop : evict_loops_)
        loop->start();
    RTP_LLM_LOG_INFO("BlockTree WorkLoop transfer: owners_per_direction=%zu C=%zu total_crc_slots_per_device=%zu M=%zu "
                     "resident_batch_budget_per_direction=%zu staging_swa_reference_blocks_per_rank=%zu "
                     "staging_private_to_load_owners=1",
                     transfer_worker_count,
                     btc_crc_slot_count,
                     needs_crc ? crcTransferWorkspaceCount(transfer_worker_count, btc_crc_slot_count) : 0,
                     max_descriptors_per_batch_,
                     transfer_queue_max_size,
                     device_disk_staging_block_count);
}

PerRankBlockTransferEngine::~PerRankBlockTransferEngine() {
    shutdown();
}

void PerRankBlockTransferEngine::stopAdmission() {
    for (auto& loop : load_loops_)
        loop->stopAdmission();
    for (auto& loop : evict_loops_)
        loop->stopAdmission();
}

void PerRankBlockTransferEngine::shutdown() {
    stopAdmission();
    for (auto& loop : load_loops_)
        loop->shutdown();
    for (auto& loop : evict_loops_)
        loop->shutdown();
    // Accepted CRC stages may still submit their disk successors while draining.
    if (disk_io_executor_)
        disk_io_executor_->stop();
}

void PerRankBlockTransferEngine::cancelPendingStagingTransfers() {
    for (auto& loop : load_loops_)
        loop->cancelPendingStagingTransfers();
}

BlockTreeQueueSizes PerRankBlockTransferEngine::queueSizes() const {
    BlockTreeQueueSizes sizes;
    for (const auto& loop : load_loops_)
        sizes.load += loop->outstandingBatches();
    for (const auto& loop : evict_loops_)
        sizes.background += loop->outstandingBatches();
    return sizes;
}

std::shared_ptr<TransferAsyncContext> PerRankBlockTransferEngine::execute(TransferTask task) {
    const auto                             proof = task.completionProof();
    block_tree_cache_detail::ScopeRollback unsubmitted([proof] {
        if (proof)
            proof->report(true);
    });
    const auto                             descriptors = task.descriptors();
    const size_t                           count       = descriptors.size();
    if (count == 0 || !task.sanityCheck()) {
        return invalidTransferContext(task);
    }
    if (task.expired()) {
        TransferResult result(ErrorInfo(ErrorCode::DEADLINE_EXCEEDED, "transfer deadline exceeded before dispatch"),
                              CacheCopyError::NOT_EXECUTED);
        result.buffer_safe = true;
        return std::make_shared<CompletedTransferAsyncContext>(std::move(result), count, proof);
    }

    auto context = std::make_shared<TransferBatchAsyncContext>(count);
    auto state   = std::make_shared<TransferStageState>(
        [context](TransferResult result) { context->complete(std::move(result)); }, count);
    auto                                   safety = std::make_shared<TransferBufferSafetyTracker>([proof] {
        if (proof)
            proof->report(true);
    });
    block_tree_cache_detail::ScopeRollback close_submission([safety] { safety->finishSubmitting(); });
    unsubmitted.dismiss();
    bool any_registered = false;
    for (size_t begin = 0; begin < count;) {
        std::vector<size_t>                   indices;
        bool                                  registered = false;
        TransferResult                        error;
        std::shared_ptr<TransferBufferSafety> batch_proof;
        std::shared_ptr<TransferAsyncContext> child;
        TransferAsyncContext::DoneCallback    complete;
        bool                                  attempted = false;
        try {
            const size_t batch_begin = begin;
            const auto&  first       = descriptors[begin];
            const size_t limit       = batchLimit();
            const size_t max_end     = begin + std::min(limit, count - begin);
            size_t       end         = begin + 1;
            while (end < max_end && descriptors[end].group_set_id == first.group_set_id
                   && descriptors[end].source_tier == first.source_tier
                   && descriptors[end].target_tier == first.target_tier) {
                ++end;
            }
            begin = end;
            indices.resize(end - batch_begin);
            std::iota(indices.begin(), indices.end(), batch_begin);
            batch_proof    = safety->addBatch();
            auto completed = std::make_shared<std::atomic<bool>>(false);
            complete       = [state, indices, completed](TransferResult result) {
                if (!completed->exchange(true, std::memory_order_acq_rel))
                    state->completeBatch(std::move(result), indices);
            };
            auto batch_task = task.subtask(batch_begin, end, batch_proof);
            state->addBatch();
            registered = any_registered = true;
            attempted                   = true;
            child                       = executeBatch(std::move(batch_task));
            if (!child) {
                error = ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "transfer engine returned a null context");
            } else {
                child->onDone(complete);
            }
        } catch (const std::exception& exception) {
            if (batch_proof && !attempted)
                batch_proof->report(true);
            if (!any_registered) {
                throw;
            }
            error = TransferResult::executionException(false, exception.what());
            if (child)
                error.status.setErrorCode(ErrorCode::TRANSFER_COMPLETION_UNKNOWN);
        } catch (...) {
            if (batch_proof && !attempted)
                batch_proof->report(true);
            if (!any_registered) {
                throw;
            }
            error = TransferResult::executionException(false, "unknown transfer submission exception");
            if (child)
                error.status.setErrorCode(ErrorCode::TRANSFER_COMPLETION_UNKNOWN);
        }
        if (!error.ok()) {
            if (!registered) {
                close_submission.run();
                state->finishSubmitting(std::move(error));
                return context;
            }
            complete(std::move(error));
        }
    }
    // Already submitted batches retain their resources until every completion
    // arrives, including when a later batch is rejected or throws.
    close_submission.run();
    state->finishSubmitting();
    return context;
}

std::shared_ptr<TransferAsyncContext> PerRankBlockTransferEngine::executeBatch(TransferTask task) {
    block_tree_cache_detail::ScopeRollback unsubmitted([proof = task.completionProof()] {
        if (proof)
            proof->report(true);
    });
    const auto&                            descriptors  = task.descriptors();
    const Tier                             source       = descriptors.front().source_tier;
    const Tier                             target       = descriptors.front().target_tier;
    const auto                             group_set_id = descriptors.front().group_set_id;
    if (group_set_id >= group_sets_.size() || !group_sets_[group_set_id])
        return invalidTransferContext(task);
    const auto* group_set = group_sets_[group_set_id].get();
    for (const auto& descriptor : descriptors) {
        if (!validDescriptor(descriptor, *group_set))
            return invalidTransferContext(task);
    }
    if ((source == Tier::DISK || target == Tier::DISK) && !enable_disk_cache_)
        return invalidTransferContext(task);
    if ((source == Tier::DEVICE || target == Tier::DEVICE) && !group_set->crcEnabled()
        && !isPlainDeviceTransferGroup(*group_set))
        return invalidTransferContext(task);
    auto&      loops  = isLoadTransfer(source, target) ? load_loops_ : evict_loops_;
    auto&      next   = isLoadTransfer(source, target) ? next_load_ : next_evict_;
    const auto worker = loops.size() == 1 ? 0 : next.fetch_add(1, std::memory_order_relaxed) % loops.size();
    std::shared_ptr<TransferAsyncContext> context;
    unsubmitted.dismiss();
    for (size_t offset = 0; offset < loops.size(); ++offset) {
        bool retryable = false;
        // The last owner returns the final rejection and records it once.
        context = loops[(worker + offset) % loops.size()]->execute(
            std::move(task), {}, offset + 1 < loops.size() ? &retryable : nullptr);
        if (!retryable) {
            break;
        }
    }
    return context;
}

void PerRankBlockTransferEngine::waitForIdle() {
    BlockTreeQueueSizes pending;
    do {
        for (auto& loop : load_loops_)
            loop->waitForIdle();
        for (auto& loop : evict_loops_)
            loop->waitForIdle();
        // A later owner's callback may have submitted to an earlier owner.
        pending = queueSizes();
    } while (pending.load != 0 || pending.background != 0);
}

}  // namespace rtp_llm
