#include "rtp_llm/cpp/cache/block_tree_cache/transfer/PerRankBlockTransferEngine.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceHostCopyGeometry.h"

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
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/AlignedHostMemory.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DeviceBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DiskBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/HostBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/remote_mock/RemoteMockAdapter.h"
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

// Each load owner has private FULL/default and SWA pools with equal byte budgets.
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
        const size_t lane   = group->groupType() == CacheGroupType::SWA ? 1 : 0;
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
                              lane == 0 ? "FULL" : "SWA",
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
        if (group && group->crcEnabled() && !group->devicePools().empty())
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
    if (descriptors.front().remote_target) {
        auto adapter = remote_adapter_.lock();
        if (!adapter)
            return invalidTransferContext(task);
        const auto proof   = task.completionProof();
        auto       context = std::make_shared<TransferBatchAsyncContext>(descriptors.size(), nullptr, proof);
        unsubmitted.dismiss();
        adapter->writeback(std::move(task), [context](ErrorInfo error) {
            TransferResult result(std::move(error));
            result.buffer_safe = true;
            context->complete(std::move(result));
        });
        return context;
    }
    if (descriptors.front().remote_source || descriptors.front().source_tier == Tier::NONE) {
        unsubmitted.dismiss();
        return executeRemote(std::move(task));
    }
    const Tier source       = descriptors.front().source_tier;
    const Tier target       = descriptors.front().target_tier;
    const auto group_set_id = descriptors.front().group_set_id;
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
    if (source == Tier::DEVICE && target == Tier::DISK) {
        unsubmitted.dismiss();
        return executeDeviceToDisk(std::move(task));
    }
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

std::shared_ptr<TransferAsyncContext> PerRankBlockTransferEngine::executeDeviceToDisk(TransferTask task) {
    const auto                             proof = task.completionProof();
    block_tree_cache_detail::ScopeRollback unsubmitted([proof] {
        if (proof)
            proof->report(true);
    });
    const auto& descriptors = task.descriptors();
    const auto  group_id    = descriptors.front().group_set_id;
    const auto& group_set   = *group_sets_[group_id];
    const auto& disk_pool   = *group_set.diskPool();
    const auto  stride      = disk_pool.strideBytes();
    const auto  payload     = group_set.payloadBytes();
    if (group_set.storageBytes() > stride || descriptors.size() > std::numeric_limits<size_t>::max() / stride) {
        return invalidTransferContext(task);
    }
    auto staging = std::make_shared<AlignedHostMemory>(descriptors.size() * stride, 4096, "device_disk_store");
    std::vector<TransferDescriptor> d2h_descs;
    std::vector<TransferDescriptor> h2disk_descs;
    std::vector<HostBufferView>     host_views;
    d2h_descs.reserve(descriptors.size());
    h2disk_descs.reserve(descriptors.size());
    host_views.reserve(descriptors.size());
    for (size_t i = 0; i < descriptors.size(); ++i) {
        d2h_descs.push_back(TransferDescriptor::deviceToHost(group_id, descriptors[i].source_blocks, 0));
        h2disk_descs.push_back(TransferDescriptor::hostToDisk(group_id, 0, descriptors[i].singleBlockAt(Tier::DISK)));
        host_views.push_back({staging->data() + i * stride, payload, stride});
    }
    auto context = std::make_shared<TransferBatchAsyncContext>(descriptors.size(), nullptr, proof);
    unsubmitted.dismiss();
    auto stage1_task = task.subtask(std::move(d2h_descs));
    auto stage2_task = task.subtask(std::move(h2disk_descs));
    executeHostBuffers(
        std::move(stage1_task),
        host_views,
        staging,
        [this, context, stage2_task = std::move(stage2_task), host_views = std::move(host_views), staging](
            TransferResult r1) mutable {
            if (!r1.ok()) {
                context->complete(std::move(r1));
                return;
            }
            if (stage2_task.expired()) {
                TransferResult expired(
                    ErrorInfo(ErrorCode::DEADLINE_EXCEEDED, "device-to-disk transfer expired before disk write"),
                    CacheCopyError::NOT_EXECUTED);
                expired.buffer_safe = true;
                context->complete(std::move(expired));
                return;
            }
            executeHostBuffers(std::move(stage2_task),
                               std::move(host_views),
                               std::move(staging),
                               [context](TransferResult r2) { context->complete(std::move(r2)); });
        });
    return context;
}

void PerRankBlockTransferEngine::setRemoteMockAdapter(const std::shared_ptr<RemoteMockAdapter>& adapter,
                                                      bool owns_device_allocations) {
    RTP_LLM_CHECK(remote_adapter_.expired());
    if (adapter)
        adapter->initializeWriteback(adapter, owns_device_allocations, max_descriptors_per_batch_);
    remote_adapter_          = adapter;
    owns_device_allocations_ = owns_device_allocations;
}

RemoteMockWritebackStats PerRankBlockTransferEngine::remoteWritebackStats() const {
    const auto adapter = remote_adapter_.lock();
    return adapter ? adapter->writebackStats() : RemoteMockWritebackStats{};
}

std::shared_ptr<TransferAsyncContext> PerRankBlockTransferEngine::executeRemote(TransferTask task) {
    const auto                             proof = task.completionProof();
    block_tree_cache_detail::ScopeRollback unsubmitted([proof] {
        if (proof)
            proof->report(true);
    });
    const auto adapter = remote_adapter_.lock();
    if (!adapter || task.expired()) {
        return invalidTransferContext(task);
    }
    RemoteMockLoadPlan plan;
    plan.deadline = task.deadline();
    try {
        for (const auto& desc : task.descriptors()) {
            if (!desc.isExecutable() || !desc.isRemoteMock() || desc.group_set_id >= group_sets_.size()
                || !group_sets_[desc.group_set_id]) {
                return invalidTransferContext(task);
            }
            const auto& pools = group_sets_[desc.group_set_id]->devicePools();
            if (desc.target_blocks.size() != pools.size()) {
                return invalidTransferContext(task);
            }
            for (size_t member = 0; member < pools.size(); ++member) {
                if (!pools[member]->validBlock(desc.target_blocks[member])) {
                    return invalidTransferContext(task);
                }
            }
            const auto& source = *desc.remote_source;
            const auto  frozen = adapter->freezeSource(desc.group_set_id, source.admission_offset, source.logical_index);
            if (!(source.session == frozen.session) || source.layout_hash != frozen.layout_hash
                || source.rank != frozen.rank || source.group != frozen.group || source.pool != frozen.pool
                || source.index != frozen.index || source.checksum != frozen.checksum) {
                return invalidTransferContext(task);
            }
            plan.targets.push_back({source, desc.target_blocks});
        }
        struct TargetPins {
            std::vector<std::pair<DeviceBlockPoolPtr, BlockIdxType>> pins;
            std::vector<GroupSetPtr>                                 groups;
            ~TargetPins() {
                for (const auto& pin : pins) {
                    pin.first->decTreeRef(pin.second, BlockTreeRefType::LOAD);
                }
            }
        };
        auto owner    = std::make_shared<TargetPins>();
        owner->groups = group_sets_;
        size_t count  = 0;
        for (const auto& desc : task.descriptors())
            count += desc.target_blocks.size();
        owner->pins.reserve(count);
        if (owns_device_allocations_)
            for (const auto& desc : task.descriptors()) {
                const auto& pools = group_sets_[desc.group_set_id]->devicePools();
                for (size_t member = 0; member < pools.size(); ++member) {
                    pools[member]->incTreeRef(desc.target_blocks[member], BlockTreeRefType::LOAD);
                    owner->pins.emplace_back(pools[member], desc.target_blocks[member]);
                }
            }
        plan.target_owner = std::move(owner);
        auto context      = std::make_shared<TransferBatchAsyncContext>(task.descriptors().size(), nullptr, proof);
        unsubmitted.dismiss();
        adapter->load(plan, [context](ErrorInfo error) {
            TransferResult result(std::move(error));
            result.buffer_safe = true;
            context->complete(std::move(result));
        });
        return context;
    } catch (const std::exception& error) {
        TransferResult result(ErrorInfo(ErrorCode::INVALID_PARAMS, error.what()));
        result.buffer_safe = true;
        return std::make_shared<CompletedTransferAsyncContext>(
            std::move(result), task.descriptors().size(), proof);
    }
}

std::shared_ptr<TransferAsyncContext>
PerRankBlockTransferEngine::executeHostBuffers(TransferTask                       task,
                                               std::vector<HostBufferView>        hosts,
                                               std::shared_ptr<void>              owner,
                                               TransferAsyncContext::DoneCallback terminal) {
    const auto& descriptors = task.descriptors();
    const auto  invalid     = [&] {
        auto context = invalidTransferContext(task);
        if (terminal)
            terminal(context->result());
        return context;
    };
    if (!owner || descriptors.empty() || hosts.size() != descriptors.size()) {
        return invalid();
    }
    const auto& first   = descriptors.front();
    const Tier  source  = first.source_tier;
    const Tier  target  = first.target_tier;
    const bool  to_dev  = source == Tier::HOST && target == Tier::DEVICE;
    const bool  from_dev = source == Tier::DEVICE && target == Tier::HOST;
    const bool  to_disk = source == Tier::HOST && target == Tier::DISK;
    if (!to_dev && !from_dev && !to_disk) {
        return invalid();
    }
    if (to_disk && !enable_disk_cache_) {
        return invalid();
    }
    const size_t group_id = first.group_set_id;
    if (group_id >= group_sets_.size() || !group_sets_[group_id]) {
        return invalid();
    }
    const auto& group = *group_sets_[group_id];
    if ((to_dev || from_dev) && !group.crcEnabled() && !isPlainDeviceTransferGroup(group)) {
        return invalid();
    }
    if (to_disk && !group.diskPool()) {
        return invalid();
    }
    const size_t required_access = to_disk ? group.diskPool()->strideBytes() : group.storageBytes();
    for (size_t i = 0; i < descriptors.size(); ++i) {
        const auto& desc = descriptors[i];
        if (desc.group_set_id != group_id || desc.source_tier != source || desc.target_tier != target) {
            return invalid();
        }
        if (!isValidHostBufferView(hosts[i], group.payloadBytes(), required_access)) {
            return invalid();
        }
        if (to_dev || from_dev) {
            const auto& blocks = desc.blocksAt(Tier::DEVICE);
            if (blocks.size() != group.devicePools().size()) {
                return invalid();
            }
            for (size_t member = 0; member < blocks.size(); ++member) {
                if (!group.devicePools()[member]->validBlock(blocks[member])) {
                    return invalid();
                }
            }
        } else {
            if (desc.target_blocks.size() != 1 || !group.diskPool()->validBlock(desc.target_blocks[0])) {
                return invalid();
            }
        }
    }
    auto&      loops  = to_dev ? load_loops_ : evict_loops_;
    auto&      next   = to_dev ? next_load_ : next_evict_;
    const auto worker = loops.size() == 1 ? 0 : next.fetch_add(1, std::memory_order_relaxed) % loops.size();
    TransferWorkLoop::BeforeCompleteCallback before_complete;
    if (terminal) {
        before_complete = [terminal = std::move(terminal)](const TransferResult& result) mutable { terminal(result); };
    }
    std::shared_ptr<TransferAsyncContext> context;
    for (size_t offset = 0; offset < loops.size(); ++offset) {
        bool retryable = false;
        context        = loops[(worker + offset) % loops.size()]->executeHostBuffers(
            std::move(task),
            std::move(hosts),
            std::move(owner),
            std::move(before_complete),
            offset + 1 < loops.size() ? &retryable : nullptr);
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
