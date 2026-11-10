#include "rtp_llm/cpp/cache/block_tree_cache/transfer/PerRankBlockTransferEngine.h"

#include <memory>
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/legacy/LegacyTransferEngine.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferEngineSelection.h"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <array>
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/CrcTransferService.h"
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
#include "rtp_llm/cpp/utils/RemoteCacheConfig.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferBatchAsyncContext.h"

namespace rtp_llm {

namespace {

std::shared_ptr<TransferAsyncContext> invalidTransferContext(size_t block_count) {
    TransferResult result(ErrorInfo(ErrorCode::INVALID_PARAMS, "invalid block transfer request"));
    result.buffer_safe = true;
    return std::make_shared<CompletedTransferAsyncContext>(std::move(result), block_count);
}

// Metadata follows the same physical allocations as the payload. HOST/DISK
// records pack multiple member tags; keep those regions independent. Existing
// task references, not this callback, protect the source/destination buffers.
struct RegionOriginTransfer {
    std::shared_ptr<IBlockPool>   source;
    std::shared_ptr<IBlockPool>   target;
    BlockIdxType                  source_block;
    BlockIdxType                  target_block;
    uint64_t                      source_incarnation;
    uint64_t                      target_incarnation;
    std::string                   tag;
    PhysicalDecodeOrigins::Origin origin;
};

std::shared_ptr<IBlockPool> originPool(const GroupSet& group, Tier tier, size_t member) {
    if (tier == Tier::DEVICE)
        return group.devicePools().at(member);
    if (tier == Tier::HOST)
        return group.hostPool();
    if (tier == Tier::DISK)
        return group.diskPool();
    return nullptr;
}

std::vector<RegionOriginTransfer> prepareOriginTransfers(const std::vector<TransferDescriptor>& descriptors,
                                                         const std::vector<const GroupSet*>&    groups) {
    std::vector<RegionOriginTransfer> transfers;
    bool                              collect = true;
    for (size_t index = 0; index < descriptors.size(); ++index) {
        const auto&  descriptor   = descriptors[index];
        const auto&  group        = *groups[index];
        const size_t source_count = descriptor.source_tier == Tier::DEVICE ? group.groupTags().size() : 1;
        const size_t target_count = descriptor.target_tier == Tier::DEVICE ? group.groupTags().size() : 1;
        if (descriptor.source_blocks.size() != source_count || descriptor.target_blocks.size() != target_count
            || ((descriptor.source_tier == Tier::DEVICE || descriptor.target_tier == Tier::DEVICE)
                && group.devicePools().size() != group.groupTags().size()))
            continue;  // Let the existing executor validate malformed payload geometry.
        for (size_t member = 0; member < group.groupTags().size(); ++member) {
            const auto source = originPool(group, descriptor.source_tier, member);
            const auto target = originPool(group, descriptor.target_tier, member);
            if (!source || !target)
                continue;
            const auto  source_block = descriptor.source_blocks.at(descriptor.source_tier == Tier::DEVICE ? member : 0);
            const auto  target_block = descriptor.target_blocks.at(descriptor.target_tier == Tier::DEVICE ? member : 0);
            const auto& tag          = group.groupTags()[member];
            const auto  source_incarnation = source->decodeOriginIncarnation(source_block);
            const auto  origin             = source->decodeOrigin(source_block, tag);
            if (origin && collect)
                (void)target->enableDecodeOriginTracking();
            const auto target_incarnation = target->decodeOriginIncarnation(target_block);
            // Unknown sources and failed transfers must not inherit a previous
            // destination's source assertion. Never clear a newer allocation.
            target->bindDecodeOrigin(target_block, target_incarnation, tag, nullptr);
            if (origin && collect && source_incarnation != 0 && target_incarnation != 0) {
                try {
                    transfers.push_back({source,
                                         target,
                                         source_block,
                                         target_block,
                                         source_incarnation,
                                         target_incarnation,
                                         tag,
                                         origin});
                } catch (const std::bad_alloc&) {
                    transfers.clear();
                    collect = false;
                }
            }
        }
    }
    return transfers;
}

void completeOriginTransfers(const std::vector<RegionOriginTransfer>& transfers, bool success) {
    if (!success)
        return;
    for (const auto& transfer : transfers) {
        if (transfer.source->decodeOriginIncarnation(transfer.source_block) == transfer.source_incarnation
            && transfer.source->decodeOrigin(transfer.source_block, transfer.tag) == transfer.origin) {
            transfer.target->bindDecodeOrigin(
                transfer.target_block, transfer.target_incarnation, transfer.tag, transfer.origin);
        }
    }
}

// One loop pair owns this many blocks, split equally by direction. Every group
// with a Device/Disk route shares slots sized for the largest physical access.
std::array<std::shared_ptr<HostStagingBlockPool>, 2>
makeAsyncStaging(const std::vector<GroupSetPtr>& groups, size_t block_count, bool enabled) {
    std::array<std::shared_ptr<HostStagingBlockPool>, 2> result;
    if (!enabled)
        return result;
    size_t stride = 0;
    for (const auto& group : groups) {
        if (!group->diskPool() || group->devicePools().empty())
            continue;
        const auto required = std::max(group->storageBytes(), group->diskPool()->strideBytes());
        if (required > std::numeric_limits<size_t>::max() - 4095)
            throw std::overflow_error("async disk staging stride overflow");
        stride = std::max(stride, ((required + 4095) / 4096) * 4096);
    }
    if (!stride)
        return result;
    if (block_count < 2 || block_count % 2)
        throw std::invalid_argument("invalid async staging budget");
    const auto count = block_count / 2;
    if (block_count > std::numeric_limits<size_t>::max() / stride)
        throw std::overflow_error("async disk staging size overflow");
    result[0] = std::make_shared<HostStagingBlockPool>(count, stride);
    result[1] = std::make_shared<HostStagingBlockPool>(count, stride);
    RTP_LLM_LOG_INFO("async staging: stride=%zu load_blocks=%zu offload_blocks=%zu data_bytes=%zu",
                     stride,
                     count,
                     count,
                     block_count * stride);
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
                                                       DeviceHostCopyOptions    device_host_options,
                                                       size_t                   device_disk_staging_block_count,
                                                       size_t                   max_descriptors_per_batch,
                                                       size_t                   transfer_worker_count,
                                                       size_t                   transfer_queue_max_size,
                                                       std::shared_ptr<BlockTreeCacheMetricsReporter> metrics_reporter,
                                                       int64_t                                        world_rank,
                                                       size_t               btc_crc_slot_count,
                                                       bool                 btc_transfer_engine_async,
                                                       DiskIOExecutorConfig disk_io_config):
    group_sets_(std::move(group_sets)),
    max_descriptors_per_batch_(max_descriptors_per_batch),
    enable_disk_cache_(enable_disk_cache),
    transfer_worker_count_(transfer_worker_count),
    track_decode_origins_(remoteCacheDecodePublicationEnabled()),
    crc_slot_count_(btc_crc_slot_count) {
    RTP_LLM_CHECK(max_descriptors_per_batch > 0);
    RTP_LLM_CHECK(transfer_worker_count > 0 && transfer_worker_count <= std::numeric_limits<size_t>::max() / 2);
    RTP_LLM_CHECK(btc_crc_slot_count > 0 && btc_crc_slot_count <= std::numeric_limits<size_t>::max() / 2);
    const auto fallback_reason = workLoopCompatibility(group_sets_, enable_disk_cache);
    async_                     = btc_transfer_engine_async && fallback_reason.empty();
    RTP_LLM_LOG_INFO("BlockTree transfer: requested_async=%d implementation=%s fallback_reason=%s",
                     btc_transfer_engine_async,
                     async_ ? "workloop" : "legacy",
                     btc_transfer_engine_async && !async_ ? fallback_reason.c_str() : "none");
    std::array<std::shared_ptr<HostStagingBlockPool>, 2> staging;
    if (async_) {
        RTP_LLM_CHECK(transfer_worker_count <= std::numeric_limits<size_t>::max() / (2 * btc_crc_slot_count));
        if (transfer_queue_max_size < transfer_worker_count
            || transfer_queue_max_size > std::numeric_limits<size_t>::max() / max_descriptors_per_batch)
            throw std::invalid_argument("async transfer requires Q >= worker count and nonoverflowing Q*M");
        staging                = makeAsyncStaging(group_sets_, device_disk_staging_block_count, enable_disk_cache);
        const bool has_staging = staging[0] != nullptr;
        max_descriptors_per_batch_ =
            asyncTransferBatchLimit(max_descriptors_per_batch_, device_disk_staging_block_count, has_staging);
    }
    if (enable_disk_cache_ && std::any_of(group_sets_.begin(), group_sets_.end(), [](const auto& group) {
            return group && group->diskPool();
        })) {
        // Disk reads use LOAD and writes use OFFLOAD. Q covers all owners/files;
        // each resident batch has at most one disk stage in flight.
        if (async_
            && !disk_io_config.setQueueSizesFromBatchCapacity(
                transfer_queue_max_size, transfer_queue_max_size, max_descriptors_per_batch_))
            throw std::invalid_argument("async disk I/O queue capacity overflow");
        if (!disk_io_config.valid())
            throw std::invalid_argument("invalid engine-wide disk I/O budgets");
        disk_io_executor_ = std::make_unique<DiskIOExecutor>(disk_io_config);
        if (async_)
            RTP_LLM_LOG_INFO("disk I/O queue capacities: read=%zu write=%zu resident_batches_per_direction=%zu "
                             "batch_limit=%zu read_threads=%zu write_threads=%zu",
                             disk_io_config.read_queue_size,
                             disk_io_config.write_queue_size,
                             transfer_queue_max_size,
                             max_descriptors_per_batch_,
                             disk_io_config.read_thread_count,
                             disk_io_config.write_thread_count);
    }
    if (async_) {
        const auto crc_groups = crcTransferGroups(group_sets_, enable_disk_cache);
        const bool needs_crc  = !crc_groups.empty();
        const auto make_crc   = [&]() -> std::unique_ptr<CrcTransferService> {
            return needs_crc ? std::make_unique<CrcTransferService>(
                                   crc_groups, max_descriptors_per_batch_, crc_slot_count_, world_rank) :
                                 nullptr;
        };
        load_loops_.reserve(transfer_worker_count);
        offload_loops_.reserve(transfer_worker_count);
        for (size_t worker = 0; worker < transfer_worker_count; ++worker) {
            const auto queue_capacity = transfer_queue_max_size / transfer_worker_count
                                        + (worker < transfer_queue_max_size % transfer_worker_count ? 1 : 0);
            if (worker != 0)
                staging = makeAsyncStaging(group_sets_, device_disk_staging_block_count, enable_disk_cache);
            load_loops_.push_back(std::make_unique<TransferWorkLoop>(group_sets_,
                                                                     make_crc(),
                                                                     disk_io_executor_.get(),
                                                                     TransferDirection::LOAD,
                                                                     max_descriptors_per_batch_,
                                                                     queue_capacity,
                                                                     std::move(staging[0]),
                                                                     metrics_reporter,
                                                                     world_rank,
                                                                     worker));
            offload_loops_.push_back(std::make_unique<TransferWorkLoop>(group_sets_,
                                                                        make_crc(),
                                                                        disk_io_executor_.get(),
                                                                        TransferDirection::OFFLOAD,
                                                                        max_descriptors_per_batch_,
                                                                        queue_capacity,
                                                                        std::move(staging[1]),
                                                                        metrics_reporter,
                                                                        world_rank,
                                                                        worker));
        }
        // Warm both disk pools before owners or business producers can submit.
        if (disk_io_executor_)
            disk_io_executor_->start();
        for (auto& loop : load_loops_)
            loop->start();
        for (auto& loop : offload_loops_)
            loop->start();
        RTP_LLM_LOG_INFO(
            "BlockTree async transfer: owners_per_direction=%zu C=%zu total_crc_slots_per_device=%zu M=%zu "
            "resident_batch_budget_per_direction=%zu staging_blocks_per_loop_pair=%zu",
            transfer_worker_count,
            btc_crc_slot_count,
            needs_crc ? crcTransferWorkspaceCount(transfer_worker_count, btc_crc_slot_count, true) : 0,
            max_descriptors_per_batch_,
            transfer_queue_max_size,
            device_disk_staging_block_count);
        return;
    }
    legacy_ = std::make_unique<LegacyTransferEngine>(group_sets_,
                                                     enable_disk_cache,
                                                     std::move(device_host_options),
                                                     device_disk_staging_block_count,
                                                     max_descriptors_per_batch,
                                                     transfer_worker_count,
                                                     transfer_queue_max_size,
                                                     std::move(metrics_reporter),
                                                     world_rank);
}

PerRankBlockTransferEngine::~PerRankBlockTransferEngine() {
    shutdown();
}

void PerRankBlockTransferEngine::stopAdmission() {
    if (async_) {
        for (auto& loop : load_loops_)
            loop->stopAdmission();
        for (auto& loop : offload_loops_)
            loop->stopAdmission();
    } else {
        legacy_->stopAdmission();
    }
}

void PerRankBlockTransferEngine::shutdown() {
    stopAdmission();
    if (async_) {
        for (auto& loop : load_loops_)
            loop->shutdown();
        for (auto& loop : offload_loops_)
            loop->shutdown();
    } else {
        cancelPendingStagingTransfers();
        legacy_->drain();
    }
    // Accepted CRC stages may still submit their disk successors while draining.
    if (disk_io_executor_)
        disk_io_executor_->stop();
}

void PerRankBlockTransferEngine::cancelPendingStagingTransfers() {
    if (async_) {
        for (auto& loop : load_loops_)
            loop->cancelPendingStagingTransfers();
        for (auto& loop : offload_loops_)
            loop->cancelPendingStagingTransfers();
    } else {
        legacy_->cancelPendingStagingTransfers();
    }
}

BlockTreeQueueSizes PerRankBlockTransferEngine::queueSizes() const {
    if (async_) {
        BlockTreeQueueSizes sizes;
        for (const auto& loop : load_loops_)
            sizes.load += loop->outstandingBatches();
        for (const auto& loop : offload_loops_)
            // Preserve the historical monitoring field/tag for the offload lane.
            sizes.background += loop->outstandingBatches();
        return sizes;
    }
    return legacy_->queueSizes();
}

std::shared_ptr<TransferAsyncContext> PerRankBlockTransferEngine::execute(TransferTask task) {
    const auto& descriptors = task.descriptors();
    if (descriptors.empty()) {
        return invalidTransferContext(descriptors.size());
    }
    const Tier source = descriptors.front().source_tier;
    const Tier target = descriptors.front().target_tier;
    if (async_ && descriptors.size() > batchLimit())
        return invalidTransferContext(descriptors.size());
    std::vector<const GroupSet*> group_sets;
    group_sets.reserve(descriptors.size());
    for (const auto& descriptor : descriptors) {
        if (descriptor.source_tier != source || descriptor.target_tier != target
            || (async_ && descriptor.group_set_id != descriptors.front().group_set_id)) {
            return invalidTransferContext(descriptors.size());
        }
        if (descriptor.group_set_id >= group_sets_.size() || !group_sets_[descriptor.group_set_id])
            return invalidTransferContext(descriptors.size());
        const auto* group_set = group_sets_[descriptor.group_set_id].get();
        if (!validDescriptor(descriptor, *group_set))
            return invalidTransferContext(descriptors.size());
        group_sets.push_back(group_set);
    }

    if ((source == Tier::DEVICE || target == Tier::DEVICE)
        && std::any_of(group_sets.begin(), group_sets.end(), [&](const auto* group) {
               return group->crcEnabled() != group_sets.front()->crcEnabled();
           }))
        return invalidTransferContext(descriptors.size());
    if ((source == Tier::DISK || target == Tier::DISK) && !enable_disk_cache_)
        return invalidTransferContext(descriptors.size());
    if (((source == Tier::DISK && target == Tier::DEVICE) || (source == Tier::DEVICE && target == Tier::DISK))
        && std::any_of(group_sets.begin(), group_sets.end(), [&](const auto* group) {
               return group->groupType() != group_sets.front()->groupType();
           }))
        return invalidTransferContext(descriptors.size());
    if (async_) {
        TransferWorkLoop::BeforeCompleteCallback before_complete;
        if (track_decode_origins_) {
            auto origins    = prepareOriginTransfers(descriptors, group_sets);
            before_complete = [origins = std::move(origins)](const TransferResult& result) {
                completeOriginTransfers(origins, result.ok());
            };
        }
        auto&      loops  = isLoadTransfer(source, target) ? load_loops_ : offload_loops_;
        auto&      next   = isLoadTransfer(source, target) ? next_load_ : next_offload_;
        const auto worker = loops.size() == 1 ? 0 : next.fetch_add(1, std::memory_order_relaxed) % loops.size();
        return loops[worker]->execute(std::move(task), std::move(before_complete));
    }
    const auto submit = [&] { return legacy_->execute(std::move(task), group_sets); };
    if (!track_decode_origins_)
        return submit();
    auto origins = prepareOriginTransfers(descriptors, group_sets);
    if (origins.empty())
        return submit();
    std::shared_ptr<TransferBatchAsyncContext> observed;
    try {
        observed = std::make_shared<TransferBatchAsyncContext>(descriptors.size());
    } catch (const std::bad_alloc&) {
        // Attribution is optional; the already-cleared destination remains
        // unknown and ordinary transfer semantics stay unchanged.
        return submit();
    }
    auto context = submit();
    if (!context)
        return context;
    context->onDone([observed, origins = std::move(origins)](TransferResult result) {
        try {
            completeOriginTransfers(origins, result.ok());
        } catch (const std::exception& error) {
            RTP_LLM_LOG_WARNING("local transfer source attribution unavailable: %s", error.what());
        } catch (...) {
            RTP_LLM_LOG_WARNING("local transfer source attribution unavailable");
        }
        // Do not expose completion to waitDone/done/result/onDone callers
        // before the metadata settlement above. No new GPU sync is needed.
        observed->complete(std::move(result));
    });
    return observed;
}

void PerRankBlockTransferEngine::waitForIdle() {
    if (!async_) {
        legacy_->waitForIdle();
        return;
    }
    BlockTreeQueueSizes pending;
    do {
        for (auto& loop : load_loops_)
            loop->waitForIdle();
        for (auto& loop : offload_loops_)
            loop->waitForIdle();
        // A later owner's callback may have submitted to an earlier owner.
        pending = queueSizes();
    } while (pending.load != 0 || pending.background != 0);
}

}  // namespace rtp_llm
