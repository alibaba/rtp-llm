#include "rtp_llm/cpp/cache/block_tree_cache/transfer/PerRankBlockTransferEngine.h"

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheLayout.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DeviceBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DiskBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/HostBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/HostStagingBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/metrics/BlockTreeCacheMetricsReporter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/CrcCopyExecutor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceHostCopyGeometry.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {

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
    // Staged Disk-to-Host restore is executed by the portable transport.
    if (!descriptor.isExecutable() || (descriptor.source_tier == Tier::DISK && descriptor.target_tier == Tier::HOST))
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
    validateLimits(transfer_queue_max_size);
    RTP_LLM_CHECK(btc_crc_slot_count > 0 && btc_crc_slot_count <= std::numeric_limits<size_t>::max() / 2);
    RTP_LLM_LOG_INFO("transfer_lifecycle event=worker_init rank=%ld engine=%p group_set=%p",
                     world_rank,
                     static_cast<const void*>(this),
                     group_sets_.empty() ? nullptr : static_cast<const void*>(group_sets_.front().get()));
    RTP_LLM_CHECK(transfer_worker_count <= std::numeric_limits<size_t>::max() / (2 * btc_crc_slot_count));
    validate_batch_ = [this](const TransferTask& task) {
        const auto descriptors = task.descriptors();
        const auto& first = descriptors.front();
        const auto group = first.group_set_id;
        if (group >= group_sets_.size() || !group_sets_[group])
            return false;
        const auto& owner = *group_sets_[group];
        for (const auto& descriptor : descriptors) {
            if (!validDescriptor(descriptor, owner))
                return false;
        }
        const auto source = first.source_tier;
        const auto target = first.target_tier;
        if ((source == Tier::DISK || target == Tier::DISK) && !enable_disk_cache_)
            return false;
        return (source != Tier::DEVICE && target != Tier::DEVICE) ||
               owner.crcEnabled() || isPlainDeviceTransferGroup(owner);
    };

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
        disk_io_executor_ = std::make_shared<DiskIOExecutor>(disk_io_config);
        stop_io_ = [io = disk_io_executor_] { io->stop(); };
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
    initializeLoops(transfer_queue_max_size, [&](TransferOperation operation, size_t worker, size_t capacity) {
        std::array<std::shared_ptr<HostStagingBlockPool>, 2> staging;
        const bool load = operation == TransferOperation::LOAD;
        if (load) {
            const size_t staging_blocks = device_disk_staging_block_count / transfer_worker_count
                                          + (worker < device_disk_staging_block_count % transfer_worker_count ? 1 : 0);
            staging = makeLoadStaging(
                group_sets_, staging_blocks, max_descriptors_per_batch_, enable_disk_cache, world_rank, worker);
        }
        return std::make_unique<TransferWorkLoop>(group_sets_,
                                                  make_crc(load ? CrcCopyDirection::LOAD : CrcCopyDirection::EVICT),
                                                  disk_io_executor_.get(),
                                                  operation,
                                                  max_descriptors_per_batch_,
                                                  capacity,
                                                  std::move(staging[0]),
                                                  metrics_reporter,
                                                  world_rank,
                                                  worker,
                                                  std::move(staging[1]));
    });
    // Warm both disk pools before owners or business producers can submit.
    if (disk_io_executor_)
        disk_io_executor_->start();
    startLoops();
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

}  // namespace rtp_llm
