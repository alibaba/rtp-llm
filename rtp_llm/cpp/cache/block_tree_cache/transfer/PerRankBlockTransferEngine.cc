#include "rtp_llm/cpp/cache/block_tree_cache/transfer/PerRankBlockTransferEngine.h"

#include <memory>
#include <algorithm>
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/CrcTransferService.h"
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
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
    return std::make_shared<CompletedTransferAsyncContext>(
        ErrorInfo(ErrorCode::INVALID_PARAMS, "invalid block transfer request"), block_count);
}

// Metadata follows the same physical allocations as the payload. HOST/DISK
// records pack multiple member tags; keep those regions independent. Existing
// task references, not this observer, protect the source/destination buffers.
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

}  // namespace

PerRankBlockTransferEngine::PerRankBlockTransferEngine(std::vector<GroupSetPtr> group_sets,
                                                       bool                     enable_disk_cache,
                                                       DeviceHostCopyOptions    device_host_options,
                                                       size_t                   device_disk_staging_block_count,
                                                       size_t                   max_descriptors_per_batch,
                                                       size_t                   transfer_worker_count,
                                                       size_t                   transfer_queue_max_size,
                                                       std::shared_ptr<BlockTreeCacheMetricsReporter> metrics_reporter,
                                                       int64_t                                        world_rank):
    group_sets_(std::move(group_sets)),
    transfer_worker_count_(transfer_worker_count),
    track_decode_origins_(remoteCacheDecodePublicationEnabled()) {
    RTP_LLM_CHECK(max_descriptors_per_batch > 0);
    RTP_LLM_CHECK(transfer_worker_count > 0);
    std::shared_ptr<CrcTransferService> crc_service;
    if (std::any_of(
            group_sets_.begin(), group_sets_.end(), [](const GroupSetPtr& group) { return group->crcEnabled(); })) {
        crc_service = std::make_shared<CrcTransferService>(
            group_sets_, max_descriptors_per_batch, 2 * transfer_worker_count, world_rank);
    }
    load_transfer_task_pool_ =
        std::make_unique<BlockTreeTaskPool>(transfer_worker_count, transfer_queue_max_size, "btc_load");
    transfer_task_pool_ =
        std::make_unique<BlockTreeTaskPool>(transfer_worker_count, transfer_queue_max_size, "btc_evict_store");
    RTP_LLM_CHECK(load_transfer_task_pool_->start());
    RTP_LLM_CHECK(transfer_task_pool_->start());
    device_host_executor_ = std::make_unique<DeviceHostTransferExecutor>(*transfer_task_pool_,
                                                                         max_descriptors_per_batch,
                                                                         std::move(device_host_options),
                                                                         metrics_reporter,
                                                                         std::move(crc_service),
                                                                         load_transfer_task_pool_.get());
    host_disk_executor_   = std::make_unique<HostDiskTransferExecutor>(
        *transfer_task_pool_, max_descriptors_per_batch, metrics_reporter, load_transfer_task_pool_.get());
    if (enable_disk_cache && !group_sets_.empty()) {
        device_disk_executor_ = std::make_unique<DeviceDiskTransferExecutor>(*device_host_executor_,
                                                                             *host_disk_executor_,
                                                                             group_sets_,
                                                                             device_disk_staging_block_count,
                                                                             *transfer_task_pool_,
                                                                             max_descriptors_per_batch,
                                                                             std::move(metrics_reporter),
                                                                             load_transfer_task_pool_.get());
    }
}

PerRankBlockTransferEngine::~PerRankBlockTransferEngine() {
    load_transfer_task_pool_->stopAdmission();
    transfer_task_pool_->stopAdmission();
    cancelPendingStagingTransfers();
    load_transfer_task_pool_->waitForIdle();
    transfer_task_pool_->waitForIdle();
    load_transfer_task_pool_.reset();
    transfer_task_pool_.reset();
}

void PerRankBlockTransferEngine::cancelPendingStagingTransfers() {
    if (device_disk_executor_ != nullptr) {
        device_disk_executor_->cancelPendingTransfers();
    }
}

BlockTreeQueueSizes PerRankBlockTransferEngine::queueSizes() const {
    auto load_sizes       = load_transfer_task_pool_->queueSizes();
    auto background_sizes = transfer_task_pool_->queueSizes();
    return {load_sizes.load + background_sizes.load,
            load_sizes.background + background_sizes.background,
            load_sizes.completion + background_sizes.completion};
}

std::shared_ptr<TransferAsyncContext> PerRankBlockTransferEngine::execute(TransferTask task) {
    const auto& descriptors = task.descriptors();
    if (descriptors.empty()) {
        return invalidTransferContext(descriptors.size());
    }
    const Tier                   source = descriptors.front().source_tier;
    const Tier                   target = descriptors.front().target_tier;
    std::vector<const GroupSet*> group_sets;
    std::vector<HostBufferView>  hosts;
    group_sets.reserve(descriptors.size());
    hosts.reserve(descriptors.size());
    for (const auto& descriptor : descriptors) {
        if (descriptor.source_tier != source || descriptor.target_tier != target) {
            return invalidTransferContext(descriptors.size());
        }
        const auto* group_set = group_sets_[descriptor.group_set_id].get();
        group_sets.push_back(group_set);
        if (source == Tier::HOST || target == Tier::HOST) {
            hosts.push_back(resolveHostView(*group_set, descriptor.singleBlockAt(Tier::HOST)));
        }
    }

    const auto submit = [&]() -> std::shared_ptr<TransferAsyncContext> {
        if (source == Tier::DISK && target == Tier::DEVICE) {
            if (device_disk_executor_ == nullptr) {
                return invalidTransferContext(descriptors.size());
            }
            return device_disk_executor_->executeDiskToDevice(std::move(task), group_sets);
        }
        if (source == Tier::DEVICE && target == Tier::DISK) {
            if (device_disk_executor_ == nullptr) {
                return invalidTransferContext(descriptors.size());
            }
            return device_disk_executor_->executeDeviceToDisk(std::move(task), *group_sets.front());
        }
        if ((source == Tier::DEVICE && target == Tier::HOST) || (source == Tier::HOST && target == Tier::DEVICE)) {
            return device_host_executor_->execute(std::move(task), std::move(hosts), std::move(group_sets));
        }
        if ((source == Tier::HOST && target == Tier::DISK) || (source == Tier::DISK && target == Tier::HOST)) {
            return host_disk_executor_->execute(std::move(task), std::move(hosts), std::move(group_sets));
        }
        return invalidTransferContext(descriptors.size());
    };
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

HostBufferView PerRankBlockTransferEngine::resolveHostView(const GroupSet& group_set, BlockIdxType host_block) {
    const HostBlockBuffer buffer = group_set.hostPool()->blockBuffer(host_block);
    return HostBufferView{buffer.addr, buffer.payload_bytes, buffer.stride_bytes};
}

}  // namespace rtp_llm
