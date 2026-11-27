#include "rtp_llm/cpp/cache/block_tree_cache/evict/EvictionStrategy.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/metrics/BlockTreeCacheMetricsReporter.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

void OnDemandEvictionStrategy::finishBatchDrop(Tier                       tier,
                                               const GroupSet&            group_set,
                                               const EvictionBatchResult& result) const {
    const bool device = tier == Tier::DEVICE;
    if (device ? result.dropped : result.freed != 0) {
        metrics_.reportEvictionTriggered(tier, group_set.groupType(), true);
    }
    if (device || result.required != 0) {
        metrics_.reportEvictionBlocks(tier,
                                      group_set.groupType(),
                                      true,
                                      result.required,
                                      device ? std::min(result.freed, result.required) : result.freed);
    }
}

WatermarkEvictionStrategy::WatermarkEvictionStrategy(Tier                                 source,
                                                     Tier                                 target,
                                                     EvictionBatchLimits                  limits,
                                                     size_t                               group_set_count,
                                                     const BlockTreeCacheMetricsReporter& metrics):
    EvictionStrategy(metrics),
    source_(source),
    target_(target),
    limits_(limits),
    watermark_rounds_(group_set_count),
    watermark_triggered_(group_set_count) {}

size_t WatermarkEvictionStrategy::batchSizeWithoutPending(size_t block_bytes) const {
    if (limits_.max_pending_bytes == 0 || block_bytes == 0) {
        return limits_.max_descriptors_per_batch;
    }
    const size_t blocks = std::max<size_t>(1, limits_.max_pending_bytes / block_bytes);
    return std::min(limits_.max_descriptors_per_batch, blocks);
}

void WatermarkEvictionStrategy::updatePending(const GroupSet& group_set, bool reserve) {
    auto&        count = watermark_rounds_[group_set.groupSetId()].pending_count;
    const size_t bytes = limits_.max_pending_bytes == 0 ? 0 : group_set.evictionBlockBytes();
    if (reserve) {
        RTP_LLM_CHECK(bytes <= std::numeric_limits<size_t>::max() - pending_bytes_);
        pending_bytes_ += bytes;
        ++count;
    } else {
        RTP_LLM_CHECK(count > 0 && bytes <= pending_bytes_);
        pending_bytes_ -= bytes;
        --count;
    }
}

bool WatermarkEvictionStrategy::startWatermark(size_t group_set_id, std::optional<TierWatermark> watermark) {
    auto& round = watermark_rounds_[group_set_id];
    if (watermark
        && (target_ != Tier::DISK || (!round.watermark && round.required_count == 0 && round.pending_count == 0))) {
        round.watermark = watermark;
    }
    return round.watermark.has_value();
}

void WatermarkEvictionStrategy::stopWatermark(size_t group_set_id) {
    watermark_rounds_[group_set_id].watermark.reset();
}

void WatermarkEvictionStrategy::finishTransferRound(const GroupSet& group_set) {
    if (target_ != Tier::DISK || !watermark_rounds_[group_set.groupSetId()].watermark) {
        finishWatermarkRound(group_set);
    }
}

void WatermarkEvictionStrategy::finishWatermarkRound(const GroupSet& group_set) {
    auto& round = watermark_rounds_[group_set.groupSetId()];
    if (round.pending_count != 0) {
        return;
    }
    if (round.required_count != 0) {
        metrics_.reportEvictionBlocks(
            source_, group_set.groupType(), false, round.required_count, round.scheduled_count);
    }
    round = {};
}

size_t
WatermarkEvictionStrategy::transferBatchSize(const GroupSet& group_set, size_t required, bool& waiting_for_budget) {
    const size_t group_set_id = group_set.groupSetId();
    auto&        round        = watermark_rounds_[group_set_id];
    if (required == 0) {
        round.freed_blocks_while_waiting = false;
        finishWatermarkRound(group_set);
        return 0;
    }
    if (waiting_for_budget) {
        return 0;
    }
    const size_t batch_limit = evictionBatchSize(group_set, required);
    if (batch_limit != 0) {
        round.freed_blocks_while_waiting = false;
        return batch_limit;
    }
    next_evict_group_set_id_ = group_set_id;
    waiting_for_budget       = true;
    return round.freed_blocks_while_waiting ?
               0 :
               std::min(required, batchSizeWithoutPending(group_set.evictionBlockBytes()));
}

size_t WatermarkEvictionStrategy::transferBatchSizeAfterDrop(const GroupSet&            group_set,
                                                             size_t                     batch_limit,
                                                             const EvictionBatchResult& drop_result,
                                                             bool                       waiting_for_budget) {
    auto& round = watermark_rounds_[group_set.groupSetId()];
    if (waiting_for_budget) {
        round.freed_blocks_while_waiting = drop_result.freed != 0;
        return 0;
    }
    if (drop_result.freed < drop_result.required) {
        batch_limit -= drop_result.required - drop_result.freed;
    }
    return batch_limit;
}

void WatermarkEvictionStrategy::finishTransferBatchScheduling(const GroupSet& group_set, size_t scheduled_count) {
    const auto group_set_id = group_set.groupSetId();
    auto&      round        = watermark_rounds_[group_set_id];
    if (scheduled_count != 0) {
        if (target_ != Tier::DISK || round.scheduled_count == 0) {
            metrics_.reportEvictionTriggered(source_, group_set.groupType(), false);
        }
        round.scheduled_count += scheduled_count;
        next_evict_group_set_id_ = (group_set_id + 1) % watermark_rounds_.size();
    } else if (target_ != Tier::DISK || round.pending_count == 0) {
        stopWatermark(group_set_id);
    }
    finishTransferRound(group_set);
}

size_t WatermarkEvictionStrategy::computeWatermarkEvictCount(
    const GroupSet& group_set, const std::unordered_map<IBlockPool*, size_t>& pending_release_counts) {
    const auto  group_set_id   = group_set.groupSetId();
    auto&       round          = watermark_rounds_[group_set_id];
    const auto& watermark      = *round.watermark;
    bool        has_pending    = false;
    bool        high_reached   = false;
    size_t      required_count = 0;
    const auto  inspect        = [&](IBlockPool* pool) {
        const auto   pending       = pending_release_counts.find(pool);
        const size_t pending_count = pending == pending_release_counts.end() ? 0 : pending->second;
        const size_t used          = pool->usedBlocksNum();
        const size_t total         = pool->totalBlocksNum();
        if (pending_count > 0) {
            RTP_LLM_CHECK_WITH_INFO(pending_count <= used,
                                    "pending eviction releases exceed used blocks: "
                                            "pool=%s address=%p pending=%zu used=%zu total=%zu low_ratio=%f high_ratio=%f",
                                    pool->poolName().c_str(),
                                    static_cast<void*>(pool),
                                    pending_count,
                                    used,
                                    total,
                                    watermark.low_ratio,
                                    watermark.high_ratio);
        }
        const size_t effective_used = used - pending_count;
        const size_t high_used_blocks =
            static_cast<size_t>(std::ceil(static_cast<double>(total) * watermark.high_ratio));
        const size_t low_used_blocks =
            static_cast<size_t>(std::floor(static_cast<double>(total) * watermark.low_ratio));
        has_pending  = has_pending || pending_count != 0;
        high_reached = high_reached || effective_used >= high_used_blocks;
        required_count =
            std::max(required_count, effective_used > low_used_blocks ? effective_used - low_used_blocks : 0);
    };
    if (source_ == Tier::DEVICE) {
        for (const auto& pool : group_set.devicePools()) {
            inspect(pool.get());
        }
    } else if (auto* pool = group_set.hostOrDiskPool(source_)) {
        inspect(pool);
    }
    if (required_count == 0 && !has_pending) {
        watermark_triggered_[group_set_id] = false;
    } else if (high_reached) {
        watermark_triggered_[group_set_id] = true;
    }
    const size_t required = watermark_triggered_[group_set_id] ? required_count : 0;
    if (round.required_count == 0) {
        round.required_count = required;
    }
    return required;
}

bool WatermarkEvictionStrategy::finishBatchDrop(const GroupSet& group_set, const EvictionBatchResult& result) {
    auto& round = watermark_rounds_[group_set.groupSetId()];
    if (result.dropped && round.scheduled_count == 0) {
        metrics_.reportEvictionTriggered(source_, group_set.groupType(), false);
    }
    round.scheduled_count += std::min(result.freed, limits_.max_descriptors_per_batch);
    if (result.freed != 0 && result.freed < result.required) {
        return true;
    }
    finishWatermarkRound(group_set);
    return false;
}

size_t WatermarkEvictionStrategy::evictionBatchSize(const GroupSet& group_set, size_t required) const {
    if (target_ == Tier::NONE) {
        return std::min(required, limits_.max_descriptors_per_batch);
    }
    const size_t bytes = group_set.evictionBlockBytes();
    if (pending_bytes_ == 0 || bytes == 0) {
        return std::min(required, batchSizeWithoutPending(bytes));
    }
    const size_t available = limits_.max_pending_bytes - std::min(limits_.max_pending_bytes, pending_bytes_);
    return std::min({required, limits_.max_descriptors_per_batch, available / bytes});
}

std::unique_ptr<WatermarkEvictionStrategy>
WatermarkEvictionStrategy::create(Tier                                 tier,
                                  const BlockTreeCacheConfig&          config,
                                  size_t                               group_set_count,
                                  const BlockTreeCacheMetricsReporter& metrics) {
    const Tier target = config.evictionTarget(tier);
    if (target == Tier::NONE && !config.usesWatermark(tier)) {
        return nullptr;
    }
    const size_t              pending_bytes = tier == Tier::DEVICE ? config.device_evict_max_pending_bytes :
                                              tier == Tier::HOST   ? config.host_evict_max_pending_bytes :
                                                                     0;
    const EvictionBatchLimits limits{pending_bytes, config.max_descriptors_per_transfer_batch};
    return std::make_unique<WatermarkEvictionStrategy>(tier, target, limits, group_set_count, metrics);
}

}  // namespace rtp_llm
