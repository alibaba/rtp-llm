#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheConfig.h"

namespace rtp_llm {

class BlockTreeCacheMetricsReporter;
class GroupSet;
class IBlockPool;

struct EvictionBatchResult {
    size_t required{0};
    size_t freed{0};
    bool   dropped{false};
    bool   needs_next_batch{false};
};

class EvictionStrategy {
public:
    explicit EvictionStrategy(const BlockTreeCacheMetricsReporter& metrics): metrics_(metrics) {}
    virtual ~EvictionStrategy()                                                        = default;
    virtual size_t evictionBatchSize(const GroupSet& group_set, size_t required) const = 0;

protected:
    const BlockTreeCacheMetricsReporter& metrics_;
};

class OnDemandEvictionStrategy final: public EvictionStrategy {
public:
    using EvictionStrategy::EvictionStrategy;

    size_t evictionBatchSize(const GroupSet&, size_t required) const override {
        return required;
    }
    void finishBatchDrop(Tier tier, const GroupSet& group_set, const EvictionBatchResult& result) const;
};

class WatermarkEvictionStrategy final: public EvictionStrategy {
public:
    static std::unique_ptr<WatermarkEvictionStrategy> create(Tier                                 tier,
                                                             const BlockTreeCacheConfig&          config,
                                                             size_t                               group_set_count,
                                                             const BlockTreeCacheMetricsReporter& metrics);
    WatermarkEvictionStrategy(Tier                                 source,
                              Tier                                 target,
                              EvictionBatchLimits                  limits,
                              size_t                               group_set_count,
                              const BlockTreeCacheMetricsReporter& metrics);
    Tier targetTier() const {
        return target_;
    }
    size_t nextGroupSetId() const {
        return next_evict_group_set_id_;
    }

    size_t evictionBatchSize(const GroupSet& group_set, size_t required) const override;

    size_t computeWatermarkEvictCount(const GroupSet&                                group_set,
                                      const std::unordered_map<IBlockPool*, size_t>& pending_release_counts);
    bool   finishBatchDrop(const GroupSet& group_set, const EvictionBatchResult& result);
    bool   startWatermark(size_t group_set_id, std::optional<TierWatermark> watermark);
    void   stopWatermark(size_t group_set_id);
    size_t transferBatchSize(const GroupSet& group_set, size_t required, bool& waiting_for_budget);
    size_t transferBatchSizeAfterDrop(const GroupSet&            group_set,
                                      size_t                     batch_limit,
                                      const EvictionBatchResult& drop_result,
                                      bool                       waiting_for_budget);
    void   finishTransferBatchScheduling(const GroupSet& group_set, size_t scheduled_count);
    void   finishTransferRound(const GroupSet& group_set);

    void updatePending(const GroupSet& group_set, bool reserve);

private:
    struct WatermarkEvictionRound {
        std::optional<TierWatermark> watermark;
        size_t                       required_count{0};
        size_t                       scheduled_count{0};
        size_t                       pending_count{0};
        bool                         freed_blocks_while_waiting{false};
    };

    size_t batchSizeWithoutPending(size_t block_bytes) const;
    void   finishWatermarkRound(const GroupSet& group_set);

    const Tier                          source_;
    const Tier                          target_;
    EvictionBatchLimits                 limits_;
    std::vector<WatermarkEvictionRound> watermark_rounds_;

    size_t pending_bytes_{0};

    size_t            next_evict_group_set_id_{0};
    std::vector<bool> watermark_triggered_;
};

}  // namespace rtp_llm
