#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "aios/navi_sched/common/NaviSchedCommon.h"
#include "aios/navi_sched/common/NaviSchedScheduleData.h"
#include "aios/navi_sched/data/AgentPendingCacheToMasterData.h"
#include "aios/navi_sched/resource/PendingCacheUtil.h"
#include "autil/Lock.h"
#include "navi/engine/Resource.h"

namespace navi_sched {

class MasterPendingCacheR : public navi::Resource {
public:
    using PhysicalBlockSet = absl::flat_hash_set<uint64_t>;

    struct Stats {
        size_t batchCount = 0;
        size_t uniqueBlockCount = 0;
        size_t uniqueTailReadyBlockCount = 0;
        size_t scheduledBatchCount = 0;
        size_t agentBatchCount = 0;
        uint64_t agentRevision = 0;
        uint64_t agentChecksum = 0;
        uint64_t agentRevisionGapCount = 0;
        uint64_t agentStateMismatchCount = 0;
        bool agentSnapshotReady = false;
    };
    struct ApplyResult {
        bool applied = false;
        bool revisionGap = false;
        bool stateMismatch = false;
        bool fullSnapshot = false;
        bool snapshotReadyBefore = false;
        bool candidateStateComputed = false;
        bool staleSnapshotIgnored = false;
        std::string reason;
        uint64_t masterRevisionBefore = 0;
        uint64_t expectedRevision = 0;
        uint64_t receivedRevision = 0;
        uint64_t masterChecksumBefore = 0;
        uint64_t candidateChecksum = 0;
        uint64_t receivedChecksum = 0;
        int64_t masterBatchCountBefore = 0;
        int64_t candidateBatchCount = 0;
        int64_t receivedBatchCount = 0;
        int64_t masterBlockCountBefore = 0;
        int64_t candidateBlockCount = 0;
        int64_t receivedBlockCount = 0;
        int64_t masterTailBlockCountBefore = 0;
        int64_t candidateTailBlockCount = 0;
        int64_t receivedTailBlockCount = 0;
        Stats stats;
    };
    struct MatchBreakdown {
        int64_t fullMatchedBlockCount = 0;
        // Effective Full group hit after query policy/alignment, before Tail
        // reconciliation.
        int64_t effectiveFullMatchedBlockCount = 0;
        int64_t staticMatchedBlockCount = 0;
        int64_t matchedBlockCount = 0;
        int64_t firstPendingBlockSourceBatchId = 0;
        int64_t pendingTailSourceBatchId = 0;
    };
    struct BatchMatchTiming {
        int64_t nodeLookupUs = 0;
        int64_t lockWaitUs = 0;
        int64_t matchUs = 0;
        size_t prefixComparisonCount = 0;
        size_t reusedFullPrefixBlockCount = 0;
        size_t fullPrefixProbeCount = 0;
        size_t tailProbeCount = 0;
        size_t scheduledBatches = 0;
        size_t agentBatches = 0;
        size_t scheduledBlocks = 0;
        size_t agentBlocks = 0;
    };
    struct ReplaySnapshot {
        bool present = false;
        std::vector<NaviSchedPendingCacheReservation> scheduled;
        std::vector<NaviSchedPendingCacheReservation> agent;
        PendingCacheStateStats scheduledStats;
        PendingCacheStateStats agentStats;
        uint64_t agentRevision = 0;
        bool agentSnapshotReady = false;

        autil::legacy::json::JsonMap toJson() const;
    };

    using ReferencedRequest = PendingCacheReferencedRequest;

    void def(navi::ResourceDefBuilder &builder) const override;
    navi::ErrorCode init(navi::ResourceInitContext &ctx) override;

    bool reserve(uint64_t agentIdHash, int64_t batchId, int64_t scheduleTimeUs,
                 const std::vector<ReferencedRequest> &requests,
                 const NaviSchedCachePolicy &policy,
                 NaviSchedCacheKeyType cacheKeyType = NaviSchedCacheKeyType::BLOCK_POOL);
    bool updateScheduled(
        uint64_t agentIdHash, const NaviSchedPendingCacheReservation &reservation,
        NaviSchedCacheKeyType cacheKeyType = NaviSchedCacheKeyType::BLOCK_POOL);
    bool retire(uint64_t agentIdHash, int64_t batchId,
                NaviSchedCacheKeyType cacheKeyType = NaviSchedCacheKeyType::BLOCK_POOL);
    bool removeScheduledBlocks(uint64_t agentIdHash, int64_t batchId,
                               const std::vector<uint64_t> &fullReadyBlockHashes,
                               const std::vector<uint64_t> &tailReadyBlockHashes,
                               NaviSchedCacheKeyType cacheKeyType = NaviSchedCacheKeyType::BLOCK_POOL);
    ApplyResult applyAgentUpdate(uint64_t agentIdHash,
                                 const AgentPendingCacheToMasterData &update);
    void clear(uint64_t agentIdHash);
    void remove(uint64_t agentIdHash);
    void clearAll();

    int64_t getMatchedBlockCount(
        uint64_t agentIdHash, const NaviSchedBlockHashes &blockHashes,
        const PhysicalBlockSet &cachedBlockHashes,
        const PhysicalBlockSet &fullReadyBlockHashes,
        const PhysicalBlockSet &tailReadyBlockHashes, bool hybridTailRequired,
        size_t matchableBlockCount = std::numeric_limits<size_t>::max(),
        int64_t fullAttentionQueryDropBlockCount = 0,
        NaviSchedCacheKeyType cacheKeyType = NaviSchedCacheKeyType::BLOCK_POOL) const;
    int64_t getMatchedBlockCountExcludingSchedule(
        uint64_t agentIdHash, int64_t excludedScheduleBatchId,
        const NaviSchedBlockHashes &blockHashes,
        const PhysicalBlockSet &cachedBlockHashes,
        const PhysicalBlockSet &fullReadyBlockHashes,
        const PhysicalBlockSet &tailReadyBlockHashes, bool hybridTailRequired,
        size_t matchableBlockCount = std::numeric_limits<size_t>::max(),
        int64_t fullAttentionQueryDropBlockCount = 0,
        NaviSchedCacheKeyType cacheKeyType = NaviSchedCacheKeyType::BLOCK_POOL,
        size_t startBlock = 0, size_t alignmentBlocks = 1) const;
    MatchBreakdown getMatchBreakdown(
        uint64_t agentIdHash, const NaviSchedBlockHashes &blockHashes,
        const PhysicalBlockSet &cachedBlockHashes,
        const PhysicalBlockSet &fullReadyBlockHashes,
        const PhysicalBlockSet &tailReadyBlockHashes, bool hybridTailRequired,
        size_t matchableBlockCount = std::numeric_limits<size_t>::max(),
        int64_t fullAttentionQueryDropBlockCount = 0,
        NaviSchedCacheKeyType cacheKeyType = NaviSchedCacheKeyType::BLOCK_POOL,
        size_t startBlock = 0, size_t alignmentBlocks = 1,
        int64_t excludedScheduleBatchId = 0) const;
    uint64_t getMatchedBlockCounts(
        uint64_t agentIdHash, const std::vector<NaviSchedScheduleItem> &requests,
        int64_t effectiveBlockSize, const PhysicalBlockSet &cachedBlockHashes,
        const PhysicalBlockSet &fullReadyBlockHashes,
        const PhysicalBlockSet &tailReadyBlockHashes, bool hybridTailRequired,
        int64_t *matchedBlockCounts, int64_t fullAttentionQueryDropBlockCount = 0,
        NaviSchedCacheKeyType cacheKeyType = NaviSchedCacheKeyType::BLOCK_POOL,
        int64_t *fullMatchedBlockCounts = nullptr,
        int64_t *effectiveFullMatchedBlockCounts = nullptr,
        const int64_t *startBlocks = nullptr,
        const size_t *matchableBlockCounts = nullptr, size_t alignmentBlocks = 1,
        bool includePending = true, BatchMatchTiming *timing = nullptr,
        const std::vector<size_t> *commonPrefixes = nullptr) const;
    Stats getStats(uint64_t agentIdHash, NaviSchedCacheKeyType cacheKeyType =
                                             NaviSchedCacheKeyType::BLOCK_POOL) const;
    uint64_t getMatchedBlockCounts(
        uint64_t agentIdHash, const std::vector<NaviSchedScheduleItem> &requests,
        int64_t effectiveBlockSize, const PhysicalBlockSet &cachedBlockHashes,
        const PhysicalBlockSet &fullReadyBlockHashes,
        const PhysicalBlockSet &tailReadyBlockHashes, bool hybridTailRequired,
        int64_t *matchedBlockCounts, BatchMatchTiming *timing,
        const std::vector<size_t> *commonPrefixes = nullptr) const;
    ReplaySnapshot captureReplaySnapshot(uint64_t agentIdHash) const;
    autil::legacy::json::JsonMap makeReplayState(uint64_t agentIdHash) const;
    bool restoreReplayState(uint64_t agentIdHash,
                            const autil::legacy::json::JsonMap &state);

public:
    static const std::string RESOURCE_ID;

private:
    struct NodeCache {
        mutable autil::ReadWriteLock lock;
        PendingCacheState scheduled;
        PendingCacheState agent;
        PendingCacheStateStats overlap;
        uint64_t agentRevision = 0;
        uint64_t agentRevisionGapCount = 0;
        uint64_t agentStateMismatchCount = 0;
        bool agentSnapshotReady = false;
        mutable std::shared_ptr<const Stats> cachedStats;
    };
    using NodeCachePtr = std::shared_ptr<NodeCache>;

    NodeCachePtr getNodeCache(uint64_t agentIdHash,
                              NaviSchedCacheKeyType cacheKeyType = NaviSchedCacheKeyType::BLOCK_POOL) const;
    NodeCachePtr getOrCreateNodeCache(uint64_t agentIdHash,
                                      NaviSchedCacheKeyType cacheKeyType);
    static Stats getStatsLocked(const NodeCache &cache);
    static bool validateAgentStateLocked(const NodeCache &cache,
                                         const AgentPendingCacheToMasterData &update);
    static int64_t getMatchedBlockCountLocked(
        const NodeCache *cache, const NaviSchedBlockHashes &blockHashes,
        const PhysicalBlockSet &cachedBlockHashes,
        const PhysicalBlockSet &fullReadyBlockHashes,
        const PhysicalBlockSet &tailReadyBlockHashes, bool hybridTailRequired,
        size_t matchableBlockCount, int64_t excludedScheduleBatchId = 0,
        int64_t fullAttentionQueryDropBlockCount = 0, size_t startBlock = 0,
        size_t alignmentBlocks = 1, int64_t *fullMatchedBlockCount = nullptr,
        int64_t *effectiveFullMatchedBlockCount = nullptr,
        size_t initialMatchedBlockCount = 0);
    static int64_t getPendingSourceBatchIdLocked(const NodeCache &cache,
                                                 uint64_t blockHash, bool tailReady);

private:
    mutable autil::ReadWriteLock _nodeLock;
    std::array<std::unordered_map<uint64_t, NodeCachePtr>, 2> _nodeCaches;
};

NAVI_TYPEDEF_PTR(MasterPendingCacheR);

} // namespace navi_sched
