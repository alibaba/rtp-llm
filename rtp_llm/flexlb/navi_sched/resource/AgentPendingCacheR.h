#pragma once

#include "aios/navi_sched/data/AgentPendingCacheToMasterData.h"
#include "aios/navi_sched/resource/AgentCacheR.h"
#include "aios/navi_sched/resource/PendingCacheUtil.h"
#include "autil/Lock.h"
#include "navi/engine/Resource.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>
namespace navi_sched {

class AgentPendingCacheR : public navi::Resource {
public:
    AgentPendingCacheR &forKeyType(NaviSchedCacheKeyType cacheKeyType) const;
    bool hasKeyType(NaviSchedCacheKeyType cacheKeyType) const;

    struct State {
        uint64_t revision = 0;
        size_t batchCount = 0;
        size_t uniqueBlockCount = 0;
        size_t uniqueTailReadyBlockCount = 0;
        uint64_t checksum = 0;
        size_t completedBatchCount = 0;
        int64_t oldestCompletedAgeMs = 0;
    };

    void def(navi::ResourceDefBuilder &builder) const override;
    navi::ErrorCode init(navi::ResourceInitContext &ctx) override;

    bool reserve(const NaviSchedPendingCacheReservation &reservation,
                 AgentPendingCacheToMasterData &update, bool &changed);
    bool replace(const NaviSchedPendingCacheReservation &reservation,
                 AgentPendingCacheToMasterData &update, bool &changed);
    bool retain(const NaviSchedPendingCacheReservation &reservation,
                AgentPendingCacheToMasterData &update, bool &changed);
    bool retire(int64_t batchId, AgentPendingCacheToMasterData &update, bool &changed);
    bool removePublications(int64_t batchId, const NaviSchedBlockHashes &failedFull,
                            const NaviSchedBlockHashes &failedTail,
                            AgentPendingCacheToMasterData &update, bool &changed,
                            bool &retired);
    bool complete(int64_t batchId, bool hybridTailRequired,
                  const NaviSchedBlockHashes &missingFullReadyBlockHashes,
                  const NaviSchedBlockHashes &missingTailReadyBlockHashes,
                  bool &readyToRetire);
    std::vector<int64_t> resolveCompleted(const NaviSchedCacheUpdate &cacheUpdate);
    bool getReservation(int64_t batchId,
                        NaviSchedPendingCacheReservation &reservation) const;
    int64_t getMatchedBlockCount(
        const NaviSchedBlockHashes &blockHashes,
        const std::unordered_set<uint64_t> &cachedBlockHashes,
        const std::unordered_set<uint64_t> &fullReadyBlockHashes,
        const std::unordered_set<uint64_t> &tailReadyBlockHashes, bool hybridTailRequired,
        size_t matchableBlockCount = std::numeric_limits<size_t>::max(),
        int64_t fullAttentionQueryDropBlockCount = 0, size_t alignmentBlocks = 1,
        int64_t *effectiveFullMatchedBlockCount = nullptr) const;
    AgentPendingCacheToMasterData makeSnapshot() const;
    State getState() const;
    void clear();
    navi::SummaryErrorCode getSummary(const std::string &query,
                                      navi::SummaryBuilder &builder) const override;

    static const std::string RESOURCE_ID;

private:
    struct CompletionState {
        int64_t completedTimeUs = 0;
        bool hybridTailRequired = false;
        std::unordered_set<uint64_t> unresolvedFullReadyBlockHashes;
        std::unordered_set<uint64_t> unresolvedTailReadyBlockHashes;
    };

    void fillStateLocked(AgentPendingCacheToMasterData &update) const;

private:
    NaviSchedCacheKeyType _cacheKeyType = NaviSchedCacheKeyType::BLOCK_POOL;
    mutable autil::ThreadMutex _typedCacheLock;
    mutable std::unique_ptr<AgentPendingCacheR> _v6dCache;
    mutable autil::ReadWriteLock _lock;
    PendingCacheState _state;
    std::unordered_map<int64_t, CompletionState> _completed;
    uint64_t _revision = 0;
};

NAVI_TYPEDEF_PTR(AgentPendingCacheR);

}
