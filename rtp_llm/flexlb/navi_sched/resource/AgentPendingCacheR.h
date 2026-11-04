#pragma once

#include "rtp_llm/flexlb/navi_sched/data/AgentPendingCacheToMasterData.h"
#include "rtp_llm/flexlb/navi_sched/resource/PendingCacheUtil.h"
#include "autil/Lock.h"
#include "navi/engine/Resource.h"

#include <cstdint>
#include <limits>
#include <unordered_set>
#include <vector>
namespace navi_sched {

class AgentPendingCacheR : public navi::Resource {
public:
    struct State {
        uint64_t revision = 0;
        size_t batchCount = 0;
        size_t uniqueBlockCount = 0;
        size_t uniqueTailReadyBlockCount = 0;
        uint64_t checksum = 0;
    };

    void def(navi::ResourceDefBuilder &builder) const override;
    navi::ErrorCode init(navi::ResourceInitContext &ctx) override;

    bool reserve(const NaviSchedPendingCacheReservation &reservation,
                 AgentPendingCacheToMasterData &update, bool &changed);
    bool replace(const NaviSchedPendingCacheReservation &reservation,
                 AgentPendingCacheToMasterData &update, bool &changed);
    // Cancellation may arrive out of order; only retain currently owned blocks.
    bool retain(const NaviSchedPendingCacheReservation &reservation,
                AgentPendingCacheToMasterData &update, bool &changed);
    bool retire(int64_t batchId, AgentPendingCacheToMasterData &update,
                bool &changed);
    bool getReservation(int64_t batchId,
                        NaviSchedPendingCacheReservation &reservation) const;
    int64_t getMatchedBlockCount(
        const NaviSchedBlockHashes &blockHashes,
        const std::unordered_set<uint64_t> &cachedBlockHashes,
        const std::unordered_set<uint64_t> &fullReadyBlockHashes,
        const std::unordered_set<uint64_t> &tailReadyBlockHashes,
        bool hybridTailRequired,
        size_t matchableBlockCount = std::numeric_limits<size_t>::max()) const;
    AgentPendingCacheToMasterData makeSnapshot() const;
    State getState() const;
    void clear();
    navi::SummaryErrorCode getSummary(const std::string &query,
                                      navi::SummaryBuilder &builder) const override;

    static const std::string RESOURCE_ID;

private:
    void fillStateLocked(AgentPendingCacheToMasterData &update) const;

private:
    mutable autil::ReadWriteLock _lock;
    PendingCacheState _state;
    uint64_t _revision = 0;
};

NAVI_TYPEDEF_PTR(AgentPendingCacheR);

} // namespace navi_sched
