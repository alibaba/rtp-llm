#pragma once

#include "aios/navi_sched/common/NaviSchedCommon.h"
#include "autil/Lock.h"
#include "navi/engine/Resource.h"
#include <unordered_set>

namespace navi_sched {

class AgentPendingCacheR;

struct NaviSchedCacheUpdate {
    int64_t totalBlocks = -1;
    bool fullSnapshot = false;
    NaviSchedBlockHashesConstPtr addedBlockHashes;
    NaviSchedBlockHashesConstPtr removedBlockHashes;
    NaviSchedBlockHashesConstPtr addedFullReadyBlockHashes;
    NaviSchedBlockHashesConstPtr removedFullReadyBlockHashes;
    NaviSchedBlockHashesConstPtr addedTailReadyBlockHashes;
    NaviSchedBlockHashesConstPtr removedTailReadyBlockHashes;
    int64_t blockSize = -1;
    int64_t linearStep = -1;
    int8_t hybridTailRequired = -1;
    int64_t eventTimeUs = 0;
    uint64_t revision = 0;
    int64_t cachedBlockCount = -1;
    uint64_t hashChecksum = 0;
    int64_t fullReadyBlockCount = -1;
    uint64_t fullReadyHashChecksum = 0;
    int64_t tailReadyBlockCount = -1;
    uint64_t tailReadyHashChecksum = 0;
    int64_t rtpCachedBlockCount = -1;
};

class AgentCacheR : public navi::Resource {
public:
    struct State {
        uint64_t revision = 0;
        size_t cachedBlockCount = 0;
        uint64_t hashChecksum = 0;
        size_t fullReadyBlockCount = 0;
        uint64_t fullReadyHashChecksum = 0;
        size_t tailReadyBlockCount = 0;
        uint64_t tailReadyHashChecksum = 0;
        int64_t rtpCachedBlockCount = -1;
        NaviSchedCachePolicy policy;
    };

    struct RequestAudit {
        NaviSchedRequestStats stats;
        int64_t physicalCacheHitTokenCount = 0;
        State cacheState;
        int64_t eventTimeUs = 0;
        bool hybridTailRequired = false;
        bool valid = true;
    };

    void def(navi::ResourceDefBuilder &builder) const override;
    bool config(navi::ResourceConfigContext &ctx) override;
    navi::ErrorCode init(navi::ResourceInitContext &ctx) override;

    NaviSchedRequestStats makeRequestStats(const NaviSchedBatchRequest &request) const;
    RequestAudit makeRequestAudit(const NaviSchedBatchRequest &request) const;
    RequestAudit makeRequestAudit(const NaviSchedBatchRequest &request,
                                  const AgentPendingCacheR *pendingCache) const;
    bool apply(const NaviSchedCacheUpdate &update, State &state);
    NaviSchedCacheUpdate makeSnapshot() const;
    State getState() const;
    size_t getCachedBlockCount() const;
    navi::SummaryErrorCode getSummary(const std::string &query,
                                      navi::SummaryBuilder &builder) const override;

public:
    static const std::string RESOURCE_ID;

private:
    mutable autil::ReadWriteLock _lock;
    std::unordered_set<uint64_t> _cachedBlockHashes;
    std::unordered_set<uint64_t> _fullReadyBlockHashes;
    std::unordered_set<uint64_t> _tailReadyBlockHashes;
    NaviSchedCachePolicy _policy;
    bool _policyReady = false;
    bool _hybridTailRequired = false;
    int64_t _totalBlocks = 0;
    uint64_t _revision = 0;
    uint64_t _hashChecksum = 0;
    uint64_t _fullReadyHashChecksum = 0;
    uint64_t _tailReadyHashChecksum = 0;
    int64_t _rtpCachedBlockCount = -1;
};

NAVI_TYPEDEF_PTR(AgentCacheR);

}
