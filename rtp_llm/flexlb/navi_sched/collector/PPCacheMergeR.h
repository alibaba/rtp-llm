#pragma once

#include "aios/navi_sched/collector/PPCacheCollectorState.h"
#include "navi/engine/Resource.h"
#include <deque>

namespace navi_sched {

// One instance per logical PP group. Mutations and downstream notifications
// share an ordering queue, so concurrent shard graphs cannot reorder add/remove.
class PPCacheMergeR : public navi::ExternalResource {
public:
    using Callback = std::function<void(const PPCacheCollectorChanges &)>;
    PPCacheMergeR();
    bool initialize(const std::string &group, uint32_t shardCount, Callback callback);
    PPCacheCollectorChanges connect(const std::string &group, uint32_t shardId,
                                    uint32_t shardCount, uint64_t generation);
    PPCacheCollectorChanges disconnect(uint32_t shardId, uint64_t generation);
    PPCacheCollectorChanges apply(const PPCacheShardUpdate &update);
    PPCacheCollectorChanges beginRequest(const PPRequestAttempt &request);
    std::vector<PPRequestShardConnection> pendingRequestShards(
        const PPRequestAttempt &request);
    uint64_t allocateGeneration();
    static const std::string RESOURCE_ID;

private:
    PPCacheCollectorChanges mutate(
        const std::function<PPCacheCollectorChanges(PPCacheCollectorState &)> &operation);
    void drain();
    std::mutex _mutex;
    std::string _group;
    uint32_t _shardCount = 0;
    std::unique_ptr<PPCacheCollectorState> _state;
    Callback _callback;
    std::deque<PPCacheCollectorChanges> _pending;
    bool _draining = false;
    uint64_t _nextGeneration = 0;
};

}
