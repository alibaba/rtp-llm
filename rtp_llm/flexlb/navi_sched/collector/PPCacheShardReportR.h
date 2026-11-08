#pragma once

#include "aios/navi_sched/collector/PPCacheShardUpdateData.h"
#include "navi/engine/AsyncPipe.h"
#include "navi/engine/Resource.h"
#include <atomic>
#include <deque>
#include <mutex>
#include <unordered_set>

namespace navi_sched {

// KV-only counterpart of AgentReportR. Each subscription starts with a
// complete cache view, then receives ordered deltas and request terminals.
class PPCacheShardReportR : public navi::ExternalResource {
public:
    PPCacheShardReportR();
    bool initializeIdentity(const std::string &group, uint32_t shardId,
                            uint32_t shardCount);
    bool subscribe(const std::string &group, uint32_t shardId, uint32_t shardCount,
                   const std::string &collectorId, uint64_t generation,
                   const navi::AsyncPipePtr &pipe);
    bool unsubscribe(const std::string &collectorId, const navi::AsyncPipePtr &pipe);
    bool publish(const PPCacheShardUpdate &localChange);
    bool publishRequestFinished(const std::string &group, uint32_t shardCount,
                                const std::string &collectorId,
                                const PPCacheShardUpdateData &data);
    void stop();
    static const std::string RESOURCE_ID;

private:
    struct Subscriber {
        std::string collectorId;
        uint64_t generation = 0;
        uint64_t sequence = 0;
        uint64_t requestFinishedSequence = 0;
        std::weak_ptr<navi::AsyncPipe> pipe;
        std::atomic<bool> active{true};
    };
    struct Delivery {
        std::shared_ptr<Subscriber> subscriber;
        PPCacheShardUpdateDataPtr data;
    };
    void enqueueLocked(const PPCacheShardUpdate &change,
                       const std::shared_ptr<Subscriber> &subscriber);
    void drain();

    std::mutex _mutex;
    std::string _group;
    uint32_t _shardId = 0;
    uint32_t _shardCount = 0;
    bool _peerCacheReadAvailable = false;
    bool _initialized = false;
    bool _stopping = false;
    bool _draining = false;
    std::unordered_set<uint64_t> _full;
    std::unordered_set<uint64_t> _tail;
    std::shared_ptr<Subscriber> _subscriber;
    std::deque<Delivery> _deliveries;
};

}
