#pragma once

#include "aios/navi_sched/collector/PPCacheShardUpdateData.h"
#include "navi/engine/AsyncPipe.h"
#include "navi/engine/Resource.h"
#include <atomic>
#include <deque>
#include <functional>
#include <mutex>

namespace navi_sched {

// Carries native request outcomes to the PP collectors. These notifications
// never assert that asynchronous KV publication has completed.
class PPRequestFinishedR : public navi::ExternalResource {
public:
    using FailureCallback = std::function<void(uint32_t, uint64_t)>;
    PPRequestFinishedR();
    bool initialize(const std::string &group, uint32_t shardCount,
                    FailureCallback onStreamFailure);
    bool reserve(uint32_t shardId, uint64_t generation);
    bool subscribe(const std::string &group, uint32_t shardId, uint32_t shardCount,
                   uint64_t generation, const navi::AsyncPipePtr &pipe);
    bool unsubscribe(uint32_t shardId, uint64_t generation,
                     const navi::AsyncPipePtr &pipe);
    bool disconnect(uint32_t shardId, uint64_t generation);
    bool publish(uint32_t shardId, uint64_t generation,
                 const PPShardRequestTerminal &finished);
    void stop();
    static const std::string RESOURCE_ID;

private:
    struct Subscriber {
        uint32_t shardId = 0;
        uint64_t generation = 0;
        std::weak_ptr<navi::AsyncPipe> pipe;
        std::atomic<bool> active{true};
    };
    struct Slot {
        std::shared_ptr<Subscriber> subscriber;
        uint64_t sequence = 0;
        bool subscribed = false;
        std::deque<PPCacheShardUpdateDataPtr> pending;
    };
    struct Delivery {
        std::shared_ptr<Subscriber> subscriber;
        PPCacheShardUpdateDataPtr data;
    };
    bool detach(uint32_t shardId, uint64_t generation,
                const navi::AsyncPipePtr &expectedPipe, bool terminate);
    void drain();
    void fail(const std::shared_ptr<Subscriber> &subscriber, bool terminate);
    std::mutex _mutex;
    std::string _group;
    std::vector<Slot> _slots;
    FailureCallback _onStreamFailure;
    std::deque<Delivery> _deliveries;
    bool _draining = false;
    bool _stopping = false;
};

}
