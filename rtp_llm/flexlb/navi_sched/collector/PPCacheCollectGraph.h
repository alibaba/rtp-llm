#pragma once

#include "aios/navi_sched/collector/PPCacheMergeR.h"
#include "aios/navi_sched/collector/PPRequestFinishedR.h"
#include "navi/engine/Navi.h"
#include "navi/engine/NaviUserResult.h"
#include <condition_variable>
#include <thread>

namespace navi_sched {

struct PPCacheShardEndpoint {
    uint32_t shardId = 0;
    std::string bizName;
    std::string host;
    int32_t grpcPort = 0;
};

// Persistent C++ Navi streams, one for every required physical PP shard.
// A stream error retracts that shard before reconnecting with a full snapshot.
class PPCacheCollectGraph {
public:
    PPCacheCollectGraph(const navi::NaviPtr &navi, const std::string &localBiz,
                        const std::string &group,
                        const std::shared_ptr<PPCacheMergeR> &merge,
                        const std::shared_ptr<PPRequestFinishedR> &requestFinished);
    ~PPCacheCollectGraph();
    bool start(const std::vector<PPCacheShardEndpoint> &endpoints);
    size_t publishRequestFinished(const PPShardRequestTerminal &finished);
    void stop();

private:
    void collect(const PPCacheShardEndpoint &endpoint, uint32_t shardCount);
    std::unique_ptr<navi::GraphDef> build(const PPCacheShardEndpoint &endpoint,
                                          uint32_t shardCount, uint64_t generation) const;
    navi::NaviPtr _navi;
    std::string _localBiz;
    std::string _group;
    std::shared_ptr<PPCacheMergeR> _merge;
    std::shared_ptr<PPRequestFinishedR> _requestFinished;
    std::mutex _mutex;
    std::condition_variable _condition;
    bool _stopping = false;
    std::vector<std::thread> _threads;
    std::vector<navi::NaviUserResultPtr> _active;
};

}
