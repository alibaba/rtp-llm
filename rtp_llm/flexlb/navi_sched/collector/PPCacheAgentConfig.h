#pragma once

#include "aios/navi_sched/collector/PPCacheCollectGraph.h"
#include "aios/navi_sched/common/NaviSchedCommon.h"
#include <set>

namespace navi_sched {

// Explicit opt-in for the PP adapter. The topology must name every stage.
struct PPCacheAgentConfig {
    std::string ppGroup;
    uint32_t pipelineParallelSize = 0;
    std::vector<PPCacheShardEndpoint> shards;
    NaviSchedCachePolicy policy;
    int64_t totalBlocks = 0;

    bool valid() const
    {
        if (ppGroup.empty() || pipelineParallelSize <= 1 ||
            pipelineParallelSize != shards.size() || totalBlocks < 0 ||
            !policy.valid(NaviSchedCacheKeyType::V6D))
            return false;
        std::set<uint32_t> ids;
        std::set<std::pair<std::string, int32_t>> addresses;
        for (const auto &shard : shards) {
            if (shard.shardId >= pipelineParallelSize || shard.bizName.empty() ||
                shard.host.empty() || shard.grpcPort <= 0 || shard.grpcPort > 65535 ||
                !ids.insert(shard.shardId).second ||
                !addresses.insert({shard.host, shard.grpcPort}).second)
                return false;
        }
        return true;
    }
};

}
