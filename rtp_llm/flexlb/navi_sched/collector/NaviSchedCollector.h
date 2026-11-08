#pragma once

#include "aios/navi_sched/collector/PPCacheShardReportR.h"
#include "aios/navi_sched/collector/PPCacheResidentState.h"
#include "aios/navi_sched/service/NaviSchedRole.h"
#include <atomic>

namespace navi_sched {

struct NaviSchedCollectorConfig {
    NaviSchedConfig service;
    NaviSchedRuntimeConfig runtime;
    std::string ppGroup;
    uint32_t shardId = 0;
    uint32_t shardCount = 0;
    std::vector<PPCacheGroupRole> groupRoles;
    std::vector<uint32_t> groupBlockScales;
};

// A shard-local Navi role. It publishes KV residency and request outcomes only;
// execution scheduling and latency learning stay in NaviSchedAgent.
class NAVI_SCHED_API NaviSchedCollector final : public NaviSchedRole {
public:
    explicit NaviSchedCollector(const NaviSchedCollectorConfig &config);
    ~NaviSchedCollector() override;
    bool init();
    bool publish(const PPCacheShardUpdate &localChange);
    bool replaceResidentObjects(const std::vector<PPCacheResidentObject> &objects);
    bool objectResident(const PPCacheResidentObject &object);
    bool objectRemoved(const std::string &objectId);
    bool reportPeerCacheReadAvailable(bool available);
    void stop() override;

private:
    bool publishResidentChange(const PPCacheShardUpdate &change);
    std::mutex _producerMutex;
    std::unique_ptr<PPCacheResidentState> _residentState;
    NaviSchedCollectorConfig _collectorConfig;
    std::shared_ptr<PPCacheShardReportR> _report;
    std::atomic<bool> _started{false};
    std::atomic<bool> _stopped{false};
};

}
