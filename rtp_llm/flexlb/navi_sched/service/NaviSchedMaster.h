#pragma once

#include "aios/navi_sched/common/NaviSchedScheduleData.h"
#include "aios/navi_sched/resource/MasterStateR.h"
#include "aios/navi_sched/service/NaviSchedRole.h"
#include "autil/Log.h"
#include <map>
#include <memory>
#include <string>
#include <unordered_map>

namespace navi_sched {

class MasterCollectorManager;
class MasterLatencyModelR;
class MasterScheduleGraph;
class MasterScheduleR;
class SchedulerConfigR;
class ReplayControlR;
struct NaviSchedMasterNodeSummary;

struct NaviSchedMasterStats {
    uint64_t agentRequestReceivedCount = 0;
    uint64_t agentRequestFinishedCount = 0;
    int64_t agentRequestCacheHitTokens = 0;
    int64_t agentRequestCacheMissTokens = 0;
    int64_t agentRequestTotalLatencyUs = 0;
};

class NaviSchedMaster final : public NaviSchedRole {
public:
    explicit NaviSchedMaster(const NaviSchedMasterConfig &config);
    ~NaviSchedMaster() override;

public:
    bool init();
    std::map<std::string, NaviSchedNodeInfo> getNodes() const;
    NaviSchedMasterStats getStats() const;
    void stop() override;

protected:
    void onUpdating() override;
    bool onUpdated() override;
    std::string makeSnapshot() const override;
    void reportMetrics() const override;
    bool enableHttpRpc() const override;

private:
    AUTIL_LOG_DECLARE();
    static std::map<std::string, uint64_t> makeCacheStateSummary(
        const MasterStateR::CacheSyncStats &stats);
    static NaviSchedMasterNodeSummary makeNodeSummary(
        const NaviSchedNodeInfo &node, const MasterStateR::CacheSyncStats &cacheStats,
        uint64_t sentRequestCount, uint64_t sentBatchCount, double timeCompression);
    std::string _masterBizName;
    NaviSchedRuntimeConfig _masterConfig;
    size_t _collectorThreadNum = 0;
    size_t _collectorQueueCount = 0;
    std::vector<std::string> _collectorTaskQueueNames;
    size_t _configuredMaxWaitScheduleCount = 30;
    int64_t _configuredMaxWaitScheduleTimeMs = 10;
    int64_t _configuredQueueOverloadStartCycleMultiplier = 10;
    int64_t _configuredQueueOverloadFullCycleMultiplier = 30;
    std::string _configuredSchedulerName;
    std::string _configuredSchedulerConfig;
    double _configuredTimeCompression = 1.0;
    std::string _configuredLatencyModelPoolStateFile;
    int64_t _configuredLatencyModelPoolSourceMinRevision = 128;
    size_t _configuredLatencyModelPoolMaxCandidates = 4;
    int64_t _configuredLatencyModelPoolPersistIntervalMs = 10000;
    size_t _maxWaitScheduleCount = 30;
    int64_t _maxWaitScheduleTimeMs = 10;
    std::string _schedulerKernelId;
    std::string _contextKernelConfig;
    std::string _schedulerKernelConfig;
    std::shared_ptr<MasterStateR> _masterStateR;
    std::shared_ptr<MasterLatencyModelR> _masterLatencyModelR;
    std::shared_ptr<MasterScheduleR> _masterScheduleR;
    std::shared_ptr<SchedulerConfigR> _schedulerConfigR;
    std::shared_ptr<ReplayControlR> _replayControlR;
    std::shared_ptr<MasterScheduleGraph> _scheduleGraph;
    std::shared_ptr<MasterCollectorManager> _collectorManager;
    mutable std::unordered_map<std::string, NaviSchedLatencyValidationStats>
        _lastValidationStats;
};

}
