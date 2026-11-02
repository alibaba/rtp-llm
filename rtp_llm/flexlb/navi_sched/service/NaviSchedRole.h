#pragma once

#include "navi_sched/NaviSchedCommon.h"
#include <atomic>
#include <memory>
#include <string>
#include <unordered_map>

namespace autil {
class LoopThread;
}

namespace kmonitor {
class MetricsReporter;
class MetricsTags;
}

namespace multi_call {
class GigRpcServer;
}

namespace navi {
class Navi;
class ResourceMap;
}

namespace navi_sched {

class NaviSchedAgent;
class NaviSchedAgentImpl;
class NaviSchedClient;
class NaviSchedMaster;
class NaviSchedRoleRuntimeAccess;
struct NaviSchedRoleSummary;

class NAVI_SCHED_API NaviSchedRole {
public:
    virtual ~NaviSchedRole();

public:
    bool init();
    virtual void stop();
    int32_t getGrpcPort() const;
    int32_t getHttpPort() const;

protected:
    explicit NaviSchedRole(const NaviSchedConfig &config);
    const NaviSchedConfig &getConfig() const;
    virtual void onUpdating();
    virtual bool onUpdated();
    virtual std::string makeSnapshot() const;
    virtual void reportMetrics() const;
    virtual bool enableHttpRpc() const;
    void reportMetric(const std::string &name, double value) const;
    void reportMetric(const std::string &name, double value,
                      const kmonitor::MetricsTags &tags) const;
    void reportSampleMetric(const std::string &name, double value) const;
    void reportSampleMetric(const std::string &name, double value,
                            const kmonitor::MetricsTags &tags) const;
    uint64_t reportCounterQpsMetric(const std::string &name, uint64_t total) const;
    uint64_t reportCounterQpsMetricAt(const std::string &name, uint64_t total,
                                      int64_t reportTimeUs) const;

private:
    friend class NaviSchedAgentImpl;
    friend class NaviSchedClient;
    friend class NaviSchedMaster;
    friend class NaviSchedRoleRuntimeAccess;

    bool update(const std::string &configStr,
                const std::shared_ptr<navi::ResourceMap> &rootResourceMap);
    const std::shared_ptr<navi::Navi> &getNavi() const;
    void stopSnapshot();
    std::shared_ptr<NaviSchedRoleSummary> makeSnapshotSummary() const;
    bool initKmon();
    bool initGig();
    bool startSnapshot();
    void recordSnapshot() const;

private:
    NaviSchedConfig _config;
    std::shared_ptr<navi::Navi> _navi;
    std::shared_ptr<kmonitor::MetricsReporter> _metricsReporter;
    std::shared_ptr<multi_call::GigRpcServer> _gigRpcServer;
    std::shared_ptr<autil::LoopThread> _metricsThread;
    std::shared_ptr<autil::LoopThread> _snapshotThread;
    mutable std::atomic<size_t> _snapshotId{0};
    struct QpsCounterState {
        uint64_t total = 0;
        int64_t reportTimeUs = 0;
    };
    mutable std::unordered_map<std::string, QpsCounterState> _lastQpsCounters;
    bool _kmonAcquired = false;
    bool _started = false;
};

}
