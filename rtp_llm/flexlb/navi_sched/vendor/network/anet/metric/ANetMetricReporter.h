#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/metric/ANetMetricReporterConfig.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/transport.h"
#include "autil/Log.h"
#include "autil/LoopThread.h"

namespace anet {

class ANetMetricReporter {
public:
    ANetMetricReporter(const ANetMetricReporterConfig &config) : _config(config) {}
    virtual ~ANetMetricReporter();

public:
    virtual bool init(Transport *transport);
    void release();

protected:
    void reportThreadProc();
    virtual void doReportConnStats(const std::map<std::string, ConnStat> &stats) = 0;

protected:
    ANetMetricReporterConfig _config;
    Transport *_transport{nullptr};

    std::map<std::string, ConnStat> _lastConnStats;
    autil::LoopThreadPtr _reportThread;

private:
    AUTIL_LOG_DECLARE();
};

} // namespace anet
