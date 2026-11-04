#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/metric/ANetMetricReporter.h"
#include "kmonitor/client/KMonitor.h"
#include "kmonitor/client/core/MutableMetric.h"

namespace anet {

/*
 * KMonitorANetMetricReporter use to report anet metric to kmon
 * metrics get from anet connection connstat
 */
class KMonitorANetMetricReporter : public ANetMetricReporter {
public:
    KMonitorANetMetricReporter(const ANetMetricReporterConfig &config, kmonitor::MetricLevel level)
        : ANetMetricReporter(config), _level(level) {}
    virtual ~KMonitorANetMetricReporter();

public:
    bool init(Transport *transport) override { return init(transport, nullptr); }
    bool init(Transport *transport, kmonitor::KMonitor *kMonitor);

protected:
    bool doRegisterMetrics();
    virtual void doReportConnStats(const std::map<std::string, ConnStat> &stats) override;

private:
    void reportOneConnStat(const ConnStat &stats);

private:
    kmonitor::KMonitor *_kMonitor{nullptr};
    kmonitor::MetricLevel _level;

    std::unique_ptr<kmonitor::MutableMetric> _connCountMetric;
    std::unique_ptr<kmonitor::MutableMetric> _rxBytesMetric;
    std::unique_ptr<kmonitor::MutableMetric> _txBytesMetric;
    std::unique_ptr<kmonitor::MutableMetric> _packetPostedMetric;
    std::unique_ptr<kmonitor::MutableMetric> _packetHandledMetric;
    std::unique_ptr<kmonitor::MutableMetric> _packetTimeoutMetric;
    std::unique_ptr<kmonitor::MutableMetric> _writeCntMetric;
    std::unique_ptr<kmonitor::MutableMetric> _readCntMetric;
    std::unique_ptr<kmonitor::MutableMetric> _queueSizeMetric;
    std::unique_ptr<kmonitor::MutableMetric> _writeOnceBytesMetric;
    std::unique_ptr<kmonitor::MutableMetric> _readOnceBytesMetric;
    std::unique_ptr<kmonitor::MutableMetric> _queueSizeMetricMax;
    std::unique_ptr<kmonitor::MutableMetric> _writeOnceBytesMetricMax;
    std::unique_ptr<kmonitor::MutableMetric> _readOnceBytesMetricMax;

private:
    AUTIL_LOG_DECLARE();
};

} // namespace anet
