#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/metric/KMonitorANetMetricReporter.h"

#include "kmonitor/client/KMonitorFactory.h"
using namespace std;
namespace anet {

AUTIL_LOG_SETUP(anet, KMonitorANetMetricReporter);

KMonitorANetMetricReporter::~KMonitorANetMetricReporter() {
    // should release report thread before dtor
    release();
}

bool KMonitorANetMetricReporter::init(Transport *transport, kmonitor::KMonitor *kMonitor) {
    if (!_config.enableANetMetric) {
        return true;
    }

    _transport = transport;
    if (_transport == nullptr) {
        AUTIL_LOG(WARN, "init kmon anet metric reporter failed, null transport");
        return false;
    }

    if (kMonitor == nullptr) {
        _kMonitor = kmonitor::KMonitorFactory::GetKMonitor("anet_kmonitor");
    } else {
        _kMonitor = kMonitor;
    }

    if (_kMonitor == nullptr) {
        AUTIL_LOG(WARN, "init kmon anet metric reporter failed, get kmonitor failed");
        return false;
    }
    std::string serviceName = "anet";
    if (!_config.metricPrefix.empty()) {
        serviceName = _config.metricPrefix + "." + serviceName;
    }
    _kMonitor->SetServiceName(serviceName);
    if (!doRegisterMetrics()) {
        return false;
    }

    return ANetMetricReporter::init(transport);
}

bool KMonitorANetMetricReporter::doRegisterMetrics() {
#define REGISTER_GAUGE_METRIC(metric, metric_name)                                                                           \
    metric.reset(_kMonitor->RegisterMetric(#metric_name, kmonitor::GAUGE, _level));                                    \
    if (metric == nullptr) {                                                                                           \
        AUTIL_LOG(WARN, "init kmon anet metric reporter failed, register metric anet.%s failed", #metric_name);        \
        return false;                                                                                                  \
    }

#define REGISTER_QPS_METRIC(metric, metric_name)                            \
    metric.reset(_kMonitor->RegisterMetric(#metric_name, kmonitor::QPS, _level));                                    \
    if (metric == nullptr) {                                                                                           \
        AUTIL_LOG(WARN, "init kmon anet metric reporter failed, register metric anet.%s failed", #metric_name);        \
        return false;                                                                                                  \
    }
    REGISTER_QPS_METRIC(_txBytesMetric, connection.tx_bytes);
    REGISTER_QPS_METRIC(_rxBytesMetric, connection.rx_bytes);
    REGISTER_QPS_METRIC(_packetPostedMetric, connection.packet_posted);
    REGISTER_QPS_METRIC(_packetHandledMetric, connection.packet_handled);
    REGISTER_QPS_METRIC(_packetTimeoutMetric, connection.packet_timeout);
    REGISTER_QPS_METRIC(_writeCntMetric, connection.write_cnt);
    REGISTER_QPS_METRIC(_readCntMetric, connection.read_cnt);
    REGISTER_GAUGE_METRIC(_connCountMetric, connection.connection_cnt);
    REGISTER_GAUGE_METRIC(_queueSizeMetric, connection.queue_size);
    REGISTER_GAUGE_METRIC(_writeOnceBytesMetric, connection.write_once_bytes);
    REGISTER_GAUGE_METRIC(_readOnceBytesMetric, connection.read_once_bytes);
    REGISTER_GAUGE_METRIC(_queueSizeMetricMax, connection.queue_size_max);
    REGISTER_GAUGE_METRIC(_writeOnceBytesMetricMax, connection.write_once_bytes_max);
    REGISTER_GAUGE_METRIC(_readOnceBytesMetricMax, connection.read_once_bytes_max);

#undef REGISTER_GAUGE_METRIC
#undef REGISTER_QPS_METRIC
    return true;
}

void KMonitorANetMetricReporter::doReportConnStats(const map<string, ConnStat> &stats) {
    _connCountMetric->Report(nullptr, stats.size());
    if (_config.fastMode) {
        ConnStat reportStat;
        int32_t maxQueueSize = 0;
        double maxWriteOnceByte = 0;
        double maxReadOnceByte = 0;
        for (const auto &pair : stats) {
            const auto & stat = pair.second;
            reportStat.totalTxBytes += stat.totalTxBytes;
            reportStat.totalRxBytes += stat.totalRxBytes;
            reportStat.packetPosted += stat.packetPosted;
            reportStat.packetHandled += stat.packetHandled;
            reportStat.packetTimeout += stat.packetTimeout;
            reportStat.callWriteCount += stat.callWriteCount;
            reportStat.callReadCount += stat.callReadCount;
            reportStat.queueSize += stat.queueSize;
            if (maxQueueSize < stat.queueSize) {
                maxQueueSize = stat.queueSize;
            }
            if (stat.callWriteCount > 0) {
                double writeOnce = double(stat.totalTxBytes) / double(stat.callWriteCount);
                if (maxWriteOnceByte < writeOnce) {
                    maxWriteOnceByte = writeOnce;
                }
            }
            if (stat.callReadCount > 0) {
                double readOnce = double(stat.totalRxBytes) / double(stat.callReadCount);
                if (maxReadOnceByte < readOnce) {
                    maxReadOnceByte = readOnce;
                }
            }
        }
        if (stats.size() > 0) {
            _queueSizeMetric->Report(nullptr, (double)reportStat.queueSize / (double)stats.size());
        }
        reportStat.queueSize = 0;
        _queueSizeMetricMax->Report(nullptr, maxQueueSize);
        if (reportStat.callWriteCount > 0) {
            _writeOnceBytesMetricMax->Report(nullptr, maxWriteOnceByte);
        }
        if (reportStat.callReadCount > 0) {
            _readOnceBytesMetricMax->Report(nullptr, maxReadOnceByte);
        }
        reportOneConnStat(reportStat);
    } else {
        for (const auto &stat : stats) {
            reportOneConnStat(stat.second);
        }
    }
}

void KMonitorANetMetricReporter::reportOneConnStat(const ConnStat &stat) {
    kmonitor::MetricsTagsPtr tags;
    if (_config.enableAddressTag) {
        auto transportName = _transport->getName();
        if (transportName.empty()) {
            transportName = "unknown";
        }
        map<string, string> tagsKv = {{"transport", transportName},
                                      {"peer_addr", stat.peerAddr.empty() ? "unknown" : stat.peerAddr},
                                      {"local_addr", stat.localAddr.empty() ? "unknown" : stat.localAddr}};
        tags.reset(new kmonitor::MetricsTags(tagsKv));
    }
    _txBytesMetric->Report(tags.get(), stat.totalTxBytes);
    _rxBytesMetric->Report(tags.get(), stat.totalRxBytes);
    _packetPostedMetric->Report(tags.get(), stat.packetPosted);
    _packetHandledMetric->Report(tags.get(), stat.packetHandled);
    _packetTimeoutMetric->Report(tags.get(), stat.packetTimeout);
    _queueSizeMetric->Report(tags.get(), stat.queueSize);
    _writeCntMetric->Report(tags.get(), stat.callWriteCount);
    _readCntMetric->Report(tags.get(), stat.callReadCount);
    if (stat.callWriteCount > 0) {
        _writeOnceBytesMetric->Report(tags.get(), double(stat.totalTxBytes) / double(stat.callWriteCount));
    }
    if (stat.callReadCount > 0) {
        _readOnceBytesMetric->Report(tags.get(), double(stat.totalRxBytes) / double(stat.callReadCount));
    }
}

} // namespace anet
