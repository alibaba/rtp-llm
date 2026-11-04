#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/metric/ANetMetricReporter.h"

namespace anet {

AUTIL_LOG_SETUP(anet, ANetMetricReporter);

ANetMetricReporter::~ANetMetricReporter() { release(); }

bool ANetMetricReporter::init(Transport *transport) {
    if (!_config.enableANetMetric) {
        return true;
    }

    _transport = transport;
    if (_transport == nullptr) {
        AUTIL_LOG(WARN, "init anet metric reporter failed, null transport");
        return false;
    }

    if (_config.fastMode && _config.intervalMs != 1000) {
        AUTIL_LOG(WARN, "init anet metric reporter in fast mode, interval should be 1000ms");
        _config.intervalMs = 1000;
    }

    auto name = _transport->getName() + "ANetMetric";
    _reportThread = autil::LoopThread::createLoopThread(
        std::bind(&ANetMetricReporter::reportThreadProc, this), _config.intervalMs * 1000, name);
    AUTIL_LOG(INFO,
              "init anet metric reporter done, report interval %ld, report thread %s",
              _config.intervalMs,
              name.c_str());
    return true;
}

void ANetMetricReporter::release() {
    if (_reportThread != nullptr) {
        _reportThread->stop();
        _reportThread.reset();
    }
    AUTIL_LOG(INFO, "anet metric reporter released!");
}

void ANetMetricReporter::reportThreadProc() {
    std::vector<ConnStat> connStats;
    _transport->getTcpConnStats(connStats);
    std::map<std::string, ConnStat> statsMap;
    std::map<std::string, ConnStat> reportStatsMap;
    for (auto &connStat : connStats) {
        auto connName = connStat.localAddr + "|" + connStat.peerAddr;
        auto iter = _lastConnStats.find(connName);
        if (iter != _lastConnStats.end()) {
            iter->second.totalRxBytes = connStat.totalRxBytes - iter->second.totalRxBytes;
            iter->second.totalTxBytes = connStat.totalTxBytes - iter->second.totalTxBytes;
            iter->second.packetPosted = connStat.packetPosted - iter->second.packetPosted;
            iter->second.packetHandled = connStat.packetHandled - iter->second.packetHandled;
            iter->second.packetTimeout = connStat.packetTimeout - iter->second.packetTimeout;
            iter->second.queueSize = connStat.queueSize;
            iter->second.callWriteCount = connStat.callWriteCount - iter->second.callWriteCount;
            iter->second.callReadCount = connStat.callReadCount - iter->second.callReadCount;
            reportStatsMap[connName] = iter->second;
        } else {
            reportStatsMap[connName] = connStat;
        }
        statsMap[connName] = connStat;
    }
    doReportConnStats(reportStatsMap);
    _lastConnStats.swap(statsMap);
}

} // namespace anet
