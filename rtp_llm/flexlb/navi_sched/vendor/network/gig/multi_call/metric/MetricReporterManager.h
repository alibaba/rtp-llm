#pragma once

#include <mutex>
#include <unordered_map>

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/MetaEnv.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/metric/BizMapMetricReporter.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/metric/HeartbeatMetricReporter.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/metric/LinkMetricReporter.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/metric/ReplyInfoCollector.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/metric/SnapshotInfoCollector.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/metric/WorkerMetricReporter.h"

namespace multi_call {

class MetricReporterManager
{
public:
    MetricReporterManager();
    virtual ~MetricReporterManager();

private:
    MetricReporterManager(const MetricReporterManager &);
    MetricReporterManager &operator=(const MetricReporterManager &);

public:
    // virtual for ut
    virtual bool init(bool isAgent, bool disableMetricReport = false,
                      bool simplifyMetricReport = false);

    void reportSnapshotInfo(const SnapshotInfoCollector &snapInfo);
    void reportReplyInfo(const ReplyInfoCollector &replyInfo);
    BizMetricReporterPtr getBizMetricReporter(const std::string &bizName);

    const WorkerMetricReporterPtr &getWorkerMetricReporter() const;
    const HeartbeatMetricReporterPtr &getHeartbeatMetricReporter() const;

    const MetaEnv &getMetaEnv() const {
        return _metaEnv;
    }
    void setMetaEnv(MetaEnv &metaEnv) {
        _metaEnv = metaEnv;
    }

    void reportLinkMetric(const MetaEnv &targetMetaEnv, const std::string &biz,
                          const std::string &targetBiz, const std::string &src,
                          const std::string &srcAb, const std::string &stressTest, int64_t latency,
                          bool timeout, bool error);

    void setReportSampling(int32_t sampling);

private:
    static void initKMonitor();

private:
    static std::once_flag _sKMonitorInitFlag;
    kmonitor::KMonitor *_kMonitor;
    bool _isAgent;
    MetaEnv _metaEnv;
    BizMapMetricReporterPtr _bizMapMetricReporter;
    WorkerMetricReporterPtr _workerMetricReporter;
    HeartbeatMetricReporterPtr _heartbeatReporter;
    LinkMetricReporterPtr _linkMetricReporter;

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(MetricReporterManager);
} // namespace multi_call

