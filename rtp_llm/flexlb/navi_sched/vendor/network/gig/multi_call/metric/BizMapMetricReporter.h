#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/metric/BizMetricReporter.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/metric/ReplyInfoCollector.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/metric/SnapshotInfoCollector.h"

namespace multi_call {

class BizMapMetricReporter
{
public:
    BizMapMetricReporter(kmonitor::KMonitor *kMonitor, bool isAgent);
    virtual ~BizMapMetricReporter();

private:
    BizMapMetricReporter(const BizMapMetricReporter &);
    BizMapMetricReporter &operator=(const BizMapMetricReporter &);

public:
    void reportSnapshotInfo(const SnapshotInfoCollector &snapInfo);
    void reportReplyInfo(const ReplyInfoCollector &replyInfo);
    BizMetricReporterPtr getBizMetricReporter(const std::string &bizName);

    typedef std::unordered_map<std::string, BizMetricReporterPtr> BizMetricReporterMap;

private:
    kmonitor::KMonitor *_kMonitor;
    bool _isAgent;
    BizMetricReporterMap _bizMetricReporterMap;
    autil::ReadWriteLock _lock;

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(BizMapMetricReporter);
} // namespace multi_call

