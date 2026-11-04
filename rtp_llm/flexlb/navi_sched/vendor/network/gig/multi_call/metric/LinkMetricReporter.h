#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/MetaEnv.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/util/MetricUtil.h"

namespace multi_call {

class LinkMetricReporter
{
public:
    LinkMetricReporter(kmonitor::KMonitor *kMonitor, MetaEnv &metaEnv);
    ~LinkMetricReporter();

private:
    LinkMetricReporter(const LinkMetricReporter &);
    LinkMetricReporter &operator=(const LinkMetricReporter &);

    DECLARE_MUTABLE_METRIC(Qps);
    DECLARE_MUTABLE_METRIC(Latency);
    DECLARE_MUTABLE_METRIC(ErrorQps);
    DECLARE_MUTABLE_METRIC(TimeoutQps);

public:
    void reportMetric(const MetaEnv &targetMetaEnv, const std::string &biz,
                      const std::string &targetBiz, const std::string &src,
                      const std::string &srcAb, const std::string &stressTest, int64_t latency,
                      bool timeout, bool error);
    void setReportSampling(int32_t interval);

private:
    void doReportMetric(const MetaEnv &targetMetaEnv, const std::string &biz,
                        const std::string &targetBiz, const std::string &src,
                        const std::string &srcAb, const std::string &stressTest, int64_t latency,
                        bool timeout, bool error);

private:
    kmonitor::MetricsTagsPtr getLinkMetricsTags(const MetaEnv &targetMetaEnv,
                                                const std::string &biz,
                                                const std::string &targetBiz,
                                                const std::string &src, const std::string &srcAb,
                                                const std::string &stressTest);

private:
    kmonitor::KMonitor *_kMonitor;
    MetaEnv _metaEnv;
    int32_t _reportSampling;
    std::atomic<int32_t> _counter;

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(LinkMetricReporter);

} // namespace multi_call

