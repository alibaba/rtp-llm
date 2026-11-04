#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/agent/GigAgent.h"

#include <assert.h>

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/agent/GigStatistic.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/metric/MetricReporterManager.h"

using namespace std;

namespace multi_call {
AUTIL_LOG_SETUP(multi_call, GigAgent);

GigAgent::GigAgent() : _statistic(NULL), _metricReporterManager(new MetricReporterManager()) {
}

GigAgent::~GigAgent() {
    DELETE_AND_SET_NULL(_statistic);
    AUTIL_LOG(INFO, "agent[%p] exited", this);
}

bool GigAgent::init(const std::string &logPrefix, bool enableAgentStat,
                    const std::string &snapshotLogDir) {
    if (enableAgentStat) {
        _statistic = new GigStatistic(logPrefix, snapshotLogDir);
        _statistic->init();
    }
    return _metricReporterManager->init(true);
}

std::string GigAgent::getLogPrefix() const {
    if (_statistic) {
        return _statistic->getLogPrefix();
    } else {
        return "";
    }
}

std::string GigAgent::getSnapshotLogDir() const {
    if (_statistic) {
        return _statistic->getSnapshotLogDir();
    } else {
        return "";
    }
}

QueryInfoPtr GigAgent::getQueryInfo(const std::string &queryInfo, const std::string &warmUpStrategy,
                                    bool withBizStat) {
    return QueryInfoPtr(new QueryInfo(queryInfo, warmUpStrategy, withBizStat ? _statistic : nullptr,
                                      _metricReporterManager));
}

void GigAgent::updateWarmUpStrategy(const std::string &warmUpStrategy, int64_t timeoutInSecond,
                                    int64_t queryCountLimit) {
    if (_statistic) {
        _statistic->updateWarmUpStrategy(warmUpStrategy, timeoutInSecond, queryCountLimit);
    }
}

void GigAgent::start(int64_t smoothIntervalUs) {
    if (_statistic) {
        _statistic->start(smoothIntervalUs);
    }
    AUTIL_LOG(INFO, "agent is started");
}

void GigAgent::stop(int64_t smoothIntervalUs) {
    if (_statistic) {
        _statistic->stop(smoothIntervalUs);
    }
    AUTIL_LOG(INFO, "agent is stopped");
}

bool GigAgent::isStopped() const {
    return _statistic ? _statistic->stopped() : true;
}

void GigAgent::start(const std::string &bizName, PartIdTy partId, int64_t smoothIntervalUs) {
    if (_statistic) {
        _statistic->start(bizName, partId, smoothIntervalUs);
    }
    AUTIL_LOG(INFO, "biz[%s] partId [%d] smooth interval [%ld] is started", bizName.c_str(), partId,
              smoothIntervalUs);
}

void GigAgent::stop(const std::string &bizName, PartIdTy partId, int64_t smoothIntervalUs) {
    if (_statistic) {
        _statistic->stop(bizName, partId, smoothIntervalUs);
    }
    AUTIL_LOG(INFO, "biz[%s] partId [%d] smooth interval [%ld] is stopped", bizName.c_str(), partId,
              smoothIntervalUs);
}

bool GigAgent::isStopped(const std::string &bizName, PartIdTy partId) const {
    return _statistic ? _statistic->stopped(bizName, partId) : true;
}

bool GigAgent::longTimeNoQuery(int64_t noQueryTimeInSecond) const {
    return _statistic ? _statistic->longTimeNoQuery(noQueryTimeInSecond) : true;
}

bool GigAgent::longTimeNoQuery(const std::string &bizName, int64_t noQueryTimeInSecond) const {
    return _statistic ? _statistic->longTimeNoQuery(bizName, noQueryTimeInSecond) : true;
}

std::shared_ptr<BizStat> GigAgent::getBizStat(const std::string &bizName, PartIdTy partId) {
    if (_statistic) {
        return _statistic->getBizStat(bizName, partId);
    } else {
        return nullptr;
    }
}

} // namespace multi_call
