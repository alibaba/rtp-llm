#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/controller/ControllerBase.h"

namespace multi_call {

class ServerRatioFilter;
class GigResponseInfo;
class PropagationStatDef;

class RatioControllerBase : public ControllerBase
{
public:
    RatioControllerBase(const std::string &controllerName, float decStep);

private:
    RatioControllerBase(const RatioControllerBase &);
    RatioControllerBase &operator=(const RatioControllerBase &);

public:
    float update(ControllerFeedBack &feedBack);

protected:
    virtual void updateLocalRatio(const QueryResultStatistic &stat) = 0;
    virtual bool isLegal(const ControllerChain *bestChain,
                         const MetricLimits &metricLimits) const = 0;
    virtual const ServerRatioFilter &
    getServerRatioFilter(const GigResponseInfo &agentInfo) const = 0;
    virtual const ServerRatioFilter &
    getMirrorServerRatioFilter(const PropagationStatDef &stat) const = 0;

protected:
    void updateServerRatio(const QueryResultStatistic &stat);
    void updateServerRatioMirror(const QueryResultStatistic &stat);

private:
    void doUpdateServerRatio(const ServerRatioFilter &serverRatioFilter);

private:
    float _decStep;

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(RatioControllerBase);

} // namespace multi_call

