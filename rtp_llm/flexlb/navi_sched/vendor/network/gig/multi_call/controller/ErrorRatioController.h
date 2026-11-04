#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/config/FlowControlConfig.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/controller/RatioControllerBase.h"

namespace multi_call {

class ErrorRatioController : public RatioControllerBase
{
public:
    ErrorRatioController();

private:
    ErrorRatioController(const ErrorRatioController &);
    ErrorRatioController &operator=(const ErrorRatioController &);

public:
    float controllerOutput() const override;
    void updateLoadBalance(ControllerFeedBack &feedBack);

public:
    bool isLegal(const ControllerChain *bestChain, const MetricLimits &metricLimits) const override;
    float getLegalLimit(const FlowControlConfigPtr &flowControlConfig) const;

protected:
    void updateLocalRatio(const QueryResultStatistic &stat) override;
    const ServerRatioFilter &getServerRatioFilter(const GigResponseInfo &agentInfo) const override;
    const ServerRatioFilter &
    getMirrorServerRatioFilter(const PropagationStatDef &stat) const override;

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(ErrorRatioController);

} // namespace multi_call

