#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/controller/RatioControllerBase.h"

namespace multi_call {

class DegradeRatioController : public RatioControllerBase
{
public:
    DegradeRatioController();

private:
    DegradeRatioController(const DegradeRatioController &);
    DegradeRatioController &operator=(const DegradeRatioController &);

public:
    float updateLoadBalance(ControllerFeedBack &feedBack);
    float controllerOutput() const override;
    float controllerLoadBalanceOutput() const;
    bool isLegal(const ControllerChain *bestChain, const MetricLimits &metricLimits) const override;
    float getLegalLimit() const;

protected:
    void updateLocalRatio(const QueryResultStatistic &stat) override;
    const ServerRatioFilter &getServerRatioFilter(const GigResponseInfo &agentInfo) const override;
    const ServerRatioFilter &
    getMirrorServerRatioFilter(const PropagationStatDef &stat) const override;
    bool getBestRatio(const ControllerFeedBack &feedBack, float &bestRatio) const;

private:
    static float calculateWeightPercent(float thisRatio, float bestRatio);

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(DegradeRatioController);

} // namespace multi_call

