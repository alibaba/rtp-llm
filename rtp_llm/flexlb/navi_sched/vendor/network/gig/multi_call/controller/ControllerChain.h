#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/controller/DegradeRatioController.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/controller/ErrorController.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/controller/ErrorRatioController.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/controller/LatencyController.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/controller/WarmUpController.h"

namespace multi_call {

struct ControllerChain {
    ControllerChain() : targetWeight(MAX_WEIGHT) {
    }

private:
    static float trimSmall(float value);

public:
    void toString(std::string &debugStr) const;
    DegradeRatioController degradeRatioController;
    LatencyController latencyController;
    ErrorRatioController errorRatioController;
    ErrorController errorController;
    WarmUpController warmUpController;
    volatile WeightTy targetWeight;
};

} // namespace multi_call

