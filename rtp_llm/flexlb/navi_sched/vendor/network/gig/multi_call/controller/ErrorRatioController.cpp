#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/controller/ErrorRatioController.h"

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/controller/ControllerChain.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/proto/GigAgent.pb.h"
#include "autil/StringUtil.h"

using namespace std;
using namespace autil;

namespace multi_call {
AUTIL_LOG_SETUP(multi_call, ErrorRatioController);

ErrorRatioController::ErrorRatioController()
    : RatioControllerBase("errorRatio", ERROR_RATIO_WEIGHT_DEC_STEP) {
}

void ErrorRatioController::updateLocalRatio(const QueryResultStatistic &stat) {
    float isFailed = stat.isFailed() ? MAX_RATIO : MIN_RATIO;
    updateLocal(isFailed);
}

const ServerRatioFilter &
ErrorRatioController::getServerRatioFilter(const GigResponseInfo &agentInfo) const {
    return agentInfo.error_ratio();
}

const ServerRatioFilter &
ErrorRatioController::getMirrorServerRatioFilter(const PropagationStatDef &stat) const {
    return stat.error();
}

bool ErrorRatioController::isLegal(const ControllerChain *bestChain,
                                   const MetricLimits &metricLimits) const {
    if (!bestChain) {
        return true;
    }
    const auto &best = bestChain->errorRatioController;
    if (&best == this) {
        return true;
    }
    auto allow = metricLimits.errorRatioLimit * MAX_RATIO;
    auto flag = compare(best, allow);
    return WorseCompareFlag != flag;
}

float ErrorRatioController::controllerOutput() const {
    if (serverValueReady()) {
        return serverValue();
    } else {
        return localValue();
    }
}

float ErrorRatioController::getLegalLimit(const FlowControlConfigPtr &flowControlConfig) const {
    if (!isValid()) {
        return INVALID_FILTER_VALUE;
    }
    auto output = controllerOutput();
    return output + flowControlConfig->errorRatioLimit * MAX_RATIO;
}

void ErrorRatioController::updateLoadBalance(ControllerFeedBack &feedBack) {
    if (!hasServerAgent()) {
        return;
    }
    if (feedBack.mirrorResponse) {
        updateServerRatioMirror(feedBack.stat);
    }
}

} // namespace multi_call
