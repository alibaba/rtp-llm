#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/controller/WarmUpController.h"

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/proto/GigAgent.pb.h"

using namespace std;

namespace multi_call {
AUTIL_LOG_SETUP(multi_call, WarmUpController);

float WarmUpController::update(ControllerFeedBack &feedBack) {
    if (!feedBack.stat.agentInfo) {
        return 0.0f;
    }
    const auto &agentInfo = *feedBack.stat.agentInfo;
    if (agentInfo.has_warm_up_status() && !agentInfo.warm_up_status().finished()) {
        _isFinished = false;
        return -MAX_WEIGHT_FLOAT;
    } else {
        _isFinished = true;
        return 0.0f;
    }
}

} // namespace multi_call
