#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/controller/ErrorController.h"

using namespace std;

namespace multi_call {
AUTIL_LOG_SETUP(multi_call, ErrorController);

ErrorController::ErrorController() {
}

ErrorController::~ErrorController() {
}

float ErrorController::update(ControllerFeedBack &feedBack) {
    if (feedBack.stat.isFailed()) {
        return -min(2.0f * ControllerParam::WEIGHT_UPDATE_STEP, MAX_WEIGHT_UPDATE_STEP_FLOAT);
    } else {
        return 0.0f;
    }
}

} // namespace multi_call
