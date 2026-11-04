#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/config/FlowControlConfig.h"

namespace multi_call {

AUTIL_LOG_SETUP(multi_call, FlowControlConfig);

FlowControlConfig *FlowControlConfig::clone() {
    FlowControlConfigMapPtr tmpConfigMap;
    if (subConfigMap && !subConfigMap->empty()) {
        tmpConfigMap = std::make_shared<FlowControlConfigMap>();
        for (const auto &pair : *subConfigMap) {
            if (!pair.second) {
                continue;
            }
            auto cloneConfig = FlowControlConfigPtr(pair.second->clone());
            (*tmpConfigMap)[pair.first] = cloneConfig;
        }
    }
    auto ret = new FlowControlConfig(*this);
    ret->subConfigMap = tmpConfigMap;
    return ret;
}

void FlowControlConfig::validate() {
    errorRatioLimit = getValidPercent(errorRatioLimit, "error_limit_ratio");
    etTriggerPercent = getValidPercent(etTriggerPercent, "et percent");
    retryTriggerPercent = getValidPercent(retryTriggerPercent, "retry percent");
    latencyTimeWindowSize = getValidInteger(latencyTimeWindowSize, "window_size");
    if (beginServerDegradeLatency > beginDegradeLatency) {
        AUTIL_LOG(ERROR,
                  "beginServerDegradeLatency [%u] is great than "
                  "beginDegradeLatency [%u]",
                  beginServerDegradeLatency, beginDegradeLatency);
        beginDegradeLatency = beginServerDegradeLatency;
    }
    if (beginDegradeLatency > fullDegradeLatency) {
        AUTIL_LOG(ERROR, "beginDegradeLatency [%u] is great than fullDegradeLatency [%u]",
                  beginDegradeLatency, fullDegradeLatency);
        fullDegradeLatency = beginDegradeLatency;
    }

    beginServerDegradeErrorRatio =
        getValidPercent(beginServerDegradeErrorRatio, "begin_server_degrade_error_ratio");
    beginDegradeErrorRatio = getValidPercent(beginDegradeErrorRatio, "begin_degrade_error_ratio");
    fullDegradeErrorRatio = getValidPercent(fullDegradeErrorRatio, "full_degrade_error_ratio");
    if (beginServerDegradeErrorRatio > beginDegradeErrorRatio) {
        AUTIL_LOG(ERROR,
                  "beginServerDegradeErrorRatio [%f] is great"
                  " than beginDegradeErrorRatio [%f]",
                  beginServerDegradeErrorRatio, beginDegradeErrorRatio);
        beginDegradeErrorRatio = beginServerDegradeErrorRatio;
    }
    if (beginDegradeErrorRatio > fullDegradeErrorRatio) {
        AUTIL_LOG(ERROR,
                  "beginDegradeErrorRatio [%f] is great "
                  "than fullDegradeErrorRatio [%f]",
                  beginDegradeErrorRatio, fullDegradeErrorRatio);
        fullDegradeErrorRatio = beginDegradeErrorRatio;
    }
    if (subConfigMap) {
        for (const auto &pair : *subConfigMap) {
            if (pair.second) {
                pair.second->validate();
            }
        }
    }
}

} // namespace multi_call
