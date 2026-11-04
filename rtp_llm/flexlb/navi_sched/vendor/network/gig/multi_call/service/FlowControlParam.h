#pragma once

#include <limits>

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/config/FlowControlConfig.h"
namespace multi_call {

class FlowControlParam
{
public:
    FlowControlParam(const FlowControlConfigPtr &flowControlConfig_)
        : flowControlConfig(flowControlConfig_)
        , partitionCount(0)
        , partitionIndex(0) {
    }
    ~FlowControlParam() {
    }

public:
    FlowControlParam(const FlowControlParam &other) {
        flowControlConfig = other.flowControlConfig;
        partitionCount = other.partitionCount;
        partitionIndex = other.partitionIndex;
    }

public:
    FlowControlConfigPtr flowControlConfig;
    uint32_t partitionCount;
    uint32_t partitionIndex;
    bool disableDegrade = false;
    bool ignoreWeightLabelInConsistentHash = false;

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(FlowControlParam);

} // namespace multi_call

