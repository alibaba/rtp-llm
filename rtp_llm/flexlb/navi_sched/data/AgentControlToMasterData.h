#pragma once

#include "aios/navi_sched/common/NaviSchedScheduleData.h"
#include <autil/DataBuffer.h>
#include "navi/common.h"

namespace navi_sched {

struct AgentControlToMasterData {
public:
    void serialize(autil::DataBuffer &dataBuffer) const;
    void deserialize(autil::DataBuffer &dataBuffer);

public:
    NaviSchedServerInfo serverInfo;
    int64_t agentStartTimeUs = 0;
    bool pdSeparated = true;
    bool unifiedLatencyPrediction = false;
    bool executionGeometryEnabled = false;
    bool projectedExecutionGeometryEnabled = false;
    bool peerCacheEnabled = false;
    TPUPrefillExecutionPolicyConfig prefillExecutionPolicy;
};

NAVI_TYPEDEF_PTR(AgentControlToMasterData);

}
