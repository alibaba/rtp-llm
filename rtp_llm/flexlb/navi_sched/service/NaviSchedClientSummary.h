#pragma once

#include "aios/navi_sched/service/NaviSchedRoleSummary.h"
#include <cstddef>
#include <cstdint>
#include <string>

namespace navi_sched {

struct NaviSchedClientSummary : public NaviSchedRoleSummary {
public:
    void Jsonize(JsonWrapper &json) override;
    std::string format() const;

public:
    std::string masterBizName;
    uint64_t requestReceivedCount = 0;
    uint64_t invalidRequestCount = 0;
    uint64_t stoppedRequestCount = 0;
    uint64_t scheduleCount = 0;
    uint64_t scheduleSuccessCount = 0;
    uint64_t scheduleErrorCount = 0;
    uint64_t masterRateLimitedCount = 0;
    uint64_t masterTimeoutCount = 0;
    uint64_t masterTransportErrorCount = 0;
    uint64_t masterRoutingErrorCount = 0;
    uint64_t masterNoDecisionCount = 0;
    uint64_t masterOtherErrorCount = 0;
};

}
