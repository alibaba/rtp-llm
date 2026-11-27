#pragma once

#include "aios/navi_sched/common/NaviSchedScheduleData.h"
#include <string>
#include <optional>

namespace navi_sched {

class NaviSchedScheduleReplayJson {
public:
    static std::string toJsonString(const NaviSchedScheduleReplay &replay);
    static void fromJsonString(const std::string &json, NaviSchedScheduleReplay &replay);
    // Diagnostic capture contains only the reported workload, never request payloads or cache keys.
    static std::optional<std::string> captureNativeWorkload(
        const NaviSchedWorkloadSnapshot &workload, uint64_t scheduleId,
        bool enabled = false, size_t maxBytes = 256 * 1024);
    static void workloadFromJsonString(const std::string &json,
                                      NaviSchedWorkloadSnapshot &workload);
};

}
