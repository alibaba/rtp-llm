#pragma once

#include "rtp_llm/flexlb/navi_sched/common/NaviSchedScheduleData.h"
#include <string>

namespace navi_sched {

class NaviSchedScheduleReplayJson {
public:
    static std::string toJsonString(const NaviSchedScheduleReplay &replay);
    static void fromJsonString(const std::string &json, NaviSchedScheduleReplay &replay);
};

} // namespace navi_sched
