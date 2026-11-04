#pragma once

#include "rtp_llm/flexlb/navi_sched/scheduler/ScheduleContext.h"
#include "navi/engine/Data.h"

namespace navi_sched {

extern const std::string MASTER_SCHEDULE_CONTEXT_TYPE_ID;

class MasterScheduleContextData : public navi::Data {
public:
    explicit MasterScheduleContextData(NaviSchedScheduleContextPtr context =
                                           std::make_shared<NaviSchedScheduleContext>());
    static NaviSchedScheduleContextPtr getContext(const navi::DataPtr &data);

public:
    NaviSchedScheduleContextPtr context;
};

NAVI_TYPEDEF_PTR(MasterScheduleContextData);

} // namespace navi_sched
