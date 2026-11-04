#include "rtp_llm/flexlb/navi_sched/scheduler/MasterScheduleContextData.h"

#include "navi/engine/Type.h"

namespace navi_sched {

const std::string MASTER_SCHEDULE_CONTEXT_TYPE_ID = "NaviSchedMasterScheduleContextType";

MasterScheduleContextData::MasterScheduleContextData(NaviSchedScheduleContextPtr context)
    : navi::Data(MASTER_SCHEDULE_CONTEXT_TYPE_ID), context(std::move(context))
{
}

NaviSchedScheduleContextPtr MasterScheduleContextData::getContext(
    const navi::DataPtr &data)
{
    auto contextData = std::dynamic_pointer_cast<MasterScheduleContextData>(data);
    return contextData ? contextData->context : nullptr;
}

class MasterScheduleContextType : public navi::Type {
public:
    MasterScheduleContextType()
        : navi::Type(NAVI_TYPE_PARAM, MASTER_SCHEDULE_CONTEXT_TYPE_ID)
    {
    }
};

REGISTER_TYPE(MasterScheduleContextType);

} // namespace navi_sched
