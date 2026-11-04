#include "rtp_llm/flexlb/navi_sched/resource/AgentInstanceR.h"
#include <autil/TimeUtility.h>
#include <atomic>

namespace navi_sched {
static std::atomic<uint64_t> AGENT_INSTANCE_SEQUENCE{0};

const std::string AgentInstanceR::RESOURCE_ID = "navi_sched.agent_instance_r";

void AgentInstanceR::def(navi::ResourceDefBuilder &builder) const
{
    builder.name(RESOURCE_ID, navi::RS_BIZ);
}

navi::ErrorCode AgentInstanceR::init(navi::ResourceInitContext &)
{
    _meta = std::to_string(autil::TimeUtility::currentTime()) + "_" +
            std::to_string(AGENT_INSTANCE_SEQUENCE.fetch_add(1));
    return navi::EC_NONE;
}

bool AgentInstanceR::getMeta(std::string &meta) const
{
    meta = _meta;
    return !meta.empty();
}

REGISTER_RESOURCE(AgentInstanceR);

} // namespace navi_sched
