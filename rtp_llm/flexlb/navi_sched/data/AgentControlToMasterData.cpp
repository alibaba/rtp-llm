#include "rtp_llm/flexlb/navi_sched/data/AgentControlToMasterData.h"

namespace navi_sched {

void AgentControlToMasterData::serialize(autil::DataBuffer &dataBuffer) const
{
    dataBuffer.write(serverInfo);
    dataBuffer.write(agentStartTimeUs);
    dataBuffer.write(pdSeparated);
    dataBuffer.write(unifiedLatencyPrediction);
}

void AgentControlToMasterData::deserialize(autil::DataBuffer &dataBuffer)
{
    dataBuffer.read(serverInfo);
    dataBuffer.read(agentStartTimeUs);
    dataBuffer.read(pdSeparated);
    dataBuffer.read(unifiedLatencyPrediction);
}

} // namespace navi_sched
