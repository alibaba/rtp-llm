#pragma once

#include "rtp_llm/flexlb/navi_sched/common/NaviSchedScheduleData.h"
#include "navi/engine/Data.h"
#include <cstdint>
#include <string>
#include <vector>

namespace navi_sched {

extern const std::string AGENT_RESPONSE_TO_CLIENT_TYPE_ID;

enum class AgentResponseKind : uint8_t {
    ACK = 0,
    OUTPUT = 1,
    ERROR = 2,
};

class AgentResponseToClientData : public navi::Data {
public:
    AgentResponseToClientData();

    void serialize(autil::DataBuffer &dataBuffer) const;
    void deserialize(autil::DataBuffer &dataBuffer);
    bool valid() const;
    navi::SummaryErrorCode getSummary(const std::string &query,
                                      navi::SummaryBuilder &builder) const override;

public:
    uint64_t requestId = 0;
    AgentResponseKind kind = AgentResponseKind::ACK;
    std::vector<std::string> frames;
    bool terminal = false;
    std::string errorMessage;
    NaviSchedRequestTiming requestTiming;
};

using AgentResponseToClientDataPtr = std::shared_ptr<AgentResponseToClientData>;

} // namespace navi_sched
