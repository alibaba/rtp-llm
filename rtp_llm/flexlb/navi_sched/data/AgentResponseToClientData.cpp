#include "aios/navi_sched/data/AgentResponseToClientData.h"
#include "navi/engine/Type.h"

namespace navi_sched {
const std::string AGENT_RESPONSE_TO_CLIENT_TYPE_ID = "NaviSchedAgentResponseType";

AgentResponseToClientData::AgentResponseToClientData()
    : navi::Data(AGENT_RESPONSE_TO_CLIENT_TYPE_ID)
{
}

void AgentResponseToClientData::serialize(autil::DataBuffer &dataBuffer) const
{
    dataBuffer.write(requestId);
    dataBuffer.write(static_cast<uint8_t>(kind));
    dataBuffer.write(terminal);
    switch (kind) {
    case AgentResponseKind::ACK:
        dataBuffer.write(requestTiming);
        break;
    case AgentResponseKind::OUTPUT:
        dataBuffer.write(frames);
        break;
    case AgentResponseKind::ERROR:
        dataBuffer.write(errorMessage);
        break;
    }
}

void AgentResponseToClientData::deserialize(autil::DataBuffer &dataBuffer)
{
    uint8_t rawKind = 0;
    dataBuffer.read(requestId);
    dataBuffer.read(rawKind);
    kind = static_cast<AgentResponseKind>(rawKind);
    dataBuffer.read(terminal);
    frames.clear();
    errorMessage.clear();
    requestTiming = {};
    switch (kind) {
    case AgentResponseKind::ACK:
        dataBuffer.read(requestTiming);
        break;
    case AgentResponseKind::OUTPUT:
        dataBuffer.read(frames);
        break;
    case AgentResponseKind::ERROR:
        dataBuffer.read(errorMessage);
        break;
    }
}

bool AgentResponseToClientData::valid() const
{
    if (requestId == 0) {
        return false;
    }
    switch (kind) {
    case AgentResponseKind::ACK:
        return frames.empty() && errorMessage.empty();
    case AgentResponseKind::OUTPUT:
        return !frames.empty() && errorMessage.empty();
    case AgentResponseKind::ERROR:
        return terminal && frames.empty() && !errorMessage.empty();
    }
    return false;
}

navi::SummaryErrorCode AgentResponseToClientData::getSummary(
    const std::string &, navi::SummaryBuilder &builder) const
{
    size_t payloadSize = 0;
    for (const auto &frame : frames) {
        payloadSize += frame.size();
    }
    auto table = builder.newTable("AgentResponseToClientData");
    TABLE_ADD_COLUMN_VALUE(table, "request_id", requestId);
    TABLE_ADD_COLUMN_VALUE(table, "kind", static_cast<uint32_t>(kind));
    TABLE_ADD_COLUMN_VALUE(table, "frame_count", frames.size());
    TABLE_ADD_COLUMN_VALUE(table, "payload_size", payloadSize);
    TABLE_ADD_COLUMN_VALUE(table, "terminal", terminal);
    TABLE_ADD_COLUMN_VALUE(table, "error_message", errorMessage);
    return navi::SEC_OK;
}

class AgentResponseToClientType : public navi::Type {
public:
    AgentResponseToClientType()
        : navi::Type(NAVI_TYPE_PARAM, AGENT_RESPONSE_TO_CLIENT_TYPE_ID)
    {
    }

public:
    navi::TypeErrorCode serialize(navi::TypeContext &ctx,
                                  const navi::DataPtr &data) const override
    {
        auto response = dynamic_cast<AgentResponseToClientData *>(data.get());
        if (!response) {
            return navi::TEC_TYPE_MISMATCH;
        }
        ctx.getDataBuffer().write(response);
        return navi::TEC_NONE;
    }

    navi::TypeErrorCode deserialize(navi::TypeContext &ctx,
                                    navi::DataPtr &data) const override
    {
        AgentResponseToClientData *response = nullptr;
        ctx.getDataBuffer().read(response);
        if (!response) {
            return navi::TEC_NULL_DATA;
        }
        data.reset(response);
        return navi::TEC_NONE;
    }
};

REGISTER_TYPE(AgentResponseToClientType);

}
