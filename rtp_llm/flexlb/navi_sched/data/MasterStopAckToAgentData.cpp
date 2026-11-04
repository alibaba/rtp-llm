#include "aios/navi_sched/data/MasterStopAckToAgentData.h"
#include "navi/engine/Type.h"

namespace navi_sched {

MasterStopAckToAgentData::MasterStopAckToAgentData()
    : navi::Data(MASTER_STOP_ACK_TO_AGENT_TYPE_ID)
{
}

navi::SummaryErrorCode MasterStopAckToAgentData::getSummary(
    const std::string &, navi::SummaryBuilder &builder) const
{
    auto table = builder.newTable("MasterStopAckToAgentData");
    TABLE_ADD_COLUMN_VALUE(table, "collector_id", collectorId);
    return navi::SEC_OK;
}

void MasterStopAckToAgentData::serialize(autil::DataBuffer &dataBuffer) const
{
    dataBuffer.write(collectorId);
}

void MasterStopAckToAgentData::deserialize(autil::DataBuffer &dataBuffer)
{
    dataBuffer.read(collectorId);
}

class MasterStopAckToAgentType : public navi::Type {
public:
    MasterStopAckToAgentType()
        : navi::Type(NAVI_TYPE_PARAM, MASTER_STOP_ACK_TO_AGENT_TYPE_ID)
    {
    }

public:
    navi::TypeErrorCode serialize(navi::TypeContext &ctx,
                                  const navi::DataPtr &data) const override
    {
        auto ack = dynamic_cast<MasterStopAckToAgentData *>(data.get());
        if (!ack) {
            return navi::TEC_TYPE_MISMATCH;
        }
        ctx.getDataBuffer().write(ack);
        return navi::TEC_NONE;
    }

    navi::TypeErrorCode deserialize(navi::TypeContext &ctx,
                                    navi::DataPtr &data) const override
    {
        MasterStopAckToAgentData *ack = nullptr;
        ctx.getDataBuffer().read(ack);
        if (!ack) {
            return navi::TEC_NULL_DATA;
        }
        data.reset(ack);
        return navi::TEC_NONE;
    }
};

REGISTER_TYPE(MasterStopAckToAgentType);

}
