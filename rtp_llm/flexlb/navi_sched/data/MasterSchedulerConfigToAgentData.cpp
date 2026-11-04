#include "aios/navi_sched/data/MasterSchedulerConfigToAgentData.h"

#include "navi/engine/Type.h"

namespace navi_sched {

MasterSchedulerConfigToAgentData::MasterSchedulerConfigToAgentData()
    : navi::Data(MASTER_SCHEDULER_CONFIG_TO_AGENT_TYPE_ID)
{
}

navi::SummaryErrorCode MasterSchedulerConfigToAgentData::getSummary(
    const std::string &, navi::SummaryBuilder &builder) const
{
    auto table = builder.newTable("MasterSchedulerConfigToAgentData");
    TABLE_ADD_COLUMN_VALUE(table, "version", version);
    TABLE_ADD_COLUMN_VALUE(table, "generation", generation);
    TABLE_ADD_COLUMN_VALUE(table, "config_hash", configHash);
    TABLE_ADD_COLUMN_VALUE(table, "config_json", configJson);
    return navi::SEC_OK;
}

void MasterSchedulerConfigToAgentData::serialize(
    autil::DataBuffer &dataBuffer) const
{
    dataBuffer.write(version);
    dataBuffer.write(generation);
    dataBuffer.write(configHash);
    dataBuffer.write(configJson);
}

void MasterSchedulerConfigToAgentData::deserialize(
    autil::DataBuffer &dataBuffer)
{
    dataBuffer.read(version);
    if (version != NAVI_SCHED_MASTER_CONFIG_VERSION) {
        return;
    }
    dataBuffer.read(generation);
    dataBuffer.read(configHash);
    dataBuffer.read(configJson);
}

bool MasterSchedulerConfigToAgentData::valid() const
{
    return version == NAVI_SCHED_MASTER_CONFIG_VERSION && generation > 0 &&
           !configHash.empty() && !configJson.empty();
}

class MasterSchedulerConfigToAgentType : public navi::Type {
public:
    MasterSchedulerConfigToAgentType()
        : navi::Type(NAVI_TYPE_PARAM,
                     MASTER_SCHEDULER_CONFIG_TO_AGENT_TYPE_ID)
    {
    }

    navi::TypeErrorCode serialize(navi::TypeContext &ctx,
                                  const navi::DataPtr &data) const override
    {
        auto config =
            dynamic_cast<MasterSchedulerConfigToAgentData *>(data.get());
        if (!config) {
            return navi::TEC_TYPE_MISMATCH;
        }
        ctx.getDataBuffer().write(config);
        return navi::TEC_NONE;
    }

    navi::TypeErrorCode deserialize(navi::TypeContext &ctx,
                                    navi::DataPtr &data) const override
    {
        MasterSchedulerConfigToAgentData *config = nullptr;
        ctx.getDataBuffer().read(config);
        if (!config) {
            return navi::TEC_NULL_DATA;
        }
        data.reset(config);
        return navi::TEC_NONE;
    }
};

REGISTER_TYPE(MasterSchedulerConfigToAgentType);

}
