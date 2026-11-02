#pragma once

#include "navi/engine/Data.h"

#include <autil/DataBuffer.h>
#include <cstdint>
#include <string>

namespace navi_sched {

static const int32_t NAVI_SCHED_MASTER_CONFIG_VERSION = 1;
static const std::string MASTER_SCHEDULER_CONFIG_TO_AGENT_TYPE_ID =
    "NaviSchedMasterSchedulerConfigType";

class MasterSchedulerConfigToAgentData : public navi::Data {
public:
    MasterSchedulerConfigToAgentData();
    navi::SummaryErrorCode getSummary(const std::string &query,
                                      navi::SummaryBuilder &builder) const override;
    void serialize(autil::DataBuffer &dataBuffer) const;
    void deserialize(autil::DataBuffer &dataBuffer);
    bool valid() const;

public:
    int32_t version = NAVI_SCHED_MASTER_CONFIG_VERSION;
    uint64_t generation = 0;
    std::string configHash;
    std::string configJson;
};

NAVI_TYPEDEF_PTR(MasterSchedulerConfigToAgentData);

}
