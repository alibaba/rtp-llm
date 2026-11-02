#pragma once

#include "navi/engine/Data.h"
#include <autil/DataBuffer.h>
#include <string>

namespace navi_sched {

static const std::string MASTER_STOP_ACK_TO_AGENT_TYPE_ID = "NaviSchedMasterStopAckType";

class MasterStopAckToAgentData : public navi::Data {
public:
    MasterStopAckToAgentData();
    navi::SummaryErrorCode getSummary(const std::string &query,
                                      navi::SummaryBuilder &builder) const override;
    void serialize(autil::DataBuffer &dataBuffer) const;
    void deserialize(autil::DataBuffer &dataBuffer);

public:
    std::string collectorId;
};

NAVI_TYPEDEF_PTR(MasterStopAckToAgentData);

}
