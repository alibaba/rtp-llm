#pragma once

#include "rtp_llm/flexlb/navi_sched/common/NaviSchedScheduleData.h"
#include "rtp_llm/flexlb/navi_sched/data/LatencyModelGlobalData.h"
#include "rtp_llm/flexlb/navi_sched/data/MasterDispatchHandle.h"
#include "navi/engine/Data.h"
#include <autil/DataBuffer.h>
#include <string>
#include <vector>

namespace navi_sched {

static const std::string MASTER_RESPONSE_TO_CLIENT_TYPE_ID = "NaviSchedMasterType";

class MasterResponseToClientData : public navi::Data {
public:
    MasterResponseToClientData();
    std::shared_ptr<MasterResponseToClientData> makeCopy() const;
    std::shared_ptr<MasterResponseToClientData> makeForwardResponse() const;
    navi::SummaryErrorCode getSummary(const std::string &query,
                                      navi::SummaryBuilder &builder) const override;
    void serialize(autil::DataBuffer &dataBuffer) const;
    void deserialize(autil::DataBuffer &dataBuffer);

public:
    int32_t version = NAVI_SCHED_DATA_VERSION;
    uint64_t requestId = 0;
    uint64_t scheduleId = 0;
    double scheduleLoss = 0.0;
    int64_t scheduleLatencyUs = 0;
    int64_t scheduleTimeUs = 0;
    std::string reason;
    int64_t batchId = 0;
    int64_t expectedBatchSize = 0;
    int64_t predictedCacheHitTokenCount = 0;
    double predictedQueueLatencyMs = 0.0;
    double predictedExecutionLatencyMs = 0.0;
    NaviSchedLatencyPredictionConstPtr latencyPrediction;
    NaviSchedLatencyTrainingPhase latencyTrainingPhase =
        NaviSchedLatencyTrainingPhase::NORMAL;
    NaviSchedTarget target;
    LatencyModelGlobalData latencyModelGlobal;
    NaviSchedScheduleReplayPtr replay;
    NaviSchedRequestTimingPtr requestTiming;
    MasterDispatchHandle dispatchHandle;
};

NAVI_TYPEDEF_PTR(MasterResponseToClientData);

} // namespace navi_sched
