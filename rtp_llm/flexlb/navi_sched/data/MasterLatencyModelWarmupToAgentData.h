#pragma once

#include "rtp_llm/flexlb/navi_sched/data/AgentLatencyToMasterData.h"
#include "navi/engine/Data.h"

#include <autil/DataBuffer.h>
#include <cstdint>
#include <string>
#include <vector>

namespace navi_sched {

static const std::string MASTER_LATENCY_MODEL_WARMUP_TO_AGENT_TYPE_ID =
    "NaviSchedMasterLatencyModelWarmupType";

struct NaviSchedLatencyModelCandidate {
    int64_t sourceRevision = 0;
    std::vector<double> parameters;
    NaviSchedLatencyOptimizerState optimizerState;
    int32_t residualVersion = NAVI_SCHED_LATENCY_RESIDUAL_VERSION;
    std::vector<double> residualParameters =
        std::vector<double>(NAVI_SCHED_LATENCY_RESIDUAL_PARAMETER_COUNT, 0.0);
    int64_t residualLastUpdateTimeUs = 0;

    void serialize(autil::DataBuffer &dataBuffer) const;
    void deserialize(autil::DataBuffer &dataBuffer);
};

class MasterLatencyModelWarmupToAgentData : public navi::Data {
public:
    MasterLatencyModelWarmupToAgentData();
    navi::SummaryErrorCode getSummary(
        const std::string &query, navi::SummaryBuilder &builder) const override;
    void serialize(autil::DataBuffer &dataBuffer) const;
    void deserialize(autil::DataBuffer &dataBuffer);
    bool valid() const;

public:
    int32_t version = 3;
    std::string fingerprint;
    int32_t formula = 0;
    int64_t binaryVersion = 0;
    std::vector<NaviSchedLatencyModelCandidate> candidates;
};

NAVI_TYPEDEF_PTR(MasterLatencyModelWarmupToAgentData);

} // namespace navi_sched
