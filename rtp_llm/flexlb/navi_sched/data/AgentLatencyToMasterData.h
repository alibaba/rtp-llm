#pragma once

#include "rtp_llm/flexlb/navi_sched/common/NaviSchedCommon.h"

#include <autil/DataBuffer.h>
#include "navi/common.h"
#include <string>
#include <vector>

namespace navi_sched {

enum NaviSchedLatencyFormula : int32_t {
    NAVI_SCHED_LATENCY_FORMULA_UNKNOWN = 0,
    NAVI_SCHED_LATENCY_FORMULA_NONLINEAR_PREFILL = 1,
};

std::string getNaviSchedLatencyFormulaName(int32_t formula);

struct NaviSchedLatencyOptimizerState {
public:
    void serialize(autil::DataBuffer &dataBuffer) const;
    void deserialize(autil::DataBuffer &dataBuffer);
    bool valid(size_t parameterCount) const;

public:
    std::vector<double> adamMoment1;
    std::vector<double> adamMoment2;
    int64_t step = 0;
};

struct AgentLatencyToMasterData {
public:
    void serialize(autil::DataBuffer &dataBuffer) const;
    void deserialize(autil::DataBuffer &dataBuffer);

public:
    int32_t formula = 0;
    int64_t binaryVersion = 0;
    int64_t modelRevision = 0;
    std::vector<double> parameters;
    NaviSchedLatencyOptimizerState optimizerState;
    int32_t residualVersion = NAVI_SCHED_LATENCY_RESIDUAL_VERSION;
    std::vector<double> residualParameters =
        std::vector<double>(NAVI_SCHED_LATENCY_RESIDUAL_PARAMETER_COUNT, 0.0);
    int64_t residualLastUpdateTimeUs = 0;
    double zeroPoint = 0.0;
    uint64_t sessionSampleCount = 0;
};

NAVI_TYPEDEF_PTR(AgentLatencyToMasterData);

} // namespace navi_sched
