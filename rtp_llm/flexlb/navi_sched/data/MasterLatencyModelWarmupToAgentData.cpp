#include "aios/navi_sched/data/MasterLatencyModelWarmupToAgentData.h"

#include "aios/navi_sched/data/AgentLatencyToMasterData.h"
#include "aios/navi_sched/data/SummaryUtil.h"
#include "navi/engine/Type.h"

#include <cmath>

namespace navi_sched {

void NaviSchedLatencyModelCandidate::serialize(
    autil::DataBuffer &dataBuffer) const
{
    dataBuffer.write(sourceRevision);
    dataBuffer.write(parameters);
    dataBuffer.write(optimizerState);
    dataBuffer.write(residualVersion);
    dataBuffer.write(residualParameters);
    dataBuffer.write(residualLastUpdateTimeUs);
}

void NaviSchedLatencyModelCandidate::deserialize(autil::DataBuffer &dataBuffer)
{
    dataBuffer.read(sourceRevision);
    dataBuffer.read(parameters);
    dataBuffer.read(optimizerState);
    dataBuffer.read(residualVersion);
    dataBuffer.read(residualParameters);
    dataBuffer.read(residualLastUpdateTimeUs);
}

MasterLatencyModelWarmupToAgentData::MasterLatencyModelWarmupToAgentData()
    : navi::Data(MASTER_LATENCY_MODEL_WARMUP_TO_AGENT_TYPE_ID)
{
}

navi::SummaryErrorCode MasterLatencyModelWarmupToAgentData::getSummary(
    const std::string &, navi::SummaryBuilder &builder) const
{
    auto table = builder.newTable("MasterLatencyModelWarmupToAgentData");
    TABLE_ADD_COLUMN_VALUE(table, "version", version);
    TABLE_ADD_COLUMN_VALUE(table, "fingerprint", fingerprint);
    TABLE_ADD_COLUMN_VALUE(table, "formula", formula);
    TABLE_ADD_COLUMN_VALUE(table, "binary_version", binaryVersion);
    TABLE_ADD_COLUMN_VALUE(table, "candidate_count", candidates.size());
    return navi::SEC_OK;
}

void MasterLatencyModelWarmupToAgentData::serialize(
    autil::DataBuffer &dataBuffer) const
{
    dataBuffer.write(version);
    dataBuffer.write(fingerprint);
    dataBuffer.write(formula);
    dataBuffer.write(binaryVersion);
    dataBuffer.write(candidates);
}

void MasterLatencyModelWarmupToAgentData::deserialize(
    autil::DataBuffer &dataBuffer)
{
    dataBuffer.read(version);
    if (version != 3) {
        return;
    }
    dataBuffer.read(fingerprint);
    dataBuffer.read(formula);
    dataBuffer.read(binaryVersion);
    dataBuffer.read(candidates);
}

bool MasterLatencyModelWarmupToAgentData::valid() const
{
    if (version != 3 || fingerprint.empty() || candidates.empty() ||
        fingerprint.size() > 4096 || binaryVersion <= 0 ||
        formula != NAVI_SCHED_LATENCY_FORMULA_NONLINEAR_PREFILL ||
        candidates.size() > 64) {
        return false;
    }
    size_t parameterCount = 0;
    for (const auto &candidate : candidates) {
        if (candidate.sourceRevision <= 0 || candidate.parameters.empty() ||
            candidate.parameters.size() > 64 ||
            (parameterCount != 0 &&
             parameterCount != candidate.parameters.size()) ||
            !candidate.optimizerState.valid(candidate.parameters.size()) ||
            candidate.residualVersion != NAVI_SCHED_LATENCY_RESIDUAL_VERSION ||
            candidate.residualParameters.size() !=
                NAVI_SCHED_LATENCY_RESIDUAL_PARAMETER_COUNT ||
            candidate.residualLastUpdateTimeUs < 0) {
            return false;
        }
        parameterCount = candidate.parameters.size();
        for (const auto parameter : candidate.parameters) {
            if (!std::isfinite(parameter)) {
                return false;
            }
        }
        for (const auto parameter : candidate.residualParameters) {
            if (!std::isfinite(parameter)) {
                return false;
            }
        }
    }
    return true;
}

class MasterLatencyModelWarmupToAgentType : public navi::Type {
public:
    MasterLatencyModelWarmupToAgentType()
        : navi::Type(NAVI_TYPE_PARAM,
                     MASTER_LATENCY_MODEL_WARMUP_TO_AGENT_TYPE_ID)
    {
    }

    navi::TypeErrorCode serialize(navi::TypeContext &ctx,
                                  const navi::DataPtr &data) const override
    {
        auto warmup =
            dynamic_cast<MasterLatencyModelWarmupToAgentData *>(data.get());
        if (!warmup) {
            return navi::TEC_TYPE_MISMATCH;
        }
        ctx.getDataBuffer().write(warmup);
        return navi::TEC_NONE;
    }

    navi::TypeErrorCode deserialize(navi::TypeContext &ctx,
                                    navi::DataPtr &data) const override
    {
        MasterLatencyModelWarmupToAgentData *warmup = nullptr;
        ctx.getDataBuffer().read(warmup);
        if (!warmup || !warmup->valid()) {
            delete warmup;
            return navi::TEC_FAILED;
        }
        data.reset(warmup);
        return navi::TEC_NONE;
    }
};

REGISTER_TYPE(MasterLatencyModelWarmupToAgentType);

}
