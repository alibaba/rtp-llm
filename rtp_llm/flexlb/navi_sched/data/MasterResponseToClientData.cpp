#include "rtp_llm/flexlb/navi_sched/data/MasterResponseToClientData.h"
#include "rtp_llm/flexlb/navi_sched/data/SummaryUtil.h"
#include "navi/engine/Type.h"

namespace navi_sched {

MasterResponseToClientData::MasterResponseToClientData()
    : navi::Data(MASTER_RESPONSE_TO_CLIENT_TYPE_ID)
{
}

MasterResponseToClientDataPtr MasterResponseToClientData::makeCopy() const
{
    auto output = std::make_shared<MasterResponseToClientData>();
    output->version = version;
    output->requestId = requestId;
    output->scheduleId = scheduleId;
    output->scheduleLoss = scheduleLoss;
    output->scheduleLatencyUs = scheduleLatencyUs;
    output->scheduleTimeUs = scheduleTimeUs;
    output->reason = reason;
    output->batchId = batchId;
    output->expectedBatchSize = expectedBatchSize;
    output->predictedCacheHitTokenCount = predictedCacheHitTokenCount;
    output->predictedQueueLatencyMs = predictedQueueLatencyMs;
    output->predictedExecutionLatencyMs = predictedExecutionLatencyMs;
    output->latencyPrediction = latencyPrediction;
    output->latencyTrainingPhase = latencyTrainingPhase;
    output->target = target;
    output->latencyModelGlobal = latencyModelGlobal;
    output->replay = replay;
    output->requestTiming = requestTiming;
    output->dispatchHandle = dispatchHandle;
    return output;
}

MasterResponseToClientDataPtr MasterResponseToClientData::makeForwardResponse() const
{
    auto output = makeCopy();
    output->target.agentAddress.clear();
    output->latencyModelGlobal = {};
    output->dispatchHandle = {};
    return output;
}

navi::SummaryErrorCode MasterResponseToClientData::getSummary(
    const std::string &query, navi::SummaryBuilder &builder) const
{
    auto table = builder.newTable("MasterResponseToClientData");
    TABLE_ADD_COLUMN_VALUE(table, "version", version);
    TABLE_ADD_COLUMN_VALUE(table, "request_id", requestId);
    TABLE_ADD_COLUMN_VALUE(table, "schedule_id", scheduleId);
    TABLE_ADD_COLUMN_VALUE(table, "schedule_loss", scheduleLoss);
    TABLE_ADD_COLUMN_VALUE(table, "schedule_latency_us", scheduleLatencyUs);
    TABLE_ADD_COLUMN_VALUE(table, "schedule_time_us", scheduleTimeUs);
    TABLE_ADD_COLUMN_VALUE(table, "batch_id", batchId);
    TABLE_ADD_COLUMN_VALUE(table, "expected_batch_size", expectedBatchSize);
    TABLE_ADD_COLUMN_VALUE(table, "predicted_cache_hit_tokens",
                           predictedCacheHitTokenCount);
    TABLE_ADD_COLUMN_VALUE(table, "predicted_queue_latency_ms",
                           predictedQueueLatencyMs);
    TABLE_ADD_COLUMN_VALUE(table, "predicted_execution_latency_ms",
                           predictedExecutionLatencyMs);
    TABLE_ADD_COLUMN_VALUE(table, "latency_prediction_recomputable",
                           latencyPrediction && latencyPrediction->valid());
    TABLE_ADD_COLUMN_VALUE(
        table, "latency_training_phase",
        static_cast<int32_t>(latencyTrainingPhase));
    TABLE_ADD_COLUMN_VALUE(table, "has_target", !target.address.empty());
    TABLE_ADD_COLUMN_VALUE(table, "latency_model_global_valid",
                           latencyModelGlobal.valid());
    TABLE_ADD_COLUMN_VALUE(table, "latency_model_global_epoch",
                           latencyModelGlobal.masterEpoch);
    TABLE_ADD_COLUMN_VALUE(table, "latency_model_global_version",
                           latencyModelGlobal.version);
    TABLE_ADD_COLUMN_VALUE(table, "latency_model_global_zero_point",
                           latencyModelGlobal.globalZeroPoint);
    TABLE_ADD_COLUMN_VALUE(table, "reason", reason);
    TABLE_ADD_COLUMN_VALUE(table, "has_replay", bool(replay));
    if (SummaryUtil::getSummaryLevel(query) >= 2) {
        auto targetTable = builder.newTable("NaviSchedTarget");
        TABLE_ADD_COLUMN_VALUE(targetTable, "version", target.version);
        TABLE_ADD_COLUMN_VALUE(targetTable, "biz_name", target.bizName);
        TABLE_ADD_COLUMN_VALUE(targetTable, "address", target.address);
        TABLE_ADD_COLUMN_VALUE(targetTable, "agent_address", target.agentAddress);
    }
    return navi::SEC_OK;
}

void MasterResponseToClientData::serialize(autil::DataBuffer &dataBuffer) const
{
    NaviSchedDecision decision;
    decision.version = version;
    decision.requestId = requestId;
    decision.scheduleId = scheduleId;
    decision.scheduleLoss = scheduleLoss;
    decision.scheduleLatencyUs = scheduleLatencyUs;
    decision.scheduleTimeUs = scheduleTimeUs;
    decision.reason = reason;
    decision.batchId = batchId;
    decision.expectedBatchSize = expectedBatchSize;
    decision.predictedCacheHitTokenCount = predictedCacheHitTokenCount;
    decision.predictedQueueLatencyMs = predictedQueueLatencyMs;
    decision.predictedExecutionLatencyMs = predictedExecutionLatencyMs;
    decision.latencyPrediction = latencyPrediction;
    decision.latencyTrainingPhase = latencyTrainingPhase;
    decision.target = target;
    dataBuffer.write(decision);
    dataBuffer.write(latencyModelGlobal);
    dataBuffer.write(bool(replay));
    if (replay) {
        replay->serialize(dataBuffer);
    }
    dataBuffer.write(requestTiming);
}

void MasterResponseToClientData::deserialize(autil::DataBuffer &dataBuffer)
{
    dispatchHandle = {};
    NaviSchedDecision decision;
    dataBuffer.read(decision);
    version = decision.version;
    if (version != NAVI_SCHED_DATA_VERSION) {
        return;
    }
    requestId = decision.requestId;
    scheduleId = decision.scheduleId;
    scheduleLoss = decision.scheduleLoss;
    scheduleLatencyUs = decision.scheduleLatencyUs;
    scheduleTimeUs = decision.scheduleTimeUs;
    reason = std::move(decision.reason);
    batchId = decision.batchId;
    expectedBatchSize = decision.expectedBatchSize;
    predictedCacheHitTokenCount = decision.predictedCacheHitTokenCount;
    predictedQueueLatencyMs = decision.predictedQueueLatencyMs;
    predictedExecutionLatencyMs = decision.predictedExecutionLatencyMs;
    latencyPrediction = std::move(decision.latencyPrediction);
    latencyTrainingPhase = decision.latencyTrainingPhase;
    target = std::move(decision.target);
    dataBuffer.read(latencyModelGlobal);
    bool hasReplay = false;
    dataBuffer.read(hasReplay);
    replay.reset();
    if (hasReplay) {
        auto output = std::make_shared<NaviSchedScheduleReplay>();
        output->deserialize(dataBuffer);
        replay = std::move(output);
    }
    dataBuffer.read(requestTiming);
}

class MasterResponseToClientType : public navi::Type {
public:
    MasterResponseToClientType()
        : navi::Type(NAVI_TYPE_PARAM, MASTER_RESPONSE_TO_CLIENT_TYPE_ID)
    {
    }

public:
    navi::TypeErrorCode serialize(navi::TypeContext &ctx,
                                  const navi::DataPtr &data) const override
    {
        auto response = dynamic_cast<MasterResponseToClientData *>(data.get());
        if (!response) {
            return navi::TEC_TYPE_MISMATCH;
        }
        ctx.getDataBuffer().write(response);
        return navi::TEC_NONE;
    }
    navi::TypeErrorCode deserialize(navi::TypeContext &ctx,
                                    navi::DataPtr &data) const override
    {
        MasterResponseToClientData *response = nullptr;
        ctx.getDataBuffer().read(response);
        if (!response) {
            return navi::TEC_NULL_DATA;
        }
        if (response->version != NAVI_SCHED_DATA_VERSION) {
            delete response;
            return navi::TEC_FAILED;
        }
        data.reset(response);
        return navi::TEC_NONE;
    }
};

REGISTER_TYPE(MasterResponseToClientType);

} // namespace navi_sched
