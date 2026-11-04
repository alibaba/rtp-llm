#include "rtp_llm/flexlb/navi_sched/data/AgentUpdateToMasterData.h"
#include "rtp_llm/flexlb/navi_sched/data/BlockHashUtil.h"
#include "rtp_llm/flexlb/navi_sched/data/SummaryUtil.h"
#include "autil/Scope.h"
#include "autil/TimeUtility.h"
#include "navi/engine/Type.h"
#include <algorithm>
#include <cmath>

namespace navi_sched {

AgentUpdateToMasterData::AgentUpdateToMasterData()
    : navi::Data(AGENT_UPDATE_TO_MASTER_TYPE_ID)
{
}

navi::SummaryErrorCode AgentUpdateToMasterData::getSummary(
    const std::string &query, navi::SummaryBuilder &builder) const
{
    auto table = builder.newTable("AgentUpdateToMasterData");
    TABLE_ADD_COLUMN_VALUE(table, "version", version);
    TABLE_ADD_COLUMN_VALUE(table, "event", static_cast<int32_t>(event));
    TABLE_ADD_COLUMN_VALUE(table, "event_type", NaviSchedAgentEvent::getName(event));
    TABLE_ADD_COLUMN_VALUE(table, "event_time_us", eventTimeUs);
    TABLE_ADD_COLUMN_VALUE(table, "deserialize_begin_mono_us", deserializeBeginMonoUs);
    TABLE_ADD_COLUMN_VALUE(table, "deserialize_end_mono_us", deserializeEndMonoUs);
    TABLE_ADD_COLUMN_VALUE(table, "has_control", control != nullptr);
    TABLE_ADD_COLUMN_VALUE(table, "has_request_event", requestEvent != nullptr);
    TABLE_ADD_COLUMN_VALUE(table, "has_cache", cache != nullptr);
    TABLE_ADD_COLUMN_VALUE(table, "has_pending_cache", pendingCache != nullptr);
    TABLE_ADD_COLUMN_VALUE(table, "has_latency", latency != nullptr);
    TABLE_ADD_COLUMN_VALUE(table, "has_workload", workload != nullptr);
    TABLE_ADD_COLUMN_VALUE(table, "latency_model_fingerprint",
                           latencyModelFingerprint);
    TABLE_ADD_COLUMN_VALUE(table, "latency_model_trusted",
                           latencyModelTrusted);
    TABLE_ADD_COLUMN_VALUE(table, "latency_model_warm_started",
                           latencyModelWarmStarted);
    TABLE_ADD_COLUMN_VALUE(table, "scheduler_config_request_generation",
                           schedulerConfigRequestGeneration);
    TABLE_ADD_COLUMN_VALUE(table, "scheduler_config_generation",
                           schedulerConfigGeneration);
    TABLE_ADD_COLUMN_VALUE(table, "scheduler_config_hash",
                           schedulerConfigHash);
    TABLE_ADD_COLUMN_VALUE(table, "scheduler_config_json", schedulerConfigJson);
    TABLE_ADD_COLUMN_VALUE(table, "scheduler_config_error", schedulerConfigError);
    if (SummaryUtil::getSummaryLevel(query) < 1) {
        return navi::SEC_OK;
    }
    if (control) {
        auto section = builder.newTable("AgentControlToMasterData");
        TABLE_ADD_COLUMN_VALUE(section, "server_ip", control->serverInfo.ip);
        TABLE_ADD_COLUMN_VALUE(section, "server_port", control->serverInfo.port);
        TABLE_ADD_COLUMN_VALUE(section, "agent_start_time_us",
                               control->agentStartTimeUs);
        TABLE_ADD_COLUMN_VALUE(section, "pd_separated", control->pdSeparated);
        TABLE_ADD_COLUMN_VALUE(section, "unified_latency_prediction",
                               control->unifiedLatencyPrediction);
    }
    if (requestEvent) {
        auto section = builder.newTable("AgentRequestEventToMasterData");
        TABLE_ADD_COLUMN_VALUE(section, "request_id", requestEvent->requestId);
        TABLE_ADD_COLUMN_VALUE(section, "inflight_stats_revision",
                               requestEvent->inflightStatsRevision);
        TABLE_ADD_COLUMN_VALUE(section, "inflight_requests",
                               requestEvent->inflightRequests);
        TABLE_ADD_COLUMN_VALUE(section, "inflight_predicted_latency_sum_ms",
                               requestEvent->inflightPredictedLatencySumMs);
        TABLE_ADD_COLUMN_VALUE(section, "batch_waiting_requests",
                               requestEvent->batchWaitingRequests);
        TABLE_ADD_COLUMN_VALUE(section, "queued_batch_requests",
                               requestEvent->queuedBatchRequests);
        TABLE_ADD_COLUMN_VALUE(section, "running_requests",
                               requestEvent->runningRequests);
        TABLE_ADD_COLUMN_VALUE(section, "schedule_batch_id",
                               requestEvent->localBatch.scheduleBatchId);
        TABLE_ADD_COLUMN_VALUE(section, "local_batch_id",
                               requestEvent->localBatch.localBatchId);
        TABLE_ADD_COLUMN_VALUE(section, "local_batch_request_count",
                               requestEvent->localBatch.requestCount);
    }
    if (cache) {
        auto section = builder.newTable("AgentCacheToMasterData");
        TABLE_ADD_COLUMN_VALUE(section, "total_blocks", cache->totalBlocks);
        TABLE_ADD_COLUMN_VALUE(section, "full_snapshot", cache->fullSnapshot);
        TABLE_ADD_COLUMN_VALUE(section, "added_block_hash_count",
                               BlockHashUtil::size(cache->addedBlockHashes));
        TABLE_ADD_COLUMN_VALUE(section, "removed_block_hash_count",
                               BlockHashUtil::size(cache->removedBlockHashes));
        TABLE_ADD_COLUMN_VALUE(section, "added_full_ready_block_hash_count",
                               BlockHashUtil::size(cache->addedFullReadyBlockHashes));
        TABLE_ADD_COLUMN_VALUE(section, "removed_full_ready_block_hash_count",
                               BlockHashUtil::size(cache->removedFullReadyBlockHashes));
        TABLE_ADD_COLUMN_VALUE(section, "added_tail_ready_block_hash_count",
                               BlockHashUtil::size(cache->addedTailReadyBlockHashes));
        TABLE_ADD_COLUMN_VALUE(section, "removed_tail_ready_block_hash_count",
                               BlockHashUtil::size(cache->removedTailReadyBlockHashes));
        TABLE_ADD_COLUMN_VALUE(section, "block_size", cache->blockSize);
        TABLE_ADD_COLUMN_VALUE(section, "linear_step", cache->linearStep);
        TABLE_ADD_COLUMN_VALUE(section, "hybrid_tail_required",
                               cache->hybridTailRequired);
        TABLE_ADD_COLUMN_VALUE(section, "cache_revision", cache->revision);
        TABLE_ADD_COLUMN_VALUE(section, "cached_block_count", cache->cachedBlockCount);
        TABLE_ADD_COLUMN_VALUE(section, "cache_hash_checksum", cache->hashChecksum);
        TABLE_ADD_COLUMN_VALUE(section, "full_ready_block_count",
                               cache->fullReadyBlockCount);
        TABLE_ADD_COLUMN_VALUE(section, "full_ready_hash_checksum",
                               cache->fullReadyHashChecksum);
        TABLE_ADD_COLUMN_VALUE(section, "tail_ready_block_count",
                               cache->tailReadyBlockCount);
        TABLE_ADD_COLUMN_VALUE(section, "tail_ready_hash_checksum",
                               cache->tailReadyHashChecksum);
        TABLE_ADD_COLUMN_VALUE(section, "rtp_cached_block_count",
                               cache->rtpCachedBlockCount);
    }
    if (pendingCache) {
        auto section = builder.newTable("AgentPendingCacheToMasterData");
        TABLE_ADD_COLUMN_VALUE(section, "full_snapshot", pendingCache->fullSnapshot);
        TABLE_ADD_COLUMN_VALUE(section, "added_batch_count",
                               pendingCache->addedReservations.size());
        TABLE_ADD_COLUMN_VALUE(section, "removed_batch_count",
                               pendingCache->removedBatchIds.size());
        TABLE_ADD_COLUMN_VALUE(section, "revision", pendingCache->revision);
        TABLE_ADD_COLUMN_VALUE(section, "batch_count", pendingCache->batchCount);
        TABLE_ADD_COLUMN_VALUE(section, "unique_block_count",
                               pendingCache->uniqueBlockCount);
        TABLE_ADD_COLUMN_VALUE(section, "unique_tail_ready_block_count",
                               pendingCache->uniqueTailReadyBlockCount);
        TABLE_ADD_COLUMN_VALUE(section, "checksum", pendingCache->checksum);
    }
    if (latency) {
        auto section = builder.newTable("AgentLatencyToMasterData");
        TABLE_ADD_COLUMN_VALUE(section, "formula_id", latency->formula);
        TABLE_ADD_COLUMN_VALUE(section, "formula_name",
                               getNaviSchedLatencyFormulaName(latency->formula));
        TABLE_ADD_COLUMN_VALUE(section, "binary_version", latency->binaryVersion);
        TABLE_ADD_COLUMN_VALUE(section, "model_revision", latency->modelRevision);
        TABLE_ADD_COLUMN_VALUE(section, "parameter_count", latency->parameters.size());
        TABLE_ADD_COLUMN_VALUE(section, "zero_point", latency->zeroPoint);
        TABLE_ADD_COLUMN_VALUE(section, "session_sample_count",
                               latency->sessionSampleCount);
    }
    if (workload) {
        auto section = builder.newTable("NaviSchedWorkloadSnapshot");
        TABLE_ADD_COLUMN_VALUE(section, "revision", workload->revision);
        TABLE_ADD_COLUMN_VALUE(section, "event_time_us", workload->eventTimeUs);
        TABLE_ADD_COLUMN_VALUE(section, "batch_count", workload->batchCount);
        TABLE_ADD_COLUMN_VALUE(section, "request_count", workload->requestCount);
        TABLE_ADD_COLUMN_VALUE(section, "remaining_prefill_request_count",
                               workload->remainingPrefillRequestCount);
        TABLE_ADD_COLUMN_VALUE(section, "running_request_count",
                               workload->runningRequestCount);
        TABLE_ADD_COLUMN_VALUE(section, "running_elapsed_time_ms",
                               workload->runningElapsedTimeMs);
        TABLE_ADD_COLUMN_VALUE(section, "running_decode_request_count",
                               workload->runningDecodeRequestCount);
        TABLE_ADD_COLUMN_VALUE(section, "running_decode_generated_tokens",
                               workload->runningDecodeGeneratedTokenCount);
        TABLE_ADD_COLUMN_VALUE(section, "running_decode_remaining_token_budget",
                               workload->runningDecodeRemainingTokenBudget);
        TABLE_ADD_COLUMN_VALUE(section, "waiting_request_count",
                               workload->waitingRequestCount);
        TABLE_ADD_COLUMN_VALUE(section, "remaining_prefill_context_tokens",
                               workload->remainingPrefillContextTokenCount);
        TABLE_ADD_COLUMN_VALUE(section, "remaining_prefill_tokens",
                               workload->remainingPrefillTokenCount);
        TABLE_ADD_COLUMN_VALUE(section, "remaining_prefill_token_square_sum",
                               workload->remainingPrefillTokenSquareSum);
        TABLE_ADD_COLUMN_VALUE(
            section, "context_remaining_prefill_token_product_sum",
            workload->contextRemainingPrefillTokenProductSum);
        TABLE_ADD_COLUMN_VALUE(section, "prefill_capacity_split",
                               workload->prefillCapacitySplit);
        TABLE_ADD_COLUMN_VALUE(section, "max_running_request_count",
                               workload->maxRunningRequestCount);
        TABLE_ADD_COLUMN_VALUE(section, "running_capacity_slot_count",
                               workload->runningCapacitySlotCount);
        TABLE_ADD_COLUMN_VALUE(
            section, "running_capacity_prefill_request_count",
            workload->runningCapacityPrefillRequestCount);
        TABLE_ADD_COLUMN_VALUE(
            section, "running_capacity_prefill_context_tokens",
            workload->runningCapacityPrefillContextTokenCount);
        TABLE_ADD_COLUMN_VALUE(section, "running_capacity_prefill_tokens",
                               workload->runningCapacityPrefillTokenCount);
        TABLE_ADD_COLUMN_VALUE(
            section, "running_capacity_prefill_token_square_sum",
            workload->runningCapacityPrefillTokenSquareSum);
        TABLE_ADD_COLUMN_VALUE(
            section, "running_capacity_prefill_product_sum",
            workload
                ->runningCapacityContextRemainingPrefillTokenProductSum);
        TABLE_ADD_COLUMN_VALUE(
            section, "waiting_capacity_prefill_request_count",
            workload->waitingCapacityPrefillRequestCount);
        TABLE_ADD_COLUMN_VALUE(
            section, "waiting_capacity_prefill_context_tokens",
            workload->waitingCapacityPrefillContextTokenCount);
        TABLE_ADD_COLUMN_VALUE(section, "waiting_capacity_prefill_tokens",
                               workload->waitingCapacityPrefillTokenCount);
        TABLE_ADD_COLUMN_VALUE(
            section, "waiting_capacity_prefill_token_square_sum",
            workload->waitingCapacityPrefillTokenSquareSum);
        TABLE_ADD_COLUMN_VALUE(
            section, "waiting_capacity_prefill_product_sum",
            workload
                ->waitingCapacityContextRemainingPrefillTokenProductSum);
        TABLE_ADD_COLUMN_VALUE(section, "remaining_total_context_tokens",
                               workload->remainingTotalContextTokenCount);
        TABLE_ADD_COLUMN_VALUE(section, "remaining_total_tokens",
                               workload->remainingTotalTokenCount);
        TABLE_ADD_COLUMN_VALUE(section, "remaining_total_token_square_sum",
                               workload->remainingTotalTokenSquareSum);
        TABLE_ADD_COLUMN_VALUE(
            section, "context_remaining_total_token_product_sum",
            workload->contextRemainingTotalTokenProductSum);
        TABLE_ADD_COLUMN_VALUE(section, "total_prompt_tokens",
                               workload->totalPromptTokenCount);
        TABLE_ADD_COLUMN_VALUE(section, "inflight_scheduled_tokens",
                               workload->inFlightScheduledTokenCount);
        TABLE_ADD_COLUMN_VALUE(section, "admitted_schedule_batch_id",
                               workload->admittedBatch.scheduleBatchId);
        TABLE_ADD_COLUMN_VALUE(section, "admitted_local_batch_id",
                               workload->admittedBatch.localBatchId);
        TABLE_ADD_COLUMN_VALUE(section, "admitted_request_count",
                               workload->admittedBatch.requestCount);
    }
    return navi::SEC_OK;
}

void AgentUpdateToMasterData::serialize(autil::DataBuffer &dataBuffer) const
{
    dataBuffer.write(version);
    dataBuffer.write(event);
    dataBuffer.write(eventTimeUs);
    dataBuffer.write(control);
    dataBuffer.write(requestEvent);
    dataBuffer.write(cache);
    dataBuffer.write(pendingCache);
    dataBuffer.write(latency);
    dataBuffer.write(workload);
    dataBuffer.write(latencyModelFingerprint);
    dataBuffer.write(latencyModelTrusted);
    dataBuffer.write(latencyModelWarmStarted);
    dataBuffer.write(schedulerConfigRequestGeneration);
    dataBuffer.write(schedulerConfigGeneration);
    dataBuffer.write(schedulerConfigHash);
    dataBuffer.write(schedulerConfigJson);
    dataBuffer.write(schedulerConfigError);
}

void AgentUpdateToMasterData::deserialize(autil::DataBuffer &dataBuffer)
{
    deserializeBeginMonoUs = autil::TimeUtility::monotonicTimeUs();
    deserializeEndMonoUs = 0;
    autil::ScopeGuard finishReceiveTiming([this]() {
        deserializeEndMonoUs = autil::TimeUtility::monotonicTimeUs();
    });
    dataBuffer.read(version);
    if (version < NAVI_SCHED_AGENT_UPDATE_MIN_SUPPORTED_VERSION ||
        version > NAVI_SCHED_AGENT_UPDATE_VERSION) {
        return;
    }
    dataBuffer.read(event);
    dataBuffer.read(eventTimeUs);
    dataBuffer.read(control);
    dataBuffer.read(requestEvent);
    dataBuffer.read(cache);
    dataBuffer.read(pendingCache);
    dataBuffer.read(latency);
    dataBuffer.read(workload);
    dataBuffer.read(latencyModelFingerprint);
    dataBuffer.read(latencyModelTrusted);
    dataBuffer.read(latencyModelWarmStarted);
    if (version >= 8) {
        dataBuffer.read(schedulerConfigRequestGeneration);
        dataBuffer.read(schedulerConfigGeneration);
        dataBuffer.read(schedulerConfigHash);
        dataBuffer.read(schedulerConfigJson);
        dataBuffer.read(schedulerConfigError);
    }
}

bool AgentUpdateToMasterData::valid() const
{
    if (version < NAVI_SCHED_AGENT_UPDATE_MIN_SUPPORTED_VERSION ||
        version > NAVI_SCHED_AGENT_UPDATE_VERSION ||
        NaviSchedAgentEvent::UNKNOWN == event) {
        return false;
    }
    const auto validInflightStats = [](const AgentRequestEventToMasterDataPtr &request) {
        return request && request->inflightStatsRevision > 0 &&
               request->inflightRequests >= 0 &&
               std::isfinite(request->inflightPredictedLatencySumMs) &&
               request->inflightPredictedLatencySumMs >= 0.0 &&
               (request->inflightRequests != 0 ||
                request->inflightPredictedLatencySumMs == 0.0);
    };
    switch (event) {
        case NaviSchedAgentEvent::AGENT_HANDSHAKE:
            return control && control->agentStartTimeUs > 0 &&
                   (control->pdSeparated || control->unifiedLatencyPrediction) &&
                   validInflightStats(requestEvent) && !cache && !pendingCache &&
                   (!workload || workload->valid()) &&
                   schedulerConfigRequestGeneration == 0 &&
                   schedulerConfigError.empty() &&
                   ((schedulerConfigGeneration == 0 &&
                     schedulerConfigHash.empty() && schedulerConfigJson.empty()) ||
                    (schedulerConfigGeneration > 0 &&
                     !schedulerConfigHash.empty() &&
                     !schedulerConfigJson.empty()));
        case NaviSchedAgentEvent::AGENT_STOPPING:
            return control && !requestEvent && !cache && !pendingCache &&
                   !latency && !workload;
        case NaviSchedAgentEvent::REQUEST_RECEIVED:
            return validInflightStats(requestEvent) && requestEvent->requestId > 0 &&
                   !control && !cache && !pendingCache && !latency &&
                   (!workload || (workload->valid() && workload->prefillCapacitySplit &&
                                  workload->admittedBatch.empty()));
        case NaviSchedAgentEvent::EXECUTION_LOAD_UPDATED:
            return validInflightStats(requestEvent) && !control && !cache &&
                   !pendingCache && !latency && !workload;
        case NaviSchedAgentEvent::BATCH_REQUESTS_ABORTED:
            return validInflightStats(requestEvent) &&
                   requestEvent->batchWaitingRequests >= 0 &&
                   requestEvent->queuedBatchRequests >= 0 &&
                   requestEvent->runningRequests >= 0 &&
                   requestEvent->completedRequestCount > 0 &&
                   requestEvent->abnormalBatchFinished &&
                   std::isfinite(requestEvent->queueTotalPredictedLatencyMs) &&
                   requestEvent->queueTotalPredictedLatencyMs >= 0.0 &&
                   std::isfinite(requestEvent->queueDecayHeadRemainingMs) &&
                   requestEvent->queueDecayHeadRemainingMs >= 0.0 &&
                   requestEvent->queueDecayHeadRemainingMs <=
                       requestEvent->queueTotalPredictedLatencyMs &&
                   requestEvent->localBatch.valid() &&
                   !requestEvent->localBatch.requestIds.empty() &&
                   requestEvent->localBatch.requestCount ==
                       requestEvent->completedRequestCount &&
                   !control && !cache && !pendingCache && !latency && !workload;
        case NaviSchedAgentEvent::SCHEDULER_CONFIG_UPDATED:
            return schedulerConfigRequestGeneration > 0 &&
                   schedulerConfigGeneration > 0 &&
                   !schedulerConfigHash.empty() && !schedulerConfigJson.empty() &&
                   !control && !requestEvent && !cache && !pendingCache &&
                   !latency && !workload;
        case NaviSchedAgentEvent::CACHE_UPDATED:
            return cache && !control && !requestEvent && !pendingCache && !latency &&
                   !workload;
        case NaviSchedAgentEvent::PENDING_CACHE_UPDATED:
            return pendingCache && !control && !requestEvent && !cache && !latency &&
                   !workload;
        case NaviSchedAgentEvent::LATENCY_MODEL_UPDATED:
            return latency && validInflightStats(requestEvent) && !control && !cache &&
                   !pendingCache && !workload;
        case NaviSchedAgentEvent::WORKLOAD_UPDATED:
            return workload && workload->valid() && !control && !requestEvent &&
                   !cache && !pendingCache && !latency;
        case NaviSchedAgentEvent::BATCH_FINISHED:
            return validInflightStats(requestEvent) && !control && !latency &&
                   !pendingCache && !workload;
        default:
            return false;
    }
}

class AgentUpdateToMasterType : public navi::Type {
public:
    AgentUpdateToMasterType()
        : navi::Type(NAVI_TYPE_PARAM, AGENT_UPDATE_TO_MASTER_TYPE_ID)
    {
    }

public:
    navi::TypeErrorCode serialize(navi::TypeContext &ctx,
                                  const navi::DataPtr &data) const override
    {
        auto update = dynamic_cast<AgentUpdateToMasterData *>(data.get());
        if (!update) {
            return navi::TEC_TYPE_MISMATCH;
        }
        ctx.getDataBuffer().write(update);
        return navi::TEC_NONE;
    }

    navi::TypeErrorCode deserialize(navi::TypeContext &ctx,
                                    navi::DataPtr &data) const override
    {
        AgentUpdateToMasterData *update = nullptr;
        ctx.getDataBuffer().read(update);
        if (!update) {
            return navi::TEC_NULL_DATA;
        }
        if (update->version < NAVI_SCHED_AGENT_UPDATE_MIN_SUPPORTED_VERSION ||
            update->version > NAVI_SCHED_AGENT_UPDATE_VERSION) {
            delete update;
            return navi::TEC_FAILED;
        }
        data.reset(update);
        return navi::TEC_NONE;
    }
};

REGISTER_TYPE(AgentUpdateToMasterType);

} // namespace navi_sched
