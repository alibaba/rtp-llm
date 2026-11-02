#include "aios/navi_sched/debug/NaviSchedScheduleReplayJson.h"
#include "aios/navi_sched/data/BlockHashUtil.h"
#include <autil/legacy/jsonizable.h>

namespace navi_sched {
namespace {

using JsonWrapper = autil::legacy::Jsonizable::JsonWrapper;
using JsonMode = autil::legacy::Jsonizable::Mode;

struct ReplayServerInfoJson : public autil::legacy::Jsonizable {
    NaviSchedServerInfo value;
    ReplayServerInfoJson() = default;
    explicit ReplayServerInfoJson(const NaviSchedServerInfo &input) : value(input) {}
    void Jsonize(JsonWrapper &json) override
    {
        json.Jsonize("ip", value.ip);
        json.Jsonize("port", value.port);
    }
};

struct ReplayTargetJson : public autil::legacy::Jsonizable {
    NaviSchedTarget value;
    ReplayTargetJson() = default;
    explicit ReplayTargetJson(const NaviSchedTarget &input) : value(input) {}
    void Jsonize(JsonWrapper &json) override
    {
        json.Jsonize("version", value.version);
        json.Jsonize("address", value.address);
        json.Jsonize("agent_address", value.agentAddress);
    }
};

struct ReplayWorkloadJson : public autil::legacy::Jsonizable {
    NaviSchedWorkloadSnapshot value;
    ReplayWorkloadJson() = default;
    explicit ReplayWorkloadJson(const NaviSchedWorkloadSnapshot &input)
        : value(input)
    {
    }
    void Jsonize(JsonWrapper &json) override
    {
        json.Jsonize("revision", value.revision);
        json.Jsonize("event_time_us", value.eventTimeUs);
        json.Jsonize("batch_count", value.batchCount);
        json.Jsonize("request_count", value.requestCount);
        json.Jsonize("remaining_prefill_request_count",
                     value.remainingPrefillRequestCount);
        json.Jsonize("running_request_count", value.runningRequestCount);
        json.Jsonize("running_elapsed_time_ms", value.runningElapsedTimeMs);
        json.Jsonize("running_decode_request_count",
                     value.runningDecodeRequestCount,
                     value.runningDecodeRequestCount);
        json.Jsonize("running_decode_generated_token_count",
                     value.runningDecodeGeneratedTokenCount,
                     value.runningDecodeGeneratedTokenCount);
        json.Jsonize("running_decode_remaining_token_budget",
                     value.runningDecodeRemainingTokenBudget,
                     value.runningDecodeRemainingTokenBudget);
        json.Jsonize("waiting_request_count", value.waitingRequestCount);
        json.Jsonize("remaining_prefill_context_token_count",
                     value.remainingPrefillContextTokenCount);
        json.Jsonize("remaining_prefill_token_count",
                     value.remainingPrefillTokenCount);
        json.Jsonize("remaining_prefill_token_square_sum",
                     value.remainingPrefillTokenSquareSum);
        json.Jsonize("context_remaining_prefill_token_product_sum",
                     value.contextRemainingPrefillTokenProductSum);
        json.Jsonize("prefill_capacity_split", value.prefillCapacitySplit,
                     value.prefillCapacitySplit);
        json.Jsonize("max_running_request_count",
                     value.maxRunningRequestCount,
                     value.maxRunningRequestCount);
        json.Jsonize("running_capacity_slot_count",
                     value.runningCapacitySlotCount,
                     value.runningCapacitySlotCount);
        json.Jsonize("running_capacity_prefill_request_count",
                     value.runningCapacityPrefillRequestCount,
                     value.runningCapacityPrefillRequestCount);
        json.Jsonize("running_capacity_prefill_context_token_count",
                     value.runningCapacityPrefillContextTokenCount,
                     value.runningCapacityPrefillContextTokenCount);
        json.Jsonize("running_capacity_prefill_token_count",
                     value.runningCapacityPrefillTokenCount,
                     value.runningCapacityPrefillTokenCount);
        json.Jsonize("running_capacity_prefill_token_square_sum",
                     value.runningCapacityPrefillTokenSquareSum,
                     value.runningCapacityPrefillTokenSquareSum);
        json.Jsonize(
            "running_capacity_context_remaining_prefill_token_product_sum",
            value.runningCapacityContextRemainingPrefillTokenProductSum,
            value.runningCapacityContextRemainingPrefillTokenProductSum);
        json.Jsonize("waiting_capacity_prefill_request_count",
                     value.waitingCapacityPrefillRequestCount,
                     value.waitingCapacityPrefillRequestCount);
        json.Jsonize("waiting_capacity_prefill_context_token_count",
                     value.waitingCapacityPrefillContextTokenCount,
                     value.waitingCapacityPrefillContextTokenCount);
        json.Jsonize("waiting_capacity_prefill_token_count",
                     value.waitingCapacityPrefillTokenCount,
                     value.waitingCapacityPrefillTokenCount);
        json.Jsonize("waiting_capacity_prefill_token_square_sum",
                     value.waitingCapacityPrefillTokenSquareSum,
                     value.waitingCapacityPrefillTokenSquareSum);
        json.Jsonize(
            "waiting_capacity_context_remaining_prefill_token_product_sum",
            value.waitingCapacityContextRemainingPrefillTokenProductSum,
            value.waitingCapacityContextRemainingPrefillTokenProductSum);
        json.Jsonize("remaining_total_context_token_count",
                     value.remainingTotalContextTokenCount);
        json.Jsonize("remaining_total_token_count",
                     value.remainingTotalTokenCount);
        json.Jsonize("remaining_total_token_square_sum",
                     value.remainingTotalTokenSquareSum);
        json.Jsonize("context_remaining_total_token_product_sum",
                     value.contextRemainingTotalTokenProductSum);
        json.Jsonize("total_prompt_token_count", value.totalPromptTokenCount);
        json.Jsonize("in_flight_scheduled_token_count",
                     value.inFlightScheduledTokenCount);
        auto admittedScheduleBatchId = value.admittedBatch.scheduleBatchId;
        auto admittedLocalBatchId = value.admittedBatch.localBatchId;
        auto admittedRequestCount = value.admittedBatch.requestCount;
        auto admittedRequestIds = value.admittedBatch.requestIds;
        json.Jsonize("admitted_schedule_batch_id", admittedScheduleBatchId);
        json.Jsonize("admitted_local_batch_id", admittedLocalBatchId);
        json.Jsonize("admitted_request_count", admittedRequestCount);
        json.Jsonize("admitted_request_ids", admittedRequestIds);
    }
};

struct ReplayLatencyFeaturesJson : public autil::legacy::Jsonizable {
    NaviSchedLatencyFeatures value;
    ReplayLatencyFeaturesJson() = default;
    explicit ReplayLatencyFeaturesJson(const NaviSchedLatencyFeatures &input)
        : value(input)
    {
    }
    void Jsonize(JsonWrapper &json) override
    {
        auto batchCount = value.getBatchCount();
        auto precedingRequestCount = value.getPrecedingRequestCount();
        auto runningElapsedTimeMs = value.getRunningElapsedTimeMs();
        auto runningDecodeRequestCount =
            value.getRunningDecodeRequestCount();
        auto runningDecodeGeneratedTokenCount =
            value.getRunningDecodeGeneratedTokenCount();
        auto runningDecodeRemainingTokenBudget =
            value.getRunningDecodeRemainingTokenBudget();
        auto batchSize = value.getBatchSize();
        auto hitTokens = value.getCacheHitTokenCount();
        auto computeTokens = value.getCacheMissTokenCount();
        auto computeTokensSquared = value.getCacheMissTokenSquareSum();
        auto hitComputeProduct = value.getCacheHitMissTokenProductSum();
        auto capacitySplit = value.usesCapacitySplit();
        auto runningBatchSize = value.getRunningBatchSize();
        auto runningHitTokens = value.getRunningCacheHitTokenCount();
        auto runningComputeTokens = value.getRunningCacheMissTokenCount();
        auto runningComputeTokensSquared =
            value.getRunningCacheMissTokenSquareSum();
        auto runningHitComputeProduct =
            value.getRunningCacheHitMissTokenProductSum();
        auto waitingBatchSize = value.getWaitingBatchSize();
        auto waitingHitTokens = value.getWaitingCacheHitTokenCount();
        auto waitingComputeTokens = value.getWaitingCacheMissTokenCount();
        auto waitingComputeTokensSquared =
            value.getWaitingCacheMissTokenSquareSum();
        auto waitingHitComputeProduct =
            value.getWaitingCacheHitMissTokenProductSum();
        json.Jsonize("batch_count", batchCount);
        json.Jsonize("preceding_request_count", precedingRequestCount,
                     precedingRequestCount);
        json.Jsonize("running_elapsed_time_ms", runningElapsedTimeMs,
                     runningElapsedTimeMs);
        json.Jsonize("running_decode_request_count",
                     runningDecodeRequestCount,
                     runningDecodeRequestCount);
        json.Jsonize("running_decode_generated_token_count",
                     runningDecodeGeneratedTokenCount,
                     runningDecodeGeneratedTokenCount);
        json.Jsonize("running_decode_remaining_token_budget",
                     runningDecodeRemainingTokenBudget,
                     runningDecodeRemainingTokenBudget);
        json.Jsonize("batch_size", batchSize);
        json.Jsonize("hit_tokens", hitTokens);
        json.Jsonize("compute_tokens", computeTokens);
        json.Jsonize("compute_tokens_squared", computeTokensSquared);
        json.Jsonize("hit_compute_product", hitComputeProduct);
        json.Jsonize("capacity_split", capacitySplit, capacitySplit);
        json.Jsonize("running_batch_size", runningBatchSize,
                     runningBatchSize);
        json.Jsonize("running_hit_tokens", runningHitTokens,
                     runningHitTokens);
        json.Jsonize("running_compute_tokens", runningComputeTokens,
                     runningComputeTokens);
        json.Jsonize("running_compute_tokens_squared",
                     runningComputeTokensSquared,
                     runningComputeTokensSquared);
        json.Jsonize("running_hit_compute_product",
                     runningHitComputeProduct,
                     runningHitComputeProduct);
        json.Jsonize("waiting_batch_size", waitingBatchSize,
                     waitingBatchSize);
        json.Jsonize("waiting_hit_tokens", waitingHitTokens,
                     waitingHitTokens);
        json.Jsonize("waiting_compute_tokens", waitingComputeTokens,
                     waitingComputeTokens);
        json.Jsonize("waiting_compute_tokens_squared",
                     waitingComputeTokensSquared,
                     waitingComputeTokensSquared);
        json.Jsonize("waiting_hit_compute_product",
                     waitingHitComputeProduct,
                     waitingHitComputeProduct);
        if (json.GetMode() != JsonMode::TO_JSON) {
            value = NaviSchedLatencyFeatures();
            if (capacitySplit) {
                value.appendCapacitySplitAggregate(
                    batchCount, runningBatchSize, runningHitTokens,
                    runningComputeTokens, runningComputeTokensSquared,
                    runningHitComputeProduct, waitingBatchSize,
                    waitingHitTokens, waitingComputeTokens,
                    waitingComputeTokensSquared,
                    waitingHitComputeProduct);
                value.appendPrecedingRequestCount(precedingRequestCount);
            } else {
                value.appendAggregate(batchCount, batchSize, hitTokens,
                                      computeTokens, computeTokensSquared,
                                      hitComputeProduct);
            }
            value.appendRunningElapsedTimeMs(runningElapsedTimeMs);
            value.appendRunningDecodeProgress(
                runningDecodeRequestCount,
                runningDecodeGeneratedTokenCount,
                runningDecodeRemainingTokenBudget);
        }
    }
};

struct ReplayLatencyPredictionJson : public autil::legacy::Jsonizable {
    NaviSchedLatencyPrediction value;
    bool present = false;

    ReplayLatencyPredictionJson() = default;
    explicit ReplayLatencyPredictionJson(
        const NaviSchedLatencyPrediction &input)
        : value(input), present(true)
    {
    }

    void Jsonize(JsonWrapper &json) override
    {
        ReplayLatencyFeaturesJson features(value.features);
        ReplayLatencyFeaturesJson floorFeatures(value.floorFeatures);
        json.Jsonize("features", features);
        json.Jsonize("floor_features", floorFeatures);
        if (json.GetMode() != JsonMode::TO_JSON) {
            value.features = features.value;
            value.floorFeatures = floorFeatures.value;
        }
        json.Jsonize("fixed_latency_ms", value.fixedLatencyMs);
        json.Jsonize("has_floor", value.hasFloor, value.hasFloor);
        json.Jsonize("recomputable", value.recomputable, value.recomputable);
        present = true;
    }
};

struct ReplayDecisionJson : public autil::legacy::Jsonizable {
    NaviSchedDecision value;
    ReplayDecisionJson() = default;
    explicit ReplayDecisionJson(const NaviSchedDecision &input) : value(input) {}
    void Jsonize(JsonWrapper &json) override
    {
        json.Jsonize("version", value.version);
        json.Jsonize("request_id", value.requestId);
        json.Jsonize("schedule_id", value.scheduleId);
        json.Jsonize("schedule_loss", value.scheduleLoss);
        json.Jsonize("schedule_latency_us", value.scheduleLatencyUs,
                     value.scheduleLatencyUs);
        json.Jsonize("schedule_time_us", value.scheduleTimeUs,
                     value.scheduleTimeUs);
        json.Jsonize("reason", value.reason);
        json.Jsonize("batch_id", value.batchId);
        json.Jsonize("expected_batch_size", value.expectedBatchSize);
        json.Jsonize("predicted_cache_hit_token_count",
                     value.predictedCacheHitTokenCount,
                     value.predictedCacheHitTokenCount);
        json.Jsonize("predicted_queue_latency_ms",
                     value.predictedQueueLatencyMs,
                     value.predictedQueueLatencyMs);
        json.Jsonize("predicted_execution_latency_ms",
                     value.predictedExecutionLatencyMs,
                     value.predictedExecutionLatencyMs);
        int32_t trainingPhase =
            static_cast<int32_t>(value.latencyTrainingPhase);
        json.Jsonize("latency_training_phase", trainingPhase, trainingPhase);
        if (json.GetMode() != JsonMode::TO_JSON) {
            value.latencyTrainingPhase =
                static_cast<NaviSchedLatencyTrainingPhase>(trainingPhase);
        }
        if (json.GetMode() == JsonMode::TO_JSON) {
            if (value.latencyPrediction) {
                ReplayLatencyPredictionJson prediction(*value.latencyPrediction);
                json.Jsonize("latency_prediction", prediction);
            }
        } else {
            ReplayLatencyPredictionJson prediction;
            json.Jsonize("latency_prediction", prediction, prediction);
            if (prediction.present) {
                value.latencyPrediction =
                    std::make_shared<NaviSchedLatencyPrediction>(prediction.value);
            } else {
                value.latencyPrediction.reset();
            }
        }
        ReplayTargetJson target(value.target);
        json.Jsonize("target", target);
        if (json.GetMode() != JsonMode::TO_JSON) value.target = target.value;
    }
};

struct ReplayNodeInfoJson : public autil::legacy::Jsonizable {
    NaviSchedNodeInfo value;
    ReplayNodeInfoJson() = default;
    explicit ReplayNodeInfoJson(const NaviSchedNodeInfo &input) : value(input) {}
    void Jsonize(JsonWrapper &json) override
    {
        json.Jsonize("agent_id_hash", value.agentIdHash);
        json.Jsonize("agent_id", value.agentId);
        json.Jsonize("collector_address", value.collectorAddress);
        ReplayServerInfoJson serverInfo(value.serverInfo);
        json.Jsonize("server_info", serverInfo);
        if (json.GetMode() != JsonMode::TO_JSON) value.serverInfo = serverInfo.value;
        json.Jsonize("server_address", value.serverAddress);
        json.Jsonize("total_blocks", value.totalBlocks);
        json.Jsonize("inflight_stats_revision", value.inflightStatsRevision);
        json.Jsonize("inflight_requests", value.inflightRequests);
        json.Jsonize("inflight_predicted_latency_sum_ms",
                     value.inflightPredictedLatencySumMs);
        json.Jsonize("batch_waiting_requests", value.batchWaitingRequests);
        json.Jsonize("queued_batch_requests", value.queuedBatchRequests);
        json.Jsonize("running_requests", value.runningRequests);
        json.Jsonize("pd_separated", value.pdSeparated);
        json.Jsonize("unified_latency_prediction",
                     value.unifiedLatencyPrediction);
        ReplayWorkloadJson workload(value.workload);
        json.Jsonize("workload", workload);
        if (json.GetMode() != JsonMode::TO_JSON) value.workload = workload.value;
        ReplayLatencyFeaturesJson reservedWorkloadFeatures(
            value.reservedWorkloadFeatures);
        json.Jsonize("reserved_workload_features", reservedWorkloadFeatures);
        if (json.GetMode() != JsonMode::TO_JSON) {
            value.reservedWorkloadFeatures = reservedWorkloadFeatures.value;
        }
        json.Jsonize("workload_feature_snapshot_valid",
                     value.workloadFeatureSnapshotValid,
                     value.workloadFeatureSnapshotValid);
        ReplayLatencyFeaturesJson workloadFeatures(value.workloadFeatures);
        json.Jsonize("workload_features", workloadFeatures,
                     workloadFeatures);
        if (json.GetMode() != JsonMode::TO_JSON) {
            value.workloadFeatures = workloadFeatures.value;
        }
        json.Jsonize("max_running_request_count",
                     value.maxRunningRequestCount,
                     value.maxRunningRequestCount);
        json.Jsonize("workload_running_capacity_slot_count",
                     value.workloadRunningCapacitySlotCount,
                     value.workloadRunningCapacitySlotCount);
        json.Jsonize("reserved_running_capacity_slot_count",
                     value.reservedRunningCapacitySlotCount,
                     value.reservedRunningCapacitySlotCount);
        json.Jsonize("workload_report_receive_time_us",
                     value.workloadReportReceiveTimeUs);
        json.Jsonize("workload_stale_update_count",
                     value.workloadStaleUpdateCount);
        json.Jsonize("batch_timeout_count", value.batchTimeoutCount);
        json.Jsonize("abnormal_batch_finish_count", value.abnormalBatchFinishCount);
        json.Jsonize("last_request_latency_ms", value.lastRequestLatencyMs);
        json.Jsonize("queue_total_predicted_latency_ms",
                     value.queueTotalPredictedLatencyMs);
        json.Jsonize("queue_decay_head_remaining_ms",
                     value.queueDecayHeadRemainingMs);
        json.Jsonize("queue_report_receive_time_us", value.queueReportReceiveTimeUs);
        json.Jsonize("online", value.online);
        json.Jsonize("last_event_type", value.lastEventType);
        json.Jsonize("last_event_time_us", value.lastEventTimeUs);
        json.Jsonize("last_request_event_type", value.lastRequestEventType);
        json.Jsonize("last_request_event_time_us", value.lastRequestEventTimeUs);
        json.Jsonize("last_receive_time_us", value.lastReceiveTimeUs);
        json.Jsonize("latency_formula", value.latencyFormula);
        json.Jsonize("latency_binary_version", value.latencyBinaryVersion);
        json.Jsonize("latency_model_revision", value.latencyModelRevision);
        json.Jsonize("latency_parameters", value.latencyParameters);
        json.Jsonize("latency_residual_version", value.latencyResidualVersion,
                     value.latencyResidualVersion);
        json.Jsonize("latency_residual_parameters",
                     value.latencyResidualParameters,
                     value.latencyResidualParameters);
        json.Jsonize("latency_residual_last_update_time_us",
                     value.latencyResidualLastUpdateTimeUs,
                     value.latencyResidualLastUpdateTimeUs);
        json.Jsonize("latency_model_fingerprint",
                     value.latencyModelFingerprint,
                     value.latencyModelFingerprint);
        json.Jsonize("latency_model_trusted", value.latencyModelTrusted,
                     value.latencyModelTrusted);
        json.Jsonize("latency_model_warm_started",
                     value.latencyModelWarmStarted,
                     value.latencyModelWarmStarted);
        json.Jsonize("latency_model_consecutive_rejected_count",
                     value.latencyModelConsecutiveRejectedCount,
                     value.latencyModelConsecutiveRejectedCount);
        json.Jsonize("latency_model_last_reject_reason",
                     value.latencyModelLastRejectReason,
                     value.latencyModelLastRejectReason);
        json.Jsonize("scheduler_config_generation",
                     value.schedulerConfigGeneration,
                     value.schedulerConfigGeneration);
        json.Jsonize("scheduler_config_hash", value.schedulerConfigHash,
                     value.schedulerConfigHash);
        json.Jsonize("scheduler_config_json", value.schedulerConfigJson,
                     value.schedulerConfigJson);
    }
};

struct ReplayScheduleRequestJson : public autil::legacy::Jsonizable {
    NaviSchedScheduleRequest value;
    ReplayScheduleRequestJson() = default;
    explicit ReplayScheduleRequestJson(const NaviSchedScheduleRequest &input)
        : value(input)
    {
    }
    void Jsonize(JsonWrapper &json) override
    {
        json.Jsonize("version", value.version);
        json.Jsonize("request_id", value.requestId);
        json.Jsonize("prompt_token_count", value.promptTokenCount);
        json.Jsonize("block_size", value.blockSize);
        auto blockHashes = BlockHashUtil::get(value.blockHashes);
        json.Jsonize("block_hashes", blockHashes);
        if (json.GetMode() != JsonMode::TO_JSON) {
            value.blockHashes =
                std::make_shared<const NaviSchedBlockHashes>(std::move(blockHashes));
        }
        json.Jsonize("timeout_ms", value.timeoutMs);
    }
};

struct ReplayScheduleItemJson : public autil::legacy::Jsonizable {
    NaviSchedScheduleItem value;
    ReplayScheduleItemJson() = default;
    explicit ReplayScheduleItemJson(const NaviSchedScheduleItem &input) : value(input) {}
    void Jsonize(JsonWrapper &json) override
    {
        ReplayScheduleRequestJson request(value.request);
        json.Jsonize("request", request);
        if (json.GetMode() != JsonMode::TO_JSON) value.request = request.value;
        json.Jsonize("wait_time_ms", value.waitTimeMs);
        json.Jsonize("remaining_timeout_ms", value.remainingTimeoutMs);
        json.Jsonize("timeout_triggered", value.timeoutTriggered);
    }
};

struct ReplayNodeMatchJson : public autil::legacy::Jsonizable {
    NaviSchedReplayNodeMatch value;
    ReplayNodeMatchJson() = default;
    explicit ReplayNodeMatchJson(const NaviSchedReplayNodeMatch &input) : value(input) {}
    void Jsonize(JsonWrapper &json) override
    {
        json.Jsonize("matched_block_count", value.matchedBlockCount);
    }
};

struct ReplayJson : public autil::legacy::Jsonizable {
    NaviSchedScheduleReplay value;
    ReplayJson() = default;
    explicit ReplayJson(const NaviSchedScheduleReplay &input) : value(input) {}
    void Jsonize(JsonWrapper &json) override
    {
        json.Jsonize("version", value.version);
        json.Jsonize("schedule_time_us", value.scheduleTimeUs);
        json.Jsonize("biz_name", value.bizName);
        if (json.GetMode() == JsonMode::TO_JSON) {
            std::vector<ReplayScheduleItemJson> requests;
            std::vector<ReplayNodeInfoJson> nodes;
            std::vector<std::vector<ReplayNodeMatchJson>> nodeMatches;
            std::vector<ReplayDecisionJson> decisions;
            for (const auto &request : value.requests) requests.emplace_back(request);
            for (const auto &node : value.nodes) nodes.emplace_back(node);
            for (const auto &matches : value.nodeMatches) {
                auto &output = nodeMatches.emplace_back();
                for (const auto &match : matches) output.emplace_back(match);
            }
            for (const auto &decision : value.decisions) decisions.emplace_back(decision);
            json.Jsonize("requests", requests);
            json.Jsonize("nodes", nodes);
            json.Jsonize("node_matches", nodeMatches);
            json.Jsonize("decisions", decisions);
            if (!value.schedulerInfo.empty()) {
                json.Jsonize("scheduler_info", value.schedulerInfo);
            }
        } else {
            std::vector<ReplayScheduleItemJson> requests;
            std::vector<ReplayNodeInfoJson> nodes;
            std::vector<std::vector<ReplayNodeMatchJson>> nodeMatches;
            std::vector<ReplayDecisionJson> decisions;
            json.Jsonize("requests", requests);
            json.Jsonize("nodes", nodes);
            json.Jsonize("node_matches", nodeMatches);
            json.Jsonize("decisions", decisions);
            json.Jsonize("scheduler_info", value.schedulerInfo,
                         value.schedulerInfo);
            value.requests.clear();
            value.nodes.clear();
            value.nodeMatches.clear();
            value.decisions.clear();
            for (const auto &request : requests) {
                auto item = request.value;
                item.request.bizName = value.bizName;
                value.requests.emplace_back(std::move(item));
            }
            for (const auto &node : nodes) {
                auto info = node.value;
                info.bizName = value.bizName;
                value.nodes.emplace_back(std::move(info));
            }
            for (const auto &matches : nodeMatches) {
                auto &output = value.nodeMatches.emplace_back();
                for (const auto &match : matches) output.emplace_back(match.value);
            }
            for (const auto &decision : decisions) {
                auto output = decision.value;
                output.target.bizName = value.bizName;
                value.decisions.emplace_back(std::move(output));
            }
        }
    }
};

}

std::string NaviSchedScheduleReplayJson::toJsonString(
    const NaviSchedScheduleReplay &replay)
{
    return autil::legacy::FastToJsonString(ReplayJson(replay));
}

void NaviSchedScheduleReplayJson::fromJsonString(const std::string &json,
                                                 NaviSchedScheduleReplay &replay)
{
    ReplayJson value;
    autil::legacy::FastFromJsonString(value, json);
    replay = value.value;
}

}
