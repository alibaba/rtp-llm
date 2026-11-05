#include "aios/navi_sched/service/NaviSchedMasterSummary.h"
#include "aios/navi_sched/data/AgentLatencyToMasterData.h"

#include <iomanip>
#include <sstream>

namespace navi_sched {

void NaviSchedMasterNodeSummary::Jsonize(JsonWrapper &json)
{
    json.Jsonize("agent_id_hash", agentIdHash);
    json.Jsonize("agent_id", agentId);
    json.Jsonize("biz_name", bizName);
    json.Jsonize("collector_address", collectorAddress);
    json.Jsonize("server_ip", serverIp);
    json.Jsonize("server_port", serverPort);
    json.Jsonize("server_address", serverAddress);
    json.Jsonize("online", online);
    json.Jsonize("pd_separated", pdSeparated);
    json.Jsonize("peer_cache_enabled", peerCacheEnabled);
    json.Jsonize("peer_cache_read_available", peerCacheReadAvailable);
    json.Jsonize("peer_cache_read_revision", peerCacheReadRevision);
    json.Jsonize("peer_cache_read_agent_start_time_us", peerCacheReadAgentStartTimeUs);
    json.Jsonize("total_blocks", totalBlocks);
    json.Jsonize("cached_block_count", cachedBlockCount);
    json.Jsonize("agent_cache_revision", agentCacheRevision);
    json.Jsonize("agent_cached_block_count", agentCachedBlockCount);
    json.Jsonize("agent_cache_hash_checksum", agentCacheHashChecksum);
    json.Jsonize("agent_full_ready_block_count", agentFullReadyBlockCount);
    json.Jsonize("agent_full_ready_hash_checksum", agentFullReadyHashChecksum);
    json.Jsonize("agent_tail_ready_block_count", agentTailReadyBlockCount);
    json.Jsonize("agent_tail_ready_hash_checksum", agentTailReadyHashChecksum);
    json.Jsonize("master_cache_hash_checksum", masterCacheHashChecksum);
    json.Jsonize("master_full_ready_block_count", masterFullReadyBlockCount);
    json.Jsonize("master_full_ready_hash_checksum", masterFullReadyHashChecksum);
    json.Jsonize("master_tail_ready_block_count", masterTailReadyBlockCount);
    json.Jsonize("master_tail_ready_hash_checksum", masterTailReadyHashChecksum);
    json.Jsonize("rtp_cached_block_count", rtpCachedBlockCount);
    json.Jsonize("cache_revision_gap_count", cacheRevisionGapCount);
    json.Jsonize("cache_state_mismatch_count", cacheStateMismatchCount);
    json.Jsonize("rtp_cache_mirror_sample_count", rtpCacheMirrorSampleCount);
    json.Jsonize("rtp_cache_mirror_mismatch_count", rtpCacheMirrorMismatchCount);
    if (json.GetMode() == FROM_JSON || !v6dCacheStats.empty()) {
        json.Jsonize("v6d_cache", v6dCacheStats, v6dCacheStats);
    }
    json.Jsonize("pending_cache_batch_count", pendingCacheBatchCount);
    json.Jsonize("pending_cache_block_count", pendingCacheBlockCount);
    json.Jsonize("pending_cache_scheduled_batch_count", pendingCacheScheduledBatchCount);
    json.Jsonize("pending_cache_agent_batch_count", pendingCacheAgentBatchCount);
    json.Jsonize("pending_cache_agent_revision", pendingCacheAgentRevision);
    json.Jsonize("pending_cache_agent_checksum", pendingCacheAgentChecksum);
    json.Jsonize("pending_cache_revision_gap_count", pendingCacheRevisionGapCount);
    json.Jsonize("pending_cache_state_mismatch_count", pendingCacheStateMismatchCount);
    json.Jsonize("pending_cache_resync_trigger_count", pendingCacheResyncTriggerCount);
    json.Jsonize("pending_cache_snapshot_ready", pendingCacheSnapshotReady);
    json.Jsonize("inflight_stats_revision", inflightStatsRevision);
    json.Jsonize("inflight_requests", inflightRequests);
    json.Jsonize("inflight_predicted_latency_sum_ms", inflightPredictedLatencySumMs);
    json.Jsonize("batch_waiting_requests", batchWaitingRequests);
    json.Jsonize("queued_batch_requests", queuedBatchRequests);
    json.Jsonize("running_requests", runningRequests);
    json.Jsonize("execution_utilization", executionUtilization);
    json.Jsonize("execution_busy_utilization", executionBusyUtilization);
    json.Jsonize("batch_timeout_count", batchTimeoutCount);
    json.Jsonize("abnormal_batch_finish_count", abnormalBatchFinishCount);
    json.Jsonize("master_sent_request_count", masterSentRequestCount);
    json.Jsonize("master_sent_batch_count", masterSentBatchCount);
    json.Jsonize("master_sent_requests_per_batch_mean", masterSentRequestsPerBatchMean);
    json.Jsonize("reservation_batch_count", reservationBatchCount);
    json.Jsonize("reservation_request_count", reservationRequestCount);
    json.Jsonize("reservation_waiting_request_count", reservationWaitingRequestCount);
    json.Jsonize("reservation_dispatched_request_count",
                 reservationDispatchedRequestCount);
    json.Jsonize("dispatch_waiter_count", dispatchWaiterCount);
    json.Jsonize("live_dispatch_waiter_count", liveDispatchWaiterCount);
    json.Jsonize("last_request_latency_ms", lastRequestLatencyMs);
    json.Jsonize("queue_total_predicted_latency_ms", queueTotalPredictedLatencyMs);
    json.Jsonize("queue_report_receive_time_us", queueReportReceiveTimeUs);
    json.Jsonize("workload_revision", workload.revision);
    json.Jsonize("workload_event_time_us", workload.eventTimeUs);
    json.Jsonize("workload_report_receive_time_us", workloadReportReceiveTimeUs);
    json.Jsonize("workload_stale_update_count", workloadStaleUpdateCount);
    json.Jsonize("workload_request_count", workload.requestCount);
    json.Jsonize("workload_remaining_prefill_request_count",
                 workload.remainingPrefillRequestCount);
    json.Jsonize("workload_running_request_count", workload.runningRequestCount);
    json.Jsonize("workload_running_elapsed_time_ms", workload.runningElapsedTimeMs);
    json.Jsonize("workload_running_decode_request_count",
                 workload.runningDecodeRequestCount);
    json.Jsonize("workload_running_decode_generated_tokens",
                 workload.runningDecodeGeneratedTokenCount);
    json.Jsonize("workload_running_decode_remaining_token_budget",
                 workload.runningDecodeRemainingTokenBudget);
    json.Jsonize("workload_effective_running_elapsed_time_ms",
                 workloadEffectiveRunningElapsedTimeMs);
    json.Jsonize("workload_waiting_request_count", workload.waitingRequestCount);
    json.Jsonize("workload_remaining_prefill_context_tokens",
                 workload.remainingPrefillContextTokenCount);
    json.Jsonize("workload_remaining_prefill_tokens",
                 workload.remainingPrefillTokenCount);
    json.Jsonize("workload_remaining_prefill_token_square_sum",
                 workload.remainingPrefillTokenSquareSum);
    json.Jsonize("workload_context_remaining_prefill_product_sum",
                 workload.contextRemainingPrefillTokenProductSum);
    json.Jsonize("workload_prefill_capacity_split", workload.prefillCapacitySplit);
    json.Jsonize("workload_max_running_request_count", workload.maxRunningRequestCount);
    json.Jsonize("workload_running_capacity_slot_count",
                 workload.runningCapacitySlotCount);
    json.Jsonize("workload_running_capacity_prefill_request_count",
                 workload.runningCapacityPrefillRequestCount);
    json.Jsonize("workload_running_capacity_prefill_context_tokens",
                 workload.runningCapacityPrefillContextTokenCount);
    json.Jsonize("workload_running_capacity_prefill_tokens",
                 workload.runningCapacityPrefillTokenCount);
    json.Jsonize("workload_running_capacity_prefill_token_square_sum",
                 workload.runningCapacityPrefillTokenSquareSum);
    json.Jsonize("workload_running_capacity_prefill_product_sum",
                 workload.runningCapacityContextRemainingPrefillTokenProductSum);
    json.Jsonize("workload_waiting_capacity_prefill_request_count",
                 workload.waitingCapacityPrefillRequestCount);
    json.Jsonize("workload_waiting_capacity_prefill_context_tokens",
                 workload.waitingCapacityPrefillContextTokenCount);
    json.Jsonize("workload_waiting_capacity_prefill_tokens",
                 workload.waitingCapacityPrefillTokenCount);
    json.Jsonize("workload_waiting_capacity_prefill_token_square_sum",
                 workload.waitingCapacityPrefillTokenSquareSum);
    json.Jsonize("workload_waiting_capacity_prefill_product_sum",
                 workload.waitingCapacityContextRemainingPrefillTokenProductSum);
    json.Jsonize("workload_remaining_total_context_tokens",
                 workload.remainingTotalContextTokenCount);
    json.Jsonize("workload_remaining_total_tokens", workload.remainingTotalTokenCount);
    json.Jsonize("workload_remaining_total_token_square_sum",
                 workload.remainingTotalTokenSquareSum);
    json.Jsonize("workload_context_remaining_total_product_sum",
                 workload.contextRemainingTotalTokenProductSum);
    json.Jsonize("workload_total_prompt_tokens", workload.totalPromptTokenCount);
    json.Jsonize("workload_inflight_scheduled_tokens",
                 workload.inFlightScheduledTokenCount);
    json.Jsonize("last_event_type", lastEventType);
    json.Jsonize("last_event_time_us", lastEventTimeUs);
    json.Jsonize("last_request_event_type", lastRequestEventType);
    json.Jsonize("last_request_event_time_us", lastRequestEventTimeUs);
    json.Jsonize("last_receive_time_us", lastReceiveTimeUs);
    json.Jsonize("latency_formula", latencyFormula);
    json.Jsonize("latency_formula_name", getNaviSchedLatencyFormulaName(latencyFormula));
    json.Jsonize("latency_binary_version", latencyBinaryVersion);
    json.Jsonize("latency_model_revision", latencyModelRevision);
    json.Jsonize("latency_parameter_count", latencyParameterCount);
    json.Jsonize("latency_parameters", latencyParameters);
    json.Jsonize("latency_residual_version", latencyResidualVersion);
    json.Jsonize("latency_residual_parameters", latencyResidualParameters);
    json.Jsonize("latency_residual_last_update_time_us", latencyResidualLastUpdateTimeUs);
    json.Jsonize("latency_model_fingerprint", latencyModelFingerprint);
    json.Jsonize("latency_model_trusted", latencyModelTrusted);
    json.Jsonize("latency_model_warm_started", latencyModelWarmStarted);
    json.Jsonize("latency_model_consecutive_rejected_count",
                 latencyModelConsecutiveRejectedCount);
    json.Jsonize("latency_model_last_reject_reason", latencyModelLastRejectReason);
    json.Jsonize("scheduler_config_generation", schedulerConfigGeneration);
    json.Jsonize("scheduler_config_hash", schedulerConfigHash);
    json.Jsonize("scheduler_config_json", schedulerConfigJson);
    json.Jsonize("latency_validation_sample_count", latencyValidation.sampleCount);
    json.Jsonize("latency_validation_unmatched_batch_count",
                 latencyValidation.unmatchedBatchCount);
    json.Jsonize("latency_validation_batch_request_count_mean",
                 latencyValidation.batchRequestCountMean);
    json.Jsonize("latency_validation_predicted_cache_hit_tokens_mean",
                 latencyValidation.predictedCacheHitTokensMean);
    json.Jsonize("latency_validation_actual_cache_hit_tokens_mean",
                 latencyValidation.actualCacheHitTokensMean);
    json.Jsonize("latency_validation_cache_hit_tokens_bias_mean",
                 latencyValidation.cacheHitTokensBiasMean);
    json.Jsonize("latency_validation_cache_hit_tokens_mae",
                 latencyValidation.cacheHitTokensMae);
    json.Jsonize("master_predicted_queue_mean_ms",
                 latencyValidation.masterPredictedQueueMeanMs);
    json.Jsonize("actual_queue_mean_ms", latencyValidation.actualQueueMeanMs);
    json.Jsonize("backend_queue_mean_ms", latencyValidation.backendQueueMeanMs);
    json.Jsonize("queue_bias_mean_ms", latencyValidation.queueBiasMeanMs);
    json.Jsonize("queue_mae_ms", latencyValidation.queueMaeMs);
    json.Jsonize("queue_raw_predicted_mean_ms",
                 latencyValidation.queueRawPredictedMeanMs);
    json.Jsonize("queue_raw_bias_mean_ms", latencyValidation.queueRawBiasMeanMs);
    json.Jsonize("queue_raw_mae_ms", latencyValidation.queueRawMaeMs);
    json.Jsonize("agent_predicted_queue_mean_ms",
                 latencyValidation.agentPredictedQueueMeanMs);
    json.Jsonize("agent_queue_bias_mean_ms", latencyValidation.agentQueueBiasMeanMs);
    json.Jsonize("agent_queue_mae_ms", latencyValidation.agentQueueMaeMs);
    json.Jsonize("queue_feature_batch_count_mean",
                 latencyValidation.queueFeatureBatchCountMean);
    json.Jsonize("queue_model_weights", latencyValidation.queueModelWeights);
    json.Jsonize("queue_model_fit_step", latencyValidation.queueModelFitStep);
    json.Jsonize("zero_predicted_queue_sample_count",
                 latencyValidation.zeroPredictedQueueSampleCount);
    json.Jsonize("zero_predicted_queue_actual_mean_ms",
                 latencyValidation.zeroPredictedQueueActualMeanMs);
    json.Jsonize("zero_predicted_queue_mae_ms",
                 latencyValidation.zeroPredictedQueueMaeMs);
    json.Jsonize("positive_predicted_queue_sample_count",
                 latencyValidation.positivePredictedQueueSampleCount);
    json.Jsonize("positive_predicted_queue_mean_ms",
                 latencyValidation.positivePredictedQueueMeanMs);
    json.Jsonize("positive_predicted_queue_actual_mean_ms",
                 latencyValidation.positivePredictedQueueActualMeanMs);
    json.Jsonize("positive_predicted_queue_bias_mean_ms",
                 latencyValidation.positivePredictedQueueBiasMeanMs);
    json.Jsonize("positive_predicted_queue_mae_ms",
                 latencyValidation.positivePredictedQueueMaeMs);
    json.Jsonize("master_predicted_execution_mean_ms",
                 latencyValidation.masterPredictedExecutionMeanMs);
    json.Jsonize("agent_pre_update_predicted_execution_mean_ms",
                 latencyValidation.agentPredictedExecutionMeanMs);
    json.Jsonize("actual_execution_mean_ms", latencyValidation.actualExecutionMeanMs);
    json.Jsonize("master_execution_bias_mean_ms",
                 latencyValidation.masterExecutionBiasMeanMs);
    json.Jsonize("master_execution_mae_ms", latencyValidation.masterExecutionMaeMs);
    json.Jsonize("master_execution_mape_pct", latencyValidation.masterExecutionMapePct);
    json.Jsonize("agent_pre_update_execution_bias_mean_ms",
                 latencyValidation.agentExecutionBiasMeanMs);
    json.Jsonize("agent_pre_update_execution_mae_ms",
                 latencyValidation.agentExecutionMaeMs);
    json.Jsonize("agent_pre_update_execution_mape_pct",
                 latencyValidation.agentExecutionMapePct);
    json.Jsonize("master_predicted_total_mean_ms",
                 latencyValidation.masterPredictedTotalMeanMs);
    json.Jsonize("actual_total_mean_ms", latencyValidation.actualTotalMeanMs);
    json.Jsonize("total_bias_mean_ms", latencyValidation.totalBiasMeanMs);
    json.Jsonize("total_mae_ms", latencyValidation.totalMaeMs);
    json.Jsonize("total_mape_pct", latencyValidation.totalMapePct);
}

std::string NaviSchedMasterNodeSummary::format() const
{
    std::ostringstream output;
    output << "  Agent\n"
           << "    id: " << agentId << "\n"
           << "    biz_name: " << bizName << "\n"
           << "    online: " << (online ? "true" : "false") << "\n"
           << "    pd_separated: " << (pdSeparated ? "true" : "false") << "\n"
           << "    peer_cache_enabled: " << (peerCacheEnabled ? "true" : "false") << "\n"
           << "    peer_cache_read_available: "
           << (peerCacheReadAvailable ? "true" : "false") << "\n"
           << "    peer_cache_read_revision: " << peerCacheReadRevision << "\n"
           << "    peer_cache_read_agent_start_time_us: " << peerCacheReadAgentStartTimeUs
           << "\n"
           << "    collector_address: " << collectorAddress << "\n"
           << "    server_address: " << serverAddress << "\n"
           << "    cache_blocks: total=" << totalBlocks << ", cached=" << cachedBlockCount
           << "\n"
           << "    cache_sync: revision=" << agentCacheRevision
           << ", agent_cached=" << agentCachedBlockCount
           << ", master_cached=" << cachedBlockCount
           << ", agent_checksum=" << agentCacheHashChecksum
           << ", master_checksum=" << masterCacheHashChecksum
           << ", rtp_cached=" << rtpCachedBlockCount
           << ", revision_gaps=" << cacheRevisionGapCount
           << ", state_mismatches=" << cacheStateMismatchCount << "\n"
           << "    hybrid_cache_sync: agent_full=" << agentFullReadyBlockCount
           << ", master_full=" << masterFullReadyBlockCount
           << ", agent_full_checksum=" << agentFullReadyHashChecksum
           << ", master_full_checksum=" << masterFullReadyHashChecksum
           << ", agent_tail=" << agentTailReadyBlockCount
           << ", master_tail=" << masterTailReadyBlockCount
           << ", agent_tail_checksum=" << agentTailReadyHashChecksum
           << ", master_tail_checksum=" << masterTailReadyHashChecksum << "\n"
           << "    rtp_cache_mirror: samples=" << rtpCacheMirrorSampleCount
           << ", mismatches=" << rtpCacheMirrorMismatchCount << "\n"
           << "    pending_cache: batches=" << pendingCacheBatchCount
           << ", unique_blocks=" << pendingCacheBlockCount
           << ", scheduled_batches=" << pendingCacheScheduledBatchCount
           << ", agent_batches=" << pendingCacheAgentBatchCount
           << ", agent_revision=" << pendingCacheAgentRevision
           << ", agent_checksum=" << pendingCacheAgentChecksum
           << ", snapshot_ready=" << pendingCacheSnapshotReady
           << ", revision_gaps=" << pendingCacheRevisionGapCount
           << ", state_mismatches=" << pendingCacheStateMismatchCount
           << ", resync_triggers=" << pendingCacheResyncTriggerCount << "\n";
    if (!v6dCacheStats.empty()) {
        output << "    v6d_cache:\n";
        for (const auto &[name, value] : v6dCacheStats) {
            output << "      " << name << ": " << value << "\n";
        }
    }
    output
        << "    requests: inflight=" << inflightRequests
        << ", inflight_revision=" << inflightStatsRevision
        << ", inflight_predicted_latency_sum_ms=" << inflightPredictedLatencySumMs
        << ", waiting_for_batch=" << batchWaitingRequests
        << ", queued_for_execution=" << queuedBatchRequests
        << ", running=" << runningRequests << "\n"
        << std::fixed << std::setprecision(3)
        << "    execution_busy_utilization: " << executionBusyUtilization << "\n"
        << "    execution_utilization: " << executionUtilization << "\n"
        << "    master_dispatch: sent_requests=" << masterSentRequestCount
        << ", sent_batches=" << masterSentBatchCount << std::fixed << std::setprecision(3)
        << ", requests_per_batch_mean=" << masterSentRequestsPerBatchMean << "\n"
        << "    reservations: batches=" << reservationBatchCount
        << ", requests=" << reservationRequestCount
        << ", waiting=" << reservationWaitingRequestCount
        << ", dispatched=" << reservationDispatchedRequestCount
        << ", dispatch_waiters=" << dispatchWaiterCount
        << ", live_dispatch_waiters=" << liveDispatchWaiterCount << "\n"
        << "    batch_timeout_count: " << batchTimeoutCount << "\n"
        << "    abnormal_batch_finish_count: " << abnormalBatchFinishCount << "\n"
        << std::fixed << std::setprecision(3)
        << "    queue_predicted_ms: " << queueTotalPredictedLatencyMs << "\n"
        << "    last_request_latency_ms: " << lastRequestLatencyMs << "\n"
        << "    workload: revision=" << workload.revision
        << ", request_count=" << workload.requestCount
        << ", remaining_prefill_requests=" << workload.remainingPrefillRequestCount
        << ", running_elapsed_time_ms=" << workload.runningElapsedTimeMs
        << ", running_decode_requests=" << workload.runningDecodeRequestCount
        << ", running_decode_generated_tokens="
        << workload.runningDecodeGeneratedTokenCount
        << ", running_decode_remaining_token_budget="
        << workload.runningDecodeRemainingTokenBudget
        << ", effective_running_elapsed_time_ms=" << workloadEffectiveRunningElapsedTimeMs
        << ", remaining_prefill_context_tokens="
        << workload.remainingPrefillContextTokenCount
        << ", remaining_prefill_tokens=" << workload.remainingPrefillTokenCount
        << ", capacity_split=" << (workload.prefillCapacitySplit ? "true" : "false")
        << ", running_capacity_slots=" << workload.runningCapacitySlotCount << "/"
        << workload.maxRunningRequestCount << ", running_capacity_prefill_requests="
        << workload.runningCapacityPrefillRequestCount
        << ", waiting_capacity_prefill_requests="
        << workload.waitingCapacityPrefillRequestCount
        << ", remaining_total_context_tokens=" << workload.remainingTotalContextTokenCount
        << ", remaining_total_tokens=" << workload.remainingTotalTokenCount
        << ", inflight_scheduled_tokens=" << workload.inFlightScheduledTokenCount
        << ", stale_updates=" << workloadStaleUpdateCount << "\n"
        << "    latency_model: formula=" << getNaviSchedLatencyFormulaName(latencyFormula)
        << ", binary_version=" << latencyBinaryVersion
        << ", revision=" << latencyModelRevision
        << ", parameter_count=" << latencyParameterCount
        << ", trusted=" << latencyModelTrusted
        << ", warm_started=" << latencyModelWarmStarted
        << ", consecutive_rejected=" << latencyModelConsecutiveRejectedCount
        << ", last_reject_reason=" << latencyModelLastRejectReason
        << ", fingerprint=" << latencyModelFingerprint << "\n"
        << "    scheduler_config: generation=" << schedulerConfigGeneration
        << ", hash=" << schedulerConfigHash << ", config=" << schedulerConfigJson << "\n"
        << "    latency_parameters: [";
    for (size_t index = 0; index < latencyParameters.size(); ++index) {
        if (index > 0) {
            output << ", ";
        }
        output << std::defaultfloat << std::setprecision(17) << latencyParameters[index];
    }
    output << "]\n"
           << "    latency_residual: version=" << latencyResidualVersion
           << ", last_update_time_us=" << latencyResidualLastUpdateTimeUs
           << ", parameters=[";
    for (size_t index = 0; index < latencyResidualParameters.size(); ++index) {
        if (index > 0) {
            output << ", ";
        }
        output << std::defaultfloat << std::setprecision(17)
               << latencyResidualParameters[index];
    }
    output << "]\n"
           << std::fixed << std::setprecision(3)
           << "    latency_validation: samples=" << latencyValidation.sampleCount
           << ", unmatched_batches=" << latencyValidation.unmatchedBatchCount << "\n"
           << "      batch: request_count_mean="
           << latencyValidation.batchRequestCountMean << "\n"
           << "      cache_hit_tokens: predicted_mean="
           << latencyValidation.predictedCacheHitTokensMean
           << ", actual_mean=" << latencyValidation.actualCacheHitTokensMean
           << ", bias_mean=" << latencyValidation.cacheHitTokensBiasMean
           << ", mae=" << latencyValidation.cacheHitTokensMae << "\n"
           << "      queue_ms: predicted_mean="
           << latencyValidation.masterPredictedQueueMeanMs
           << ", actual_mean=" << latencyValidation.actualQueueMeanMs
           << ", backend_mean=" << latencyValidation.backendQueueMeanMs
           << ", bias_mean=" << latencyValidation.queueBiasMeanMs
           << ", mae=" << latencyValidation.queueMaeMs << "\n"
           << "      agent_queue_model_ms: raw_predicted_mean="
           << latencyValidation.queueRawPredictedMeanMs
           << ", fitted_predicted_mean=" << latencyValidation.agentPredictedQueueMeanMs
           << ", actual_mean=" << latencyValidation.actualQueueMeanMs
           << ", raw_bias_mean=" << latencyValidation.queueRawBiasMeanMs
           << ", raw_mae=" << latencyValidation.queueRawMaeMs
           << ", fitted_bias_mean=" << latencyValidation.agentQueueBiasMeanMs
           << ", fitted_mae=" << latencyValidation.agentQueueMaeMs
           << ", preceding_batch_count_mean="
           << latencyValidation.queueFeatureBatchCountMean
           << ", fit_step=" << latencyValidation.queueModelFitStep << ", weights=[";
    for (size_t index = 0; index < latencyValidation.queueModelWeights.size(); ++index) {
        if (index > 0) {
            output << ", ";
        }
        output << latencyValidation.queueModelWeights[index];
    }
    output << "]\n"
           << "      queue_zero_prediction: samples="
           << latencyValidation.zeroPredictedQueueSampleCount
           << ", actual_mean=" << latencyValidation.zeroPredictedQueueActualMeanMs
           << ", mae=" << latencyValidation.zeroPredictedQueueMaeMs << "\n"
           << "      queue_positive_prediction: samples="
           << latencyValidation.positivePredictedQueueSampleCount
           << ", predicted_mean=" << latencyValidation.positivePredictedQueueMeanMs
           << ", actual_mean=" << latencyValidation.positivePredictedQueueActualMeanMs
           << ", bias_mean=" << latencyValidation.positivePredictedQueueBiasMeanMs
           << ", mae=" << latencyValidation.positivePredictedQueueMaeMs << "\n"
           << "      execution_ms: master_predicted_mean="
           << latencyValidation.masterPredictedExecutionMeanMs
           << ", agent_pre_update_predicted_mean="
           << latencyValidation.agentPredictedExecutionMeanMs
           << ", actual_mean=" << latencyValidation.actualExecutionMeanMs
           << ", master_bias_mean=" << latencyValidation.masterExecutionBiasMeanMs
           << ", master_mae=" << latencyValidation.masterExecutionMaeMs
           << ", master_mape_pct=" << latencyValidation.masterExecutionMapePct
           << ", agent_bias_mean=" << latencyValidation.agentExecutionBiasMeanMs
           << ", agent_mae=" << latencyValidation.agentExecutionMaeMs
           << ", agent_mape_pct=" << latencyValidation.agentExecutionMapePct << "\n"
           << "      total_ms: predicted_mean="
           << latencyValidation.masterPredictedTotalMeanMs
           << ", actual_mean=" << latencyValidation.actualTotalMeanMs
           << ", bias_mean=" << latencyValidation.totalBiasMeanMs
           << ", mae=" << latencyValidation.totalMaeMs
           << ", mape_pct=" << latencyValidation.totalMapePct << "\n"
           << "    last_event: " << lastEventType << ", event_time_us=" << lastEventTimeUs
           << ", receive_time_us=" << lastReceiveTimeUs;
    return output.str();
}

void NaviSchedMasterSummary::Jsonize(JsonWrapper &json)
{
    NaviSchedRoleSummary::Jsonize(json);
    json.Jsonize("master_biz_name", masterBizName);
    json.Jsonize("collector_count", collectorCount);
    json.Jsonize("schedule_graph_running", scheduleGraphRunning);
    json.Jsonize("schedule_graph_state", scheduleGraphState);
    json.Jsonize("schedule_graph_restart_count", scheduleGraphRestartCount);
    json.Jsonize("schedule_request_pipe_generation", scheduleRequestPipeGeneration);
    json.Jsonize("schedule_submit_count", scheduleSubmitCount);
    json.Jsonize("schedule_pending_request_count", schedulePendingRequestCount);
    json.Jsonize("schedule_processing_request_count", scheduleProcessingRequestCount);
    json.Jsonize("active_request_count", activeRequestCount);
    json.Jsonize("schedule_round_count", scheduleRoundCount);
    json.Jsonize("schedule_round_request_count", scheduleRoundRequestCount);
    json.Jsonize("schedule_round_requests_mean", scheduleRoundRequestsMean);
    json.Jsonize("schedule_round_requests_min", scheduleRoundRequestsMin);
    json.Jsonize("schedule_round_requests_max", scheduleRoundRequestsMax);
    json.Jsonize("schedule_round_request_count_histogram",
                 scheduleRoundRequestCountHistogram);
    json.Jsonize("max_wait_schedule_count", maxWaitScheduleCount);
    json.Jsonize("max_wait_schedule_time_ms", maxWaitScheduleTimeMs);
    json.Jsonize("queue_overload_start_cycle_multiplier",
                 queueOverloadStartCycleMultiplier);
    json.Jsonize("queue_overload_full_cycle_multiplier",
                 queueOverloadFullCycleMultiplier);
    json.Jsonize("queue_overload_evaluated_round_count",
                 queueOverloadEvaluatedRoundCount);
    json.Jsonize("queue_overload_evaluated_request_count",
                 queueOverloadEvaluatedRequestCount);
    json.Jsonize("queue_overload_rejected_request_count",
                 queueOverloadRejectedRequestCount);
    json.Jsonize("queue_overload_latest_average_predicted_latency_ms",
                 queueOverloadLatestAveragePredictedLatencyMs);
    json.Jsonize("queue_overload_latest_drop_probability",
                 queueOverloadLatestDropProbability);
    json.Jsonize("execution_utilization_control_enabled",
                 executionUtilizationControlEnabled);
    json.Jsonize("execution_utilization_average", executionUtilizationAverage);
    json.Jsonize("execution_utilization_raw_average", executionUtilizationRawAverage);
    json.Jsonize("execution_utilization_running", executionUtilizationRunning);
    json.Jsonize("execution_utilization_waiting", executionUtilizationWaiting);
    json.Jsonize("execution_utilization_global_scaling",
                 executionUtilizationGlobalScaling);
    json.Jsonize("execution_utilization_integral", executionUtilizationIntegral);
    json.Jsonize("execution_utilization_reporting_nodes",
                 executionUtilizationReportingNodes);
    json.Jsonize("scheduler_config_generation", schedulerConfigGeneration);
    json.Jsonize("scheduler_config_hash", schedulerConfigHash);
    json.Jsonize("scheduler_config_json", schedulerConfigJson);
    json.Jsonize("agent_request_received_count", agentRequestReceivedCount);
    json.Jsonize("agent_request_finished_count", agentRequestFinishedCount);
    json.Jsonize("agent_workload_update_count", agentWorkloadUpdateCount);
    json.Jsonize("agent_request_cache_hit_tokens", agentRequestCacheHitTokens);
    json.Jsonize("agent_request_cache_miss_tokens", agentRequestCacheMissTokens);
    json.Jsonize("agent_request_total_latency_us", agentRequestTotalLatencyUs);
    json.Jsonize("nodes", nodes);
}

std::string NaviSchedMasterSummary::format() const
{
    std::ostringstream output;
    output << NaviSchedRoleSummary::format() << "\n\nMaster\n"
           << "  biz_name: " << masterBizName << "\n"
           << "  collector_count: " << collectorCount << "\n"
           << "  schedule_graph_running: " << (scheduleGraphRunning ? "true" : "false")
           << "\n"
           << "  schedule_graph_state: " << scheduleGraphState << "\n"
           << "  schedule_graph_restart_count: " << scheduleGraphRestartCount << "\n"
           << "  schedule_request_pipe_generation: " << scheduleRequestPipeGeneration
           << "\n"
           << "  schedule_submit_count: " << scheduleSubmitCount << "\n"
           << "  schedule_pending_request_count: " << schedulePendingRequestCount << "\n"
           << "  schedule_round_count: " << scheduleRoundCount << "\n"
           << "  schedule_round_request_count: " << scheduleRoundRequestCount << "\n"
           << std::fixed << std::setprecision(3)
           << "  schedule_round_requests_mean: " << scheduleRoundRequestsMean << "\n"
           << "  schedule_round_requests_min: " << scheduleRoundRequestsMin << "\n"
           << "  schedule_round_requests_max: " << scheduleRoundRequestsMax << "\n"
           << "  schedule_round_request_count_histogram:";
    if (scheduleRoundRequestCountHistogram.empty()) {
        output << " empty";
    } else {
        for (const auto &[requestCount, roundCount] :
             scheduleRoundRequestCountHistogram) {
            output << " " << requestCount << ":" << roundCount;
        }
    }
    output << "\n"
           << "  max_wait_schedule_count: " << maxWaitScheduleCount << "\n"
           << "  max_wait_schedule_time_ms: " << maxWaitScheduleTimeMs
           << "\n  queue_overload_start_cycle_multiplier: "
           << queueOverloadStartCycleMultiplier
           << "\n  queue_overload_full_cycle_multiplier: "
           << queueOverloadFullCycleMultiplier
           << "\n  queue_overload_evaluated_round_count: "
           << queueOverloadEvaluatedRoundCount
           << "\n  queue_overload_evaluated_request_count: "
           << queueOverloadEvaluatedRequestCount
           << "\n  queue_overload_rejected_request_count: "
           << queueOverloadRejectedRequestCount
           << "\n  queue_overload_latest_average_predicted_latency_ms: "
           << queueOverloadLatestAveragePredictedLatencyMs
           << "\n  queue_overload_latest_drop_probability: "
           << queueOverloadLatestDropProbability
           << "\n  execution_utilization_control_enabled: "
           << (executionUtilizationControlEnabled ? "true" : "false")
           << "\n  execution_utilization_raw_average: " << executionUtilizationRawAverage
           << "\n  execution_utilization_running: " << executionUtilizationRunning
           << "\n  execution_utilization_waiting: " << executionUtilizationWaiting
           << "\n  execution_utilization_global_scaling: "
           << executionUtilizationGlobalScaling
           << "\n  execution_utilization_average: " << executionUtilizationAverage
           << "\n  execution_utilization_integral: " << executionUtilizationIntegral
           << "\n  execution_utilization_reporting_nodes: "
           << executionUtilizationReportingNodes
           << "\n  schedule_processing_request_count: " << scheduleProcessingRequestCount
           << "\n  active_request_count: " << activeRequestCount
           << "\n  scheduler_config_generation: " << schedulerConfigGeneration
           << "\n  scheduler_config_hash: " << schedulerConfigHash
           << "\n  scheduler_config_json: " << schedulerConfigJson
           << "\n\nAgent requests\n"
           << "  received_requests: " << agentRequestReceivedCount << "\n"
           << "  finished_requests: " << agentRequestFinishedCount << "\n"
           << "  workload_updates: " << agentWorkloadUpdateCount << "\n"
           << "  cache_hit_tokens: " << agentRequestCacheHitTokens << "\n"
           << "  cache_miss_tokens: " << agentRequestCacheMissTokens << "\n"
           << "  total_latency_us: " << agentRequestTotalLatencyUs << "\n\nAgents\n"
           << "  count: " << nodes.size();
    for (const auto &node : nodes) {
        output << "\n" << node.format();
    }
    return output.str();
}

}
