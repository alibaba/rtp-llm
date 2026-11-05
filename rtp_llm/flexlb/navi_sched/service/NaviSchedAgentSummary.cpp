#include "aios/navi_sched/service/NaviSchedAgentSummary.h"

namespace navi_sched {

void NaviSchedAgentSummary::Jsonize(JsonWrapper &json)
{
    NaviSchedRoleSummary::Jsonize(json);
    json.Jsonize("collector_count", collectorCount);
    json.Jsonize("server_ip", serverIp);
    json.Jsonize("server_port", serverPort);
    json.Jsonize("pd_separated", pdSeparated);
    json.Jsonize("execution_busy_ema", executionBusyEma);
    json.Jsonize("execution_busy_window", executionBusyWindow);
    json.Jsonize("execution_window_busy_us", executionWindowBusyUs);
    json.Jsonize("execution_window_elapsed_us", executionWindowElapsedUs);
    json.Jsonize("execution_window_sample_mono_us", executionWindowSampleMonoUs);
    json.Jsonize("execution_capacity_ema", executionCapacity);
    json.Jsonize("execution_capacity_denominator", executionCapacityDenominator);
    json.Jsonize("gpu_ready_queued_requests", gpuReadyQueuedRequests);
    json.Jsonize("execution_active_count", executionActiveCount);
    json.Jsonize("execution_sample_busy_us", executionSampleBusyUs);
    json.Jsonize("execution_sample_elapsed_us", executionSampleElapsedUs);
    json.Jsonize("execution_running_requests", executionRunningRequests);
    json.Jsonize("execution_queued_requests", executionQueuedRequests);
    json.Jsonize("execution_utilization", executionUtilization);
    json.Jsonize("workload_revision", workload.revision);
    json.Jsonize("native_workload_batch_count", nativeWorkloadBatchCount);
    json.Jsonize("native_workload_request_count", nativeWorkloadRequestCount);
    json.Jsonize("native_unbatched_request_count", nativeUnbatchedRequestCount);
    json.Jsonize("raw_observer_event_time_us", rawObserverEventTimeUs);
    json.Jsonize("workload_event_time_us", workload.eventTimeUs);
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
    json.Jsonize("pending_batch_count", pendingBatchCount);
    json.Jsonize("inflight_stats_revision", inflightStatsRevision);
    json.Jsonize("inflight_requests", inflightRequests);
    json.Jsonize("inflight_predicted_latency_sum_ms", inflightPredictedLatencySumMs);
    json.Jsonize("inflight_invalid_add_count", inflightInvalidAddCount);
    json.Jsonize("inflight_duplicate_add_count", inflightDuplicateAddCount);
    json.Jsonize("inflight_missing_remove_count", inflightMissingRemoveCount);
    json.Jsonize("inflight_recompute_count", inflightRecomputeCount);
    json.Jsonize("inflight_recomputed_request_count", inflightRecomputedRequestCount);
    json.Jsonize("inflight_recompute_failure_count", inflightRecomputeFailureCount);
    json.Jsonize("inflight_invalid_state_transition_count",
                 inflightInvalidStateTransitionCount);
    json.Jsonize("cached_block_count", cachedBlockCount);
    json.Jsonize("cache_revision", cacheRevision);
    json.Jsonize("cache_hash_checksum", cacheHashChecksum);
    json.Jsonize("full_ready_block_count", fullReadyBlockCount);
    json.Jsonize("full_ready_hash_checksum", fullReadyHashChecksum);
    json.Jsonize("tail_ready_block_count", tailReadyBlockCount);
    json.Jsonize("tail_ready_hash_checksum", tailReadyHashChecksum);
    json.Jsonize("rtp_cached_block_count", rtpCachedBlockCount);
    json.Jsonize("cache_hit_path_request_count", cacheHitPathRequestCount);
    json.Jsonize("cache_hit_master_predicted_mean", cacheHitMasterPredictedMean);
    json.Jsonize("cache_hit_agent_arrival_mean", cacheHitAgentArrivalMean);
    json.Jsonize("cache_hit_agent_pending_aware_mean", cacheHitAgentPendingAwareMean);
    json.Jsonize("cache_hit_rtp_actual_mean", cacheHitRtpActualMean);
    json.Jsonize("cache_hit_master_to_agent_bias_mean", cacheHitMasterToAgentBiasMean);
    json.Jsonize("cache_hit_agent_to_rtp_bias_mean", cacheHitAgentToRtpBiasMean);
    json.Jsonize("cache_hit_master_to_agent_pending_aware_bias_mean",
                 cacheHitMasterToAgentPendingAwareBiasMean);
    json.Jsonize("cache_hit_agent_pending_aware_to_rtp_bias_mean",
                 cacheHitAgentPendingAwareToRtpBiasMean);
    json.Jsonize("cache_hit_master_to_rtp_bias_mean", cacheHitMasterToRtpBiasMean);
    json.Jsonize("cache_hit_aligned_request_count", cacheHitAlignedRequestCount);
    json.Jsonize("cache_hit_aligned_agent_to_rtp_bias_mean",
                 cacheHitAlignedAgentToRtpBiasMean);
    json.Jsonize("cache_hit_aligned_agent_to_rtp_mae", cacheHitAlignedAgentToRtpMae);
    json.Jsonize("cache_hit_unaligned_request_count", cacheHitUnalignedRequestCount);
    json.Jsonize("cache_hit_unaligned_agent_to_rtp_bias_mean",
                 cacheHitUnalignedAgentToRtpBiasMean);
    json.Jsonize("cache_hit_unaligned_agent_to_rtp_mae", cacheHitUnalignedAgentToRtpMae);
    json.Jsonize("hybrid_reuse_debug_request_count", hybridReuseDebugRequestCount);
    json.Jsonize("hybrid_reuse_agent_predicted_blocks_mean",
                 hybridReuseAgentPredictedBlocksMean);
    json.Jsonize("hybrid_reuse_candidate_blocks_mean", hybridReuseCandidateBlocksMean);
    json.Jsonize("hybrid_reuse_flat_matched_blocks_mean",
                 hybridReuseFlatMatchedBlocksMean);
    json.Jsonize("hybrid_reuse_min_full_blocks_mean", hybridReuseMinFullBlocksMean);
    json.Jsonize("hybrid_reuse_selected_blocks_mean", hybridReuseSelectedBlocksMean);
    json.Jsonize("hybrid_reuse_agent_to_full_overprediction_blocks_mean",
                 hybridReuseAgentToFullOverpredictionBlocksMean);
    json.Jsonize("hybrid_reuse_tail_rollback_blocks_mean",
                 hybridReuseTailRollbackBlocksMean);
    json.Jsonize("hybrid_reuse_invalid_order_count", hybridReuseInvalidOrderCount);
    json.Jsonize("ready_without_running_gap_count", readyWithoutRunningGapCount);
    json.Jsonize("ready_without_running_total_ms", readyWithoutRunningTotalMs);
    json.Jsonize("ready_without_running_mean_ms", readyWithoutRunningMeanMs);
    json.Jsonize("ready_without_running_max_ms", readyWithoutRunningMaxMs);
    json.Jsonize("ready_without_running_current_ms", readyWithoutRunningCurrentMs);
}

std::string NaviSchedAgentSummary::format() const
{
    return NaviSchedRoleSummary::format() +
           "\n\nAgent\n"
           "  collector_count: " +
           std::to_string(collectorCount) +
           "\n"
           "  server: " +
           serverIp + ":" + std::to_string(serverPort) +
           "\n"
           "  pd_separated: " +
           (pdSeparated ? "true" : "false") + "\n  execution_capacity_denominator: " +
           std::to_string(executionCapacityDenominator) +
           "\n  execution_capacity_ema: " + std::to_string(executionCapacity) +
           "\n  gpu_ready_queued_requests: " + std::to_string(gpuReadyQueuedRequests) +
           "\n  execution_busy_ema: " + std::to_string(executionBusyEma) +
           "\n  execution_busy_window: " + std::to_string(executionBusyWindow) +
           "\n  execution_window_busy_us: " + std::to_string(executionWindowBusyUs) +
           "\n  execution_window_elapsed_us: " +
           std::to_string(executionWindowElapsedUs) +
           "\n  execution_window_sample_mono_us: " +
           std::to_string(executionWindowSampleMonoUs) +
           "\n  execution_active_count: " + std::to_string(executionActiveCount) +
           "\n  execution_sample_busy_us: " + std::to_string(executionSampleBusyUs) +
           "\n  execution_sample_elapsed_us: " +
           std::to_string(executionSampleElapsedUs) +
           "\n  execution_running_requests: " + std::to_string(executionRunningRequests) +
           "\n  execution_queued_requests: " + std::to_string(executionQueuedRequests) +
           "\n  execution_utilization: " + std::to_string(executionUtilization) +
           "\n"
           "  workload: revision=" +
           std::to_string(workload.revision) +
           ", requests=" + std::to_string(workload.requestCount) +
           ", remaining_prefill_requests=" +
           std::to_string(workload.remainingPrefillRequestCount) +
           ", running_elapsed_time_ms=" + std::to_string(workload.runningElapsedTimeMs) +
           ", running_decode_requests=" +
           std::to_string(workload.runningDecodeRequestCount) +
           ", running_decode_generated_tokens=" +
           std::to_string(workload.runningDecodeGeneratedTokenCount) +
           ", running_decode_remaining_token_budget=" +
           std::to_string(workload.runningDecodeRemainingTokenBudget) +
           ", remaining_prefill_context_tokens=" +
           std::to_string(workload.remainingPrefillContextTokenCount) +
           ", remaining_prefill_tokens=" +
           std::to_string(workload.remainingPrefillTokenCount) +
           ", capacity_split=" + (workload.prefillCapacitySplit ? "true" : "false") +
           ", running_capacity_slots=" +
           std::to_string(workload.runningCapacitySlotCount) + "/" +
           std::to_string(workload.maxRunningRequestCount) +
           ", running_capacity_prefill_requests=" +
           std::to_string(workload.runningCapacityPrefillRequestCount) +
           ", waiting_capacity_prefill_requests=" +
           std::to_string(workload.waitingCapacityPrefillRequestCount) +
           ", remaining_total_context_tokens=" +
           std::to_string(workload.remainingTotalContextTokenCount) +
           ", remaining_total_tokens=" +
           std::to_string(workload.remainingTotalTokenCount) +
           ", inflight_scheduled_tokens=" +
           std::to_string(workload.inFlightScheduledTokenCount) +
           "\n"
           "  native_workload: batches=" +
           std::to_string(nativeWorkloadBatchCount) +
           ", requests=" + std::to_string(nativeWorkloadRequestCount) +
           ", unbatched_requests=" + std::to_string(nativeUnbatchedRequestCount) +
           ", raw_observer_event_time_us=" + std::to_string(rawObserverEventTimeUs) +
           "\n"
           "  batches: pending=" +
           std::to_string(pendingBatchCount) +
           "\n"
           "  inflight_requests: " +
           std::to_string(inflightRequests) +
           ", revision=" + std::to_string(inflightStatsRevision) +
           ", predicted_latency_sum_ms=" + std::to_string(inflightPredictedLatencySumMs) +
           ", invalid_add_count=" + std::to_string(inflightInvalidAddCount) +
           ", duplicate_add_count=" + std::to_string(inflightDuplicateAddCount) +
           ", missing_remove_count=" + std::to_string(inflightMissingRemoveCount) +
           ", recompute_count=" + std::to_string(inflightRecomputeCount) +
           ", recomputed_request_count=" +
           std::to_string(inflightRecomputedRequestCount) +
           ", recompute_failure_count=" + std::to_string(inflightRecomputeFailureCount) +
           ", invalid_state_transition_count=" +
           std::to_string(inflightInvalidStateTransitionCount) +
           "\n"
           "  cached_block_count: " +
           std::to_string(cachedBlockCount) +
           "\n"
           "  cache_revision: " +
           std::to_string(cacheRevision) +
           "\n"
           "  cache_hash_checksum: " +
           std::to_string(cacheHashChecksum) +
           "\n"
           "  hybrid_cache_state: full_ready=" +
           std::to_string(fullReadyBlockCount) +
           ", full_checksum=" + std::to_string(fullReadyHashChecksum) +
           ", tail_ready=" + std::to_string(tailReadyBlockCount) +
           ", tail_checksum=" + std::to_string(tailReadyHashChecksum) +
           "\n"
           "  rtp_cached_block_count: " +
           std::to_string(rtpCachedBlockCount) +
           "\n"
           "  cache_hit_path: requests=" +
           std::to_string(cacheHitPathRequestCount) +
           ", master_predicted_mean=" + std::to_string(cacheHitMasterPredictedMean) +
           ", agent_arrival_mean=" + std::to_string(cacheHitAgentArrivalMean) +
           ", agent_pending_aware_mean=" + std::to_string(cacheHitAgentPendingAwareMean) +
           ", rtp_actual_mean=" + std::to_string(cacheHitRtpActualMean) +
           ", master_to_agent_bias_mean=" +
           std::to_string(cacheHitMasterToAgentBiasMean) +
           ", agent_to_rtp_bias_mean=" + std::to_string(cacheHitAgentToRtpBiasMean) +
           ", master_to_agent_pending_aware_bias_mean=" +
           std::to_string(cacheHitMasterToAgentPendingAwareBiasMean) +
           ", agent_pending_aware_to_rtp_bias_mean=" +
           std::to_string(cacheHitAgentPendingAwareToRtpBiasMean) +
           ", master_to_rtp_bias_mean=" + std::to_string(cacheHitMasterToRtpBiasMean) +
           "\n"
           "  cache_hit_alignment: aligned_requests=" +
           std::to_string(cacheHitAlignedRequestCount) +
           ", aligned_agent_to_rtp_bias_mean=" +
           std::to_string(cacheHitAlignedAgentToRtpBiasMean) +
           ", aligned_agent_to_rtp_mae=" + std::to_string(cacheHitAlignedAgentToRtpMae) +
           ", unaligned_requests=" + std::to_string(cacheHitUnalignedRequestCount) +
           ", unaligned_agent_to_rtp_bias_mean=" +
           std::to_string(cacheHitUnalignedAgentToRtpBiasMean) +
           ", unaligned_agent_to_rtp_mae=" +
           std::to_string(cacheHitUnalignedAgentToRtpMae) +
           "\n"
           "  hybrid_reuse_debug: requests=" +
           std::to_string(hybridReuseDebugRequestCount) +
           ", agent_predicted_blocks_mean=" +
           std::to_string(hybridReuseAgentPredictedBlocksMean) +
           ", candidate_blocks_mean=" + std::to_string(hybridReuseCandidateBlocksMean) +
           ", flat_matched_blocks_mean=" +
           std::to_string(hybridReuseFlatMatchedBlocksMean) +
           ", min_full_blocks_mean=" + std::to_string(hybridReuseMinFullBlocksMean) +
           ", selected_blocks_mean=" + std::to_string(hybridReuseSelectedBlocksMean) +
           ", agent_to_full_overprediction_blocks_mean=" +
           std::to_string(hybridReuseAgentToFullOverpredictionBlocksMean) +
           ", tail_rollback_blocks_mean=" +
           std::to_string(hybridReuseTailRollbackBlocksMean) +
           ", invalid_order_count=" + std::to_string(hybridReuseInvalidOrderCount) +
           "\n"
           "  ready_without_running: gaps=" +
           std::to_string(readyWithoutRunningGapCount) +
           ", total_ms=" + std::to_string(readyWithoutRunningTotalMs) +
           ", mean_ms=" + std::to_string(readyWithoutRunningMeanMs) +
           ", max_ms=" + std::to_string(readyWithoutRunningMaxMs) +
           ", current_ms=" + std::to_string(readyWithoutRunningCurrentMs);
}

}
