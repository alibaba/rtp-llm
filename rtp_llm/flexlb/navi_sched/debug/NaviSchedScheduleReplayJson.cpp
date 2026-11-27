#include <algorithm>
#include <stdexcept>
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

struct ReplayNativeCacheQueryJson : public autil::legacy::Jsonizable {
    NaviSchedNativeCacheQuery value;
    ReplayNativeCacheQueryJson() = default;
    explicit ReplayNativeCacheQueryJson(const NaviSchedNativeCacheQuery &input)
        : value(input)
    {
    }
    void Jsonize(JsonWrapper &json) override
    {
        auto source = int32_t(value.inputSource);
        auto future = int32_t(value.futureState);
        auto load = int32_t(value.loadState);
        auto receive = int32_t(value.receiveState);
        json.Jsonize("request_id", value.requestId);
        json.Jsonize("generation", value.generation);
        json.Jsonize("local_prefix_tokens", value.localPrefixTokenCount);
        json.Jsonize("input_source", source);
        json.Jsonize("future_state", future);
        json.Jsonize("load_state", load);
        json.Jsonize("receive_state", receive);
        if (json.GetMode() != JsonMode::TO_JSON) {
            value.inputSource = static_cast<NativeCacheQueryInputSource>(source);
            value.futureState = static_cast<NativeCacheQueryFutureState>(future);
            value.loadState = static_cast<NativeCacheLoadState>(load);
            value.receiveState = static_cast<NativeCacheReceiveState>(receive);
        }
    }
};

struct ReplayNativePPSpanJson : public autil::legacy::Jsonizable {
    NaviSchedNativePPSpan value;
    ReplayNativePPSpanJson() = default;
    explicit ReplayNativePPSpanJson(const NaviSchedNativePPSpan &input) : value(input) {}
    void Jsonize(JsonWrapper &json) override
    {
        json.Jsonize("request_id", value.requestId, value.requestId);
        json.Jsonize("computed_tokens", value.computedTokens, value.computedTokens);
        json.Jsonize("scheduled_tokens", value.scheduledTokens, value.scheduledTokens);
    }
};

struct ReplayNativePPSlotJson : public autil::legacy::Jsonizable {
    NaviSchedNativePPSlot value;
    ReplayNativePPSlotJson() = default;
    explicit ReplayNativePPSlotJson(const NaviSchedNativePPSlot &input) : value(input) {}
    void Jsonize(JsonWrapper &json) override
    {
        json.Jsonize("ordinal", value.ordinal, value.ordinal);
        json.Jsonize("dispatch_age_us", value.dispatchAgeUs, value.dispatchAgeUs);
        json.Jsonize("expected_forward_ms", value.expectedForwardMs,
                     value.expectedForwardMs);
        json.Jsonize("last_stage_reply_age_us", value.lastStageReplyAgeUs,
                     value.lastStageReplyAgeUs);
        std::vector<double> features(value.features.begin(), value.features.end());
        json.Jsonize("features", features, features);
        if (json.GetMode() != JsonMode::TO_JSON) {
            if (features.size() != 8)
                throw std::invalid_argument("PP stage geometry requires eight features");
            std::copy(features.begin(), features.end(), value.features.begin());
        }
        std::vector<ReplayNativePPSpanJson> spans;
        for (const auto &item : value.spans) spans.emplace_back(item);
        json.Jsonize("spans", spans, spans);
        if (json.GetMode() != JsonMode::TO_JSON) {
            value.spans.clear();
            for (const auto &item : spans) value.spans.emplace_back(item.value);
        }
    }
};

struct ReplayNativePPResultJson : public autil::legacy::Jsonizable {
    NaviSchedNativePPResult value;
    ReplayNativePPResultJson() = default;
    explicit ReplayNativePPResultJson(const NaviSchedNativePPResult &input) : value(input)
    {
    }
    void Jsonize(JsonWrapper &json) override
    {
        json.Jsonize("scheduler_output_id", value.schedulerOutputId,
                     value.schedulerOutputId);
        json.Jsonize("load_request_ids", value.loadRequestIds);
        json.Jsonize("dispatch_age_us", value.dispatchAgeUs, value.dispatchAgeUs);
        json.Jsonize("slot_ordinal", value.slotOrdinal, value.slotOrdinal);
        json.Jsonize("execute_reply_ready", value.executeReplyReady,
                     value.executeReplyReady);
        json.Jsonize("result_ready", value.resultReady, value.resultReady);
    }
};

struct ReplayNativePPRequestJson : public autil::legacy::Jsonizable {
    NaviSchedNativePPRequest value;
    ReplayNativePPRequestJson() = default;
    explicit ReplayNativePPRequestJson(const NaviSchedNativePPRequest &input)
        : value(input)
    {
    }
    void Jsonize(JsonWrapper &json) override
    {
        json.Jsonize("request_id", value.requestId, value.requestId);
        auto queue = int32_t(value.queue);
        json.Jsonize("queue", queue, queue);
        if (json.GetMode() != JsonMode::TO_JSON)
            value.queue = static_cast<NativePPQueue>(queue);
        json.Jsonize("queue_index", value.queueIndex, value.queueIndex);
        auto phase = int32_t(value.phase);
        json.Jsonize("phase", phase, phase);
        if (json.GetMode() != JsonMode::TO_JSON)
            value.phase = static_cast<NativePPRequestPhase>(phase);
        json.Jsonize("prompt_tokens", value.promptTokens, value.promptTokens);
        json.Jsonize("computed_tokens", value.computedTokens, value.computedTokens);
        json.Jsonize("cache_hit_tokens", value.cacheHitTokens, value.cacheHitTokens);
        json.Jsonize("shared_prefix_boundary", value.sharedPrefixBoundary,
                     value.sharedPrefixBoundary);
        json.Jsonize("query_age_us", value.queryAgeUs, value.queryAgeUs);
        json.Jsonize("load_age_us", value.loadAgeUs, value.loadAgeUs);
        auto release = int32_t(value.prefillSlotRelease);
        json.Jsonize("prefill_slot_release", release);
        if (json.GetMode() != JsonMode::TO_JSON)
            value.prefillSlotRelease = static_cast<NativePPPrefillSlotRelease>(release);
    }
};

struct ReplayNativePPDecisionJson : public autil::legacy::Jsonizable {
    NaviSchedNativePPDecision value;
    ReplayNativePPDecisionJson() = default;
    explicit ReplayNativePPDecisionJson(const NaviSchedNativePPDecision &input)
        : value(input)
    {
    }
    void Jsonize(JsonWrapper &json) override
    {
        json.Jsonize("scheduler_output_id", value.schedulerOutputId,
                     value.schedulerOutputId);
        json.Jsonize("load_request_ids", value.loadRequestIds);
        std::vector<double> features(value.features.begin(), value.features.end());
        json.Jsonize("features", features, features);
        if (json.GetMode() != JsonMode::TO_JSON) {
            if (features.size() != 8)
                throw std::invalid_argument("PP stage geometry requires eight features");
            std::copy(features.begin(), features.end(), value.features.begin());
        }
        std::vector<ReplayNativePPSpanJson> spans;
        for (const auto &item : value.spans) spans.emplace_back(item);
        json.Jsonize("spans", spans, spans);
        if (json.GetMode() != JsonMode::TO_JSON) {
            value.spans.clear();
            for (const auto &item : spans) value.spans.emplace_back(item.value);
        }
    }
};

struct ReplayNativePPStateJson : public autil::legacy::Jsonizable {
    NaviSchedNativePPState value;
    ReplayNativePPStateJson() = default;
    explicit ReplayNativePPStateJson(const NaviSchedNativePPState &input) : value(input)
    {
    }
    void Jsonize(JsonWrapper &json) override
    {
        json.Jsonize("occupied_request_slot_count", value.occupiedRequestSlotCount);
        json.Jsonize("observation_id", value.observationId, value.observationId);
        json.Jsonize("burst", value.burst, value.burst);
        json.Jsonize("wave_open", value.waveOpen, value.waveOpen);
        json.Jsonize("last_real_ordinal", value.lastRealOrdinal, value.lastRealOrdinal);
        json.Jsonize("measured_forward_ms", value.measuredForwardMs,
                     value.measuredForwardMs);
        std::vector<ReplayNativePPSlotJson> slots;
        for (const auto &item : value.slots) slots.emplace_back(item);
        json.Jsonize("slots", slots, slots);
        if (json.GetMode() != JsonMode::TO_JSON) {
            value.slots.clear();
            for (const auto &item : slots) value.slots.emplace_back(item.value);
        }
        std::vector<ReplayNativePPResultJson> results;
        for (const auto &item : value.results) results.emplace_back(item);
        json.Jsonize("results", results, results);
        if (json.GetMode() != JsonMode::TO_JSON) {
            value.results.clear();
            for (const auto &item : results) value.results.emplace_back(item.value);
        }
        std::vector<ReplayNativePPRequestJson> requests;
        for (const auto &item : value.requests) requests.emplace_back(item);
        json.Jsonize("requests", requests, requests);
        if (json.GetMode() != JsonMode::TO_JSON) {
            value.requests.clear();
            for (const auto &item : requests) value.requests.emplace_back(item.value);
        }
        bool hasDecision =
            json.GetMode() == JsonMode::TO_JSON && bool(value.pendingDecision);
        json.Jsonize("pending_decision_present", hasDecision, hasDecision);
        if (hasDecision) {
            ReplayNativePPDecisionJson decision;
            if (value.pendingDecision) decision.value = *value.pendingDecision;
            json.Jsonize("pending_decision", decision);
            if (json.GetMode() != JsonMode::TO_JSON)
                value.pendingDecision = std::move(decision.value);
        } else if (json.GetMode() != JsonMode::TO_JSON)
            value.pendingDecision.reset();
    }
};

struct ReplayKVPreparationJson : public autil::legacy::Jsonizable {
    NaviSchedKVPreparation value;
    ReplayKVPreparationJson() = default;
    explicit ReplayKVPreparationJson(const NaviSchedKVPreparation &input) : value(input) {}
    void Jsonize(JsonWrapper &json) override {
        json.Jsonize("request_id", value.requestId);
        json.Jsonize("local_batch_id", value.localBatchId);
        json.Jsonize("load_generation", value.loadGeneration);
        json.Jsonize("preparation_elapsed_ms", value.preparationElapsedMs, -1.0);
        json.Jsonize("context_tokens", value.contextTokens);
        json.Jsonize("compute_tokens", value.computeTokens);
        json.Jsonize("running_capacity", value.runningCapacity);
        int32_t state = int32_t(value.state);
        json.Jsonize("state", state);
        if (json.GetMode() != JsonMode::TO_JSON) value.state = NativeKVExecutionState(state);
    }
};

struct ReplayWorkloadJson : public autil::legacy::Jsonizable {
    NaviSchedWorkloadSnapshot value;
    ReplayWorkloadJson() = default;
    explicit ReplayWorkloadJson(const NaviSchedWorkloadSnapshot &input) : value(input) {}
    void Jsonize(JsonWrapper &json) override
    {
        json.Jsonize("revision", value.revision);
        json.Jsonize("execution_load_tracked", value.executionLoadTracked, value.executionLoadTracked);
        std::vector<ReplayKVPreparationJson> executionLoads;
        for (const auto &load : value.executionLoadRequests) executionLoads.emplace_back(load);
        json.Jsonize("execution_load_requests", executionLoads, executionLoads);
        if (json.GetMode() != JsonMode::TO_JSON) {
            value.executionLoadRequests.clear();
            for (const auto &load : executionLoads) value.executionLoadRequests.push_back(load.value);
        }
        json.Jsonize("event_time_us", value.eventTimeUs);
        json.Jsonize("batch_count", value.batchCount);
        json.Jsonize("request_count", value.requestCount);
        json.Jsonize("remaining_prefill_request_count",
                     value.remainingPrefillRequestCount);
        json.Jsonize("running_request_count", value.runningRequestCount);
        json.Jsonize("running_elapsed_time_ms", value.runningElapsedTimeMs);
        json.Jsonize("running_elapsed_request_count", value.runningElapsedRequestCount);
        json.Jsonize("running_decode_request_count", value.runningDecodeRequestCount,
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
        json.Jsonize("remaining_prefill_token_count", value.remainingPrefillTokenCount);
        json.Jsonize("remaining_prefill_token_square_sum",
                     value.remainingPrefillTokenSquareSum);
        json.Jsonize("context_remaining_prefill_token_product_sum",
                     value.contextRemainingPrefillTokenProductSum);
        json.Jsonize("prefill_capacity_split", value.prefillCapacitySplit,
                     value.prefillCapacitySplit);
        json.Jsonize("max_running_request_count", value.maxRunningRequestCount,
                     value.maxRunningRequestCount);
        json.Jsonize("running_capacity_slot_count", value.runningCapacitySlotCount,
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
        json.Jsonize("running_capacity_context_remaining_prefill_token_product_sum",
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
        json.Jsonize("waiting_capacity_context_remaining_prefill_token_product_sum",
                     value.waitingCapacityContextRemainingPrefillTokenProductSum,
                     value.waitingCapacityContextRemainingPrefillTokenProductSum);
        json.Jsonize("remaining_total_context_token_count",
                     value.remainingTotalContextTokenCount);
        json.Jsonize("remaining_total_token_count", value.remainingTotalTokenCount);
        json.Jsonize("remaining_total_token_square_sum",
                     value.remainingTotalTokenSquareSum);
        json.Jsonize("context_remaining_total_token_product_sum",
                     value.contextRemainingTotalTokenProductSum);
        json.Jsonize("total_prompt_token_count", value.totalPromptTokenCount);
        json.Jsonize("in_flight_scheduled_token_count",
                     value.inFlightScheduledTokenCount);
        bool hasNativePP =
            json.GetMode() == JsonMode::TO_JSON && bool(value.nativePPState);
        json.Jsonize("native_pp_state_present", hasNativePP, hasNativePP);
        if (hasNativePP) {
            ReplayNativePPStateJson observed;
            if (value.nativePPState) observed.value = *value.nativePPState;
            json.Jsonize("native_pp_state", observed);
            json.Jsonize("native_pp_observation_age_us", value.nativePPObservationAgeUs);
            if (json.GetMode() != JsonMode::TO_JSON)
                value.nativePPState = std::make_shared<const NaviSchedNativePPState>(
                    std::move(observed.value));
        } else if (json.GetMode() != JsonMode::TO_JSON) {
            value.nativePPState.reset();
            value.nativePPObservationAgeUs = -1;
        }
        auto admittedScheduleBatchId = value.admittedBatch.scheduleBatchId;
        auto admittedLocalBatchId = value.admittedBatch.localBatchId;
        auto admittedRequestCount = value.admittedBatch.requestCount;
        auto admittedRequestIds = value.admittedBatch.requestIds;
        json.Jsonize("admitted_schedule_batch_id", admittedScheduleBatchId);
        json.Jsonize("admitted_local_batch_id", admittedLocalBatchId);
        json.Jsonize("admitted_request_count", admittedRequestCount);
        json.Jsonize("admitted_request_ids", admittedRequestIds);
        if (json.GetMode() != JsonMode::TO_JSON || !value.nativeCacheQueries.empty()) {
            std::vector<ReplayNativeCacheQueryJson> queries;
            for (const auto &query : value.nativeCacheQueries)
                queries.emplace_back(query);
            json.Jsonize("native_cache_queries", queries, queries);
            if (json.GetMode() != JsonMode::TO_JSON) {
                value.nativeCacheQueries.clear();
                for (const auto &query : queries)
                    value.nativeCacheQueries.emplace_back(query.value);
            }
        }
    }
};

struct ReplayExpectedKVWorkloadJson : public autil::legacy::Jsonizable {
    NaviSchedExpectedKVWorkload value;
    void Jsonize(JsonWrapper &json) override {
        std::vector<double> running(value.running.begin(),value.running.end());
        std::vector<double> waiting(value.waiting.begin(),value.waiting.end());
        json.Jsonize("running",running);json.Jsonize("waiting",waiting);
        json.Jsonize("preceding_requests",value.precedingRequests);
        json.Jsonize("batch_count",value.batchCount);
        if(running.size()!=5 || waiting.size()!=5)
            throw std::invalid_argument("expected KV workload requires five moments per capacity");
        std::copy(running.begin(),running.end(),value.running.begin());
        std::copy(waiting.begin(),waiting.end(),value.waiting.begin());
        if(!value.valid()) throw std::invalid_argument("invalid expected KV workload");
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
        std::optional<NaviSchedExpectedKVWorkload> expected;
        if(json.GetMode()==JsonMode::TO_JSON) {
            if(const auto p=value.getExpectedKVWorkload()) {
                ReplayExpectedKVWorkloadJson encoded;encoded.value=*p;
                json.Jsonize("expected_kv_workload",encoded);
            }
        } else if(json.GetMap().count("expected_kv_workload")) {
            ReplayExpectedKVWorkloadJson encoded;json.Jsonize("expected_kv_workload",encoded);expected=encoded.value;
        }
        auto batchCount = value.getBatchCount();
        auto precedingRequestCount = value.getPrecedingRequestCount();
        auto runningElapsedTimeMs = value.getRunningElapsedTimeMs();
        auto runningDecodeRequestCount = value.getRunningDecodeRequestCount();
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
        auto runningComputeTokensSquared = value.getRunningCacheMissTokenSquareSum();
        auto runningHitComputeProduct = value.getRunningCacheHitMissTokenProductSum();
        auto waitingBatchSize = value.getWaitingBatchSize();
        auto waitingHitTokens = value.getWaitingCacheHitTokenCount();
        auto waitingComputeTokens = value.getWaitingCacheMissTokenCount();
        auto waitingComputeTokensSquared = value.getWaitingCacheMissTokenSquareSum();
        auto waitingHitComputeProduct = value.getWaitingCacheHitMissTokenProductSum();
        json.Jsonize("batch_count", batchCount);
        json.Jsonize("preceding_request_count", precedingRequestCount,
                     precedingRequestCount);
        json.Jsonize("running_elapsed_time_ms", runningElapsedTimeMs,
                     runningElapsedTimeMs);
        json.Jsonize("running_decode_request_count", runningDecodeRequestCount,
                     runningDecodeRequestCount);
        json.Jsonize("running_decode_generated_token_count",
                     runningDecodeGeneratedTokenCount, runningDecodeGeneratedTokenCount);
        json.Jsonize("running_decode_remaining_token_budget",
                     runningDecodeRemainingTokenBudget,
                     runningDecodeRemainingTokenBudget);
        json.Jsonize("batch_size", batchSize);
        json.Jsonize("hit_tokens", hitTokens);
        json.Jsonize("compute_tokens", computeTokens);
        json.Jsonize("compute_tokens_squared", computeTokensSquared);
        json.Jsonize("hit_compute_product", hitComputeProduct);
        json.Jsonize("capacity_split", capacitySplit, capacitySplit);
        json.Jsonize("running_batch_size", runningBatchSize, runningBatchSize);
        json.Jsonize("running_hit_tokens", runningHitTokens, runningHitTokens);
        json.Jsonize("running_compute_tokens", runningComputeTokens,
                     runningComputeTokens);
        json.Jsonize("running_compute_tokens_squared", runningComputeTokensSquared,
                     runningComputeTokensSquared);
        json.Jsonize("running_hit_compute_product", runningHitComputeProduct,
                     runningHitComputeProduct);
        json.Jsonize("waiting_batch_size", waitingBatchSize, waitingBatchSize);
        json.Jsonize("waiting_hit_tokens", waitingHitTokens, waitingHitTokens);
        json.Jsonize("waiting_compute_tokens", waitingComputeTokens,
                     waitingComputeTokens);
        json.Jsonize("waiting_compute_tokens_squared", waitingComputeTokensSquared,
                     waitingComputeTokensSquared);
        json.Jsonize("waiting_hit_compute_product", waitingHitComputeProduct,
                     waitingHitComputeProduct);
        int64_t executionSteps = 0;
        int64_t paddedTokens = 0;
        bool geometryPresent = false;
        bool sourcePresent = false;
        int64_t sourceRequests = -1, sourceMaximum = -1;
        if (json.GetMode() == JsonMode::TO_JSON) {
            if (value.hasExecutionGeometry()) {
                executionSteps = value.getExecutionStepCount();
                paddedTokens = value.getGlobalPaddedTokens();
                json.Jsonize("execution_step_count", executionSteps);
                json.Jsonize("global_padded_tokens", paddedTokens);
            }
            if (value.hasExecutionCacheSource()) {
                sourceRequests = value.getExternalReadRequestCount();
                sourceMaximum = value.getMaxExternalReadTokens();
                json.Jsonize("external_read_request_count", sourceRequests);
                json.Jsonize("max_external_read_tokens", sourceMaximum);
            }
        } else {
            const auto fields = json.GetMap();
            geometryPresent = fields.count("execution_step_count") ||
                              fields.count("global_padded_tokens");
            json.Jsonize("execution_step_count", executionSteps, int64_t(0));
            json.Jsonize("global_padded_tokens", paddedTokens, int64_t(0));
            sourcePresent = fields.count("external_read_request_count") ||
                            fields.count("max_external_read_tokens");
            json.Jsonize("external_read_request_count", sourceRequests, int64_t(-1));
            json.Jsonize("max_external_read_tokens", sourceMaximum, int64_t(-1));
        }
        if (json.GetMode() != JsonMode::TO_JSON) {
            value = NaviSchedLatencyFeatures();
            if (capacitySplit) {
                value.appendCapacitySplitAggregate(
                    batchCount, runningBatchSize, runningHitTokens, runningComputeTokens,
                    runningComputeTokensSquared, runningHitComputeProduct,
                    waitingBatchSize, waitingHitTokens, waitingComputeTokens,
                    waitingComputeTokensSquared, waitingHitComputeProduct);
                value.appendPrecedingRequestCount(precedingRequestCount);
            } else {
                value.appendAggregate(batchCount, batchSize, hitTokens, computeTokens,
                                      computeTokensSquared, hitComputeProduct);
            }
            value.appendRunningElapsedTimeMs(runningElapsedTimeMs);
            value.appendRunningDecodeProgress(runningDecodeRequestCount,
                                              runningDecodeGeneratedTokenCount,
                                              runningDecodeRemainingTokenBudget);
            if (geometryPresent &&
                !value.setExecutionGeometry(executionSteps, paddedTokens)) {
                AUTIL_LEGACY_THROW(autil::legacy::NotJsonizableException,
                                   "invalid or incomplete execution geometry");
            }
            if (sourcePresent &&
                !value.setExecutionCacheSource(sourceRequests, sourceMaximum)) {
                AUTIL_LEGACY_THROW(autil::legacy::NotJsonizableException,
                                   "invalid execution cache source");
            }
            if(expected && !value.setExpectedKVWorkload(*expected))
                throw std::invalid_argument("expected KV workload requires capacity features");
        }
    }
};

struct ReplayLatencyPredictionJson : public autil::legacy::Jsonizable {
    NaviSchedLatencyPrediction value;
    bool present = false;

    ReplayLatencyPredictionJson() = default;
    explicit ReplayLatencyPredictionJson(const NaviSchedLatencyPrediction &input)
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
        int32_t included = value.executionLoadIncluded ? (*value.executionLoadIncluded ? 1 : 0) : -1;
        json.Jsonize("execution_load_included", included, included);
        if (json.GetMode() != JsonMode::TO_JSON) {
            if (included < -1 || included > 1) throw std::invalid_argument("invalid execution load inclusion");
            value.executionLoadIncluded.reset();
            if (included >= 0) value.executionLoadIncluded = included != 0;
        }
        json.Jsonize("schedule_loss", value.scheduleLoss);
        json.Jsonize("schedule_latency_us", value.scheduleLatencyUs,
                     value.scheduleLatencyUs);
        json.Jsonize("schedule_time_us", value.scheduleTimeUs, value.scheduleTimeUs);
        json.Jsonize("reason", value.reason);
        json.Jsonize("batch_id", value.batchId);
        json.Jsonize("expected_batch_size", value.expectedBatchSize);
        json.Jsonize("predicted_cache_hit_token_count", value.predictedCacheHitTokenCount,
                     value.predictedCacheHitTokenCount);
        json.Jsonize("predicted_queue_latency_ms", value.predictedQueueLatencyMs,
                     value.predictedQueueLatencyMs);
        json.Jsonize("predicted_execution_latency_ms", value.predictedExecutionLatencyMs,
                     value.predictedExecutionLatencyMs);
        int32_t trainingPhase = static_cast<int32_t>(value.latencyTrainingPhase);
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

struct ReplayNativePPPolicyJson : public autil::legacy::Jsonizable {
    TPUPrefillExecutionPolicyConfig value;
    void Jsonize(JsonWrapper &json) override
    {
        json.Jsonize("pcp_size", value.pcpSize, value.pcpSize);
        json.Jsonize("interleave_size", value.interleaveSize, value.interleaveSize);
        json.Jsonize("compile_sizes", value.compileSizes, value.compileSizes);
        json.Jsonize("max_scheduled_tokens", value.maxScheduledTokens,
                     value.maxScheduledTokens);
        json.Jsonize("long_prefill_token_threshold", value.longPrefillTokenThreshold,
                     value.longPrefillTokenThreshold);
        json.Jsonize("hash_block_size", value.hashBlockSize, value.hashBlockSize);
        json.Jsonize("scheduler_block_size", value.schedulerBlockSize,
                     value.schedulerBlockSize);
        json.Jsonize("use_eagle", value.useEagle, value.useEagle);
        json.Jsonize("mamba_has_prefill_checkpoint_blocks",
                     value.mambaHasPrefillCheckpointBlocks,
                     value.mambaHasPrefillCheckpointBlocks);
        json.Jsonize("mamba_partial_cache_hit", value.mambaPartialCacheHit,
                     value.mambaPartialCacheHit);
        json.Jsonize("async_kv_lookup_enabled", value.asyncKVLookupEnabled,
                     value.asyncKVLookupEnabled);
        json.Jsonize("async_kv_lookup_hash_block_size", value.asyncKVLookupHashBlockSize,
                     value.asyncKVLookupHashBlockSize);
        json.Jsonize("async_kv_lookup_uses_eagle", value.asyncKVLookupUsesEagle,
                     value.asyncKVLookupUsesEagle);
        json.Jsonize("fixed_prefill_chunk_plan", value.fixedPrefillChunkPlan,
                     value.fixedPrefillChunkPlan);
        json.Jsonize("mamba_block_aligned_split", value.mambaBlockAlignedSplit,
                     value.mambaBlockAlignedSplit);
        json.Jsonize("attention_schedule_capacity", value.attentionScheduleCapacity,
                     value.attentionScheduleCapacity);
        json.Jsonize("attention_query_tile", value.attentionQueryTile,
                     value.attentionQueryTile);
        json.Jsonize("attention_kv_tile", value.attentionKvTile, value.attentionKvTile);
        json.Jsonize("prefill_lookahead", value.prefillLookahead, value.prefillLookahead);
        json.Jsonize("max_input_tokens", value.maxInputTokens, value.maxInputTokens);
        json.Jsonize("draft_slots_per_request", value.draftSlotsPerRequest,
                     value.draftSlotsPerRequest);
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
        json.Jsonize("unified_latency_prediction", value.unifiedLatencyPrediction);
        json.Jsonize("native_prefill_pipeline_stages", value.nativePrefillPipelineStages,
                     value.nativePrefillPipelineStages);
        if (value.nativePrefillPipelineStages > 0) {
            json.Jsonize("pd_unified_capacity_split", value.pdUnifiedCapacitySplit);
            ReplayNativePPPolicyJson policy;
            policy.value = value.prefillExecutionPolicy;
            json.Jsonize("native_prefill_pipeline_policy", policy);
            if (json.GetMode() != JsonMode::TO_JSON)
                value.prefillExecutionPolicy = std::move(policy.value);
        }
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
        json.Jsonize("workload_features", workloadFeatures, workloadFeatures);
        if (json.GetMode() != JsonMode::TO_JSON) {
            value.workloadFeatures = workloadFeatures.value;
        }
        json.Jsonize("max_running_request_count", value.maxRunningRequestCount,
                     value.maxRunningRequestCount);
        json.Jsonize("workload_running_capacity_slot_count",
                     value.workloadRunningCapacitySlotCount,
                     value.workloadRunningCapacitySlotCount);
        json.Jsonize("reserved_running_capacity_slot_count",
                     value.reservedRunningCapacitySlotCount,
                     value.reservedRunningCapacitySlotCount);
        json.Jsonize("workload_report_receive_time_us",
                     value.workloadReportReceiveTimeUs);
        json.Jsonize("workload_stale_update_count", value.workloadStaleUpdateCount);
        json.Jsonize("batch_timeout_count", value.batchTimeoutCount);
        json.Jsonize("abnormal_batch_finish_count", value.abnormalBatchFinishCount);
        json.Jsonize("last_request_latency_ms", value.lastRequestLatencyMs);
        json.Jsonize("queue_total_predicted_latency_ms",
                     value.queueTotalPredictedLatencyMs);
        json.Jsonize("queue_decay_head_remaining_ms", value.queueDecayHeadRemainingMs);
        json.Jsonize("queue_base_clip_adjustment_ms",
                     value.queueModel.baseClipAdjustmentMs);
        json.Jsonize("queue_cache_member_weight_ms",
                     value.queueModel.cacheMemberWeightMs);
        json.Jsonize("queue_lookup_member_weight_ms",
                     value.queueModel.lookupMemberWeightMs);
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
        json.Jsonize("latency_residual_parameters", value.latencyResidualParameters,
                     value.latencyResidualParameters);
        json.Jsonize("latency_residual_last_update_time_us",
                     value.latencyResidualLastUpdateTimeUs,
                     value.latencyResidualLastUpdateTimeUs);
        json.Jsonize("latency_model_fingerprint", value.latencyModelFingerprint,
                     value.latencyModelFingerprint);
        json.Jsonize("latency_model_trusted", value.latencyModelTrusted,
                     value.latencyModelTrusted);
        json.Jsonize("latency_model_warm_started", value.latencyModelWarmStarted,
                     value.latencyModelWarmStarted);
        json.Jsonize("latency_model_consecutive_rejected_count",
                     value.latencyModelConsecutiveRejectedCount,
                     value.latencyModelConsecutiveRejectedCount);
        json.Jsonize("latency_model_last_reject_reason",
                     value.latencyModelLastRejectReason,
                     value.latencyModelLastRejectReason);
        json.Jsonize("scheduler_config_generation", value.schedulerConfigGeneration,
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
        json.Jsonize("explicit_cache_checkpoint_position",
                     value.explicitCacheCheckpointPosition);
        auto release = int32_t(value.nativePrefillSlotRelease);
        json.Jsonize("native_prefill_slot_release", release, release);
        if (json.GetMode() != JsonMode::TO_JSON)
            value.nativePrefillSlotRelease =
                static_cast<NativePPPrefillSlotRelease>(release);
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
            json.Jsonize("scheduler_info", value.schedulerInfo, value.schedulerInfo);
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
