package org.flexlb.dao.pv;

import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.SchedulerConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.route.RequestPhase;
import org.flexlb.dao.route.RoleType;
import org.flexlb.util.JsonUtils;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PvLogDataTest {

    @Test
    void keepsCacheSelectionWhenSelectedCandidateHasNoEndpoint() {
        BalanceContext context = new BalanceContext();
        context.recordCacheSelection(RoleType.PREFILL, "10.0.0.2", 512L);
        context.recordRoutingDecision(new RoutingDecision(RoleType.PREFILL, "default", "CostBasedPrefill",
                "CACHE_LEADER", null, null, null, 1600L, 1, "10.0.0.2:8080", 1, 1, false, Map.of(), List.of(
                        new RoutingDecision.Candidate(null, true, 90L, 20L, 70L, 512L, 512L,
                                1L, null, null, null, "MODELED", 1L)), null));

        PvLogData data = new PvLogData(context);

        assertEquals(1, data.getCacheMatchSelections().size());
        assertEquals("10.0.0.2", data.getCacheMatchSelections().getFirst().selectedIp());
    }

    @Test
    void includesPhaseToDistinguishTwoDecisionsWithTheSameRequestId() throws Exception {
        BalanceContext context = new BalanceContext();
        context.setRequestPhase(RequestPhase.ENCODER);
        Request request = new Request();
        request.setRequestId("shared-id");
        context.setRequest(request);

        var json = new com.fasterxml.jackson.databind.ObjectMapper()
                .readTree(JsonUtils.toStringOrEmpty(new PvLogData(context)));

        assertEquals("shared-id", json.path("requestId").asText());
        assertEquals("ENCODER", json.path("phase").asText());
    }

    @Test
    void keepsEpochMillisecondsAndFormatsAbsoluteTimes() throws Exception {
        long requestTimeMs = 1_792_339_299_097L;
        Request request = new Request();
        request.setRequestId("timed-request");
        request.setRequestTimeMs(requestTimeMs);
        BalanceContext context = new BalanceContext();
        context.setRequest(request);
        context.setStartTime(requestTimeMs + 1_000);
        context.setEnqueueTime(requestTimeMs + 2_000);
        context.setSchedulingMetadata(SchedulingMetadata.explicit(50, requestTimeMs + 60_000));
        context.setDecisionGroup(new DecisionGroup("group", "SINGLE", "NON_BATCH", "worker",
                1, "ready", requestTimeMs + 4_000, 7));
        context.recordRoutingDecision(new RoutingDecision(RoleType.PREFILL, "default", "CostBasedPrefill",
                "ready", null, null, null, requestTimeMs + 3_000, 1, "worker", 1, 1, false, Map.of(), List.of(), null));

        var json = new com.fasterxml.jackson.databind.ObjectMapper().readTree(
                JsonUtils.toStringOrEmpty(new PvLogData(context, 0, null, "LOCAL_MASTER", 0,
                        "COMPLETED", null, requestTimeMs + 5_000)));

        assertEquals(requestTimeMs, json.path("requestTimeMs").asLong());
        assertEquals("2026-10-19 00:01:39.097", json.path("requestTime").asText());
        assertEquals(requestTimeMs + 1_000, json.path("startTime").asLong());
        assertEquals("2026-10-19 00:01:40.097", json.path("startedAt").asText());
        assertEquals(requestTimeMs + 2_000, json.path("enqueueTime").asLong());
        assertEquals("2026-10-19 00:01:41.097", json.path("enqueuedAt").asText());
        assertEquals(requestTimeMs + 60_000, json.path("requestExpiresAtMs").asLong());
        assertEquals("2026-10-19 00:02:39.097", json.path("requestExpiresAt").asText());
        var group = json.path("decisionGroup");
        assertEquals(requestTimeMs + 4_000, group.path("committedAtMs").asLong());
        assertEquals("2026-10-19 00:01:43.097", group.path("committedAt").asText());
        assertEquals(7, group.path("requestWaitMs").asLong());
        var decision = json.path("routingDecisions").get(0);
        assertEquals(requestTimeMs + 3_000, decision.path("decisionTimeMs").asLong());
        assertEquals("2026-10-19 00:01:42.097", decision.path("decisionTime").asText());
        assertEquals(4_000, json.path("latencyMs").asLong());
        assertFalse(json.has("latency"));
        assertFalse(group.has("requestWait"));
    }

    @Test
    void omitsFormattedTimesWhenTimestampsAreAbsentOrUnlimited() throws Exception {
        FlexlbConfig config = new FlexlbConfig();
        config.setScheduler(SchedulerConfig.direct());
        BalanceContext context = new BalanceContext(config);
        context.setStartTime(0);
        context.setRequest(new Request());
        context.setDecisionGroup(new DecisionGroup("group", "SINGLE", "NON_BATCH", "worker",
                1, "ready", 0, 0));
        context.recordRoutingDecision(new RoutingDecision(RoleType.PREFILL, "default", "CostBasedPrefill",
                "ready", null, null, null, 0, 1, "worker", 1, 1, false, Map.of(), List.of(), null));

        var json = new com.fasterxml.jackson.databind.ObjectMapper().readTree(
                JsonUtils.toStringOrEmpty(new PvLogData(context, 0, null, "LOCAL_MASTER", 0,
                        "COMPLETED", null, 0)));

        assertEquals(0, json.path("requestTimeMs").asLong());
        assertEquals(0, json.path("startTime").asLong());
        assertEquals(0, json.path("enqueueTime").asLong());
        assertEquals(Long.MAX_VALUE, json.path("requestExpiresAtMs").asLong());
        assertFalse(json.has("requestTime"));
        assertFalse(json.has("startedAt"));
        assertFalse(json.has("enqueuedAt"));
        assertFalse(json.has("requestExpiresAt"));
        assertEquals(0, json.path("decisionGroup").path("committedAtMs").asLong());
        assertFalse(json.path("decisionGroup").has("committedAt"));
        assertEquals(0, json.path("routingDecisions").get(0).path("decisionTimeMs").asLong());
        assertFalse(json.path("routingDecisions").get(0).has("decisionTime"));
    }

    @Test
    void includesEncoderCacheHitLengthOnlyWhenClientProvidesIt() throws Exception {
        BalanceContext context = new BalanceContext();
        Request request = new Request();
        request.setRequestId("encoder-stage");
        context.setRequest(request);
        var mapper = new com.fasterxml.jackson.databind.ObjectMapper();

        var omitted = mapper.readTree(JsonUtils.toStringOrEmpty(new PvLogData(context)));
        assertFalse(omitted.has("encoderCacheHitLen"));

        request.setEncoderCacheHitLen(0L);
        var knownMiss = mapper.readTree(JsonUtils.toStringOrEmpty(new PvLogData(context)));
        assertTrue(knownMiss.has("encoderCacheHitLen"));
        assertEquals(0L, knownMiss.path("encoderCacheHitLen").asLong());
    }

    @Test
    void terminalResponseUsesFinalOutcomeWithoutMutatingRoutingResponse() throws Exception {
        BalanceContext context = new BalanceContext();
        Response routed = Response.error(org.flexlb.dao.loadbalance.StrategyErrorType.REQUEST_CANCELLED);
        context.setResponse(routed);
        context.setSuccess(false);
        context.setErrorMessage("Schedule RPC deadline exceeded");
        int deadlineCode = org.flexlb.dao.loadbalance.StrategyErrorType.BATCH_SLO_EXPIRED.getErrorCode();
        PvLogData data = new PvLogData(context, deadlineCode, null, "LOCAL_MASTER", 0,
                "REQUEST_STATE_TIMED_OUT", null, System.currentTimeMillis());
        var json = new com.fasterxml.jackson.databind.ObjectMapper().readTree(JsonUtils.toStringOrEmpty(data));
        assertEquals(deadlineCode, json.path("code").asInt());
        assertFalse(json.path("response").has("code"));
        assertEquals("Schedule RPC deadline exceeded", json.path("error").asText());
        assertFalse(json.path("response").has("error_message"));
        assertFalse(json.path("response").has("success"));
        assertFalse(json.has("realMasterHost"));
        assertFalse(json.path("response").has("real_master_host"));
        assertFalse(json.has("admissionRejectReason"));
        assertFalse(json.path("response").has("admission_reject_reason"));
        assertEquals(org.flexlb.dao.loadbalance.StrategyErrorType.REQUEST_CANCELLED.getErrorCode(), routed.getCode());
    }

    @Test
    void compactPvKeepsProtocolIdentityAndRuntimeGroupFields() throws Exception {
        BalanceContext context = new BalanceContext();
        Request request = new Request();
        request.setRequestId("compact");
        context.setRequest(request);
        org.flexlb.dao.loadbalance.ServerStatus worker = new org.flexlb.dao.loadbalance.ServerStatus();
        worker.setRequestId("compact");
        worker.setPrefillTime(42);
        worker.setDebugInfo(new org.flexlb.dao.loadbalance.DebugInfo());
        worker.setServerIp("10.0.0.1");
        Response response = new Response();
        response.setServerStatus(List.of(worker));
        context.setResponse(response);
        context.setDecisionGroup(new DecisionGroup("group", "FIXED_WINDOW", "NON_BATCH", "10.0.0.1",
                2, "batch_full", 1234, 10));
        RoutingDecision.PrefillPolicy policy = new RoutingDecision.PrefillPolicy(100, "BEST_ONLY", 10L,
                0.2, 20, 1, 0L, 15.0, 0.0, 0, 50, 3.0, 3.0);
        context.recordRoutingDecision(new RoutingDecision(RoleType.PREFILL, "default", "CostBasedPrefill",
                "BEST_ONLY", null, null, null, 1234, 1, "10.0.0.1:8080", 1, 1, false, java.util.Map.of(), List.of(), policy));
        var mapper = new com.fasterxml.jackson.databind.ObjectMapper();
        var json = mapper.readTree(JsonUtils.toStringOrEmpty(new PvLogData(context)));
        assertEquals("compact", json.path("requestId").asText());
        assertFalse(json.path("response").path("server_status").get(0).has("request_id"));
        assertFalse(json.path("response").path("server_status").get(0).has("prefill_time"));
        assertFalse(json.path("response").path("server_status").get(0).has("debug_info"));
        assertEquals("compact", mapper.readTree(JsonUtils.toStringOrEmpty(response))
                .path("server_status").get(0).path("request_id").asText());
        assertFalse(json.path("routingDecisions").get(0).has("prefillPolicy"));
        var group = json.path("decisionGroup");
        assertEquals(6, group.size());
        assertEquals("group", group.path("id").asText());
        assertEquals(2, group.path("committedSize").asInt());
        assertEquals("batch_full", group.path("reason").asText());
        assertEquals(1234, group.path("committedAtMs").asLong());
        assertEquals("1970-01-01 08:00:01.234", group.path("committedAt").asText());
        assertEquals(10, group.path("requestWaitMs").asLong());
    }

    @Test
    void pvKeepsDecodePreemptRequestIdsUnderServerStatusCompaction() throws Exception {
        BalanceContext context = new BalanceContext();
        Request request = new Request();
        request.setRequestId("incoming");
        context.setRequest(request);
        org.flexlb.dao.loadbalance.ServerStatus decode = new org.flexlb.dao.loadbalance.ServerStatus();
        decode.setRole(RoleType.DECODE);
        decode.setServerIp("10.0.0.3");
        decode.setRequestId("decode-internal");
        decode.setPrefillTime(7);
        decode.setDebugInfo(new org.flexlb.dao.loadbalance.DebugInfo());
        decode.setPreemptRequestIds(List.of("victim-a", "victim-b"));
        org.flexlb.dao.loadbalance.ServerStatus prefill = new org.flexlb.dao.loadbalance.ServerStatus();
        prefill.setRole(RoleType.PREFILL);
        prefill.setServerIp("10.0.0.1");
        Response response = new Response();
        response.setServerStatus(List.of(decode, prefill));
        context.setResponse(response);

        var json = new com.fasterxml.jackson.databind.ObjectMapper()
                .readTree(JsonUtils.toStringOrEmpty(new PvLogData(context)));

        var statuses = json.path("response").path("server_status");
        var decodeIds = statuses.get(0).path("preempt_request_ids");
        assertEquals(2, decodeIds.size());
        assertEquals("victim-a", decodeIds.get(0).asText());
        assertEquals("victim-b", decodeIds.get(1).asText());
        assertEquals("DECODE", statuses.get(0).path("role").asText());
        assertEquals(0, statuses.get(1).path("preempt_request_ids").size());
        assertFalse(statuses.get(0).has("request_id"));
        assertFalse(statuses.get(0).has("prefill_time"));
        assertFalse(statuses.get(0).has("debug_info"));
        assertEquals(List.of("victim-a", "victim-b"), decode.getPreemptRequestIds());
    }

    @Test
    void omitsRoutingDecisionWhenSnapshotIsAbsent() {
        Request request = new Request();
        request.setRequestId("1001");

        BalanceContext context = new BalanceContext();
        context.setRequest(request);
        context.setResponse(new Response());

        String json = JsonUtils.toStringOrEmpty(new PvLogData(context));

        assertFalse(json.contains("routingDecisions"));
    }

    @Test
    void includesSelectionReasonWithoutDecisionSnapshot() {
        BalanceContext context = new BalanceContext();
        context.recordSelectionReason(RoleType.PREFILL, "SHORTEST_TTFT_FALLBACK");

        String json = JsonUtils.toStringOrEmpty(new PvLogData(context));

        assertTrue(json.contains("\"selectionReasons\":{\"PREFILL\":\"SHORTEST_TTFT_FALLBACK\"}"));
        assertFalse(json.contains("routingDecisions"));
    }

    @Test
    void includesKvcmTimings() {
        Request request = new Request();
        request.setRequestId("1");
        request.setSeqLen(128);
        request.setRequestTimeMs(1000);

        BalanceContext context = new BalanceContext();
        context.setStartTime(1500);
        context.setRequest(request);
        context.setResponse(new Response());
        context.recordRequestTiming(request.getRequestTimeMs(), 9L);
        context.recordCacheQuery("KVCM", 56);
        context.recordCacheSelection(RoleType.PREFILL, "10.0.0.1", 256);
        context.recordCacheQuery("KVCM", 78);
        context.recordCacheSelection(RoleType.PREFILL, "10.0.0.2", 512);
        context.recordCacheQuery("KVCM", 10);
        context.recordCacheSelection(RoleType.DECODE, "10.0.0.3", 128);
        context.recordSelectionReason(RoleType.PREFILL, "CACHE_LEADER");
        context.recordRoutingDecision(new RoutingDecision(RoleType.PREFILL, "default", "CostBasedPrefill",
                "CACHE_LEADER", null, null, null, 1600L, 1, "10.0.0.2:8080", 2, 1, false, java.util.Map.of(), List.of(
                        new RoutingDecision.Candidate("10.0.0.2:8080", true, 90L, 20L, 70L, 512L, 512L, 1L, null, null, null, "MODELED", 1L)), null));
        context.finishRequestTiming();

        PvLogData data = new PvLogData(context);

        assertEquals(context.getTotalTimeUs(), data.getTotalUs());
        assertEquals(500, data.getArrivalMs());
        assertEquals(9, data.getReqParseUs());
        assertEquals("KVCM", data.getCacheMatchSource());
        assertEquals(144, data.getCacheMatchUs());
        assertEquals(3, data.getCacheMatchCount());
        assertEquals(1, data.getCacheMatchSelections().size());
        assertTrue(data.getSelectionReasons().isEmpty());
        assertEquals(1, data.getRoutingDecisions().size());
        assertEquals("10.0.0.3", data.getCacheMatchSelections().getFirst().selectedIp());
        assertEquals(128, data.getCacheMatchSelections().getFirst().hitCacheTokens());

        String json = JsonUtils.toStringOrEmpty(data);
        assertTrue(json.contains("\"totalUs\":" + context.getTotalTimeUs()));
        assertTrue(json.contains("\"arrivalMs\":500"));
        assertTrue(json.contains("\"reqParseUs\":9"));
        assertFalse(json.contains("\"hashWaitUs\""));
        assertFalse(json.contains("\"hashUs\""));
        assertTrue(json.contains("\"cacheMatchSource\":\"KVCM\""));
        assertTrue(json.contains("\"cacheMatchUs\":144"));
        assertTrue(json.contains("\"cacheMatchCount\":3"));
        assertTrue(json.contains("\"cacheMatchSelections\":[{\"role\":\"DECODE\""));
        assertFalse(json.contains("\"selectionReasons\""));
        assertTrue(json.contains("\"routingDecisions\":[{\"role\":\"PREFILL\""));
        assertTrue(json.contains("\"strategy\":\"CostBasedPrefill\""));
        assertTrue(json.contains("\"projectedTtftMs\":90"));
        assertFalse(json.contains("trackedTasks"));
        assertFalse(json.contains("waitingTasks"));
        assertFalse(json.contains("runningTasks"));
    }

    @Test
    void includesSequenceLengthAndOmitsPayloadDiagnostics() {
        Request request = new Request();
        request.setRequestId("2");
        request.setSeqLen(999);

        BalanceContext context = new BalanceContext();
        context.setRequest(request);
        context.setRequestMessageBytes(1234L);
        context.setRequestBodyBytes(1_234L);

        PvLogData data = new PvLogData(context);

        assertEquals(Long.valueOf(999), data.getSeqLen());
        assertEquals(Long.valueOf(1_234), data.getRequestBodyBytes());
        String json = JsonUtils.toStringOrEmpty(data);
        assertTrue(json.contains("\"seqLen\":999"));
        assertFalse(json.contains("\"inputIdsCount\""));
        assertFalse(json.contains("\"requestMessageBytes\""));
        assertTrue(json.contains("\"requestBodyBytes\":1234"));
    }

    @Test
    void keepsFailurePvWhenRequestBodyCannotBeDeserialized() {
        BalanceContext context = new BalanceContext();
        context.setSuccess(false);
        context.setErrorMessage("Exceeded limit on max bytes to buffer");
        context.setRequestBodyBytes(5_242_881L);

        PvLogData data = new PvLogData(context);

        assertNull(data.getRequestId());
        assertNull(data.getSeqLen());
        assertEquals(Long.valueOf(5_242_881), data.getRequestBodyBytes());
        String json = JsonUtils.toStringOrEmpty(data);
        assertFalse(json.contains("\"requestId\""));
        assertFalse(json.contains("\"seqLen\""));
        assertFalse(json.contains("\"inputIdsCount\""));
        assertTrue(json.contains("\"requestBodyBytes\":5242881"));
        assertTrue(json.contains("\"success\":false"));
    }
    @Test
    void retainedPvSnapshotDoesNotChangeAfterSequentialRoutingUpdates() throws Exception {
        BalanceContext context = new BalanceContext();
        var published = new java.util.concurrent.CountDownLatch(1);
        var resume = new java.util.concurrent.CountDownLatch(1);
        try (var executor = java.util.concurrent.Executors.newSingleThreadExecutor()) {
            var writer = executor.submit(() -> {
                for (int attempt = 0; attempt < 1_000; attempt++) {
                    context.beginRoutingAttempt(RoleType.PREFILL);
                    context.recordCacheQuery("LOCAL_SYNC", 2);
                    context.recordRoutingDecision(new RoutingDecision(RoleType.PREFILL, "default",
                            "CostBasedPrefill", "attempt-" + attempt, null, null, null, attempt, attempt + 1, "worker:8080",
                            1, 0, false, java.util.Map.of(), List.of(), null));
                    if (attempt == 0) {
                        published.countDown();
                        if (!resume.await(5, java.util.concurrent.TimeUnit.SECONDS)) {
                            throw new AssertionError("PV reader did not resume the route writer");
                        }
                    }
                }
                return null;
            });
            try {
                assertTrue(published.await(5, java.util.concurrent.TimeUnit.SECONDS));
                PvLogData terminalPv = new PvLogData(context);
                String terminalJson = JsonUtils.toStringOrEmpty(terminalPv);
                resume.countDown();
                writer.get(5, java.util.concurrent.TimeUnit.SECONDS);
                for (int i = 0; i < 1_000; i++) {
                    PvLogData snapshot = new PvLogData(context);
                    assertEquals(snapshot.getCacheMatchCount() * 2L, snapshot.getCacheMatchUs());
                    for (RoutingDecision decision : snapshot.getRoutingDecisions()) {
                        assertFalse(snapshot.getSelectionReasons().containsKey(decision.role()));
                    }
                    assertFalse(JsonUtils.toStringOrEmpty(snapshot).isEmpty());
                }
                writer.get(5, java.util.concurrent.TimeUnit.SECONDS);
                assertEquals(terminalJson, JsonUtils.toStringOrEmpty(terminalPv));
                assertEquals(1, terminalPv.getCacheMatchCount());
                assertEquals(1_000, new PvLogData(context).getCacheMatchCount());
            } finally {
                resume.countDown();
            }
        }
    }

    @Test
    void completedProcessingPublishesTimingFieldsToFinalReader() throws Exception {
        BalanceContext context = new BalanceContext();
        try (var executor = java.util.concurrent.Executors.newSingleThreadExecutor()) {
            executor.submit(() -> {
                for (int i = 1; i <= 5_000; i++) {
                    context.recordCacheQuery("KVCM", 2);
                }
            }).get(5, java.util.concurrent.TimeUnit.SECONDS);
        }
        PvLogData result = new PvLogData(context);
        assertEquals(5_000, result.getCacheMatchCount());
        assertEquals(10_000, result.getCacheMatchUs());
    }

    @Test
    void rejectedSelectionReasonCannotCorruptLaterPvSnapshot() {
        BalanceContext context = new BalanceContext();
        context.recordSelectionReason(RoleType.PREFILL, "BEST_ONLY");
        assertThrows(NullPointerException.class, () -> context.recordSelectionReason(RoleType.PREFILL, null));
        assertEquals("BEST_ONLY", new PvLogData(context).getSelectionReasons().get(RoleType.PREFILL));
    }

}
