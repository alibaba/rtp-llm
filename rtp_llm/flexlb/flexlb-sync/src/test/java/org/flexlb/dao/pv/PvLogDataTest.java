package org.flexlb.dao.pv;

import org.flexlb.balance.scheduler.RequestData;
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

/**
 * Checks request log data construction and serialized fields.
 */
class PvLogDataTest {

    private static PvLogData pv(RequestData data) {
        var response = data.getResponse();
        return new PvLogData(
                data,
                response != null && response.isSuccess(),
                response == null ? null : response.getErrorMessage(),
                response == null ? 0 : response.getCode(),
                null,
                "LOCAL_MASTER",
                0,
                null,
                null,
                System.currentTimeMillis());
    }

    @Test
    void keepsCacheSelectionWhenSelectedCandidateHasNoEndpoint() {
        RequestData context = new RequestData(new org.flexlb.config.FlexlbConfig());
        context.recordCacheSelection(RoleType.PREFILL, "10.0.0.2", 512L);
        context.recordRoutingDecision(
                new RoutingDecision(
                        RoleType.PREFILL,
                        "default",
                        "CostBasedPrefill",
                        "CACHE_LEADER",
                        1600L,
                        1,
                        "10.0.0.2:8080",
                        1,
                        1,
                        false,
                        Map.of(),
                        List.of(
                                new RoutingDecision.Candidate(
                                        null, true, 90L, 20L, 70L, 512L, 512L, 1L, null, null, null,
                                        "MODELED", 1L)),
                        null));

        PvLogData data = pv(context);

        assertEquals(1, data.getCacheMatchSelections().size());
        assertEquals("10.0.0.2", data.getCacheMatchSelections().getFirst().selectedIp());
    }

    @Test
    void includesPhaseToDistinguishTwoDecisionsWithTheSameRequestId() throws Exception {
        RequestData context = new RequestData(new org.flexlb.config.FlexlbConfig());
        context.setPhase(RequestPhase.ENCODER);
        Request request = Request.builder().requestId("shared-id").build();
        context.setRequest(request);

        var json =
                new com.fasterxml.jackson.databind.ObjectMapper()
                        .readTree(JsonUtils.toStringOrEmpty(pv(context)));

        assertEquals("shared-id", json.path("requestId").asText());
        assertEquals("ENCODER", json.path("phase").asText());
    }

    @Test
    void includesEncoderCacheHitLengthOnlyWhenClientProvidesIt() throws Exception {
        RequestData context = new RequestData(new org.flexlb.config.FlexlbConfig());
        Request request = Request.builder().requestId("encoder-stage").build();
        context.setRequest(request);
        var mapper = new com.fasterxml.jackson.databind.ObjectMapper();

        var omitted = mapper.readTree(JsonUtils.toStringOrEmpty(pv(context)));
        assertFalse(omitted.has("encoderCacheHitLen"));

        context.setRequest(request.toBuilder().encoderCacheHitLen(0L).build());
        var knownMiss = mapper.readTree(JsonUtils.toStringOrEmpty(pv(context)));
        assertTrue(knownMiss.has("encoderCacheHitLen"));
        assertEquals(0L, knownMiss.path("encoderCacheHitLen").asLong());
    }

    @Test
    void terminalResponseUsesFinalOutcomeWithoutMutatingRoutingResponse() throws Exception {
        RequestData context = new RequestData(new org.flexlb.config.FlexlbConfig());
        Response routed =
                Response.error(org.flexlb.dao.loadbalance.StrategyErrorType.REQUEST_CANCELLED);
        context.setFuture(java.util.concurrent.CompletableFuture.completedFuture(routed));
        int deadlineCode =
                org.flexlb.dao.loadbalance.StrategyErrorType.BATCH_SLO_EXPIRED.getErrorCode();
        PvLogData data =
                new PvLogData(
                        context,
                        false,
                        "Schedule RPC deadline exceeded",
                        deadlineCode,
                        null,
                        "LOCAL_MASTER",
                        0,
                        "REQUEST_STATE_TIMED_OUT",
                        null,
                        System.currentTimeMillis());
        var json =
                new com.fasterxml.jackson.databind.ObjectMapper()
                        .readTree(JsonUtils.toStringOrEmpty(data));
        assertEquals(deadlineCode, json.path("code").asInt());
        assertFalse(json.path("response").has("code"));
        assertEquals("Schedule RPC deadline exceeded", json.path("error").asText());
        assertFalse(json.path("response").has("error_message"));
        assertFalse(json.path("response").has("success"));
        assertFalse(json.has("realMasterHost"));
        assertFalse(json.path("response").has("real_master_host"));
        assertFalse(json.has("admissionRejectReason"));
        assertFalse(json.path("response").has("admission_reject_reason"));
        assertEquals(
                org.flexlb.dao.loadbalance.StrategyErrorType.REQUEST_CANCELLED.getErrorCode(),
                routed.getCode());
    }

    @Test
    void compactPvKeepsProtocolIdentityAndRuntimeGroupFields() throws Exception {
        RequestData context = new RequestData(new org.flexlb.config.FlexlbConfig());
        Request request = Request.builder().requestId("compact").build();
        context.setRequest(request);
        org.flexlb.dao.loadbalance.ServerStatus worker =
                new org.flexlb.dao.loadbalance.ServerStatus();
        worker.setRequestId("compact");
        worker.setPrefillTime(42);
        worker.setDebugInfo(new org.flexlb.dao.loadbalance.DebugInfo());
        worker.setServerIp("10.0.0.1");
        Response response = new Response();
        response.setServerStatus(List.of(worker));
        context.setFuture(java.util.concurrent.CompletableFuture.completedFuture(response));
        context.setDecisionGroup(
                new DecisionGroup(
                        "group",
                        "FIXED_WINDOW",
                        "NON_BATCH",
                        "10.0.0.1",
                        2,
                        "batch_full",
                        1234,
                        10));
        RoutingDecision.PrefillPolicy policy =
                new RoutingDecision.PrefillPolicy(
                        100, "BEST_ONLY", 10L, 0.2, 20, 1, 0L, 15.0, 0.0, 50000L, 0, 50, 3.0, 3.0);
        context.recordRoutingDecision(
                new RoutingDecision(
                        RoleType.PREFILL,
                        "default",
                        "CostBasedPrefill",
                        "BEST_ONLY",
                        1234,
                        1,
                        "10.0.0.1:8080",
                        1,
                        1,
                        false,
                        java.util.Map.of(),
                        List.of(),
                        policy));
        var mapper = new com.fasterxml.jackson.databind.ObjectMapper();
        var json = mapper.readTree(JsonUtils.toStringOrEmpty(pv(context)));
        assertEquals("compact", json.path("requestId").asText());
        assertFalse(json.path("response").path("server_status").get(0).has("request_id"));
        assertFalse(json.path("response").path("server_status").get(0).has("prefill_time"));
        assertFalse(json.path("response").path("server_status").get(0).has("debug_info"));
        assertEquals(
                "compact",
                mapper.readTree(JsonUtils.toStringOrEmpty(response))
                        .path("server_status")
                        .get(0)
                        .path("request_id")
                        .asText());
        assertFalse(json.path("routingDecisions").get(0).has("prefillPolicy"));
        var group = json.path("decisionGroup");
        assertEquals(5, group.size());
        assertEquals("group", group.path("id").asText());
        assertEquals(2, group.path("committedSize").asInt());
        assertEquals("batch_full", group.path("reason").asText());
        assertEquals(1234, group.path("committedAtMs").asLong());
        assertEquals(10, group.path("requestWaitMs").asLong());
    }

    @Test
    void pvKeepsDecodePreemptRequestIdsUnderServerStatusCompaction() throws Exception {
        RequestData context = new RequestData(new org.flexlb.config.FlexlbConfig());
        Request request = Request.builder().requestId("incoming").build();
        context.setRequest(request);
        org.flexlb.dao.loadbalance.ServerStatus decode =
                new org.flexlb.dao.loadbalance.ServerStatus();
        decode.setRole(RoleType.DECODE);
        decode.setServerIp("10.0.0.3");
        decode.setRequestId("decode-internal");
        decode.setPrefillTime(7);
        decode.setDebugInfo(new org.flexlb.dao.loadbalance.DebugInfo());
        decode.setPreemptRequestIds(List.of("victim-a", "victim-b"));
        org.flexlb.dao.loadbalance.ServerStatus prefill =
                new org.flexlb.dao.loadbalance.ServerStatus();
        prefill.setRole(RoleType.PREFILL);
        prefill.setServerIp("10.0.0.1");
        Response response = new Response();
        response.setServerStatus(List.of(decode, prefill));
        context.setFuture(java.util.concurrent.CompletableFuture.completedFuture(response));

        var json =
                new com.fasterxml.jackson.databind.ObjectMapper()
                        .readTree(JsonUtils.toStringOrEmpty(pv(context)));

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
        Request request = Request.builder().requestId("1001").build();

        RequestData context = new RequestData(new org.flexlb.config.FlexlbConfig());
        context.setRequest(request);
        context.setFuture(java.util.concurrent.CompletableFuture.completedFuture(new Response()));

        String json = JsonUtils.toStringOrEmpty(pv(context));

        assertFalse(json.contains("routingDecisions"));
    }

    @Test
    void includesSelectionReasonInDecisionSnapshot() {
        RequestData context = new RequestData(new org.flexlb.config.FlexlbConfig());
        context.recordRoutingDecision(
                new org.flexlb.dao.pv.RoutingDecision(
                        RoleType.PREFILL,
                        null,
                        null,
                        "SHORTEST_TTFT_FALLBACK",
                        0L,
                        0,
                        null,
                        0,
                        0,
                        false,
                        java.util.Map.of(),
                        java.util.List.of(),
                        null));

        String json = JsonUtils.toStringOrEmpty(pv(context));

        assertTrue(json.contains("\"selectionReason\":\"SHORTEST_TTFT_FALLBACK\""));
        assertTrue(json.contains("routingDecisions"));
        assertFalse(json.contains("\"selectionReasons\""));
    }

    @Test
    void includesKvcmTimings() {
        Request request = Request.builder().requestId("1").seqLen(128).requestTimeMs(1000).build();

        RequestData context = new RequestData(new org.flexlb.config.FlexlbConfig());
        context.setRequest(request);
        context.setFuture(java.util.concurrent.CompletableFuture.completedFuture(new Response()));
        context.recordRequestTiming(context.getStartTime() - 500, 9L);
        context.recordCacheQuery("KVCM", 56);
        context.recordCacheSelection(RoleType.PREFILL, "10.0.0.1", 256);
        context.recordCacheQuery("KVCM", 78);
        context.recordCacheSelection(RoleType.PREFILL, "10.0.0.2", 512);
        context.recordCacheQuery("KVCM", 10);
        context.recordCacheSelection(RoleType.DECODE, "10.0.0.3", 128);
        context.recordRoutingDecision(
                new RoutingDecision(
                        RoleType.PREFILL,
                        "default",
                        "CostBasedPrefill",
                        "CACHE_LEADER",
                        1600L,
                        1,
                        "10.0.0.2:8080",
                        2,
                        1,
                        false,
                        java.util.Map.of(),
                        List.of(
                                new RoutingDecision.Candidate(
                                        "10.0.0.2:8080",
                                        true,
                                        90L,
                                        20L,
                                        70L,
                                        512L,
                                        512L,
                                        1L,
                                        null,
                                        null,
                                        null,
                                        "MODELED",
                                        1L)),
                        null));
        context.finishRequestTiming();

        PvLogData data = pv(context);

        assertEquals(context.getTotalTimeUs(), data.getTotalUs());
        assertEquals(500, data.getArrivalMs());
        assertEquals(9, data.getReqParseUs());
        assertEquals("KVCM", data.getCacheMatchSource());
        assertEquals(144, data.getCacheMatchUs());
        assertEquals(3, data.getCacheMatchCount());
        assertEquals(1, data.getCacheMatchSelections().size());
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
        Request request = Request.builder().requestId("2").seqLen(999).build();

        RequestData context = new RequestData(new org.flexlb.config.FlexlbConfig());
        context.setRequest(request);
        context.setRequestMessageBytes(1234L);
        context.setRequestBodyBytes(1_234L);

        PvLogData data = pv(context);

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
        RequestData context = new RequestData(new org.flexlb.config.FlexlbConfig());
        context.setRequestBodyBytes(5_242_881L);

        PvLogData data = pv(context);

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
        RequestData context = new RequestData(new org.flexlb.config.FlexlbConfig());
        var published = new java.util.concurrent.CountDownLatch(1);
        var resume = new java.util.concurrent.CountDownLatch(1);
        try (var executor = java.util.concurrent.Executors.newSingleThreadExecutor()) {
            var writer =
                    executor.submit(
                            () -> {
                                for (int attempt = 0; attempt < 1_000; attempt++) {
                                    context.beginRoutingAttempt(RoleType.PREFILL);
                                    context.recordCacheQuery("LOCAL_SYNC", 2);
                                    context.recordRoutingDecision(
                                            new RoutingDecision(
                                                    RoleType.PREFILL,
                                                    "default",
                                                    "CostBasedPrefill",
                                                    "attempt-" + attempt,
                                                    attempt,
                                                    attempt + 1,
                                                    "worker:8080",
                                                    1,
                                                    0,
                                                    false,
                                                    java.util.Map.of(),
                                                    List.of(),
                                                    null));
                                    if (attempt == 0) {
                                        published.countDown();
                                        if (!resume.await(
                                                5, java.util.concurrent.TimeUnit.SECONDS)) {
                                            throw new AssertionError(
                                                    "PV reader did not resume the route writer");
                                        }
                                    }
                                }
                                return null;
                            });
            try {
                assertTrue(published.await(5, java.util.concurrent.TimeUnit.SECONDS));
                PvLogData terminalPv = pv(context);
                String terminalJson = JsonUtils.toStringOrEmpty(terminalPv);
                resume.countDown();
                writer.get(5, java.util.concurrent.TimeUnit.SECONDS);
                for (int i = 0; i < 1_000; i++) {
                    PvLogData snapshot = pv(context);
                    assertEquals(snapshot.getCacheMatchCount() * 2L, snapshot.getCacheMatchUs());
                    assertEquals(
                            "attempt-999",
                            snapshot.getRoutingDecisions().getFirst().selectionReason());
                    assertFalse(JsonUtils.toStringOrEmpty(snapshot).isEmpty());
                }
                writer.get(5, java.util.concurrent.TimeUnit.SECONDS);
                assertEquals(terminalJson, JsonUtils.toStringOrEmpty(terminalPv));
                assertEquals(1, terminalPv.getCacheMatchCount());
                assertEquals(1_000, pv(context).getCacheMatchCount());
            } finally {
                resume.countDown();
            }
        }
    }

    @Test
    void completedProcessingPublishesTimingFieldsToFinalReader() throws Exception {
        RequestData context = new RequestData(new org.flexlb.config.FlexlbConfig());
        try (var executor = java.util.concurrent.Executors.newSingleThreadExecutor()) {
            executor.submit(
                            () -> {
                                for (int i = 1; i <= 5_000; i++) {
                                    context.recordCacheQuery("KVCM", 2);
                                }
                            })
                    .get(5, java.util.concurrent.TimeUnit.SECONDS);
        }
        PvLogData result = pv(context);
        assertEquals(5_000, result.getCacheMatchCount());
        assertEquals(10_000, result.getCacheMatchUs());
    }

    @Test
    void rejectedSelectionReasonCannotCorruptLaterPvSnapshot() {
        RequestData context = new RequestData(new org.flexlb.config.FlexlbConfig());
        context.recordRoutingDecision(
                new org.flexlb.dao.pv.RoutingDecision(
                        RoleType.PREFILL,
                        null,
                        null,
                        "BEST_ONLY",
                        0L,
                        0,
                        null,
                        0,
                        0,
                        false,
                        java.util.Map.of(),
                        java.util.List.of(),
                        null));
        assertThrows(
                NullPointerException.class,
                () ->
                        context.recordRoutingDecision(
                                new org.flexlb.dao.pv.RoutingDecision(
                                        RoleType.PREFILL,
                                        null,
                                        null,
                                        null,
                                        0L,
                                        0,
                                        null,
                                        0,
                                        0,
                                        false,
                                        java.util.Map.of(),
                                        java.util.List.of(),
                                        null)));
        assertEquals("BEST_ONLY", pv(context).getRoutingDecisions().getFirst().selectionReason());
    }
}
