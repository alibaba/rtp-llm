package org.flexlb.dao.pv;

import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.route.RoleType;
import org.flexlb.util.JsonUtils;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PvLogDataTest {

    @Test
    void terminalResponseUsesFinalOutcomeWithoutMutatingRoutingResponse() throws Exception {
        BalanceContext context = new BalanceContext();
        Response routed = Response.error(org.flexlb.dao.loadbalance.StrategyErrorType.REQUEST_CANCELLED);
        context.setResponse(routed);
        context.setSuccess(false);
        context.setErrorMessage("Schedule RPC deadline exceeded");
        int deadlineCode = org.flexlb.dao.loadbalance.StrategyErrorType.BATCH_SLO_EXPIRED.getErrorCode();
        PvLogData data = new PvLogData(context, deadlineCode, null, "LOCAL_MASTER", 0,
                "REQUEST_STATE_TIMED_OUT", "", System.currentTimeMillis());
        var json = new com.fasterxml.jackson.databind.ObjectMapper().readTree(JsonUtils.toStringOrEmpty(data));
        assertEquals(json.path("code"), json.path("response").path("code"));
        assertEquals(json.path("error"), json.path("response").path("error_message"));
        assertFalse(json.path("response").path("success").asBoolean());
        assertFalse(json.has("admissionRejectReason"));
        assertFalse(json.path("response").has("admission_reject_reason"));
        assertEquals(org.flexlb.dao.loadbalance.StrategyErrorType.REQUEST_CANCELLED.getErrorCode(), routed.getCode());
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
    void includesBlockHashAndKvcmTimings() {
        Request request = new Request();
        request.setRequestId("1");
        request.setSeqLen(128);
        request.setRequestTimeMs(1000);

        BalanceContext context = new BalanceContext();
        context.setStartTime(1500);
        context.setRequest(request);
        context.setResponse(new Response());
        context.recordRequestTiming(request.getRequestTimeMs(), 9L);
        context.recordBlockHashTiming(12, 34);
        context.recordCacheQuery("KVCM", 56);
        context.recordCacheSelection(RoleType.PREFILL, "10.0.0.1", 256);
        context.recordCacheQuery("KVCM", 78);
        context.recordCacheSelection(RoleType.PREFILL, "10.0.0.2", 512);
        context.recordCacheQuery("KVCM", 10);
        context.recordCacheSelection(RoleType.DECODE, "10.0.0.3", 128);
        context.recordSelectionReason(RoleType.PREFILL, "CACHE_LEADER");
        context.recordRoutingDecision(new RoutingDecision(RoleType.PREFILL, "default", "CostBasedPrefill",
                "CACHE_LEADER", 1600L, 1, "10.0.0.2:8080", 2, 1, false, java.util.Map.of(), List.of(
                        new RoutingDecision.Candidate("10.0.0.2:8080", true, 90L, 20L, 70L, 512L, 512L, 1L, null, null, null, "MODELED", 1L)), null));
        context.finishRequestTiming();

        PvLogData data = new PvLogData(context);

        assertEquals(context.getTotalTimeUs(), data.getTotalUs());
        assertEquals(500, data.getArrivalMs());
        assertEquals(9, data.getReqParseUs());
        assertEquals(12, data.getHashWaitUs());
        assertEquals(34, data.getHashUs());
        assertEquals("KVCM", data.getCacheMatchSource());
        assertEquals(144, data.getCacheMatchUs());
        assertEquals(3, data.getCacheMatchCount());
        assertEquals(2, data.getCacheMatchSelections().size());
        assertEquals("CACHE_LEADER", data.getSelectionReasons().get(RoleType.PREFILL));
        assertEquals(1, data.getRoutingDecisions().size());
        assertEquals("10.0.0.2", data.getCacheMatchSelections().getFirst().selectedIp());
        assertEquals(512, data.getCacheMatchSelections().getFirst().hitCacheTokens());

        String json = JsonUtils.toStringOrEmpty(data);
        assertTrue(json.contains("\"totalUs\":" + context.getTotalTimeUs()));
        assertTrue(json.contains("\"arrivalMs\":500"));
        assertTrue(json.contains("\"reqParseUs\":9"));
        assertTrue(json.contains("\"hashWaitUs\":12"));
        assertTrue(json.contains("\"hashUs\":34"));
        assertTrue(json.contains("\"cacheMatchSource\":\"KVCM\""));
        assertTrue(json.contains("\"cacheMatchUs\":144"));
        assertTrue(json.contains("\"cacheMatchCount\":3"));
        assertTrue(json.contains("\"cacheMatchSelections\":[{\"role\":\"PREFILL\",\"selectedIp\":\"10.0.0.2\",\"hitCacheTokens\":512}"));
        assertTrue(json.contains("\"selectionReasons\":{\"PREFILL\":\"CACHE_LEADER\"}"));
        assertTrue(json.contains("\"routingDecisions\":[{\"role\":\"PREFILL\""));
        assertTrue(json.contains("\"strategy\":\"CostBasedPrefill\""));
        assertTrue(json.contains("\"projectedTtftMs\":90"));
        assertFalse(json.contains("trackedTasks"));
        assertFalse(json.contains("waitingTasks"));
        assertFalse(json.contains("runningTasks"));
    }

    @Test
    void includesActualInputIdsCountAndRequestBodyBytes() {
        Request request = new Request();
        request.setRequestId("2");
        request.setSeqLen(999);
        request.setInputIds(new int[]{1, 2, 3});

        BalanceContext context = new BalanceContext();
        context.setRequest(request);
        context.setInputIdsCount(3L);
        context.setRequestBodyBytes(1_234L);

        PvLogData data = new PvLogData(context);

        assertEquals(Long.valueOf(3), data.getInputIdsCount());
        assertEquals(Long.valueOf(1_234), data.getRequestBodyBytes());
        String json = JsonUtils.toStringOrEmpty(data);
        assertTrue(json.contains("\"inputIdsCount\":3"));
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
        assertNull(data.getInputIdsCount());
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
                            "CostBasedPrefill", "attempt-" + attempt, attempt, attempt + 1, "worker:8080",
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
                        assertEquals(decision.selectionReason(), snapshot.getSelectionReasons().get(decision.role()));
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
                    context.recordBlockHashTiming(i, i);
                    context.recordCacheQuery("KVCM", 2);
                }
            }).get(5, java.util.concurrent.TimeUnit.SECONDS);
        }
        PvLogData result = new PvLogData(context);
        assertEquals(5_000, result.getHashWaitUs());
        assertEquals(5_000, result.getHashUs());
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
