package org.flexlb.balance.schedule;

import org.flexlb.balance.strategy.RoleSelection;
import org.flexlb.config.service.RoleType;
import org.flexlb.worker.endpoint.CapacityKey;
import org.flexlb.worker.endpoint.EndpointTestSupport;
import org.flexlb.worker.status.DebugInfo;
import org.flexlb.worker.status.ServerStatus;
import org.flexlb.worker.status.WorkerStatus;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Checks frozen role selections, lightweight request state and preemption fields on the unified result.
 */
class RequestResultTest {
    @Test
    void frozenSelectionSurvivesInputResponseAndEndpointChanges() {
        var context = RequestProtocolTestSupport.context(SchedulingTestConfig.newConfig(), 42L);
        var status =
                WorkerStatus.createDiscovered(
                        RoleType.PREFILL, "original", "chosen", 8080, 8090, null);
        var endpoint =
                EndpointTestSupport.realPrefill(
                        status,
                        SchedulingTestConfig.newConfig(),
                        org.mockito.Mockito.mock(EngineCancellation.class),
                        new RequestRepository(),
                        org.mockito.Mockito.mock(
                                org.flexlb.balance.metrics.DeliveryMetricsReporter.class));
        var roles = new java.util.ArrayList<RoleSelection>();
        try {
            var metadata =
                    ServerStatus.forWorker(
                            RoleType.PREFILL,
                            "42",
                            status.topologySnapshot(),
                            status.committedEngineObservation());
            var debug = new DebugInfo();
            debug.setHitCacheLen(7);
            metadata.setDebugInfo(debug);
            roles.add(RoleSelection.prefill(endpoint, metadata, 13, null));
            var result = new RequestResult(context, roles);
            assertThrows(
                    IllegalStateException.class,
                    () ->
                            new RequestResult(
                                    context,
                                    java.util.List.of(roles.getFirst(), roles.getFirst())));
            metadata.setGroup("changed");
            debug.setHitCacheLen(99);
            result.successResponse()
                    .getServerStatus()
                    .getFirst()
                    .getDebugInfo()
                    .setHitCacheLen(100);
            org.flexlb.worker.endpoint.EndpointTestSupport.closeAndWait(endpoint);

            var prefill = result.role(RoleType.PREFILL);
            assertSame(endpoint, prefill.endpoint());
            assertEquals(13, prefill.computeTokens());
            assertEquals(7, prefill.hitCacheTokens());
            assertEquals(
                    "original", result.successResponse().getServerStatus().getFirst().getGroup());
            assertThrows(UnsupportedOperationException.class, () -> result.roles().clear());

            var another = new RequestResult(context, result.roles());
            assertNotSame(result, another);
            assertSame(context, result.ctx());
            assertSame(context, another.ctx());
            assertSame(result.role(RoleType.PREFILL), another.role(RoleType.PREFILL));
            assertNull(result.decodeRegistration());
            assertNull(another.prefillRegistration());
        } finally {
            org.flexlb.worker.endpoint.EndpointTestSupport.closeAndWait(endpoint);
            EndpointTestSupport.awaitClosed(endpoint);
        }
    }

    @Test
    void roleFailureBecomesOneRequestDecisionWithItsOriginalCause() {
        var blocker = new CapacityKey(RoleType.PREFILL, "group", null);
        var failure =
                org.flexlb.worker.endpoint.Response.error(
                        org.flexlb.worker.endpoint.StrategyErrorType.NO_PREFILL_WORKER);
        var diagnostics = new java.util.HashMap<String, Object>();
        diagnostics.put("cause", "capacity");
        var decision =
                org.flexlb.balance.strategy.RoutingResult.blocked(blocker, failure, diagnostics);
        var result = RequestResult.fromDecision(decision);
        diagnostics.clear();
        assertEquals(org.flexlb.balance.strategy.RoutingResult.Status.BLOCKED, result.status());
        assertSame(blocker, result.blocker());
        assertSame(failure, result.failure());
        assertThrows(IllegalStateException.class, result::successResponse);
        assertEquals("capacity", result.diagnostics().get("cause"));
        assertThrows(UnsupportedOperationException.class, () -> result.diagnostics().clear());
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        RequestResult.fromDecision(
                                org.flexlb.balance.strategy.RoutingResult.success("selected")));
    }

    @Test
    void endedRequestStateRetainsNoRequestOrRoleReferences() {
        var context = new RequestContext(SchedulingTestConfig.newConfig());
        RequestProtocolTestSupport.register(
                context,
                "ended",
                1000L,
                5000L,
                1000L,
                org.flexlb.worker.endpoint.StrategyErrorType.SCHEDULING_DEADLINE_EXCEEDED);
        context.endRequest(
                RequestContext.RequestEndEvent.failure(
                        null,
                        RequestContext.EndKind.FAILURE,
                        org.flexlb.worker.endpoint.StrategyErrorType.DISPATCH_FAILED,
                        "failed"),
                1001L,
                new RequestResult(context, RequestContext.ResponseKind.TERMINAL, 1001L));
        context.recordCleanupProgress(context.snapshot().route(), false, false, true, 1002L);
        var state = RequestResult.from(context.snapshot());
        assertSame(context.snapshot().requestRecordForArchive(), state);
        assertEquals(RequestResult.State.FAILED, state.state());
        assertEquals("ended", state.requestId());
        assertNull(state.ctx());
        assertEquals(java.util.List.of(), state.roles());
        assertEquals(
                state,
                new RequestResult(
                        "ended",
                        0L,
                        org.flexlb.balance.schedule.RequestPhase.GENERATION,
                        RequestResult.State.FAILED,
                        1000L,
                        1002L,
                        "failed"));
    }

    @Test
    void requestStateJsonKeepsItsPublicFieldsAndOmitsRuntimeReferences() throws Exception {
        var owner = org.mockito.Mockito.mock(AbstractRequestScheduler.class);
        var state =
                new RequestResult(
                        "request",
                        42L,
                        org.flexlb.balance.schedule.RequestPhase.GENERATION,
                        RequestResult.State.FAILED,
                        1000L,
                        1002L,
                        "failed",
                        owner);
        var json =
                new com.fasterxml.jackson.databind.ObjectMapper()
                        .readTree(
                                org.flexlb.util.JsonUtils.toFormattedString(
                                        java.util.List.of(state)))
                        .get(0);
        assertEquals(7, json.size());
        assertEquals("request", json.get("requestId").asText());
        assertEquals(42L, json.get("originalNumericRequestId").asLong());
        assertEquals("GENERATION", json.get("requestPhase").asText());
        assertEquals("FAILED", json.get("state").asText());
        assertEquals(1000L, json.get("createdAtMs").asLong());
        assertEquals(1002L, json.get("updatedAtMs").asLong());
        assertEquals("failed", json.get("detail").asText());
    }
}
