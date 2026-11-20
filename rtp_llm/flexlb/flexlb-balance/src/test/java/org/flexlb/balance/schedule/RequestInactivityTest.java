package org.flexlb.balance.schedule;

import org.flexlb.balance.request.Request;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.service.RoleType;
import org.flexlb.worker.endpoint.DecodeEndpoint;
import org.flexlb.worker.endpoint.StrategyErrorType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import static org.flexlb.worker.status.WorkerStatusTestSupport.createDiscovered;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Checks inactivity and worker handoff deadlines against exact committed request identities.
 */
class RequestInactivityTest {
    @ParameterizedTest
    @EnumSource(
            value = RoleType.class,
            names = {"PREFILL", "DECODE"})
    void matchingActivityRenewsInactivityWithoutChangingRoute(RoleType role) {
        var context = context();
        var route = context.snapshot().route();
        var previous = context.snapshot().deadlines().inactivity();
        org.junit.jupiter.api.Assertions.assertFalse(
                WorkerEventTestSupport.record(
                        context,
                        event(
                                route,
                                role,
                                org.flexlb.worker.endpoint.RequestEventType.ACTIVE,
                                1500L)));
        assertEquals(2500L, context.snapshot().inactivityExpiresAtMs());
        assertSame(route, context.snapshot().route());
        assertSame(previous, context.snapshot().deadlines().inactivity());
        assertEquals(RequestContext.Phase.OPEN, context.snapshot().phase());
    }

    @Test
    void obsoleteRouteAndDecodeReservationCannotRenewInactivity() {
        var context = context();
        var before = context.snapshot();
        var oldRoute = mock(RequestRoute.class);
        when(oldRoute.ctx()).thenReturn(context);
        org.junit.jupiter.api.Assertions.assertFalse(
                WorkerEventTestSupport.record(
                        context,
                        event(
                                oldRoute,
                                RoleType.PREFILL,
                                org.flexlb.worker.endpoint.RequestEventType.ACTIVE,
                                1500L)));
        var route = before.route();
        org.junit.jupiter.api.Assertions.assertFalse(
                WorkerEventTestSupport.record(
                        context,
                        new WorkerEventTestSupport.RequestEvent(
                                route,
                                new DecodeEndpoint.RequestRegistration(1L, "request"),
                                RoleType.DECODE,
                                null,
                                org.flexlb.worker.endpoint.RequestEventType.ACTIVE,
                                0L,
                                null,
                                1500L)));
        assertEquals(before, context.snapshot());
    }

    @Test
    void onlyCompletedPrefillStartsDecodeHandoffDeadline() {
        var context = context();
        var route = context.snapshot().route();
        assertNull(context.snapshot().deadlines().visibility());
        WorkerEventTestSupport.record(
                context,
                event(
                        route,
                        RoleType.PREFILL,
                        org.flexlb.worker.endpoint.RequestEventType.ACTIVE,
                        1200L));
        assertNull(context.snapshot().deadlines().visibility());
        WorkerEventTestSupport.record(
                context,
                event(
                        route,
                        RoleType.PREFILL,
                        org.flexlb.worker.endpoint.RequestEventType.COMPLETED,
                        1500L));
        assertEquals(11500L, context.snapshot().deadlines().visibility().atMs());
        WorkerEventTestSupport.record(
                context,
                event(
                        route,
                        RoleType.DECODE,
                        org.flexlb.worker.endpoint.RequestEventType.ACTIVE,
                        1600L));
        assertNull(context.snapshot().deadlines().visibility());
        assertEquals(2600L, context.snapshot().inactivityExpiresAtMs());
    }

    @Test
    void inactivityCancelsUnreleasedRoleAfterSuccessfulCompletion() {
        var context = context();
        var route = context.snapshot().route();
        WorkerEventTestSupport.record(
                context,
                event(
                        route,
                        RoleType.DECODE,
                        org.flexlb.worker.endpoint.RequestEventType.COMPLETED,
                        1500L));
        var completed = context.snapshot();
        context.recordCleanupProgress(route, false, true, true, 1500L);
        org.junit.jupiter.api.Assertions.assertTrue(context.snapshot().notificationsDone());
        context.recordDeadline(context.snapshot().deadlines().inactivity(), 2500L, null);
        assertSame(completed.endState(), context.snapshot().endState());
        org.junit.jupiter.api.Assertions.assertNotNull(context.snapshot().cancellationReason());
        org.junit.jupiter.api.Assertions.assertFalse(context.snapshot().notificationsDone());
        org.junit.jupiter.api.Assertions.assertFalse(
                context.snapshot().prefillWorkerReleaseConfirmed());
    }

    private static RequestContext context() {
        var context = new RequestContext(new FlexlbConfig());
        context.setRequest(Request.builder().requestId("request").seqLen(100L).build());
        RequestProtocolTestSupport.register(
                context,
                "request",
                1000L,
                60000L,
                1000L,
                StrategyErrorType.SCHEDULING_DEADLINE_EXCEEDED);
        var prefill = org.mockito.Mockito.mock(org.flexlb.worker.endpoint.PrefillEndpoint.class);
        var decode = mock(DecodeEndpoint.class);
        var prefillStatus = createDiscovered(RoleType.PREFILL, null, "127.0.0.1", 8080, 8090, null);
        var decodeStatus = createDiscovered(RoleType.DECODE, null, "127.0.0.2", 8080, 8090, null);
        when(prefill.getStatus()).thenReturn(prefillStatus);
        when(decode.getStatus()).thenReturn(decodeStatus);
        var prefillRegistration =
                mock(org.flexlb.worker.endpoint.PrefillEndpoint.RequestRegistration.class);
        when(prefillRegistration.requestId()).thenReturn("request");
        var route =
                new RequestRoute(
                        context,
                        java.util.List.of(
                                org.flexlb.balance.strategy.RoleSelection.prefill(
                                        prefill,
                                        prefillStatus.topologySnapshot(),
                                        0L,
                                        0L,
                                        prefillRegistration),
                                org.flexlb.balance.strategy.RoleSelection.decode(
                                        decode,
                                        decodeStatus.topologySnapshot(),
                                        new DecodeEndpoint.RequestRegistration(1L, "request"))));
        var selecting = context.beginRouteSelection();
        synchronized (context) {
            context.registerSelectedRoute(selecting, route);
            org.junit.jupiter.api.Assertions.assertTrue(
                    context.tryCommitRoute(
                            selecting,
                            route,
                            SchedulerTestSupport.responseCandidate(
                                    context, RequestContext.ResponseKind.ACK),
                            1000L,
                            0L));
        }
        assertNull(context.currentSelectionAttempt());
        org.junit.jupiter.api.Assertions.assertNotNull(context.snapshot().responseForFuture());
        return context;
    }

    private static WorkerEventTestSupport.RequestEvent event(
            RequestRoute route,
            RoleType role,
            org.flexlb.worker.endpoint.RequestEventType type,
            long nowMs) {
        return new WorkerEventTestSupport.RequestEvent(
                route,
                role == RoleType.DECODE ? route.decodeRegistration() : null,
                role,
                null,
                type,
                0L,
                null,
                nowMs);
    }
}
