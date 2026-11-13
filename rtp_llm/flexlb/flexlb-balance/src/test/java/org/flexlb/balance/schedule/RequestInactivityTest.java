package org.flexlb.balance.schedule;

import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.service.RoleType;
import org.flexlb.worker.endpoint.DecodeEndpoint;
import org.flexlb.worker.endpoint.Request;
import org.flexlb.worker.endpoint.StrategyErrorType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

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
                context.recordEvent(
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
        var oldRoute = mock(RequestResult.class);
        when(oldRoute.ctx()).thenReturn(context);
        org.junit.jupiter.api.Assertions.assertFalse(
                context.recordEvent(
                        event(
                                oldRoute,
                                RoleType.PREFILL,
                                org.flexlb.worker.endpoint.RequestEventType.ACTIVE,
                                1500L)));
        var route = before.route();
        org.junit.jupiter.api.Assertions.assertFalse(
                context.recordEvent(
                        new RequestContext.RequestEvent(
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
        context.recordEvent(
                event(
                        route,
                        RoleType.PREFILL,
                        org.flexlb.worker.endpoint.RequestEventType.ACTIVE,
                        1200L));
        assertNull(context.snapshot().deadlines().visibility());
        context.recordEvent(
                event(
                        route,
                        RoleType.PREFILL,
                        org.flexlb.worker.endpoint.RequestEventType.COMPLETED,
                        1500L));
        assertEquals(11500L, context.snapshot().deadlines().visibility().atMs());
        context.recordEvent(
                event(
                        route,
                        RoleType.DECODE,
                        org.flexlb.worker.endpoint.RequestEventType.ACTIVE,
                        1600L));
        assertNull(context.snapshot().deadlines().visibility());
        assertEquals(2600L, context.snapshot().inactivityExpiresAtMs());
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
        var route = mock(RequestResult.class);
        when(route.ctx()).thenReturn(context);
        when(route.requestId()).thenReturn("request");
        when(route.decodeEndpoint()).thenReturn(mock(DecodeEndpoint.class));
        when(route.decodeRegistration())
                .thenReturn(new DecodeEndpoint.RequestRegistration(1L, "request"));
        var selecting = context.beginRouteSelection();
        synchronized (context) {
            context.registerSelectedRoute(selecting, route);
            org.junit.jupiter.api.Assertions.assertTrue(
                    context.tryCommitRoute(
                            selecting,
                            route,
                            new RequestResult(context, RequestContext.ResponseKind.ACK, 1000L),
                            1000L));
        }
        assertNull(context.currentSelectionAttempt());
        org.junit.jupiter.api.Assertions.assertNotNull(context.snapshot().responseForFuture());
        return context;
    }

    private static RequestContext.RequestEvent event(
            RequestResult route,
            RoleType role,
            org.flexlb.worker.endpoint.RequestEventType type,
            long nowMs) {
        return new RequestContext.RequestEvent(
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
