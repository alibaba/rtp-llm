package org.flexlb.balance.scheduler;

import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import static org.flexlb.balance.scheduler.ContextResult.Granted;
import static org.flexlb.balance.scheduler.ContextResult.Rejected;
import static org.flexlb.balance.scheduler.ContextResult.ResponseCandidate;
import static org.flexlb.balance.scheduler.ContextResult.ResponseKind;
import static org.flexlb.balance.scheduler.ContextResult.RouteDelivered;
import static org.flexlb.balance.scheduler.ContextResult.SendState;
import static org.flexlb.balance.scheduler.ContextResult.UpdateKind;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DispatchDeadlineTest {
    @ParameterizedTest
    @EnumSource(value = DeliveryClaimKind.class, names = {"BATCH_ENQUEUE", "ROUTE_DECISION"})
    void preparedDeliveryKeepsTheOriginalDeadlineWithoutAnInactivityTimer(DeliveryClaimKind kind) {
        var fixture = RequestDeliveryClaimStateTest.fixture(kind, false);
        var context = fixture.context();
        assertTrue(context.isAwaitingDispatch());
        assertEquals(100_000L, context.snapshot().deadlines().request().atMs());
        assertNull(context.snapshot().deadlines().inactivity());
        assertEquals(Long.MAX_VALUE, context.snapshot().inactivityExpiresAtMs());
        var deadline = context.snapshot().deadlines().request();
        context.recordDeadline(deadline, deadline.atMs(), null);
        assertEquals(CancelReason.DEADLINE_EXCEEDED, context.snapshot().cancellation().reason());
        assertInstanceOf(Rejected.class, context.beginSend(fixture.delivery(), deadline.atMs()));
    }

    @Test
    void engineSendAtDeadlineIsRejectedEvenBeforeTimerCallback() {
        var fixture = RequestDeliveryClaimStateTest.fixture(DeliveryClaimKind.BATCH_ENQUEUE, false);
        var context = fixture.context();
        assertInstanceOf(Rejected.class, context.beginSend(fixture.delivery(), 100_000L));
        assertEquals(SendState.NOT_STARTED, context.snapshot().delivery().send());
        assertEquals(CancelReason.DEADLINE_EXCEEDED, context.snapshot().cancellation().reason());
        assertEquals(StrategyErrorType.BATCH_SLO_EXPIRED, context.snapshot().pendingEnd().errorType());
    }

    @Test
    void timelyEngineSendDisarmsDispatchDeadlineAndStartsWatchdogAtSendTime() {
        var fixture = RequestDeliveryClaimStateTest.fixture(DeliveryClaimKind.BATCH_ENQUEUE, false);
        var context = fixture.context();
        var original = context.snapshot().deadlines().request();
        assertInstanceOf(Granted.class, context.beginSend(fixture.delivery(), 99_999L));
        assertFalse(context.isAwaitingDispatch());
        assertNull(context.snapshot().deadlines().request());
        assertEquals(109_999L, context.snapshot().deadlines().inactivity().atMs());
        assertEquals(UpdateKind.STALE, context.recordDeadline(original, 100_000L, null).kind());
        assertNull(context.snapshot().cancellation());
    }

    @Test
    void routeResultMustBePublishedBeforeDeadline() {
        var fixture = RequestDeliveryClaimStateTest.fixture(DeliveryClaimKind.ROUTE_DECISION, false);
        var context = fixture.context();
        context.recordDeliveryResult(new RouteDelivered(fixture.delivery(), 200_000L, 99_999L));
        assertInstanceOf(Granted.class, context.selectResponse(
                new ResponseCandidate(context, ResponseKind.CONFIRMATION, null, 99_999L)));
        assertTrue(context.isAwaitingDispatch());
        assertInstanceOf(Rejected.class,
                context.selectResponse(new ResponseCandidate(context, ResponseKind.ACK, null, 100_000L)));
        assertNull(context.snapshot().selectedResponse());
        assertEquals(CancelReason.DEADLINE_EXCEEDED, context.snapshot().cancellation().reason());
    }

    @Test
    void longRouteQueueWaitDoesNotConsumePostDispatchWatchdog() {
        var fixture = RequestDeliveryClaimStateTest.fixture(DeliveryClaimKind.ROUTE_DECISION, false);
        var context = fixture.context();
        var original = context.snapshot().deadlines().request();
        context.recordDeliveryResult(new RouteDelivered(fixture.delivery(), 200_000L, 50_000L));
        assertInstanceOf(Granted.class,
                context.selectResponse(new ResponseCandidate(context, ResponseKind.ACK, null, 99_999L)));
        assertFalse(context.isAwaitingDispatch());
        assertNull(context.snapshot().deadlines().request());
        assertEquals(109_999L, context.snapshot().deadlines().inactivity().atMs());
        assertEquals(UpdateKind.STALE, context.recordDeadline(original, 100_000L, null).kind());
    }
}
