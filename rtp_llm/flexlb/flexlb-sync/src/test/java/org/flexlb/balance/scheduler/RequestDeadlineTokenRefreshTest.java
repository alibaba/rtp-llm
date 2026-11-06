package org.flexlb.balance.scheduler;

import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import static org.flexlb.balance.scheduler.ContextResult.CancellationFacts;
import static org.flexlb.balance.scheduler.ContextResult.Phase;
import static org.flexlb.balance.scheduler.ContextResult.Prediction;
import static org.flexlb.balance.scheduler.ContextResult.WorkerRequestEvent;
import static org.flexlb.balance.scheduler.ContextResult.WorkerRequestEventType;
import static org.flexlb.balance.scheduler.RequestChangeAssertions.UpdateKind;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** An exact early wake refreshes scheduling identity without acquiring any termination authority. */
class RequestDeadlineTokenRefreshTest {
    @ParameterizedTest
    @EnumSource(ExpirationTimer.Kind.class)
    void earlyWakeRefreshesOnlyItsTokenWithoutConsumingDeadlineOrFirstCause(
            ExpirationTimer.Kind kind) {
        var fixture = RequestDeliveryClaimStateTest.fixture(Delivery.SendMode.BATCH_ENQUEUE, false);
        var context = fixture.context();
        org.flexlb.balance.scheduler.RequestChangeAssertions.record(
                context,
                observedRequest ->
                        observedRequest.recordDeliveryResult(
                                new Prediction(fixture.delivery(), 3000L, 1002L)));
        var before = context.snapshot();
        var previous = before.deadlines().token(kind);
        assertNotNull(previous);
        var diagnosis =
                new CancellationFacts(
                        CancelReason.DEADLINE_EXCEEDED,
                        StrategyErrorType.RESOURCE_EXHAUSTED,
                        "diagnosis must not be used early");
        var update =
                org.flexlb.balance.scheduler.RequestChangeAssertions.record(
                        context,
                        observedRequest ->
                                observedRequest.recordDeadline(
                                        previous, previous.atMs() - 1L, diagnosis));
        assertEquals(UpdateKind.APPLIED, update.kind());
        assertSame(before, update.before());
        assertNotSame(before, update.after());
        var current = update.after().deadlines().token(kind);
        assertNotNull(current);
        assertNotSame(previous, current);
        assertEquals(
                kind == ExpirationTimer.Kind.INACTIVITY
                        ? before.inactivityExpiresAtMs()
                        : previous.atMs(),
                current.atMs());
        assertTrue(current.version() > previous.version());
        for (var other : ExpirationTimer.Kind.values()) {
            if (other != kind) {
                assertSame(
                        before.deadlines().token(other), update.after().deadlines().token(other));
            }
        }
        assertNull(update.after().cancellation());
        assertNull(update.after().outcome());
        assertNull(update.after().pendingEnd());
        assertFalse(update.after().close().expired());
        assertFalse(update.after().engine().visibilityOverdue());
        assertSame(fixture.delivery(), update.after().delivery().identity());
        assertEquals(Phase.OPEN, update.after().phase());
        assertSame(
                previous,
                before.deadlines().token(kind),
                "refresh cannot mutate a previously returned snapshot");
        var old =
                org.flexlb.balance.scheduler.RequestChangeAssertions.record(
                        context,
                        observedRequest ->
                                observedRequest.recordDeadline(
                                        previous, previous.atMs(), diagnosis));
        assertEquals(UpdateKind.STALE, old.kind());
        assertSame(update.after(), old.before());
        assertSame(old.before(), old.after());
        var forged = new ExpirationTimer.Token(context, kind, current.version(), current.atMs());
        assertEquals(
                UpdateKind.STALE,
                org.flexlb.balance.scheduler.RequestChangeAssertions.record(
                                context,
                                observedRequest ->
                                        observedRequest.recordDeadline(
                                                forged, current.atMs(), diagnosis))
                        .kind());
        assertSame(current, context.snapshot().deadlines().token(kind));
        if (kind == ExpirationTimer.Kind.REQUEST) {
            org.flexlb.balance.scheduler.RequestChangeAssertions.record(
                    context,
                    observedRequest ->
                            observedRequest.recordWorkerRequestEvent(
                                    new WorkerRequestEvent(
                                            fixture.route(),
                                            fixture.reservation(),
                                            RoleType.PREFILL,
                                            WorkerRequestEventType.ACTIVE,
                                            0L,
                                            null,
                                            current.atMs() - 1L)));
        }
        org.flexlb.balance.scheduler.RequestChangeAssertions.record(
                context,
                observedRequest -> observedRequest.recordDeadline(current, current.atMs(), null));
        if (kind == ExpirationTimer.Kind.REQUEST) {
            assertNull(context.snapshot().deadlines().request());
            org.junit.jupiter.api.Assertions.assertEquals(
                    RequestContext.SendResult.CANCELLED,
                    context.beginSend(fixture.delivery(), current.atMs()),
                    "consuming the refreshed request timer cannot remove the original absolute send deadline");
        } else if (kind == ExpirationTimer.Kind.VISIBILITY) {
            assertNull(context.snapshot().deadlines().visibility());
            assertTrue(context.snapshot().engine().visibilityOverdue());
        } else {
            assertTrue(context.snapshot().close().expired());
        }
    }
}
