package org.flexlb.balance.scheduler;

import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.flexlb.balance.scheduler.ContextResult.*;
import static org.junit.jupiter.api.Assertions.*;

/** Public transitions acquire the monitor themselves; observations cannot partially bypass it. */
class RequestEngineObservationTest {
    @Test
    void prefillObservationCannotChangeActivityOrVisibilityOutsideItsAtomicTransition() throws Exception {
        verifyAtomicObservation(RoleType.PREFILL);
    }

    @Test
    void decodeObservationCannotChangeActivityOrAcceptanceOutsideItsAtomicTransition() throws Exception {
        verifyAtomicObservation(RoleType.DECODE);
    }

    @Test
    void laterActivityAndCancellationCannotMutateHistoricalSnapshotFacts() {
        var fixture = RequestDeliveryClaimStateTest.fixture(DeliveryClaimKind.BATCH_ENQUEUE, false);
        var context = fixture.context();
        var first = context.recordEngineEvidence(new EngineEvidence(fixture.route(), fixture.reservation(),
                RoleType.PREFILL, EngineKind.ACTIVE, 0L, null, 1100L)).after();
        var firstDuplicate = context.recordEngineEvidence(new EngineEvidence(fixture.route(), fixture.reservation(),
                RoleType.PREFILL, EngineKind.ACTIVE, 0L, null, 1100L));
        assertEquals(UpdateKind.NO_CHANGE, firstDuplicate.kind());
        assertSame(first, firstDuplicate.before());
        assertSame(first, firstDuplicate.after());
        var later = context.recordEngineEvidence(new EngineEvidence(fixture.route(), fixture.reservation(),
                RoleType.PREFILL, EngineKind.ACTIVE, 0L, null, 1200L)).after();
        assertNotSame(first, later);
        var laterDuplicate = context.recordEngineEvidence(new EngineEvidence(fixture.route(), fixture.reservation(),
                RoleType.PREFILL, EngineKind.ACTIVE, 0L, null, 1200L));
        assertEquals(UpdateKind.NO_CHANGE, laterDuplicate.kind());
        assertSame(later, laterDuplicate.before());
        assertSame(later, laterDuplicate.after());
        var foreign = context.recordEngineEvidence(new EngineEvidence(null, fixture.reservation(),
                RoleType.PREFILL, EngineKind.ACTIVE, 0L, null, 1250L));
        assertEquals(UpdateKind.STALE, foreign.kind());
        assertSame(later, foreign.before());
        assertSame(later, foreign.after());
        assertSame(first, firstDuplicate.after(), "later results cannot repoint an already-returned update");
        assertEquals(1100L, first.lastWorkerAtMs());
        assertEquals(1200L, later.lastWorkerAtMs());
        assertEquals(1100L, first.updatedAtMs());
        assertTrue(first.engine().prefillObserved());
        assertSame(fixture.route(), first.route());
        var cancelled = context.requestCancel(new Cancellation(0L, CancelReason.CLIENT_CANCELLED, false, 1300L,
                org.flexlb.dao.loadbalance.StrategyErrorType.DISPATCH_FAILED, "first cancellation"));
        assertNull(first.cancellation());
        assertNull(later.cancellation());
        assertNotNull(cancelled.after().cancellation());
        assertEquals("first cancellation", cancelled.after().cancellation().detail());
        assertEquals(1100L, first.lastWorkerAtMs());
        assertSame(fixture.delivery(), first.delivery().identity());
        assertNull(first.outcome());
    }

    private static void verifyAtomicObservation(RoleType role) throws Exception {
        var fixture = RequestDeliveryClaimStateTest.fixture(DeliveryClaimKind.BATCH_ENQUEUE, false);
        var context = fixture.context();
        context.recordDeliveryResult(new Prediction(fixture.delivery(), 3000L, 1002L));
        var before = context.snapshot();
        var visibility = before.deadlines().visibility();
        var inactivity = before.deadlines().inactivity();
        assertNotNull(visibility);
        assertNotNull(inactivity);
        var entered = new CountDownLatch(1);
        try (var executor = Executors.newSingleThreadExecutor()) {
            java.util.concurrent.Future<Update> applied;
            synchronized (context) {
                applied = executor.submit(() -> {
                    entered.countDown();
                    return context.recordEngineEvidence(new EngineEvidence(fixture.route(), fixture.reservation(),
                            role, EngineKind.ACTIVE, 0L, null, 2000L));
                });
                assertTrue(entered.await(5, TimeUnit.SECONDS));
                assertFalse(applied.isDone(), "the transition must wait for the existing request monitor owner");
                assertSame(before, context.snapshot());
                assertSame(visibility, context.snapshot().deadlines().visibility());
                assertSame(inactivity, context.snapshot().deadlines().inactivity());
                assertFalse(context.snapshot().engine().decodeAccepted());
                assertFalse(context.snapshot().delivery().acknowledged());
                assertNull(context.snapshot().selectedResponse());
            }
            var update = applied.get(5, TimeUnit.SECONDS);
            assertEquals(UpdateKind.APPLIED, update.kind());
            assertSame(fixture.delivery(), update.after().delivery().identity());
            assertNull(update.after().deadlines().visibility());
            assertEquals(role == RoleType.DECODE, update.after().engine().decodeAccepted());
            assertEquals(2000L, update.after().lastWorkerAtMs());
            assertFalse(update.after().delivery().acknowledged());
            assertNull(update.after().selectedResponse());
        }
    }
}
