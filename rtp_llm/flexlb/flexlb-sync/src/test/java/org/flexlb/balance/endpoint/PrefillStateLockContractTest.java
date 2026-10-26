package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.RequestRoute;
import org.junit.jupiter.api.Test;

import java.util.Comparator;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.FutureTask;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.ReentrantLock;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PrefillStateLockContractTest {

    @Test
    void everyLockedEntryRejectsCallsBeforeTouchingArgumentsOrState() {
        PrefillState state = state();
        List<Runnable> unguardedCalls =
                List.of(
                        () -> state.enableQueueLocked(null),
                        state::schedulingInputsChangedLocked,
                        () -> state.ownsSelectionLocked(List.of(), 0L),
                        () -> state.queueWaitCurrentLocked(null, 0L, 0L, false, 0L),
                        () -> state.commitLocked(null, null),
                        () -> state.commitStatusLocked(null, null),
                        state::snapshotLocked);

        for (Runnable call : unguardedCalls) {
            IllegalStateException failure = assertThrows(IllegalStateException.class, call::run);
            assertEquals("Prefill ownership requires queue lock", failure.getMessage());
        }
        assertEquals(0L, state.progress().ownershipVersion());
    }

    @Test
    void queueSignalAndWaitRequireTheOwnershipGuard() {
        var guard = state().queueGuard();

        assertThrows(IllegalStateException.class, guard::signal);
        assertThrows(IllegalStateException.class, guard::signalAll);
        assertThrows(IllegalStateException.class, guard::await);
        assertThrows(IllegalStateException.class, () -> guard.awaitNanos(1L));
    }

    @Test
    void sameGuardKeepsPredicateCheckAndWaitAtomic() throws Exception {
        PrefillState state = state();
        var guard = state.queueGuard();
        CountDownLatch checking = new CountDownLatch(1);
        FutureTask<Long> waiting =
                new FutureTask<>(
                        () -> {
                            guard.lockInterruptibly();
                            try {
                                while (state.progress().schedulingInputVersion() == 0L) {
                                    assertTrue(
                                            state.queueWaitCurrentLocked(null, 0L, 0L, false, 0L));
                                    checking.countDown();
                                    guard.await();
                                }
                                assertTrue(
                                        guard.isHeldByCurrentThread(),
                                        "await must reacquire the ownership guard");
                                return state.progress().schedulingInputVersion();
                            } finally {
                                guard.unlock();
                            }
                        });
        Thread waiter = Thread.ofVirtual().start(waiting);
        try {
            assertTrue(checking.await(5L, TimeUnit.SECONDS));
            guard.lock();
            try {
                state.schedulingInputsChangedLocked();
                guard.signalAll();
            } finally {
                guard.unlock();
            }
            assertEquals(1L, waiting.get(5L, TimeUnit.SECONDS));
        } finally {
            waiter.interrupt();
            waiter.join(TimeUnit.SECONDS.toMillis(5L));
            assertFalse(waiter.isAlive());
        }
    }

    private static PrefillState state() {
        return new PrefillState(
                new ReentrantLock(),
                PrefillActiveIndex.ordered(16, Comparator.comparing(RequestRoute::requestId)),
                () -> 0L);
    }
}
