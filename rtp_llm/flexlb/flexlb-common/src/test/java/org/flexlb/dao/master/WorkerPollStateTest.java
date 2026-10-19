package org.flexlb.dao.master;

import org.junit.jupiter.api.Test;
import java.util.ArrayList;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.atomic.AtomicInteger;
import static org.junit.jupiter.api.Assertions.*;

class WorkerPollStateTest {
    @Test void transportSuccessRemainsFreshWhileLocalApplicationIsQueued() {
        var state = new WorkerPollState();
        state.configure(10000);
        state.applied(1);
        state.rpcSucceeded(20_000_000L);
        assertFalse(state.isExpired(20_000_001L, 1, 10_000_000L));
        assertFalse(state.isSchedulingFresh(20_000_001L));
    }

    @Test void outstandingRpcProtectionHasADeadline() {
        var state = new WorkerPollState();
        state.rpcStarted(9_000_000L, 5000);
        assertFalse(state.isExpired(11_000_000L, 1, 10_000_000L));
        assertTrue(state.isExpired(14_000_001L, 1, 10_000_000L));
    }

    @Test void mailboxKeepsOnlyTheLatestUnappliedResponse() {
        var state = new WorkerPollState();
        var queued = new ArrayList<Runnable>();
        var applied = new AtomicInteger(-1);
        for (int i=0; i<1000; i++) {
            int version=i;
            state.offer(i, () -> applied.set(version), queued::add);
        }
        assertEquals(1, queued.size());
        queued.remove(0).run();
        assertEquals(999, applied.get());
        assertEquals(0, queued.size());
    }

    @Test void rejectedApplicationDoesNotRunOnTransportCompletionThread() {
        var state = new WorkerPollState();
        var applied = new AtomicInteger();
        state.offer(1, applied::incrementAndGet, command -> { throw new RejectedExecutionException(); });
        assertEquals(0, applied.get());
        state.offer(2, applied::incrementAndGet, Runnable::run);
        assertEquals(1, applied.get());
    }

    @Test void retiredGenerationCannotApplyQueuedResponse() {
        var state = new WorkerPollState();
        var queued = new ArrayList<Runnable>();
        var applied = new AtomicInteger();
        state.offer(1, applied::incrementAndGet, queued::add);
        assertTrue(state.retire());
        queued.remove(0).run();
        state.offer(2, applied::incrementAndGet, Runnable::run);
        assertEquals(0, applied.get());
    }

    @Test void localConversionFailureDoesNotEraseTransportSuccess() {
        var state = new WorkerPollState();
        state.rpcSucceeded(100);
        assertThrows(IllegalArgumentException.class, () -> state.offer(100,
                () -> { throw new IllegalArgumentException("conversion"); }, Runnable::run));
        assertEquals(100, state.lastRpcSuccessUs());
        assertEquals(0, state.lastAppliedUs());
        assertFalse(state.isRetired());
    }
}
