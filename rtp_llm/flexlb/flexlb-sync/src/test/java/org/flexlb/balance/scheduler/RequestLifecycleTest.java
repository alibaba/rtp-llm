package org.flexlb.balance.scheduler;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class RequestLifecycleTest {

    @Test
    void normalDispatchTransitionsToCompleted() {
        RequestLifecycle lifecycle = new RequestLifecycle(1L);

        lifecycle.startDispatch(101L);
        assertEquals(RequestLifecycleState.DISPATCHING, lifecycle.snapshot().state());
        assertEquals(RequestLifecycleState.ACKNOWLEDGED, lifecycle.acknowledge().state());
        RequestLifecycleSnapshot completed = lifecycle.complete("decode finished");

        assertEquals(RequestLifecycleState.COMPLETED, completed.state());
        assertEquals(101L, completed.batchId());
        assertTrue(completed.state().isTerminal());
    }

    @Test
    void cancelDuringDispatchRequiresReconciliation() {
        RequestLifecycle lifecycle = new RequestLifecycle(2L);
        lifecycle.startDispatch(102L);

        RequestLifecycleSnapshot requested = lifecycle.requestCancel(CancelReason.CLIENT_CANCELLED);
        assertEquals(RequestLifecycleState.CANCEL_REQUESTED, requested.state());
        assertFalse(requested.state().isTerminal());
        assertEquals(RequestLifecycleState.CANCEL_REQUESTED, lifecycle.acknowledge().state());

        RequestLifecycleSnapshot cancelled = lifecycle.finishCancellation();
        assertEquals(RequestLifecycleState.CANCELLED, cancelled.state());
    }

    @Test
    void deadlineTransitionsDirectlyToTimedOutAndCannotBeOverwritten() {
        RequestLifecycle lifecycle = new RequestLifecycle(3L);
        lifecycle.startDispatch(103L);

        RequestLifecycleSnapshot timedOut = lifecycle.requestCancel(CancelReason.DEADLINE_EXCEEDED);
        assertEquals(RequestLifecycleState.TIMED_OUT, timedOut.state());
        assertEquals(RequestLifecycleState.TIMED_OUT, lifecycle.acknowledge().state());
        assertEquals(RequestLifecycleState.TIMED_OUT, lifecycle.fail("late failure").state());
        assertEquals(RequestLifecycleState.TIMED_OUT, lifecycle.complete("late completion").state());
    }

}
