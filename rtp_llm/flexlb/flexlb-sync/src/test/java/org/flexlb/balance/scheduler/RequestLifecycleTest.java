package org.flexlb.balance.scheduler;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
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
    void deadlineTransitionsDirectlyToTimedOutAndCannotBeOverwritten() {
        RequestLifecycle lifecycle = new RequestLifecycle(3L);
        lifecycle.startDispatch(103L);

        RequestLifecycleSnapshot timedOut = lifecycle.timeout("deadline exceeded");
        assertEquals(RequestLifecycleState.TIMED_OUT, timedOut.state());
        assertEquals(RequestLifecycleState.TIMED_OUT, lifecycle.acknowledge().state());
        assertEquals(RequestLifecycleState.TIMED_OUT, lifecycle.fail("late failure").state());
        assertEquals(RequestLifecycleState.TIMED_OUT, lifecycle.complete("late completion").state());
    }

}
