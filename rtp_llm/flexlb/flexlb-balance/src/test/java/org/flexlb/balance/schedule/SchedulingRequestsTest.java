package org.flexlb.balance.schedule;

import org.flexlb.balance.queue.GlobalQueueEntry;
import org.flexlb.config.FlexlbConfig;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Checks waiting counts, overlapping tasks and callbacks that outlive their queue registration.
 */
class SchedulingRequestsTest {
    @Test
    void waitingExcludesOneRequestOnlyOnceWhenDecisionAndPreemptionOverlap() {
        var requests = new SchedulingRequests();
        var first = entry();
        var second = entry();
        requests.register(first);
        requests.register(second);
        assertEquals(2, requests.waitingRequestCount());
        requests.startDecisions(List.of(first));
        assertEquals(1, requests.waitingRequestCount());
        assertTrue(requests.tryStartPreemption(first, 1));
        assertEquals(1, requests.waitingRequestCount());
        var pending = requests.pendingPreemptions().getFirst();
        requests.finishDecision(first);
        assertEquals(0, requests.decisionCount());
        assertEquals(1, requests.preemptionCount());
        assertEquals(1, requests.waitingRequestCount());
        assertSame(pending, requests.finishPreemption(first));
        assertFalse(
                pending.isDone(), "the caller completes the notification after releasing its lock");
        assertEquals(2, requests.waitingRequestCount());
        assertFalse(requests.hasWork());
    }

    @Test
    void removedRequestsKeepTaskSlotsUntilTheirCallbacksReturn() {
        var requests = new SchedulingRequests();
        var first = entry();
        var second = entry();
        requests.register(first);
        requests.register(second);
        requests.startDecisions(List.of(first));
        assertTrue(requests.tryStartPreemption(first, 1));
        requests.remove(first);
        assertEquals(1, requests.waitingRequestCount());
        assertFalse(requests.tryStartPreemption(second, 1));
        assertEquals(1, requests.decisionCount());
        assertEquals(1, requests.preemptionCount());
        requests.finishDecision(first);
        var completion = requests.finishPreemption(first);
        assertFalse(completion.isDone());
        assertEquals(0, requests.decisionCount());
        assertEquals(0, requests.preemptionCount());
        assertTrue(requests.tryStartPreemption(second, 1));
        assertEquals(0, requests.waitingRequestCount());
    }

    @Test
    void oldCallbacksCannotRemoveOrDiscountAReplacementRegistration() {
        var requests = new SchedulingRequests();
        var original = entry();
        requests.register(original);
        requests.startDecisions(List.of(original));
        assertTrue(requests.tryStartPreemption(original, 1));
        requests.remove(original);
        var replacement = new GlobalQueueEntry(original.context(), "replacement");
        requests.register(replacement);
        assertEquals(1, requests.waitingRequestCount());
        requests.remove(original);
        requests.finishPreemption(original);
        requests.finishDecision(original);
        requests.finishDecision(original);
        assertNull(requests.finishPreemption(original));
        assertSame(replacement, requests.find(original.context()));
        assertEquals(1, requests.waitingRequestCount());
        assertEquals(0, requests.decisionCount());
        assertEquals(0, requests.preemptionCount());
    }

    @Test
    void drainingRegistrationRetainsOutstandingPreemptionForShutdown() {
        var requests = new SchedulingRequests();
        var entry = entry();
        requests.register(entry);
        requests.startDecisions(List.of(entry));
        assertTrue(requests.tryStartPreemption(entry, 1));
        var pending = requests.pendingPreemptions().getFirst();
        assertEquals(List.of(entry), requests.drainRegistered());
        assertNull(requests.find(entry.context()));
        assertEquals(0, requests.waitingRequestCount());
        assertEquals(List.of(pending), requests.pendingPreemptions());
        requests.finishDecision(entry);
        assertTrue(requests.hasWork());
        assertSame(pending, requests.finishPreemption(entry));
        assertFalse(requests.hasWork());
    }

    private static GlobalQueueEntry entry() {
        return new GlobalQueueEntry(new RequestContext(new FlexlbConfig()), "group");
    }
}
