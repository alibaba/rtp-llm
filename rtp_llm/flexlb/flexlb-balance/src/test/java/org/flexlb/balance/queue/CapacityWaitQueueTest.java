package org.flexlb.balance.queue;

import org.flexlb.balance.schedule.RequestContext;
import org.flexlb.config.service.RoleType;
import org.flexlb.worker.endpoint.CapacityKey;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Checks bounded capacity retries, progress after removal and notifications during a retry.
 */
class CapacityWaitQueueTest {
    private final CapacityWaitQueue queue =
            new CapacityWaitQueue(Comparator.comparingLong(RequestContext::getPlacementSequence));
    private final CapacityKey capacity = CapacityKey.anyGroup(RoleType.DECODE);

    @Test
    void oneNotificationAllowsOnlyOneRetryUntilCapacityIsAvailable() {
        var first = request(1);
        var second = request(2);
        queue.park(first, capacity);
        queue.park(second, capacity);
        assertTrue(resume().isEmpty());
        queue.capacityChanged();
        assertEquals(List.of(first), resume());
        assertTrue(resume().isEmpty());
        queue.park(first, capacity);
        assertTrue(queue.isWaiting(first));
        assertTrue(resume().isEmpty());
        queue.capacityChanged();
        assertEquals(List.of(first), resume());
        queue.remove(first);
        assertEquals(List.of(second), resume());
        queue.remove(second);
        assertFalse(queue.hasReadyRequests());
    }

    @Test
    void changeDuringRetryIsRetainedAndCancellationPassesTheRetry() {
        var first = request(1);
        var second = request(2);
        queue.park(first, capacity);
        queue.park(second, capacity);
        queue.capacityChanged();
        assertEquals(List.of(first), resume());
        queue.capacityChanged();
        queue.park(first, capacity);
        assertEquals(List.of(first), resume());
        queue.remove(first);
        assertEquals(List.of(second), resume());
    }

    @Test
    void independentDomainsCanRetryTogetherInQueueOrder() {
        var first = request(1);
        var second = request(2);
        var third = request(3);
        queue.park(third, capacity);
        queue.park(first, capacity);
        queue.park(second, CapacityKey.anyGroup(RoleType.PREFILL));
        queue.capacityChanged();
        assertEquals(List.of(first, second), resume());
        assertTrue(queue.isWaiting(third));
        queue.clear();
        assertFalse(queue.isWaiting(third));
        assertFalse(queue.hasReadyRequests());
    }

    @Test
    void runningWaiterDoesNotBlockAnIndependentDomain() {
        var first = request(1);
        var second = request(2);
        queue.park(first, capacity);
        queue.park(second, CapacityKey.anyGroup(RoleType.PREFILL));
        queue.capacityChanged();
        var resumed = new ArrayList<RequestContext>();
        queue.resumeReady(2, request -> request != first, resumed::add);
        assertEquals(List.of(second), resumed);
        assertEquals(List.of(first), resume());
    }

    private List<RequestContext> resume() {
        var resumed = new ArrayList<RequestContext>();
        queue.resumeReady(10, request -> true, resumed::add);
        return resumed;
    }

    private static RequestContext request(long sequence) {
        var request = mock(RequestContext.class);
        when(request.getPlacementSequence()).thenReturn(sequence);
        return request;
    }
}
