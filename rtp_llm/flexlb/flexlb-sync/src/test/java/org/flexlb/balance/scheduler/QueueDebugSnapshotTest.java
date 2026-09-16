package org.flexlb.balance.scheduler;

import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class QueueDebugSnapshotTest {
    @Test
    void boundedVisitPreservesPriorityFifoAndQueueMembership() {
        var queue = new OrderedRequestQueue(true);
        queue.add(entry(1, 10));
        queue.add(entry(2, 90));
        queue.add(entry(3, 90));
        var ids = new ArrayList<Long>();
        queue.debugVisit(entry -> {
            if (ids.size() == 2) {
                return false;
            }
            ids.add(entry.context.getRequestId());
            return true;
        });
        assertEquals(java.util.List.of(2L, 3L), ids);
        assertEquals(3, queue.size());
        var first = new ArrayList<Long>();
        queue.debugVisit(entry -> {
            first.add(entry.context.getRequestId());
            return false;
        });
        assertEquals(java.util.List.of(2L), first);
    }

    @Test
    void observingClaimantNeverConsumesTheCapacityEdge() {
        var blocked = new PlacementWaitQueue(true, new PlacementAvailability());
        var first = entry(1, 90);
        var second = entry(2, 10);
        first.sequence = 1;
        second.sequence = 2;
        var key = PlacementKey.exact(RoleType.PREFILL, "g", "p:1");
        assertTrue(blocked.park(first, key, 0));
        assertTrue(blocked.park(second, key, 0));
        blocked.capacityChanged(key);
        assertTrue(blocked.isWaiting(second));
        var resumed = new ArrayList<GlobalQueueEntry>();
        blocked.resumeReady(1, resumed::add);
        assertEquals(java.util.List.of(first), resumed);
        assertFalse(blocked.isWaiting(first));
        assertTrue(blocked.isWaiting(second));
    }

    private static GlobalQueueEntry entry(long id, int priority) {
        var context = mock(BalanceContext.class);
        when(context.getRequestId()).thenReturn(id);
        return new GlobalQueueEntry(context, new CompletableFuture<>(), priority, "g");
    }
}
