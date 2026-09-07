package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.master.WorkerStatus;
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
        assertEquals(2L, queue.peekHead().context.getRequestId());
    }

    @Test
    void observingClaimantNeverConsumesTheCapacityEdge() {
        var endpoint = mock(PrefillEndpoint.class);
        var status = mock(WorkerStatus.class);
        when(endpoint.ipPort()).thenReturn("p:1");
        when(endpoint.getStatus()).thenReturn(status);
        when(status.getGenerationId()).thenReturn(7L);
        var blocked = new BlockedRequestIndex(true);
        var first = entry(1, 90);
        var second = entry(2, 10);
        var key = PlacementKey.exact(RoleType.PREFILL, "g", "p:1");
        blocked.parkExact(first, key, endpoint);
        blocked.parkExact(second, key, endpoint);
        blocked.capacityChanged(key);
        var captured = blocked.debugEntry(second);
        assertEquals("1", captured.get("claimant_request_id"));
        assertEquals("7", captured.get("blocked_endpoint_generation"));
        assertEquals(captured, blocked.debugEntry(second));
        assertFalse(blocked.isBlocked(first));
        assertTrue(blocked.isBlocked(second));
    }

    private static GlobalQueueEntry entry(long id, int priority) {
        var context = mock(BalanceContext.class);
        when(context.getRequestId()).thenReturn(id);
        return new GlobalQueueEntry(context, new CompletableFuture<>(), priority, "g");
    }
}
