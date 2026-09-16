package org.flexlb.balance.scheduler;

import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;

import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class BlockedRequestIndexTest {
    private final OrderedRequestQueue queue = new OrderedRequestQueue(true);
    private final BlockedRequestIndex blocked = new BlockedRequestIndex(true);

    @Test
    void higherPriorityCanRescueAnEndpointWithLowerPriorityWaiters() {
        var endpoint = mock(org.flexlb.balance.endpoint.PrefillEndpoint.class);
        when(endpoint.ipPort()).thenReturn("p:1");
        var admission = mock(QueueRouteAdmission.class);
        when(admission.selectedPrefillEndpoint()).thenReturn(endpoint);
        var key = PlacementKey.exact(RoleType.PREFILL, "a", "p:1");
        GlobalQueueEntry low = entry("a", 10);
        blocked.parkExact(low, key, endpoint);
        GlobalQueueEntry high = entry("a", 90);
        assertNull(blocked.conflict(high, admission), "priority rescue must reach exact admission");
        blocked.parkExact(high, key, endpoint);
        blocked.capacityChanged(key);
        assertFalse(blocked.isBlocked(high));
        assertTrue(blocked.isBlocked(low));
    }

    @Test
    void isolatedGroupsProgressWhileSharedAndWildcardRequestsWait() {
        GlobalQueueEntry missing = entry("a", 50);
        GlobalQueueEntry shared = entry("a", 50);
        GlobalQueueEntry independent = entry("b", 50);
        GlobalQueueEntry wildcard = entry(null, 50);
        blocked.parkSelector(missing, new PlacementKey(RoleType.DECODE, "a"));

        assertTrue(blocked.isBlocked(missing));
        assertTrue(blocked.isBlocked(shared));
        assertTrue(blocked.isBlocked(wildcard));
        assertFalse(blocked.isBlocked(independent));
        assertFalse(blocked.isBlocked(entry("a", 90)),
                "a later high-priority request still precedes the lower-priority blocker");
    }

    @Test
    void unboundSelectorMissCannotClaimIsolationFromItsLastSelectedGroup() {
        GlobalQueueEntry missing = entry(null, 50);
        blocked.parkSelector(missing, new PlacementKey(RoleType.DECODE, "a"));
        assertTrue(blocked.isBlocked(entry("b", 50)));
    }

    @Test
    void capacityEventsWakeOnlyMatchingRoleAndGroup() {
        GlobalQueueEntry a = entry("a", 50);
        GlobalQueueEntry b = entry("b", 50);
        blocked.parkSelector(a, new PlacementKey(RoleType.DECODE, "a"));
        blocked.parkSelector(b, new PlacementKey(RoleType.DECODE, "b"));

        blocked.capacityChanged(PlacementKey.exact(RoleType.PREFILL, "a", "p"));
        assertTrue(blocked.isBlocked(a));
        blocked.capacityChanged(PlacementKey.exact(RoleType.DECODE, "a", "d"));
        assertFalse(blocked.isBlocked(a));
        assertTrue(blocked.isBlocked(b));
        blocked.topologyChanged(PlacementKey.exact(RoleType.DECODE, "b", "d2"));
        assertFalse(blocked.isBlocked(b));
    }

    @Test
    void cancellingTheFirstBlockerPreservesTheNextPriorityFrontier() {
        GlobalQueueEntry first = entry("a", 50);
        blocked.parkSelector(first, new PlacementKey(RoleType.DECODE, "a"));
        GlobalQueueEntry higher = entry("a", 90);
        blocked.parkSelector(higher, new PlacementKey(RoleType.DECODE, "a"));
        GlobalQueueEntry between = entry("a", 70);
        assertTrue(blocked.isBlocked(between));

        blocked.clearEntry(higher);
        assertFalse(blocked.isBlocked(between));
        assertTrue(blocked.isBlocked(entry("a", 40)));
        blocked.clearEntry(first);
        assertFalse(blocked.isBlocked(entry("a", 40)));
    }

    private GlobalQueueEntry entry(String group, int priority) {
        GlobalQueueEntry entry = new GlobalQueueEntry(new BalanceContext(),
                new CompletableFuture<Response>(), priority, group);
        queue.add(entry);
        return entry;
    }
}
