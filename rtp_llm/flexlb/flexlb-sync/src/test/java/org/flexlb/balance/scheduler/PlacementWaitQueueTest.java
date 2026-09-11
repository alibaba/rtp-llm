package org.flexlb.balance.scheduler;

import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.*;

class PlacementWaitQueueTest {
    private final PlacementAvailability availability = new PlacementAvailability();
    private final PlacementWaitQueue queue = new PlacementWaitQueue(true, availability);
    private long sequence;
    private static final PlacementKey A = PlacementKey.exact(RoleType.PREFILL, "g", "a:8000");
    private static final PlacementKey B = PlacementKey.exact(RoleType.PREFILL, "g", "b:8000");

    @Test
    void oneCapacityEdgeDoesNotReplayAnEntireBacklog() {
        List<GlobalQueueEntry> backlog = new ArrayList<>();
        for (int i = 0; i < 10_000; i++) {
            var entry = entry(50);
            backlog.add(entry);
            queue.park(entry, A, 0);
        }
        queue.capacityChanged(A);
        assertEquals(List.of(backlog.getFirst()), drain(16));
        assertTrue(drain(16).isEmpty(), "an active attempt owns the domain's retry opportunity");
        queue.remove(backlog.getFirst());
        assertEquals(List.of(backlog.get(1)), drain(16), "successful progress permits one more attempt");
        queue.park(backlog.get(1), A, 0);
        assertTrue(drain(16).isEmpty(), "the confirming failure ends this capacity round");
        assertTrue(queue.isWaiting(backlog.getLast()));
    }

    @Test
    void retriesAreOrderedAcrossDomainsAndThenWithinEachDomain() {
        var first = entry(50);
        var second = entry(50);
        var high = entry(90);
        queue.park(first, A, 0);
        queue.park(second, B, 0);
        queue.park(high, B, 0);
        queue.capacityChanged(B);
        queue.capacityChanged(A);
        assertTrue(queue.hasEarlierReadyRequest(second));
        assertEquals(List.of(high, first), drain(10));
        queue.remove(high);
        assertEquals(List.of(second), drain(10));
    }

    @Test
    void fifoIgnoresPriority() {
        var fifo = new PlacementWaitQueue(false, availability);
        var first = entry(10);
        var second = entry(90);
        fifo.park(first, A, 0);
        fifo.park(second, A, 0);
        fifo.capacityChanged(A);
        List<GlobalQueueEntry> resumed = new ArrayList<>();
        fifo.resumeReady(2, resumed::add);
        assertEquals(List.of(first), resumed);
        fifo.remove(first);
        fifo.resumeReady(2, resumed::add);
        assertEquals(List.of(first, second), resumed);
    }

    @Test
    void retryingElsewhereDoesNotBindRouteToTheWakeSource() {
        var first = entry(50);
        var second = entry(50);
        queue.park(first, A, 0);
        queue.park(second, A, 0);
        queue.capacityChanged(A);
        assertEquals(List.of(first), drain(1));
        queue.park(first, B, 0);
        assertTrue(drain(10).isEmpty());
        queue.capacityChanged(B);
        assertEquals(List.of(first), drain(1));
        queue.remove(first);
        assertTrue(queue.isWaiting(second));
        queue.capacityChanged(A);
        assertEquals(List.of(second), drain(1));
    }

    @Test
    void cancellingActiveRetryPreservesTheUnusedOpportunity() {
        var first = entry(50);
        var second = entry(50);
        queue.park(first, A, 0);
        queue.park(second, A, 0);
        queue.capacityChanged(A);
        assertEquals(List.of(first), drain(1));
        queue.remove(first);
        assertEquals(List.of(second), drain(1));
    }

    @Test
    void aNewEdgeDuringAnActiveRetryIsNotLost() {
        var first = entry(50);
        var second = entry(50);
        queue.park(first, A, 0);
        queue.park(second, A, 0);
        queue.capacityChanged(A);
        assertEquals(List.of(first), drain(1));
        availability.capacityChanged(A);
        queue.capacityChanged(A);
        assertTrue(drain(10).isEmpty(), "another edge cannot create two active retries");
        assertFalse(queue.park(first, A, 0), "the owner must retry the raced placement");
        queue.remove(first);
        assertEquals(List.of(second), drain(1));
    }

    @Test
    void exactReleaseMatchesRoleAndAddressAndAlsoWakesGroupAndWildcard() {
        var exact = entry(50);
        var other = entry(50);
        var group = entry(50);
        var wildcard = entry(50);
        var decode = entry(50);
        queue.park(exact, A, 0);
        queue.park(other, B, 0);
        queue.park(group, new PlacementKey(RoleType.PREFILL, "g"), 0);
        queue.park(wildcard, PlacementKey.anyGroup(RoleType.PREFILL), 0);
        queue.park(decode, PlacementKey.exact(RoleType.DECODE, "g", "a:8000"), 0);
        queue.capacityChanged(A);
        assertEquals(List.of(exact, group, wildcard), drain(10));
        assertTrue(queue.isWaiting(other));
        assertTrue(queue.isWaiting(decode));
    }

    @Test
    void groupEventDoesNotWakeExactEndpointWaiters() {
        var exact = entry(50);
        queue.park(exact, A, 0);
        queue.capacityChanged(new PlacementKey(RoleType.PREFILL, "g"));
        assertTrue(drain(10).isEmpty());
    }

    @Test
    void cancellationNeitherWakesPeersNorLosesAlreadyReadyPeers() {
        var first = entry(50);
        var second = entry(50);
        queue.park(first, A, 0);
        queue.park(second, A, 0);
        queue.remove(first);
        assertTrue(drain(10).isEmpty());
        queue.capacityChanged(A);
        var third = entry(90);
        queue.park(third, A, 0);
        queue.capacityChanged(A);
        queue.remove(third);
        assertEquals(List.of(second), drain(10));
    }

    @Test
    void removingReleasedBatchDoesNotRemoveNewWaitersInSameDomain() {
        var first = entry(50);
        var second = entry(50);
        queue.park(first, A, 0);
        queue.capacityChanged(A);
        queue.park(second, A, 0);
        queue.remove(first);
        queue.capacityChanged(A);
        assertEquals(List.of(second), drain(10));
    }

    @Test
    void edgeDuringPlanningRequiresRetryInsteadOfSleeping() {
        long snapshot = availability.sequence();
        availability.capacityChanged(A);
        assertFalse(queue.park(entry(50), A, snapshot));
        assertFalse(queue.park(entry(50), new PlacementKey(RoleType.PREFILL, "g"), snapshot));
        assertFalse(queue.park(entry(50), PlacementKey.anyGroup(RoleType.PREFILL), snapshot));
        assertTrue(queue.park(entry(50), B, snapshot));
        assertTrue(queue.park(entry(50), A, availability.sequence()));
    }

    @Test
    void topologyReplacementWakesExactWaitEvenAfterGroupChange() {
        var first = entry(50);
        queue.park(first, A, 0);
        PlacementKey replacement = PlacementKey.exact(A.role(), "new-group", A.endpoint());
        queue.capacityChanged(replacement);
        assertEquals(List.of(first), drain(1));
        long snapshot = availability.sequence();
        availability.topologyChanged(replacement);
        assertFalse(queue.park(first, A, snapshot));
    }

    @Test
    void completionWithDelayedCallbackStillGetsCleanupOpportunity() {
        var first = entry(50);
        queue.park(first, A, 0);
        first.future.complete(null);
        queue.capacityChanged(A);
        assertEquals(List.of(first), drain(1), "the owner must remove completed nodes from the ordered queue");
        assertFalse(queue.isWaiting(first));
    }

    @Test
    void clearDropsBothWaitingAndReadyBatches() {
        var first = entry(50);
        var second = entry(50);
        queue.park(first, A, 0);
        queue.capacityChanged(A);
        queue.park(second, A, 0);
        queue.clear();
        queue.capacityChanged(A);
        assertTrue(drain(10).isEmpty());
        assertFalse(queue.isWaiting(first));
        assertFalse(queue.isWaiting(second));
    }

    private GlobalQueueEntry entry(int priority) {
        var entry = new GlobalQueueEntry(null, new CompletableFuture<>(), priority);
        entry.sequence = ++sequence;
        return entry;
    }

    private List<GlobalQueueEntry> drain(int limit) {
        List<GlobalQueueEntry> resumed = new ArrayList<>();
        queue.resumeReady(limit, resumed::add);
        return resumed;
    }
}
