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
        availability.changed(A);
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
        queue.park(group, new PlacementKey(RoleType.PREFILL, "g", null), 0);
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
        queue.capacityChanged(new PlacementKey(RoleType.PREFILL, "g", null));
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
        availability.changed(A);
        assertFalse(queue.park(entry(50), A, snapshot));
        assertFalse(queue.park(entry(50), new PlacementKey(RoleType.PREFILL, "g", null), snapshot));
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
        availability.changed(replacement);
        assertFalse(queue.park(first, A, snapshot));
    }

    @Test
    void completionWithDelayedCallbackStillGetsCleanupOpportunity() {
        var first = entry(50);
        queue.park(first, A, 0);
        first.context.getFuture().complete(null);
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

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.MethodSource("capacityEventCases")
    void allCapacityScopesMatchBothBeforeAndAfterParking(String scope, String eventKind, boolean beforePark, boolean matches) {
        PlacementKey blocker = switch (scope) {
            case "EXACT" -> A;
            case "GROUP" -> new PlacementKey(RoleType.PREFILL, "g", null);
            case "ROLE" -> PlacementKey.anyGroup(RoleType.PREFILL);
            default -> throw new AssertionError(scope);
        };
        PlacementKey event = switch (eventKind) {
            case "SAME" -> A;
            case "OTHER_ADDRESS" -> B;
            case "MOVED_GROUP" -> PlacementKey.exact(RoleType.PREFILL, "other", A.endpoint());
            case "GROUP_ONLY" -> new PlacementKey(RoleType.PREFILL, "g", null);
            case "ROLE_ONLY" -> PlacementKey.anyGroup(RoleType.PREFILL);
            case "OTHER_ROLE" -> PlacementKey.exact(RoleType.DECODE, "g", A.endpoint());
            default -> throw new AssertionError(eventKind);
        };
        var request = entry(50);
        long observed = availability.sequence();
        if (beforePark) {
            availability.changed(event);
            queue.capacityChanged(event);
            assertEquals(!matches, queue.park(request, blocker, observed));
            assertEquals(!matches, queue.isWaiting(request));
            assertTrue(drain(1).isEmpty(), "a pre-park edge is represented as immediate replan, not a duplicate wakeup");
        } else {
            assertTrue(queue.park(request, blocker, observed));
            availability.changed(event);
            queue.capacityChanged(event);
            assertEquals(matches ? List.of(request) : List.of(), drain(1));
            assertEquals(!matches, queue.isWaiting(request));
            assertTrue(drain(1).isEmpty(), "one capacity edge cannot issue the same attempt twice");
        }
        queue.remove(request);
        assertFalse(queue.isWaiting(request));
        assertTrue(drain(1).isEmpty());
    }

    static java.util.stream.Stream<org.junit.jupiter.params.provider.Arguments> capacityEventCases() {
        return java.util.stream.Stream.of("EXACT", "GROUP", "ROLE").flatMap(scope ->
                java.util.stream.Stream.of("SAME", "OTHER_ADDRESS", "MOVED_GROUP", "GROUP_ONLY", "ROLE_ONLY", "OTHER_ROLE")
                        .flatMap(event -> java.util.stream.Stream.of(false, true).map(before -> {
                            boolean matches = switch (scope) {
                                case "EXACT" -> event.equals("SAME") || event.equals("MOVED_GROUP");
                                case "GROUP" -> event.equals("SAME") || event.equals("OTHER_ADDRESS") || event.equals("GROUP_ONLY");
                                case "ROLE" -> !event.equals("OTHER_ROLE");
                                default -> throw new AssertionError(scope);
                            };
                            return org.junit.jupiter.params.provider.Arguments.of(scope, event, before, matches);
                        })));
    }

    private GlobalQueueEntry entry(int priority) {
        var context = new RequestContext(SchedulingTestConfig.newConfig());
        context.setFuture(new CompletableFuture<>());
        var request = new org.flexlb.dao.loadbalance.Request();
        request.setPriority(priority);
        context.setRequest(request);
        SchedulingTestConfig.freezeInputs(context);
        var entry = new GlobalQueueEntry(context, null);
        entry.sequence = ++sequence;
        return entry;
    }

    private List<GlobalQueueEntry> drain(int limit) {
        List<GlobalQueueEntry> resumed = new ArrayList<>();
        queue.resumeReady(limit, resumed::add);
        return resumed;
    }
}
