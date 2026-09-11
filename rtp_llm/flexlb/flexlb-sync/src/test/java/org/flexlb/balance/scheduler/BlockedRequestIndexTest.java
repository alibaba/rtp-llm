package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.concurrent.CompletableFuture;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
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
        var admission = mock(RouteAdmission.class);
        when(admission.prefillEndpoint()).thenReturn(endpoint);
        var key = PlacementKey.exact(RoleType.PREFILL, "a", "p:1");
        GlobalQueueEntry low = entry("a", 10);
        blocked.parkExact(low, key, endpoint);
        GlobalQueueEntry high = entry("a", 90);
        assertNull(blocked.findBlockingEndpoint(high, admission), "priority rescue must reach exact admission");
        blocked.parkExact(high, key, endpoint);
        blocked.capacityChanged(key);
        assertFalse(blocked.isBlocked(high));
        assertTrue(blocked.isBlocked(low));
    }

    @Test
    void selectorMissParksOnlyTheAffectedRequest() {
        GlobalQueueEntry missing = entry("a", 50);
        GlobalQueueEntry shared = entry("a", 50);
        GlobalQueueEntry independent = entry("b", 50);
        GlobalQueueEntry wildcard = entry(null, 50);
        blocked.parkSelector(missing, new PlacementKey(RoleType.DECODE, "a"));

        assertTrue(blocked.isBlocked(missing));
        assertFalse(blocked.isBlocked(shared));
        assertFalse(blocked.isBlocked(wildcard));
        assertFalse(blocked.isBlocked(independent));
        assertFalse(blocked.isBlocked(entry("a", 90)),
                "a later high-priority request still precedes the lower-priority blocker");
    }

    @Test
    void unboundSelectorMissDoesNotBlockFollowingRequests() {
        GlobalQueueEntry missing = entry(null, 50);
        blocked.parkSelector(missing, new PlacementKey(RoleType.DECODE, "a"));
        assertFalse(blocked.isBlocked(entry("b", 50)));
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
    void cancellingOneSelectorWaitDoesNotWakeAnother() {
        GlobalQueueEntry first = entry("a", 50);
        blocked.parkSelector(first, new PlacementKey(RoleType.DECODE, "a"));
        GlobalQueueEntry higher = entry("a", 90);
        blocked.parkSelector(higher, new PlacementKey(RoleType.DECODE, "a"));
        GlobalQueueEntry between = entry("a", 70);
        assertFalse(blocked.isBlocked(between));

        blocked.removeAndWakeNext(higher);
        assertFalse(blocked.isBlocked(between));
        assertTrue(blocked.isBlocked(first));
        blocked.removeAndWakeNext(first);
        assertFalse(blocked.isBlocked(entry("a", 40)));
    }

    @Test
    void selectorCapacityWakeIsConstantWorkAndReparkingRequiresAnotherEvent() {
        var wakes = new ArrayList<GlobalQueueEntry>();
        var index = new BlockedRequestIndex(true, wakes::add);
        var key = PlacementKey.anyGroup(RoleType.PREFILL);
        var waiters = new ArrayList<GlobalQueueEntry>();
        for (int i = 0; i < 1_000; i++) {
            var waiter = entry(null, 50);
            waiters.add(waiter);
            index.parkSelector(waiter, key);
        }
        index.capacityChanged(PlacementKey.exact(RoleType.PREFILL, "g", "p"));
        assertEquals(List.of(waiters.getFirst()), wakes);
        for (var waiter : waiters) {
            assertFalse(index.isBlocked(waiter));
        }
        index.parkSelector(waiters.getFirst(), key);
        assertTrue(index.isBlocked(waiters.getFirst()));
        assertFalse(index.isBlocked(waiters.getLast()));
        index.capacityChanged(PlacementKey.exact(RoleType.PREFILL, "g", "p"));
        assertFalse(index.isBlocked(waiters.getFirst()));
        assertEquals(List.of(waiters.getFirst(), waiters.get(1), waiters.getFirst()), wakes);
    }

    @Test
    void selectorCapacityWakeResumesEveryPriorityBucket() {
        var index = new BlockedRequestIndex(true, queue::markRequestReadyForRetry);
        var key = PlacementKey.anyGroup(RoleType.PREFILL);
        var low = entry(null, 10);
        var high = entry(null, 90);
        index.parkSelector(low, key);
        index.parkSelector(high, key);
        assertTrue(queue.scanForPlanningCandidates(10, 10, entry -> !index.isBlocked(entry)).isEmpty());
        assertFalse(queue.hasUnscannedRequests());

        index.capacityChanged(PlacementKey.exact(RoleType.PREFILL, "g", "p"));

        assertEquals(List.of(high, low),
                queue.scanForPlanningCandidates(10, 10, entry -> !index.isBlocked(entry)));
        index.removeAndWakeNext(high);
        index.removeAndWakeNext(low);
        index.capacityChanged(PlacementKey.exact(RoleType.PREFILL, "g", "p"));
        assertFalse(queue.hasUnscannedRequests());
    }

    @Test
    void selectorRetryChainBypassesUnrelatedBacklogAfterOneCapacityEvent() {
        for (boolean priorityOrdering : new boolean[]{false, true}) {
            var ordered = new OrderedRequestQueue(priorityOrdering);
            var index = new BlockedRequestIndex(priorityOrdering, ordered::markRequestReadyForRetry);
            var available = new PlacementKey(RoleType.PREFILL, "available");
            var unavailable = new PlacementKey(RoleType.PREFILL, "unavailable");
            var ready = new ArrayList<GlobalQueueEntry>();
            for (int i = 0; i < 10_000; i++) {
                var entry = new GlobalQueueEntry(null, new CompletableFuture<>(), 50);
                ordered.add(entry);
                index.parkSelector(entry, i < 100 ? available : unavailable);
                if (i < 100) {
                    ready.add(entry);
                }
            }
            assertTrue(ordered.scanForPlanningCandidates(15, 100,
                    entry -> !index.isBlocked(entry)).isEmpty());
            index.capacityChanged(available);
            for (var expected : ready) {
                assertEquals(List.of(expected), ordered.scanForPlanningCandidates(15, 30,
                        entry -> !index.isBlocked(entry)));
                ordered.remove(expected);
                index.removeAndWakeNext(expected);
            }
        }
    }

    @Test
    void reparkingSelectorHeadStillReleasesItsReadySuccessor() {
        var index = new BlockedRequestIndex(true, queue::markRequestReadyForRetry);
        var key = PlacementKey.anyGroup(RoleType.PREFILL);
        var first = entry(null, 50);
        var second = entry(null, 50);
        index.parkSelector(first, key);
        index.parkSelector(second, key);
        assertTrue(queue.scanForPlanningCandidates(15, 30, e -> !index.isBlocked(e)).isEmpty());
        index.capacityChanged(key);
        assertEquals(List.of(first), queue.scanForPlanningCandidates(15, 30,
                e -> !index.isBlocked(e)));
        index.parkSelector(first, key);
        assertTrue(index.isBlocked(first));
        assertEquals(List.of(second), queue.scanForPlanningCandidates(15, 30,
                e -> !index.isBlocked(e)));
        queue.remove(second);
        index.removeAndWakeNext(second);
        index.capacityChanged(key);
        assertEquals(List.of(first), queue.scanForPlanningCandidates(15, 30,
                e -> !index.isBlocked(e)));
    }

    @Test
    void overlappingSelectorWakeChainsRetainPriorityAndCancellationProgress() {
        var index = new BlockedRequestIndex(true, queue::markRequestReadyForRetry);
        var key = PlacementKey.anyGroup(RoleType.PREFILL);
        var first = entry(null, 50);
        var second = entry(null, 50);
        var high = entry(null, 90);
        index.parkSelector(first, key);
        index.parkSelector(second, key);
        queue.scanForPlanningCandidates(15, 30, e -> false);
        index.capacityChanged(key);
        index.parkSelector(high, key);
        index.capacityChanged(key);
        queue.remove(first);
        index.removeAndWakeNext(first);
        assertEquals(List.of(high, second), queue.scanForPlanningCandidates(15, 30,
                e -> !index.isBlocked(e)));
        queue.remove(high);
        index.removeAndWakeNext(high);
        queue.remove(second);
        index.removeAndWakeNext(second);
        index.capacityChanged(key);
        assertFalse(queue.hasUnscannedRequests());
    }

    @Test
    void endpointRetryChainAdvancesWithoutNewArrivalsAcrossABlockedBacklog() {
        var index = new BlockedRequestIndex(true, queue::markRequestReadyForRetry);
        var endpoint = endpoint("p:1");
        var key = PlacementKey.exact(RoleType.PREFILL, "a", "p:1");
        var entries = new ArrayList<GlobalQueueEntry>();
        for (int i = 0; i < 10_000; i++) {
            var entry = new GlobalQueueEntry(null, new CompletableFuture<>(), 50, "a");
            queue.add(entry);
            entries.add(entry);
            index.parkExact(entry, key, endpoint);
        }
        assertTrue(queue.scanForPlanningCandidates(15, 30, entry -> !index.isBlocked(entry)).isEmpty());
        index.capacityChanged(key);
        for (int i = 0; i < 100; i++) {
            var expected = entries.get(i);
            assertEquals(List.of(expected),
                    queue.scanForPlanningCandidates(15, 30, entry -> !index.isBlocked(entry)));
            queue.remove(expected);
            index.removeAndWakeNext(expected);
        }
    }

    @Test
    void oneActiveRetryAdvancesUntilItParksAgain() {
        var endpoint = mock(org.flexlb.balance.endpoint.PrefillEndpoint.class);
        when(endpoint.ipPort()).thenReturn("p:1");
        var key = PlacementKey.exact(RoleType.PREFILL, "a", "p:1");
        GlobalQueueEntry first = entry("a", 50);
        GlobalQueueEntry second = entry("a", 50);
        GlobalQueueEntry third = entry("a", 50);
        assertNull(blocked.retrySource(first));
        blocked.parkExact(first, key, endpoint);
        blocked.parkExact(second, key, endpoint);
        blocked.parkExact(third, key, endpoint);
        assertNull(blocked.retrySource(first));
        assertNull(blocked.retrySource(second));
        assertNull(blocked.retrySource(third));

        blocked.capacityChanged(key);
        blocked.capacityChanged(key);
        assertFalse(blocked.isBlocked(first));
        assertTrue(blocked.isBlocked(second));
        assertTrue(blocked.isBlocked(third));
        var source = new BlockedRequestIndex.WaitTarget(endpoint, key);
        assertEquals(source, blocked.retrySource(first));
        assertNull(blocked.retrySource(second));
        assertNull(blocked.retrySource(third));

        blocked.removeAndWakeNext(first);
        blocked.removeAndWakeNext(first);
        assertFalse(blocked.isBlocked(second));
        assertTrue(blocked.isBlocked(third), "duplicate retirement must not advance twice");
        assertNull(blocked.retrySource(first));
        assertEquals(source, blocked.retrySource(second));
        assertNull(blocked.retrySource(third));

        blocked.parkExact(second, key, endpoint);
        assertTrue(blocked.isBlocked(second));
        assertTrue(blocked.isBlocked(third), "a failed active retry stops the chain");
        assertNull(blocked.retrySource(second));
        assertNull(blocked.retrySource(third));

        blocked.capacityChanged(key);
        assertFalse(blocked.isBlocked(second));
        assertTrue(blocked.isBlocked(third));
        assertEquals(source, blocked.retrySource(second));
        assertNull(blocked.retrySource(third));
        blocked.removeAndWakeNext(second);
        assertFalse(blocked.isBlocked(third));
        assertNull(blocked.retrySource(second));
        assertEquals(source, blocked.retrySource(third));
    }

    @Test
    void movingActiveRetryHandsOffOnlyItsOriginalEndpoint() {
        var source = mock(org.flexlb.balance.endpoint.PrefillEndpoint.class);
        var target = mock(org.flexlb.balance.endpoint.PrefillEndpoint.class);
        when(source.ipPort()).thenReturn("p:1");
        when(target.ipPort()).thenReturn("p:2");
        var sourceKey = PlacementKey.exact(RoleType.PREFILL, "a", "p:1");
        var targetKey = PlacementKey.exact(RoleType.PREFILL, "a", "p:2");
        GlobalQueueEntry moving = entry("a", 50);
        GlobalQueueEntry sourceSuccessor = entry("a", 50);
        GlobalQueueEntry targetSuccessor = entry("a", 50);
        blocked.parkExact(moving, sourceKey, source);
        blocked.parkExact(sourceSuccessor, sourceKey, source);
        blocked.parkExact(targetSuccessor, targetKey, target);
        blocked.capacityChanged(sourceKey);

        blocked.parkExact(moving, targetKey, target);

        assertFalse(blocked.isBlocked(sourceSuccessor));
        assertTrue(blocked.isBlocked(moving));
        assertTrue(blocked.isBlocked(targetSuccessor));
        blocked.capacityChanged(targetKey);
        assertFalse(blocked.isBlocked(moving), "the target retains queue ordering");
        assertTrue(blocked.isBlocked(targetSuccessor));
    }

    @Test
    void pausedContinuationDoesNotWakeItsSourceWhenTheActiveRetryMoves() {
        ExactWaiters source = threeWaitingRequests();
        var target = endpoint("p:2");
        var targetKey = PlacementKey.exact(RoleType.PREFILL, "a", "p:2");
        blocked.pauseContinuation(blocked.retryContinuation(source.active()));

        blocked.parkExact(source.active(), targetKey, target);

        assertTrue(blocked.isBlocked(source.next()));
        assertTrue(blocked.isBlocked(source.last()));
        blocked.capacityChanged(targetKey);
        assertFalse(blocked.isBlocked(source.active()));
        assertTrue(blocked.isBlocked(source.next()), "B's event must not release A's paused successor");

        blocked.capacityChanged(source.key());
        assertFalse(blocked.isBlocked(source.next()));
        assertTrue(blocked.isBlocked(source.last()), "A resumes one ordered retry at a time");
    }

    @Test
    void exactCapacityEventRestoresContinuationWhileTheRetryIsStillActive() {
        ExactWaiters waiters = threeWaitingRequests();
        blocked.pauseContinuation(blocked.retryContinuation(waiters.active()));

        blocked.capacityChanged(waiters.key());

        assertFalse(blocked.isBlocked(waiters.active()));
        assertTrue(blocked.isBlocked(waiters.next()), "an event must not create a second active retry");
        blocked.removeAndWakeNext(waiters.active());
        assertFalse(blocked.isBlocked(waiters.next()), "the event must clear the previous capacity pause");
        assertTrue(blocked.isBlocked(waiters.last()));
    }

    @Test
    void cancellingTheProbeCandidateBeforePauseCannotPauseTheUncheckedSuccessor() {
        ExactWaiters waiters = threeWaitingRequests();
        var candidate = blocked.retryContinuation(waiters.active());
        assertEquals(waiters.next(), candidate.next());

        waiters.next().future.cancel(false);
        blocked.pauseContinuation(candidate);
        blocked.removeAndWakeNext(waiters.active());

        assertFalse(blocked.isBlocked(waiters.last()),
                "a canceled probe candidate cannot justify pausing a different request");
        assertEquals(new BlockedRequestIndex.WaitTarget(waiters.endpoint(), waiters.key()),
                blocked.retrySource(waiters.last()));
    }

    @Test
    void aNewPrioritySuccessorInvalidatesTheOldContinuationProbe() {
        ExactWaiters waiters = threeWaitingRequests();
        var candidate = blocked.retryContinuation(waiters.active());
        GlobalQueueEntry higher = entry("a", 90);
        blocked.parkExact(higher, waiters.key(), waiters.endpoint());

        blocked.pauseContinuation(candidate);
        blocked.removeAndWakeNext(waiters.active());

        assertFalse(blocked.isBlocked(higher), "the old successor's capacity check cannot pause a new priority head");
        assertTrue(blocked.isBlocked(waiters.next()));
        assertTrue(blocked.isBlocked(waiters.last()));
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void cancellingThePausedSuccessorReleasesTheUncheckedSuccessor(boolean cancelBeforeActiveRetires) {
        ExactWaiters waiters = threeWaitingRequests();
        blocked.pauseContinuation(blocked.retryContinuation(waiters.active()));

        if (cancelBeforeActiveRetires) {
            waiters.next().future.cancel(false);
            blocked.removeAndWakeNext(waiters.next());
            assertTrue(blocked.isBlocked(waiters.last()), "the original retry still owns the active slot");
            blocked.removeAndWakeNext(waiters.active());
        } else {
            blocked.removeAndWakeNext(waiters.active());
            assertTrue(blocked.isBlocked(waiters.next()));
            assertTrue(blocked.isBlocked(waiters.last()));
            waiters.next().future.cancel(false);
            blocked.removeAndWakeNext(waiters.next());
        }

        assertFalse(blocked.isBlocked(waiters.last()),
                "canceling the checked successor must not leave the unchecked request waiting for another event");
        assertEquals(new BlockedRequestIndex.WaitTarget(waiters.endpoint(), waiters.key()),
                blocked.retrySource(waiters.last()));
    }

    private ExactWaiters threeWaitingRequests() {
        var endpoint = endpoint("p:1");
        var key = PlacementKey.exact(RoleType.PREFILL, "a", "p:1");
        GlobalQueueEntry active = entry("a", 50);
        GlobalQueueEntry next = entry("a", 50);
        GlobalQueueEntry last = entry("a", 50);
        blocked.parkExact(active, key, endpoint);
        blocked.parkExact(next, key, endpoint);
        blocked.parkExact(last, key, endpoint);
        blocked.capacityChanged(key);
        return new ExactWaiters(endpoint, key, active, next, last);
    }

    private static PrefillEndpoint endpoint(String address) {
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.ipPort()).thenReturn(address);
        return endpoint;
    }

    private record ExactWaiters(PrefillEndpoint endpoint, PlacementKey key,
            GlobalQueueEntry active, GlobalQueueEntry next, GlobalQueueEntry last) {
    }

    private GlobalQueueEntry entry(String group, int priority) {
        GlobalQueueEntry entry = new GlobalQueueEntry(new BalanceContext(),
                new CompletableFuture<Response>(), priority, group);
        queue.add(entry);
        return entry;
    }
}
