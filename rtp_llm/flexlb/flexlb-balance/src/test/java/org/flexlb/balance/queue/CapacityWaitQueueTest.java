package org.flexlb.balance.queue;

import org.flexlb.balance.schedule.RequestContext;
import org.flexlb.config.service.RoleType;
import org.flexlb.worker.endpoint.CapacityKey;
import org.flexlb.worker.endpoint.WorkerAdmissionCapacity;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Checks group retry batches, shared capacity, scan progress and notifications during a retry.
 */
class CapacityWaitQueueTest {
    private final CapacityWaitQueue queue = new CapacityWaitQueue();
    private final OrderedRequestQueue ordered = new OrderedRequestQueue(false, false);
    private final CapacityKey group = new CapacityKey(RoleType.PREFILL, "g", null);

    @Test
    void workerNotificationCanRetryAnOlderRequestWithAStaleHint() {
        var first = request(1);
        var second = request(2);
        park(first, group, Set.of(1L));
        park(second, group, Set.of(2L));

        queue.capacityChanged(capacity(RoleType.PREFILL, "g", 2, 1, 1));

        assertEquals(List.of(first), resume(10));
        assertFalse(queue.canSchedule(second), "both waiters share the one slot");
        remove(first);
        queue.capacityChanged(capacity(RoleType.PREFILL, "g", 2, 2, 1));
        assertEquals(List.of(second), resume(10));
    }

    @Test
    void overlappingHintsShareOneWorkerBudgetWithoutCreatingWaitingDomains() {
        var first = request(1);
        var second = request(2);
        park(first, group, java.util.Set.of(1L));
        park(second, group, java.util.Set.of(1L, 2L));
        changed(1, 1);
        assertEquals(List.of(first), resume(10));
        remove(first);
        changed(1, 1);
        assertTrue(resume(10).isEmpty());
        changed(2, 1);
        assertEquals(List.of(second), resume(10));
    }

    @Test
    void aFailedRetryUsesItsNewHintBeforeOtherCapacity() {
        var first = request(1);
        var second = request(2);
        var anyGroup = new CapacityKey(RoleType.PREFILL, null, null);
        park(first, anyGroup, Set.of(1L));
        changed(1, 1);
        assertEquals(List.of(first), resume(10));

        park(first, anyGroup, Set.of(2L));
        park(second, group, Set.of(1L));
        changed(2, 1);
        queue.capacityChanged(capacity(RoleType.PREFILL, "other", 2, 1, 1));

        assertEquals(
                List.of(first, second),
                resume(10),
                "the first request should use its new hint and leave the only g slot");
    }

    @Test
    void periodicProbeCanReconsiderAStaleHintBeforeMatchedTrafficSpendsEverySlot() {
        var stale = request(1);
        var matching = request(2);
        park(stale, group, java.util.Set.of(1L));
        park(matching, group, java.util.Set.of(2L));
        var available = capacity(RoleType.PREFILL, "g", 2, 1, 1);
        queue.checkCapacity(List.of(available), queue.workerAdditionVersion());
        assertEquals(List.of(stale), resume(10));
        park(stale, group, java.util.Set.of(2L));
        assertTrue(resume(10).isEmpty());
        queue.checkCapacity(List.of(available), queue.workerAdditionVersion());
        assertEquals(
                List.of(matching),
                resume(10),
                "the current finite pass continues before the failed probe");
    }

    @Test
    void unusedPeriodicCapacityCanReconsiderMultipleStaleHints() {
        var first = request(1);
        var second = request(2);
        park(first, group, java.util.Set.of(1L));
        park(second, group, java.util.Set.of(2L));
        queue.checkCapacity(
                List.of(capacity(RoleType.PREFILL, "g", 3, 1, 2)), queue.workerAdditionVersion());
        assertEquals(List.of(first, second), resume(10));
        assertTrue(resume(10).isEmpty());
    }

    @Test
    void orderedScanSkipsWaiterWithoutCapacityInItsGroup() {
        var blocked = request(1);
        var otherGroup = request(2);
        var sameGroup = request(3);
        park(blocked, new CapacityKey(RoleType.PREFILL, "blocked", null), Set.of(1L));
        park(sameGroup, group, java.util.Set.of(2L));
        park(otherGroup, new CapacityKey(RoleType.PREFILL, "other", null), java.util.Set.of(3L));
        queue.capacityChanged(capacity(RoleType.PREFILL, "g", 2, 1, 1));
        queue.capacityChanged(capacity(RoleType.PREFILL, "other", 3, 1, 1));
        assertEquals(List.of(otherGroup, sameGroup), resume(10));
        assertFalse(queue.canSchedule(blocked));
    }

    @Test
    void newArrivalsCannotKeepExtendingTheCurrentPass() {
        var first = request(1);
        var second = request(2);
        var arriving = request(3);
        park(first, group, java.util.Set.of());
        park(second, group, java.util.Set.of());
        changed(1, 1);
        assertEquals(List.of(first), resume(10));
        park(first, group, java.util.Set.of());
        park(arriving, group, java.util.Set.of());
        changed(2, 1);
        assertEquals(List.of(second), resume(10));
        park(second, group, java.util.Set.of());
        changed(3, 1);
        assertEquals(
                List.of(first),
                resume(10),
                "a new pass must reconsider the failed head before later arrivals");
    }

    @Test
    void oneGroupCanUseMultipleSlotsWithoutWaitingForRetryCompletion() {
        var first = request(1);
        var second = request(2);
        var third = request(3);
        park(third, group, java.util.Set.of());
        park(first, group, java.util.Set.of());
        park(second, group, java.util.Set.of());
        assertTrue(resume(10).isEmpty());
        changed(1, 2);
        assertEquals(List.of(first, second), resume(10));
        assertFalse(queue.canSchedule(first));
        assertFalse(queue.canSchedule(second));
        assertFalse(queue.canSchedule(third));
        assertTrue(resume(10).isEmpty());
        remove(first);
        remove(second);
        assertTrue(resume(10).isEmpty(), "completion does not replenish the budget");
        changed(2, 1);
        assertEquals(List.of(third), resume(10));
        remove(third);
        assertFalse(queue.hasWaitingRequests());
    }

    @Test
    void plannerLimitAndCapacityApplyAcrossMultipleResumeCalls() {
        var first = request(1);
        var second = request(2);
        var third = request(3);
        park(first, group, java.util.Set.of());
        park(second, group, java.util.Set.of());
        park(third, group, java.util.Set.of());
        changed(1, 2);
        assertEquals(List.of(first), resume(1));
        assertEquals(List.of(second), resume(1));
        park(first, group, java.util.Set.of());
        park(second, group, java.util.Set.of());
        assertTrue(resume(10).isEmpty(), "failed retries do not replenish the budget");
        changed(2, 1);
        assertEquals(List.of(third), resume(10));
    }

    @Test
    void failedHeadDoesNotHideTheNextRequestOnAnotherCapacityObservation() {
        var first = request(1);
        var second = request(2);
        park(first, group, java.util.Set.of());
        park(second, group, java.util.Set.of());
        changed(1, 1);
        assertEquals(List.of(first), resume(10));
        park(first, group, java.util.Set.of());
        changed(2, 1);
        assertEquals(List.of(second), resume(10));
        remove(second);
        changed(3, 1);
        assertEquals(List.of(first), resume(10));
    }

    @Test
    void periodicChecksContinueTheScanBeforeRetryingTheHead() {
        var first = request(1);
        var second = request(2);
        park(first, group, java.util.Set.of());
        park(second, group, java.util.Set.of());
        check(1);
        assertEquals(List.of(first), resume(10));
        park(first, group, java.util.Set.of());
        check(1);
        assertEquals(List.of(second), resume(10));
        park(second, group, java.util.Set.of());
        check(1);
        assertEquals(List.of(first), resume(10));
    }

    @Test
    void exactAndAnyGroupWaitersShareTheSameWorkerBudget() {
        var first = request(1);
        var second = request(2);
        park(first, group, java.util.Set.of());
        park(second, new CapacityKey(RoleType.PREFILL, null, null), java.util.Set.of());
        changed(1, 1);
        assertEquals(List.of(first), resume(10));
        remove(first);
        assertTrue(resume(10).isEmpty());
        changed(2, 1);
        assertEquals(List.of(second), resume(10));
    }

    @Test
    void unrelatedRolesAndGroupsDoNotResumeWaiters() {
        var first = request(1);
        park(first, group, java.util.Set.of());
        queue.capacityChanged(capacity(RoleType.DECODE, "g", 2, 1, 10));
        queue.capacityChanged(capacity(RoleType.PREFILL, "other", 3, 1, 10));
        assertTrue(resume(10).isEmpty());
        changed(1, 0);
        assertTrue(resume(10).isEmpty());
        changed(2, 1);
        assertEquals(List.of(first), resume(10));
    }

    @Test
    void olderPeriodicSnapshotPreservesNewWorkerCapacity() {
        var first = request(1);
        var second = request(2);
        park(first, group, Set.of());
        park(second, group, Set.of());

        long sampledVersion = queue.workerAdditionVersion();
        var snapshotBeforeWorkerJoined = List.<WorkerAdmissionCapacity>of();
        queue.capacityChanged(capacity(RoleType.PREFILL, "g", 2, 1, 2));
        queue.checkCapacity(snapshotBeforeWorkerJoined, sampledVersion);

        assertEquals(
                List.of(first, second),
                resume(2),
                "An older snapshot must preserve both retry slots announced by the new worker");
    }

    @Test
    void nextSnapshotDropsMissingWorkerCapacity() {
        var first = request(1);
        var second = request(2);
        park(first, group, Set.of());
        park(second, group, Set.of());

        long sampledVersion = queue.workerAdditionVersion();
        queue.capacityChanged(capacity(RoleType.PREFILL, "g", 2, 1, 2));
        queue.checkCapacity(List.of(), sampledVersion);
        queue.checkCapacity(List.of(), queue.workerAdditionVersion());

        assertEquals(List.of(first), resume(2), "Only the group probe can run after withdrawal");
    }

    @Test
    void lowerOrDuplicateRevisionDoesNotRestoreSpentBudget() {
        var first = request(1);
        park(first, group, java.util.Set.of());
        changed(3, 1);
        assertEquals(List.of(first), resume(10));
        park(first, group, java.util.Set.of());
        changed(2, 100);
        changed(3, 100);
        assertTrue(resume(10).isEmpty());
    }

    @Test
    void notificationDuringRetryAccountsForEveryInFlightClaim() {
        var first = request(1);
        var second = request(2);
        var third = request(3);
        park(first, group, java.util.Set.of());
        park(second, group, java.util.Set.of());
        park(third, group, java.util.Set.of());
        changed(1, 2);
        assertEquals(List.of(first, second), resume(10));
        changed(2, 2);
        assertTrue(resume(10).isEmpty());
        park(first, group, java.util.Set.of());
        assertEquals(List.of(third), resume(10));
        remove(second);
        assertTrue(
                resume(10).isEmpty(), "completed old attempts must consume their refreshed claim");
    }

    @Test
    void notificationDuringTheOnlyRetryIsRetainedAfterParking() {
        var first = request(1);
        park(first, group, java.util.Set.of());
        changed(1, 1);
        assertEquals(List.of(first), resume(10));
        changed(2, 1);
        park(first, group, java.util.Set.of());
        assertEquals(List.of(first), resume(10));
    }

    @Test
    void fullGroupGetsOnlyOneControlCheckPerPeriod() {
        var first = request(1);
        var second = request(2);
        park(first, group, java.util.Set.of());
        park(second, group, java.util.Set.of());
        check(0);
        assertEquals(List.of(first), resume(10));
        park(first, group, java.util.Set.of());
        assertTrue(resume(10).isEmpty());
        check(0);
        assertEquals(List.of(second), resume(10));
    }

    @Test
    void zeroCapacityNotificationKeepsThePeriodicProbe() {
        var request = request(1);
        park(request, group, Set.of());
        check(0);
        changed(1, 0);
        assertEquals(List.of(request), resume(1));
    }

    @Test
    void retryResultSettlesThePreviousAttemptBeforeRefreshingCapacity() {
        var first = request(1);
        var second = request(2);
        park(first, group, Set.of());
        park(second, group, Set.of());
        changed(1, 1);
        assertEquals(List.of(first), resume(1));

        queue.retry(first);
        assertTrue(resume(1).isEmpty(), "a retry does not refund its current budget");
        queue.checkCapacity(
                List.of(capacity(RoleType.PREFILL, "g", 1, 1, 1)), queue.workerAdditionVersion());
        assertEquals(List.of(second), resume(1));
        remove(second);
        changed(2, 1);
        assertEquals(List.of(first), resume(1));
    }

    @Test
    void inFlightControlCheckStillCountsWhenWorkerCapacityReturnsThenFills() {
        var probe = request(1);
        var retry = request(2);
        var waiting = request(3);
        park(probe, group, java.util.Set.of());
        park(retry, group, java.util.Set.of());
        park(waiting, group, java.util.Set.of());
        check(0);
        assertEquals(List.of(probe), resume(10));

        check(1);
        assertEquals(List.of(retry), resume(10));
        check(0);
        assertTrue(resume(10).isEmpty());

        park(probe, group, java.util.Set.of());
        assertEquals(List.of(waiting), resume(10));
    }

    @Test
    void removingTheNextWaiterPreservesProgress() {
        var first = request(1);
        var second = request(2);
        var third = request(3);
        park(first, group, java.util.Set.of());
        park(second, group, java.util.Set.of());
        park(third, group, java.util.Set.of());
        changed(1, 2);
        assertEquals(List.of(first), resume(1));
        remove(second);
        assertEquals(List.of(third), resume(1));
    }

    @Test
    void aRunningHeadDoesNotBlockAnotherGroup() {
        var first = request(1);
        var second = request(2);
        park(first, group, java.util.Set.of());
        park(second, new CapacityKey(RoleType.DECODE, "g", null), java.util.Set.of());
        changed(1, 1);
        queue.capacityChanged(capacity(RoleType.DECODE, "g", 2, 1, 1));
        assertEquals(List.of(second), resume(2, request -> request != first));
        assertEquals(List.of(first), resume(2));
    }

    @Test
    void movingToAnotherGroupAndClearingReleaseMembership() {
        var first = request(1);
        park(first, group, java.util.Set.of());
        changed(1, 1);
        assertEquals(List.of(first), resume(1));
        var other = new CapacityKey(RoleType.DECODE, "g", null);
        park(first, other, java.util.Set.of());
        assertFalse(queue.canSchedule(first));
        queue.capacityChanged(capacity(RoleType.DECODE, "g", 2, 1, 1));
        assertEquals(List.of(first), resume(1));
        queue.clear();
        assertTrue(queue.canSchedule(first));
        assertFalse(queue.hasWaitingRequests());
    }

    private void changed(long revision, long slots) {
        queue.capacityChanged(capacity(RoleType.PREFILL, "g", 1, revision, slots));
    }

    private static WorkerAdmissionCapacity capacity(
            RoleType role, String group, long generation, long revision, long slots) {
        return new WorkerAdmissionCapacity(
                CapacityKey.exact(role, group, "worker-" + generation),
                generation,
                revision,
                slots);
    }

    private void check(long slots) {
        queue.checkCapacity(
                List.of(capacity(RoleType.PREFILL, "g", 1, 0, slots)),
                queue.workerAdditionVersion());
    }

    private List<RequestContext> resume(int limit) {
        return resume(limit, ignored -> true);
    }

    private List<RequestContext> resume(
            int limit, java.util.function.Predicate<RequestContext> canRun) {
        var resumed = new ArrayList<RequestContext>();
        ordered.beginScan();
        while (resumed.size() < limit && ordered.hasUnscannedRequests()) {
            for (var request :
                    ordered.scanForPlanningCandidates(
                            limit - resumed.size(),
                            Integer.MAX_VALUE,
                            candidate -> canRun.test(candidate) && queue.canSchedule(candidate))) {
                if (queue.tryStartRetry(request)) resumed.add(request);
            }
        }
        return resumed;
    }

    private void park(RequestContext request, CapacityKey blocker, Set<Long> hints) {
        if (!ordered.contains(request)) ordered.add(request);
        queue.park(request, blocker, hints);
    }

    private void remove(RequestContext request) {
        ordered.remove(request);
        queue.remove(request);
    }

    private static RequestContext request(long sequence) {
        var request = mock(RequestContext.class);
        when(request.getPlacementSequence()).thenReturn(sequence);
        return request;
    }
}
