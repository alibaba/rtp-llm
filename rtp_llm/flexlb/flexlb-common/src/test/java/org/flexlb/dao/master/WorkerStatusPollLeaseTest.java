package org.flexlb.dao.master;

import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.LockSupport;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

class WorkerStatusPollLeaseTest {

    @Test
    @Timeout(20)
    void competingPollsCannotReplaceTheAdmittedPollsDiagnostics() throws Exception {
        try (var executor = Executors.newFixedThreadPool(8)) {
            for (int round = 0; round < 100; round++) {
                WorkerStatus status = workerStatus();
                var start = new CountDownLatch(1);
                var calls = new ArrayList<Future<WorkerStatus.PollLease>>();
                for (long cycle = 1L; cycle <= 8L; cycle++) {
                    long syncCycle = cycle;
                    calls.add(executor.submit(() -> {
                        start.await();
                        return syncCycle % 2 == 0
                                ? status.tryBeginStatusPoll(syncCycle)
                                : status.tryBeginCachePoll(syncCycle);
                    }));
                }
                start.countDown();
                var admitted = new HashMap<Long, WorkerStatus.PollLease>();
                for (int i = 0; i < calls.size(); i++) {
                    var lease = calls.get(i).get(3, TimeUnit.SECONDS);
                    if (lease != null) {
                        admitted.put(i + 1L, lease);
                    }
                }
                try {
                    assertEquals(2, admitted.size(), "one status poll and one cache poll may be active");
                    var snapshot = status.trace().snapshot();
                    assertTrue(admitted.containsKey(snapshot.statusPoll().syncCycleId()));
                    assertTrue(admitted.containsKey(snapshot.cachePoll().syncCycleId()));
                    assertEquals(0L, snapshot.statusPoll().syncCycleId() % 2);
                    assertEquals(1L, snapshot.cachePoll().syncCycleId() % 2);
                    assertNotEquals(snapshot.statusPoll().sequence(), snapshot.cachePoll().sequence());
                } finally {
                    admitted.values().forEach(WorkerStatus.PollLease::close);
                }
            }
        }
    }

    @Test
    void retiredGenerationIgnoresLateWireSuccess() {
        WorkerStatus status = workerStatus();
        var lease = status.tryBeginStatusPoll(1L);
        var health = status.pollHealth();
        status.lock.lock();
        try {
            assertTrue(status.beginRetirementAfterEndpointGateClosed());
        } finally { status.lock.unlock(); }
        status.recordStatusRpcResult(lease, true, true, null);
        assertSame(health, status.pollHealth());
        lease.close();
        assertNull(status.tryBeginStatusPoll(2L));
        assertNull(status.tryBeginCachePoll(2L));
    }

    @Test
    void oldCompletionCannotUpdateTheNextStatusPollHealth() {
        WorkerStatus status = workerStatus();
        WorkerStatus.PollLease first = status.tryBeginStatusPoll(1L);
        first.close();
        WorkerStatus.PollLease second = status.tryBeginStatusPoll(2L);
        var health = status.pollHealth();
        status.recordStatusRpcResult(first, true, true, null);
        assertSame(health, status.pollHealth());
        status.recordStatusRpcResult(second, true, true, null);
        assertTrue(status.pollHealth().reportedAlive());
        second.close();
    }

    @Test
    void staleCloseCannotReleaseTheNextStatusPoll() {
        WorkerStatus status = workerStatus();
        WorkerStatus.PollLease first = status.tryBeginStatusPoll(1L);
        assertNotNull(first);
        first.close();

        WorkerStatus.PollLease second = status.tryBeginStatusPoll(1L);
        assertNotNull(second);
        first.close();

        assertNull(status.tryBeginStatusPoll(1L));
        second.close();
        assertNotNull(status.tryBeginStatusPoll(1L));
    }

    @Test
    @Timeout(5)
    void concurrentDuplicateNotAliveReportsCountOnlyOneWorkerFailure() throws Exception {
        WorkerStatus status = workerStatus();
        try (var initial = status.tryBeginStatusPoll(1L)) {
            status.recordStatusRpcResult(initial, true, true, null);
        }
        try (var lease = status.tryBeginStatusPoll(2L);
             var executor = Executors.newFixedThreadPool(8)) {
            var start = new CountDownLatch(1);
            var calls = new ArrayList<Future<?>>();
            for (int i = 0; i < 8; i++) {
                calls.add(executor.submit(() -> {
                    start.await();
                    status.recordStatusRpcResult(lease, true, false, null);
                    return null;
                }));
            }
            start.countDown();
            for (var call : calls) {
                call.get(2, TimeUnit.SECONDS);
            }
            assertEquals(1L, status.pollHealth().consecutiveWorkerFailures());
            assertTrue(status.trace().snapshot().statusPoll().times().healthMs() >= 0L);
            assertTrue(status.isAvailableForScheduling());
            assertNull(status.retirementReason());
            var recorded = status.pollHealth();
            status.recordStatusRpcResult(lease, true, true, null);
            assertSame(recorded, status.pollHealth());
        }
    }

    @Test
    void invalidFirstResultCannotBeReclassifiedByDuplicateCallbacks() {
        WorkerStatus status = workerStatus();
        try (var lease = status.tryBeginStatusPoll(1L)) {
            status.recordStatusRpcResult(lease, false, false, null);
            var health = status.pollHealth();
            for (int i = 0; i < 3; i++) {
                status.recordStatusRpcResult(lease, true, false, null);
            }
            status.recordStatusRpcResult(lease, true, true, null);
            assertSame(health, status.pollHealth());
            assertEquals(0L, health.consecutiveWorkerFailures());
            assertFalse(status.isAvailableForScheduling());
            assertNull(status.retirementReason());
        }
    }

    @Test
    void unconfirmedFirstResultConsumesTheLeaseWithoutPublishingHealth() {
        WorkerStatus status = workerStatus();
        try (var lease = status.tryBeginStatusPoll(1L)) {
            var health = status.pollHealth();
            status.recordStatusRpcResult(lease, false, false, new java.net.ConnectException());
            for (int i = 0; i < 3; i++) {
                status.recordStatusRpcResult(lease, true, false, null);
            }
            status.recordStatusRpcResult(lease, true, true, null);
            assertSame(health, status.pollHealth());
            assertEquals(0L, health.consecutiveWorkerFailures());
            assertFalse(status.isAvailableForScheduling());
            assertNull(status.retirementReason());
        }
    }

    @Test
    void oldGenerationLeaseCannotUpdateTheReplacementAtTheSameAddress() {
        WorkerStatus old = workerStatus();
        WorkerStatus replacement = workerStatus();
        try (var oldLease = old.tryBeginStatusPoll(1L);
             var replacementLease = replacement.tryBeginStatusPoll(1L)) {
            var health = replacement.pollHealth();
            replacement.recordStatusRpcResult(oldLease, true, true, null);
            replacement.recordStatusRpcResult(oldLease, true, false, null);
            assertSame(health, replacement.pollHealth());
            replacement.recordStatusRpcResult(replacementLease, true, true, null);
            assertTrue(replacement.isAvailableForScheduling());
            assertEquals(0L, replacement.pollHealth().consecutiveWorkerFailures());
        }
    }

    @Test
    void statusAndCacheLeasesAreIndependent() {
        WorkerStatus status = workerStatus();

        assertNotNull(status.tryBeginStatusPoll(1L));
        assertNotNull(status.tryBeginCachePoll(1L));
        assertNull(status.tryBeginStatusPoll(1L));
        assertNull(status.tryBeginCachePoll(1L));
    }

    @Test
    void cacheSuccessTransitionReturnsThePreviousSuccessPeriod() {
        WorkerStatus status = workerStatus();

        assertEquals(0L, status.recordSuccessfulCachePoll());
        LockSupport.parkNanos(2_000_000L);

        assertTrue(status.recordSuccessfulCachePoll() > 0L);
    }

    private static WorkerStatus workerStatus() {
        return WorkerStatus.createDiscovered(
                RoleType.PREFILL, null, "127.0.0.1", 8080, 8081, "test-site");
    }
}
