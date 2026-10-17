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
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
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
                var admitted = new HashMap<String, WorkerStatus.PollLease>();
                for (var call : calls) {
                    var lease = call.get(3, TimeUnit.SECONDS);
                    if (lease != null) {
                        assertNull(admitted.put(lease.pollTrace().snapshot().kind(), lease));
                    }
                }
                try {
                    assertEquals(2, admitted.size(), "one status poll and one cache poll may be active");
                    var snapshot = status.trace().snapshot();
                    assertEquals(admitted.get("STATUS").pollTrace().snapshot().syncCycleId(),
                            snapshot.statusPoll().syncCycleId());
                    assertEquals(admitted.get("CACHE").pollTrace().snapshot().syncCycleId(),
                            snapshot.cachePoll().syncCycleId());
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
        org.junit.jupiter.api.Assertions.assertSame(health, status.pollHealth());
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
        org.junit.jupiter.api.Assertions.assertSame(health, status.pollHealth());
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
                RoleType.PREFILL, null, "127.0.0.1",
                8080, 8081, "test-site");
    }
}
