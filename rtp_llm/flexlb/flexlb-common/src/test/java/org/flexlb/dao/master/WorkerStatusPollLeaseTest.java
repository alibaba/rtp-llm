package org.flexlb.dao.master;

import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;

import java.util.concurrent.locks.LockSupport;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class WorkerStatusPollLeaseTest {

    @Test
    void transportProgressInvalidatesStaleRetirementAndClaimFencesLateSuccess() {
        WorkerStatus status = workerStatus();
        var lease = status.tryBeginStatusPoll(1L);
        var beforeStart = status.pollHealth();
        lease.statusRpcStarted(5000L);
        status.lock.lock();
        try {
            org.junit.jupiter.api.Assertions.assertFalse(status.tryClaimHealthRetirement(beforeStart));
            var beforeSuccess = status.pollHealth();
            status.recordStatusRpcResult(lease, true, true, null);
            org.junit.jupiter.api.Assertions.assertFalse(status.tryClaimHealthRetirement(beforeSuccess));
            var current = status.pollHealth();
            org.junit.jupiter.api.Assertions.assertTrue(status.tryClaimHealthRetirement(current));
            status.recordStatusRpcResult(lease, true, true, null);
            org.junit.jupiter.api.Assertions.assertSame(current, status.pollHealth());
        } finally { status.lock.unlock(); lease.close(); }
        assertNull(status.tryBeginStatusPoll(2L));
    }

    @Test
    void rpcAndCommitProtectionAreBoundedAndLeaseFenced() {
        WorkerStatus status = workerStatus();
        WorkerStatus.PollLease first = status.tryBeginStatusPoll(1L);
        assertTrue(!status.isStatusPollProtected(System.nanoTime() / 1000, 60_000_000L),
                "queued polls do not prove transport health");
        first.statusRpcStarted(5000);
        long now = System.nanoTime() / 1000;
        assertTrue(status.isStatusPollProtected(now, 60_000_000L));
        assertTrue(!status.isStatusPollProtected(now + 5_001_000L, 60_000_000L));
        first.statusRpcCompleted(true);
        now = System.nanoTime() / 1000;
        assertTrue(status.isStatusPollProtected(now, 60_000_000L));
        assertTrue(!status.isStatusPollProtected(now + 60_001_000L, 60_000_000L));
        first.statusRpcCompleted(false);
        assertTrue(!status.isStatusPollProtected(now, 60_000_000L));
        first.close();
        WorkerStatus.PollLease second = status.tryBeginStatusPoll(1L);
        first.statusRpcCompleted(true);
        assertTrue(!status.isStatusPollProtected(System.nanoTime() / 1000, 60_000_000L),
                "late old completions cannot protect the next poll");
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
