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
    void staleCloseCannotReleaseTheNextStatusPoll() {
        WorkerStatus status = workerStatus();
        WorkerStatus.PollLease first = status.tryBeginStatusPoll();
        assertNotNull(first);
        first.close();

        WorkerStatus.PollLease second = status.tryBeginStatusPoll();
        assertNotNull(second);
        first.close();

        assertNull(status.tryBeginStatusPoll());
        second.close();
        assertNotNull(status.tryBeginStatusPoll());
    }

    @Test
    void statusAndCacheLeasesAreIndependent() {
        WorkerStatus status = workerStatus();

        assertNotNull(status.tryBeginStatusPoll());
        assertNotNull(status.tryBeginCachePoll());
        assertNull(status.tryBeginStatusPoll());
        assertNull(status.tryBeginCachePoll());
    }

    @Test
    void cacheSuccessTransitionReturnsThePreviousSuccessPeriod() {
        WorkerStatus status = workerStatus();

        assertEquals(0L, status.recordSuccessfulCachePoll());
        LockSupport.parkNanos(2_000_000L);

        assertTrue(status.recordSuccessfulCachePoll() > 0L);
    }

    @Test
    void rpcAndCommitProtectionAreBoundedAndLeaseFenced() {
        WorkerStatus status = workerStatus();
        WorkerStatus.PollLease first = status.tryBeginStatusPoll();
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
        WorkerStatus.PollLease second = status.tryBeginStatusPoll();
        first.statusRpcCompleted(true);
        assertTrue(!status.isStatusPollProtected(System.nanoTime() / 1000, 60_000_000L),
                "late old completions cannot protect the next poll");
        second.close();
    }

    private static WorkerStatus workerStatus() {
        return WorkerStatus.createDiscovered(
                RoleType.PREFILL, null, "127.0.0.1",
                8080, 8081, "test-site");
    }
}
