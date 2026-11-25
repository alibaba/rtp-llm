package org.flexlb.dao.master;

import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;

import java.util.concurrent.locks.LockSupport;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Checks exact poll-lease acquisition, transfer and release.
 */
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
    void lateStatusCompletionLeavesLatestStatusAndCacheRecordsIndependent() {
        WorkerStatus status = workerStatus();
        assertNull(status.latestStatusPollDiagnostics());
        assertNull(status.latestCachePollDiagnostics());

        try (var firstStatus = status.tryBeginStatusPoll();
                var cache = status.tryBeginCachePoll()) {
            firstStatus.close();
            try (var nextStatus = status.tryBeginStatusPoll()) {
                long rpcStartedUs = System.nanoTime() / 1000;
                nextStatus.recordRpcStarted(rpcStartedUs);
                firstStatus.completeDiagnostics(WorkerStatus.PollOutcome.APPLIED);
                assertEquals(rpcStartedUs, status.latestStatusPollDiagnostics().rpcStartedUs());
                assertNull(status.latestStatusPollDiagnostics().outcome());

                cache.completeDiagnostics(WorkerStatus.PollOutcome.FULL_SNAPSHOT_NEEDED);
                assertEquals(
                        WorkerStatus.PollOutcome.FULL_SNAPSHOT_NEEDED,
                        status.latestCachePollDiagnostics().outcome());
                assertNull(status.latestStatusPollDiagnostics().outcome());

                nextStatus.completeDiagnostics(WorkerStatus.PollOutcome.APPLICATION_BUSY);
                assertEquals(
                        WorkerStatus.PollOutcome.APPLICATION_BUSY,
                        status.latestStatusPollDiagnostics().outcome());
                assertEquals(
                        WorkerStatus.PollOutcome.FULL_SNAPSHOT_NEEDED,
                        status.latestCachePollDiagnostics().outcome());
            }
        }
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
