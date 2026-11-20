package org.flexlb.worker;

import org.flexlb.config.service.RoleType;
import org.flexlb.worker.status.WorkerStatus;
import org.junit.jupiter.api.Test;

import java.util.concurrent.locks.LockSupport;

import static org.flexlb.worker.status.WorkerStatusTestSupport.createDiscovered;
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
        WorkerStatus.PollLease first = status.tryBeginStatusPoll(0L);
        assertNotNull(first);
        first.close();

        WorkerStatus.PollLease second = status.tryBeginStatusPoll(0L);
        assertNotNull(second);
        first.close();

        assertNull(status.tryBeginStatusPoll(0L));
        second.close();
        assertNotNull(status.tryBeginStatusPoll(0L));
    }

    @Test
    void statusAndCacheLeasesAreIndependent() {
        WorkerStatus status = workerStatus();

        assertNotNull(status.tryBeginStatusPoll(0L));
        assertNotNull(status.tryBeginCachePoll(0L));
        assertNull(status.tryBeginStatusPoll(0L));
        assertNull(status.tryBeginCachePoll(0L));
    }

    @Test
    void cacheSuccessTransitionReturnsThePreviousSuccessPeriod() {
        WorkerStatus status = workerStatus();

        assertEquals(0L, status.recordSuccessfulCachePoll());
        LockSupport.parkNanos(2_000_000L);

        assertTrue(status.recordSuccessfulCachePoll() > 0L);
    }

    private static WorkerStatus workerStatus() {
        return createDiscovered(RoleType.PREFILL, null, "127.0.0.1", 8080, 8081, "test-site");
    }
}
