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
    void preparedPollCadenceMatchesExecutionTimeCadenceAcrossIntervalChanges() {
        WorkerStatus legacy = workerStatus();
        WorkerStatus prepared = workerStatus();
        var random = new java.util.Random(1701L);
        long cycle = 0L;
        for (int i = 0; i < 10_000; i++) {
            cycle += random.nextInt(8);
            long interval = 1L + random.nextInt(150);
            boolean expected = legacy.shouldPollCache(cycle, interval);
            assertEquals(expected, prepared.prepareCachePoll(cycle, interval), "iteration=" + i);
            if (expected) {
                assertTrue(prepared.shouldPollCache(cycle, interval));
            }
        }
    }

    @Test
    void rejectedDuePollRemainsDueAfterItsBoundaryTick() {
        WorkerStatus status = workerStatus();
        org.junit.jupiter.api.Assertions.assertFalse(status.prepareCachePoll(49, 50));
        assertTrue(status.prepareCachePoll(50, 50));
        assertTrue(status.prepareCachePoll(51, 50), "rejected task did not consume the due poll");
        assertTrue(status.shouldPollCache(51, 50));
        org.junit.jupiter.api.Assertions.assertFalse(status.prepareCachePoll(52, 50));
        assertTrue(status.prepareCachePoll(101, 50), "a coalesced due tick is caught up");
    }

    private static WorkerStatus workerStatus() {
        return WorkerStatus.createDiscovered(
                RoleType.PREFILL, null, "127.0.0.1",
                8080, 8081, "test-site");
    }
}
