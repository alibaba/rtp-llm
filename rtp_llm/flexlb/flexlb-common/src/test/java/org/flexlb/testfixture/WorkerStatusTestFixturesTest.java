package org.flexlb.testfixture;

import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;

/**
 * Checks poll counting and application lease cleanup when test generations are busy or unavailable.
 */
class WorkerStatusTestFixturesTest {
    @Test
    void notAlivePollsCountOnceAndUnavailableGenerationsAreSkipped() {
        WorkerStatus worker = worker();
        var observation = observation(worker, false);
        for (int round = 1; round <= 5; round++) {
            assertNull(WorkerStatusTestFixtures.tryBeginStatusApplication(worker, observation));
            assertEquals(Math.min(round, 3), worker.pollHealth().consecutiveWorkerFailures());
        }
        assertEquals(-1L, worker.appliedStatusCursor().statusVersion());
        assertEquals(
                WorkerStatus.RetirementReason.CONSECUTIVE_WORKER_FAILURES,
                worker.retirementReason());
    }

    @Test
    void busyAndRemovedWorkersLeaveTheRecordedHealthUnchanged() {
        WorkerStatus worker = worker();
        var health = worker.pollHealth();
        try (var busy = worker.tryBeginStatusPoll(2L)) {
            assertNotNull(busy);
            assertSame(health, WorkerStatusTestFixtures.receive(worker, false, null));
        }
        worker.markRemovedFromDiscovery();
        assertSame(health, WorkerStatusTestFixtures.receive(worker, false, null));
    }

    @Test
    void applicationKeepsItsOwnLeaseAfterTheWireSlotIsReleased() {
        WorkerStatus worker = worker();
        var observation = observation(worker, true);
        var application = WorkerStatusTestFixtures.tryBeginStatusApplication(worker, observation);
        assertNotNull(application);
        try {
            try (var nextPoll = worker.tryBeginStatusPoll(2L)) {
                assertNotNull(nextPoll);
            }
            assertNull(WorkerStatusTestFixtures.tryBeginStatusApplication(worker, observation));
        } finally {
            worker.finishStatusApplication(application);
        }
        var next = WorkerStatusTestFixtures.tryBeginStatusApplication(worker, observation);
        assertNotNull(next);
        worker.finishStatusApplication(next);
    }

    private static WorkerStatus worker() {
        return WorkerStatusTestFixtures.readyForScheduling(
                WorkerStatus.createDiscovered(RoleType.PREFILL, "g", "127.0.0.1", 8000, 8001, "s"));
    }

    private static WorkerStatus.StatusObservation observation(WorkerStatus worker, boolean alive) {
        var response = new WorkerStatusResponse();
        response.setRole(RoleType.PREFILL);
        response.setAlive(alive);
        response.setStatusVersion(1L);
        response.setLatestFinishedVersion(0L);
        return WorkerStatusTestFixtures.observation(worker, response);
    }
}
