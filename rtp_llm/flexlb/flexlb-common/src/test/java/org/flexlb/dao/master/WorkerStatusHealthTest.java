package org.flexlb.dao.master;

import org.flexlb.dao.route.RoleType;
import org.flexlb.testfixture.WorkerStatusResponse;
import org.flexlb.testfixture.WorkerStatusTestFixtures;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

class WorkerStatusHealthTest {
    private static WorkerStatus worker() {
        return WorkerStatus.createDiscovered(RoleType.PREFILL, "g", "127.0.0.1", 8080, 8081, "test");
    }

    private static void publish(WorkerStatus worker, long version, WorkerStatus.PollLease lease) {
        var response = new WorkerStatusResponse();
        response.setRole(RoleType.PREFILL);
        response.setAlive(true);
        response.setStatusVersion(version);
        assertTrue(worker.tryBeginStatusApplication(lease));
        lease.close();
        worker.lock.lock();
        try {
            worker.publishPreparedStatus(worker.prepareNewStatus(WorkerStatusTestFixtures.observation(worker, response), lease));
        } finally { worker.lock.unlock(); worker.finishStatusApplication(lease); }
    }

    private static WorkerStatus aliveWorker() {
        var worker = worker();
        var lease = WorkerStatusTestFixtures.beginStatusPoll(worker, true);
        try {
            publish(worker, 1L, lease);
        } finally { lease.close(); }
        return worker;
    }

    @Test
    void discoveryStartsUnknownWithoutInventingAWorkerFailure() {
        var worker = worker();
        assertFalse(worker.pollHealth().reportedAlive());
        assertFalse(worker.isAvailableForScheduling());
        assertNull(worker.retirementReason());
        assertEquals(0L, worker.pollHealth().lastSuccessfulPollUs());
        WorkerStatusTestFixtures.receive(worker, true, null);
        assertTrue(worker.pollHealth().lastSuccessfulPollUs() > 0L);
        assertEquals(0L, worker.pollHealth().successfulPollIntervalUs(), "the first success has no prior interval");
    }

    @Test
    void queuedApplicationKeepsTheLastCommittedStateAvailable() {
        var worker = aliveWorker();
        var committed = worker.committedWorkerStatus();
        var applying = worker.tryBeginStatusPoll(2L);
        worker.recordStatusRpcResult(applying, true, true, null);
        assertTrue(worker.tryBeginStatusApplication(applying));
        applying.close();
        try {
            for (long cycle = 3L; cycle < 103L; cycle++) {
                try (var next = worker.tryBeginStatusPoll(cycle)) {
                    assertNotNull(next, "pending application must not block heartbeats");
                    worker.recordStatusRpcResult(next, true, true, null);
                    assertFalse(worker.tryBeginStatusApplication(next), "retain only one application");
                    worker.finishStatusApplication(next);
                    assertFalse(worker.tryBeginStatusApplication(next), "a different poll cannot release the owner");
                }
            }
            assertSame(committed, worker.committedWorkerStatus());
            assertEquals(1L, worker.appliedStatusCursor().statusVersion());
            assertTrue(worker.isAvailableForScheduling(), "local backlog does not invalidate a live worker");
            assertNull(worker.retirementReason());
        } finally { worker.finishStatusApplication(applying); }
    }

    @Test
    void threeConsecutiveTransportFailuresLatchUnavailability() {
        var worker = aliveWorker();
        WorkerStatusTestFixtures.receive(worker, false, new IllegalStateException("first RPC failed"));
        WorkerStatusTestFixtures.receive(worker, false, new IllegalStateException("second RPC failed"));
        assertTrue(worker.isAvailableForScheduling());
        assertNull(worker.retirementReason());
        var third = worker.tryBeginStatusPoll(4L);
        worker.recordStatusRpcResult(third, false, false, new IllegalStateException("third RPC failed"));
        var failedHealth = worker.pollHealth();
        assertEquals(3L, failedHealth.consecutiveTransportFailures());
        assertFalse(worker.isAvailableForScheduling());
        assertEquals(WorkerStatus.RetirementReason.CONSECUTIVE_STATUS_FAILURES,
                worker.retirementReason());
        worker.recordStatusRpcResult(third, true, true, null);
        assertSame(failedHealth, worker.pollHealth(), "late success cannot revive an unavailable generation");
        third.close();
        assertNull(worker.tryBeginStatusPoll(5L));
        assertNull(worker.tryBeginCachePoll(5L));
    }

    @Test
    void successBeforeTheFailureLimitResetsTheConsecutiveCount() {
        var worker = aliveWorker();
        WorkerStatusTestFixtures.receive(worker, false, new IllegalStateException("RPC failed"));
        WorkerStatusTestFixtures.receive(worker, true, null);
        assertEquals(0L, worker.pollHealth().consecutiveTransportFailures());
        assertTrue(worker.isAvailableForScheduling());
        assertNull(worker.retirementReason());
    }

    @Test
    void explicitWorkerDeathIsUnavailableBeforeApplicationOrCleanup() {
        var worker = aliveWorker();
        var lease = worker.tryBeginStatusPoll(2L);
        worker.recordStatusRpcResult(lease, true, false, null);
        var deadHealth = worker.pollHealth();
        assertFalse(deadHealth.reportedAlive());
        assertFalse(worker.isAvailableForScheduling());
        assertEquals(WorkerStatus.RetirementReason.WORKER_REPORTED_NOT_ALIVE,
                worker.retirementReason());
        worker.recordStatusRpcResult(lease, true, true, null);
        assertSame(deadHealth, worker.pollHealth());
        lease.close();
        assertNull(worker.tryBeginStatusPoll(3L));
        assertNull(worker.tryBeginCachePoll(3L));
    }

    @Test
    @Timeout(5)
    void wireContactDoesNotWaitForTheWorkerApplicationLock() throws Exception {
        var worker = worker();
        var committed = worker.committedWorkerStatus();
        try (var lease = worker.tryBeginStatusPoll(1L)) {
            worker.lock.lock();
            try {
                CompletableFuture.runAsync(() -> worker.recordStatusRpcResult(lease, true, true, null))
                        .get(1, TimeUnit.SECONDS);
                assertTrue(worker.pollHealth().reportedAlive());
                assertSame(committed, worker.committedWorkerStatus());
            } finally { worker.lock.unlock(); }
        }
    }

    @Test
    @Timeout(5)
    void confirmedDiscoveryRemovalDoesNotWaitForTheWorkerApplicationLock() throws Exception {
        var worker = aliveWorker();
        var lease = worker.tryBeginStatusPoll(2L);
        var committed = worker.committedWorkerStatus();
        worker.lock.lock();
        try {
            CompletableFuture.runAsync(worker::markRemovedFromDiscovery).get(1, TimeUnit.SECONDS);
            assertEquals(WorkerStatus.RetirementReason.DISCOVERY_REMOVED, worker.retirementReason());
            assertFalse(worker.isAvailableForScheduling());
            var health = worker.pollHealth();
            worker.recordStatusRpcResult(lease, true, true, null);
            assertSame(health, worker.pollHealth(), "a late RPC cannot undo confirmed removal");
            assertSame(committed, worker.committedWorkerStatus());
        } finally {
            worker.lock.unlock();
            lease.close();
        }
        assertEquals(WorkerStatus.RetirementReason.DISCOVERY_REMOVED, worker.retirementReason());
        assertNull(worker.tryBeginStatusPoll(3L));
        assertNull(worker.tryBeginCachePoll(3L));
    }
}
