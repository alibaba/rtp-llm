package org.flexlb.dao.master;

import org.flexlb.dao.route.RoleType;
import org.flexlb.testfixture.WorkerStatusResponse;
import org.flexlb.testfixture.WorkerStatusTestFixtures;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.net.ConnectException;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

class WorkerStatusHealthTest {
    private static WorkerStatus worker() {
        return WorkerStatus.createDiscovered(
                RoleType.PREFILL, "g", "127.0.0.1", 8080, 8081, "test");
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
            worker.publishPreparedStatus(
                    worker.prepareNewStatus(
                            WorkerStatusTestFixtures.observation(worker, response), lease));
        } finally {
            worker.lock.unlock();
            worker.finishStatusApplication(lease);
        }
    }

    private static WorkerStatus aliveWorker() {
        var worker = worker();
        var lease = WorkerStatusTestFixtures.beginStatusPoll(worker, true);
        try {
            publish(worker, 1L, lease);
        } finally {
            lease.close();
        }
        return worker;
    }

    @Test
    void discoveryStartsUnknownAndOnlyAValidHealthyResponseEstablishesHealth() {
        var worker = worker();
        assertFalse(worker.pollHealth().reportedAlive());
        assertFalse(worker.pollHealth().healthy());
        assertFalse(worker.isAvailableForScheduling());
        assertNull(worker.retirementReason());
        assertEquals(0L, worker.pollHealth().lastSuccessfulPollUs());
        assertEquals(0L, worker.pollHealth().consecutiveWorkerFailures());

        WorkerStatusTestFixtures.receive(worker, false, new TimeoutException());
        try (var invalid = worker.tryBeginStatusPoll(2L)) {
            worker.recordStatusRpcResult(invalid, false, true, null);
        }
        assertFalse(
                worker.pollHealth().healthy(),
                "an invalid alive field is not a health observation");
        assertFalse(worker.isAvailableForScheduling());
        assertEquals(0L, worker.pollHealth().consecutiveWorkerFailures());
        assertNull(worker.retirementReason());

        WorkerStatusTestFixtures.receive(worker, true, null);
        assertTrue(worker.pollHealth().lastSuccessfulPollUs() > 0L);
        assertTrue(worker.pollHealth().healthy());
        assertTrue(worker.isAvailableForScheduling());
    }

    @Test
    void firstSuccessfulPollHasNoPriorSuccessInterval() {
        var worker = worker();
        WorkerStatusTestFixtures.receive(worker, true, null);
        assertEquals(0L, worker.pollHealth().successfulPollIntervalUs());
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
                    assertFalse(
                            worker.tryBeginStatusApplication(next), "retain only one application");
                    worker.finishStatusApplication(next);
                    assertFalse(
                            worker.tryBeginStatusApplication(next),
                            "a different poll cannot release the owner");
                }
            }
            assertSame(committed, worker.committedWorkerStatus());
            assertEquals(1L, worker.appliedStatusCursor().statusVersion());
            assertTrue(
                    worker.isAvailableForScheduling(),
                    "local backlog does not invalidate a live worker");
            assertNull(worker.retirementReason());
        } finally {
            worker.finishStatusApplication(applying);
        }
    }

    @Test
    void repeatedLocalAndUnattributedTransportFailuresPreserveEstablishedHealth() {
        var worker = aliveWorker();
        var initialHealth = worker.pollHealth();
        long lastSuccessUs = worker.pollHealth().lastSuccessfulPollUs();
        List<Throwable> failures =
                List.of(
                        new OutOfMemoryError("direct buffer"),
                        new TimeoutException(),
                        new ConnectException(),
                        new OutOfMemoryError("heap"));
        for (int index = 0; index < failures.size(); index++) {
            try (var lease = worker.tryBeginStatusPoll(index + 2L);
                    var cacheLease = worker.tryBeginCachePoll(index + 2L)) {
                assertNotNull(lease, "failure must not stop the next status poll");
                assertNotNull(cacheLease, "failure must not stop independent cache polling");
                worker.recordStatusRpcResult(lease, false, false, failures.get(index));
                assertSame(
                        initialHealth,
                        worker.pollHealth(),
                        "unconfirmed failures do not republish health");
                assertEquals(0L, worker.pollHealth().consecutiveWorkerFailures());
                assertEquals(lastSuccessUs, worker.pollHealth().lastSuccessfulPollUs());
                assertTrue(worker.pollHealth().reportedAlive());
                assertTrue(worker.pollHealth().healthy());
                assertTrue(worker.isAvailableForScheduling());
                assertNull(worker.retirementReason());
                assertTrue(worker.isActiveGeneration());
            }
        }
        WorkerStatusTestFixtures.receive(worker, true, null);
        assertEquals(0L, worker.pollHealth().consecutiveWorkerFailures());
        assertTrue(worker.isAvailableForScheduling());
    }

    @Test
    void thirdValidNotAliveReportMakesTheGenerationUnavailable() {
        var worker = aliveWorker();
        for (int attempt = 1; attempt <= 3; attempt++) {
            WorkerStatusTestFixtures.receive(worker, false, null);
            assertEquals(attempt, worker.pollHealth().consecutiveWorkerFailures());
            assertFalse(worker.pollHealth().reportedAlive());
            assertEquals(attempt < 3, worker.pollHealth().healthy());
            assertEquals(attempt < 3, worker.isAvailableForScheduling());
            assertEquals(
                    attempt < 3 ? null : WorkerStatus.RetirementReason.CONSECUTIVE_WORKER_FAILURES,
                    worker.retirementReason());
        }
        assertNull(worker.tryBeginStatusPoll(5L));
        assertNull(worker.tryBeginCachePoll(5L));
    }

    @Test
    void unknownFailuresAndInvalidResponsesNeitherCountNorClearConfirmedWorkerFailures() {
        var worker = aliveWorker();
        var firstFailure = WorkerStatusTestFixtures.receive(worker, false, null);
        WorkerStatusTestFixtures.receive(worker, false, new TimeoutException());
        assertSame(firstFailure, worker.pollHealth());
        try (var invalid = worker.tryBeginStatusPoll(4L)) {
            worker.recordStatusRpcResult(invalid, false, true, null);
        }
        assertEquals(1L, worker.pollHealth().consecutiveWorkerFailures());
        assertFalse(worker.pollHealth().reportedAlive());
        assertTrue(worker.isAvailableForScheduling());

        var secondFailure = WorkerStatusTestFixtures.receive(worker, false, null);
        WorkerStatusTestFixtures.receive(
                worker, false, new IllegalStateException(new ConnectException()));
        WorkerStatusTestFixtures.receive(worker, false, new OutOfMemoryError());
        assertSame(secondFailure, worker.pollHealth());
        assertEquals(
                2L,
                worker.pollHealth().consecutiveWorkerFailures(),
                "RPC exceptions cannot confirm or clear a not-alive report");
        assertFalse(worker.pollHealth().reportedAlive());
        assertTrue(worker.isAvailableForScheduling());
        assertNull(worker.retirementReason());

        WorkerStatusTestFixtures.receive(worker, false, null);
        assertEquals(3L, worker.pollHealth().consecutiveWorkerFailures());
        assertFalse(worker.isAvailableForScheduling());
        assertEquals(
                WorkerStatus.RetirementReason.CONSECUTIVE_WORKER_FAILURES,
                worker.retirementReason());
    }

    @Test
    void healthySameVersionResponseClearsWorkerFailuresWithoutReplayingCommittedState() {
        var worker = aliveWorker();
        var committed = worker.committedWorkerStatus();
        WorkerStatusTestFixtures.receive(worker, false, null);
        WorkerStatusTestFixtures.receive(worker, false, null);
        assertEquals(2L, worker.pollHealth().consecutiveWorkerFailures());
        WorkerStatusTestFixtures.receive(worker, true, null);
        assertSame(committed, worker.committedWorkerStatus());
        assertEquals(1L, worker.appliedStatusCursor().statusVersion());
        assertEquals(0L, worker.pollHealth().consecutiveWorkerFailures());
        assertTrue(worker.pollHealth().reportedAlive());
        assertTrue(worker.isAvailableForScheduling());

        WorkerStatusTestFixtures.receive(worker, false, null);
        WorkerStatusTestFixtures.receive(worker, false, null);
        assertEquals(2L, worker.pollHealth().consecutiveWorkerFailures());
        assertTrue(
                worker.isAvailableForScheduling(),
                "a healthy response starts a new failure sequence");
        assertNull(worker.retirementReason());
    }

    @Test
    void firstUnhealthyReportsDoNotCreateEstablishedHealth() {
        var worker = worker();
        WorkerStatusTestFixtures.receive(worker, false, null);
        WorkerStatusTestFixtures.receive(worker, false, null);
        assertFalse(worker.pollHealth().healthy());
        assertFalse(worker.isAvailableForScheduling());
        assertEquals(2L, worker.pollHealth().consecutiveWorkerFailures());
        assertNull(worker.retirementReason());
        WorkerStatusTestFixtures.receive(worker, true, null);
        assertTrue(worker.isAvailableForScheduling());
        assertEquals(0L, worker.pollHealth().consecutiveWorkerFailures());
    }

    @Test
    void latePollSuccessCannotClearANewerPollsConfirmedFailure() {
        var worker = aliveWorker();
        var old = worker.tryBeginStatusPoll(2L);
        old.close();
        try (var current = worker.tryBeginStatusPoll(3L)) {
            worker.recordStatusRpcResult(current, true, false, null);
            var failedHealth = worker.pollHealth();
            worker.recordStatusRpcResult(old, true, true, null);
            worker.recordStatusRpcResult(current, true, true, null);
            assertSame(
                    failedHealth,
                    worker.pollHealth(),
                    "a duplicate result cannot rewrite the first outcome");
            assertEquals(1L, worker.pollHealth().consecutiveWorkerFailures());
            assertTrue(worker.isAvailableForScheduling());
        }
        WorkerStatusTestFixtures.receive(worker, true, null);
        assertEquals(0L, worker.pollHealth().consecutiveWorkerFailures());
    }

    @Test
    @Timeout(5)
    void wireContactDoesNotWaitForTheWorkerApplicationLock() throws Exception {
        var worker = worker();
        var committed = worker.committedWorkerStatus();
        try (var lease = worker.tryBeginStatusPoll(1L)) {
            worker.lock.lock();
            try {
                CompletableFuture.runAsync(
                                () -> worker.recordStatusRpcResult(lease, true, true, null))
                        .get(1, TimeUnit.SECONDS);
                assertTrue(worker.pollHealth().reportedAlive());
                assertSame(committed, worker.committedWorkerStatus());
            } finally {
                worker.lock.unlock();
            }
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
            assertEquals(
                    WorkerStatus.RetirementReason.DISCOVERY_REMOVED, worker.retirementReason());
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
