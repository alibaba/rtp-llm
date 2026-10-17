package org.flexlb.dao.master;

import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

class WorkerStatusProgressTest {
    private static WorkerStatus worker() {
        return WorkerStatus.createDiscovered(RoleType.PREFILL, "g", "127.0.0.1", 8080, 8081, "test", 20_000L);
    }

    private static void publish(WorkerStatus worker, long version) {
        var response = new WorkerStatusResponse();
        response.setRole(RoleType.PREFILL);
        response.setAlive(true);
        response.setStatusVersion(version);
        worker.lock.lock();
        try {
            worker.publishPreparedStatus(worker.prepareNewStatus(worker.freezeStatusResponse(response), null));
        } finally { worker.lock.unlock(); }
    }

    @Test
    void unknownContactExpiresOnlyAfterTimeoutAndClaimCannotRepeat() {
        var worker = worker();
        long lastContact = worker.pollHealth().lastSuccessfulPollUs();
        worker.lock.lock();
        try {
            assertNull(worker.tryClaimHealthRetirement(lastContact + 10_000L, 10_000L, 60_000L));
            assertEquals(WorkerStatus.HealthRetirementReason.STATUS_EXPIRED,
                    worker.tryClaimHealthRetirement(lastContact + 10_001L, 10_000L, 60_000L));
            assertNull(worker.tryClaimHealthRetirement(lastContact + 10_002L, 10_000L, 60_000L));
        } finally { worker.lock.unlock(); }
    }

    @Test
    void rpcProtectionDefersExpiryButDoesNotHideReportedDeath() {
        var worker = worker();
        try (var lease = worker.tryBeginStatusPoll(1L)) {
            lease.statusRpcStarted(5000L);
            worker.lock.lock();
            try {
                assertNull(worker.tryClaimHealthRetirement(System.nanoTime()/1000, 0L, 60_000L));
                worker.recordStatusRpcResult(lease, true, false, null);
                assertEquals(WorkerStatus.HealthRetirementReason.WORKER_REPORTED_NOT_ALIVE,
                        worker.tryClaimHealthRetirement(System.nanoTime()/1000, Long.MAX_VALUE/2, 60_000L));
            } finally { worker.lock.unlock(); }
        }
    }

    @Test
    void threeTransportFailuresClaimRetirementBeforeTimeout() {
        var worker = worker();
        worker.lock.lock();
        try {
            worker.recordTransportFailure();
            worker.recordTransportFailure();
            assertNull(worker.tryClaimHealthRetirement(System.nanoTime()/1000, Long.MAX_VALUE/2, 60_000L));
            worker.recordTransportFailure();
            assertEquals(WorkerStatus.HealthRetirementReason.CONSECUTIVE_STATUS_FAILURES,
                    worker.tryClaimHealthRetirement(System.nanoTime()/1000, Long.MAX_VALUE/2, 60_000L));
        } finally { worker.lock.unlock(); }
    }

    @Test
    void discoveryIsUnknownAndCannotScheduleUntilAResponseIsApplied() {
        var worker = worker();
        assertEquals(WorkerStatus.StatusContact.UNKNOWN, worker.pollHealth().contact());
        assertFalse(worker.hasReceivedStatusRpc());
        assertFalse(worker.isSchedulingStatusFresh(System.nanoTime()/1000));
        try (var lease = worker.tryBeginStatusPoll(1L)) {
            worker.recordStatusRpcResult(lease, true, false, null);
        }
        assertTrue(worker.hasReceivedStatusRpc());
        assertEquals(WorkerStatus.StatusContact.DEAD, worker.pollHealth().contact());
        assertFalse(worker.isSchedulingStatusFresh(System.nanoTime()/1000));
    }

    @Test
    void snapshotAndItsOriginalReceiptArePublishedTogether() {
        var worker = worker();
        long received = System.nanoTime()/1000-50_000L;
        var lease = worker.tryBeginStatusPoll(1L);
        worker.recordStatusRpcResult(lease, true, true, null);
        assertFalse(worker.isSchedulingStatusFresh(System.nanoTime()/1000), "wire contact alone is insufficient");
        assertTrue(worker.tryBeginStatusApplication(lease, received));
        lease.close();
        try {
            publish(worker, 1L);
            assertEquals(received, worker.committedWorkerStatus().snapshotReceivedUs());
            assertEquals(received, worker.appliedSnapshotReceivedUs());
            assertTrue(worker.isSchedulingStatusFresh(received+20_000L));
            assertFalse(worker.isSchedulingStatusFresh(received+20_001L));
        } finally { worker.finishStatusApplication(lease); }
    }

    @Test
    void staleFinishCannotClearTheNextApplicationOrReplaceItsReceipt() {
        var worker = worker();
        var first = worker.tryBeginStatusPoll(1L);
        assertTrue(worker.tryBeginStatusApplication(first, 10L)); first.close();
        worker.finishStatusApplication(first);
        var second = worker.tryBeginStatusPoll(2L);
        assertTrue(worker.tryBeginStatusApplication(second, 20L)); second.close();
        worker.finishStatusApplication(first);
        try (var third = worker.tryBeginStatusPoll(3L)) {
            assertFalse(worker.tryBeginStatusApplication(third, 30L));
            publish(worker, 1L);
            assertEquals(20L, worker.appliedSnapshotReceivedUs());
        } finally { worker.finishStatusApplication(second); }
    }

    @Test
    @Timeout(5)
    void wireContactDoesNotWaitForTheWorkerApplicationLock() throws Exception {
        var worker = worker();
        try (var lease = worker.tryBeginStatusPoll(1L)) {
            worker.lock.lock();
            try {
                CompletableFuture.runAsync(() -> worker.recordStatusRpcResult(lease, true, true, null))
                        .get(1, TimeUnit.SECONDS);
                assertEquals(WorkerStatus.StatusContact.ALIVE, worker.pollHealth().contact());
                assertEquals(0L, worker.appliedSnapshotReceivedUs());
            } finally { worker.lock.unlock(); }
        }
    }

    @Test
    @Timeout(10)
    void retirementAndWireSuccessHaveOneWinner() throws Exception {
        var executor = Executors.newFixedThreadPool(2);
        try {
            for (int index = 0; index < 100; index++) {
                var worker = worker();
                try (var lease = worker.tryBeginStatusPoll(1L)) {
                    var expected = worker.pollHealth();
                    var barrier = new CyclicBarrier(2);
                    var claimed = executor.submit(() -> {
                        barrier.await();
                        worker.lock.lock();
                        try { return worker.tryClaimHealthRetirement(expected); }
                        finally { worker.lock.unlock(); }
                    });
                    var received = executor.submit(() -> {
                        barrier.await();
                        worker.recordStatusRpcResult(lease, true, true, null);
                        return null;
                    });
                    boolean retired = claimed.get(2, TimeUnit.SECONDS); received.get(2, TimeUnit.SECONDS);
                    if (retired) {
                        assertSame(expected, worker.pollHealth(), "late contact cannot revive a claimed generation");
                        assertFalse(worker.tryBeginStatusApplication(lease, 123L));
                        assertNull(worker.tryBeginStatusPoll(2L));
                    } else {
                        assertNotSame(expected, worker.pollHealth(), "fresh contact invalidates the old scan");
                        assertTrue(worker.hasReceivedStatusRpc());
                    }
                }
            }
        } finally { executor.shutdownNow(); }
    }
}
