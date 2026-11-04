package org.flexlb.sync.runner;

import org.flexlb.dao.master.EndpointTrace;
import org.flexlb.dao.master.EndpointTrace.Stage;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.lang.management.ManagementFactory;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Measures diagnostic state and writes, excluding RPC payloads and business reducers. */
@Tag("performance-regression")
class WorkerPollDiagnosticsPerformanceTest {
    private static final int WARMUP = 10_000;
    private static final int ITERATIONS = 20_000;

    @Test
    void recordDoesNotAllocatePerScalarEvent() {
        var bean = allocationBean();
        long threadId = Thread.currentThread().threadId();
        var trace = new EndpointTrace();
        var poll = trace.beginPoll("STATUS", 1L);
        Stage point = Stage.CALLBACK_STARTED;
        for (int i = 0; i < WARMUP; i++) {
            poll.record(point);
        }
        long before = bean.getThreadAllocatedBytes(threadId);
        for (int i = 0; i < ITERATIONS; i++) {
            poll.record(point);
        }
        double bytesPerEvent =
                (double) (bean.getThreadAllocatedBytes(threadId) - before) / ITERATIONS;
        System.out.printf("Trace recorder: bytes_per_event=%.2f%n", bytesPerEvent);
        assertTrue(
                bytesPerEvent <= 8.0,
                "recording must update fixed fields without per-event allocation");
        assertEquals(1, trace.snapshot().statusPoll().syncCycleId());
    }

    @Test
    void normalPhaseRecordingDoesNotAllocateAfterLeaseCreation() {
        var bean = allocationBean();
        long threadId = Thread.currentThread().threadId();
        WorkerStatus worker = worker();
        for (int i = 0; i < WARMUP; i++) {
            WorkerStatus.PollLease lease = worker.tryBeginStatusPoll(i);
            recordNormalPhases(lease.pollTrace());
            lease.close();
        }

        long allocated = 0L;
        for (int i = 0; i < ITERATIONS; i++) {
            WorkerStatus.PollLease lease = worker.tryBeginStatusPoll(i);
            long before = bean.getThreadAllocatedBytes(threadId);
            recordNormalPhases(lease.pollTrace());
            allocated += bean.getThreadAllocatedBytes(threadId) - before;
            lease.close();
        }
        double bytesPerPoll = (double) allocated / ITERATIONS;
        System.out.printf("Worker poll diagnostic phases: bytes_per_poll=%.2f%n", bytesPerPoll);
        assertTrue(
                bytesPerPoll <= 16.0,
                "fixed primitive phase writes must not allocate maps, boxed values or event objects per poll");
    }

    @Test
    void completePollDiagnosticAllocationIsBounded() {
        var bean = allocationBean();
        long threadId = Thread.currentThread().threadId();
        WorkerStatus worker = worker();
        for (int i = 0; i < WARMUP; i++) {
            try (var lease = worker.tryBeginStatusPoll(i)) {
                recordNormalPhases(lease.pollTrace());
            }
        }
        long before = bean.getThreadAllocatedBytes(threadId);
        for (int i = 0; i < ITERATIONS; i++) {
            try (var lease = worker.tryBeginStatusPoll(i)) {
                recordNormalPhases(lease.pollTrace());
            }
        }
        double bytesPerPoll =
                (double) (bean.getThreadAllocatedBytes(threadId) - before) / ITERATIONS;
        System.out.printf(
                "Complete poll diagnostics including state: bytes_per_poll=%.2f%n", bytesPerPoll);
        assertTrue(
                bytesPerPoll <= 2048,
                "each poll must retain only bounded primitive diagnostic state");
    }

    @Test
    void busyLeaseAttemptsDoNotAllocatePerSkippedTick() {
        var bean = allocationBean();
        long threadId = Thread.currentThread().threadId();
        WorkerStatus worker = worker();
        WorkerStatus.PollLease held = worker.tryBeginStatusPoll(1L);
        try {
            held.pollTrace().record(Stage.CALLBACK_READY);
            for (int i = 0; i < WARMUP; i++) {
                if (worker.tryBeginStatusPoll(i) != null) {
                    throw new AssertionError("an active poll must retain its lease");
                }
            }
            long before = bean.getThreadAllocatedBytes(threadId);
            for (int i = 0; i < ITERATIONS; i++) {
                if (worker.tryBeginStatusPoll(i) != null) {
                    throw new AssertionError("an active poll must retain its lease");
                }
            }
            double bytesPerSkip =
                    (double) (bean.getThreadAllocatedBytes(threadId) - before) / ITERATIONS;
            System.out.printf("Worker poll busy lease: bytes_per_skip=%.2f%n", bytesPerSkip);
            assertTrue(
                    bytesPerSkip <= 8.0,
                    "busy skips must aggregate without per-tick strings, maps or snapshots");

            var snapshot = worker.trace().snapshot();
            assertEquals(WARMUP + ITERATIONS, snapshot.statusPoll().skippedBusy());
            assertEquals("CALLBACK_READY", snapshot.statusPoll().stage());
        } finally {
            held.close();
        }
    }

    @Test
    void fleetHeartbeatsKeepOneApplicationPerWorker() {
        // Exercise the new coordination at 750P/750D; excludes transport payload allocation.
        int workersPerRole = 750;
        int warmupRounds = 20;
        int measuredRounds = 100;
        WorkerStatus[] workers = new WorkerStatus[2 * workersPerRole];
        WorkerStatus.PollLease[] applications = new WorkerStatus.PollLease[workers.length];
        for (int i = 0; i < workers.length; i++) {
            workers[i] =
                    WorkerStatus.createDiscovered(
                            i < workersPerRole ? RoleType.PREFILL : RoleType.DECODE,
                            "test",
                            "127.0.0.1",
                            10000 + i,
                            20000 + i,
                            "test");
            applications[i] = workers[i].tryBeginStatusPoll(0L);
            assertTrue(workers[i].tryBeginStatusApplication(applications[i]));
            applications[i].close();
        }
        var bean = allocationBean();
        long threadId = Thread.currentThread().threadId();
        long started = 0L;
        long allocated = 0L;
        try {
            for (int round = 0; round < warmupRounds + measuredRounds; round++) {
                if (round == warmupRounds) {
                    started = System.nanoTime();
                    allocated = bean.getThreadAllocatedBytes(threadId);
                }
                for (WorkerStatus worker : workers) {
                    try (var lease = worker.tryBeginStatusPoll(round + 1L)) {
                        if (lease == null) {
                            throw new AssertionError("application must not block RPC");
                        }
                        worker.recordStatusRpcResult(lease, true, true, null);
                        if (worker.tryBeginStatusApplication(lease)) {
                            throw new AssertionError("a second application must not be retained");
                        }
                    }
                }
            }
            long polls = (long) measuredRounds * workers.length;
            double bytesPerPoll =
                    (double) (bean.getThreadAllocatedBytes(threadId) - allocated) / polls;
            double nsPerPoll = (double) (System.nanoTime() - started) / polls;
            System.out.printf(
                    "750P/750D blocked application bookkeeping: polls=%d bytes_per_poll=%.2f ns_per_poll=%.2f%n",
                    polls, bytesPerPoll, nsPerPoll);
            assertTrue(
                    bytesPerPoll <= 2048,
                    "continuing heartbeats must retain only bounded poll state");
        } finally {
            for (int i = 0; i < workers.length; i++) {
                workers[i].finishStatusApplication(applications[i]);
            }
        }
        for (WorkerStatus worker : workers) {
            try (var lease = worker.tryBeginStatusPoll(warmupRounds + measuredRounds + 1L)) {
                assertTrue(
                        worker.tryBeginStatusApplication(lease),
                        "application ownership must be reusable");
                worker.finishStatusApplication(lease);
            }
        }
    }

    @Test
    void failureEpisodesKeepFixedPollStateWithoutEventHistory() {
        WorkerStatus worker = worker();
        WorkerStatus.PollLease held = worker.tryBeginStatusPoll(1L);
        try {
            held.pollTrace()
                    .record(Stage.RPC_COMPLETED, new IllegalStateException("worker unavailable"));
            for (int i = 0; i < 20_000; i++) {
                held.pollTrace().record(Stage.STATUS_LOCK_ACQUIRED);
            }
            var first = worker.trace().snapshot();
            for (int i = 0; i < 20_000; i++) {
                held.pollTrace().record(Stage.STATUS_LOCK_ACQUIRED);
            }
            var second = worker.trace().snapshot();
            assertEquals(first.statusPoll().pollId(), second.statusPoll().pollId());
            assertEquals("RPC_COMPLETED", second.statusPoll().failureStage());
            assertTrue(second.statusPoll().error().contains("worker unavailable"));
            assertEquals(1, second.statusPoll().syncCycleId());
        } finally {
            held.close();
        }
    }

    private static void recordNormalPhases(EndpointTrace.Poll pollTrace) {
        pollTrace.record(Stage.RUNNER_SUBMITTED);
        pollTrace.record(Stage.RUNNER_STARTED);
        pollTrace.record(Stage.RPC_STARTED);
        pollTrace.record(Stage.RPC_COMPLETED);
        pollTrace.record(Stage.STATUS_HEALTH_UPDATED);
        pollTrace.record(Stage.CALLBACK_READY);
        pollTrace.record(Stage.CALLBACK_STARTED);
        pollTrace.record(Stage.CONVERSION_STARTED);
        pollTrace.record(Stage.CONVERSION_COMPLETED);
        pollTrace.record(Stage.STATUS_LOCK_ATTEMPT);
        pollTrace.record(Stage.STATUS_LOCK_ACQUIRED);
        pollTrace.record(Stage.STATUS_COMMITTED);
        pollTrace.record(Stage.STATUS_APPLIED);
        pollTrace.record(Stage.STATUS_LOCK_RELEASED);
        pollTrace.record(Stage.CALLBACK_ENDED);
    }

    private static WorkerStatus worker() {
        return WorkerStatus.createDiscovered(
                RoleType.DECODE, "test", "127.0.0.1", 8080, 8081, "test");
    }

    private static com.sun.management.ThreadMXBean allocationBean() {
        var bean = (com.sun.management.ThreadMXBean) ManagementFactory.getThreadMXBean();
        assertTrue(bean.isThreadAllocatedMemorySupported());
        bean.setThreadAllocatedMemoryEnabled(true);
        return bean;
    }
}
