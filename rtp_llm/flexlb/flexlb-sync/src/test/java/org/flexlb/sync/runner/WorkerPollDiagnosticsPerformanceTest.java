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

/** Measures diagnostic writes, excluding lease construction, RPC payloads and business reducers. */
@Tag("performance-regression")
class WorkerPollDiagnosticsPerformanceTest {
    private static final int WARMUP = 10_000;
    private static final int ITERATIONS = 20_000;

    @Test
    void syncPhaseRecordingDoesNotAllocatePerCycle() {
        var bean = allocationBean();
        long threadId = Thread.currentThread().threadId();
        var trace = new EndpointTrace.Poll("SYNC", 1L);
        for (int i = 0; i < WARMUP; i++) {
            recordSyncPhases(trace);
        }
        long before = bean.getThreadAllocatedBytes(threadId);
        for (int i = 0; i < ITERATIONS; i++) {
            recordSyncPhases(trace);
        }
        double bytesPerCycle = (double) (bean.getThreadAllocatedBytes(threadId) - before) / ITERATIONS;
        System.out.printf("Sync phase trace: bytes_per_cycle=%.2f%n", bytesPerCycle);
        assertTrue(bytesPerCycle <= 8.0,
                "sync phase recording must use fixed fields without per-worker events or snapshots");
        assertEquals(Stage.RUNNER_ENDED.name(), trace.snapshot().stage());
    }

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
        double bytesPerEvent = (double) (bean.getThreadAllocatedBytes(threadId) - before) / ITERATIONS;
        System.out.printf("Trace recorder: bytes_per_event=%.2f%n", bytesPerEvent);
        assertTrue(bytesPerEvent <= 8.0, "recording must update fixed fields without per-event allocation");
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
        assertTrue(bytesPerPoll <= 16.0,
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
        double bytesPerPoll = (double) (bean.getThreadAllocatedBytes(threadId) - before) / ITERATIONS;
        System.out.printf("Complete poll diagnostics including state: bytes_per_poll=%.2f%n", bytesPerPoll);
        assertTrue(bytesPerPoll <= 2048, "each poll must retain only bounded primitive diagnostic state");
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
            double bytesPerSkip = (double)
                    (bean.getThreadAllocatedBytes(threadId) - before) / ITERATIONS;
            System.out.printf("Worker poll busy lease: bytes_per_skip=%.2f%n", bytesPerSkip);
            assertTrue(bytesPerSkip <= 8.0,
                    "busy skips must aggregate without per-tick strings, maps or snapshots");

            var snapshot = worker.trace().snapshot();
            assertEquals(WARMUP + ITERATIONS, snapshot.statusPoll().skippedBusy());
            assertEquals("CALLBACK_READY", snapshot.statusPoll().stage());
        } finally {
            held.close();
        }
    }

    @Test
    void failureEpisodesKeepFixedPollStateWithoutEventHistory() {
        WorkerStatus worker = worker();
        WorkerStatus.PollLease held = worker.tryBeginStatusPoll(1L);
        try {
            held.pollTrace().record(Stage.RPC_COMPLETED, new IllegalStateException("worker unavailable"));
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
        pollTrace.record(Stage.CONVERSION_STARTED);
        pollTrace.record(Stage.CALLBACK_READY);
        pollTrace.record(Stage.CALLBACK_STARTED);
        pollTrace.record(Stage.STATUS_LOCK_ATTEMPT);
        pollTrace.record(Stage.STATUS_LOCK_ACQUIRED);
        pollTrace.record(Stage.STATUS_HEALTH_UPDATED);
        pollTrace.record(Stage.STATUS_COMMITTED);
        pollTrace.record(Stage.STATUS_APPLIED);
        pollTrace.record(Stage.STATUS_LOCK_RELEASED);
        pollTrace.record(Stage.CALLBACK_ENDED);
    }

    private static void recordSyncPhases(EndpointTrace.Poll trace) {
        trace.record(Stage.RUNNER_SUBMITTED);
        trace.record(Stage.RUNNER_STARTED);
        trace.record(Stage.SYNC_DISCOVERY);
        trace.record(Stage.SYNC_RETIREMENT);
        trace.record(Stage.SYNC_DISPATCH);
        trace.record(Stage.SYNC_REPORT);
        trace.record(Stage.RUNNER_ENDED);
    }

    private static WorkerStatus worker() {
        return WorkerStatus.createDiscovered(RoleType.DECODE, "test", "127.0.0.1",
                8080, 8081, "test");
    }

    private static com.sun.management.ThreadMXBean allocationBean() {
        var bean = (com.sun.management.ThreadMXBean) ManagementFactory.getThreadMXBean();
        assertTrue(bean.isThreadAllocatedMemorySupported());
        bean.setThreadAllocatedMemoryEnabled(true);
        return bean;
    }
}
