package org.flexlb.mockengine;

import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import static org.flexlb.mockengine.MockEngineTestSupport.awaitDecodeQuiescence;
import static org.flexlb.mockengine.MockEngineTestSupport.decodeModel;
import static org.flexlb.mockengine.MockEngineTestSupport.requestShape;
import static org.flexlb.mockengine.MockEngineTestSupport.scheduleDecodeCompletion;
import static org.flexlb.mockengine.MockEngineTestSupport.workerStatus;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Two-state tests for the opt-in accepted-layer window (performance JSON
 * {@code decode.report_queued_as_kv_allocated}, C1-①):
 *
 * <ol>
 *   <li>default OFF — queued decode requests keep today's reporting
 *       (TASK_PHASE_RUNNING entries in running_task_info, runningQueryLen
 *       counts only truly running requests as before): zero behavior
 *       change,</li>
 *   <li>ON — queued (admitted, not yet running) requests surface as
 *       TASK_PHASE_KV_ALLOCATED so the scheduler's accepted layer sees them;
 *       once drained into a slot the phase flips back to RUNNING; cancel of a
 *       queued request reports the KV_ALLOCATED phase (the exact contract the
 *       8429 accepted-eviction path consumes).</li>
 * </ol>
 */
class KvAllocatedReportOptInTest {

    private static final int BASE_PORT = 63400;

    @TempDir
    Path tempDir;

    private ScheduledExecutorService scheduler;
    private Map<Integer, JavaMockEngineCluster.FastRpcService> services;
    private int nextPortOffset;

    @BeforeEach
    void setUp() {
        scheduler = Executors.newScheduledThreadPool(4, runnable -> {
            Thread thread = new Thread(runnable, "kv-allocated-optin-scheduler");
            thread.setDaemon(true);
            return thread;
        });
        services = new ConcurrentHashMap<>();
        nextPortOffset = 0;
    }

    @AfterEach
    void tearDown() throws InterruptedException {
        for (JavaMockEngineCluster.FastRpcService service : services.values()) {
            service.shutdown();
        }
        scheduler.shutdownNow();
        scheduler.awaitTermination(3, TimeUnit.SECONDS);
    }

    // ──────────── State 1: flag absent → today's reporting, zero change ────────────

    @Test
    void defaultOffKeepsQueuedTasksReportedAsRunning() throws Exception {
        MockPerformanceModel model = decodeModel(tempDir, 10_000.0, 3, false);
        JavaMockEngineCluster.FastRpcService decode = newDecodeService(model, 1);

        // 1 running + 2 queued behind the concurrency gate.
        for (long rid = 1; rid <= 3; rid++) {
            assertTrue(scheduleDecodeCompletion(decode, requestShape(model, rid, 8), -1, null));
        }

        EngineRpcService.WorkerStatusPB status = workerStatus(decode, 0);
        assertEquals(2, status.getWaitingQueryLen(), "two requests queued");
        for (EngineRpcService.TaskInfoPB task : status.getRunningTaskInfoList()) {
            assertEquals(EngineRpcService.TaskPhase.TASK_PHASE_RUNNING, task.getPhase(),
                    "flag OFF: queued decode tasks must keep today's "
                            + "TASK_PHASE_RUNNING reporting (zero behavior change), "
                            + "request " + task.getRequestId());
        }
        assertEquals(3, status.getRunningTaskInfoCount(),
                "all admitted requests keep their runningTasks claim");
        // KV accounting, flag OFF (P2-5 byte-for-byte guard): only the one
        // truly RUNNING request holds KV — queued requests stay uncounted
        // until run start, exactly today's behavior.
        assertEquals(8, decode.getActiveKvTokens(),
                "default OFF: only the running request's input tokens are counted");

        awaitDecodeQuiescence(decode, 30_000);
        assertEquals(3, decode.getCompletedCount());
        assertEquals(0, decode.getActiveKvTokens(), "KV must net to zero after quiescence");
        decode.checkLeakDrain(0L);
        assertFalse(decode.isLeakDetected());
    }

    // ──────────── State 2: flag ON → queued = KV_ALLOCATED, flips back on start ────────────

    @Test
    void optInReportsQueuedAsKvAllocatedAndFlipsBackWhenRunning() throws Exception {
        MockPerformanceModel model = decodeModel(tempDir, 10_000.0, 3, true);
        JavaMockEngineCluster.FastRpcService decode = newDecodeService(model, 1);

        for (long rid = 1; rid <= 3; rid++) {
            assertTrue(scheduleDecodeCompletion(decode, requestShape(model, rid, 8), -1, null));
        }

        EngineRpcService.WorkerStatusPB status = workerStatus(decode, 0);
        assertEquals(2, status.getWaitingQueryLen());
        assertEquals(1, phaseCount(status, EngineRpcService.TaskPhase.TASK_PHASE_RUNNING),
                "exactly the one truly running request reports RUNNING");
        assertEquals(2, phaseCount(status, EngineRpcService.TaskPhase.TASK_PHASE_KV_ALLOCATED),
                "flag ON: both queued requests must surface as KV_ALLOCATED "
                        + "(accepted layer window)");
        assertEquals(1, status.getRunningQueryLen(),
                "runningQueryLen must not count KV_ALLOCATED (queued) tasks");
        // KV accounting, flag ON (P2-5): KV_ALLOCATED means "KV reserved", so
        // the two queued requests hold their reservation from enqueue — all
        // three requests (1 running + 2 queued) are counted.
        assertEquals(24, decode.getActiveKvTokens(),
                "opt-in: queued requests must hold their KV reservation from enqueue");

        // As slots free, queued requests drain and their phase flips to
        // RUNNING; eventually everything completes with no leak.
        awaitDecodeQuiescence(decode, 30_000);
        assertEquals(3, decode.getCompletedCount(),
                "queued requests must still drain and complete normally");
        assertEquals(0, decode.getActiveKvTokens(),
                "opt-in: enqueue-time KV must not be double-counted at run start "
                        + "(net zero after quiescence)");
        decode.checkLeakDrain(0L);
        assertFalse(decode.isLeakDetected());
    }

    @Test
    void optInCancelOfQueuedRequestReportsKvAllocatedPhase() throws Exception {
        MockPerformanceModel model = decodeModel(tempDir, 10_000.0, 3, true);
        JavaMockEngineCluster.FastRpcService decode = newDecodeService(model, 1);

        assertTrue(scheduleDecodeCompletion(decode, requestShape(model, 1L, 8), -1, null));
        assertTrue(scheduleDecodeCompletion(decode, requestShape(model, 2L, 8), -1, null));
        assertEquals(16, decode.getActiveKvTokens(),
                "opt-in: running + queued requests both hold KV before the cancel");

        // Cancel the QUEUED request (rid=2) through Decode's ordinary internal
        // stream-cancellation path. Priority Cancel RPC itself is Prefill-only.
        EngineRpcService.TaskPhase phase = decode.cancel(2L);
        assertEquals(EngineRpcService.TaskPhase.TASK_PHASE_KV_ALLOCATED, phase,
                "cancel of a queued request must report the KV_ALLOCATED phase");
        // Cancel of a queued opt-in request must release its enqueue-time KV
        // reservation immediately (P2-5), leaving only the running request's.
        assertEquals(8, decode.getActiveKvTokens(),
                "cancelling the queued request must release its KV reservation");

        // Iron rule 4: CANCELLED terminal surfaces in the next WorkerStatus.
        EngineRpcService.WorkerStatusPB status = workerStatus(decode, 0);
        boolean cancelledReported = status.getFinishedTaskListList().stream()
                .anyMatch(task -> task.getRequestId() == 2L
                        && task.getErrorInfo().getErrorCode()
                        == EngineRpcService.ErrorCodePB.CANCELLED.getNumber());
        assertTrue(cancelledReported,
                "CANCELLED completion for request 2 must appear in WorkerStatus");

        awaitDecodeQuiescence(decode, 30_000);
        assertEquals(1, decode.getCompletedCount(), "only the running request completes");
        assertEquals(0, decode.getActiveKvTokens(), "KV must net to zero after quiescence");
        decode.checkLeakDrain(0L);
        assertFalse(decode.isLeakDetected());
    }

    // ──────────── Helpers ────────────

    private static int phaseCount(EngineRpcService.WorkerStatusPB status,
                                  EngineRpcService.TaskPhase phase) {
        return (int) status.getRunningTaskInfoList().stream()
                .filter(task -> task.getPhase() == phase)
                .count();
    }

    private JavaMockEngineCluster.FastRpcService newDecodeService(
            MockPerformanceModel model, int decodeMaxConcurrency) {
        int port = BASE_PORT + nextPortOffset++;
        return MockEngineTestSupport.decodeService(
                model, port, services, scheduler, decodeMaxConcurrency);
    }

}
