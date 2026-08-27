package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.grpc.stub.StreamObserver;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.lang.reflect.Method;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

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

    private static final ObjectMapper MAPPER = new ObjectMapper();
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
        MockPerformanceModel model = model(10_000.0, false);
        JavaMockEngineCluster.FastRpcService decode = newDecodeService(model, 1);

        // 1 running + 2 queued behind the concurrency gate.
        for (long rid = 1; rid <= 3; rid++) {
            assertTrue(invokeScheduleDecodeCompletion(decode, shapeOf(model, rid, 8), -1, null));
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

        awaitQuiescence(decode, 30_000);
        assertEquals(3, decode.getCompletedCount());
        assertEquals(0, decode.getActiveKvTokens(), "KV must net to zero after quiescence");
        decode.checkLeakDrain(0L);
        assertFalse(decode.isLeakDetected());
    }

    // ──────────── State 2: flag ON → queued = KV_ALLOCATED, flips back on start ────────────

    @Test
    void optInReportsQueuedAsKvAllocatedAndFlipsBackWhenRunning() throws Exception {
        MockPerformanceModel model = model(10_000.0, true);
        JavaMockEngineCluster.FastRpcService decode = newDecodeService(model, 1);

        for (long rid = 1; rid <= 3; rid++) {
            assertTrue(invokeScheduleDecodeCompletion(decode, shapeOf(model, rid, 8), -1, null));
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
        awaitQuiescence(decode, 30_000);
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
        MockPerformanceModel model = model(10_000.0, true);
        JavaMockEngineCluster.FastRpcService decode = newDecodeService(model, 1);

        assertTrue(invokeScheduleDecodeCompletion(decode, shapeOf(model, 1L, 8), -1, null));
        assertTrue(invokeScheduleDecodeCompletion(decode, shapeOf(model, 2L, 8), -1, null));
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

        awaitQuiescence(decode, 30_000);
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
        JavaMockEngineCluster.FastRpcService service = new JavaMockEngineCluster.FastRpcService(
                "decode-" + port, "127.0.0.1", "decode",
                EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE,
                port, services, scheduler, model, 100,
                new JavaMockEngineCluster.ClusterStats(), 10_000_000L, decodeMaxConcurrency);
        services.put(port, service);
        return service;
    }

    private MockPerformanceModel model(double decodeStepMs,
                                       boolean reportQueuedAsKvAllocated) throws Exception {
        Path performance = tempDir.resolve("performance-" + System.nanoTime() + ".json");
        Path master = tempDir.resolve("master-" + System.nanoTime() + ".json");
        Map<String, Object> decodeConfig = new LinkedHashMap<>();
        decodeConfig.put("scale", 1.0);
        decodeConfig.put("step_ms_by_batch", List.of(List.of(1, decodeStepMs)));
        if (reportQueuedAsKvAllocated) {
            decodeConfig.put("report_queued_as_kv_allocated", true);
        }
        MAPPER.writeValue(performance.toFile(), Map.of(
                "block_size", 1024,
                "sleep_scale", 0.1,
                "jitter_pct", 0.0,
                "prefill", Map.of("scale", 1.0),
                "decode", decodeConfig));
        MockMasterConfig.writeWithPrefillExpression(master, "10");
        return MockPerformanceModel.load(performance.toString(), master.toString());
    }

    private static MockPerformanceModel.RequestShape shapeOf(
            MockPerformanceModel model, long requestId, int inputTokens) {
        EngineRpcService.GenerateInputPB.Builder input = EngineRpcService.GenerateInputPB.newBuilder()
                .setRequestId(requestId)
                .setGenerateConfig(EngineRpcService.GenerateConfigPB.newBuilder()
                        .setMaxNewTokens(1)
                        .build());
        for (int token = 0; token < inputTokens; token++) {
            input.addTokenIds(token);
        }
        return model.shape(input.build(), new MockLruBlockCache(100));
    }

    private void awaitQuiescence(JavaMockEngineCluster.FastRpcService service, long timeoutMs)
            throws Exception {
        long deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(timeoutMs);
        while (System.nanoTime() < deadline) {
            if (service.getInflightCount() == 0 && service.getRunningCount() == 0) {
                return;
            }
            Thread.sleep(10);
        }
        fail("engine did not quiesce: inflight=" + service.getInflightCount()
                + " running=" + service.getRunningCount());
    }

    private static EngineRpcService.WorkerStatusPB workerStatus(
            JavaMockEngineCluster.FastRpcService service, long sinceVersion) {
        AtomicReference<EngineRpcService.WorkerStatusPB> response = new AtomicReference<>();
        AtomicReference<Throwable> error = new AtomicReference<>();
        CountDownLatch latch = new CountDownLatch(1);
        service.getWorkerStatus(
                EngineRpcService.StatusVersionPB.newBuilder()
                        .setLatestFinishedVersion(sinceVersion)
                        .build(),
                new StreamObserver<>() {
                    @Override
                    public void onNext(EngineRpcService.WorkerStatusPB value) {
                        response.set(value);
                    }

                    @Override
                    public void onError(Throwable throwable) {
                        error.set(throwable);
                        latch.countDown();
                    }

                    @Override
                    public void onCompleted() {
                        latch.countDown();
                    }
                });
        try {
            if (!latch.await(5, TimeUnit.SECONDS)) {
                fail("worker status timeout");
            }
        } catch (InterruptedException e) {
            fail("interrupted waiting for worker status");
        }
        Optional.ofNullable(error.get()).ifPresent(t -> fail(String.valueOf(t)));
        assertNotNull(response.get());
        return response.get();
    }

    private static boolean invokeScheduleDecodeCompletion(
            JavaMockEngineCluster.FastRpcService service,
            MockPerformanceModel.RequestShape shape,
            long batchId,
            LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB> responseQueue)
            throws Exception {
        Method method = JavaMockEngineCluster.FastRpcService.class.getDeclaredMethod(
                "scheduleDecodeCompletion",
                MockPerformanceModel.RequestShape.class,
                long.class,
                LinkedBlockingQueue.class);
        method.setAccessible(true);
        return (Boolean) method.invoke(service, shape, batchId, responseQueue);
    }
}
