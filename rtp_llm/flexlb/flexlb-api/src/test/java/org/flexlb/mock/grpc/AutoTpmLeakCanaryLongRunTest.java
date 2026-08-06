package org.flexlb.mock.grpc;

import org.flexlb.autotpm.CancelReasonMapper;
import org.flexlb.autotpm.PriorityPressureController;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.scheduler.InflightItem;
import org.flexlb.balance.scheduler.Router;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.enums.TaskPhase;
import org.flexlb.mock.FlexLBMockTestBase;
import org.flexlb.mock.MockWorkerBehavior;
import org.flexlb.mock.StabilityMonitor;
import org.flexlb.service.monitor.FlexlbMetricHelper;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Stage 4 leak-canary long-run: a six-phase mixed workload — success,
 * enqueue failure, mid-flight cancel, queue yield (8400), running-decode
 * preemption (4290), and queue-deadline timeout (4511) — with all Auto-TPM
 * switches ON, followed by the full canary gate:
 *
 * <ul>
 *   <li>{@link StabilityMonitor} proves every accounting layer (inflight
 *       store actives, prefill/decode endpoint reservations and engine
 *       tasks, every tracked future) drains back to zero</li>
 *   <li>the tombstone population equals exactly the number of submitted
 *       requests — every terminal item is retained until TTL, nothing
 *       leaked, nothing double-freed</li>
 * </ul>
 *
 * <p>Backpressure is switched live between phases: unlimited for the
 * high-volume phases, single-slot for the yield / timeout phases (both
 * config keys are re-read by the batcher on every decide cycle).
 */
class AutoTpmLeakCanaryLongRunTest extends FlexLBMockTestBase {

    private static final long RELAXED_DEADLINE_MS = 60_000L;
    private static final int SUCCESS_COUNT = 20;
    private static final int FAILURE_COUNT = 10;
    private static final int CANCEL_COUNT = 6;
    private static final int YIELD_LOW_COUNT = 6;

    /** Request IDs whose first route attempt fails with NO_AVAILABLE_WORKER. */
    private final Set<Long> capacityExhaustedOnce = ConcurrentHashMap.newKeySet();

    @Override
    protected MockWorkerBehavior createDecodeBehavior() {
        return MockWorkerBehavior.builder()
                .cancelFoundRunning(true)
                .build();
    }

    @Override
    protected FlexlbConfig createConfig() {
        FlexlbConfig cfg = super.createConfig();
        cfg.setAutoTpmEnabled(true);
        cfg.setAutoTpmQueueYieldEnabled(true);
        cfg.setAutoTpmDecodeRunningPreemptEnabled(true);
        cfg.setAutoTpmPreemptRateLimitPerMin(10);
        cfg.setAutoTpmEndpointPreemptQpsLimit(0);
        cfg.setAutoTpmCommitWaitReleaseTimeoutMs(300L);
        cfg.setAutoTpmPreemptCriticalSectionMs(0);
        cfg.setFlexlbBatchEnqueueDeadlineMs(RELAXED_DEADLINE_MS);
        cfg.setFlexlbBatchFixedMaxInflightBatches(0);
        return cfg;
    }

    @Override
    protected Router createRouter() {
        Router capacityRouter = mock(Router.class);
        when(capacityRouter.route(any(BalanceContext.class))).thenAnswer(inv -> {
            BalanceContext ctx = inv.getArgument(0);
            if (capacityExhaustedOnce.remove(ctx.getRequestId())) {
                return Response.error(StrategyErrorType.NO_AVAILABLE_WORKER);
            }
            return routeToMockWorkers(ctx.getRequestId());
        });
        return capacityRouter;
    }

    @BeforeEach
    void wirePressureController() {
        scheduler.setPressureController(new PriorityPressureController(
                configService, endpointRegistry, grpcClient, inflightStore,
                scheduler.priorityRegistry(), mock(FlexlbMetricHelper.class)));
    }

    @Test
    void mixedLongRun_allLayersDrainToZero_tombstoneCountExact() throws Exception {
        StabilityMonitor monitor = new StabilityMonitor(inflightStore)
                .watchPrefill(getPrefillEndpoint())
                .watchDecode(getDecodeEndpoint())
                .pump(this::simulatePrefillFinishedReport);
        int submitted = 0;

        // ---- Phase 1: successful requests ----
        for (int i = 0; i < SUCCESS_COUNT; i++) {
            Response response = monitor.track(submitWithPriority(8100 + i, 50))
                    .get(5, TimeUnit.SECONDS);
            assertTrue(response.isSuccess(), "phase-1 request should succeed");
            submitted++;
        }

        // ---- Phase 2: enqueue failures ----
        mockPrefillWorker.setBehavior(MockWorkerBehavior.builder()
                .failOnEnqueue(true)
                .enqueueErrorMessage("mock engine overloaded")
                .enqueueErrorCode(13)
                .build());
        for (int i = 0; i < FAILURE_COUNT; i++) {
            Response response = monitor.track(submitWithPriority(8200 + i, 50))
                    .get(5, TimeUnit.SECONDS);
            assertFalse(response.isSuccess(), "phase-2 request should fail");
            submitted++;
        }

        // ---- Phase 3: mid-flight cancels while the enqueue RPC is delayed ----
        mockPrefillWorker.setBehavior(MockWorkerBehavior.builder()
                .enqueueDelayMs(800L)
                .build());
        List<CompletableFuture<Response>> cancelled = new ArrayList<>();
        for (int i = 0; i < CANCEL_COUNT; i++) {
            long requestId = 8300 + i;
            cancelled.add(monitor.track(submitWithPriority(requestId, 50)));
            submitted++;
            awaitTracked(String.valueOf(requestId), 3_000);
            InflightItem item = inflightStore.get(String.valueOf(requestId));
            assertTrue(item.cancel(), "mid-flight cancel should win the CAS");
            item.fireOnCancel();
            grpcClient.cancelAsync(
                    prefillIp, prefillGrpcPort,
                    EngineRpcService.CancelRequestPB.newBuilder().setRequestId(requestId).build(),
                    2_000L).get(3, TimeUnit.SECONDS);
        }
        for (CompletableFuture<Response> future : cancelled) {
            assertTrue(future.isDone(), "cancelled future must be settled");
        }
        mockPrefillWorker.setBehavior(MockWorkerBehavior.builder().build());
        monitor.assertQuiescent(6_000);

        // ---- Phase 4: queue yields — parked lows evicted with 8400 ----
        config.setFlexlbBatchFixedMaxInflightBatches(1);
        CompletableFuture<Response> yieldPrimer = monitor.track(submitWithPriority(8400, 70));
        submitted++;
        awaitTrue(() -> yieldPrimer.isDone(), 3_000, "yield primer must dispatch");
        long lowEnqueuedAt = System.currentTimeMillis();
        List<CompletableFuture<Response>> lows = new ArrayList<>();
        for (int i = 0; i < YIELD_LOW_COUNT; i++) {
            lows.add(monitor.track(submitWithPriority(8410 + i, 30)));
            submitted++;
        }
        Thread.sleep(500);
        long highEnqueuedAt = System.currentTimeMillis();
        CompletableFuture<Response> yieldHigh = monitor.track(submitWithPriority(8420, 70));
        submitted++;
        Thread.sleep(500);
        long now = System.currentTimeMillis();
        long lowElapsed = now - lowEnqueuedAt;
        long highElapsed = now - highEnqueuedAt;
        assertTrue(lowElapsed - highElapsed >= 200, "timing guard: separation collapsed");
        config.setFlexlbBatchEnqueueDeadlineMs((lowElapsed + highElapsed) / 2);
        for (CompletableFuture<Response> low : lows) {
            Response lowResponse = low.get(3, TimeUnit.SECONDS);
            assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(), lowResponse.getCode(),
                    "yield victim must be 8400");
            assertTrue(lowResponse.getErrorMessage().contains("auto_tpm: yielded for priority=70"));
        }
        config.setFlexlbBatchEnqueueDeadlineMs(RELAXED_DEADLINE_MS);
        assertTrue(pumpUntilDone(yieldHigh, 10_000).isSuccess(), "yield-phase high must succeed");
        config.setFlexlbBatchFixedMaxInflightBatches(0);
        assertEquals(0, mockPrefillWorker.getCancelCount() + mockDecodeWorker.getCancelCount()
                        - CANCEL_COUNT,
                "queue yield must never add an Engine cancel");
        monitor.assertQuiescent(6_000);

        // ---- Phase 5: running-decode preemption — victim 4290 ----
        mockPrefillWorker.setBehavior(MockWorkerBehavior.builder()
                .enqueueDelayMs(1_500L)
                .build());
        int enqueuesBeforePhase5 = mockPrefillWorker.getEnqueueCount();
        CompletableFuture<Response> victim = monitor.track(submitWithPriority(8501, 30));
        submitted++;
        awaitTrue(() -> mockPrefillWorker.getEnqueueCount() == enqueuesBeforePhase5 + 1, 3_000,
                "victim enqueue should reach the prefill worker");
        reportDecodeRunning(8501);
        Thread feeder = scheduleVictimRelease(8501);
        capacityExhaustedOnce.add(8701L);
        CompletableFuture<Response> preemptHigh = monitor.track(submitWithPriority(8701, 70));
        submitted++;
        Response victimResponse = victim.get(2, TimeUnit.SECONDS);
        assertEquals(StrategyErrorType.AUTO_TPM_PREEMPTED.getErrorCode(), victimResponse.getCode(),
                "preempt victim must be 4290");
        assertTrue(preemptHigh.get(6, TimeUnit.SECONDS).isSuccess(), "preempting high must succeed");
        assertEquals(1, mockDecodeWorker.getCancelCount(), "exactly one preempt Cancel on decode");
        feeder.join(3_000);
        mockPrefillWorker.setBehavior(MockWorkerBehavior.builder().build());
        monitor.assertQuiescent(8_000);

        // ---- Phase 6: queue-deadline timeout at the head — 4511 ----
        config.setFlexlbBatchFixedMaxInflightBatches(1);
        CompletableFuture<Response> timeoutPrimer = monitor.track(submitWithPriority(8600, 70));
        submitted++;
        awaitTrue(() -> timeoutPrimer.isDone(), 3_000, "timeout primer must dispatch");
        config.setFlexlbBatchEnqueueDeadlineMs(300L);
        CompletableFuture<Response> timedOut = monitor.track(submitWithPriority(8601, 50));
        submitted++;
        Response timedOutResponse = timedOut.get(5, TimeUnit.SECONDS);
        assertEquals(StrategyErrorType.BATCH_SLO_EXPIRED.getErrorCode(), timedOutResponse.getCode(),
                "plain head-of-queue deadline expiry must stay 4511 (not a yield)");
        config.setFlexlbBatchEnqueueDeadlineMs(RELAXED_DEADLINE_MS);
        config.setFlexlbBatchFixedMaxInflightBatches(0);

        // ---- Canary gate: zero leak on every layer, exact tombstones ----
        monitor.assertQuiescent(8_000);
        assertEquals(0, inflightStore.activeCount());
        assertEquals(submitted, inflightStore.totalSize(),
                "every submitted request must remain exactly once as a tombstone until TTL");
    }

    // ==================== helpers ====================

    private CompletableFuture<Response> submitWithPriority(long requestId, int priority) {
        BalanceContext ctx = createBalanceContext(requestId);
        ctx.setPriority(priority);
        return scheduler.submit(ctx);
    }

    private void reportDecodeRunning(long requestId) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(requestId);
        task.setPhase(TaskPhase.RUNNING);
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(RoleType.DECODE);
        response.setRunningTaskInfo(Map.of(String.valueOf(requestId), task));
        DecodeEndpoint decodeEp = getDecodeEndpoint();
        decodeEp.onWorkerStatusUpdate(decodeEp.getStatus(), response);
    }

    private void reportDecodeFinishedPreempted(long requestId) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(requestId);
        task.setErrorCode(CancelReasonMapper.ENGINE_ERROR_CODE_CANCELLED);
        task.setCancelReason(CancelReasonMapper.CANCEL_REASON_PRIORITY_PREEMPTED);
        task.setErrorMessage("cancelled");
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(RoleType.DECODE);
        response.setFinishedTaskInfo(Map.of(String.valueOf(requestId), task));
        DecodeEndpoint decodeEp = getDecodeEndpoint();
        decodeEp.onWorkerStatusUpdate(decodeEp.getStatus(), response);
    }

    /** Feed the preempted-finished report once the Cancel RPC lands. */
    private Thread scheduleVictimRelease(long victimId) {
        Thread feeder = new Thread(() -> {
            try {
                long deadline = System.currentTimeMillis() + 5_000;
                while (System.currentTimeMillis() < deadline
                        && mockDecodeWorker.getRpcService().getCancelledRequests().stream()
                                .noneMatch(r -> r.getRequestId() == victimId)) {
                    Thread.sleep(10);
                }
                reportDecodeFinishedPreempted(victimId);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }, "victim-release-feeder-" + victimId);
        feeder.setDaemon(true);
        feeder.start();
        return feeder;
    }

    /** Same shape as the base fixed route (private there): both mock workers. */
    private Response routeToMockWorkers(long requestId) {
        Response response = new Response();
        response.setSuccess(true);
        response.setServerStatus(List.of(
                serverStatus(RoleType.PREFILL, prefillIp, prefillHttpPort, prefillGrpcPort, requestId),
                serverStatus(RoleType.DECODE, decodeIp, decodeHttpPort, decodeGrpcPort, requestId)));
        return response;
    }

    private static ServerStatus serverStatus(RoleType role, String ip, int httpPort,
                                             int grpcPort, long requestId) {
        ServerStatus status = new ServerStatus();
        status.setSuccess(true);
        status.setRole(role);
        status.setServerIp(ip);
        status.setHttpPort(httpPort);
        status.setGrpcPort(grpcPort);
        status.setDpRank(0);
        status.setGroup("test-group");
        status.setRequestId(requestId);
        return status;
    }

    private Response pumpUntilDone(CompletableFuture<Response> future, long timeoutMs) throws Exception {
        long deadline = System.currentTimeMillis() + timeoutMs;
        while (!future.isDone() && System.currentTimeMillis() < deadline) {
            simulatePrefillFinishedReport();
            Thread.sleep(20);
        }
        return future.get(1, TimeUnit.SECONDS);
    }

    private void awaitTracked(String requestId, long timeoutMs) throws InterruptedException {
        long deadline = System.currentTimeMillis() + timeoutMs;
        while (System.currentTimeMillis() < deadline) {
            if (inflightStore.get(requestId) != null) {
                return;
            }
            Thread.sleep(10);
        }
        assertNotNull(inflightStore.get(requestId), "request should be tracked in the store");
    }

    private static void awaitTrue(java.util.function.BooleanSupplier condition,
                                  long timeoutMs, String message) throws InterruptedException {
        long deadline = System.currentTimeMillis() + timeoutMs;
        while (System.currentTimeMillis() < deadline) {
            if (condition.getAsBoolean()) {
                return;
            }
            Thread.sleep(20);
        }
        assertTrue(condition.getAsBoolean(), message);
    }
}
