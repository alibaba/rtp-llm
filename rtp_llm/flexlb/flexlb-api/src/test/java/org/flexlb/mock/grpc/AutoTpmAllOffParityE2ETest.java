package org.flexlb.mock.grpc;

import org.flexlb.autotpm.PriorityPressureController;
import org.flexlb.balance.endpoint.DecodeEndpoint;
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
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Stage 4 switch-matrix E2E spot check: with every AUTO_TPM switch at its
 * DEFAULT (off) value the same mixed-priority traffic behaves exactly like
 * the pre-Auto-TPM baseline (iron rule 1 — default-off parity), even though
 * a real {@link PriorityPressureController} is wired in:
 *
 * <ul>
 *   <li>dispatch is plain FIFO — priorities are carried but never re-order
 *       the queue (no tiering, no yield)</li>
 *   <li>a capacity-starved high-priority request keeps the plain 8400 route
 *       failure: no preemption, no Engine Cancel, no 4290 anywhere</li>
 *   <li>queue-deadline expiry stays the baseline 4511 head drop, never the
 *       yield 8400, even with a higher-priority request queued behind</li>
 *   <li>all accounting layers drain to zero</li>
 * </ul>
 */
class AutoTpmAllOffParityE2ETest extends FlexLBMockTestBase {

    private static final long RELAXED_DEADLINE_MS = 60_000L;

    /** Request IDs whose first route attempt fails with NO_AVAILABLE_WORKER. */
    private final Set<Long> capacityExhaustedOnce = ConcurrentHashMap.newKeySet();

    @Override
    protected FlexlbConfig createConfig() {
        // AUTO_TPM keys stay at their defaults — that IS the scenario.
        FlexlbConfig cfg = super.createConfig();
        cfg.setFlexlbBatchFixedMaxInflightBatches(1);
        cfg.setFlexlbBatchEnqueueDeadlineMs(RELAXED_DEADLINE_MS);
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
    void wirePressureControllerAndVerifyDefaults() {
        // The controller is wired exactly as in the enabled scenarios — the
        // master switch alone must keep it inert.
        scheduler.setPressureController(new PriorityPressureController(
                configService, endpointRegistry, grpcClient, inflightStore,
                scheduler.priorityRegistry(), mock(FlexlbMetricHelper.class)));
        assertFalse(config.isAutoTpmEnabled(), "parity precondition: master switch off");
        assertFalse(config.isAutoTpmQueueYieldEnabled(), "parity precondition: yield off");
        assertFalse(config.isAutoTpmDecodeRunningPreemptEnabled(), "parity precondition: preempt off");
    }

    // ---- gate: FIFO dispatch, priorities never re-order the queue ----

    @Test
    void mixedPriorities_dispatchStaysFifo_noYieldNoPreempt() throws Exception {
        StabilityMonitor monitor = new StabilityMonitor(inflightStore)
                .watchPrefill(getPrefillEndpoint())
                .watchDecode(getDecodeEndpoint())
                .pump(this::simulatePrefillFinishedReport);

        // Primer parks the queue on the single backpressure slot.
        CompletableFuture<Response> primer = monitor.track(submitWithPriority(7100, 70));
        awaitTrue(() -> mockPrefillWorker.getEnqueueCount() == 1, 3_000,
                "primer must reach the engine before the scenario starts");

        // A P30 arrives first, then P70s — baseline keeps arrival order.
        CompletableFuture<Response> low = monitor.track(submitWithPriority(7301, 30));
        Thread.sleep(200);
        List<CompletableFuture<Response>> highs = new ArrayList<>();
        long[] highIds = {7701, 7702};
        for (long id : highIds) {
            highs.add(monitor.track(submitWithPriority(id, 70)));
        }

        // Drain one slot per pump; every request must succeed.
        long deadline = System.currentTimeMillis() + 15_000;
        while (System.currentTimeMillis() < deadline
                && !(low.isDone() && highs.stream().allMatch(CompletableFuture::isDone))) {
            simulatePrefillFinishedReport();
            Thread.sleep(20);
        }
        assertTrue(primer.get(1, TimeUnit.SECONDS).isSuccess());
        assertTrue(low.get(1, TimeUnit.SECONDS).isSuccess(),
                "baseline: earlier P30 is served, never yielded or rejected");
        for (CompletableFuture<Response> high : highs) {
            assertTrue(high.get(1, TimeUnit.SECONDS).isSuccess());
        }

        // Baseline parity: strict arrival (FIFO) order — the earlier P30 is
        // dispatched BEFORE the later P70s, priority has no effect.
        List<Long> arrivalOrder = enqueueArrivalOrder();
        int lowIndex = arrivalOrder.indexOf(7301L);
        for (long id : highIds) {
            assertTrue(lowIndex < arrivalOrder.indexOf(id),
                    "all-off parity: dispatch must stay FIFO, got: " + arrivalOrder);
        }

        // No yield, no preempt: zero Engine Cancels in the whole run.
        assertEquals(0, mockPrefillWorker.getCancelCount());
        assertEquals(0, mockDecodeWorker.getCancelCount());

        monitor.assertQuiescent(5_000);
        assertEquals(0, inflightStore.activeCount());
    }

    // ---- gate: capacity exhaustion keeps plain 8400, no preemption ----

    @Test
    void capacityExhausted_highKeepsPlain8400_noPreemptNoCancel() throws Exception {
        StabilityMonitor monitor = new StabilityMonitor(inflightStore)
                .watchPrefill(getPrefillEndpoint())
                .watchDecode(getDecodeEndpoint())
                .pump(this::simulatePrefillFinishedReport);

        // An eligible-looking P30 victim is RUNNING on the decode engine.
        CompletableFuture<Response> wouldBeVictim = monitor.track(submitWithPriority(7302, 30));
        awaitTrue(() -> mockPrefillWorker.getEnqueueCount() == 1, 3_000,
                "victim-lookalike enqueue should reach the prefill worker");
        assertTrue(wouldBeVictim.get(3, TimeUnit.SECONDS).isSuccess());
        reportDecodeRunning(7302);

        // Capacity exhausted for an incoming P70 → with the master switch
        // off the plain route failure passes straight through.
        capacityExhaustedOnce.add(7703L);
        CompletableFuture<Response> high = monitor.track(submitWithPriority(7703, 70));
        Response highResponse = high.get(2, TimeUnit.SECONDS);
        assertFalse(highResponse.isSuccess());
        assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(), highResponse.getCode(),
                "all-off parity: plain 8400 passthrough, never a preempt attempt");

        // Absolutely no Cancel and no 4290 attribution anywhere.
        assertEquals(0, mockDecodeWorker.getCancelCount(),
                "all-off parity: no preemption Cancel may ever be issued");
        assertEquals(0, mockPrefillWorker.getCancelCount());

        reportDecodeFinishedNormal(7302);
        monitor.assertQuiescent(5_000);
        assertEquals(0, inflightStore.activeCount());
    }

    // ---- gate: deadline expiry stays baseline 4511, never the yield 8400 ----

    @Test
    void queueDeadlineExpiry_staysBaseline4511_notYield8400() throws Exception {
        StabilityMonitor monitor = new StabilityMonitor(inflightStore)
                .watchPrefill(getPrefillEndpoint())
                .watchDecode(getDecodeEndpoint())
                .pump(this::simulatePrefillFinishedReport);

        CompletableFuture<Response> primer = monitor.track(submitWithPriority(7101, 70));
        awaitTrue(() -> mockPrefillWorker.getEnqueueCount() == 1, 3_000,
                "primer must reach the engine before the scenario starts");

        // P30 queues first, a P70 queues behind it while parked.
        long lowEnqueuedAt = System.currentTimeMillis();
        CompletableFuture<Response> low = monitor.track(submitWithPriority(7303, 30));
        Thread.sleep(500);
        long highEnqueuedAt = System.currentTimeMillis();
        CompletableFuture<Response> high = monitor.track(submitWithPriority(7704, 70));
        Thread.sleep(500);

        // Only the P30 head exceeds the mid-point deadline.
        long now = System.currentTimeMillis();
        long lowElapsed = now - lowEnqueuedAt;
        long highElapsed = now - highEnqueuedAt;
        assertTrue(lowElapsed - highElapsed >= 200, "timing guard: separation collapsed");
        config.setFlexlbBatchEnqueueDeadlineMs((lowElapsed + highElapsed) / 2);

        // Baseline attribution: 4511 head expiry — NOT the yield 8400.
        Response lowResponse = low.get(3, TimeUnit.SECONDS);
        assertFalse(lowResponse.isSuccess());
        assertEquals(StrategyErrorType.BATCH_SLO_EXPIRED.getErrorCode(), lowResponse.getCode(),
                "all-off parity: deadline expiry must stay the baseline 4511");
        assertTrue(lowResponse.getErrorMessage() == null
                        || !lowResponse.getErrorMessage().contains("yielded"),
                "all-off parity: no yield attribution may appear");

        // The queued P70 is unaffected and completes once released.
        config.setFlexlbBatchEnqueueDeadlineMs(RELAXED_DEADLINE_MS);
        long deadline = System.currentTimeMillis() + 10_000;
        while (System.currentTimeMillis() < deadline && !high.isDone()) {
            simulatePrefillFinishedReport();
            Thread.sleep(20);
        }
        assertTrue(primer.get(1, TimeUnit.SECONDS).isSuccess());
        assertTrue(high.get(1, TimeUnit.SECONDS).isSuccess());

        assertEquals(0, mockPrefillWorker.getCancelCount());
        assertEquals(0, mockDecodeWorker.getCancelCount());
        monitor.assertQuiescent(5_000);
        assertEquals(0, inflightStore.activeCount());
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

    private void reportDecodeFinishedNormal(long requestId) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(requestId);
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(RoleType.DECODE);
        response.setFinishedTaskInfo(Map.of(String.valueOf(requestId), task));
        DecodeEndpoint decodeEp = getDecodeEndpoint();
        decodeEp.onWorkerStatusUpdate(decodeEp.getStatus(), response);
    }

    /** Flattened requestId arrival order across all EnqueueBatch calls. */
    private List<Long> enqueueArrivalOrder() {
        List<Long> order = new ArrayList<>();
        for (EngineRpcService.EnqueueBatchRequestPB batch
                : mockPrefillWorker.getRpcService().getEnqueuedRequests()) {
            for (EngineRpcService.EnqueueBatchDpSlotPB slot : batch.getDpSlotsList()) {
                for (EngineRpcService.EnqueueBatchExternalInputPB ext : slot.getRequestsList()) {
                    order.add(ext.getInput().getRequestId());
                }
            }
        }
        return order;
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
