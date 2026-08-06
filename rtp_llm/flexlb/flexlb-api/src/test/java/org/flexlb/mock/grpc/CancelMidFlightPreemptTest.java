package org.flexlb.mock.grpc;

import org.flexlb.autotpm.CancelReasonMapper;
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
import org.flexlb.mock.MockWorkerBehavior;
import org.flexlb.mock.StabilityMonitor;
import org.flexlb.service.monitor.FlexlbMetricHelper;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

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
 * Stage 3 end-to-end preemption over the Stage 1 mock-worker harness: a real
 * {@link PriorityPressureController} wired into the real BatchScheduler, with
 * the Cancel RPC travelling through the real {@code EngineGrpcClient} to the
 * mock decode worker.
 *
 * <ul>
 *   <li>immediate release — cancel confirmed (found=true, structured
 *       PRIORITY_PREEMPTED reason recorded by the worker), release observed
 *       within the bounded wait, high-priority request re-routed and
 *       dispatched, victim settled as AUTO_TPM_PREEMPTED (4290)</li>
 *   <li>release slower than the wait budget — never optimistic: the incoming
 *       request keeps the original route failure (8400), the victim is closed
 *       out as 4290 only when the WorkerStatus finished report lands</li>
 *   <li>after both scenarios every accounting layer drains to zero</li>
 * </ul>
 *
 * <p>Timing: the slow prefill enqueue (1.5s) keeps the victim's future
 * pending so it stays in the PriorityRegistry while being preempted; the
 * scenario-two release delay (700ms) sits between the wait budget (300ms)
 * and the enqueue ACK so the close-out settles the victim first.
 */
class CancelMidFlightPreemptTest extends FlexLBMockTestBase {

    private static final long ENQUEUE_DELAY_MS = 1_500L;
    private static final long WAIT_RELEASE_TIMEOUT_MS = 300L;
    private static final long SLOW_RELEASE_DELAY_MS = 700L;
    private static final int VICTIM_PRIORITY = 30;
    private static final int HIGH_PRIORITY = 70;

    /** Request IDs whose first route attempt fails with NO_AVAILABLE_WORKER. */
    private final Set<Long> capacityExhaustedOnce = ConcurrentHashMap.newKeySet();

    @Override
    protected MockWorkerBehavior createPrefillBehavior() {
        // Slow enqueue keeps the victim's future pending (still registered in
        // the PriorityRegistry) while the preemption runs.
        return MockWorkerBehavior.builder()
                .enqueueDelayMs(ENQUEUE_DELAY_MS)
                .build();
    }

    @Override
    protected MockWorkerBehavior createDecodeBehavior() {
        // The decode engine still tracks the victim: Cancel answers found=true.
        return MockWorkerBehavior.builder()
                .cancelFoundRunning(true)
                .build();
    }

    @Override
    protected FlexlbConfig createConfig() {
        FlexlbConfig cfg = super.createConfig();
        cfg.setAutoTpmEnabled(true);
        cfg.setAutoTpmDecodeRunningPreemptEnabled(true);
        cfg.setAutoTpmPreemptRateLimitPerMin(10);
        cfg.setAutoTpmEndpointPreemptQpsLimit(0);
        cfg.setAutoTpmCommitWaitReleaseTimeoutMs(WAIT_RELEASE_TIMEOUT_MS);
        cfg.setAutoTpmPreemptCriticalSectionMs(0);
        return cfg;
    }

    @Override
    protected Router createRouter() {
        // First route attempt of a marked request fails with NO_AVAILABLE_WORKER
        // (capacity exhausted); the post-preempt re-route succeeds.
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

    // ---- scenario 1: immediate release → high dispatched, victim 4290 ----

    @Test
    void preempt_immediateRelease_highPriorityDispatched_victimSettled4290() throws Exception {
        StabilityMonitor monitor = new StabilityMonitor(inflightStore)
                .watchPrefill(getPrefillEndpoint())
                .watchDecode(getDecodeEndpoint())
                .pump(this::simulatePrefillFinishedReport);

        // Low-priority request reaches RUNNING on the decode engine.
        CompletableFuture<Response> victim = monitor.track(submitWithPriority(101, VICTIM_PRIORITY));
        awaitTrue(() -> mockPrefillWorker.getEnqueueCount() == 1, 3_000,
                "victim enqueue should reach the prefill worker");
        reportDecodeRunning(101);

        // Engine-side release: finished report follows the Cancel immediately.
        Thread feeder = scheduleVictimRelease(101, 0);

        // Capacity exhausted for the incoming high-priority request → preempt.
        capacityExhaustedOnce.add(201L);
        CompletableFuture<Response> high = monitor.track(submitWithPriority(201, HIGH_PRIORITY));

        // Cancel reached the decode worker with the structured preempt reason.
        List<EngineRpcService.CancelRequestPB> cancels =
                mockDecodeWorker.getRpcService().getCancelledRequests();
        assertEquals(1, cancels.size(), "exactly one Cancel should reach decode");
        assertEquals(101, cancels.get(0).getRequestId());
        assertEquals(EngineRpcService.EngineCancelReasonPB.ENGINE_CANCEL_REASON_PRIORITY_PREEMPTED,
                cancels.get(0).getReason());

        // Victim settled with structured attribution AUTO_TPM_PREEMPTED (4290).
        Response victimResponse = victim.get(2, TimeUnit.SECONDS);
        assertFalse(victimResponse.isSuccess());
        assertEquals(StrategyErrorType.AUTO_TPM_PREEMPTED.getErrorCode(), victimResponse.getCode());

        // High-priority request re-routed onto the freed capacity and dispatched.
        Response highResponse = high.get(ENQUEUE_DELAY_MS + 4_000, TimeUnit.MILLISECONDS);
        assertTrue(highResponse.isSuccess(), "high-priority request should complete after preempt");
        assertEquals(2, mockPrefillWorker.getEnqueueCount(),
                "high-priority request must be dispatched after the confirmed release");

        feeder.join(3_000);
        monitor.assertQuiescent(ENQUEUE_DELAY_MS + 5_000);
    }

    // ---- scenario 2: release slower than the wait budget → never optimistic ----

    @Test
    void preempt_releaseTimeout_highKeeps8400_victimClosedOutByWorkerStatus() throws Exception {
        StabilityMonitor monitor = new StabilityMonitor(inflightStore)
                .watchPrefill(getPrefillEndpoint())
                .watchDecode(getDecodeEndpoint())
                .pump(this::simulatePrefillFinishedReport);

        CompletableFuture<Response> victim = monitor.track(submitWithPriority(102, VICTIM_PRIORITY));
        awaitTrue(() -> mockPrefillWorker.getEnqueueCount() == 1, 3_000,
                "victim enqueue should reach the prefill worker");
        reportDecodeRunning(102);

        // Engine releases only after the bounded wait budget has expired.
        Thread feeder = scheduleVictimRelease(102, SLOW_RELEASE_DELAY_MS);

        capacityExhaustedOnce.add(202L);
        CompletableFuture<Response> high = monitor.track(submitWithPriority(202, HIGH_PRIORITY));

        // Wait budget expired without a confirmed release → the original route
        // failure is passed through, nothing is dispatched optimistically.
        Response highResponse = high.get(1, TimeUnit.SECONDS);
        assertFalse(highResponse.isSuccess());
        assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(), highResponse.getCode());
        assertEquals(1, mockPrefillWorker.getEnqueueCount(),
                "high-priority request must not be dispatched after an unconfirmed release");

        // Cancel did reach the engine; the victim stays non-terminal until the
        // WorkerStatus close-out lands (cancel intent still pending).
        assertEquals(1, mockDecodeWorker.getRpcService().getCancelCount());
        assertEquals(102, mockDecodeWorker.getRpcService()
                .getCancelledRequests().get(0).getRequestId());
        assertFalse(victim.isDone(),
                "victim settlement belongs to the WorkerStatus close-out, not the timed-out wait");

        // Close-out: the delayed finished report (cancelReason=2) settles 4290.
        Response victimResponse = victim.get(3, TimeUnit.SECONDS);
        assertFalse(victimResponse.isSuccess());
        assertEquals(StrategyErrorType.AUTO_TPM_PREEMPTED.getErrorCode(), victimResponse.getCode());

        feeder.join(3_000);
        monitor.assertQuiescent(ENQUEUE_DELAY_MS + 5_000);
    }

    // ==================== helpers ====================

    private CompletableFuture<Response> submitWithPriority(long requestId, int priority) {
        BalanceContext ctx = createBalanceContext(requestId);
        ctx.setPriority(priority);
        return scheduler.submit(ctx);
    }

    /** Feed a decode RUNNING report so the request enters layer 2 via calibrate. */
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

    /** Feed the decode finished report carrying the structured preempt attribution. */
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

    /**
     * Engine-release feeder: once the Cancel RPC for {@code victimId} is
     * recorded by the mock decode worker, wait {@code delayMs} and feed the
     * finished report (the engine's asynchronous release).
     */
    private Thread scheduleVictimRelease(long victimId, long delayMs) {
        Thread feeder = new Thread(() -> {
            try {
                long deadline = System.currentTimeMillis() + 5_000;
                while (System.currentTimeMillis() < deadline
                        && mockDecodeWorker.getRpcService().getCancelledRequests().stream()
                                .noneMatch(r -> r.getRequestId() == victimId)) {
                    Thread.sleep(10);
                }
                if (delayMs > 0) {
                    Thread.sleep(delayMs);
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
