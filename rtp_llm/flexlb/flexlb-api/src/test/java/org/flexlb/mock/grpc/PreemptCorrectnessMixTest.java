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

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Stage 4 E2E — preemption correctness under a mixed-priority population
 * (P30/P50/P60/P70) with capacity-limited routing, over the mock-worker
 * harness with a real {@link PriorityPressureController}. Gates asserted:
 *
 * <ul>
 *   <li>the victim is ALWAYS strictly lower priority than the incoming
 *       request — the Cancel record set on the mock decode worker never
 *       contains a peer- or higher-priority requestId (iron rule 2)</li>
 *   <li>4290 attribution appears ONLY on the preempted victim; every
 *       non-victim completes successfully and the refused incoming keeps
 *       the plain route failure 8400, never 429x</li>
 *   <li>preemption volume is capped by the rate limiter: with
 *       {@code autoTpmPreemptRateLimitPerMin=1} the second capacity-starved
 *       high request is refused without a second Cancel</li>
 *   <li>every accounting layer drains to zero after each scenario</li>
 * </ul>
 *
 * <p>Determinism: victim selection is a pure lexicographic order (priority
 * asc → iterateCount → kvTokens → requestId asc), so with two P30 victims
 * the lower requestId is always picked; the slow prefill enqueue (1.5s)
 * keeps candidate futures pending (still in the PriorityRegistry) while the
 * preemption runs, exactly as in CancelMidFlightPreemptTest.
 */
class PreemptCorrectnessMixTest extends FlexLBMockTestBase {

    private static final long ENQUEUE_DELAY_MS = 1_500L;
    private static final long WAIT_RELEASE_TIMEOUT_MS = 300L;

    private static final int P30 = 30;
    private static final int P50 = 50;
    private static final int P60 = 60;
    private static final int P70 = 70;

    /** Request IDs whose first route attempt fails with NO_AVAILABLE_WORKER. */
    private final Set<Long> capacityExhaustedOnce = ConcurrentHashMap.newKeySet();

    @Override
    protected MockWorkerBehavior createPrefillBehavior() {
        // Slow enqueue keeps candidate futures pending (still registered in
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

    // ---- gate: victim strictly lower priority, 4290 only on the victim ----

    @Test
    void mixedPriorities_victimIsStrictlyLower_attributionOnlyOnVictim() throws Exception {
        StabilityMonitor monitor = new StabilityMonitor(inflightStore)
                .watchPrefill(getPrefillEndpoint())
                .watchDecode(getDecodeEndpoint())
                .pump(this::simulatePrefillFinishedReport);

        // A P30 and a P50 request both reach RUNNING on the decode engine.
        CompletableFuture<Response> low = monitor.track(submitWithPriority(311, P30));
        CompletableFuture<Response> mid = monitor.track(submitWithPriority(551, P50));
        awaitTrue(() -> mockPrefillWorker.getEnqueueCount() == 2, 3_000,
                "both candidates should reach the prefill worker");
        reportDecodeRunning(311, 551);

        // Engine-side release: finished report follows the Cancel immediately.
        Thread feeder = scheduleVictimRelease(311, 0);

        // Capacity exhausted for the incoming P70 → preempt path.
        capacityExhaustedOnce.add(771L);
        CompletableFuture<Response> high = monitor.track(submitWithPriority(771, P70));

        // Victim settled with structured attribution AUTO_TPM_PREEMPTED (4290).
        Response lowResponse = low.get(2, TimeUnit.SECONDS);
        assertFalse(lowResponse.isSuccess());
        assertEquals(StrategyErrorType.AUTO_TPM_PREEMPTED.getErrorCode(), lowResponse.getCode());

        // The Cancel set contains ONLY the strictly-lower-priority victim:
        // never the P50 peer candidate, never the incoming request itself.
        List<EngineRpcService.CancelRequestPB> cancels =
                mockDecodeWorker.getRpcService().getCancelledRequests();
        assertEquals(1, cancels.size(), "exactly one Cancel should reach decode");
        Set<Long> cancelledIds = cancels.stream()
                .map(EngineRpcService.CancelRequestPB::getRequestId)
                .collect(Collectors.toSet());
        assertEquals(Set.of(311L), cancelledIds,
                "victim must be the strictly-lowest-priority candidate, got: " + cancelledIds);
        assertEquals(EngineRpcService.EngineCancelReasonPB.ENGINE_CANCEL_REASON_PRIORITY_PREEMPTED,
                cancels.get(0).getReason());
        assertEquals(0, mockPrefillWorker.getCancelCount(),
                "preemption must only ever cancel on the decode engine");

        // 4290 only on the victim: the P50 candidate and the incoming P70
        // both complete successfully.
        Response highResponse = high.get(ENQUEUE_DELAY_MS + 4_000, TimeUnit.MILLISECONDS);
        assertTrue(highResponse.isSuccess(), "incoming P70 should complete after preempt");
        Response midResponse = mid.get(ENQUEUE_DELAY_MS + 4_000, TimeUnit.MILLISECONDS);
        assertTrue(midResponse.isSuccess(), "P50 candidate must be untouched by the preemption");

        feeder.join(3_000);
        reportDecodeFinishedNormal(551);
        monitor.assertQuiescent(ENQUEUE_DELAY_MS + 5_000);
        assertEquals(0, inflightStore.activeCount());
    }

    // ---- gate: peer/higher priority is never preempted, incoming keeps 8400 ----

    @Test
    void peerAndHigherPriority_neverPreempted_incomingKeepsPlain8400() throws Exception {
        StabilityMonitor monitor = new StabilityMonitor(inflightStore)
                .watchPrefill(getPrefillEndpoint())
                .watchDecode(getDecodeEndpoint())
                .pump(this::simulatePrefillFinishedReport);

        // Only P50 and P60 requests are RUNNING on the decode engine.
        CompletableFuture<Response> peer = monitor.track(submitWithPriority(552, P50));
        CompletableFuture<Response> higher = monitor.track(submitWithPriority(661, P60));
        awaitTrue(() -> mockPrefillWorker.getEnqueueCount() == 2, 3_000,
                "both candidates should reach the prefill worker");
        reportDecodeRunning(552, 661);

        // Capacity exhausted for an incoming P50: no strictly-lower candidate
        // exists → no preemption, the plain route failure passes through.
        capacityExhaustedOnce.add(553L);
        CompletableFuture<Response> incoming = monitor.track(submitWithPriority(553, P50));

        Response incomingResponse = incoming.get(2, TimeUnit.SECONDS);
        assertFalse(incomingResponse.isSuccess());
        assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(), incomingResponse.getCode(),
                "refused incoming must keep the plain 8400 route failure, never 429x");

        // Iron rule 2: absolutely no Cancel for peer or higher priority.
        assertEquals(0, mockDecodeWorker.getCancelCount(),
                "peer/higher priority must never be preempted");
        assertEquals(0, mockPrefillWorker.getCancelCount());

        // No 4290 anywhere: both running candidates complete successfully.
        assertTrue(peer.get(ENQUEUE_DELAY_MS + 4_000, TimeUnit.MILLISECONDS).isSuccess());
        assertTrue(higher.get(ENQUEUE_DELAY_MS + 4_000, TimeUnit.MILLISECONDS).isSuccess());

        reportDecodeFinishedNormal(552, 661);
        monitor.assertQuiescent(ENQUEUE_DELAY_MS + 5_000);
        assertEquals(0, inflightStore.activeCount());
    }

    // ---- gate: preemption volume is capped by the rate limiter ----

    @Test
    void rateLimit_capsPreemption_secondHighRefusedWithoutSecondCancel() throws Exception {
        // Tighten the global window to a single permit; the controller
        // rebuilds its limiter when the config key changes.
        config.setAutoTpmPreemptRateLimitPerMin(1);

        StabilityMonitor monitor = new StabilityMonitor(inflightStore)
                .watchPrefill(getPrefillEndpoint())
                .watchDecode(getDecodeEndpoint())
                .pump(this::simulatePrefillFinishedReport);

        // Two P30 victims RUNNING; selection is deterministic: requestId asc
        // within the same priority level → 312 is picked first.
        CompletableFuture<Response> victim = monitor.track(submitWithPriority(312, P30));
        CompletableFuture<Response> survivor = monitor.track(submitWithPriority(313, P30));
        awaitTrue(() -> mockPrefillWorker.getEnqueueCount() == 2, 3_000,
                "both victims should reach the prefill worker");
        reportDecodeRunning(312, 313);

        Thread feeder = scheduleVictimRelease(312, 0);

        // First capacity-starved P70 consumes the only permit → preempts 312.
        capacityExhaustedOnce.add(772L);
        CompletableFuture<Response> firstHigh = monitor.track(submitWithPriority(772, P70));
        Response victimResponse = victim.get(2, TimeUnit.SECONDS);
        assertEquals(StrategyErrorType.AUTO_TPM_PREEMPTED.getErrorCode(), victimResponse.getCode());
        assertTrue(firstHigh.get(ENQUEUE_DELAY_MS + 4_000, TimeUnit.MILLISECONDS).isSuccess());
        assertEquals(1, mockDecodeWorker.getCancelCount());

        // Second capacity-starved P70 in the same window: the rate limiter
        // refuses → plain 8400 passthrough, NO second Cancel is issued even
        // though an eligible P30 victim (313) is still running.
        capacityExhaustedOnce.add(773L);
        CompletableFuture<Response> secondHigh = monitor.track(submitWithPriority(773, P70));
        Response secondResponse = secondHigh.get(2, TimeUnit.SECONDS);
        assertFalse(secondResponse.isSuccess());
        assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(), secondResponse.getCode());
        assertEquals(1, mockDecodeWorker.getCancelCount(),
                "rate limiter must cap preemption at one Cancel per window");

        // The surviving victim is untouched and completes successfully.
        assertTrue(survivor.get(ENQUEUE_DELAY_MS + 4_000, TimeUnit.MILLISECONDS).isSuccess());

        feeder.join(3_000);
        reportDecodeFinishedNormal(313);
        monitor.assertQuiescent(ENQUEUE_DELAY_MS + 5_000);
        assertEquals(0, inflightStore.activeCount());
    }

    // ==================== helpers ====================

    private CompletableFuture<Response> submitWithPriority(long requestId, int priority) {
        BalanceContext ctx = createBalanceContext(requestId);
        ctx.setPriority(priority);
        return scheduler.submit(ctx);
    }

    /** Feed one decode RUNNING report covering all {@code requestIds} at once. */
    private void reportDecodeRunning(long... requestIds) {
        Map<String, TaskInfo> running = new HashMap<>();
        for (long requestId : requestIds) {
            TaskInfo task = new TaskInfo();
            task.setRequestId(requestId);
            task.setPhase(TaskPhase.RUNNING);
            running.put(String.valueOf(requestId), task);
        }
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(RoleType.DECODE);
        response.setRunningTaskInfo(running);
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

    /** Feed a normal (successful) decode finished report to drain accounting. */
    private void reportDecodeFinishedNormal(long... requestIds) {
        Map<String, TaskInfo> finished = new HashMap<>();
        for (long requestId : requestIds) {
            TaskInfo task = new TaskInfo();
            task.setRequestId(requestId);
            finished.put(String.valueOf(requestId), task);
        }
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(RoleType.DECODE);
        response.setFinishedTaskInfo(finished);
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
