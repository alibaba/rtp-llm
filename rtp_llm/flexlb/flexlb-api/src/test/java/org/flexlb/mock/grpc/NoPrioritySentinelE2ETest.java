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
import org.flexlb.mock.MockWorkerBehavior;
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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

/**
 * D12 (task40 revision) guard — the 0 sentinel ("no priority carried")
 * takes the FULL legacy path even with every AUTO_TPM switch on:
 *
 * <ul>
 *   <li>never registered in the PriorityRegistry → never a victim
 *       candidate; an incoming high-priority request that finds no
 *       capacity keeps the plain 8400, no Cancel is ever issued</li>
 *   <li>never yield-skipped in the batcher (covered E2E by completing
 *       alongside prioritized traffic; unit-locked in
 *       PriorityYieldBatcherAlgorithmTest)</li>
 *   <li>never initiates preemption itself: capacity-starved no-priority
 *       request fails plain 8400 without a preempt attempt</li>
 *   <li>the engine receives {@code GenerateConfigPB.priority == 0} (field
 *       not set) — default-state metric parity on the engine side</li>
 *   <li>no auto_tpm priority metric is emitted anywhere on the path</li>
 * </ul>
 */
class NoPrioritySentinelE2ETest extends FlexLBMockTestBase {

    private static final long ENQUEUE_DELAY_MS = 800L;
    private static final int P70 = 70;

    /** Request IDs whose first route attempt fails with NO_AVAILABLE_WORKER. */
    private final Set<Long> capacityExhaustedOnce = ConcurrentHashMap.newKeySet();

    private FlexlbMetricHelper metricHelperMock;

    @Override
    protected MockWorkerBehavior createPrefillBehavior() {
        // Slow enqueue keeps candidate futures pending while preemption
        // would run — the exact window where a victim could be picked.
        return MockWorkerBehavior.builder()
                .enqueueDelayMs(ENQUEUE_DELAY_MS)
                .build();
    }

    @Override
    protected MockWorkerBehavior createDecodeBehavior() {
        return MockWorkerBehavior.builder()
                .cancelFoundRunning(true)
                .build();
    }

    @Override
    protected FlexlbConfig createConfig() {
        // Every AUTO_TPM switch ON — the sentinel must still take the
        // legacy path end-to-end.
        FlexlbConfig cfg = super.createConfig();
        cfg.setAutoTpmEnabled(true);
        cfg.setAutoTpmQueueYieldEnabled(true);
        cfg.setAutoTpmDecodeRunningPreemptEnabled(true);
        cfg.setAutoTpmPreemptRateLimitPerMin(10);
        cfg.setAutoTpmEndpointPreemptQpsLimit(0);
        cfg.setAutoTpmCommitWaitReleaseTimeoutMs(300L);
        cfg.setAutoTpmPreemptCriticalSectionMs(0);
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
        metricHelperMock = mock(FlexlbMetricHelper.class);
        scheduler.setPressureController(new PriorityPressureController(
                configService, endpointRegistry, grpcClient, inflightStore,
                scheduler.priorityRegistry(), metricHelperMock));
    }

    // ---- gate: full legacy pass-through, priority=0 on the wire ----

    @Test
    void noPriorityRequest_completesLegacyPath_zeroPriorityOnWire() throws Exception {
        CompletableFuture<Response> legacy = submitRequest(9001);

        awaitTrue(() -> mockPrefillWorker.getEnqueueCount() == 1, 3_000,
                "the no-priority request should reach the prefill worker");
        // 0 sentinel is never registered in the PriorityRegistry
        assertEquals(0, scheduler.priorityRegistry().size(),
                "no-priority requests must never enter the priority registry");

        assertTrue(legacy.get(ENQUEUE_DELAY_MS + 4_000, TimeUnit.MILLISECONDS).isSuccess());

        // Engine side: GenerateConfigPB.priority stays 0 (field not set) —
        // parity with the pre-Auto-TPM default metric shape.
        EngineRpcService.EnqueueBatchRequestPB recorded =
                mockPrefillWorker.getRpcService().getEnqueuedRequests().get(0);
        EngineRpcService.GenerateInputPB input = recorded.getDpSlots(0).getRequests(0).getInput();
        assertEquals(9001, input.getRequestId());
        assertEquals(0, input.getGenerateConfig().getPriority(),
                "engine must receive priority=0 for a no-priority request");

        // No auto_tpm metric anywhere on the legacy path.
        verifyNoInteractions(metricHelperMock);

        simulatePrefillFinishedReport();
        assertEquals(0, inflightStore.activeCount());
    }

    // ---- gate: 0-sentinel RUNNING request is never selected as a victim ----

    @Test
    void noPriorityRunning_neverVictim_incomingHighKeepsPlain8400() throws Exception {
        // A no-priority request reaches RUNNING on the decode engine.
        CompletableFuture<Response> legacy = submitRequest(9002);
        awaitTrue(() -> mockPrefillWorker.getEnqueueCount() == 1, 3_000,
                "the no-priority request should reach the prefill worker");
        reportDecodeRunning(9002);

        // Capacity exhausted for an incoming P70: the only RUNNING request
        // carries the 0 sentinel → no victim, plain 8400 passthrough.
        capacityExhaustedOnce.add(9701L);
        CompletableFuture<Response> high = submitWithPriority(9701, P70);

        Response highResponse = high.get(2, TimeUnit.SECONDS);
        assertFalse(highResponse.isSuccess());
        assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(), highResponse.getCode(),
                "no eligible victim → the incoming keeps the plain 8400, never 429x");

        // Absolutely no Cancel: the sentinel is not preemptable.
        assertEquals(0, mockDecodeWorker.getCancelCount(),
                "a no-priority request must never be preempted");
        assertEquals(0, mockPrefillWorker.getCancelCount());

        // The no-priority request is untouched and completes successfully.
        assertTrue(legacy.get(ENQUEUE_DELAY_MS + 4_000, TimeUnit.MILLISECONDS).isSuccess());

        // No victim was ever selected → no auto_tpm metric was emitted.
        verifyNoInteractions(metricHelperMock);

        reportDecodeFinishedNormal(9002);
        simulatePrefillFinishedReport();
        awaitTrue(() -> inflightStore.activeCount() == 0, 3_000,
                "all accounting must drain to zero");
    }

    // ---- gate: 0-sentinel incoming never initiates preemption ----

    @Test
    void noPriorityIncoming_capacityExhausted_plain8400_noPreemptAttempt() throws Exception {
        // A P30 request reaches RUNNING — a would-be victim if the incoming
        // carried any priority.
        CompletableFuture<Response> low = submitWithPriority(9003, 30);
        awaitTrue(() -> mockPrefillWorker.getEnqueueCount() == 1, 3_000,
                "the P30 request should reach the prefill worker");
        reportDecodeRunning(9003);

        // Capacity exhausted for a no-priority incoming: it must NOT try to
        // preempt the P30 — plain 8400, no Cancel.
        capacityExhaustedOnce.add(9004L);
        CompletableFuture<Response> legacyIncoming = submitRequest(9004);

        Response incomingResponse = legacyIncoming.get(2, TimeUnit.SECONDS);
        assertFalse(incomingResponse.isSuccess());
        assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(), incomingResponse.getCode(),
                "a no-priority request finding no capacity fails plain 8400 without preempting");
        assertEquals(0, mockDecodeWorker.getCancelCount(),
                "a no-priority incoming must never trigger a Cancel");

        // The P30 candidate is untouched and completes successfully.
        assertTrue(low.get(ENQUEUE_DELAY_MS + 4_000, TimeUnit.MILLISECONDS).isSuccess());

        verifyNoInteractions(metricHelperMock);

        reportDecodeFinishedNormal(9003);
        simulatePrefillFinishedReport();
        awaitTrue(() -> inflightStore.activeCount() == 0, 3_000,
                "all accounting must drain to zero");
    }

    // ==================== helpers ====================

    private CompletableFuture<Response> submitWithPriority(long requestId, int priority) {
        BalanceContext ctx = createBalanceContext(requestId);
        ctx.setPriority(priority);
        ctx.getRequest().setPriority(priority);
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
