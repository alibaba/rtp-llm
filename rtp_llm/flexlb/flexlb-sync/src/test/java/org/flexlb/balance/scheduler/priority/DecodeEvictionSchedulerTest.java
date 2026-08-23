package org.flexlb.balance.scheduler.priority;

import org.flexlb.balance.scheduler.SchedulingTestConfig;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.scheduler.BatchDispatcher;
import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.balance.scheduler.DefaultBatchDispatcher;
import org.flexlb.balance.scheduler.PriorityScheduler;
import org.flexlb.balance.scheduler.Router;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.PrioritySchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.BooleanSupplier;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Phase 4 tests for the decode reserved-only eviction path of
 * {@link PriorityAdmissionScheduler} wired through
 * {@link PriorityScheduler#submit}: higher priority evicts a strictly
 * lower-priority decode reservation when the decode capacity is exhausted
 * (the reserved-only victim yields with the retryable NO_AVAILABLE_WORKER,
 * contract 5.3), equal priority never yields, the gate keeps the legacy
 * failure path, version conflicts retry with a fresh plan, and victim
 * termination by id is idempotent (design doc 3.3, 10-13, 17.2-17.3).
 */
class DecodeEvictionSchedulerTest {

    private static final String PREFILL_IP_PORT = "10.0.0.1:8080";
    private static final String DECODE_IP_PORT = "10.0.0.2:8081";

    private ConfigService configService;
    private Router router;
    private EngineGrpcClient grpcClient;
    private BatchSchedulerReporter reporter;
    private PrioritySchedulerReporter priorityReporter;
    private PriorityAdmissionScheduler priorityScheduler;
    private PriorityScheduler scheduler;
    private EndpointRegistry endpointRegistry;
    private FlexlbConfig config;
    private final AtomicBoolean failDecodeEvictionPlacement = new AtomicBoolean();
    private final AtomicBoolean failNextVictimSettlement = new AtomicBoolean();

    @BeforeEach
    void setUp() {
        configService = mock(ConfigService.class);
        router = mock(Router.class);
        grpcClient = mock(EngineGrpcClient.class);
        reporter = mock(BatchSchedulerReporter.class);
        priorityReporter = mock(PrioritySchedulerReporter.class);

        config = new FlexlbConfig();
        // Large batch + window + SLO keep queued items parked (no dispatch).
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxRequests(100);
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxCollectionWaitMs(10_000);
        SchedulingTestConfig.usePriorityQueue(config);
        SchedulingTestConfig.allowVictim(config, org.flexlb.config.VictimStage.DECODE_RESERVED);
        // Single decode slot: the second admission always hits slot-full.
        config.getRouter().getRoles().getDecode().getAvailability().setMaxEngineRequests((long) (1));
        SchedulingTestConfig.useQueueCapacity(config).setMaxWaitingRequestsPerPrefillWorker(4);
        when(configService.loadBalanceConfig()).thenReturn(config);

        // Capacity-aware route stand-in: mirrors the production decode hard
        // filter (slot limit) and, on success, takes the priority-carrying
        // decode reservation exactly like the production Router.
        when(router.route(any(BalanceContext.class))).thenAnswer(inv -> routeAnswer(inv.getArgument(0)));
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(),
                any(EngineRpcService.EnqueueBatchRequestPB.class), anyLong()))
                .thenAnswer(inv -> CompletableFuture.completedFuture(ackFor(inv.getArgument(2))));

        endpointRegistry = new EndpointRegistry(configService, () -> scheduler, reporter);
        BatchDispatcher dispatcher = new DefaultBatchDispatcher(grpcClient, configService, null);
        // Test seam: after a decode eviction the prefill is picked manually
        // (the static strategy factory is not populated in unit tests).
        priorityScheduler = new PriorityAdmissionScheduler(
                configService, router, endpointRegistry, new PlanCommitter(),
                priorityReporter, reporter, new UnsupportedEngineCancelChannel()) {
            @Override
            protected ServerStatus selectPrefillForDecodeEviction(BalanceContext ctx,
                                                                  FlexlbConfig config,
                                                                  String group) {
                if (failDecodeEvictionPlacement.get()) {
                    throw new IllegalStateException("prefill selection failed");
                }
                return server(RoleType.PREFILL, "10.0.0.1", 8080, 8081, ctx.getRequestId());
            }
        };
        scheduler = new PriorityScheduler(configService, router,
                endpointRegistry, dispatcher, reporter, priorityScheduler, null,
                new UnsupportedEngineCancelChannel()) {
            @Override
            public void finishYieldedById(long requestId, String detail) {
                if (failNextVictimSettlement.compareAndSet(true, false)) {
                    throw new IllegalStateException("victim settlement interrupted");
                }
                super.finishYieldedById(requestId, detail);
            }
        };

        WorkerStatus prefillWs = new WorkerStatus();
        prefillWs.setIp("10.0.0.1");
        prefillWs.setPort(8080);
        prefillWs.setGrpcPort(8081);
        endpointRegistry.ensureEndpoint(RoleType.PREFILL, PREFILL_IP_PORT, prefillWs);

        WorkerStatus decodeWs = new WorkerStatus();
        decodeWs.setIp("10.0.0.2");
        decodeWs.setPort(8081);
        decodeWs.setGrpcPort(8082);
        decodeWs.setAvailableKvCacheTokens(new AtomicLong(128L));
        decodeWs.setTotalKvCacheTokens(new AtomicLong(256L));
        endpointRegistry.ensureEndpoint(RoleType.DECODE, DECODE_IP_PORT, decodeWs);
        // Calibrate once so reportedKvAvailable reflects the engine report
        // (plenty of KV: only the slot dimension is ever in deficit here).
        endpointRegistry.getDecode(DECODE_IP_PORT)
                .onWorkerStatusUpdate(decodeWs, new WorkerStatusResponse());
    }

    @AfterEach
    void tearDown() {
        priorityScheduler.shutdown();
        scheduler.shutdown();
    }

    private Response routeAnswer(BalanceContext ctx) {
        DecodeEndpoint decodeEp = endpointRegistry.getDecode(DECODE_IP_PORT);
        boolean slotFull = config.getRouter().getRoles().getDecode().getAvailability().getMaxEngineRequests() > 0
                && decodeEp.getEngineLoad() + 1 > config.getRouter().getRoles().getDecode().getAvailability().getMaxEngineRequests();
        boolean kvFull = decodeEp.realKvTotal() > 0
                && decodeEp.realKvAvailable() < ctx.getRequest().getSeqLen();
        if (slotFull || kvFull) {
            return Response.error(StrategyErrorType.NO_DECODE_WORKER);
        }
        long seqLen = ctx.getRequest().getSeqLen();
        decodeEp.reserve(ctx.getRequestId(), seqLen, seqLen + 8,
                ctx.getPriority());
        return successRoute(ctx.getRequestId());
    }

    // ==================== higher priority evicts a reserved lower one ====================

    @Test
    void p70_evicts_reserved_p30_victim_when_decode_slots_full() throws Exception {
        DecodeEndpoint decodeEp = endpointRegistry.getDecode(DECODE_IP_PORT);

        CompletableFuture<Response> victim = scheduler.submit(context(1, 30));
        await(() -> decodeEp.reservedView().containsKey(1L));
        // Keep the local victim Master-queued. Its reservation consumes the
        // available KV, so the incoming uses the local KV-eviction path.

        CompletableFuture<Response> incoming = scheduler.submit(context(2, 70));

        // Reserved-only victim yields with retryable NO_AVAILABLE_WORKER —
        // the engine never saw it (contract 5.3); never PRIORITY_PREEMPTED.
        Response victimResponse = victim.get(2, TimeUnit.SECONDS);
        assertFalse(victimResponse.isSuccess());
        assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(), victimResponse.getCode());
        assertTrue(victimResponse.getErrorMessage().contains("yielded to higher-priority request 2"));

        // Shadow state swapped atomically: incoming reserved, victim gone
        await(() -> decodeEp.reservedView().containsKey(2L));
        assertFalse(decodeEp.reservedView().containsKey(1L));
        assertEquals(70, decodeEp.reservedView().get(2L).priority());
        assertEquals(1, decodeEp.getInflightCount());
        assertEquals(1, decodeEp.getTotalLoad());
        assertFalse(incoming.isDone());

        verify(priorityReporter).reportEvictionPlan(eq(70), eq("decode_kv_full"), eq("feasible"));
        verify(priorityReporter).reportEvictionCommit(eq(70), eq("decode_kv_full"), eq("success"));
        verify(priorityReporter).reportVictim(eq(30), eq(70), eq("decode_reserved"), eq("decode_kv_full"));
        verify(priorityReporter).reportVictimKvTokens(eq(30), eq("decode_reserved"), eq(128L));
    }

    @Test
    void reservedEvictionReleasesIncomingWhenPlacementHandoffThrows()
            throws Exception {
        DecodeEndpoint decodeEp = endpointRegistry.getDecode(DECODE_IP_PORT);
        CompletableFuture<Response> victim = scheduler.submit(context(3, 30));
        await(() -> decodeEp.reservedView().containsKey(3L));
        failDecodeEvictionPlacement.set(true);

        Response incoming = scheduler.submit(context(4, 70))
                .get(2, TimeUnit.SECONDS);

        assertFalse(incoming.isSuccess());
        assertEquals(StrategyErrorType.BATCH_DISPATCH_FAILED.getErrorCode(),
                incoming.getCode());
        assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(),
                victim.get(1, TimeUnit.SECONDS).getCode());
        assertFalse(decodeEp.reservedView().containsKey(3L));
        assertFalse(decodeEp.reservedView().containsKey(4L),
                "failed pre-register handoff must release its provisional reservation");
        assertFalse(scheduler.ownsRequestGeneration(4L));
    }

    @Test
    void transient_victim_settlement_failure_does_not_interrupt_decode_swap()
            throws Exception {
        DecodeEndpoint decodeEp = endpointRegistry.getDecode(DECODE_IP_PORT);
        WorkerStatus decodeStatus = decodeEp.getStatus();
        decodeStatus.setAvailableKvCacheTokens(new AtomicLong(256L));
        decodeStatus.setTotalKvCacheTokens(new AtomicLong(512L));
        decodeEp.onWorkerStatusUpdate(decodeStatus, new WorkerStatusResponse());

        CompletableFuture<Response> firstVictim = scheduler.submit(context(5, 20));
        CompletableFuture<Response> secondVictim = scheduler.submit(context(6, 30));
        await(() -> decodeEp.reservedView().size() == 2);

        // A 256-token request needs both 128-token victims. The first reducer
        // call fails once after the endpoint has atomically swapped the
        // reservations; the idempotent retry and remaining drain must still
        // settle both exact generations.
        failNextVictimSettlement.set(true);
        CompletableFuture<Response> incoming = scheduler.submit(context(7, 70, 256));

        assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(),
                firstVictim.get(2, TimeUnit.SECONDS).getCode());
        assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(),
                secondVictim.get(2, TimeUnit.SECONDS).getCode());
        await(() -> decodeEp.reservedView().size() == 1
                && decodeEp.reservedView().containsKey(7L));
        assertFalse(incoming.isDone());
        assertFalse(scheduler.ownsRequestGeneration(5L));
        assertFalse(scheduler.ownsRequestGeneration(6L));
    }

    // ==================== equal priority never yields ====================

    @Test
    void equal_priority_is_never_evicted_and_incoming_fails_explicitly() throws Exception {
        DecodeEndpoint decodeEp = endpointRegistry.getDecode(DECODE_IP_PORT);

        CompletableFuture<Response> victim = scheduler.submit(context(11, 50));
        await(() -> decodeEp.reservedView().containsKey(11L));

        Response response = scheduler.submit(context(12, 50)).get(2, TimeUnit.SECONDS);

        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.PRIORITY_ADMISSION_REJECTED.getErrorCode(), response.getCode());
        assertEquals(org.flexlb.dao.loadbalance.AdmissionRejectReason.SAME_PRIORITY_AHEAD,
                response.getAdmissionRejectReason());
        verify(router, times(2)).route(any(BalanceContext.class));
        verify(priorityReporter)
                .reportEvictionPlan(eq(50), eq("decode_kv_full"), eq("infeasible"));
        verify(priorityReporter, never()).reportVictim(anyInt(), anyInt(), anyString(), anyString());

        // The reserved equal-priority request is untouched
        assertFalse(victim.isDone());
        assertTrue(decodeEp.reservedView().containsKey(11L));
        assertEquals(1, decodeEp.getInflightCount());
    }

    // ==================== gate off: legacy decode-full failure path ====================

    @Test
    void evict_switch_off_keeps_legacy_failure_and_never_plans() throws Exception {
        SchedulingTestConfig.disallowVictim(config, org.flexlb.config.VictimStage.DECODE_RESERVED);
        DecodeEndpoint decodeEp = endpointRegistry.getDecode(DECODE_IP_PORT);

        CompletableFuture<Response> victim = scheduler.submit(context(21, 30));
        await(() -> decodeEp.reservedView().containsKey(21L));

        Response response = scheduler.submit(context(22, 70)).get(2, TimeUnit.SECONDS);

        // Auto-TPM never leaks the router's generic 8403. The lower-priority
        // KV looks sufficient, but the disabled eviction gate means this
        // snapshot cannot prove a priority blocker, so the admission contract
        // falls back to resource exhaustion.
        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(), response.getCode());
        assertEquals(org.flexlb.dao.loadbalance.AdmissionRejectReason.RESOURCE_EXHAUSTED,
                response.getAdmissionRejectReason());
        verify(router, times(2)).route(any(BalanceContext.class));
        verify(priorityReporter, never()).reportEvictionPlan(anyInt(), anyString(), anyString());

        assertFalse(victim.isDone());
        assertTrue(decodeEp.reservedView().containsKey(21L));
        assertEquals(1, decodeEp.getInflightCount());
    }

    // ==================== victim termination by id is idempotent ====================

    @Test
    void finish_preempted_by_id_is_idempotent_and_unknown_id_is_noop() throws Exception {
        DecodeEndpoint decodeEp = endpointRegistry.getDecode(DECODE_IP_PORT);
        decodeEp.reserve(77, 128, 136);
        BatchItem item = dummyItem(77);
        assertTrue(scheduler.registerInflight(item));

        scheduler.finishPreemptedById(77, "preempted by higher-priority request 88");
        scheduler.finishPreemptedById(77, "second call must be ignored");
        scheduler.finishPreemptedById(888, "unknown id is a no-op");

        Response response = item.future().get(1, TimeUnit.SECONDS);
        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.PRIORITY_PREEMPTED.getErrorCode(), response.getCode());
        assertTrue(response.getErrorMessage().contains("preempted by higher-priority request 88"));

        // Decode reservation released exactly once (no double release)
        assertEquals(0, decodeEp.getInflightCount());
        assertEquals(0, decodeEp.inflightHardKvReserved());
        assertEquals(0, decodeEp.getTotalLoad());
    }

    @Test
    void finish_yielded_by_id_is_idempotent_and_unknown_id_is_noop() throws Exception {
        DecodeEndpoint decodeEp = endpointRegistry.getDecode(DECODE_IP_PORT);
        decodeEp.reserve(78, 128, 136);
        BatchItem item = dummyItem(78);
        assertTrue(scheduler.registerInflight(item));

        scheduler.finishYieldedById(78, "yielded to higher-priority request 88");
        doThrow(new IllegalStateException("metrics unavailable"))
                .when(priorityReporter)
                .reportInflightSettleMiss(anyString());
        assertDoesNotThrow(() -> scheduler.finishYieldedById(
                78, "second call must be ignored"));
        scheduler.finishYieldedById(888, "unknown id is a no-op");
        // A racing preempt terminal must not override the yielded terminal
        scheduler.finishPreemptedById(78, "late preempt must be ignored");

        Response response = item.future().get(1, TimeUnit.SECONDS);
        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(), response.getCode());
        assertTrue(response.getErrorMessage().contains("yielded to higher-priority request 88"));

        // Decode reservation released exactly once (no double release)
        assertEquals(0, decodeEp.getInflightCount());
        assertEquals(0, decodeEp.inflightHardKvReserved());
        assertEquals(0, decodeEp.getTotalLoad());
    }

    // ==================== helpers ====================

    private static void await(BooleanSupplier condition) throws InterruptedException {
        long deadline = System.currentTimeMillis() + 2_000;
        while (!condition.getAsBoolean()) {
            if (System.currentTimeMillis() > deadline) {
                throw new AssertionError("condition not met within 2s");
            }
            TimeUnit.MILLISECONDS.sleep(10);
        }
    }

    private BatchItem dummyItem(long requestId) {
        Response route = successRoute(requestId);
        return new BatchItem(context(requestId, 50), new CompletableFuture<>(), route,
                PriorityScheduler.findServer(route, RoleType.PREFILL),
                PriorityScheduler.findServer(route, RoleType.DECODE),
                endpointRegistry.getPrefill(PREFILL_IP_PORT),
                endpointRegistry.getDecode(DECODE_IP_PORT),
                System.currentTimeMillis());
    }

    private static EngineRpcService.EnqueueBatchResponsePB ackFor(
            EngineRpcService.EnqueueBatchRequestPB request) {
        EngineRpcService.EnqueueBatchResponsePB.Builder response =
                EngineRpcService.EnqueueBatchResponsePB.newBuilder().setBatchId(request.getBatchId());
        request.getDpSlotsList().stream()
                .flatMap(slot -> slot.getRequestsList().stream())
                .map(external -> external.getInput().getRequestId())
                .forEach(requestId -> response.addSuccesses(
                        EngineRpcService.EnqueueBatchSuccessPB.newBuilder()
                                .setRequestId(requestId)
                                .build()));
        return response.build();
    }

    private static BalanceContext context(long requestId, int priority) {
        return context(requestId, priority, 128);
    }

    private static BalanceContext context(long requestId, int priority, int seqLen) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(seqLen);
        request.setMaxNewTokens(8);
        request.setNumBeams(1);
        request.setModel("test-model");
        request.setPriority(priority);

        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(request);
        ctx.setConfig(new FlexlbConfig());
        ctx.setGenerateInputPbBytes(generateInputBytes(requestId));
        return ctx;
    }

    private static byte[] generateInputBytes(long requestId) {
        EngineRpcService.GenerateInputPB input = EngineRpcService.GenerateInputPB.newBuilder()
                .setRequestId(requestId)
                .addTokenIds(101)
                .addTokenIds(102)
                .setGenerateConfig(EngineRpcService.GenerateConfigPB.newBuilder()
                        .setMaxNewTokens(8)
                        .build())
                .build();
        return input.toByteArray();
    }

    private static Response successRoute(long requestId) {
        Response response = new Response();
        response.setSuccess(true);
        response.setServerStatus(List.of(
                server(RoleType.PREFILL, "10.0.0.1", 8080, 8081, requestId),
                server(RoleType.DECODE, "10.0.0.2", 8081, 8082, requestId)
        ));
        return response;
    }

    private static ServerStatus server(RoleType role, String ip, int httpPort, int grpcPort, long requestId) {
        ServerStatus status = new ServerStatus();
        status.setSuccess(true);
        status.setRole(role);
        status.setServerIp(ip);
        status.setHttpPort(httpPort);
        status.setGrpcPort(grpcPort);
        status.setDpRank(0);
        status.setGroup("g1");
        status.setRequestId(requestId);
        return status;
    }
}
