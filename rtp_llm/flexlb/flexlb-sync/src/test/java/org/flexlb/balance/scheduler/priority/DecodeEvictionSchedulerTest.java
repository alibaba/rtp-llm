package org.flexlb.balance.scheduler.priority;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.scheduler.BatchDispatcher;
import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.balance.scheduler.DefaultBatchDispatcher;
import org.flexlb.balance.scheduler.FlexlbBatchScheduler;
import org.flexlb.balance.scheduler.Router;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.PrioritySloPolicy;
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
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Phase 4 tests for the decode reserved-only eviction path of
 * {@link PriorityAdmissionScheduler} wired through
 * {@link FlexlbBatchScheduler#submit}: higher priority evicts a strictly
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
    private FlexlbBatchScheduler scheduler;
    private EndpointRegistry endpointRegistry;
    private FlexlbConfig config;

    @BeforeEach
    void setUp() {
        configService = mock(ConfigService.class);
        router = mock(Router.class);
        grpcClient = mock(EngineGrpcClient.class);
        reporter = mock(BatchSchedulerReporter.class);
        priorityReporter = mock(PrioritySchedulerReporter.class);

        config = new FlexlbConfig();
        config.setScheduleWorkerSize(1);
        // Large batch + window + SLO keep queued items parked (no dispatch).
        config.setFlexlbBatchSizeMax(100);
        config.setFlexlbBatchWindowMs(10_000);
        config.setCostSloMs(50_000L);
        config.setCostSloRiskMarginMs(50L);
        config.setAutoTpmEnabled(true);
        config.setAutoTpmDecodeReservedEvictEnabled(true);
        // Single decode slot: the second admission always hits slot-full.
        config.setDecodeConcurrencyLimit(1);
        config.setFlexlbBatchQueueMaxSize(4);
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
        PriorityAdmissionScheduler priorityScheduler = new PriorityAdmissionScheduler(
                configService, router, endpointRegistry, new PlanCommitter(),
                new PrioritySloPolicy(PrioritySloPolicy.DEFAULT_SLO_LENGTH_BUCKETS,
                        PrioritySloPolicy.DEFAULT_PRIORITY_SLO_MULTIPLIERS),
                priorityReporter, reporter, new UnsupportedEngineCancelChannel()) {
            @Override
            protected ServerStatus selectPrefillForDecodeEviction(BalanceContext ctx,
                                                                  FlexlbConfig config,
                                                                  String group) {
                return server(RoleType.PREFILL, "10.0.0.1", 8080, 8081, ctx.getRequestId());
            }
        };
        scheduler = new FlexlbBatchScheduler(configService, router,
                endpointRegistry, dispatcher, reporter, priorityScheduler, null);

        WorkerStatus prefillWs = new WorkerStatus();
        prefillWs.setIp("10.0.0.1");
        prefillWs.setPort(8080);
        prefillWs.setGrpcPort(8081);
        endpointRegistry.ensureEndpoint(RoleType.PREFILL, PREFILL_IP_PORT, prefillWs);

        WorkerStatus decodeWs = new WorkerStatus();
        decodeWs.setIp("10.0.0.2");
        decodeWs.setPort(8081);
        decodeWs.setGrpcPort(8082);
        decodeWs.setAvailableKvCacheTokens(new AtomicLong(1_000_000L));
        decodeWs.setTotalKvCacheTokens(new AtomicLong(2_000_000L));
        endpointRegistry.ensureEndpoint(RoleType.DECODE, DECODE_IP_PORT, decodeWs);
        // Calibrate once so reportedKvAvailable reflects the engine report
        // (plenty of KV: only the slot dimension is ever in deficit here).
        endpointRegistry.getDecode(DECODE_IP_PORT)
                .onWorkerStatusUpdate(decodeWs, new WorkerStatusResponse());
    }

    @AfterEach
    void tearDown() {
        scheduler.shutdown();
    }

    private Response routeAnswer(BalanceContext ctx) {
        DecodeEndpoint decodeEp = endpointRegistry.getDecode(DECODE_IP_PORT);
        if (decodeEp.getTotalLoad() + 1 > config.getDecodeConcurrencyLimit()) {
            return Response.error(StrategyErrorType.NO_DECODE_WORKER);
        }
        decodeEp.reserve(ctx.getRequestId(), 128, 136, ctx.getPriority(), ctx.getDeadlineMs());
        return successRoute(ctx.getRequestId());
    }

    // ==================== higher priority evicts a reserved lower one ====================

    @Test
    void p70_evicts_reserved_p30_victim_when_decode_slots_full() throws Exception {
        DecodeEndpoint decodeEp = endpointRegistry.getDecode(DECODE_IP_PORT);

        CompletableFuture<Response> victim = scheduler.submit(context(1, 30));
        await(() -> decodeEp.reservedView().containsKey(1L));
        // task 35 P1-3: slot eviction now targets engine-facing load only —
        // model the victim as already dispatched to prefill (non-queued), the
        // exact shape whose reservation holds a real engine slot.
        decodeEp.markDispatchedPhase(1L);

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

        verify(priorityReporter).reportEvictionPlan(eq(70), eq("decode_slot_full"), eq("feasible"));
        verify(priorityReporter).reportEvictionCommit(eq(70), eq("decode_slot_full"), eq("success"));
        verify(priorityReporter).reportVictim(eq(30), eq(70), eq("decode_reserved"), eq("decode_slot_full"));
        verify(priorityReporter).reportVictimKvTokens(eq(30), eq("decode_reserved"), eq(128L));
    }

    // ==================== equal priority never yields ====================

    @Test
    void equal_priority_is_never_evicted_and_incoming_fails_explicitly() throws Exception {
        DecodeEndpoint decodeEp = endpointRegistry.getDecode(DECODE_IP_PORT);

        CompletableFuture<Response> victim = scheduler.submit(context(11, 50));
        await(() -> decodeEp.reservedView().containsKey(11L));

        Response response = scheduler.submit(context(12, 50)).get(2, TimeUnit.SECONDS);

        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(), response.getCode());
        // Redesign C-2: infeasible is an ordinary capacity failure — the full
        // retry budget is consumed, then a reason-tagged exhaustion surfaces.
        assertTrue(response.getErrorMessage().contains("reason=capacity_no_evict_candidates"),
                "expected reason-tagged exhaustion, got: " + response.getErrorMessage());
        // One route for the victim + MAX_PLAN_RETRIES (3) for the incoming
        verify(router, times(4)).route(any(BalanceContext.class));
        verify(priorityReporter, times(3))
                .reportEvictionPlan(eq(50), eq("decode_slot_full"), eq("infeasible"));
        verify(priorityReporter, never()).reportVictim(anyInt(), anyInt(), anyString(), anyString());

        // The reserved equal-priority request is untouched
        assertFalse(victim.isDone());
        assertTrue(decodeEp.reservedView().containsKey(11L));
        assertEquals(1, decodeEp.getInflightCount());
    }

    // ==================== gate off: legacy decode-full failure path ====================

    @Test
    void evict_switch_off_keeps_legacy_failure_and_never_plans() throws Exception {
        config.setAutoTpmDecodeReservedEvictEnabled(false);
        DecodeEndpoint decodeEp = endpointRegistry.getDecode(DECODE_IP_PORT);

        CompletableFuture<Response> victim = scheduler.submit(context(21, 30));
        await(() -> decodeEp.reservedView().containsKey(21L));

        Response response = scheduler.submit(context(22, 70)).get(2, TimeUnit.SECONDS);

        // The router's own decode-full failure surfaces unchanged
        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.NO_DECODE_WORKER.getErrorCode(), response.getCode());
        verify(router, times(2)).route(any(BalanceContext.class));
        verify(priorityReporter, never()).reportEvictionPlan(anyInt(), anyString(), anyString());

        assertFalse(victim.isDone());
        assertTrue(decodeEp.reservedView().containsKey(21L));
        assertEquals(1, decodeEp.getInflightCount());
    }

    // ==================== version conflict retries with a fresh plan ====================

    @Test
    void admission_version_mismatch_retries_and_commits_on_fresh_plan() throws Exception {
        // Pin the legacy queue_version guard: under the default victim_presence
        // mode unrelated admission-version churn no longer aborts a commit
        // (redesign N3) — this case asserts the legacy OCC retry contract.
        config.setAutoTpmVictimGuardMode("queue_version");
        DecodeEndpoint decodeEp = endpointRegistry.getDecode(DECODE_IP_PORT);
        // One-shot interference: bump the admission version between the
        // incoming's plan snapshot (pre-route) and its eviction commit.
        AtomicBoolean bumpOnce = new AtomicBoolean(true);
        when(router.route(any(BalanceContext.class))).thenAnswer(inv -> {
            BalanceContext ctx = inv.getArgument(0);
            if (decodeEp.getTotalLoad() + 1 > config.getDecodeConcurrencyLimit()
                    && bumpOnce.compareAndSet(true, false)) {
                decodeEp.reserve(555L, 0, 0);
                decodeEp.release(555L);
            }
            return routeAnswer(ctx);
        });

        CompletableFuture<Response> victim = scheduler.submit(context(31, 30));
        await(() -> decodeEp.reservedView().containsKey(31L));
        // task 35 P1-3: keep the victim engine-facing (non-queued) so the
        // slot-eviction plan under test still targets it.
        decodeEp.markDispatchedPhase(31L);

        CompletableFuture<Response> incoming = scheduler.submit(context(32, 70));

        // First attempt conflicts, the retry's fresh plan commits; the
        // reserved victim yields with 8400 (contract 5.3)
        Response victimResponse = victim.get(2, TimeUnit.SECONDS);
        assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(), victimResponse.getCode());
        assertTrue(victimResponse.getErrorMessage().contains("yielded to higher-priority request 32"));
        await(() -> decodeEp.reservedView().containsKey(32L));
        assertFalse(decodeEp.reservedView().containsKey(31L));
        assertFalse(incoming.isDone());

        verify(priorityReporter).reportPlanConflict(eq("decode_admission_version"));
        verify(priorityReporter).reportEvictionCommit(eq(70), eq("decode_slot_full"), eq("version_mismatch"));
        verify(priorityReporter).reportEvictionCommit(eq(70), eq("decode_slot_full"), eq("success"));
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
        scheduler.finishYieldedById(78, "second call must be ignored");
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
                FlexlbBatchScheduler.findServer(route, RoleType.PREFILL),
                FlexlbBatchScheduler.findServer(route, RoleType.DECODE),
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
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(128);
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
