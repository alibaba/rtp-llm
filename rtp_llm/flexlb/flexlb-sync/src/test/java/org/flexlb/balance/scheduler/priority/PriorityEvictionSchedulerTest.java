package org.flexlb.balance.scheduler.priority;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.scheduler.BatchDispatcher;
import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.balance.scheduler.DefaultBatchDispatcher;
import org.flexlb.balance.scheduler.FlexlbBatchScheduler;
import org.flexlb.balance.scheduler.Router;
import org.flexlb.balance.scheduler.WorkerBatcher;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.PrioritySloPolicy;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.ScheduleBudget;
import org.flexlb.dao.loadbalance.AdmissionRejectReason;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerStatus;
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
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.TimeUnit;
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
 * Phase 3 tests for the prefill-queue eviction path of
 * {@link PriorityAdmissionScheduler} wired through
 * {@link FlexlbBatchScheduler#submit}: higher priority evicts strictly lower
 * priority on a full queue (the queued victim yields with the retryable
 * NO_AVAILABLE_WORKER, contract 5.3), equal priority never yields, the gate
 * keeps the legacy retry path, victim termination is idempotent, and Auto-TPM
 * never silently drops an expired head (design doc 8.3, 9.1-9.5, 17.2-17.3).
 */
class PriorityEvictionSchedulerTest {

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
    private final List<EngineRpcService.EnqueueBatchRequestPB> sentBatches = new CopyOnWriteArrayList<>();

    @BeforeEach
    void setUp() {
        configService = mock(ConfigService.class);
        router = mock(Router.class);
        grpcClient = mock(EngineGrpcClient.class);
        reporter = mock(BatchSchedulerReporter.class);
        priorityReporter = mock(PrioritySchedulerReporter.class);

        config = new FlexlbConfig();
        config.setScheduleWorkerSize(1);
        // Large batch + window + SLO keep queued items parked (no dispatch),
        // so eviction races with neither dispatch nor expiry.
        config.setFlexlbBatchSizeMax(100);
        config.setFlexlbBatchWindowMs(10_000);
        config.setCostSloMs(50_000L);
        config.setCostSloRiskMarginMs(50L);
        config.setAutoTpmEnabled(true);
        config.setAutoTpmPrefillQueueEvictEnabled(true);
        config.setFlexlbBatchQueueMaxSize(1);
        when(configService.loadBalanceConfig()).thenReturn(config);

        // Route reserves the decode (D reserve first), like production Router
        when(router.route(any(BalanceContext.class))).thenAnswer(inv -> {
            BalanceContext ctx = inv.getArgument(0);
            endpointRegistry.getDecode(DECODE_IP_PORT).reserve(ctx.getRequestId(), 128, 136);
            return successRoute(ctx.getRequestId());
        });
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(),
                any(EngineRpcService.EnqueueBatchRequestPB.class), anyLong()))
                .thenAnswer(inv -> {
                    EngineRpcService.EnqueueBatchRequestPB request = inv.getArgument(2);
                    sentBatches.add(request);
                    return CompletableFuture.completedFuture(ackFor(request));
                });

        endpointRegistry = new EndpointRegistry(configService, () -> scheduler, reporter);
        BatchDispatcher dispatcher = new DefaultBatchDispatcher(grpcClient, configService, null);
        PriorityAdmissionScheduler priorityScheduler = new PriorityAdmissionScheduler(
                configService, router, endpointRegistry, new PlanCommitter(),
                new PrioritySloPolicy(PrioritySloPolicy.DEFAULT_SLO_LENGTH_BUCKETS,
                        PrioritySloPolicy.DEFAULT_PRIORITY_SLO_MULTIPLIERS),
                priorityReporter, reporter, new UnsupportedEngineCancelChannel());
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
    }

    @AfterEach
    void tearDown() {
        scheduler.shutdown();
    }

    // ==================== higher priority evicts lower on full queue ====================

    @Test
    void p70_evicts_queued_p30_victim_when_prefill_queue_full() throws Exception {
        WorkerBatcher batcher = endpointRegistry.getPrefill(PREFILL_IP_PORT).getBatcher();
        DecodeEndpoint decodeEp = endpointRegistry.getDecode(DECODE_IP_PORT);

        CompletableFuture<Response> victim = scheduler.submit(context(1, 30));
        await(() -> batcher.queueSize() == 1);

        CompletableFuture<Response> incoming = scheduler.submit(context(2, 70));

        // Queued victim yields with retryable NO_AVAILABLE_WORKER — the engine
        // never saw it (contract 5.3); never PRIORITY_PREEMPTED.
        Response victimResponse = victim.get(2, TimeUnit.SECONDS);
        assertFalse(victimResponse.isSuccess());
        assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(), victimResponse.getCode());
        assertTrue(victimResponse.getErrorMessage().contains("yielded to higher-priority request 2"));

        // Incoming replaced the victim in the single queue slot and stays queued
        await(() -> batcher.queueSize() == 1);
        assertFalse(incoming.isDone());
        // Victim's decode reservation released; only the incoming's remains
        await(() -> decodeEp.getInflightCount() == 1);
        assertEquals(1, decodeEp.getTotalLoad());

        verify(priorityReporter).reportEvictionPlan(eq(70), eq("prefill_queue_full"), eq("feasible"));
        verify(priorityReporter).reportEvictionCommit(eq(70), eq("prefill_queue_full"), eq("success"));
        verify(priorityReporter).reportVictim(eq(30), eq(70), eq("prefill_queued"), eq("prefill_queue_full"));
    }

    // ==================== equal priority never yields ====================

    @Test
    void equal_priority_is_never_evicted_and_incoming_fails_explicitly() throws Exception {
        WorkerBatcher batcher = endpointRegistry.getPrefill(PREFILL_IP_PORT).getBatcher();
        DecodeEndpoint decodeEp = endpointRegistry.getDecode(DECODE_IP_PORT);

        CompletableFuture<Response> victim = scheduler.submit(context(11, 50));
        await(() -> batcher.queueSize() == 1);

        Response response = scheduler.submit(context(12, 50)).get(2, TimeUnit.SECONDS);

        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.PRIORITY_ADMISSION_REJECTED.getErrorCode(),
                response.getCode());
        assertEquals(AdmissionRejectReason.SAME_PRIORITY_AHEAD,
                response.getAdmissionRejectReason());
        assertTrue(response.getErrorMessage().contains("same-priority requests are ahead"),
                "expected typed same-priority detail, got: " + response.getErrorMessage());
        // One route for the victim + one causally classified incoming attempt.
        verify(router, times(2)).route(any(BalanceContext.class));
        verify(priorityReporter)
                .reportEvictionPlan(eq(50), eq("prefill_queue_full"), eq("infeasible"));
        verify(priorityReporter, never()).reportVictim(anyInt(), anyInt(), anyString(), anyString());

        // The queued equal-priority request is untouched
        assertFalse(victim.isDone());
        assertEquals(1, batcher.queueSize());
        // Incoming's decode reservation rolled back; the victim's remains
        await(() -> decodeEp.getInflightCount() == 1);
    }

    // ==================== gate off: legacy retry-exhausted path ====================

    @Test
    void evict_switch_off_never_plans_and_fast_rejects_on_capacity() throws Exception {
        config.setAutoTpmPrefillQueueEvictEnabled(false);
        WorkerBatcher batcher = endpointRegistry.getPrefill(PREFILL_IP_PORT).getBatcher();
        DecodeEndpoint decodeEp = endpointRegistry.getDecode(DECODE_IP_PORT);

        CompletableFuture<Response> victim = scheduler.submit(context(21, 30));
        await(() -> batcher.queueSize() == 1);

        Response response = scheduler.submit(context(22, 70)).get(2, TimeUnit.SECONDS);

        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(), response.getCode());
        assertEquals(AdmissionRejectReason.RESOURCE_EXHAUSTED,
                response.getAdmissionRejectReason());
        // N3 §3.3 (default lockfree): a capacity failure is not transient —
        // primary + one fallback offer, then typed resource exhaustion.
        assertTrue(response.getErrorMessage().contains("admission capacity is temporarily exhausted"),
                "expected resource-exhaustion detail, got: " + response.getErrorMessage());
        // 1 victim route + primary + one fallback re-route for the incoming
        verify(router, times(3)).route(any(BalanceContext.class));
        verify(priorityReporter, never()).reportEvictionPlan(anyInt(), anyString(), anyString());

        assertFalse(victim.isDone());
        assertEquals(1, batcher.queueSize());
        await(() -> decodeEp.getInflightCount() == 1);
    }

    // ==================== victim termination is idempotent ====================

    @Test
    void finish_preempted_is_idempotent_and_releases_decode_once() throws Exception {
        DecodeEndpoint decodeEp = endpointRegistry.getDecode(DECODE_IP_PORT);
        decodeEp.reserve(77, 128, 136);
        BatchItem item = dummyItem(77);
        assertTrue(scheduler.registerInflight(item));

        scheduler.finishPreempted(item, "preempted by higher-priority request 88");
        scheduler.finishPreempted(item, "second call must be ignored");

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
    void finish_yielded_is_idempotent_and_releases_decode_once() throws Exception {
        DecodeEndpoint decodeEp = endpointRegistry.getDecode(DECODE_IP_PORT);
        decodeEp.reserve(78, 128, 136);
        BatchItem item = dummyItem(78);
        assertTrue(scheduler.registerInflight(item));

        scheduler.finishYielded(item, "yielded to higher-priority request 88");
        scheduler.finishYielded(item, "second call must be ignored");
        // A racing preempt terminal must not override the yielded terminal
        scheduler.finishPreempted(item, "late preempt must be ignored");

        Response response = item.future().get(1, TimeUnit.SECONDS);
        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(), response.getCode());
        assertTrue(response.getErrorMessage().contains("yielded to higher-priority request 88"));

        // Decode reservation released exactly once (no double release)
        assertEquals(0, decodeEp.getInflightCount());
        assertEquals(0, decodeEp.inflightHardKvReserved());
        assertEquals(0, decodeEp.getTotalLoad());
    }

    // ==================== 8.3: never a silent drop ====================

    @Test
    void expired_head_is_dispatched_not_dropped_when_auto_tpm_enabled() throws Exception {
        // Legacy would drop the head as deadline_expired with this SLO;
        // Auto-TPM must instead dispatch it via the deadline guard.
        config.setCostSloMs(1L);
        config.setFlexlbBatchSizeMax(2);

        Response response = scheduler.submit(context(61, 50)).get(2, TimeUnit.SECONDS);

        assertTrue(response.isSuccess());
        assertTrue(response.isEnqueuedByMaster());
        assertEquals(1, sentBatches.size());
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
        // Mirror production admission: set a ScheduleBudget so that
        // item.priority() / item.deadlineMs() delegate correctly.
        ctx.setBudget(ScheduleBudget.forDeadline(priority, ctx.getStartTime(), ctx.getStartTime() + 30_000));
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
