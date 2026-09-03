package org.flexlb.balance.scheduler.priority;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.scheduler.BatchDispatcher;
import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.balance.scheduler.DefaultBatchDispatcher;
import org.flexlb.balance.scheduler.PriorityScheduler;
import org.flexlb.balance.scheduler.RequestIdFixtures;
import org.flexlb.balance.scheduler.Router;
import org.flexlb.balance.scheduler.SchedulingTestConfig;
import org.flexlb.balance.scheduler.WorkerBatcher;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.VictimStage;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.AdmissionRejectReason;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.RequestId;
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
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Phase 3 tests for the prefill-queue eviction path of
 * {@link PriorityAdmissionScheduler} wired through
 * {@link PriorityScheduler#submit}: higher priority evicts strictly lower
 * priority on a full queue (the queued victim yields with the retryable
 * NO_AVAILABLE_WORKER, contract 5.3), equal priority never yields, admission
 * retries remain fenced, and victim termination is idempotent.
 */
class PriorityEvictionSchedulerTest {

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
    private final List<EngineRpcService.EnqueueBatchRequestPB> sentBatches = new CopyOnWriteArrayList<>();

    @BeforeEach
    void setUp() {
        configService = mock(ConfigService.class);
        router = mock(Router.class);
        grpcClient = mock(EngineGrpcClient.class);
        reporter = mock(BatchSchedulerReporter.class);
        priorityReporter = mock(PrioritySchedulerReporter.class);

        config = new FlexlbConfig();
        // Large batch + window + SLO keep queued items parked (no dispatch),
        // so eviction races with neither dispatch nor expiry.
        SchedulingTestConfig.useBatchDispatcher(config).setMaxRequests(100);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxCollectionWaitMs(10_000);
        SchedulingTestConfig.usePriorityQueue(config);
        SchedulingTestConfig.allowVictim(config, VictimStage.PREFILL_QUEUED);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxWaitingRequestsPerPrefillWorker(1);
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
        priorityScheduler = new PriorityAdmissionScheduler(
                configService, router, endpointRegistry, new PlanCommitter(),
                priorityReporter, reporter, new UnsupportedEngineCancelChannel());
        scheduler = new PriorityScheduler(configService, router,
                endpointRegistry, dispatcher, reporter, priorityScheduler, null,
                new UnsupportedEngineCancelChannel());

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
        priorityScheduler.shutdown();
        scheduler.shutdown();
    }

    // ==================== higher priority evicts lower on full queue ====================

    @Test
    void p70_evicts_queued_p30_victim_when_prefill_queue_full() throws Exception {
        WorkerBatcher batcher = endpointRegistry.getPrefill(PREFILL_IP_PORT).getBatcher();
        DecodeEndpoint decodeEp = endpointRegistry.getDecode(DECODE_IP_PORT);

        CompletableFuture<Response> victim = scheduler.submit(context("1", 30));
        await(() -> batcher.queueSize() == 1);

        CompletableFuture<Response> incoming = scheduler.submit(context("2", 70));

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

    @Test
    void telemetry_failure_does_not_interrupt_committed_victim_settlement() throws Exception {
        WorkerBatcher batcher = endpointRegistry.getPrefill(PREFILL_IP_PORT).getBatcher();
        DecodeEndpoint decodeEp = endpointRegistry.getDecode(DECODE_IP_PORT);

        // Admit two victims, then lower the live hard limit so the next
        // request must atomically replace both of them.
        SchedulingTestConfig.useBatchDispatcher(config).setMaxWaitingRequestsPerPrefillWorker(2);
        CompletableFuture<Response> firstVictim = scheduler.submit(context("3", 20));
        CompletableFuture<Response> secondVictim = scheduler.submit(context("4", 30));
        await(() -> batcher.queueSize() == 2);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxWaitingRequestsPerPrefillWorker(1);

        doThrow(new IllegalStateException("victim metrics unavailable"))
                .doNothing()
                .when(priorityReporter)
                .reportVictim(anyInt(), anyInt(), anyString(), anyString());
        doThrow(new IllegalStateException("commit metrics unavailable"))
                .when(priorityReporter)
                .reportEvictionCommit(eq(70), eq("prefill_queue_full"), eq("success"));

        CompletableFuture<Response> incoming = scheduler.submit(context("5", 70));

        Response firstResponse = firstVictim.get(2, TimeUnit.SECONDS);
        Response secondResponse = secondVictim.get(2, TimeUnit.SECONDS);
        assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(),
                firstResponse.getCode());
        assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(),
                secondResponse.getCode());

        // The replacement remains committed even though both observer paths
        // fail: every removed victim is settled and only the incoming Decode
        // reservation/queue member remains.
        await(() -> batcher.queueSize() == 1 && decodeEp.getInflightCount() == 1);
        assertFalse(incoming.isDone());
        verify(priorityReporter, times(2)).reportVictim(
                anyInt(), eq(70), eq("prefill_queued"), eq("prefill_queue_full"));
        verify(priorityReporter).reportEvictionCommit(
                eq(70), eq("prefill_queue_full"), eq("success"));
    }

    // ==================== equal priority never yields ====================

    @Test
    void equal_priority_is_never_evicted_and_incoming_fails_explicitly() throws Exception {
        WorkerBatcher batcher = endpointRegistry.getPrefill(PREFILL_IP_PORT).getBatcher();
        DecodeEndpoint decodeEp = endpointRegistry.getDecode(DECODE_IP_PORT);

        CompletableFuture<Response> victim = scheduler.submit(context("11", 50));
        await(() -> batcher.queueSize() == 1);

        Response response = scheduler.submit(context("12", 50)).get(2, TimeUnit.SECONDS);

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

    // ==================== preemption disabled ====================

    @Test
    void evict_switch_off_never_plans_and_fast_rejects_on_capacity() throws Exception {
        SchedulingTestConfig.disallowVictim(config, VictimStage.PREFILL_QUEUED);
        WorkerBatcher batcher = endpointRegistry.getPrefill(PREFILL_IP_PORT).getBatcher();
        DecodeEndpoint decodeEp = endpointRegistry.getDecode(DECODE_IP_PORT);

        CompletableFuture<Response> victim = scheduler.submit(context("21", 30));
        await(() -> batcher.queueSize() == 1);

        Response response = scheduler.submit(context("22", 70)).get(2, TimeUnit.SECONDS);

        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(), response.getCode());
        assertEquals(AdmissionRejectReason.RESOURCE_EXHAUSTED,
                response.getAdmissionRejectReason());
        // A capacity failure is not transient: primary + one fallback offer,
        // then typed resource exhaustion.
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
        decodeEp.reserve("77", 128, 136);
        BatchItem item = dummyItem("77");
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
        decodeEp.reserve("78", 128, 136);
        BatchItem item = dummyItem("78");
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

    private BatchItem dummyItem(String requestId) {
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
                .map(external -> Long.parseLong(RequestId.parse(external.getInput())))
                .forEach(requestId -> response.addSuccesses(
                        EngineRpcService.EnqueueBatchSuccessPB.newBuilder()
                                .setRequestId(requestId)
                                .build()));
        return response.build();
    }

    private static BalanceContext context(String requestId, int priority) {
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
        // Mirror production admission: bind normalized priority and expiry once.
        ctx.setSchedulingMetadata(SchedulingMetadata.explicit(
                priority, ctx.getStartTime() + 30_000));
        return ctx;
    }

    private static byte[] generateInputBytes(String requestId) {
        EngineRpcService.GenerateInputPB input = RequestIdFixtures.write(EngineRpcService.GenerateInputPB.newBuilder(), requestId)
                .addTokenIds(101)
                .addTokenIds(102)
                .setGenerateConfig(EngineRpcService.GenerateConfigPB.newBuilder()
                        .setMaxNewTokens(8)
                        .build())
                .build();
        return input.toByteArray();
    }

    private static Response successRoute(String requestId) {
        Response response = new Response();
        response.setSuccess(true);
        response.setServerStatus(List.of(
                server(RoleType.PREFILL, "10.0.0.1", 8080, 8081, requestId),
                server(RoleType.DECODE, "10.0.0.2", 8081, 8082, requestId)
        ));
        return response;
    }

    private static ServerStatus server(RoleType role, String ip, int httpPort, int grpcPort, String requestId) {
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
