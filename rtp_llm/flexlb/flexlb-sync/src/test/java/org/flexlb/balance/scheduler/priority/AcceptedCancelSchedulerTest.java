package org.flexlb.balance.scheduler.priority;

import org.flexlb.balance.scheduler.priority.EngineCancelChannel.CancelReason;
import org.flexlb.balance.scheduler.priority.EngineCancelChannel.CancelTarget;
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
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.enums.TaskPhase;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.PrioritySchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.BooleanSupplier;
import java.util.function.LongFunction;

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
 * Phase 5 tests for the accepted-eviction cancel-wait-confirm commit of
 * {@link PriorityAdmissionScheduler} and the CANCELLED completion attribution
 * of {@link FlexlbBatchScheduler}: a confirmed release lets the incoming take
 * the freed capacity, a timeout fails the plan without reserving or leaking
 * (iron rule 4), ACCEPTED and FAILED outcomes both wait for an explicit
 * release record (the ack is intent registration only), the accepted-evict
 * gate keeps the legacy infeasible failure, and an engine CANCELLED completion
 * maps to PRIORITY_PREEMPTED only for marked victims with Auto-TPM on (iron
 * rule 1).
 */
class AcceptedCancelSchedulerTest {

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
    private FakeCancelChannel cancelChannel;
    private WorkerStatus decodeWs;

    @BeforeEach
    void setUp() {
        // ReleaseTracker.global() is a process-wide singleton — drop releases
        // cached by earlier tests so no stale observation confirms an eviction.
        ReleaseTracker.global().reset();
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
        config.setAutoTpmDecodeAcceptedEvictEnabled(true);
        // Single decode slot: the second admission always hits slot-full.
        config.setDecodeConcurrencyLimit(1);
        config.setFlexlbBatchQueueMaxSize(4);
        when(configService.loadBalanceConfig()).thenReturn(config);

        when(router.route(any(BalanceContext.class))).thenAnswer(inv -> routeAnswer(inv.getArgument(0)));
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(),
                any(EngineRpcService.EnqueueBatchRequestPB.class), anyLong()))
                .thenAnswer(inv -> CompletableFuture.completedFuture(ackFor(inv.getArgument(2))));

        endpointRegistry = new EndpointRegistry(configService, () -> scheduler, reporter);
        BatchDispatcher dispatcher = new DefaultBatchDispatcher(grpcClient, configService, null);
        cancelChannel = new FakeCancelChannel();
        // Test seam: after a decode eviction the prefill is picked manually
        // (the static strategy factory is not populated in unit tests).
        PriorityAdmissionScheduler priorityScheduler = new PriorityAdmissionScheduler(
                configService, router, endpointRegistry, new PlanCommitter(),
                new PrioritySloPolicy(PrioritySloPolicy.DEFAULT_SLO_LENGTH_BUCKETS,
                        PrioritySloPolicy.DEFAULT_PRIORITY_SLO_MULTIPLIERS),
                priorityReporter, reporter, cancelChannel) {
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

        decodeWs = new WorkerStatus();
        decodeWs.setIp("10.0.0.2");
        decodeWs.setPort(8081);
        decodeWs.setGrpcPort(8082);
        decodeWs.setAvailableKvCacheTokens(new AtomicLong(1_000_000L));
        decodeWs.setTotalKvCacheTokens(new AtomicLong(2_000_000L));
        endpointRegistry.ensureEndpoint(RoleType.DECODE, DECODE_IP_PORT, decodeWs);
        calibrateDecode(null);
    }

    @AfterEach
    void tearDown() {
        scheduler.shutdown();
        ReleaseTracker.global().reset();
    }

    // ==================== success: cancel confirmed within the window ====================

    @Test
    void acceptedVictim_cancelConfirmed_evictsAndReservesIncoming() throws Exception {
        DecodeEndpoint decodeEp = decodeEndpoint();
        CompletableFuture<Response> victim = submitAndConfirmAccepted(1, 30);

        // The engine "processes" the cancel: the next WorkerStatus report
        // carries an explicit finished CANCELLED record — the sole release
        // confirmation (§16.4) — fed to ReleaseTracker.global() by the
        // decode calibrate, driven synchronously from the fake RPC.
        cancelChannel.onCancel = id -> {
            calibrateDecode(Map.of(), Map.of(String.valueOf(id), cancelledFinished(id)));
            return EngineCancelChannel.CancelOutcome.accepted();
        };

        CompletableFuture<Response> incoming = scheduler.submit(context(2, 70));

        Response victimResponse = victim.get(2, TimeUnit.SECONDS);
        assertFalse(victimResponse.isSuccess());
        assertEquals(StrategyErrorType.PRIORITY_PREEMPTED.getErrorCode(), victimResponse.getCode());
        assertTrue(victimResponse.getErrorMessage().contains("preempted by higher-priority request 2"));

        // Only after the release confirmation may the incoming take the slot.
        await(() -> decodeEp.reservedView().containsKey(2L));
        assertEquals(70, decodeEp.reservedView().get(2L).priority());
        assertFalse(decodeEp.isConfirmedTracked(1L));
        assertFalse(incoming.isDone());
        assertEquals(1, cancelChannel.cancelCount.get());

        verify(priorityReporter).reportEvictionPlan(eq(70), eq("decode_slot_full"), eq("feasible"));
        verify(priorityReporter).reportCancelRequest(eq(DECODE_IP_PORT), eq(30));
        verify(priorityReporter).reportCancel(eq(30), eq("PRIORITY_PREEMPTED"));
        verify(priorityReporter).reportCancelConfirm(eq(DECODE_IP_PORT), eq(30));
        verify(priorityReporter).reportVictim(eq(30), eq(70), eq("decode_accepted"), eq("decode_slot_full"));
        verify(priorityReporter).reportVictimKvTokens(eq(30), eq("decode_accepted"), eq(128L));
        verify(priorityReporter).reportEvictionCommit(eq(70), eq("decode_slot_full"), eq("success"));
        verify(priorityReporter, never()).reportCancelTimeout(anyString(), anyInt());
    }

    // ==================== timeout: never assume the resources are free ====================

    @Test
    void cancelTimeout_failsIncoming_withoutReservingOrLeaking() throws Exception {
        config.setAutoTpmCommitWaitReleaseTimeoutMs(40);
        DecodeEndpoint decodeEp = decodeEndpoint();
        CompletableFuture<Response> victim = submitAndConfirmAccepted(1, 30);

        // Cancel accepted by the engine but no release confirmation follows.
        cancelChannel.onCancel = id ->
                EngineCancelChannel.CancelOutcome.accepted();

        Response incoming = scheduler.submit(context(2, 70)).get(2, TimeUnit.SECONDS);

        assertFalse(incoming.isSuccess());
        assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(), incoming.getCode());
        assertTrue(incoming.getErrorMessage().contains("accepted eviction cancel_timeout"));

        // Iron rule 4: the incoming never reserved, no accounting leak.
        assertFalse(decodeEp.reservedView().containsKey(2L));
        assertEquals(0, decodeEp.getInflightCount());
        assertEquals(0, decodeEp.inflightHardKvReserved());
        // The victim keeps running with the CANCEL_REQUESTED mark: still
        // tracked, but no longer offered as an eviction candidate.
        assertFalse(victim.isDone());
        assertTrue(decodeEp.isConfirmedTracked(1L));
        assertTrue(DecodeEndpointSnapshot.capture(decodeEp, 1).accepted().isEmpty());

        verify(priorityReporter).reportCancelRequest(eq(DECODE_IP_PORT), eq(30));
        verify(priorityReporter).reportCancelTimeout(eq(DECODE_IP_PORT), eq(70));
        verify(priorityReporter).reportEvictionCommit(eq(70), eq("decode_slot_full"), eq("cancel_timeout"));
        verify(priorityReporter, never()).reportVictim(anyInt(), anyInt(),
                eq("decode_accepted"), anyString());
        verify(priorityReporter, never()).reportCancelConfirm(anyString(), anyInt());
    }

    // ==================== outcome branches: unsupported / failed ====================

    @Test
    void unsupportedOutcome_failsPlanWithCancelUnsupported() throws Exception {
        submitAndConfirmAccepted(1, 30);
        cancelChannel.onCancel = id -> EngineCancelChannel.CancelOutcome.unsupported();

        Response incoming = scheduler.submit(context(2, 70)).get(2, TimeUnit.SECONDS);

        assertFalse(incoming.isSuccess());
        assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(), incoming.getCode());
        assertTrue(incoming.getErrorMessage().contains("accepted eviction cancel_unsupported"));
        assertFalse(decodeEndpoint().reservedView().containsKey(2L));
        verify(priorityReporter).reportEvictionCommit(eq(70), eq("decode_slot_full"),
                eq("cancel_unsupported"));
    }

    @Test
    void failedOutcome_waitsForReleaseRecord_failsWithoutObservation() throws Exception {
        config.setAutoTpmCommitWaitReleaseTimeoutMs(40);
        DecodeEndpoint decodeEp = decodeEndpoint();
        CompletableFuture<Response> victim = submitAndConfirmAccepted(1, 30);
        cancelChannel.onCancel = id -> EngineCancelChannel.CancelOutcome.failed();

        Response incoming = scheduler.submit(context(2, 70)).get(2, TimeUnit.SECONDS);

        // A FAILED ack is never release proof (nor cancel-failure proof) —
        // without an explicit release record from WorkerStatus the eviction
        // fails like a cancel timeout, the incoming never reserves and the
        // victim keeps running.
        assertFalse(incoming.isSuccess());
        assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(), incoming.getCode());
        assertTrue(incoming.getErrorMessage().contains("accepted eviction cancel_timeout"));
        assertFalse(decodeEp.reservedView().containsKey(2L));
        assertFalse(victim.isDone());
        assertTrue(decodeEp.isConfirmedTracked(1L));
        verify(priorityReporter).reportCancelTimeout(eq(DECODE_IP_PORT), eq(70));
        verify(priorityReporter, never()).reportCancelConfirm(anyString(), anyInt());
    }

    @Test
    void failedOutcome_withExplicitReleaseRecord_evicts() throws Exception {
        DecodeEndpoint decodeEp = decodeEndpoint();
        CompletableFuture<Response> victim = submitAndConfirmAccepted(1, 30);
        // FAILED paired with an explicit finished record in the same status
        // round (the RPC failed but the cancel still landed engine-side): the
        // ReleaseTracker observation — not the ack — confirms.
        cancelChannel.onCancel = id -> {
            calibrateDecode(Map.of(), Map.of(String.valueOf(id), cancelledFinished(id)));
            return EngineCancelChannel.CancelOutcome.failed();
        };

        scheduler.submit(context(2, 70));

        Response victimResponse = victim.get(2, TimeUnit.SECONDS);
        assertEquals(StrategyErrorType.PRIORITY_PREEMPTED.getErrorCode(), victimResponse.getCode());
        await(() -> decodeEp.reservedView().containsKey(2L));
        verify(priorityReporter).reportCancelConfirm(eq(DECODE_IP_PORT), eq(30));
        verify(priorityReporter, never()).reportCancelTimeout(anyString(), anyInt());
    }

    // ==================== gate off: accepted layer never cancelled ====================

    @Test
    void acceptedEvictGateOff_neverCancels_keepsInfeasibleFailure() throws Exception {
        config.setAutoTpmDecodeAcceptedEvictEnabled(false);
        DecodeEndpoint decodeEp = decodeEndpoint();
        CompletableFuture<Response> victim = submitAndConfirmAccepted(1, 30);

        Response incoming = scheduler.submit(context(2, 70)).get(2, TimeUnit.SECONDS);

        assertFalse(incoming.isSuccess());
        assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(), incoming.getCode());
        // Redesign C-2: infeasible is an ordinary capacity failure — the full
        // retry budget is consumed, then a reason-tagged exhaustion surfaces.
        assertTrue(incoming.getErrorMessage().contains("reason=capacity_no_evict_candidates"),
                "expected reason-tagged exhaustion, got: " + incoming.getErrorMessage());
        assertEquals(0, cancelChannel.cancelCount.get());
        assertFalse(victim.isDone());
        assertTrue(decodeEp.isConfirmedTracked(1L));
        verify(priorityReporter, times(3))
                .reportEvictionPlan(eq(70), eq("decode_slot_full"), eq("infeasible"));
        verify(priorityReporter, never()).reportCancelRequest(anyString(), anyInt());
    }

    // ==================== CANCELLED completion attribution (iron rule 1) ====================

    @Test
    void cancelledCompletion_attributedToPriorityPreempted_whenMarked() throws Exception {
        BatchItem item = dummyItem(77);
        assertTrue(scheduler.registerInflight(item));
        assertTrue(scheduler.markCancelRequested(77, "preempted by higher-priority request 88"));
        assertFalse(scheduler.markCancelRequested(999, "unknown id is a no-op"));

        scheduler.onWorkerStatusUpdate(decodeCancelledCompletion(77));

        Response response = item.future().get(1, TimeUnit.SECONDS);
        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.PRIORITY_PREEMPTED.getErrorCode(), response.getCode());
        assertTrue(response.getErrorMessage().contains("preempted by higher-priority request 88"));
        // Late confirm counted against the victim's decode endpoint; the
        // dummy item carries no Auto-TPM budget, so its priority tag is 0.
        verify(priorityReporter).reportCancelConfirm(eq(DECODE_IP_PORT), eq(0));
        verify(priorityReporter).reportPriorityPreempt(eq("decode_accepted"));
    }

    @Test
    void cancelledCompletion_keepsWorkerFailure_whenAutoTpmOff() throws Exception {
        config.setAutoTpmEnabled(false);
        BatchItem item = dummyItem(78);
        assertTrue(scheduler.registerInflight(item));
        assertTrue(scheduler.markCancelRequested(78, "stale mark"));

        scheduler.onWorkerStatusUpdate(decodeCancelledCompletion(78));

        Response response = item.future().get(1, TimeUnit.SECONDS);
        assertEquals(StrategyErrorType.WORKER_EXECUTION_FAILED.getErrorCode(), response.getCode());
        assertTrue(response.getErrorMessage().contains("worker error code 2"));
        verify(priorityReporter, never()).reportCancelConfirm(anyString(), anyInt());
    }

    @Test
    void cancelledCompletion_keepsWorkerFailure_whenUnmarked() throws Exception {
        BatchItem item = dummyItem(79);
        assertTrue(scheduler.registerInflight(item));

        scheduler.onWorkerStatusUpdate(decodeCancelledCompletion(79));

        Response response = item.future().get(1, TimeUnit.SECONDS);
        assertEquals(StrategyErrorType.WORKER_EXECUTION_FAILED.getErrorCode(), response.getCode());
        verify(priorityReporter, never()).reportCancelConfirm(anyString(), anyInt());
    }

    // ==================== helpers ====================

    /**
     * Programmable cancel channel: every endpoint supported, each cancel is
     * answered synchronously by the current {@link #onCancel} handler (which
     * may drive a calibrate to simulate the engine releasing the request).
     */
    private static final class FakeCancelChannel implements EngineCancelChannel {

        volatile LongFunction<CancelOutcome> onCancel =
                id -> CancelOutcome.accepted();
        final AtomicInteger cancelCount = new AtomicInteger();

        @Override
        public boolean isSupported(DecodeEndpoint endpoint) {
            return true;
        }

        @Override
        public CompletableFuture<CancelOutcome> cancel(CancelTarget target,
                                                       long requestId,
                                                       CancelReason reason) {
            cancelCount.incrementAndGet();
            return CompletableFuture.completedFuture(onCancel.apply(requestId));
        }
    }

    /**
     * Submit a request, wait for its decode reservation, then confirm it into
     * the accepted layer via a calibrate reporting KV_ALLOCATED (inputLength
     * 128 becomes the tracked KV estimate).
     */
    private CompletableFuture<Response> submitAndConfirmAccepted(long requestId, int priority)
            throws InterruptedException {
        DecodeEndpoint decodeEp = decodeEndpoint();
        CompletableFuture<Response> future = scheduler.submit(context(requestId, priority));
        await(() -> decodeEp.reservedView().containsKey(requestId));

        TaskInfo task = new TaskInfo();
        task.setRequestId(requestId);
        task.setPhase(TaskPhase.KV_ALLOCATED);
        task.setInputLength(128);
        calibrateDecode(Map.of(String.valueOf(requestId), task));
        assertEquals(1, decodeEp.getAcceptedLayerCount());
        assertEquals(1, decodeEp.getTotalLoad());
        return future;
    }

    private void calibrateDecode(Map<String, TaskInfo> running) {
        calibrateDecode(running, null);
    }

    private void calibrateDecode(Map<String, TaskInfo> running, Map<String, TaskInfo> finished) {
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRunningTaskInfo(running);
        response.setFinishedTaskInfo(finished);
        decodeEndpoint().onWorkerStatusUpdate(decodeWs, response);
    }

    /** Finished CANCELLED record (errorCode 2) — the explicit release proof. */
    private static TaskInfo cancelledFinished(long requestId) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(requestId);
        task.setErrorCode(2);
        task.setErrorMessage("cancelled");
        return task;
    }

    private DecodeEndpoint decodeEndpoint() {
        return endpointRegistry.getDecode(DECODE_IP_PORT);
    }

    /** Decode-side WorkerStatus completion reporting CANCELLED (errorCode 2). */
    private static WorkerStatusResponse decodeCancelledCompletion(long requestId) {
        TaskInfo cancelled = new TaskInfo();
        cancelled.setRequestId(requestId);
        cancelled.setErrorCode(2);
        cancelled.setErrorMessage("cancelled");
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(RoleType.DECODE);
        response.setFinishedTaskInfo(Map.of(String.valueOf(requestId), cancelled));
        return response;
    }

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

    private Response routeAnswer(BalanceContext ctx) {
        DecodeEndpoint decodeEp = decodeEndpoint();
        if (decodeEp.getTotalLoad() + 1 > config.getDecodeConcurrencyLimit()) {
            return Response.error(StrategyErrorType.NO_DECODE_WORKER);
        }
        decodeEp.reserve(ctx.getRequestId(), 128, 136, ctx.getPriority(), ctx.getDeadlineMs());
        return successRoute(ctx.getRequestId());
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
        status.setEndpointGeneration(1);
        return status;
    }
}
