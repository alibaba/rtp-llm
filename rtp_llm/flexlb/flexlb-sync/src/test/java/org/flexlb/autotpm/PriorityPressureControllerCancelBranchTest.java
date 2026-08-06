package org.flexlb.autotpm;

import io.grpc.Status;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.scheduler.InflightItem;
import org.flexlb.balance.scheduler.InflightStore;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.service.monitor.FlexlbMetricHelper;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.List;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Cancel-branch behaviour of {@link PriorityPressureController#tryPreempt}
 * (blueprint §1.7 response interpretation), with a mocked
 * {@link EngineGrpcClient} and mocked endpoint views:
 *
 * <ul>
 *   <li>found=true + release observed → {@link PreemptResult}, victim
 *       settled as AUTO_TPM_PREEMPTED (4290)</li>
 *   <li>found=false (race) → empty, rate-limit permit rolled back, victim
 *       untouched</li>
 *   <li>UNIMPLEMENTED → endpoint degraded, no further Cancel sent</li>
 *   <li>cancel RPC timeout / transport failure → empty, never optimistic</li>
 * </ul>
 */
class PriorityPressureControllerCancelBranchTest {

    private static final String EP = "10.0.0.1:8080";
    private static final String EP_IP = "10.0.0.1";
    private static final int EP_GRPC_PORT = 8081;
    private static final long VICTIM_ID = 100L;
    private static final int VICTIM_PRIORITY = 30;
    private static final long INCOMING_ID = 200L;
    private static final int INCOMING_PRIORITY = 70;
    /** Short bounded-wait budget keeps the timeout branches fast. */
    private static final long WAIT_TIMEOUT_MS = 30;

    private ConfigService configService;
    private EndpointRegistry endpointRegistry;
    private EngineGrpcClient grpcClient;
    private InflightStore inflightStore;
    private FlexlbMetricHelper metricHelper;
    private DecodeEndpoint decodeEp;
    private FlexlbConfig config;
    private PriorityPressureController controller;

    private CompletableFuture<Response> victimFuture;
    private InflightItem victimItem;

    @BeforeEach
    void setUp() {
        configService = mock(ConfigService.class);
        endpointRegistry = mock(EndpointRegistry.class);
        grpcClient = mock(EngineGrpcClient.class);
        inflightStore = mock(InflightStore.class);
        metricHelper = mock(FlexlbMetricHelper.class);
        decodeEp = mock(DecodeEndpoint.class);

        config = enabledConfig();
        when(configService.loadBalanceConfig()).thenReturn(config);

        ConcurrentHashMap<String, DecodeEndpoint> decodeEndpoints = new ConcurrentHashMap<>();
        decodeEndpoints.put(EP, decodeEp);
        when(endpointRegistry.getDecodeEndpoints()).thenReturn(decodeEndpoints);
        when(endpointRegistry.getDecode(EP)).thenReturn(decodeEp);
        when(decodeEp.getIp()).thenReturn(EP_IP);
        when(decodeEp.getGrpcPort()).thenReturn(EP_GRPC_PORT);
        when(decodeEp.snapshotRunningCandidates(any())).thenReturn(List.of(victimCandidate()));
        when(grpcClient.isCancelSupported(EP_IP, EP_GRPC_PORT)).thenReturn(true);

        victimFuture = new CompletableFuture<>();
        victimItem = new InflightItem(context(VICTIM_ID, VICTIM_PRIORITY), victimFuture, null);
        when(inflightStore.get(String.valueOf(VICTIM_ID))).thenReturn(victimItem);

        controller = new PriorityPressureController(configService, endpointRegistry, grpcClient,
                inflightStore, new PriorityRegistry(), metricHelper);
    }

    // ---- a) found=true + engine task disappears → PreemptResult, victim settled 4290 ----

    @Test
    void tryPreempt_foundTrue_releaseObserved_returnsResultAndSettlesVictim() {
        stubCancelResponse(cancelResponse(true, false));
        // release confirmed after two poll rounds (entry disappears from layer 2)
        when(decodeEp.hasEngineTask(VICTIM_ID)).thenReturn(true, true, false);

        Optional<PreemptResult> result = controller.tryPreempt(context(INCOMING_ID, INCOMING_PRIORITY));

        assertTrue(result.isPresent());
        assertEquals(EP, result.get().endpoint());
        assertEquals(VICTIM_ID, result.get().victimRequestId());
        assertEquals(VICTIM_PRIORITY, result.get().victimPriority());

        // victim settled with structured attribution AUTO_TPM_PREEMPTED (4290)
        assertTrue(victimItem.isTerminated());
        Response settled = victimFuture.join();
        assertFalse(settled.isSuccess());
        assertEquals(4290, settled.getCode());
        // confirmed cancel leaves no pending intent behind
        assertFalse(controller.hasCancelIntent(VICTIM_ID));
        verify(metricHelper).reportAutoTpmRunningCancel(VICTIM_PRIORITY, INCOMING_PRIORITY,
                PriorityPressureController.RESULT_SUCCESS);
    }

    // ---- b) found=false → empty, permit rolled back, victim untouched ----

    @Test
    void tryPreempt_foundFalse_returnsEmpty_rollsBackPermit_victimNotSettled() {
        // one preemption per minute: only a rollback makes the second acquire possible
        config.setAutoTpmPreemptRateLimitPerMin(1);
        stubCancelResponse(cancelResponse(false, false));

        Optional<PreemptResult> first = controller.tryPreempt(context(INCOMING_ID, INCOMING_PRIORITY));

        assertTrue(first.isEmpty());
        assertFalse(victimItem.isTerminated(), "found=false must not settle the victim");
        assertFalse(victimFuture.isDone());
        assertFalse(controller.hasCancelIntent(VICTIM_ID));
        verify(metricHelper).reportAutoTpmRunningCancel(VICTIM_PRIORITY, INCOMING_PRIORITY,
                PriorityPressureController.RESULT_NOT_FOUND);

        // permit rolled back: the same limiter admits the retry, which now succeeds
        stubCancelResponse(cancelResponse(true, false));
        when(decodeEp.hasEngineTask(VICTIM_ID)).thenReturn(false);

        Optional<PreemptResult> second = controller.tryPreempt(context(INCOMING_ID, INCOMING_PRIORITY));

        assertTrue(second.isPresent(), "rollback must return the permit for the next attempt");
        verify(metricHelper, never()).reportAutoTpmRunningCancel(VICTIM_PRIORITY, INCOMING_PRIORITY,
                PriorityPressureController.RESULT_RATE_LIMITED);
    }

    // ---- c) UNIMPLEMENTED → endpoint degraded, no further Cancel ----

    @Test
    void tryPreempt_unimplemented_degradesEndpoint_noFurtherCancelSent() {
        // Contract of EngineGrpcClient probe-degrade: the UNIMPLEMENTED answer
        // flips isCancelSupported to false for the endpoint.
        AtomicBoolean supported = new AtomicBoolean(true);
        when(grpcClient.isCancelSupported(EP_IP, EP_GRPC_PORT)).thenAnswer(inv -> supported.get());
        when(grpcClient.cancelAsync(eq(EP_IP), eq(EP_GRPC_PORT), eq(VICTIM_ID),
                eq(EngineRpcService.EngineCancelReasonPB.ENGINE_CANCEL_REASON_PRIORITY_PREEMPTED), anyLong()))
                .thenAnswer(inv -> {
                    supported.set(false);
                    CompletableFuture<EngineRpcService.CancelResponsePB> failed = new CompletableFuture<>();
                    failed.completeExceptionally(Status.UNIMPLEMENTED
                            .withDescription("engine without Cancel RPC").asRuntimeException());
                    return failed;
                });

        Optional<PreemptResult> first = controller.tryPreempt(context(INCOMING_ID, INCOMING_PRIORITY));

        assertTrue(first.isEmpty());
        assertFalse(victimItem.isTerminated());
        assertFalse(controller.hasCancelIntent(VICTIM_ID));
        verify(metricHelper).reportAutoTpmRunningCancel(VICTIM_PRIORITY, INCOMING_PRIORITY,
                PriorityPressureController.RESULT_UNSUPPORTED);

        // degraded endpoint: the next attempt bails out before any Cancel RPC
        Optional<PreemptResult> second = controller.tryPreempt(context(INCOMING_ID, INCOMING_PRIORITY));

        assertTrue(second.isEmpty());
        verify(grpcClient, times(1)).cancelAsync(any(), Mockito.anyInt(), anyLong(), any(), anyLong());
        verify(metricHelper, times(2)).reportAutoTpmRunningCancel(VICTIM_PRIORITY, INCOMING_PRIORITY,
                PriorityPressureController.RESULT_UNSUPPORTED);
    }

    // ---- d) cancel RPC timeout / transport failure → empty, never optimistic ----

    @Test
    void tryPreempt_cancelRpcTimeout_returnsEmpty_keepsIntent_noOptimisticDispatch() {
        // never-completing future → controller-side .get() times out
        stubCancelResponse(null);

        Optional<PreemptResult> result = controller.tryPreempt(context(INCOMING_ID, INCOMING_PRIORITY));

        assertTrue(result.isEmpty(), "RPC timeout must not grant capacity optimistically");
        assertFalse(victimItem.isTerminated());
        assertFalse(victimFuture.isDone());
        // the cancel may still land engine-side: intent (and permit) kept for
        // the WorkerStatus close-out path
        assertTrue(controller.hasCancelIntent(VICTIM_ID));
        verify(metricHelper).reportAutoTpmRunningCancel(VICTIM_PRIORITY, INCOMING_PRIORITY,
                PriorityPressureController.RESULT_TIMEOUT);
    }

    @Test
    void tryPreempt_cancelRpcTransportFailure_returnsEmpty_victimNotSettled() {
        CompletableFuture<EngineRpcService.CancelResponsePB> failed = new CompletableFuture<>();
        failed.completeExceptionally(Status.UNAVAILABLE
                .withDescription("connection refused").asRuntimeException());
        stubCancelResponse(failed);

        Optional<PreemptResult> result = controller.tryPreempt(context(INCOMING_ID, INCOMING_PRIORITY));

        assertTrue(result.isEmpty());
        assertFalse(victimItem.isTerminated());
        // transport failure: the cancel never reached the engine → intent dropped
        assertFalse(controller.hasCancelIntent(VICTIM_ID));
        verify(metricHelper).reportAutoTpmRunningCancel(VICTIM_PRIORITY, INCOMING_PRIORITY,
                PriorityPressureController.RESULT_NOT_FOUND);
    }

    @Test
    void tryPreempt_foundTrue_butReleaseWaitTimesOut_returnsEmpty_keepsIntent() {
        stubCancelResponse(cancelResponse(true, false));
        // the layer-2 entry never disappears within the bounded wait
        when(decodeEp.hasEngineTask(VICTIM_ID)).thenReturn(true);

        Optional<PreemptResult> result = controller.tryPreempt(context(INCOMING_ID, INCOMING_PRIORITY));

        assertTrue(result.isEmpty(), "unconfirmed release must not grant capacity");
        assertFalse(victimItem.isTerminated(),
                "settlement belongs to the WorkerStatus close-out path after a wait timeout");
        assertTrue(controller.hasCancelIntent(VICTIM_ID));
        verify(metricHelper).reportAutoTpmRunningCancel(VICTIM_PRIORITY, INCOMING_PRIORITY,
                PriorityPressureController.RESULT_TIMEOUT);
    }

    // ==================== fixtures ====================

    private static FlexlbConfig enabledConfig() {
        FlexlbConfig config = new FlexlbConfig();
        config.setAutoTpmEnabled(true);
        config.setAutoTpmDecodeRunningPreemptEnabled(true);
        config.setAutoTpmPreemptRateLimitPerMin(10);
        config.setAutoTpmEndpointPreemptQpsLimit(0);
        config.setAutoTpmCommitWaitReleaseTimeoutMs(WAIT_TIMEOUT_MS);
        config.setAutoTpmPreemptCriticalSectionMs(0);
        return config;
    }

    private static VictimCandidate victimCandidate() {
        return new VictimCandidate(VICTIM_ID, VICTIM_PRIORITY, 5, 500,
                System.currentTimeMillis() - 60_000, EP);
    }

    private static BalanceContext context(long requestId, int priority) {
        Request request = new Request();
        request.setRequestId(requestId);
        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(request);
        ctx.setPriority(priority);
        return ctx;
    }

    private static CompletableFuture<EngineRpcService.CancelResponsePB> cancelResponse(
            boolean found, boolean alreadyFinished) {
        return CompletableFuture.completedFuture(EngineRpcService.CancelResponsePB.newBuilder()
                .setFound(found)
                .setAlreadyFinished(alreadyFinished)
                .build());
    }

    /** Stub the structured cancel RPC; {@code null} → never-completing future. */
    private void stubCancelResponse(CompletableFuture<EngineRpcService.CancelResponsePB> future) {
        when(grpcClient.cancelAsync(eq(EP_IP), eq(EP_GRPC_PORT), eq(VICTIM_ID),
                eq(EngineRpcService.EngineCancelReasonPB.ENGINE_CANCEL_REASON_PRIORITY_PREEMPTED), anyLong()))
                .thenReturn(future != null ? future : new CompletableFuture<>());
    }
}
