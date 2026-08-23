package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.scheduler.priority.AdmissionLease;
import org.flexlb.balance.scheduler.priority.EngineCancelChannel;
import org.flexlb.balance.scheduler.priority.InflightRegistrar.PostDeliveryFenceResult;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.TaskPhase;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.flexlb.balance.endpoint.DecodeEndpoint.EngineDispatchPermitTransferStatus.TRANSFERRED;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.timeout;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/** Delivery-ownership tests for cancellation of the externally returned future. */
class ExternalFutureCancellationTest {

    private static final String PREFILL_IP_PORT = "10.0.0.1:8080";
    private static final String DECODE_IP_PORT = "10.0.0.2:8081";

    private PriorityScheduler scheduler;
    private EndpointRegistry endpointRegistry;
    private PrefillEndpoint prefillEndpoint;
    private DecodeEndpoint decodeEndpoint;
    private CapturingBatchDispatcher batchDispatcher;
    private HoldingRouteDecisionDelivery routeDelivery;
    private EngineCancelChannel cancelChannel;
    private CompletableFuture<EngineCancelChannel.CancelOutcome> cancelResult;
    private FlexlbConfig config;

    @BeforeEach
    void setUp() {
        ConfigService configService = mock(ConfigService.class);
        Router router = mock(Router.class);
        BatchSchedulerReporter reporter = mock(BatchSchedulerReporter.class);
        config = new FlexlbConfig();
        SchedulingTestConfig.usePriorityQueue(config);
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxRequests(100);
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxCollectionWaitMs(60_000);
        SchedulingTestConfig.useNonBatchDispatcher(config).setMaxInflightRequestsPerPrefillWorker(100);
        config.getRouter().getRoles().getDecode().getAvailability().setMaxEngineRequests((long) (100));
        when(configService.loadBalanceConfig()).thenReturn(config);

        batchDispatcher = new CapturingBatchDispatcher();
        routeDelivery = new HoldingRouteDecisionDelivery();
        cancelChannel = mock(EngineCancelChannel.class);
        cancelResult = new CompletableFuture<>();
        when(cancelChannel.cancel(any(), anyLong(), anyLong())).thenReturn(cancelResult);

        endpointRegistry = new EndpointRegistry(configService, () -> scheduler, reporter);
        scheduler = new PriorityScheduler(configService, router, endpointRegistry,
                batchDispatcher, reporter, null, null, cancelChannel,
                new PriorityScheduler.EngineFencePolicy(1, 1, 1, 1),
                routeDelivery);

        WorkerStatus prefill = new WorkerStatus();
        prefill.setIp("10.0.0.1");
        prefill.setPort(8080);
        prefill.setGrpcPort(8081);
        prefill.setAlive(true);
        endpointRegistry.ensureEndpoint(RoleType.PREFILL, PREFILL_IP_PORT, prefill);
        prefillEndpoint = endpointRegistry.getPrefill(PREFILL_IP_PORT);

        WorkerStatus decode = new WorkerStatus();
        decode.setIp("10.0.0.2");
        decode.setPort(8081);
        decode.setGrpcPort(8082);
        decode.setAlive(true);
        decode.setAvailableKvCacheTokens(new AtomicLong(1_000_000));
        decode.setTotalKvCacheTokens(new AtomicLong(2_000_000));
        endpointRegistry.ensureEndpoint(RoleType.DECODE, DECODE_IP_PORT, decode);
        decodeEndpoint = endpointRegistry.getDecode(DECODE_IP_PORT);
    }

    @AfterEach
    void tearDown() {
        scheduler.shutdown();
    }

    @Test
    void batchCancellationBetweenSendAndAckRetainsLedgersUntilEngineTombstone() {
        long requestId = 10_001L;
        BatchItem item = admittedItem(requestId, DeliveryMode.BATCH_ENQUEUE);

        deliverAdmitted(List.of(item), new DecisionGroupMetadata("test", 0));

        assertTrue(batchDispatcher.wasSent());
        long batchId = batchDispatcher.batchId;
        assertEquals(RequestLifecycleState.DISPATCHING,
                scheduler.getRequestState(requestId, batchId).state());
        assertEquals(1, prefillEndpoint.getInflightBatchCount());

        assertTrue(item.future().cancel(false));

        verify(cancelChannel, timeout(1_000)).cancel(any(), eq(requestId), anyLong());
        assertEquals(1, scheduler.getInflightSize());
        assertEquals(1, prefillEndpoint.getInflightBatchCount());
        assertTrue(decodeEndpoint.reservedView().containsKey(requestId));

        // Protection won the batch-key race. Even a concurrent ordinary
        // WorkerStatus terminal must leave the Prefill capacity gate charged
        // until the Engine fence reaches an authoritative outcome.
        prefillEndpoint.onWorkerStatusUpdate(
                prefillEndpoint.getStatus(), prefillFinished(requestId, batchId));
        assertEquals(1, prefillEndpoint.getInflightBatchCount());

        // A late positive ACK is not allowed to escape the cancellation fence.
        batchDispatcher.callback.onSuccess(item, batchId);
        assertEquals(RequestLifecycleState.DISPATCHING,
                scheduler.getRequestState(requestId, batchId).state());
        assertTrue(item.future().isCancelled());

        cancelResult.complete(EngineCancelChannel.CancelOutcome.tombstoned());

        assertEquals(0, scheduler.getInflightSize());
        assertEquals(0, prefillEndpoint.getInflightBatchCount());
        assertFalse(decodeEndpoint.reservedView().containsKey(requestId));
        assertEquals(RequestLifecycleState.TIMED_OUT,
                scheduler.getRequestState(requestId, batchId).state());
    }

    @Test
    void batchFutureCancelAfterSendDoesNotDependOnAdmissionLease() {
        long requestId = 10_115L;
        BatchItem item = admittedItemWithoutRegistration(requestId, DeliveryMode.BATCH_ENQUEUE);
        assertTrue(scheduler.registerInflight(item));
        decodeEndpoint.reserve(requestId, 128, 136, 50);
        decodeEndpoint.markQueuedPhase(requestId);
        deliverAdmitted(List.of(item), new DecisionGroupMetadata("test", 0));
        long batchId = batchDispatcher.batchId;

        assertTrue(item.future().cancel(false));

        verify(cancelChannel, timeout(1_000)).cancel(any(), eq(requestId), anyLong());
        assertEquals(1, scheduler.getInflightSize());
        assertEquals(1, prefillEndpoint.getInflightBatchCount());
        cancelResult.complete(EngineCancelChannel.CancelOutcome.tombstoned());
        assertEquals(RequestLifecycleState.TIMED_OUT,
                scheduler.getRequestState(requestId, batchId).state());
        assertEquals(0, scheduler.getInflightSize());
        assertEquals(0, prefillEndpoint.getInflightBatchCount());
        assertFalse(decodeEndpoint.reservedView().containsKey(requestId));
    }

    @Test
    void batchSettlementBeforeProtectionDoesNotPublishEngineFence() {
        long requestId = 10_003L;
        BatchItem item = admittedItem(requestId, DeliveryMode.BATCH_ENQUEUE);
        deliverAdmitted(List.of(item), new DecisionGroupMetadata("test", 0));

        long batchId = batchDispatcher.batchId;
        WorkerStatusResponse finished = prefillFinished(requestId, batchId, 500);
        prefillEndpoint.onWorkerStatusUpdate(prefillEndpoint.getStatus(), finished);
        assertEquals(0, prefillEndpoint.getInflightBatchCount(),
                "the Prefill terminal owns settlement before protection is attempted");

        assertTrue(item.future().cancel(false));

        verify(cancelChannel, never()).cancel(any(), anyLong(), anyLong());
        assertEquals(1, scheduler.getInflightSize(),
                "the matching scheduler WorkerStatus reducer still owns final cleanup");

        scheduler.onWorkerStatusUpdate(finished);
        assertEquals(0, scheduler.getInflightSize());
        assertFalse(decodeEndpoint.reservedView().containsKey(requestId));
        verify(cancelChannel, never()).cancel(any(), anyLong(), anyLong());
    }

    @Test
    void routeCancellationBeforePublicationRollsBackLocally() {
        long requestId = 10_002L;
        BatchItem item = admittedItem(requestId, DeliveryMode.ROUTE_DECISION);

        deliverAdmitted(List.of(item), new DecisionGroupMetadata("test", 0));

        assertTrue(routeDelivery.wasClaimed());
        RequestLifecycleSnapshot claimed = scheduler.getRequestState(requestId, 0);
        assertEquals(RequestLifecycleState.DISPATCHING, claimed.state());
        assertEquals(DeliveryClaimKind.ROUTE_DECISION, claimed.deliveryClaimKind());
        assertEquals(1, prefillEndpoint.getIndividuallyTrackedRequestCount());

        assertTrue(item.future().cancel(false));

        assertEquals(0, scheduler.getInflightSize());
        assertEquals(0, prefillEndpoint.getIndividuallyTrackedRequestCount());
        assertEquals(0, prefillEndpoint.getLocallyOwnedRequestCount());
        assertFalse(decodeEndpoint.reservedView().containsKey(requestId));
        assertNull(scheduler.getRequestState(requestId, 0));
        verify(cancelChannel, never()).cancel(any(), anyLong(), anyLong());

        // Publication which resumes after cancellation observes the detached
        // generation and cannot recreate accounting or a schedule response.
        routeDelivery.callback.onDelivered(item);
        assertEquals(0, scheduler.getInflightSize());
        assertTrue(item.future().isCancelled());
    }

    @Test
    void masterCancelBeforeDeliveryIsTerminalAndIdempotent() throws Exception {
        long requestId = 10_101L;
        BatchItem item = admittedItem(requestId, DeliveryMode.BATCH_ENQUEUE);

        RequestLifecycleSnapshot cancelled = scheduler.cancelRequest(
                requestId, 0, CancelReason.CLIENT_CANCELLED);

        assertEquals(RequestLifecycleState.CANCELLED, cancelled.state());
        Response response = item.future().get(1, TimeUnit.SECONDS);
        assertFalse(response.isSuccess());
        assertEquals(8504, response.getCode());
        assertEquals(0, scheduler.getInflightSize());
        assertFalse(decodeEndpoint.reservedView().containsKey(requestId));
        verify(cancelChannel, never()).cancel(any(), anyLong(), anyLong());

        RequestLifecycleSnapshot repeated = scheduler.cancelRequest(
                requestId, 0, CancelReason.CLIENT_CANCELLED);
        assertEquals(cancelled, repeated);
        assertNull(scheduler.cancelRequest(
                99_999L, 0, CancelReason.CLIENT_CANCELLED));

        BatchItem reused = admittedItemWithoutRegistration(
                requestId, DeliveryMode.BATCH_ENQUEUE);
        assertFalse(scheduler.registerInflight(reused),
                "the terminal tombstone must fence request-id reuse");
    }

    @Test
    void deadlineBeforeDeliveryTerminatesAsTimedOut() throws Exception {
        long requestId = 10_106L;
        BatchItem item = admittedItem(requestId, DeliveryMode.BATCH_ENQUEUE);

        RequestLifecycleSnapshot timedOut = scheduler.cancelRequest(
                requestId, 0, CancelReason.DEADLINE_EXCEEDED);

        assertEquals(RequestLifecycleState.TIMED_OUT, timedOut.state());
        Response response = item.future().get(1, TimeUnit.SECONDS);
        assertFalse(response.isSuccess());
        assertEquals(8511, response.getCode());
        assertEquals(0, scheduler.getInflightSize());
        verify(cancelChannel, never()).cancel(any(), anyLong(), anyLong());
    }

    @Test
    void batchCancelUsesGenerationFenceAndOneEngineOwnerUnderConcurrency()
            throws Exception {
        long requestId = 10_102L;
        BatchItem item = admittedItem(requestId, DeliveryMode.BATCH_ENQUEUE);
        deliverAdmitted(List.of(item), new DecisionGroupMetadata("test", 0));
        long batchId = batchDispatcher.batchId;

        assertNull(scheduler.cancelRequest(
                requestId, batchId + 1, CancelReason.CLIENT_CANCELLED));
        verify(cancelChannel, never()).cancel(any(), anyLong(), anyLong());

        List<CompletableFuture<RequestLifecycleSnapshot>> calls =
                java.util.stream.IntStream.range(0, 16)
                        .mapToObj(ignored -> CompletableFuture.supplyAsync(() ->
                                scheduler.cancelRequest(requestId, batchId,
                                        CancelReason.CLIENT_CANCELLED)))
                        .toList();
        CompletableFuture.allOf(calls.toArray(CompletableFuture[]::new))
                .get(2, TimeUnit.SECONDS);

        for (CompletableFuture<RequestLifecycleSnapshot> call : calls) {
            assertEquals(RequestLifecycleState.CANCEL_REQUESTED, call.join().state());
        }
        verify(cancelChannel).cancel(any(), eq(requestId), anyLong());
        assertEquals(1, scheduler.getInflightSize());
        assertTrue(decodeEndpoint.reservedView().containsKey(requestId));

        prefillEndpoint.onWorkerStatusUpdate(
                prefillEndpoint.getStatus(), prefillFinished(requestId, batchId));
        assertEquals(1, prefillEndpoint.getInflightBatchCount(),
                "the accepted cancel must protect the ACK-uncertain batch member");

        batchDispatcher.callback.onSuccess(item, batchId);
        assertEquals(RequestLifecycleState.CANCEL_REQUESTED,
                scheduler.getRequestState(requestId, batchId).state(),
                "a late ACK cannot bypass the cancellation owner");

        cancelResult.complete(EngineCancelChannel.CancelOutcome.tombstoned());
        Response response = item.future().get(1, TimeUnit.SECONDS);
        assertFalse(response.isSuccess());
        assertEquals(8504, response.getCode());
        assertEquals(RequestLifecycleState.CANCELLED,
                scheduler.getRequestState(requestId, batchId).state());
        assertEquals(0, scheduler.getInflightSize());
        assertEquals(0, prefillEndpoint.getInflightBatchCount());
        assertFalse(decodeEndpoint.reservedView().containsKey(requestId));

        assertEquals(RequestLifecycleState.CANCELLED,
                scheduler.cancelRequest(requestId, batchId,
                        CancelReason.CLIENT_CANCELLED).state());
        verify(cancelChannel).cancel(any(), eq(requestId), anyLong());
    }

    @Test
    void batchCancelDoesNotClaimAcceptanceWhenSettlementWinsProtectionRace()
            throws Exception {
        long requestId = 10_109L;
        BatchItem item = admittedItem(requestId, DeliveryMode.BATCH_ENQUEUE);
        deliverAdmitted(List.of(item), new DecisionGroupMetadata("test", 0));
        long batchId = batchDispatcher.batchId;
        WorkerStatusResponse finished = prefillFinished(requestId, batchId, 500);
        prefillEndpoint.onWorkerStatusUpdate(prefillEndpoint.getStatus(), finished);
        assertEquals(0, prefillEndpoint.getInflightBatchCount());

        RequestLifecycleSnapshot result = scheduler.cancelRequest(
                requestId, batchId, CancelReason.CLIENT_CANCELLED);

        assertEquals(RequestLifecycleState.DISPATCHING, result.state(),
                "settlement won before cancellation could acquire ownership");
        verify(cancelChannel, never()).cancel(any(), anyLong(), anyLong());

        scheduler.onWorkerStatusUpdate(finished);
        Response response = item.future().get(1, TimeUnit.SECONDS);
        assertFalse(response.isSuccess());
        assertEquals(RequestLifecycleState.FAILED,
                scheduler.getRequestState(requestId, batchId).state());
        assertEquals(0, scheduler.getInflightSize());
    }

    @Test
    void acknowledgedBatchCancelDoesNotRequireSettledBatchProtection()
            throws Exception {
        long requestId = 10_110L;
        BatchItem item = admittedItem(requestId, DeliveryMode.BATCH_ENQUEUE);
        deliverAdmitted(List.of(item), new DecisionGroupMetadata("test", 0));
        long batchId = batchDispatcher.batchId;
        batchDispatcher.callback.onSuccess(item, batchId);
        assertEquals(RequestLifecycleState.ACKNOWLEDGED,
                scheduler.getRequestState(requestId, batchId).state());

        RequestLifecycleSnapshot pending = scheduler.cancelRequest(
                requestId, batchId, CancelReason.CLIENT_CANCELLED);

        assertEquals(RequestLifecycleState.CANCEL_REQUESTED, pending.state());
        verify(cancelChannel).cancel(any(), eq(requestId), anyLong());
        cancelResult.complete(EngineCancelChannel.CancelOutcome.tombstoned());
        assertEquals(RequestLifecycleState.CANCELLED,
                scheduler.getRequestState(requestId, batchId).state());
    }

    @Test
    void decodeAcceptanceBeforeBatchAckStillInstallsExplicitCancelOwner()
            throws Exception {
        long requestId = 10_111L;
        BatchItem item = admittedItem(requestId, DeliveryMode.BATCH_ENQUEUE);
        deliverAdmitted(List.of(item), new DecisionGroupMetadata("test", 0));
        long batchId = batchDispatcher.batchId;
        scheduler.onWorkerStatusUpdate(runningDecode(requestId, TaskPhase.KV_ALLOCATED));

        RequestLifecycleSnapshot pending = scheduler.cancelRequest(
                requestId, batchId, CancelReason.CLIENT_CANCELLED);

        assertEquals(RequestLifecycleState.CANCEL_REQUESTED, pending.state());
        verify(cancelChannel).cancel(any(), eq(requestId), anyLong());
        cancelResult.complete(EngineCancelChannel.CancelOutcome.tombstoned());
        assertEquals(RequestLifecycleState.CANCELLED,
                scheduler.getRequestState(requestId, batchId).state());
        assertEquals(0, scheduler.getInflightSize());
    }

    @Test
    void batchDeadlineSettlesAsTimedOutAfterEngineTombstone() throws Exception {
        long requestId = 10_107L;
        BatchItem item = admittedItem(requestId, DeliveryMode.BATCH_ENQUEUE);
        deliverAdmitted(List.of(item), new DecisionGroupMetadata("test", 0));
        long batchId = batchDispatcher.batchId;

        RequestLifecycleSnapshot pending = scheduler.cancelRequest(
                requestId, batchId, CancelReason.DEADLINE_EXCEEDED);
        assertEquals(RequestLifecycleState.CANCEL_REQUESTED, pending.state());

        cancelResult.complete(EngineCancelChannel.CancelOutcome.tombstoned());

        Response response = item.future().get(1, TimeUnit.SECONDS);
        assertFalse(response.isSuccess());
        assertEquals(8511, response.getCode());
        assertEquals(RequestLifecycleState.TIMED_OUT,
                scheduler.getRequestState(requestId, batchId).state());
        assertEquals(0, scheduler.getInflightSize());
    }

    @Test
    void batchAdmissionDeadlineRemainsFirstCauseAfterLateClientCancel()
            throws Exception {
        long requestId = 10_112L;
        BatchItem item = admittedItem(requestId, DeliveryMode.BATCH_ENQUEUE);
        deliverAdmitted(List.of(item), new DecisionGroupMetadata("test", 0));
        long batchId = batchDispatcher.batchId;

        scheduler.onRequestExpired(requestId, item.future());
        assertEquals(RequestLifecycleState.CANCEL_REQUESTED,
                scheduler.getRequestState(requestId, batchId).state());

        RequestLifecycleSnapshot lateClientCancel = scheduler.cancelRequest(
                requestId, batchId, CancelReason.CLIENT_CANCELLED);
        assertEquals(RequestLifecycleState.CANCEL_REQUESTED, lateClientCancel.state());
        verify(cancelChannel, timeout(1_000)).cancel(any(), eq(requestId), anyLong());

        scheduler.onWorkerStatusUpdate(
                finished(RoleType.DECODE, requestId, batchId, 0));
        Response response = item.future().get(1, TimeUnit.SECONDS);
        assertFalse(response.isSuccess());
        assertEquals(8511, response.getCode(),
                "the admission deadline must not be relabeled as client cancellation");
        assertEquals(RequestLifecycleState.TIMED_OUT,
                scheduler.getRequestState(requestId, batchId).state());
        cancelResult.complete(EngineCancelChannel.CancelOutcome.accepted());
    }

    @Test
    void explicitCancelTakesOwnershipFromPriorityNotFound() throws Exception {
        long requestId = 10_113L;
        BatchItem item = admittedItem(requestId, DeliveryMode.BATCH_ENQUEUE);
        long attemptToken = 503L;
        commitQueuedDecodeDispatch(item);
        prepareNotFoundPreemption(item, attemptToken, 90_001L);

        RequestLifecycleSnapshot pending = scheduler.cancelRequest(
                requestId, 0, CancelReason.CLIENT_CANCELLED);

        assertEquals(RequestLifecycleState.CANCEL_REQUESTED, pending.state());
        verify(cancelChannel).cancel(any(), eq(requestId), anyLong());
        cancelResult.complete(EngineCancelChannel.CancelOutcome.tombstoned());
        assertEquals(8504, item.future().get(1, TimeUnit.SECONDS).getCode());
        assertEquals(RequestLifecycleState.CANCELLED,
                scheduler.getRequestState(requestId, 0).state());
        assertEquals(0, scheduler.getInflightSize());
    }

    @Test
    void softTimeoutTakesOwnershipFromPriorityNotFoundWithoutLaterStatus()
            throws Exception {
        long requestId = 10_114L;
        BatchItem item = admittedItem(requestId, DeliveryMode.ROUTE_DECISION);
        deliverAdmitted(List.of(item), new DecisionGroupMetadata("test", 0));
        routeDelivery.callback.onDelivered(item);
        assertTrue(item.future().get(1, TimeUnit.SECONDS).isSuccess());
        long attemptToken = 504L;
        prepareNotFoundPreemption(item, attemptToken, 90_002L);

        assertEquals(PostDeliveryFenceResult.STARTED,
                scheduler.fenceAfterDeliveryTimeout(item, "post-delivery soft timeout"));
        verify(cancelChannel).cancel(any(), eq(requestId), anyLong());

        cancelResult.complete(EngineCancelChannel.CancelOutcome.tombstoned());
        assertEquals(RequestLifecycleState.TIMED_OUT,
                scheduler.getRequestState(requestId, 0).state());
        assertEquals(0, scheduler.getInflightSize());
        assertEquals(0, prefillEndpoint.getIndividuallyTrackedRequestCount());
    }

    @Test
    void routeCancelStaysPendingUntilWorkerTerminal() throws Exception {
        long requestId = 10_103L;
        BatchItem item = admittedItem(requestId, DeliveryMode.ROUTE_DECISION);
        deliverAdmitted(List.of(item), new DecisionGroupMetadata("test", 0));
        routeDelivery.callback.onDelivered(item);
        assertTrue(item.future().get(1, TimeUnit.SECONDS).isSuccess());

        RequestLifecycleSnapshot pending = scheduler.cancelRequest(
                requestId, 0, CancelReason.CLIENT_CANCELLED);

        assertEquals(RequestLifecycleState.CANCEL_REQUESTED, pending.state());
        assertEquals(1, scheduler.getInflightSize());
        assertEquals(1, prefillEndpoint.getIndividuallyTrackedRequestCount());
        verify(cancelChannel).cancel(any(), eq(requestId), anyLong());

        cancelResult.complete(EngineCancelChannel.CancelOutcome.accepted());
        scheduler.onWorkerStatusUpdate(
                finished(RoleType.DECODE, requestId, 77_777L, 2));

        assertEquals(RequestLifecycleState.CANCELLED,
                scheduler.getRequestState(requestId, 0).state());
        assertEquals(0, scheduler.getInflightSize());
        assertEquals(0, prefillEndpoint.getIndividuallyTrackedRequestCount());
        assertFalse(decodeEndpoint.reservedView().containsKey(requestId));
    }

    @Test
    void completedRouteDeliveryIgnoresLateAdmissionDeadline() throws Exception {
        long requestId = 10_108L;
        BatchItem item = admittedItem(requestId, DeliveryMode.ROUTE_DECISION);
        deliverAdmitted(List.of(item), new DecisionGroupMetadata("test", 0));
        routeDelivery.callback.onDelivered(item);
        assertTrue(item.future().get(1, TimeUnit.SECONDS).isSuccess());

        RequestLifecycleSnapshot acknowledged = scheduler.cancelRequest(
                requestId, 0, CancelReason.DEADLINE_EXCEEDED);
        assertEquals(RequestLifecycleState.ACKNOWLEDGED, acknowledged.state());
        verify(cancelChannel, never()).cancel(any(), anyLong(), anyLong());

        scheduler.onWorkerStatusUpdate(
                finished(RoleType.DECODE, requestId, 77_778L, 2));

        assertEquals(RequestLifecycleState.FAILED,
                scheduler.getRequestState(requestId, 0).state());
        assertEquals(0, scheduler.getInflightSize());
        assertEquals(0, prefillEndpoint.getIndividuallyTrackedRequestCount());
    }

    @Test
    void clientCancelBeforePriorityRpcKeepsClientTerminalCause() throws Exception {
        long requestId = 10_104L;
        BatchItem item = admittedItem(requestId, DeliveryMode.BATCH_ENQUEUE);
        long token = 501L;
        assertTrue(scheduler.claimForPreemption(requestId, token, "priority attempt"));

        RequestLifecycleSnapshot pending = scheduler.cancelRequest(
                requestId, 0, CancelReason.CLIENT_CANCELLED);
        assertEquals(RequestLifecycleState.CANCEL_REQUESTED, pending.state());
        assertFalse(scheduler.markPreemptionCancelInFlight(requestId, token),
                "the later priority RPC must lose to the accepted client cancel");
        assertTrue(scheduler.releasePreemptionClaim(requestId, token));
        verify(cancelChannel).cancel(any(), eq(requestId), anyLong());

        cancelResult.complete(EngineCancelChannel.CancelOutcome.tombstoned());
        Response response = item.future().get(1, TimeUnit.SECONDS);
        assertEquals(8504, response.getCode());
        assertEquals(RequestLifecycleState.CANCELLED,
                scheduler.getRequestState(requestId, 0).state());
    }

    @Test
    void priorityRpcAlreadyInFlightKeepsPriorityTerminalCause() throws Exception {
        long requestId = 10_105L;
        BatchItem item = admittedItem(requestId, DeliveryMode.BATCH_ENQUEUE);
        long token = 502L;
        assertTrue(scheduler.claimForPreemption(requestId, token, "priority attempt"));
        assertTrue(scheduler.markPreemptionCancelInFlight(requestId, token));

        RequestLifecycleSnapshot notAccepted = scheduler.cancelRequest(
                requestId, 0, CancelReason.CLIENT_CANCELLED);
        assertEquals(RequestLifecycleState.QUEUED, notAccepted.state(),
                "the RPC must not claim that the later client cancel was accepted");

        assertTrue(scheduler.finishTombstonedById(
                requestId, token, "priority cancel tombstoned"));
        Response response = item.future().get(1, TimeUnit.SECONDS);
        assertEquals(8429, response.getCode());
        assertEquals(RequestLifecycleState.CANCELLED,
                scheduler.getRequestState(requestId, 0).state());
        verify(cancelChannel, never()).cancel(any(), anyLong(), anyLong());
    }

    @Test
    void admissionDeadlineBeforePriorityRpcWinsInBatchAndRouteModes()
            throws Exception {
        when(cancelChannel.cancel(any(), anyLong(), anyLong())).thenReturn(
                CompletableFuture.completedFuture(
                        EngineCancelChannel.CancelOutcome.tombstoned()));
        DeliveryMode[] modes = {DeliveryMode.BATCH_ENQUEUE, DeliveryMode.ROUTE_DECISION};
        for (int index = 0; index < modes.length; index++) {
            long requestId = 10_116L + index;
            long attemptToken = 505L + index;
            BatchItem item = admittedItem(requestId, modes[index]);
            assertTrue(scheduler.claimForPreemption(
                    requestId, attemptToken, "priority attempt"));

            scheduler.onRequestExpired(requestId, item.future());

            assertEquals(RequestLifecycleState.CANCEL_REQUESTED,
                    scheduler.getRequestState(requestId, 0).state());
            assertFalse(scheduler.markPreemptionCancelInFlight(
                    requestId, attemptToken),
                    "deadline first-cause must prevent a later priority RPC");
            assertTrue(scheduler.releasePreemptionClaim(requestId, attemptToken));
            assertEquals(StrategyErrorType.BATCH_SLO_EXPIRED.getErrorCode(),
                    item.future().get(1, TimeUnit.SECONDS).getCode());
            assertEquals(RequestLifecycleState.TIMED_OUT,
                    scheduler.getRequestState(requestId, 0).state());
        }
    }

    @Test
    void admissionDeadlineTakesOwnershipFromPriorityNotFound() throws Exception {
        long requestId = 10_118L;
        BatchItem item = admittedItem(requestId, DeliveryMode.BATCH_ENQUEUE);
        long attemptToken = 507L;
        commitQueuedDecodeDispatch(item);
        prepareNotFoundPreemption(item, attemptToken, 90_003L);

        scheduler.onRequestExpired(requestId, item.future());

        assertEquals(RequestLifecycleState.CANCEL_REQUESTED,
                scheduler.getRequestState(requestId, 0).state());
        verify(cancelChannel, timeout(1_000)).cancel(any(), eq(requestId), anyLong());
        cancelResult.complete(EngineCancelChannel.CancelOutcome.tombstoned());
        assertEquals(8511, item.future().get(1, TimeUnit.SECONDS).getCode());
        assertEquals(RequestLifecycleState.TIMED_OUT,
                scheduler.getRequestState(requestId, 0).state());
    }

    @Test
    void decodeAcceptanceBeforeDeadlineWinsWhilePriorityClaimIsReversible()
            throws Exception {
        long requestId = 10_119L;
        long attemptToken = 508L;
        BatchItem item = admittedItem(requestId, DeliveryMode.BATCH_ENQUEUE);
        deliverAdmitted(List.of(item), new DecisionGroupMetadata("test", 0));
        assertTrue(scheduler.claimForPreemption(
                requestId, attemptToken, "priority attempt"));
        scheduler.onWorkerStatusUpdate(runningDecode(requestId, TaskPhase.KV_ALLOCATED));

        scheduler.onRequestExpired(requestId, item.future());

        assertEquals(RequestLifecycleState.DISPATCHING,
                scheduler.getRequestState(requestId, batchDispatcher.batchId).state());
        assertTrue(scheduler.releasePreemptionClaim(requestId, attemptToken));
        assertTrue(item.future().get(1, TimeUnit.SECONDS).isSuccess());
        assertEquals(RequestLifecycleState.ACKNOWLEDGED,
                scheduler.getRequestState(requestId, batchDispatcher.batchId).state());
        verify(cancelChannel, never()).cancel(any(), anyLong(), anyLong());
    }

    @Test
    void decodeAcceptanceBeforeDeadlineReplaysPriorityNotFoundAsDelivery()
            throws Exception {
        long requestId = 10_120L;
        long attemptToken = 509L;
        BatchItem item = admittedItem(requestId, DeliveryMode.BATCH_ENQUEUE);
        deliverAdmitted(List.of(item), new DecisionGroupMetadata("test", 0));
        prepareNotFoundPreemption(item, attemptToken, 90_004L);
        scheduler.onWorkerStatusUpdate(runningDecode(requestId, TaskPhase.KV_ALLOCATED));

        scheduler.onRequestExpired(requestId, item.future());

        assertTrue(item.future().get(1, TimeUnit.SECONDS).isSuccess());
        assertEquals(RequestLifecycleState.ACKNOWLEDGED,
                scheduler.getRequestState(requestId, batchDispatcher.batchId).state());
        verify(cancelChannel, never()).cancel(any(), anyLong(), anyLong());
    }

    @Test
    void claimedDeliveryResponseWinsLaterDeadlineInBatchAndRouteModes()
            throws Exception {
        int completionWorkers = scheduler.completionExecutorSnapshot().workerLimit();
        CountDownLatch workersBlocked = new CountDownLatch(completionWorkers);
        CountDownLatch releaseWorkers = new CountDownLatch(1);
        List<BatchItem> blockers = new java.util.ArrayList<>(completionWorkers);
        for (int index = 0; index < completionWorkers; index++) {
            BatchItem blocker = admittedItem(10_130L + index, DeliveryMode.ROUTE_DECISION);
            blockers.add(blocker);
            blocker.future().thenRun(() -> {
                workersBlocked.countDown();
                await(releaseWorkers);
            });
            deliverAdmitted(
                    List.of(blocker), new DecisionGroupMetadata("block_completion", 0));
            scheduler.onDelivered(blocker);
        }
        assertTrue(workersBlocked.await(1, TimeUnit.SECONDS));

        BatchItem batch = admittedItem(10_140L, DeliveryMode.BATCH_ENQUEUE);
        BatchItem route = admittedItem(10_141L, DeliveryMode.ROUTE_DECISION);
        try {
            deliverAdmitted(
                    List.of(batch), new DecisionGroupMetadata("batch_response_claim", 0));
            long batchId = batchDispatcher.batchId;
            batchDispatcher.callback.onSuccess(batch, batchId);
            deliverAdmitted(
                    List.of(route), new DecisionGroupMetadata("route_response_claim", 0));
            scheduler.onDelivered(route);

            assertFalse(batch.future().isDone());
            assertFalse(route.future().isDone());
            assertEquals(RequestLifecycleState.ACKNOWLEDGED,
                    scheduler.getRequestState(batch.requestId(), batchId).state());
            assertEquals(RequestLifecycleState.ACKNOWLEDGED,
                    scheduler.getRequestState(route.requestId(), 0).state());

            scheduler.onRequestExpired(batch.requestId(), batch.future());
            scheduler.onRequestExpired(route.requestId(), route.future());

            assertEquals(RequestLifecycleState.ACKNOWLEDGED,
                    scheduler.getRequestState(batch.requestId(), batchId).state());
            assertEquals(RequestLifecycleState.ACKNOWLEDGED,
                    scheduler.getRequestState(route.requestId(), 0).state());
            verify(cancelChannel, never()).cancel(any(), anyLong(), anyLong());
        } finally {
            releaseWorkers.countDown();
        }

        assertTrue(batch.future().get(1, TimeUnit.SECONDS).isSuccess());
        assertTrue(route.future().get(1, TimeUnit.SECONDS).isSuccess());
        for (BatchItem item : blockers) {
            scheduler.onWorkerStatusUpdate(
                    finished(RoleType.DECODE, item.requestId(), 0, 0));
        }
        scheduler.onWorkerStatusUpdate(
                finished(RoleType.DECODE, batch.requestId(), batchDispatcher.batchId, 0));
        scheduler.onWorkerStatusUpdate(
                finished(RoleType.DECODE, route.requestId(), 0, 0));
        assertEquals(0, scheduler.getInflightSize());
    }

    @Test
    void ttlWaitsForBatchDeliveryCommitThenFencesTheEngine() throws Exception {
        long requestId = 10_142L;
        config.queueScheduler().getLifecycle().setStaleInflightTimeoutMs(-1);
        CountDownLatch dispatchEntered = new CountDownLatch(1);
        CountDownLatch releaseDispatch = new CountDownLatch(1);
        batchDispatcher.blockDispatch(dispatchEntered, releaseDispatch);
        BatchItem item = admittedItem(requestId, DeliveryMode.BATCH_ENQUEUE);

        CompletableFuture<Void> delivery = CompletableFuture.runAsync(() ->
                deliverAdmitted(
                        List.of(item), new DecisionGroupMetadata("ttl_delivery_race", 0)));
        assertTrue(dispatchEntered.await(1, TimeUnit.SECONDS));
        CountDownLatch cleanupStarted = new CountDownLatch(1);
        CompletableFuture<Void> cleanup = CompletableFuture.runAsync(() -> {
            cleanupStarted.countDown();
            scheduler.cleanupInflight();
        });

        assertTrue(cleanupStarted.await(1, TimeUnit.SECONDS));
        Thread.sleep(25);
        assertFalse(cleanup.isDone(),
                "TTL must serialize behind the delivery commit/send boundary");
        assertEquals(1, scheduler.getInflightSize());
        assertEquals(1, prefillEndpoint.getInflightBatchCount());
        assertTrue(decodeEndpoint.reservedView().containsKey(requestId));

        releaseDispatch.countDown();
        delivery.get(1, TimeUnit.SECONDS);
        cleanup.get(1, TimeUnit.SECONDS);

        assertEquals(RequestLifecycleState.CANCEL_REQUESTED,
                scheduler.getRequestState(requestId, batchDispatcher.batchId).state());
        verify(cancelChannel, timeout(1_000)).cancel(any(), eq(requestId), anyLong());
        assertEquals(1, prefillEndpoint.getInflightBatchCount());
        assertTrue(decodeEndpoint.reservedView().containsKey(requestId));

        cancelResult.complete(EngineCancelChannel.CancelOutcome.tombstoned());
        assertEquals(RequestLifecycleState.TIMED_OUT,
                scheduler.getRequestState(requestId, batchDispatcher.batchId).state());
        assertEquals(0, scheduler.getInflightSize());
        assertEquals(0, prefillEndpoint.getInflightBatchCount());
        assertFalse(decodeEndpoint.reservedView().containsKey(requestId));
    }

    @Test
    void ttlRetainsAcknowledgedBatchAndRouteLedgersUntilEngineProof()
            throws Exception {
        config.queueScheduler().getLifecycle().setStaleInflightTimeoutMs(-1);
        long batchRequestId = 10_143L;
        long routeRequestId = 10_144L;
        BatchItem batch = admittedItem(batchRequestId, DeliveryMode.BATCH_ENQUEUE);
        BatchItem route = admittedItem(routeRequestId, DeliveryMode.ROUTE_DECISION);

        deliverAdmitted(
                List.of(batch), new DecisionGroupMetadata("ttl_batch_ack", 0));
        long batchId = batchDispatcher.batchId;
        batchDispatcher.callback.onSuccess(batch, batchId);
        assertTrue(batch.future().get(1, TimeUnit.SECONDS).isSuccess());
        scheduler.onWorkerStatusUpdate(
                runningDecode(batchRequestId, TaskPhase.KV_ALLOCATED));

        deliverAdmitted(
                List.of(route), new DecisionGroupMetadata("ttl_route_ack", 0));
        routeDelivery.callback.onDelivered(route);
        assertTrue(route.future().get(1, TimeUnit.SECONDS).isSuccess());

        scheduler.cleanupInflight();

        assertEquals(RequestLifecycleState.CANCEL_REQUESTED,
                scheduler.getRequestState(batchRequestId, batchId).state());
        assertEquals(RequestLifecycleState.CANCEL_REQUESTED,
                scheduler.getRequestState(routeRequestId, 0).state());
        verify(cancelChannel, timeout(1_000).times(2))
                .cancel(any(), anyLong(), anyLong());
        assertEquals(2, scheduler.getInflightSize());
        assertEquals(1, prefillEndpoint.getInflightBatchCount());
        assertEquals(1, prefillEndpoint.getIndividuallyTrackedRequestCount());
        assertTrue(decodeEndpoint.reservedView().containsKey(batchRequestId));
        assertTrue(decodeEndpoint.reservedView().containsKey(routeRequestId));

        assertEquals(0, prefillEndpoint.evictExpiredInflight(-1),
                "pending Engine fences must protect both batch and route ledgers");
        assertEquals(1, prefillEndpoint.getInflightBatchCount());
        assertEquals(1, prefillEndpoint.getIndividuallyTrackedRequestCount());

        cancelResult.complete(EngineCancelChannel.CancelOutcome.tombstoned());

        assertEquals(RequestLifecycleState.TIMED_OUT,
                scheduler.getRequestState(batchRequestId, batchId).state());
        assertEquals(RequestLifecycleState.TIMED_OUT,
                scheduler.getRequestState(routeRequestId, 0).state());
        assertEquals(0, scheduler.getInflightSize());
        assertEquals(0, prefillEndpoint.getInflightBatchCount());
        assertEquals(0, prefillEndpoint.getIndividuallyTrackedRequestCount());
        assertFalse(decodeEndpoint.reservedView().containsKey(batchRequestId));
        assertFalse(decodeEndpoint.reservedView().containsKey(routeRequestId));
    }

    @Test
    void shutdownCompletesScheduleFutureWaitingForClientCancelProof()
            throws Exception {
        long requestId = 10_121L;
        BatchItem item = admittedItem(requestId, DeliveryMode.BATCH_ENQUEUE);
        deliverAdmitted(List.of(item), new DecisionGroupMetadata("test", 0));
        long batchId = batchDispatcher.batchId;
        RequestLifecycleSnapshot pending = scheduler.cancelRequest(
                requestId, batchId, CancelReason.CLIENT_CANCELLED);
        assertEquals(RequestLifecycleState.CANCEL_REQUESTED, pending.state());

        scheduler.shutdown();

        assertEquals(8510, item.future().get(1, TimeUnit.SECONDS).getCode());
        assertEquals(RequestLifecycleState.CANCEL_REQUESTED,
                scheduler.getRequestState(requestId, batchId).state());
        assertEquals(0, scheduler.generationGateCount());
    }

    @Test
    void shutdownCompletesScheduleFutureWaitingForDeadlineCancelProof()
            throws Exception {
        long requestId = 10_122L;
        BatchItem item = admittedItem(requestId, DeliveryMode.BATCH_ENQUEUE);
        deliverAdmitted(List.of(item), new DecisionGroupMetadata("test", 0));
        long batchId = batchDispatcher.batchId;
        RequestLifecycleSnapshot pending = scheduler.cancelRequest(
                requestId, batchId, CancelReason.DEADLINE_EXCEEDED);
        assertEquals(RequestLifecycleState.CANCEL_REQUESTED, pending.state());

        scheduler.shutdown();

        assertEquals(8510, item.future().get(1, TimeUnit.SECONDS).getCode());
        assertEquals(RequestLifecycleState.CANCEL_REQUESTED,
                scheduler.getRequestState(requestId, batchId).state());
        assertEquals(0, scheduler.generationGateCount());
    }

    private BatchItem admittedItem(long requestId, DeliveryMode scheduleMode) {
        BatchItem item = admittedItemWithoutRegistration(requestId, scheduleMode);

        assertTrue(scheduler.registerInflight(item));
        AdmissionLease lease = new AdmissionLease(item, decodeEndpoint,
                prefillEndpoint.getBatcher().queueManager(), scheduler,
                0, null, null);
        assertTrue(scheduler.attachAdmissionLease(item, lease));
        lease.bindTo(item.future());
        decodeEndpoint.reserve(requestId, 128, 136, 50);
        decodeEndpoint.markQueuedPhase(requestId);
        return item;
    }

    private BatchItem admittedItemWithoutRegistration(
            long requestId, DeliveryMode scheduleMode) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setPriority(50);
        request.setSeqLen(128);
        request.setMaxNewTokens(8);

        long nowMs = System.currentTimeMillis();
        BalanceContext context = new BalanceContext();
        context.setRequest(request);
        FlexlbConfig itemConfig = new FlexlbConfig();
        if (scheduleMode == DeliveryMode.ROUTE_DECISION) {
            SchedulingTestConfig.useNonBatchDispatcher(itemConfig);
        }
        context.setConfig(itemConfig);
        context.setSchedulingMetadata(SchedulingMetadata.explicit(50, nowMs + 30_000));

        ServerStatus prefill = server(
                RoleType.PREFILL, "10.0.0.1", 8080, 8081, requestId);
        ServerStatus decode = server(
                RoleType.DECODE, "10.0.0.2", 8081, 8082, requestId);
        Response route = new Response();
        route.setSuccess(true);
        route.setServerStatus(List.of(prefill, decode));
        BatchItem item = new BatchItem(context, new CompletableFuture<>(), route,
                prefill, decode, prefillEndpoint, decodeEndpoint, nowMs);
        return item;
    }

    private static void await(CountDownLatch latch) {
        try {
            assertTrue(latch.await(2, TimeUnit.SECONDS));
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            throw new AssertionError(error);
        }
    }

    private void prepareNotFoundPreemption(
            BatchItem item, long attemptToken, long incomingRequestId) {
        assertFalse(decodeEndpoint.layeredAdmissionView().queued()
                .contains(item.requestId()));
        assertTrue(decodeEndpoint.reservedView().containsKey(item.requestId()));
        assertTrue(scheduler.claimForPreemption(
                item.requestId(), attemptToken, "priority attempt"));
        assertEquals(DecodeEndpoint.PreemptionBeginResult.SUCCESS,
                decodeEndpoint.beginPriorityPreemption(
                        attemptToken,
                        List.of(item.requestId()),
                        incomingRequestId,
                        128,
                        136,
                        90,
                        decodeEndpoint.admissionVersion(),
                        true));
        assertTrue(decodeEndpoint.markPriorityCancelInFlight(attemptToken));
        assertTrue(scheduler.markPreemptionCancelInFlight(
                item.requestId(), attemptToken));
        assertTrue(decodeEndpoint.markPriorityCancelNotFound(
                attemptToken, item.requestId()));
        assertTrue(scheduler.markPreemptionNotFound(
                item.requestId(), attemptToken));
        decodeEndpoint.abortPriorityPreemption(attemptToken);
        assertFalse(decodeEndpoint.reservedView().containsKey(incomingRequestId));
    }

    private void commitQueuedDecodeDispatch(BatchItem item) {
        DecodeEndpoint.EngineDispatchPermitAcquisition acquisition =
                decodeEndpoint.acquireEngineDispatchPermit(item.requestId(), 100);
        assertEquals(DecodeEndpoint.EngineDispatchPermitAcquireStatus.ACQUIRED,
                acquisition.status());
        assertNotNull(acquisition.permit());
        assertEquals(TRANSFERRED,
                acquisition.permit().transferToEngineLifecycle());
    }

    private static WorkerStatusResponse finished(RoleType role,
                                                 long requestId,
                                                 long batchId,
                                                 long errorCode) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(requestId);
        task.setBatchId(batchId);
        task.setErrorCode(errorCode);
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(role);
        response.setFinishedTaskInfo(Map.of(Long.toString(requestId), task));
        response.setRunningTaskInfo(Map.of());
        return response;
    }

    private void deliverAdmitted(
            List<BatchItem> items,
            DecisionGroupMetadata metadata) {
        TestCapacityAdmission.runDeliveryCallback(
                scheduler,
                TestCapacityAdmission.admit(scheduler, items),
                metadata);
    }

    private static WorkerStatusResponse runningDecode(
            long requestId, TaskPhase phase) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(requestId);
        task.setPhase(phase);
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(RoleType.DECODE);
        response.setRunningTaskInfo(Map.of(Long.toString(requestId), task));
        response.setFinishedTaskInfo(Map.of());
        return response;
    }

    private static ServerStatus server(RoleType role,
                                       String ip,
                                       int httpPort,
                                       int grpcPort,
                                       long requestId) {
        ServerStatus status = new ServerStatus();
        status.setSuccess(true);
        status.setRole(role);
        status.setServerIp(ip);
        status.setHttpPort(httpPort);
        status.setGrpcPort(grpcPort);
        status.setRequestId(requestId);
        return status;
    }

    private static WorkerStatusResponse prefillFinished(long requestId, long batchId) {
        return prefillFinished(requestId, batchId, 0);
    }

    private static WorkerStatusResponse prefillFinished(
            long requestId, long batchId, long errorCode) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(requestId);
        task.setBatchId(batchId);
        task.setErrorCode(errorCode);
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(RoleType.PREFILL);
        response.setFinishedTaskInfo(Map.of(Long.toString(requestId), task));
        response.setRunningTaskInfo(Map.of());
        return response;
    }

    private static final class CapturingBatchDispatcher implements BatchDispatcher {
        private long batchId;
        private DispatchCallback callback;
        private CountDownLatch dispatchEntered;
        private CountDownLatch releaseDispatch;

        @Override
        public SubmissionReservationResult tryReserveSubmission() {
            return new SubmissionReserved(new SubmissionPermit() {
                @Override
                public void submit(List<BatchItem> items,
                                   PrefillEndpoint prefillEp,
                                   long submittedBatchId,
                                   long predMs,
                                   String reason,
                                   DispatchCallback submittedCallback) {
                    batchId = submittedBatchId;
                    callback = submittedCallback;
                    if (dispatchEntered != null) {
                        dispatchEntered.countDown();
                        await(releaseDispatch);
                    }
                }

                @Override
                public void release() {
                }
            });
        }

        private boolean wasSent() {
            return callback != null;
        }

        private void blockDispatch(
                CountDownLatch entered, CountDownLatch release) {
            this.dispatchEntered = entered;
            this.releaseDispatch = release;
        }
    }

    private static final class HoldingRouteDecisionDelivery
            implements DecisionDelivery<List<BatchItem>> {
        private Callback callback;

        @Override
        public void deliver(List<BatchItem> items, Callback callback) {
            this.callback = callback;
        }

        private boolean wasClaimed() {
            return callback != null;
        }
    }
}
