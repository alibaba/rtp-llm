package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.scheduler.RequestContext.AdmissionHandle;
import org.flexlb.balance.scheduler.RequestContext.DeliveryClaim;
import org.flexlb.balance.scheduler.RequestContext.PreemptionRegistration;
import org.flexlb.balance.scheduler.RequestContext.RequestStage;
import org.flexlb.balance.scheduler.RequestProtocolTestSupport.Registered;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.service.RecentCacheKeyTraceReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import static org.flexlb.balance.scheduler.RequestProtocolTestSupport.awaitCondition;
import static org.flexlb.balance.scheduler.RequestProtocolTestSupport.commitRoute;
import static org.flexlb.balance.scheduler.SchedulingTestConfig.freezeInputs;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Request lifecycle and exact generation ownership tests.
 */
class RequestContextLifecycleTest {

    private FlexlbConfig config;

    private AbstractRequestScheduler lifecycle;

    @BeforeEach
    void setUp() {
        config = SchedulingTestConfig.batchConfig();
        SchedulingTestConfig.usePriorityQueue(config);
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        lifecycle = org.flexlb.balance.scheduler.SchedulerTestSupport.create(configService, mock(DeliveryMetricsReporter.class), mock(RequestSchedulerReporter.class),
                mock(RecentCacheKeyTraceReporter.class));
    }

    @AfterEach
    void tearDown() {
        if (RequestProtocolTestSupport.closeAdmissionAndAwaitMutations(lifecycle)) {
            lifecycle.closeOutstandingAndTerminalize();
            org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(lifecycle).timer().close();
            org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(lifecycle).closeRequestExecutors();
        }
    }

    @Test
    void deadlineInstallationDoesNotReadUnboundRequestForDiagnostics() {
        RequestContext context = new RequestContext(config);
        context.setFuture(new CompletableFuture<>());
        var deadline = mock(ExpirationTimer.RequestDeadline.class);
        assertTrue(context.installRequestDeadline(deadline));
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void pendingDeliveryCountTracksTheLifecycleBeforeSenderHandoff(boolean immediate) {
        var requestConfig = immediate ? SchedulingTestConfig.newConfig() : config;
        if (immediate) { requestConfig.setScheduler(org.flexlb.config.SchedulerConfig.direct()); }
        var context = RequestProtocolTestSupport.context(requestConfig, 894L);
        var future = RequestProtocolTestSupport.register(lifecycle, context);
        var repository = SchedulerTestSupport.repository(lifecycle);
        int expectedPending = immediate ? 0 : 1;
        assertEquals(immediate ? RequestRequirements.DecodeMode.IMMEDIATE
                : RequestRequirements.DecodeMode.PREEMPT_AT_PLACEMENT, context.getRequirements().mode());
        assertEquals(RequestStage.QUEUED, context.stage());
        assertEquals(expectedPending, repository.pendingDeliveryRequestCount());
        var item = SchedulingTestConfig.createRoute(freezeInputs(context), new Response(), null, null,
                mock(PrefillEndpoint.class), null, null, System.currentTimeMillis());
        try (var admission = lifecycle.claimAdmissionHandle(context.getRequestId(), future);
             var completion = RequestProtocolTestSupport.finishOnExit(admission)) {
            assertNotNull(admission);
            assertEquals(RequestStage.ROUTING, context.stage());
            assertEquals(expectedPending, repository.pendingDeliveryRequestCount());
            assertEquals(PlacementResult.Status.SUCCESS,
                    lifecycle.commitRoute(item, RequestProtocolTestSupport.publication(() -> true)));
            assertEquals(RequestStage.READY_TO_DELIVER, context.stage());
            assertEquals(expectedPending, repository.pendingDeliveryRequestCount());
            admission.finish();
        }
        assertEquals(expectedPending, repository.pendingDeliveryRequestCount());
        var delivery = RequestProtocolTestSupport.claimBatchWithoutPrediction(lifecycle, item, 8L, () -> true);
        assertNotNull(delivery);
        assertEquals(RequestStage.DELIVERING, context.stage());
        try {
            assertEquals(0, repository.pendingDeliveryRequestCount());
            assertEquals(1, repository.liveRequestCount(), "sender handoff is not request completion");
        } finally {
            lifecycle.completeDelivery(delivery, org.flexlb.balance.delivery.DeliveryResult.notSent(
                    new IllegalStateException("test sender stopped")));
        }
        assertEquals(0, repository.pendingDeliveryRequestCount());
    }

    @Test
    void registrationCapturesCurrentIdentityAndKeepsItAfterFutureCompletion() {
        RequestContext context = context(890L);
        SchedulingTestConfig.freezeInputs(context);
        context.getRequest().setRequestId(891L);
        var future = RequestProtocolTestSupport.register(lifecycle, context);
        assertSame(future, context.getFuture());
        assertEquals(891L, context.getRequirements().requestId());
        RequestContext other = context(893L);
        assertThrows(IllegalStateException.class, () -> other.setFuture(future));
        assertEquals(893L, other.getRequestId());
        var otherFuture = RequestProtocolTestSupport.register(lifecycle, other);
        assertFalse(otherFuture.isDone());
        assertSame(otherFuture, other.getFuture());
        assertEquals(893L, other.getRequestId());
        context.getRequest().setRequestId(892L);
        assertEquals(891L, context.getRequestId());
        assertTrue(future.cancel(false));
        assertEquals(891L, context.getRequestId());
        assertThrows(IllegalStateException.class, () -> context.setFuture(new CompletableFuture<>()));
        assertTrue(RequestProtocolTestSupport.register(lifecycle, context).isDone());
        assertSame(future, context.getFuture());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void registrationFreezesPriorityAndDeadline(boolean suppliedMetadata) {
        RequestContext context = context(806L);
        context.getRequest().setPriority(73);
        context.setSchedulingMetadata(suppliedMetadata
                ? SchedulingMetadata.explicit(64, Long.MAX_VALUE) : null);
        int priority = context.getPriority();
        long expiresAt = context.getRequestExpiresAtMs();
        RequestProtocolTestSupport.register(lifecycle, context);
        assertNotNull(context.getRequirements());
        var metadata = context.getSchedulingMetadata();
        assertNotNull(metadata);
        context.setSchedulingMetadata(metadata);
        assertThrows(IllegalStateException.class, () -> context.setSchedulingMetadata(null));
        assertThrows(IllegalStateException.class, () -> context.setSchedulingMetadata(
                SchedulingMetadata.explicit(1, 1L)));
        context.getRequest().setPriority(1);
        org.springframework.test.util.ReflectionTestUtils.setField(context, "startTime", 1L);
        assertEquals(priority, context.getPriority());
        assertEquals(expiresAt, context.getRequestExpiresAtMs());
        assertEquals(priority, RequestRequirements.capture(context).priority());
    }

    @ParameterizedTest
    @ValueSource(ints = {0, -1, 101, 1, 73, 100})
    void queueReadsFrozenPriorityAndPreservesInternalDefaulting(int priority) {
        var context = context(808L);
        context.setSchedulingMetadata(null);
        context.getRequest().setPriority(priority);
        RequestProtocolTestSupport.register(lifecycle, context);
        assertSame(context.getSchedulingMetadata(), context.getRequirements().schedulingMetadata());
        int queuePriority = priority >= 1 && priority <= 100 ? priority : 50;
        var entry = new GlobalQueueEntry(context, null);
        for (boolean priorityOrdering : new boolean[]{false, true}) {
            var queue = new OrderedRequestQueue(priorityOrdering);
            queue.add(entry);
            context.getRequest().setPriority(99);
            assertEquals(priority, context.getPriority());
            assertEquals(queuePriority, entry.priority());
            assertEquals(1, queue.priorityCounts()[queuePriority]);
            queue.remove(entry);
            queue.restore(entry);
            assertEquals(queuePriority, entry.priority());
            queue.remove(entry);
        }
        assertEquals(queuePriority, new GlobalQueueEntry(context, null).priority());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void registrationFreezesRoutingInputsAcrossRouteAttempts(boolean routeDelivery) {
        var dispatcher = config.getDispatcher();
        dispatcher.setType(routeDelivery ? org.flexlb.config.DispatcherConfig.Type.NON_BATCH
                : org.flexlb.config.DispatcherConfig.Type.BATCH);
        dispatcher.setMaxInflightPerPrefillWorker(3);
        RequestContext context = context(807L);
        var request = context.getRequest();
        var cacheKeys = new ArrayList<>(List.of(11L, 22L));
        request.setSeqLen(64L);
        request.setMaxNewTokens(32);
        request.setApiKey("original");
        request.setBlockCacheKeys(cacheKeys);
        request.setCacheKeyBlockSize(16L);
        RequestProtocolTestSupport.register(lifecycle, context);
        var inputs = context.getRequirements();
        request.setRequestId(808L);
        request.setSeqLen(1L);
        request.setMaxNewTokens(2);
        request.setApiKey("changed");
        request.setCacheKeyBlockSize(1L);
        cacheKeys.clear();
        dispatcher.setType(routeDelivery ? org.flexlb.config.DispatcherConfig.Type.BATCH
                : org.flexlb.config.DispatcherConfig.Type.NON_BATCH);
        dispatcher.setMaxInflightPerPrefillWorker(99);
        var first = org.flexlb.balance.scheduler.SchedulingTestConfig.createRoute(context, null, null, null, null, null, null, 123L);
        var retry = org.flexlb.balance.scheduler.SchedulingTestConfig.createRoute(context, null, null, null, null, null, null, 456L);
        assertSame(inputs, first.requirements());
        assertSame(inputs, retry.requirements());
        assertEquals(routeDelivery, first.requiresRouteReservation());
        assertEquals(routeDelivery, retry.requiresRouteReservation());
        assertEquals(routeDelivery ? 0 : 3, retry.requirements().maxInflightBatchesPerPrefillWorker());
        assertEquals(807L, retry.requestId());
        assertEquals(64L, retry.seqLen());
        assertEquals(96L, inputs.expectedKvTokens());
        assertEquals("original", inputs.apiKey());
        assertEquals(List.of(11L, 22L), inputs.blockCacheKeys());
        assertEquals(16L, inputs.cacheKeyBlockSize());
        assertThrows(UnsupportedOperationException.class, () -> inputs.blockCacheKeys().clear());
        assertEquals(first.enqueueSeq(), retry.enqueueSeq());
        assertEquals(123L, retry.enqueuedAtMs());
    }

    @Test
    void invalidRouteInputsDoNotInitializeWorkerFifoIdentity() {
        RequestContext context = context(804L);
        assertThrows(NullPointerException.class, () -> org.flexlb.balance.scheduler.SchedulingTestConfig.createRoute(context, null, null, null, null, null, null, 123L));
        assertThrows(IllegalArgumentException.class, () -> org.flexlb.balance.scheduler.SchedulingTestConfig.createRoute(freezeInputs(context), null, null, null, null, null,
                new DecodeResources.ReservationHandle(1L, 805L, 1L), 123L));
        assertEquals(0L, context.getWorkerEnqueueSequence());
        assertEquals(0L, context.getFirstWorkerEnqueueTime());
    }

    @Test
    void registrationAndFutureInitializationShareTheContextMonitor() throws Exception {
        RequestContext context = context(803L);
        CompletableFuture<Response> placeholder = new CompletableFuture<>();
        AtomicReference<CompletableFuture<Response>> registered = new AtomicReference<>();
        Thread registrar = new Thread(() -> registered.set(RequestProtocolTestSupport.register(lifecycle, context)), "registration-test");
        synchronized (context) {
            registrar.start();
            awaitCondition(() -> registrar.getState() == Thread.State.BLOCKED);
            assertNull(context.getRequirements(), "binding must wait for the Context monitor");
            context.setFuture(placeholder);
        }
        registrar.join(5_000L);
        assertFalse(registrar.isAlive());
        assertNotNull(registered.get());
        assertSame(registered.get(), context.getFuture());
        assertTrue(context.getFuture() instanceof RequestContext.RequestFuture);
        assertFalse(context.getFuture().isDone());
    }

    @Test
    void duplicateSubmitPreservesCanonicalFutureAndRequestId() {
        RequestContext context = context(800L);
        CompletableFuture<Response> canonical = RequestProtocolTestSupport.register(lifecycle, context);
        context.getRequest().setRequestId(801L);
        assertEquals(StrategyErrorType.INVALID_REQUEST.getErrorCode(), RequestProtocolTestSupport.register(lifecycle, context).join().getCode());
        assertSame(canonical, context.getFuture());
        assertEquals(800L, context.getRequestId());
        assertEquals(800L, RequestRequirements.capture(context).requestId());
        assertThrows(IllegalStateException.class, () -> context.setFuture(new CompletableFuture<>()));
        lifecycle.cancel(800L, 0L, CancelReason.CLIENT_CANCELLED);
        ((QueuedRequestScheduler) lifecycle).onGlobalControl(context);
        assertEquals(StrategyErrorType.REQUEST_CANCELLED.getErrorCode(), canonical.join().getCode());
        assertNull(lifecycle.findRequestContext(800L));
        assertEquals(RequestState.Phase.CANCELLED, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle).getRequestState(800L, 0L).state());
        assertNull(org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle).getRequestState(801L, 0L));
    }

    @Test
    void rejectedDuplicateSubmitCannotConsumeTheCanonicalControlTicket() {
        var queue = (QueuedRequestScheduler) lifecycle;
        org.mockito.Mockito.doNothing().when(queue).signalControl(org.mockito.ArgumentMatchers.any());
        RequestContext context = context(804L);
        CompletableFuture<Response> canonical = RequestProtocolTestSupport.register(lifecycle, context);
        lifecycle.cancel(804L, 0L, CancelReason.CLIENT_CANCELLED);

        assertEquals(StrategyErrorType.INVALID_REQUEST.getErrorCode(), queue.submit(context).join().getCode());
        assertFalse(canonical.isDone(), "duplicate submission does not own the queued cancellation ticket");
        assertSame(context, lifecycle.findRequestContext(804L));

        queue.onGlobalControl(context);
        assertEquals(StrategyErrorType.REQUEST_CANCELLED.getErrorCode(), canonical.join().getCode());
    }

    @Test
    void expiredTerminalAllowsNewContextButCannotReuseOldContext() {
        RequestContext old = context(802L);
        CompletableFuture<Response> original = RequestProtocolTestSupport.register(lifecycle, old);
        lifecycle.cancel(802L, 0L, CancelReason.CLIENT_CANCELLED);
        original.join();
        assertTrue(org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle).removeExactTerminal(org.flexlb.balance.scheduler.SchedulerTestSupport.terminalRecord(lifecycle, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle).getRequestState(802L, 0L)), Long.MAX_VALUE));
        assertEquals(StrategyErrorType.INVALID_REQUEST.getErrorCode(), RequestProtocolTestSupport.register(lifecycle, old).join().getCode());
        assertSame(original, old.getFuture());
        CompletableFuture<Response> replacement = RequestProtocolTestSupport.register(lifecycle, context(802L));
        assertFalse(replacement.isDone());
        assertSame(replacement, lifecycle.findRequestContext(802L).getFuture());
        var queue = (QueuedRequestScheduler) lifecycle;
        queue.onGlobalControl(old);
        assertFalse(queue.settleGlobalQueueClose(old));
        assertSame(replacement, lifecycle.findRequestContext(802L).getFuture());
        assertFalse(replacement.isDone(), "stale control and close tickets cannot settle a reused request id");
    }

    @Test
    void duplicateRegistrationCannotReplaceTheCanonicalExactGeneration() {
        RequestContext context = context(101L);
        CompletableFuture<Response> canonical = RequestProtocolTestSupport.register(lifecycle, context);
        CompletableFuture<Response> duplicate = RequestProtocolTestSupport.register(lifecycle, context(101L));
        assertFalse(canonical.isDone());
        assertTrue(duplicate.isDone());
        assertEquals(StrategyErrorType.INVALID_REQUEST.getErrorCode(), duplicate.join().getCode());
        assertSame(canonical, lifecycle.findRequestContext(101L).future());
        assertEquals(1, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle).liveRequestCount());
    }

    @Test
    void successfulExternalCompletionBeforeDeliveryCannotStrandTheRequest() {
        CompletableFuture<Response> future = RequestProtocolTestSupport.register(lifecycle, context(10001L));
        Response success = new Response();
        success.setSuccess(true);
        assertFalse(future.complete(success));
        assertFalse(future.isDone());
        assertEquals(RequestState.Phase.QUEUED, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle).getRequestState(10001L, 0L).state());
        assertEquals(1, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle).liveRequestCount());
    }

    @Test
    void futureCancelDuringAdmissionIsSynchronousAndTheAdmissionOwnerSettlesIt() {
        QueuedRequestScheduler queue = (QueuedRequestScheduler) lifecycle;
        org.mockito.Mockito.doNothing().when(queue).signalControl(org.mockito.ArgumentMatchers.any());
        CompletableFuture<Response> future = RequestProtocolTestSupport.register(lifecycle, context(10003L));
        RequestContext requestContext = lifecycle.findRequestContext(10003L);
        AdmissionHandle admission = lifecycle.claimAdmissionHandle(10003L, future);
        assertNotNull(admission);
        assertTrue(future.cancel(false));
        assertTrue(future.isCancelled());
        assertEquals(RequestState.Phase.CANCEL_REQUESTED, requestContext.snapshot().state());
        admission.finish();
        assertEquals(RequestState.Phase.CANCELLED, requestContext.snapshot().state());
        assertEquals(0, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle).liveRequestCount());
        verify(queue).signalControl(requestContext);
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void schedulingDeadlineDuringAdmissionRetainsItsOwnEntryRuleAfterDeliveryClaim(boolean timer) {
        var context = context(10007L);
        var future = RequestProtocolTestSupport.register(lifecycle, context);
        ExpirationTimer.RequestDeadline deadline = RequestProtocolTestSupport.field(context, "requestDeadline");
        assertNotNull(deadline);
        var item = org.flexlb.balance.scheduler.SchedulingTestConfig.createRoute(freezeInputs(context), new Response(), null, null,
                mock(PrefillEndpoint.class), null, null, System.currentTimeMillis());
        try (var admission = lifecycle.claimAdmissionHandle(context.getRequestId(), future); var admissionCompletion1 = RequestProtocolTestSupport.finishOnExit(admission)) {
            assertNotNull(admission);
            assertEquals(PlacementResult.Status.SUCCESS, lifecycle.commitRoute(item, RequestProtocolTestSupport.publication(() -> true)));
            assertNotNull(RequestProtocolTestSupport.claimRouteWithoutPrediction(lifecycle, item, () -> true));
            if (timer) {
                lifecycle.onSchedulingDeadline(context, deadline);
            } else {
                lifecycle.cancelRequest(context, 0L, CancelReason.DEADLINE_EXCEEDED);
            }
            assertEquals(timer ? CancelReason.DEADLINE_EXCEEDED : null, context.cancellationReason());
            assertSame(timer ? null : deadline, RequestProtocolTestSupport.field(context, "requestDeadline"));
            assertEquals(RequestStage.DELIVERING, context.stage());
            assertSame(item, context.route());
            assertFalse(future.isDone(), "admission owner must settle the timer fact");
        }
    }

    @Test
    void notificationFailureAfterPublicationKeepsTheExactRouteAndDoesNotRollback() {
        var context = context(10005L);
        var future = RequestProtocolTestSupport.register(lifecycle, context);
        var prefill = mock(PrefillEndpoint.class);
        var item = org.flexlb.balance.scheduler.SchedulingTestConfig.createRoute(freezeInputs(context), new Response(), null, null,
                prefill, null, null, System.currentTimeMillis());
        var failure = new IllegalStateException("route-ready notification failed");
        org.mockito.Mockito.doThrow(failure).when(prefill).signalRouteReady();
        try (var admission = lifecycle.claimAdmissionHandle(context.getRequestId(), future); var admissionCompletion2 = RequestProtocolTestSupport.finishOnExit(admission)) {
            assertNotNull(admission);
            assertSame(failure, assertThrows(IllegalStateException.class,
                    () -> lifecycle.commitRoute(item, RequestProtocolTestSupport.publication(() -> true))));
            assertSame(item, context.route());
            assertEquals(RequestStage.READY_TO_DELIVER, context.stage());
            assertEquals(0, failure.getSuppressed().length, "committed binding must not be rolled back");
        }
    }

    @Test
    void publicationFailureAfterHandoffPreservesBindingAndWakesDelivery() {
        var context = context(10008L);
        var future = RequestProtocolTestSupport.register(lifecycle, context);
        var prefill = mock(PrefillEndpoint.class);
        var item = org.flexlb.balance.scheduler.SchedulingTestConfig.createRoute(freezeInputs(context), new Response(), null, null,
                prefill, null, null, System.currentTimeMillis());
        var failure = new IllegalStateException("failure after handoff");
        var publication = new AbstractRequestScheduler.Publication() {
            private boolean published;
            @Override public void publish() { published = true; throw failure; }
            @Override public boolean published() { return published; }
        };
        try (var admission = lifecycle.claimAdmissionHandle(context.getRequestId(), future);
             var completion = RequestProtocolTestSupport.finishOnExit(admission)) {
            assertSame(failure, assertThrows(IllegalStateException.class,
                    () -> lifecycle.commitRoute(item, publication)));
            assertSame(item, context.route());
            assertEquals(RequestStage.READY_TO_DELIVER, context.stage());
            verify(prefill).signalRouteReady();
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void failedPublicationUnbindsTheRouteBeforeAdmissionCloses(boolean throwsFailure) {
        var context = context(10006L);
        var future = RequestProtocolTestSupport.register(lifecycle, context);
        var item = org.flexlb.balance.scheduler.SchedulingTestConfig.createRoute(freezeInputs(context), new Response(), null, null,
                mock(PrefillEndpoint.class), null, null, System.currentTimeMillis());
        var failure = new IllegalStateException("publication failed");
        try (var admission = lifecycle.claimAdmissionHandle(context.getRequestId(), future); var admissionCompletion3 = RequestProtocolTestSupport.finishOnExit(admission)) {
            assertNotNull(admission);
            if (throwsFailure) {
                assertSame(failure, assertThrows(IllegalStateException.class,
                        () -> lifecycle.commitRoute(item, RequestProtocolTestSupport.publication(() -> { throw failure; }))));
            } else {
                assertEquals(PlacementResult.Status.BLOCKED, lifecycle.commitRoute(item, RequestProtocolTestSupport.publication(() -> false)));
            }
            assertNull(context.route());
            assertEquals(RequestStage.ROUTING, context.stage());
        }
        assertEquals(RequestStage.QUEUED, context.stage());
    }

    @Test
    void futureCancelAfterQueuePublicationStillWinsBeforeAdmissionHandleCloses() {
        RequestContext context = context(10004L);
        CompletableFuture<Response> future = RequestProtocolTestSupport.register(lifecycle, context);
        PrefillEndpoint prefill = mock(PrefillEndpoint.class);
        context.setFuture(future);
        RequestRoute item = org.flexlb.balance.scheduler.SchedulingTestConfig.createRoute(freezeInputs(context), new Response(), null, null, prefill, null, null, System.currentTimeMillis());
        when(prefill.signalQueuedControl(item)).thenReturn(true);
        try (AdmissionHandle admission = lifecycle.claimAdmissionHandle(10004L, future); var admissionCompletion4 = RequestProtocolTestSupport.finishOnExit(admission)) {
            assertNotNull(admission);
            assertEquals(PlacementResult.Status.SUCCESS, lifecycle.commitRoute(item, RequestProtocolTestSupport.publication(() -> true)));
            assertTrue(future.cancel(false));
            assertTrue(future.isCancelled());
        }
        verify(prefill, org.mockito.Mockito.atLeastOnce()).signalQueuedControl(item);
        lifecycle.onQueuedItemControl(item);
        assertEquals(RequestState.Phase.CANCELLED, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle).getRequestState(10004L, 0L).state());
        assertEquals(0, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle).liveRequestCount());
    }

    @Test
    void globalCloseConsumesAnAcceptedInactivityFactBeforeShutdownFailure() throws Exception {
        CompletableFuture<Response> future = RequestProtocolTestSupport.register(lifecycle, context(10002L));
        RequestContext context = lifecycle.findRequestContext(10002L);
        lifecycle.enqueueInactivityDeadline(context, RequestProtocolTestSupport.<ExpirationTimer.InactivityDeadline>field(context, "inactivityDeadline"), Long.MAX_VALUE, () -> {
        });
        ((QueuedRequestScheduler) lifecycle).settleGlobalQueueClose(context);
        assertEquals(RequestState.Phase.TIMED_OUT, context.snapshot().state());
        assertFalse(future.get(5, TimeUnit.SECONDS).isSuccess());
    }

    @Test
    void terminalRecordPreservesIdentityWithoutRetainingRequestContext() throws Exception {
        RequestContext context = context(103L);
        CompletableFuture<Response> future = RequestProtocolTestSupport.register(lifecycle, context);
        lifecycle.cancel(103L, 0L, CancelReason.CLIENT_CANCELLED);
        assertEquals(StrategyErrorType.REQUEST_CANCELLED.getErrorCode(), future.get(5, TimeUnit.SECONDS).getCode());
        awaitCondition(() -> org.springframework.test.util.ReflectionTestUtils.getField(future, "target") == null);
        RequestState terminal = org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle).getRequestState(103L, 0L);
        assertEquals(RequestState.Phase.CANCELLED, terminal.state());
        assertNull(lifecycle.findRequestContext(103L));
        assertSame(terminal, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle).getRequestState(103L, 0L));
        assertEquals(StrategyErrorType.INVALID_REQUEST.getErrorCode(), RequestProtocolTestSupport.register(lifecycle, context(103L)).join().getCode());
    }

    @Test
    void globalWaitingRequestsHaveNoQuantityAdmissionLimit() {
        var low = context(1L);
        low.setSchedulingMetadata(SchedulingMetadata.explicit(10, Long.MAX_VALUE));
        var waiting = RequestProtocolTestSupport.register(lifecycle, low);
        for (long id = 2; id <= 1001; id++) {
            var high = context(id);
            high.setSchedulingMetadata(SchedulingMetadata.explicit(90, Long.MAX_VALUE));
            assertFalse(RequestProtocolTestSupport.register(lifecycle, high).isDone());
        }
        assertFalse(waiting.isDone(), "higher priority arrivals must not evict waiting requests");
        assertEquals(1001, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle).liveRequestCount());
        assertEquals(StrategyErrorType.INVALID_REQUEST.getErrorCode(),
                RequestProtocolTestSupport.register(lifecycle, context(1L)).join().getCode());
        for (long id = 1; id <= 1001; id++) {
            lifecycle.cancel(id, 0L, CancelReason.CLIENT_CANCELLED);
        }
        assertEquals(0, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle).liveRequestCount());
    }

    @Test
    void globalWaitingCancellationWaitsForDecisionOwnerTicket() throws Exception {
        QueuedRequestScheduler queue = (QueuedRequestScheduler) lifecycle;
        org.mockito.Mockito.doNothing().when(queue).signalControl(org.mockito.ArgumentMatchers.any());
        CompletableFuture<Response> future = RequestProtocolTestSupport.register(lifecycle, context(1002L));

        assertEquals(RequestState.Phase.CANCEL_REQUESTED,
                lifecycle.cancel(1002L, 0L, CancelReason.CLIENT_CANCELLED).state());
        assertFalse(future.isDone());
        RequestContext context = lifecycle.findRequestContext(1002L);
        verify(queue).signalControl(context);

        ((QueuedRequestScheduler) lifecycle).onGlobalControl(context);
        assertEquals(StrategyErrorType.REQUEST_CANCELLED.getErrorCode(),
                future.get(5, TimeUnit.SECONDS).getCode());
        assertEquals(RequestState.Phase.CANCELLED,
                org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle).getRequestState(1002L, 0L).state());
    }

    @Test
    void publicQueriesOwnTheirLockAndPrivateDecisionsStillRequireIt() {
        RequestProtocolTestSupport.register(lifecycle, context(102L));
        RequestContext requestContext = lifecycle.findRequestContext(102L);
        assertNull(requestContext.activeRoute());
        assertTrue(requestContext.isOpen());
        assertTrue(requestContext.isLiveGeneration());
        IllegalStateException failure = assertThrows(IllegalStateException.class, () -> org.springframework.test.util.ReflectionTestUtils.invokeMethod(lifecycle, "recordCancellationLocked", requestContext, CancelReason.CLIENT_CANCELLED, "client cancelled"));
        assertTrue(failure.getMessage().contains("requires context lock"));
        assertEquals(RequestState.Phase.QUEUED, requestContext.snapshot().state());
    }

    @Test
    void deliveredRequestsHaveNoExtraGlobalQuantityGate() {
        for (long id = 201; id <= 401; id++) {
            Registered registered = registerItem(id);
            assertEquals(PlacementResult.Status.SUCCESS,
                    commitRoute(lifecycle, registered));
            assertNotNull(RequestProtocolTestSupport.claimRoute(
                    lifecycle, registered.item(), () -> true));
        }
        assertEquals(201, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle).liveRequestCount());
    }

    @Test
    void itemEntryPointsRejectAnotherSchedulersLiveContext() {
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        AbstractRequestScheduler other = org.flexlb.balance.scheduler.SchedulerTestSupport.create(configService,
                mock(DeliveryMetricsReporter.class), mock(RequestSchedulerReporter.class),
                mock(RecentCacheKeyTraceReporter.class));
        Registered registered = registerItem(806L);
        CompletableFuture<Response> otherFuture = other.register(context(806L), StrategyErrorType.BATCH_SLO_EXPIRED);
        RequestRoute item = registered.item();
        try {
            try (AdmissionHandle admission = lifecycle.claimAdmissionHandle(806L, registered.future()); var admissionCompletion5 = RequestProtocolTestSupport.finishOnExit(admission)) {
                assertNotNull(admission);
                assertEquals(PlacementResult.Status.CLOSED, other.commitRoute(item, RequestProtocolTestSupport.publication(() -> {
                    throw new AssertionError("foreign scheduler published the route");
                })));
                assertEquals(RequestContext.RequestStage.ROUTING, item.ctx().stage());
                assertNull(item.ctx().route());
                assertEquals(PlacementResult.Status.SUCCESS, lifecycle.commitRoute(item, RequestProtocolTestSupport.publication(() -> true)));
            }
            synchronized (item.ctx()) {
                assertFalse(other.ownsPreparedDeliveryLocked(item.ctx(), item));
            }
            assertNull(other.claimDelivery(item, DeliveryClaimKind.BATCH_ENQUEUE, 41L, RequestProtocolTestSupport.handoff(() -> {
                throw new AssertionError("foreign scheduler transferred endpoint ownership");
            })));
            other.failDeliveryPreparation(item, new IllegalStateException("foreign failure"));
            other.onQueuedItemExpired(item);
            other.onQueuedItemControl(item);
            other.onQueueOfferFailure(item, new IllegalStateException("foreign queue failure"));
            other.onQueuedItemPreempted(item, item);
            assertEquals(RequestContext.RequestStage.READY_TO_DELIVER, item.ctx().stage());
            assertEquals(RequestContext.RequestStage.QUEUED, other.findRequestContext(806L).stage());
            assertFalse(registered.future().isDone());
            assertFalse(otherFuture.isDone());
            assertNotNull(lifecycle.claimDelivery(item, DeliveryClaimKind.BATCH_ENQUEUE, 41L, RequestProtocolTestSupport.handoff(() -> true)));
        } finally {
            RequestProtocolTestSupport.closeAdmissionAndAwaitMutations(other);
            other.closeOutstandingAndTerminalize();
            org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(other).timer().close();
            org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(other).closeRequestExecutors();
        }
    }

    @Test
    void failedPublicationKeepsSchedulingStageAndAllowsExactRetry() {
        Registered registered = registerItem(706L);
        RequestContext requestContext = lifecycle.findRequestContext(706L);
        try (AdmissionHandle admission = lifecycle.claimAdmissionHandle(706L, registered.future()); var admissionCompletion6 = RequestProtocolTestSupport.finishOnExit(admission)) {
            assertNotNull(admission);
            assertEquals(PlacementResult.Status.BLOCKED, lifecycle.commitRoute(registered.item(), RequestProtocolTestSupport.publication(() -> false)));
            assertEquals("ROUTING", String.valueOf(org.springframework.test.util.ReflectionTestUtils.getField(requestContext, "stage")));
            assertNull(requestContext.activeRoute());
            assertEquals(RequestState.Phase.QUEUED, requestContext.snapshot().state());
            assertEquals(PlacementResult.Status.SUCCESS, lifecycle.commitRoute(registered.item(), RequestProtocolTestSupport.publication(() -> true)));
            assertEquals("READY_TO_DELIVER", String.valueOf(org.springframework.test.util.ReflectionTestUtils.getField(requestContext, "stage")));
        }
    }

    @Test
    void acceptedQueuedCancellationSurvivesEndpointStopFailure() throws Exception {
        RequestContext context = context(707L);
        CompletableFuture<Response> future = RequestProtocolTestSupport.register(lifecycle, context);
        PrefillEndpoint prefill = mock(PrefillEndpoint.class);
        context.setFuture(future);
        RequestRoute item = org.flexlb.balance.scheduler.SchedulingTestConfig.createRoute(freezeInputs(context), new Response(),
                null, null, prefill, null, null, System.currentTimeMillis());
        when(prefill.signalQueuedControl(item)).thenReturn(true);
        try (AdmissionHandle admission = lifecycle.claimAdmissionHandle(707L, future); var admissionCompletion7 = RequestProtocolTestSupport.finishOnExit(admission)) {
            assertNotNull(admission);
            assertEquals(PlacementResult.Status.SUCCESS,
                    lifecycle.commitRoute(item, RequestProtocolTestSupport.publication(() -> true)));
        }

        assertEquals(RequestState.Phase.CANCEL_REQUESTED,
                lifecycle.cancel(707L, 0L, CancelReason.CLIENT_CANCELLED).state());
        assertFalse(future.isDone(), "local owner has not consumed its control ticket");
        lifecycle.onQueueOfferFailure(item, new IllegalStateException("endpoint stopped"));

        assertEquals(RequestState.Phase.CANCELLED, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle).getRequestState(707L, 0L).state());
        assertEquals(StrategyErrorType.REQUEST_CANCELLED.getErrorCode(),
                future.get(5, TimeUnit.SECONDS).getCode());
    }

    @Test
    void stoppedLocalOwnerFallsBackToSharedContinuationInsteadOfTimerThread() throws Exception {
        RequestContext context = context(708L);
        CompletableFuture<Response> future = RequestProtocolTestSupport.register(lifecycle, context);
        PrefillEndpoint prefill = mock(PrefillEndpoint.class);
        context.setFuture(future);
        RequestRoute item = org.flexlb.balance.scheduler.SchedulingTestConfig.createRoute(freezeInputs(context), new Response(), null, null, prefill, null, null, System.currentTimeMillis());
        try (AdmissionHandle admission = lifecycle.claimAdmissionHandle(708L, future); var admissionCompletion8 = RequestProtocolTestSupport.finishOnExit(admission)) {
            assertNotNull(admission);
            assertEquals(PlacementResult.Status.SUCCESS, lifecycle.commitRoute(item, RequestProtocolTestSupport.publication(() -> true)));
        }
        AtomicReference<Thread> cleanupThread = new AtomicReference<>();
        when(prefill.releaseRequest(item)).thenAnswer(invocation -> {
            assertFalse(Thread.holdsLock(context));
            cleanupThread.set(Thread.currentThread());
            return true;
        });
        RequestContext requestContext = lifecycle.findRequestContext(708L);
        lifecycle.enqueueInactivityDeadline(requestContext, RequestProtocolTestSupport.<ExpirationTimer.InactivityDeadline>field(requestContext, "inactivityDeadline"), Long.MAX_VALUE, () -> {
        });
        assertFalse(future.get(5, TimeUnit.SECONDS).isSuccess());
        lifecycle.runtime.continuations().awaitIdle();
        assertNotNull(cleanupThread.get());
        assertNotEquals(Thread.currentThread(), cleanupThread.get());
        verify(prefill).releaseRequest(item);
        assertEquals(RequestState.Phase.TIMED_OUT, requestContext.snapshot().state());
    }

    @Test
    void terminalRecordDropsUnconsumedExactOwnerFactReferences() {
        RequestContext context = context(709L);
        CompletableFuture<Response> future = RequestProtocolTestSupport.register(lifecycle, context);
        PrefillEndpoint prefill = mock(PrefillEndpoint.class);
        context.setFuture(future);
        RequestRoute item = org.flexlb.balance.scheduler.SchedulingTestConfig.createRoute(freezeInputs(context), new Response(), null, null, prefill, null, null, System.currentTimeMillis());
        when(prefill.signalQueuedControl(item)).thenReturn(true);
        try (AdmissionHandle admission = lifecycle.claimAdmissionHandle(709L, future); var admissionCompletion9 = RequestProtocolTestSupport.finishOnExit(admission)) {
            assertNotNull(admission);
            assertEquals(PlacementResult.Status.SUCCESS, lifecycle.commitRoute(item, RequestProtocolTestSupport.publication(() -> true)));
        }
        RequestContext requestContext = lifecycle.findRequestContext(709L);
        RequestContinuationExecutor continuations = (RequestContinuationExecutor) org.springframework.test.util.ReflectionTestUtils.getField(lifecycle, "continuations");
        lifecycle.enqueueInactivityDeadline(requestContext, RequestProtocolTestSupport.<ExpirationTimer.InactivityDeadline>field(requestContext, "inactivityDeadline"), Long.MAX_VALUE, () -> {
        });
        try {
            assertTrue(RequestProtocolTestSupport.closeAdmissionAndAwaitMutations(lifecycle));
            lifecycle.closeOutstandingAndTerminalize();
            lifecycle.runtime.continuations().awaitIdle();
            assertEquals("FINISHED", String.valueOf(org.springframework.test.util.ReflectionTestUtils.getField(requestContext, "stage")));
            assertNull(requestContext.activeRoute());
            assertNull(requestContext.route());
            assertNull(RequestProtocolTestSupport.field(requestContext, "requestDeadline"));
            assertNull(RequestProtocolTestSupport.field(requestContext, "decisionDeadline"));
            assertNull(RequestProtocolTestSupport.<ExpirationTimer.InactivityDeadline>field(requestContext, "inactivityDeadline"));
            assertNull(lifecycle.findRequestContext(709L));
            RequestState terminal = org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle).getRequestState(709L, 0L);
            assertTrue(terminal.state().isTerminal());
            assertSame(terminal, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle).getRequestState(709L, 0L));
        } finally {
            org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(lifecycle).timer().close();
            org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(lifecycle).closeRequestExecutors();
        }
    }

    @Test
    void endpointStopCannotReplaceAnAcceptedTimeoutDuringCleanup() throws Exception {
        RequestContext context = context(710L);
        CompletableFuture<Response> future = RequestProtocolTestSupport.register(lifecycle, context);
        PrefillEndpoint prefill = mock(PrefillEndpoint.class);
        context.setFuture(future);
        RequestRoute item = org.flexlb.balance.scheduler.SchedulingTestConfig.createRoute(freezeInputs(context), new Response(), null, null, prefill, null, null, System.currentTimeMillis());
        try (AdmissionHandle admission = lifecycle.claimAdmissionHandle(710L, future); var admissionCompletion10 = RequestProtocolTestSupport.finishOnExit(admission)) {
            assertNotNull(admission);
            assertEquals(PlacementResult.Status.SUCCESS, lifecycle.commitRoute(item, RequestProtocolTestSupport.publication(() -> true)));
        }
        CountDownLatch entered = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        when(prefill.releaseRequest(item)).thenAnswer(call -> {
            assertFalse(Thread.holdsLock(context));
            entered.countDown();
            assertTrue(release.await(5, TimeUnit.SECONDS));
            return true;
        });
        try {
            lifecycle.enqueueInactivityDeadline(context, RequestProtocolTestSupport.<ExpirationTimer.InactivityDeadline>field(context, "inactivityDeadline"), Long.MAX_VALUE, () -> {
            });
            assertTrue(entered.await(2, TimeUnit.SECONDS));
            lifecycle.onQueueOfferFailure(item, new IllegalStateException("endpoint stopped"));
            assertEquals(RequestState.Phase.TIMED_OUT, context.snapshot().state());
        } finally {
            release.countDown();
        }
        assertFalse(future.get(5, TimeUnit.SECONDS).isSuccess());
        lifecycle.runtime.continuations().awaitIdle();
        verify(prefill).releaseRequest(item);
        assertEquals(RequestState.Phase.TIMED_OUT, context.snapshot().state());
    }

    @Test
    void admissionHandleDefersCancellationUntilItsExactCapabilityCloses() {
        CompletableFuture<Response> future = RequestProtocolTestSupport.register(lifecycle, context(301L));
        AdmissionHandle scope =
                lifecycle.claimAdmissionHandle(301L, future);
        assertNotNull(scope);

        RequestState requested = lifecycle.cancel(
                301L, 0L, CancelReason.CLIENT_CANCELLED);

        assertEquals(RequestState.Phase.CANCEL_REQUESTED, requested.state());
        assertFalse(future.isDone(),
                "the admission mutation still owns rollback and terminal cleanup");

        scope.finish();

        assertEquals(StrategyErrorType.REQUEST_CANCELLED.getErrorCode(),
                future.join().getCode());
        assertEquals(RequestState.Phase.CANCELLED,
                org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle).getRequestState(301L, 0L).state());
    }

    @Test
    void repeatedCancellationDuringAdmissionKeepsTheFirstCause() throws Exception {
        CompletableFuture<Response> future = RequestProtocolTestSupport.register(lifecycle, context(303L));
        AdmissionHandle admission = lifecycle.claimAdmissionHandle(303L, future);
        assertNotNull(admission);

        lifecycle.cancel(303L, 0L, CancelReason.CLIENT_CANCELLED);
        lifecycle.cancel(303L, 0L, CancelReason.DEADLINE_EXCEEDED);
        assertFalse(future.isDone());

        admission.finish();

        assertEquals(StrategyErrorType.REQUEST_CANCELLED.getErrorCode(),
                future.get(5, TimeUnit.SECONDS).getCode());
        assertEquals(RequestState.Phase.CANCELLED,
                org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle).getRequestState(303L, 0L).state());
        assertEquals(0, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle).liveRequestCount());
    }

    @Test
    void workerTerminalDuringAdmissionWinsOverRetirementAndAdmissionFailure() throws Exception {
        Registered registered = registerItem(7111L);
        RequestRoute original = registered.item();
        RequestContext context = original.ctx();
        PrefillEndpoint prefill = mock(PrefillEndpoint.class);
        when(prefill.getStatus()).thenReturn(org.flexlb.dao.master.WorkerStatus.createDiscovered(
                org.flexlb.dao.route.RoleType.PREFILL, "test", "127.0.0.1", 8000, 8001, "test"));
        RequestRoute route = org.flexlb.balance.scheduler.SchedulingTestConfig.createRoute(context, SchedulingTestConfig.routeResponse(original), null, null,
                prefill, original.decodeEp(), original.decodeReservation(), original.enqueuedAtMs());
        var callbacks = new java.util.concurrent.atomic.AtomicInteger();
        var published = registered.future().thenAccept(response -> {
            assertFalse(Thread.holdsLock(context));
            callbacks.incrementAndGet();
        });
        try (var admission = lifecycle.claimAdmissionHandle(route.requestId(), registered.future());
                var finish = RequestProtocolTestSupport.finishOnExit(admission)) {
            assertNotNull(admission);
            assertEquals(PlacementResult.Status.SUCCESS,
                    lifecycle.commitRoute(route, RequestProtocolTestSupport.publication(() -> true)));
            lifecycle.onPrefillGenerationRetired(prefill, route);
            lifecycle.runtime.continuations().awaitIdle();
            RequestProtocolTestSupport.applyPrefillStatus(lifecycle, context, prefill,
                    org.flexlb.dao.route.RoleType.PREFILL,
                    org.flexlb.balance.endpoint.PrefillState.PrefillRequestStatus.terminal(route,
                            org.flexlb.balance.endpoint.PrefillState.PrefillRequestStatus.Kind.FAILED, 42L));
            assertFalse(registered.future().isDone(), "the routing owner still retains all terminal evidence");
            admission.terminate(Response.error(StrategyErrorType.RESOURCE_EXHAUSTED));
            admission.finish();
        }
        assertEquals(StrategyErrorType.WORKER_EXECUTION_FAILED.getErrorCode(),
                registered.future().get(5, TimeUnit.SECONDS).getCode());
        published.get(5, TimeUnit.SECONDS);
        lifecycle.runtime.continuations().awaitIdle();
        assertEquals(1, callbacks.get());
        assertEquals(RequestStage.FINISHED, context.stage());
        assertEquals(RequestState.Phase.FAILED, context.snapshot().state());
        assertEquals(0, SchedulerTestSupport.repository(lifecycle).liveRequestCount());
        verify(route.decodeEp()).release(route.decodeReservation(), DecodeResources.ReleaseReason.COUNTERPART_FINISHED);
        org.mockito.Mockito.verify(prefill, org.mockito.Mockito.never()).releaseRequest(route);
    }

    @Test
    void admissionFailurePreservesAnEarlierCancellation() throws Exception {
        CompletableFuture<Response> future = RequestProtocolTestSupport.register(lifecycle, context(304L));
        AdmissionHandle admission = lifecycle.claimAdmissionHandle(304L, future);
        assertNotNull(admission);

        lifecycle.cancel(304L, 0L, CancelReason.CLIENT_CANCELLED);
        admission.terminate(Response.error(StrategyErrorType.RESOURCE_EXHAUSTED));
        admission.finish();

        assertEquals(StrategyErrorType.REQUEST_CANCELLED.getErrorCode(),
                future.get(5, TimeUnit.SECONDS).getCode());
        assertEquals(RequestState.Phase.CANCELLED,
                org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle).getRequestState(304L, 0L).state());
        assertEquals(0, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle).liveRequestCount());
    }

    @Test
    void shutdownIntentSurvivesAdmissionFailureWithoutAnItem() throws Exception {
        CompletableFuture<Response> future = RequestProtocolTestSupport.register(lifecycle, context(305L));
        AdmissionHandle admission = lifecycle.claimAdmissionHandle(305L, future);
        assertNotNull(admission);
        var requestContext = lifecycle.findRequestContext(305L);
        synchronized (requestContext) {
            assertNull(lifecycle.claimFinalizationLocked(requestContext, requestContext.decideShutdownLocked()));
        }
        admission.terminate(Response.error(StrategyErrorType.RESOURCE_EXHAUSTED));
        assertEquals(StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(), future.get(5, TimeUnit.SECONDS).getCode());
        assertEquals(RequestState.Phase.FAILED, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle).getRequestState(305L, 0L).state());
    }

    @Test
    void queueDecisionResponsePublishesOutsideTheDecisionCaller() throws Exception {
        CompletableFuture<Response> future = RequestProtocolTestSupport.register(lifecycle, context(302L));
        CountDownLatch published = new CountDownLatch(1);
        AtomicReference<String> callbackThread = new AtomicReference<>();
        future.thenAccept(response -> {
            callbackThread.set(Thread.currentThread().getName());
            published.countDown();
        });

        Response rejection = Response.error(StrategyErrorType.RESOURCE_EXHAUSTED);
        assertTrue(lifecycle.publishDecisionResponseAsync(
                302L, future, rejection));
        assertTrue(published.await(5, TimeUnit.SECONDS));
        assertNotEquals(Thread.currentThread().getName(), callbackThread.get());
        assertEquals(StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(),
                future.get(5, TimeUnit.SECONDS).getCode());
    }

    @Test
    void shutdownGateWaitsForTheExactAdmissionHandleAndRejectsNewWork()
            throws Exception {
        CompletableFuture<Response> heldFuture =
                RequestProtocolTestSupport.register(lifecycle, context(401L));
        AdmissionHandle held =
                lifecycle.claimAdmissionHandle(401L, heldFuture);
        assertNotNull(held);
        ExecutorService executor = Executors.newSingleThreadExecutor();
        try {
            Future<Boolean> shutdownOwner =
                    executor.submit(() -> RequestProtocolTestSupport.closeAdmissionAndAwaitMutations(lifecycle));
            awaitCondition(org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle)::isClosed);
            assertFalse(shutdownOwner.isDone(),
                    "shutdown must not overtake an exact admission mutation");

            CompletableFuture<Response> rejected =
                    RequestProtocolTestSupport.register(lifecycle, context(402L));
            assertEquals(StrategyErrorType.DISPATCH_FAILED.getErrorCode(),
                    rejected.join().getCode());

            held.finish();
            assertTrue(shutdownOwner.get(5, TimeUnit.SECONDS));
            lifecycle.closeOutstandingAndTerminalize();
            org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(lifecycle).timer().close();
            org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(lifecycle).closeRequestExecutors();
            assertEquals(StrategyErrorType.DISPATCH_FAILED.getErrorCode(),
                    heldFuture.get(5, TimeUnit.SECONDS).getCode());
        } finally {
            executor.shutdownNow();
        }
    }

    @Test
    void cancelRequiresTheExpectedBatchGenerationAndUnknownIdsStayAbsent() {
        CompletableFuture<Response> future = RequestProtocolTestSupport.register(lifecycle, context(501L));

        assertNull(lifecycle.cancel(
                999L, 0L, CancelReason.CLIENT_CANCELLED));
        assertNull(lifecycle.cancel(
                501L, 91L, CancelReason.CLIENT_CANCELLED));
        assertFalse(future.isDone());

        RequestState exact = lifecycle.cancel(
                501L, 0L, CancelReason.CLIENT_CANCELLED);
        assertNotNull(exact);
        assertEquals(StrategyErrorType.REQUEST_CANCELLED.getErrorCode(),
                future.join().getCode());
    }

    @Test
    void publishedQueueDeadlineReleasesLocalReservationWithoutEngineCancel() {
        Registered registered = registerItem(602L);
        assertEquals(PlacementResult.Status.SUCCESS, commitRoute(lifecycle, registered));
        RequestContext requestContext = lifecycle.findRequestContext(602L);
        synchronized (requestContext) {
            requestContext.scheduler().acceptPrefillStatus(requestContext, registered.item().prefillEp(), org.flexlb.dao.route.RoleType.PREFILL, org.flexlb.balance.endpoint.PrefillState.PrefillRequestStatus.active(registered.item()), System.currentTimeMillis());
        }
        lifecycle.cancel(602L, 0L, CancelReason.DEADLINE_EXCEEDED);
        lifecycle.runtime.continuations().awaitIdle();
        assertEquals(RequestState.Phase.TIMED_OUT, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle).getRequestState(602L, 0L).state());
        assertEquals(0, lifecycle.requests.liveRequestCount());
        verify(registered.item().prefillEp()).releaseRequest(registered.item());
        verify(registered.item().decodeEp()).release(registered.item().decodeReservation(), DecodeResources.ReleaseReason.COUNTERPART_FINISHED);
    }

    @Test
    void expiredArrivalDoesNotDisturbTheWaitingRequests() throws Exception {
        var low = RequestProtocolTestSupport.register(lifecycle, context(1));
        var expired = context(2);
        expired.setSchedulingMetadata(SchedulingMetadata.explicit(90, System.currentTimeMillis() - 1L));
        // QUEUE expiry before placement is admission capacity exhaustion (8431).
        assertEquals(StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(),
                RequestProtocolTestSupport.register(lifecycle, expired).get(5, TimeUnit.SECONDS).getCode());
        assertFalse(low.isDone());
        assertEquals(1, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle).liveRequestCount());
    }

    @Test
    void concurrentGlobalAdmissionRetainsEveryUniqueRequest() throws Exception {
        try (var executor = Executors.newFixedThreadPool(8)) {
            List<Future<CompletableFuture<Response>>> futures = new ArrayList<>();
            for (long id = 1; id <= 128; id++) {
                long requestId = id;
                futures.add(executor.submit(() -> RequestProtocolTestSupport.register(lifecycle, context(requestId))));
            }
            for (var future : futures) {
                assertFalse(future.get(5, TimeUnit.SECONDS).isDone());
            }
            assertEquals(128, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle).liveRequestCount());
        }
    }

    @Test
    void oldDeliveryAndPreemptionCapabilitiesCannotReachAReusedRequestId() throws Exception {
        Registered registered = registerItemWithCancelTarget(703L);
        assertEquals(PlacementResult.Status.SUCCESS, commitRoute(lifecycle, registered));
        RequestContext old = lifecycle.findRequestContext(703L);
        DeliveryClaim delivery = RequestProtocolTestSupport.claimBatchWithoutPrediction(lifecycle, registered.item(), 17L, () -> true);
        assertNotNull(delivery);
        PreemptionRegistration preemption = lifecycle.tryClaim(new DecodeResources.ReservationHandle(1L, 703L, 1L), 19L, "victim").orElseThrow();
        RequestProtocolTestSupport.expireInactiveRequest(lifecycle, old, RequestProtocolTestSupport.<Long>inspect(lifecycle, old, "inactivityExpiresAtMsLocked"));
        registered.future().join();
        assertTrue(lifecycle.requests.isCurrent(old), "sender still owns prepared delivery");
        delivery.item.ctx().scheduler().completeDelivery(delivery, org.flexlb.balance.delivery.DeliveryResult.notSent(new IllegalStateException("expired before send")));
        lifecycle.runtime.continuations().awaitIdle();
        assertTrue(org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle).removeExactTerminal(org.flexlb.balance.scheduler.SchedulerTestSupport.terminalRecord(lifecycle, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle).getRequestState(703L, 0L)), Long.MAX_VALUE));
        CompletableFuture<Response> replacement = RequestProtocolTestSupport.register(lifecycle, context(703L));
        assertSame(registered.future(), registered.item().future());
        assertNotSame(replacement, registered.item().future());
        assertThrows(IllegalStateException.class, () -> old.setFuture(replacement));
        assertSame(registered.future(), registered.item().future());
        assertThrows(IllegalStateException.class, () -> delivery.item.ctx().scheduler().completeDelivery(delivery, org.flexlb.balance.delivery.DeliveryResult.delivered()));
        assertFalse(lifecycle.completePreemption(preemption, "late engine cancellation"));
        assertFalse(lifecycle.releasePreemption(preemption));
        lifecycle.cancelRequest(old, 0L, CancelReason.CLIENT_CANCELLED);
        lifecycle.onQueuedItemExpired(registered.item());
        lifecycle.onQueuedItemControl(registered.item());
        lifecycle.onQueueOfferFailure(registered.item(), new IllegalStateException("late queue failure"));
        lifecycle.onQueuedItemPreempted(registered.item(), registered.item());
        RequestContext current = lifecycle.findRequestContext(703L);
        RequestState oldState = old.snapshot();
        RequestState currentState = current.snapshot();
        lifecycle.onPrefillStatus(old, registered.item().prefillEp(), org.flexlb.dao.route.RoleType.PREFILL,
                org.flexlb.balance.endpoint.PrefillState.PrefillRequestStatus.active(registered.item()));
        lifecycle.onPrefillStatus(old, registered.item().prefillEp(), org.flexlb.dao.route.RoleType.PREFILL,
                org.flexlb.balance.endpoint.PrefillState.PrefillRequestStatus.terminal(registered.item(),
                        org.flexlb.balance.endpoint.PrefillState.PrefillRequestStatus.Kind.COMPLETED, 0L));
        lifecycle.onDecodeStatus(old, registered.item().decodeEp(),
                DecodeResources.DecodeRequestStatus.active(registered.item().decodeReservation()));
        lifecycle.onDecodeStatus(old, registered.item().decodeEp(),
                DecodeResources.DecodeRequestStatus.terminal(registered.item().decodeReservation(), 0L));
        lifecycle.runtime.continuations().awaitIdle();
        assertEquals(oldState, old.snapshot(), "late Worker status must not update the archived Context");
        assertEquals(currentState, current.snapshot(), "late Worker status must not update the reused ID");
        assertFalse(replacement.isDone());
        assertEquals(RequestState.Phase.QUEUED, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle).getRequestState(703L, 0L).state());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void decodeAllocationReconcilesOnlyNotFoundProtocol(boolean acceptedCancel) {
        Registered registered = registerItemWithCancelTarget(706L);
        assertEquals(PlacementResult.Status.SUCCESS, commitRoute(lifecycle, registered));
        var delivery = RequestProtocolTestSupport.claimBatch(lifecycle, registered.item(), 17L, () -> true);
        assertNotNull(delivery);
        delivery.item.ctx().scheduler().completeDelivery(delivery, org.flexlb.balance.delivery.DeliveryResult.delivered());
        lifecycle.runtime.continuations().awaitIdle();
        var claim = lifecycle.tryClaim(new DecodeResources.ReservationHandle(1L, 706L, 1L), 20L, "victim").orElseThrow();
        assertTrue(lifecycle.updatePreemption(claim, org.flexlb.balance.preemption.PreemptionCancelPhase.CANCEL_IN_FLIGHT));
        assertTrue(lifecycle.updatePreemption(claim, acceptedCancel
                ? org.flexlb.balance.preemption.PreemptionCancelPhase.CANCEL_REQUESTED
                : org.flexlb.balance.preemption.PreemptionCancelPhase.NOT_FOUND_STALE));
        var source = registered.item().decodeEp();
        when(source.reconcilePreemptionResources(org.mockito.ArgumentMatchers.eq(20L), org.mockito.ArgumentMatchers.any())).thenReturn(true);
        org.mockito.Mockito.doAnswer(call -> {
            assertFalse(Thread.holdsLock(registered.item().ctx()));
            return null;
        }).when(source).publishCapacityRelease();
        lifecycle.onDecodeStatus(registered.item().ctx(), source, DecodeResources.DecodeRequestStatus.allocated(registered.item().decodeReservation()));
        lifecycle.runtime.continuations().awaitIdle();
        assertSame(acceptedCancel ? claim : null, registered.item().ctx().preemption());
        if (acceptedCancel) {
            verify(source, org.mockito.Mockito.never()).reconcilePreemptionResources(org.mockito.ArgumentMatchers.anyLong(), org.mockito.ArgumentMatchers.any());
        } else {
            verify(source).reconcilePreemptionResources(20L, DecodeResources.PreemptionUpdate.active(registered.item().decodeReservation()));
            verify(source).publishCapacityRelease();
        }
    }

    @Test
    void decodeAllocationSettlesCleanupAfterRetainedTerminalAndNotFound() {
        Registered registered = registerItemWithCancelTarget(707L);
        assertEquals(PlacementResult.Status.SUCCESS, commitRoute(lifecycle, registered));
        var delivery = RequestProtocolTestSupport.claimBatch(lifecycle, registered.item(), 17L, () -> true);
        delivery.item.ctx().scheduler().completeDelivery(delivery, org.flexlb.balance.delivery.DeliveryResult.delivered());
        lifecycle.runtime.continuations().awaitIdle();
        var context = registered.item().ctx();
        var source = registered.item().decodeEp();
        when(source.hasOwnedResources(registered.item().decodeReservation())).thenReturn(true);
        var claim = lifecycle.tryClaim(new DecodeResources.ReservationHandle(1L, 707L, 1L), 21L, "victim").orElseThrow();
        assertTrue(lifecycle.updatePreemption(claim, org.flexlb.balance.preemption.PreemptionCancelPhase.CANCEL_IN_FLIGHT));
        delivery.recordWorkerCompletion(registered.item());
        delivery.item.ctx().scheduler().settleDelivery(delivery);
        AbstractRequestScheduler.PublicationPermit permit;
        RequestContext.ResponseResult response;
        synchronized (context) {
            context.retainPreemptionTerminalLocked(claim, DeferredTerminal.worker(WorkerTerminalSource.PREFILL_ENDPOINT, true, 0L));
            permit = lifecycle.selectDeliveryFailureLocked(context, registered.item(), org.flexlb.balance.delivery.DeliveryResult.Status.PREFILL_REJECTED,
                    "Prefill rejected delivery while cancellation was pending");
            response = context.selectedResponse();
            var pass = context.beginCleanup();
            context.finishCleanup(pass, true, false);
        }
        if (permit != null) {
            try { AbstractRequestScheduler.completeFutureResult(permit, response); }
            finally { permit.closePublication(); }
        }
        assertTrue(lifecycle.updatePreemption(claim, org.flexlb.balance.preemption.PreemptionCancelPhase.NOT_FOUND_STALE));
        assertEquals(RequestStage.FINALIZING, context.stage());
        assertFalse(claim.isFinished(), "NOT_FOUND is not independent Decode release proof");
        assertSame(context, lifecycle.findRequestContext(707L));
        verify(source, org.mockito.Mockito.never()).release(org.mockito.ArgumentMatchers.any(), org.mockito.ArgumentMatchers.any());
        when(source.reconcilePreemptionResources(org.mockito.ArgumentMatchers.eq(21L), org.mockito.ArgumentMatchers.any())).thenReturn(true);

        lifecycle.onDecodeStatus(registered.item().ctx(), source, DecodeResources.DecodeRequestStatus.allocated(registered.item().decodeReservation()));
        lifecycle.runtime.continuations().awaitIdle();

        verify(source).reconcilePreemptionResources(21L, DecodeResources.PreemptionUpdate.finished(registered.item().decodeReservation()));
        assertTrue(claim.isFinished());
        assertEquals(RequestStage.FINISHED, context.stage());
        assertTrue(claim.requestResolution().toCompletableFuture().isDone());
        assertNull(lifecycle.findRequestContext(707L));
        assertNull(context.preemption());
    }

    @Test
    void decodeAllocationCarriesSettledResourcesIntoTheFirstTerminalCleanup() {
        Registered registered = registerItemWithCancelTarget(711L);
        assertEquals(PlacementResult.Status.SUCCESS, commitRoute(lifecycle, registered));
        var delivery = RequestProtocolTestSupport.claimBatch(lifecycle, registered.item(), 17L, () -> true);
        delivery.item.ctx().scheduler().completeDelivery(delivery, org.flexlb.balance.delivery.DeliveryResult.delivered());
        lifecycle.runtime.continuations().awaitIdle();
        var context = registered.item().ctx();
        var claim = lifecycle.tryClaim(new DecodeResources.ReservationHandle(1L, 711L, 1L), 22L, "victim").orElseThrow();
        assertTrue(lifecycle.updatePreemption(claim, org.flexlb.balance.preemption.PreemptionCancelPhase.CANCEL_IN_FLIGHT));
        assertTrue(lifecycle.updatePreemption(claim, org.flexlb.balance.preemption.PreemptionCancelPhase.NOT_FOUND_STALE));
        synchronized (context) {
            assertFalse(context.hasCleanup());
            context.retainPreemptionTerminalLocked(claim,
                    DeferredTerminal.worker(WorkerTerminalSource.PREFILL_ENDPOINT, true, 0L));
        }
        var source = registered.item().decodeEp();
        when(source.reconcilePreemptionResources(22L,
                DecodeResources.PreemptionUpdate.finished(registered.item().decodeReservation()))).thenReturn(true);
        org.mockito.Mockito.doAnswer(call -> {
            assertFalse(Thread.holdsLock(context), "capacity notification must follow the resource CAS outside the request lock");
            return null;
        }).when(source).publishCapacityRelease();

        lifecycle.onDecodeStatus(registered.item().ctx(), source, DecodeResources.DecodeRequestStatus.allocated(registered.item().decodeReservation()));
        lifecycle.runtime.continuations().awaitIdle();

        verify(source).reconcilePreemptionResources(22L,
                DecodeResources.PreemptionUpdate.finished(registered.item().decodeReservation()));
        verify(source).publishCapacityRelease();
        verify(source, org.mockito.Mockito.never()).release(org.mockito.ArgumentMatchers.any(), org.mockito.ArgumentMatchers.any());
        assertTrue(claim.isFinished());
        assertTrue(claim.requestResolution().toCompletableFuture().isDone());
        assertEquals(RequestStage.FINISHED, context.stage());
        assertEquals(RequestState.Phase.COMPLETED, context.snapshot().state());
        assertNull(lifecycle.findRequestContext(711L));
    }

    @Test
    void deliveryCannotConsumeAnotherRoutesHandoffReceipt() {
        Registered registered = registerItem(705L);
        assertEquals(PlacementResult.Status.SUCCESS, commitRoute(lifecycle, registered));
        Registered other = registerItem(706L);
        var wrongReceipt = mock(org.flexlb.balance.endpoint.DecodeEndpoint.EngineDispatchPermit.class);
        assertThrows(IllegalArgumentException.class,
                () -> lifecycle.claimDelivery(registered.item(), DeliveryClaimKind.BATCH_ENQUEUE, 7L, wrongReceipt));
        assertNull(registered.item().ctx().delivery());
        assertEquals(RequestStage.READY_TO_DELIVER, registered.item().ctx().stage());
        assertFalse(registered.future().isDone());
    }

    @Test
    void invalidBatchIdentityCannotTransferEndpointOwnership() {
        Registered registered = registerItem(704L);
        assertEquals(PlacementResult.Status.SUCCESS, commitRoute(lifecycle, registered));
        var transfer = mock(java.util.function.BooleanSupplier.class);

        assertThrows(IllegalArgumentException.class,
                () -> lifecycle.claimDelivery(registered.item(), DeliveryClaimKind.BATCH_ENQUEUE,
                        0L, RequestProtocolTestSupport.handoff(transfer)));

        org.mockito.Mockito.verify(transfer, org.mockito.Mockito.never()).getAsBoolean();
        assertEquals(RequestState.Phase.QUEUED, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle).getRequestState(704L, 0L).state());
        assertFalse(registered.future().isDone());
    }

    @Test
    void invalidResultDoesNotConsumeDeliveryAndDuplicateResultCannotChangeItsOutcome() throws Exception {
        Registered registered = registerItem(705L);
        assertEquals(PlacementResult.Status.SUCCESS, commitRoute(lifecycle, registered));
        DeliveryClaim claim = RequestProtocolTestSupport.claimBatch(lifecycle, registered.item(), 23L, () -> true);
        assertNotNull(claim);

        assertThrows(NullPointerException.class, () -> claim.item.ctx().scheduler().completeDelivery(claim, null));
        assertEquals(RequestState.Phase.DISPATCHING, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle).getRequestState(705L, 23L).state());
        claim.item.ctx().scheduler().completeDelivery(claim, org.flexlb.balance.delivery.DeliveryResult.delivered());
        assertTrue(registered.future().get(5, TimeUnit.SECONDS).isSuccess());
        assertThrows(IllegalStateException.class, () -> claim.item.ctx().scheduler().completeDelivery(claim,
                org.flexlb.balance.delivery.DeliveryResult.notSent(new IllegalStateException("duplicate failure"))));

        assertEquals(RequestState.Phase.ACKNOWLEDGED, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle).getRequestState(705L, 23L).state());
        org.mockito.Mockito.verify(registered.item().decodeEp(), org.mockito.Mockito.never())
                .release(registered.item().decodeReservation(), DecodeResources.ReleaseReason.NOT_SENT);
    }

    private RequestContext context(long requestId) {
        return RequestProtocolTestSupport.context(config, requestId);
    }

    @Test
    void preemptionClaimCapturesTheExactCancelTargetAndRejectsStaleReservation() {
        Registered registered = registerItemWithCancelTarget(709L);
        assertEquals(PlacementResult.Status.SUCCESS, commitRoute(lifecycle, registered));
        assertTrue(lifecycle.tryClaim(new DecodeResources.ReservationHandle(1L, 709L, 2L), 23L, "stale").isEmpty());
        PreemptionRegistration claim = lifecycle.tryClaim(new DecodeResources.ReservationHandle(1L, 709L, 1L), 24L, "victim").orElseThrow();
        assertEquals(new org.flexlb.balance.preemption.CancelTarget("127.0.0.1", 8090), claim.cancelTarget());
        registered.item().prefill().setServerIp("changed-after-claim");
        assertEquals("127.0.0.1", claim.cancelTarget().prefillIp());
        registered.item().prefill().setGrpcPort(0);
        assertTrue(lifecycle.tryClaim(new DecodeResources.ReservationHandle(1L, 709L, 1L), 26L, "already claimed").isEmpty());
        lifecycle.releasePreemption(claim);
    }

    @Test
    void exactPreemptionRouteWithUnroutableCancelTargetIsAControlFailureAndDoesNotInstallClaim() {
        Registered registered = registerItem(710L);
        registered.item().prefill().setGrpcPort(0);
        assertEquals(PlacementResult.Status.SUCCESS, commitRoute(lifecycle, registered));
        assertThrows(IllegalStateException.class, () -> lifecycle.tryClaim(new DecodeResources.ReservationHandle(1L, 710L, 1L), 25L, "victim"));
        assertNull(registered.item().ctx().preemption());
    }

    @Test
    void preemptionClaimRejectsAnotherDecodeGenerationWithTheSameRequestAndToken() {
        Registered registered = registerItemWithCancelTarget(711L);
        assertEquals(PlacementResult.Status.SUCCESS, commitRoute(lifecycle, registered));
        var current = registered.item().decodeReservation();
        var stale = new DecodeResources.ReservationHandle(current.endpointGenerationId() + 1L,
                current.requestId(), current.reservationToken());
        assertTrue(lifecycle.tryClaim(stale, 27L, "old endpoint").isEmpty());
        assertNull(registered.item().ctx().preemption());
        PreemptionRegistration valid = lifecycle.tryClaim(current, 28L, "current endpoint").orElseThrow();
        assertEquals(new org.flexlb.balance.preemption.CancelTarget("127.0.0.1", 8090), valid.cancelTarget());
        lifecycle.releasePreemption(valid);
    }

    private Registered registerItemWithCancelTarget(long requestId) {
        Registered registered = registerItem(requestId);
        var prefill = new org.flexlb.dao.loadbalance.ServerStatus();
        prefill.setServerIp("127.0.0.1");
        prefill.setGrpcPort(8090);
        RequestRoute original = registered.item();
        return new Registered(org.flexlb.balance.scheduler.SchedulingTestConfig.createRoute(original.ctx(), SchedulingTestConfig.routeResponse(original), prefill,
                original.decode(), original.prefillEp(), original.decodeEp(), original.decodeReservation(),
                original.enqueuedAtMs()), registered.future());
    }

    private Registered registerItem(long requestId) {
        RequestContext context = context(requestId);
        CompletableFuture<Response> future = RequestProtocolTestSupport.register(lifecycle, context);
        DecodeEndpoint decode = RequestProtocolTestSupport.decodeEndpoint();
        DecodeResources.ReservationHandle reservation =
                new DecodeResources.ReservationHandle(1L, requestId, 1L);
        context.setFuture(future);
        return new Registered(
                org.flexlb.balance.scheduler.SchedulingTestConfig.createRoute(freezeInputs(context),
                        new Response(),
                        null,
                        null,
                        null,
                        decode,
                        reservation,
                        System.currentTimeMillis()),
                future);
    }
}
