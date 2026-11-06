package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.scheduler.RequestProtocolTestSupport.Registered;
import org.flexlb.balance.scheduler.ResponseCompletionExecutor.RequestFuture;
import org.flexlb.balance.scheduler.SchedulerTestSupport.Stage;
import org.flexlb.config.ConfigService;
import org.flexlb.config.ConfigTestSupport;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.TheoryCacheHitReporter;
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
        lifecycle =
                org.flexlb.balance.scheduler.SchedulerTestSupport.create(
                        configService,
                        mock(DeliveryMetricsReporter.class),
                        mock(RequestSchedulerReporter.class),
                        mock(TheoryCacheHitReporter.class));
    }

    @AfterEach
    void tearDown() {
        if (RequestProtocolTestSupport.closeAdmissionAndAwaitMutations(lifecycle)) {
            lifecycle.closeOutstandingAndTerminalize();
            SchedulerTestSupport.timer(
                            org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(lifecycle))
                    .close();
            SchedulerTestSupport.closeRequestExecutors(
                    org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(lifecycle));
        }
    }

    @Test
    void pendingWorkerEndKeepsInactivityRecheckInstalledUntilPlacementFinishes() throws Exception {
        var context = RequestProtocolTestSupport.context(config, 895L);
        var future = RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context);
        var route =
                SchedulingTestConfig.createRoute(
                        context,
                        new Response(),
                        null,
                        null,
                        mock(PrefillEndpoint.class),
                        null,
                        null,
                        System.currentTimeMillis());
        try (var admission = SchedulerTestSupport.claim(lifecycle, "895", future);
                var completion = RequestProtocolTestSupport.finishOnExit(admission)) {
            assertNotNull(lifecycle.beginRoute(route));
            assertTrue(AbstractRequestScheduler.recordRoute(admission, route, true, null));
            var original = context.snapshot().deadlines().inactivity();
            lifecycle.onWorkerRequestEvent(
                    context,
                    new ContextResult.WorkerRequestEvent(
                            route,
                            null,
                            org.flexlb.dao.route.RoleType.PDFUSION,
                            ContextResult.WorkerRequestEventType.COMPLETED,
                            0L,
                            "worker complete",
                            original.atMs() - 1));
            assertNull(context.snapshot().outcome());
            lifecycle.onDeadline(original, original.atMs());
            awaitCondition(() -> context.snapshot().deadlines().inactivity() != original);
            awaitCondition(
                    () ->
                            !lifecycle.expirationTimer.needsUpdate(
                                    context, context.snapshot().deadlines()));
            assertNull(context.snapshot().outcome());
            assertFalse(future.isDone());
        }
    }

    @Test
    void deadlineInstallationDoesNotReadUnboundRequestForDiagnostics() {
        RequestContext context = new RequestContext(config);
        context.data().setFuture(new CompletableFuture<>());
        var timer =
                SchedulerTestSupport.timer(
                        org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(lifecycle));
        var before = context.snapshot();
        org.junit.jupiter.api.Assertions.assertDoesNotThrow(
                () -> timer.update(context, before.deadlines()));
        assertSame(before, context.snapshot());
        assertNull(context.snapshot().key());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void pendingDeliveryCountTracksTheLifecycleBeforeSenderHandoff(boolean immediate) {
        var requestConfig = immediate ? SchedulingTestConfig.newConfig() : config;
        if (immediate) {
            requestConfig.setScheduler(ConfigTestSupport.direct());
        }
        var context = RequestProtocolTestSupport.context(requestConfig, 894L);
        var future = RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context);
        var repository = SchedulerTestSupport.repository(lifecycle);
        int expectedPending = immediate ? 0 : 1;
        assertEquals(immediate, context.data().getConfig().isDirect());
        assertEquals(Stage.QUEUED, SchedulerTestSupport.stage(context));
        assertEquals(expectedPending, repository.pendingDeliveryRequestCount());
        var prefill = mock(PrefillEndpoint.class);
        when(prefill.acceptsNewRequests()).thenReturn(true);
        var item =
                SchedulingTestConfig.createRoute(
                        context,
                        new Response(),
                        null,
                        null,
                        prefill,
                        null,
                        null,
                        System.currentTimeMillis());
        try (var admission =
                        SchedulerTestSupport.claim(
                                lifecycle, context.data().getRequestId(), future);
                var completion = RequestProtocolTestSupport.finishOnExit(admission)) {
            assertNotNull(admission);
            assertEquals(Stage.ROUTING, SchedulerTestSupport.stage(context));
            assertEquals(expectedPending, repository.pendingDeliveryRequestCount());
            assertEquals(
                    PlacementResult.Status.SUCCESS,
                    RequestProtocolTestSupport.commitRoute(
                            lifecycle, item, RequestProtocolTestSupport.publication(() -> true)));
            assertEquals(Stage.READY_TO_DELIVER, SchedulerTestSupport.stage(context));
            assertEquals(expectedPending, repository.pendingDeliveryRequestCount());
            AbstractRequestScheduler.finishPlacement(admission);
        }
        assertEquals(expectedPending, repository.pendingDeliveryRequestCount());
        var delivery =
                RequestProtocolTestSupport.claimBatchWithoutPrediction(
                        lifecycle, item, 8L, () -> true);
        assertNotNull(delivery);
        assertEquals(Stage.DELIVERING, SchedulerTestSupport.stage(context));
        try {
            assertEquals(0, repository.pendingDeliveryRequestCount());
            assertEquals(
                    1, repository.liveRequestCount(), "sender handoff is not request completion");
        } finally {
            lifecycle.onDeliveryResult(
                    new ContextResult.TransportResult(
                            delivery,
                            org.flexlb.balance.scheduler.Delivery.Result.notSent(
                                    new IllegalStateException("test sender stopped")),
                            System.currentTimeMillis()));
        }
        assertEquals(0, repository.pendingDeliveryRequestCount());
    }

    @Test
    void registrationCapturesCurrentIdentityAndKeepsItAfterFutureCompletion() {
        RequestContext context = context(890L);

        context.data().setRequest(context.data().getRequest().toBuilder().requestId("891").build());
        var future = RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context);
        assertSame(future, context.data().getFuture());
        assertEquals("891", context.data().getRequestId());
        RequestContext other = context(893L);
        assertThrows(IllegalStateException.class, () -> other.data().setFuture(future));
        assertEquals("893", other.data().getRequestId());
        var otherFuture = RequestProtocolTestSupport.submitForManualPlacement(lifecycle, other);
        assertFalse(otherFuture.isDone());
        assertSame(otherFuture, other.data().getFuture());
        assertEquals("893", other.data().getRequestId());
        assertThrows(
                IllegalStateException.class,
                () ->
                        context.data()
                                .setRequest(
                                        context.data().getRequest().toBuilder()
                                                .requestId("892")
                                                .build()));
        assertEquals("891", context.data().getRequestId());
        assertTrue(future.cancel(false));
        assertEquals("891", context.data().getRequestId());
        assertThrows(
                IllegalStateException.class,
                () -> context.data().setFuture(new CompletableFuture<>()));
        assertTrue(
                RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context).isDone());
        assertSame(future, context.data().getFuture());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void registrationFreezesPriorityAndDeadline(boolean suppliedMetadata) {
        RequestContext context = context(806L);
        context.data().setRequest(context.data().getRequest().toBuilder().rawPriority(73).build());
        context.data()
                .setSchedulingMetadata(
                        suppliedMetadata ? SchedulingMetadata.explicit(64, Long.MAX_VALUE) : null);
        int priority = context.data().getPriority();
        long expiresAt = context.data().getRequestExpiresAtMs();
        RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context);
        assertNotNull(context.data().getRequest());
        var metadata = context.data().getSchedulingMetadata();
        assertNotNull(metadata);
        context.data().setSchedulingMetadata(metadata);
        assertThrows(IllegalStateException.class, () -> context.data().setSchedulingMetadata(null));
        assertThrows(
                IllegalStateException.class,
                () -> context.data().setSchedulingMetadata(SchedulingMetadata.explicit(1, 1L)));
        assertThrows(
                IllegalStateException.class,
                () ->
                        context.data()
                                .setRequest(
                                        context.data().getRequest().toBuilder()
                                                .rawPriority(1)
                                                .build()));
        org.springframework.test.util.ReflectionTestUtils.setField(context.data(), "startTime", 1L);
        assertEquals(priority, context.data().getPriority());
        assertEquals(expiresAt, context.data().getRequestExpiresAtMs());
        assertEquals(priority, metadata.priority());
    }

    @ParameterizedTest
    @ValueSource(ints = {0, -1, 101, 1, 73, 100})
    void queueReadsFrozenPriorityAndPreservesInternalDefaulting(int priority) {
        var context = context(808L);
        context.data().setSchedulingMetadata(null);
        context.data()
                .setRequest(context.data().getRequest().toBuilder().rawPriority(priority).build());
        RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context);
        assertEquals(priority, context.data().getSchedulingMetadata().priority());
        int queuePriority = priority >= 1 && priority <= 100 ? priority : 50;
        var entry = new GlobalQueueEntry(context, null);
        for (boolean priorityOrdering : new boolean[] {false, true}) {
            var queue = new OrderedRequestQueue(priorityOrdering);
            queue.add(entry);
            assertThrows(
                    IllegalStateException.class,
                    () ->
                            context.data()
                                    .setRequest(
                                            context.data().getRequest().toBuilder()
                                                    .rawPriority(99)
                                                    .build()));
            assertEquals(priority, context.data().getPriority());
            assertEquals(queuePriority, entry.priority());
            assertEquals(1, queue.priorityCounts()[queuePriority]);
            queue.remove(entry);
            queue.restore(entry);
            assertEquals(queuePriority, entry.priority());
            queue.remove(entry);
        }
        assertEquals(queuePriority, new GlobalQueueEntry(context, null).priority());
    }

    @Test
    void routeAttemptsShareRegisteredInputsAndRetainQueueIdentity() {
        RequestContext context = context(807L);
        var cacheKeys = new ArrayList<>(List.of(11L, 22L));
        var inputs =
                context.data().getRequest().toBuilder()
                        .seqLen(64L)
                        .maxNewTokens(32)
                        .apiKey("original")
                        .blockCacheKeys(cacheKeys)
                        .cacheKeyBlockSize(16L)
                        .build();
        context.data().setRequest(inputs);
        RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context);
        context.data().setRequest(inputs);
        assertThrows(
                IllegalStateException.class,
                () ->
                        context.data()
                                .setRequest(
                                        inputs.toBuilder()
                                                .requestId("808")
                                                .seqLen(1L)
                                                .maxNewTokens(2)
                                                .apiKey("changed")
                                                .build()));
        cacheKeys.clear();
        var first =
                SchedulingTestConfig.createRoute(context, null, null, null, null, null, null, 123L);
        var retry =
                SchedulingTestConfig.createRoute(context, null, null, null, null, null, null, 456L);
        assertSame(inputs, first.ctx().data().getRequest());
        assertSame(inputs, retry.ctx().data().getRequest());
        assertEquals("807", retry.requestId());
        assertEquals(64L, retry.seqLen());
        assertEquals(96L, inputs.getExpectedKvTokens());
        assertEquals("original", inputs.getApiKey());
        assertEquals(List.of(11L, 22L), inputs.getBlockCacheKeys());
        assertEquals(16L, inputs.getCacheKeyBlockSize());
        assertThrows(UnsupportedOperationException.class, () -> inputs.getBlockCacheKeys().clear());
        assertEquals(first.enqueueSeq(), retry.enqueueSeq());
        assertEquals(123L, retry.enqueuedAtMs());
    }

    @Test
    void foreignDecodeReservationDoesNotInitializeWorkerFifoIdentity() {
        RequestContext context = context(804L);
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        SchedulingTestConfig.createRoute(
                                context,
                                null,
                                null,
                                null,
                                null,
                                null,
                                new DecodeResources.RequestReservation(
                                        1L, String.valueOf(805L), 1L),
                                123L));
        assertEquals(0L, context.data().getWorkerEnqueueSequence());
        assertEquals(0L, context.data().getFirstWorkerEnqueueTime());
    }

    @Test
    void registrationAndFutureInitializationShareTheContextMonitor() throws Exception {
        RequestContext context = context(803L);
        CompletableFuture<Response> placeholder = new CompletableFuture<>();
        AtomicReference<CompletableFuture<Response>> registered = new AtomicReference<>();
        Thread registrar =
                new Thread(
                        () ->
                                registered.set(
                                        RequestProtocolTestSupport.submitForManualPlacement(
                                                lifecycle, context)),
                        "registration-test");
        synchronized (context) {
            registrar.start();
            awaitCondition(() -> registrar.getState() == Thread.State.BLOCKED);
            assertFalse(
                    context.data().getFuture() instanceof RequestFuture,
                    "registration must wait for the Context monitor");
            context.data().setFuture(placeholder);
        }
        registrar.join(5_000L);
        assertFalse(registrar.isAlive());
        assertNotNull(registered.get());
        assertSame(registered.get(), context.data().getFuture());
        assertTrue(context.data().getFuture() instanceof RequestFuture);
        assertFalse(context.data().getFuture().isDone());
    }

    @Test
    void duplicateSubmitPreservesCanonicalFutureAndRequestId() {
        RequestContext context = context(800L);
        CompletableFuture<Response> canonical =
                RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context);
        assertThrows(
                IllegalStateException.class,
                () ->
                        context.data()
                                .setRequest(
                                        context.data().getRequest().toBuilder()
                                                .requestId("801")
                                                .build()));
        assertEquals(
                StrategyErrorType.INVALID_REQUEST.getErrorCode(),
                RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context)
                        .join()
                        .getCode());
        assertSame(canonical, context.data().getFuture());
        assertEquals("800", context.data().getRequestId());
        assertEquals("800", context.data().getRequest().getRequestId());
        assertThrows(
                IllegalStateException.class,
                () -> context.data().setFuture(new CompletableFuture<>()));
        lifecycle.cancel("800", 0L, CancelReason.CLIENT_CANCELLED);
        assertEquals(
                StrategyErrorType.REQUEST_CANCELLED.getErrorCode(), canonical.join().getCode());
        assertNull(lifecycle.requests.findActive("800"));
        assertEquals(
                RequestState.Phase.CANCELLED,
                org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle)
                        .getRequestState("800", 0L)
                        .state());
        assertNull(
                org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle)
                        .getRequestState("801", 0L));
    }

    @Test
    void rejectedDuplicateSubmitCannotReplaceTheCanonicalCancellation() {
        var queue =
                (QueuedRequestScheduler)
                        org.flexlb.balance.scheduler.SchedulerTestServices.scheduler(lifecycle);
        org.mockito.Mockito.doNothing()
                .when(queue)
                .removeRequest(org.mockito.ArgumentMatchers.any());
        RequestContext context = context(804L);
        CompletableFuture<Response> canonical =
                RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context);
        lifecycle.cancel("804", 0L, CancelReason.CLIENT_CANCELLED);

        assertEquals(
                StrategyErrorType.INVALID_REQUEST.getErrorCode(),
                SchedulerTestServices.entry(queue).submit(context).join().getCode());
        assertEquals(ContextResult.Phase.CLOSED, context.snapshot().phase());
        assertNull(lifecycle.requests.findActive("804"));

        assertEquals(
                StrategyErrorType.REQUEST_CANCELLED.getErrorCode(), canonical.join().getCode());
    }

    @Test
    void expiredTerminalAllowsNewContextButCannotReuseOldContext() {
        RequestContext old = context(802L);
        CompletableFuture<Response> original =
                RequestProtocolTestSupport.submitForManualPlacement(lifecycle, old);
        lifecycle.cancel("802", 0L, CancelReason.CLIENT_CANCELLED);
        original.join();
        assertTrue(
                org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle)
                        .removeExactTerminal(
                                org.flexlb.balance.scheduler.SchedulerTestSupport.terminalRecord(
                                        lifecycle,
                                        org.flexlb.balance.scheduler.SchedulerTestSupport
                                                .repository(lifecycle)
                                                .getRequestState("802", 0L)),
                                Long.MAX_VALUE));
        assertEquals(
                StrategyErrorType.INVALID_REQUEST.getErrorCode(),
                RequestProtocolTestSupport.submitForManualPlacement(lifecycle, old)
                        .join()
                        .getCode());
        assertSame(original, old.data().getFuture());
        CompletableFuture<Response> replacement =
                RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context(802L));
        assertFalse(replacement.isDone());
        assertSame(replacement, lifecycle.requests.findActive("802").data().getFuture());
        var queue =
                (QueuedRequestScheduler)
                        org.flexlb.balance.scheduler.SchedulerTestServices.scheduler(lifecycle);
        queue.removeRequest(old);
        assertNull(SchedulerTestServices.lifecycle(queue).cancel(old, 0L, CancelReason.SHUTDOWN));
        assertSame(replacement, lifecycle.requests.findActive("802").data().getFuture());
        assertFalse(
                replacement.isDone(),
                "stale control and close tickets cannot settle a reused request id");
    }

    @Test
    void duplicateRegistrationCannotReplaceTheCanonicalExactGeneration() {
        RequestContext context = context(101L);
        CompletableFuture<Response> canonical =
                RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context);
        CompletableFuture<Response> duplicate =
                RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context(101L));
        assertFalse(canonical.isDone());
        assertTrue(duplicate.isDone());
        assertEquals(StrategyErrorType.INVALID_REQUEST.getErrorCode(), duplicate.join().getCode());
        assertSame(canonical, lifecycle.requests.findActive("101").data().getFuture());
        assertEquals(
                1,
                org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle)
                        .liveRequestCount());
    }

    @Test
    void successfulExternalCompletionBeforeDeliveryCannotStrandTheRequest() {
        CompletableFuture<Response> future =
                RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context(10001L));
        Response success = new Response();
        success.setSuccess(true);
        assertFalse(future.complete(success));
        assertFalse(future.isDone());
        assertEquals(
                RequestState.Phase.QUEUED,
                org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle)
                        .getRequestState("10001", 0L)
                        .state());
        assertEquals(
                1,
                org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle)
                        .liveRequestCount());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void futureCancelDuringAdmissionIsSynchronousAndTheAdmissionOwnerSettlesIt(
            boolean mayInterruptIfRunning) {
        QueuedRequestScheduler queue =
                (QueuedRequestScheduler)
                        org.flexlb.balance.scheduler.SchedulerTestServices.scheduler(lifecycle);
        CompletableFuture<Response> future =
                RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context(10003L));
        RequestContext requestContext = lifecycle.requests.findActive("10003");
        queue.onRequest(requestContext);
        PlacementOperation admission = SchedulerTestSupport.claim(lifecycle, "10003", future);
        assertNotNull(admission);
        assertTrue(future.cancel(mayInterruptIfRunning));
        assertTrue(future.isCancelled());
        assertEquals(0, RequestProtocolTestSupport.queuedCount(queue));
        assertEquals(
                RequestState.Phase.CANCEL_REQUESTED,
                RequestState.from(requestContext.snapshot()).state());
        AbstractRequestScheduler.finishPlacement(admission);
        assertEquals(
                RequestState.Phase.CANCELLED, RequestState.from(requestContext.snapshot()).state());
        assertEquals(
                0,
                org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle)
                        .liveRequestCount());
        assertEquals(0, RequestProtocolTestSupport.queuedCount(queue));
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void schedulingDeadlineDuringAdmissionRetainsItsOwnEntryRuleAfterDeliveryClaim(boolean timer) {
        var context = context(10007L);
        var future = RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context);
        ExpirationTimer.Token deadline =
                RequestProtocolTestSupport.field(context, "requestDeadline");
        assertNotNull(deadline);
        var item =
                SchedulingTestConfig.createRoute(
                        context,
                        new Response(),
                        null,
                        null,
                        mock(PrefillEndpoint.class),
                        null,
                        null,
                        System.currentTimeMillis());
        try (var admission =
                        SchedulerTestSupport.claim(
                                lifecycle, context.data().getRequestId(), future);
                var admissionCompletion1 = RequestProtocolTestSupport.finishOnExit(admission)) {
            assertNotNull(admission);
            assertEquals(
                    PlacementResult.Status.SUCCESS,
                    RequestProtocolTestSupport.commitRoute(
                            lifecycle, item, RequestProtocolTestSupport.publication(() -> true)));
            assertNotNull(
                    RequestProtocolTestSupport.claimRouteWithoutPrediction(
                            lifecycle, item, () -> true));
            if (timer) {
                lifecycle.onDeadline(deadline, deadline.atMs());
            } else {
                lifecycle.cancel(context, 0L, CancelReason.DEADLINE_EXCEEDED);
            }
            assertEquals(
                    timer ? CancelReason.DEADLINE_EXCEEDED : null,
                    SchedulerTestSupport.cancellationReason(context));
            assertSame(
                    timer ? null : deadline,
                    RequestProtocolTestSupport.field(context, "requestDeadline"));
            assertEquals(Stage.DELIVERING, SchedulerTestSupport.stage(context));
            assertSame(item, context.snapshot().route());
            assertFalse(future.isDone(), "admission owner must settle the timer fact");
        }
    }

    @Test
    void notificationFailureAfterPublicationKeepsTheExactRouteAndDoesNotRollback() {
        var context = context(10005L);
        var future = RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context);
        var prefill = mock(PrefillEndpoint.class);
        var item =
                SchedulingTestConfig.createRoute(
                        context,
                        new Response(),
                        null,
                        null,
                        prefill,
                        null,
                        null,
                        System.currentTimeMillis());
        var failure = new IllegalStateException("route-ready notification failed");
        org.mockito.Mockito.doThrow(failure)
                .when(prefill)
                .signal(org.flexlb.balance.endpoint.PrefillEndpoint.Signal.INPUT);
        try (var admission =
                        SchedulerTestSupport.claim(
                                lifecycle, context.data().getRequestId(), future);
                var admissionCompletion2 = RequestProtocolTestSupport.finishOnExit(admission)) {
            assertNotNull(admission);
            assertSame(
                    failure,
                    assertThrows(
                            IllegalStateException.class,
                            () ->
                                    RequestProtocolTestSupport.commitRoute(
                                            lifecycle,
                                            item,
                                            RequestProtocolTestSupport.publication(() -> true))));
            assertSame(item, context.snapshot().route());
            assertEquals(Stage.READY_TO_DELIVER, SchedulerTestSupport.stage(context));
            assertEquals(
                    0, failure.getSuppressed().length, "committed binding must not be rolled back");
        }
    }

    @Test
    void publicationFailureAfterHandoffPreservesBindingAndWakesDelivery() {
        var context = context(10008L);
        var future = RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context);
        var prefill = mock(PrefillEndpoint.class);
        var item =
                SchedulingTestConfig.createRoute(
                        context,
                        new Response(),
                        null,
                        null,
                        prefill,
                        null,
                        null,
                        System.currentTimeMillis());
        var failure = new IllegalStateException("failure after handoff");
        var publication =
                new RequestProtocolTestSupport.Publication() {
                    private boolean published;

                    @Override
                    public void publish() {
                        published = true;
                        throw failure;
                    }

                    @Override
                    public boolean published() {
                        return published;
                    }
                };
        try (var admission =
                        SchedulerTestSupport.claim(
                                lifecycle, context.data().getRequestId(), future);
                var completion = RequestProtocolTestSupport.finishOnExit(admission)) {
            assertSame(
                    failure,
                    assertThrows(
                            IllegalStateException.class,
                            () ->
                                    RequestProtocolTestSupport.commitRoute(
                                            lifecycle, item, publication)));
            assertSame(item, context.snapshot().route());
            assertEquals(Stage.READY_TO_DELIVER, SchedulerTestSupport.stage(context));
            verify(prefill).signal(org.flexlb.balance.endpoint.PrefillEndpoint.Signal.INPUT);
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void failedPublicationUnbindsTheRouteBeforeAdmissionCloses(boolean throwsFailure) {
        var context = context(10006L);
        var future = RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context);
        var item =
                SchedulingTestConfig.createRoute(
                        context,
                        new Response(),
                        null,
                        null,
                        mock(PrefillEndpoint.class),
                        null,
                        null,
                        System.currentTimeMillis());
        var failure = new IllegalStateException("publication failed");
        try (var admission =
                        SchedulerTestSupport.claim(
                                lifecycle, context.data().getRequestId(), future);
                var admissionCompletion3 = RequestProtocolTestSupport.finishOnExit(admission)) {
            assertNotNull(admission);
            if (throwsFailure) {
                assertSame(
                        failure,
                        assertThrows(
                                IllegalStateException.class,
                                () ->
                                        RequestProtocolTestSupport.commitRoute(
                                                lifecycle,
                                                item,
                                                RequestProtocolTestSupport.publication(
                                                        () -> {
                                                            throw failure;
                                                        }))));
            } else {
                assertEquals(
                        PlacementResult.Status.BLOCKED,
                        RequestProtocolTestSupport.commitRoute(
                                lifecycle,
                                item,
                                RequestProtocolTestSupport.publication(() -> false)));
            }
            context.bindPlacement(
                    admission, item, ContextResult.PublicationResult.ROLLBACK_COMPLETED);
            assertNull(context.snapshot().route());
            assertEquals(Stage.ROUTING, SchedulerTestSupport.stage(context));
        }
        assertEquals(Stage.QUEUED, SchedulerTestSupport.stage(context));
    }

    @Test
    void futureCancelAfterQueuePublicationStillWinsBeforePlacementOperationCloses() {
        RequestContext context = context(10004L);
        CompletableFuture<Response> future =
                RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context);
        PrefillEndpoint prefill = mock(PrefillEndpoint.class);
        context.data().setFuture(future);
        RouteExecution item =
                SchedulingTestConfig.createRoute(
                        context,
                        new Response(),
                        null,
                        null,
                        prefill,
                        null,
                        null,
                        System.currentTimeMillis());
        try (PlacementOperation admission = SchedulerTestSupport.claim(lifecycle, "10004", future);
                var admissionCompletion4 = RequestProtocolTestSupport.finishOnExit(admission)) {
            assertNotNull(admission);
            assertEquals(
                    PlacementResult.Status.SUCCESS,
                    RequestProtocolTestSupport.commitRoute(
                            lifecycle, item, RequestProtocolTestSupport.publication(() -> true)));
            assertTrue(future.cancel(false));
            assertTrue(future.isCancelled());
        }
        verify(prefill).releaseRequest(item);
        assertEquals(
                RequestState.Phase.CANCELLED,
                org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle)
                        .getRequestState("10004", 0L)
                        .state());
        assertEquals(
                0,
                org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle)
                        .liveRequestCount());
    }

    @Test
    void globalCloseConsumesAnAcceptedInactivityFactBeforeShutdownFailure() throws Exception {
        CompletableFuture<Response> future =
                RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context(10002L));
        RequestContext context = lifecycle.requests.findActive("10002");
        lifecycle.onDeadline(
                RequestProtocolTestSupport.<ExpirationTimer.Token>field(
                        context, "inactivityDeadline"),
                Long.MAX_VALUE);
        lifecycle.closeRequest(
                context,
                new ContextResult.EndEvidence(
                        context.snapshot().route(),
                        ContextResult.EndKind.SHUTDOWN,
                        null,
                        false,
                        0L,
                        StrategyErrorType.DISPATCH_FAILED,
                        "shutdown",
                        false,
                        false));
        assertEquals(RequestState.Phase.TIMED_OUT, RequestState.from(context.snapshot()).state());
        assertFalse(future.get(5, TimeUnit.SECONDS).isSuccess());
    }

    @Test
    void terminalRecordPreservesIdentityWithoutRetainingRequestContext() throws Exception {
        RequestContext context = context(103L);
        CompletableFuture<Response> future =
                RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context);
        lifecycle.cancel("103", 0L, CancelReason.CLIENT_CANCELLED);
        assertEquals(
                StrategyErrorType.REQUEST_CANCELLED.getErrorCode(),
                future.get(5, TimeUnit.SECONDS).getCode());
        awaitCondition(
                () ->
                        org.springframework.test.util.ReflectionTestUtils.getField(future, "target")
                                == null);
        RequestState terminal =
                org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle)
                        .getRequestState("103", 0L);
        assertEquals(RequestState.Phase.CANCELLED, terminal.state());
        assertNull(lifecycle.requests.findActive("103"));
        assertSame(
                terminal,
                org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle)
                        .getRequestState("103", 0L));
        assertEquals(
                StrategyErrorType.INVALID_REQUEST.getErrorCode(),
                RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context(103L))
                        .join()
                        .getCode());
    }

    @Test
    void globalWaitingRequestsHaveNoQuantityAdmissionLimit() {
        var low = context(1L);
        low.data().setSchedulingMetadata(SchedulingMetadata.explicit(10, Long.MAX_VALUE));
        var waiting = RequestProtocolTestSupport.submitForManualPlacement(lifecycle, low);
        for (long id = 2; id <= 1001; id++) {
            var high = context(id);
            high.data().setSchedulingMetadata(SchedulingMetadata.explicit(90, Long.MAX_VALUE));
            assertFalse(
                    RequestProtocolTestSupport.submitForManualPlacement(lifecycle, high).isDone());
        }
        assertFalse(waiting.isDone(), "higher priority arrivals must not evict waiting requests");
        assertEquals(
                1001,
                org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle)
                        .liveRequestCount());
        assertEquals(
                StrategyErrorType.INVALID_REQUEST.getErrorCode(),
                RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context(1L))
                        .join()
                        .getCode());
        for (long id = 1; id <= 1001; id++) {
            lifecycle.cancel(String.valueOf(id), 0L, CancelReason.CLIENT_CANCELLED);
        }
        assertEquals(
                0,
                org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle)
                        .liveRequestCount());
    }

    @Test
    void globalWaitingCancellationDoesNotWaitForDecisionOwnerTicket() throws Exception {
        QueuedRequestScheduler queue =
                (QueuedRequestScheduler)
                        org.flexlb.balance.scheduler.SchedulerTestServices.scheduler(lifecycle);
        org.mockito.Mockito.doNothing()
                .when(queue)
                .removeRequest(org.mockito.ArgumentMatchers.any());
        RequestContext context = context(1002L);
        CompletableFuture<Response> future =
                RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context);

        assertEquals(
                RequestState.Phase.CANCELLED,
                lifecycle.cancel("1002", 0L, CancelReason.CLIENT_CANCELLED).state());
        assertNull(lifecycle.requests.findActive("1002"));
        verify(queue).removeRequest(context);

        assertEquals(
                StrategyErrorType.REQUEST_CANCELLED.getErrorCode(),
                future.get(5, TimeUnit.SECONDS).getCode());
        assertEquals(
                RequestState.Phase.CANCELLED,
                org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle)
                        .getRequestState("1002", 0L)
                        .state());
    }

    @Test
    void publicQueriesOwnTheirLockAndPrivateDecisionsStillRequireIt() {
        RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context(102L));
        RequestContext requestContext = lifecycle.requests.findActive("102");
        assertNull(requestContext.snapshot().route());
        assertTrue(SchedulerTestSupport.isOpen(requestContext));
        assertTrue(requestContext.snapshot().phase() != ContextResult.Phase.CLOSED);
        var before = requestContext.snapshot();
        var stale =
                org.flexlb.balance.scheduler.RequestChangeAssertions.record(
                        requestContext,
                        observedRequest ->
                                observedRequest.recordWorkerRequestEvent(
                                        new ContextResult.WorkerRequestEvent(
                                                null,
                                                null,
                                                RoleType.PREFILL,
                                                ContextResult.WorkerRequestEventType.ACTIVE,
                                                0L,
                                                null,
                                                System.currentTimeMillis())));
        assertEquals(RequestChangeAssertions.UpdateKind.STALE, stale.kind());
        assertSame(before, stale.after());
        assertEquals(
                RequestState.Phase.QUEUED, RequestState.from(requestContext.snapshot()).state());
    }

    @Test
    void deliveredRequestsHaveNoExtraGlobalQuantityGate() {
        for (long id = 201; id <= 401; id++) {
            Registered registered = registerItem(id);
            assertEquals(PlacementResult.Status.SUCCESS, commitRoute(lifecycle, registered));
            assertNotNull(
                    RequestProtocolTestSupport.claimRoute(
                            lifecycle, registered.item(), () -> true));
        }
        assertEquals(
                201,
                org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle)
                        .liveRequestCount());
    }

    @Test
    void itemEntryPointsRejectAnotherSchedulersLiveContext() {
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        AbstractRequestScheduler other =
                org.flexlb.balance.scheduler.SchedulerTestSupport.create(
                        configService,
                        mock(DeliveryMetricsReporter.class),
                        mock(RequestSchedulerReporter.class),
                        mock(TheoryCacheHitReporter.class));
        Registered registered = registerItem(806L);
        CompletableFuture<Response> otherFuture = other.submit(context(806L), () -> {});
        RouteExecution item = registered.item();
        try {
            try (PlacementOperation admission =
                            SchedulerTestSupport.claim(lifecycle, "806", registered.future());
                    var admissionCompletion5 = RequestProtocolTestSupport.finishOnExit(admission)) {
                assertNotNull(admission);
                assertEquals(
                        PlacementResult.Status.CLOSED,
                        RequestProtocolTestSupport.commitRoute(
                                other,
                                item,
                                RequestProtocolTestSupport.publication(
                                        () -> {
                                            throw new AssertionError(
                                                    "foreign scheduler published the route");
                                        })));
                assertEquals(Stage.ROUTING, SchedulerTestSupport.stage(item.ctx()));
                assertNull(item.ctx().snapshot().route());
                assertEquals(
                        PlacementResult.Status.SUCCESS,
                        RequestProtocolTestSupport.commitRoute(
                                lifecycle,
                                item,
                                RequestProtocolTestSupport.publication(() -> true)));
            }
            synchronized (item.ctx()) {
                assertFalse(
                        other.requests.isCurrent(item.ctx())
                                && item.ctx().data().getRequestScheduler() == other
                                && item.ctx().snapshot().preparedDelivery());
            }
            assertNull(
                    DeliveryDispatcher.commitDelivery(
                            other,
                            new org.flexlb.balance.scheduler.Delivery.Member(
                                    item,
                                    RequestProtocolTestSupport.handoff(
                                            () -> {
                                                throw new AssertionError(
                                                        "foreign scheduler transferred endpoint ownership");
                                            })),
                            Delivery.SendMode.BATCH_ENQUEUE,
                            41L));
            other.onDeliveryResult(
                    new ContextResult.PreparationFailure(
                            null, item, "foreign failure", System.currentTimeMillis()));
            other.cancel(
                    item.ctx(), 0L, org.flexlb.balance.scheduler.CancelReason.DEADLINE_EXCEEDED);
            other.rejectRoute(item, new IllegalStateException("foreign queue failure"));
            other.rejectRoute(item, null);
            assertEquals(Stage.READY_TO_DELIVER, SchedulerTestSupport.stage(item.ctx()));
            assertEquals(
                    Stage.QUEUED, SchedulerTestSupport.stage(other.requests.findActive("806")));
            assertFalse(registered.future().isDone());
            assertFalse(otherFuture.isDone());
            assertNotNull(
                    DeliveryDispatcher.commitDelivery(
                            lifecycle,
                            new org.flexlb.balance.scheduler.Delivery.Member(
                                    item, RequestProtocolTestSupport.handoff(() -> true)),
                            Delivery.SendMode.BATCH_ENQUEUE,
                            41L));
        } finally {
            RequestProtocolTestSupport.closeAdmissionAndAwaitMutations(other);
            other.closeOutstandingAndTerminalize();
            SchedulerTestSupport.timer(
                            org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(other))
                    .close();
            SchedulerTestSupport.closeRequestExecutors(
                    org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(other));
        }
    }

    @Test
    void failedPublicationKeepsSchedulingStageAndAllowsExactRetry() {
        Registered registered = registerItem(706L);
        RequestContext requestContext = lifecycle.requests.findActive("706");
        try (PlacementOperation admission =
                        SchedulerTestSupport.claim(lifecycle, "706", registered.future());
                var admissionCompletion6 = RequestProtocolTestSupport.finishOnExit(admission)) {
            assertNotNull(admission);
            assertEquals(
                    PlacementResult.Status.BLOCKED,
                    RequestProtocolTestSupport.commitRoute(
                            lifecycle,
                            registered.item(),
                            RequestProtocolTestSupport.publication(() -> false)));
            requestContext.bindPlacement(
                    admission,
                    registered.item(),
                    ContextResult.PublicationResult.ROLLBACK_COMPLETED);
            assertEquals(Stage.ROUTING, SchedulerTestSupport.stage(requestContext));
            assertNull(requestContext.snapshot().route());
            assertEquals(
                    RequestState.Phase.QUEUED,
                    RequestState.from(requestContext.snapshot()).state());
            assertEquals(
                    PlacementResult.Status.SUCCESS,
                    RequestProtocolTestSupport.commitRoute(
                            lifecycle,
                            registered.item(),
                            RequestProtocolTestSupport.publication(() -> true)));
            assertEquals(Stage.READY_TO_DELIVER, SchedulerTestSupport.stage(requestContext));
        }
    }

    @Test
    void publicationResultCannotBindAReplacementPlacementScope() {
        Registered registered = registerItem(712L);
        var context = registered.item().ctx();
        var replacement = new java.util.concurrent.atomic.AtomicReference<PlacementOperation>();
        var expected = new java.util.concurrent.atomic.AtomicReference<ContextResult.Snapshot>();
        try (var initial = SchedulerTestSupport.claim(lifecycle, "712", registered.future())) {
            assertNotNull(initial);
            try {
                var result =
                        RequestProtocolTestSupport.commitRoute(
                                lifecycle,
                                registered.item(),
                                RequestProtocolTestSupport.publication(
                                        () -> {
                                            registered
                                                    .item()
                                                    .prefillEndpoint()
                                                    .releaseRequest(registered.item());
                                            assertEquals(
                                                    DecodeResources.ReservationReleaseResult
                                                            .RELEASED,
                                                    registered
                                                            .item()
                                                            .decodeEndpoint()
                                                            .release(
                                                                    registered
                                                                            .item()
                                                                            .decodeReservation(),
                                                                    DecodeResources.ReleaseReason
                                                                            .LOCAL_ROLLBACK)
                                                            .status());
                                            context.bindPlacement(
                                                    initial,
                                                    registered.item(),
                                                    ContextResult.PublicationResult
                                                            .ROLLBACK_COMPLETED);
                                            AbstractRequestScheduler.finishPlacement(initial);
                                            replacement.set(
                                                    SchedulerTestSupport.claim(
                                                            lifecycle, "712", registered.future()));
                                            assertNotNull(replacement.get());
                                            expected.set(context.snapshot());
                                            return false;
                                        }));
                assertEquals(PlacementResult.Status.CLOSED, result);
                assertSame(
                        expected.get(),
                        context.snapshot(),
                        "the old publication must leave the new scope untouched");
                assertSame(replacement.get(), context.snapshot().placement().operation());
                assertNull(context.snapshot().route());
                assertNull(context.snapshot().route());
                assertFalse(registered.future().isDone());
                verify(registered.item().prefillEndpoint(), org.mockito.Mockito.never())
                        .signal(org.flexlb.balance.endpoint.PrefillEndpoint.Signal.INPUT);
            } finally {
                if (replacement.get() != null) {
                    AbstractRequestScheduler.finishPlacement(replacement.get());
                }
                AbstractRequestScheduler.finishPlacement(initial);
            }
        }
    }

    @Test
    void acceptedQueuedCancellationSurvivesEndpointStopFailure() throws Exception {
        RequestContext context = context(707L);
        CompletableFuture<Response> future =
                RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context);
        PrefillEndpoint prefill = mock(PrefillEndpoint.class);
        context.data().setFuture(future);
        RouteExecution item =
                SchedulingTestConfig.createRoute(
                        context,
                        new Response(),
                        null,
                        null,
                        prefill,
                        null,
                        null,
                        System.currentTimeMillis());
        try (PlacementOperation admission = SchedulerTestSupport.claim(lifecycle, "707", future);
                var admissionCompletion7 = RequestProtocolTestSupport.finishOnExit(admission)) {
            assertNotNull(admission);
            assertEquals(
                    PlacementResult.Status.SUCCESS,
                    RequestProtocolTestSupport.commitRoute(
                            lifecycle, item, RequestProtocolTestSupport.publication(() -> true)));
        }

        assertEquals(
                RequestState.Phase.CANCELLED,
                lifecycle.cancel("707", 0L, CancelReason.CLIENT_CANCELLED).state());
        assertEquals(ContextResult.Phase.CLOSED, context.snapshot().phase());
        lifecycle.rejectRoute(item, new IllegalStateException("endpoint stopped"));

        assertEquals(
                RequestState.Phase.CANCELLED,
                org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle)
                        .getRequestState("707", 0L)
                        .state());
        assertEquals(
                StrategyErrorType.REQUEST_CANCELLED.getErrorCode(),
                future.get(5, TimeUnit.SECONDS).getCode());
    }

    @Test
    void stoppedLocalOwnerFallsBackToSharedContinuationInsteadOfTimerThread() throws Exception {
        RequestContext context = context(708L);
        CompletableFuture<Response> future =
                RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context);
        PrefillEndpoint prefill = mock(PrefillEndpoint.class);
        context.data().setFuture(future);
        RouteExecution item =
                SchedulingTestConfig.createRoute(
                        context,
                        new Response(),
                        null,
                        null,
                        prefill,
                        null,
                        null,
                        System.currentTimeMillis());
        try (PlacementOperation admission = SchedulerTestSupport.claim(lifecycle, "708", future);
                var admissionCompletion8 = RequestProtocolTestSupport.finishOnExit(admission)) {
            assertNotNull(admission);
            assertEquals(
                    PlacementResult.Status.SUCCESS,
                    RequestProtocolTestSupport.commitRoute(
                            lifecycle, item, RequestProtocolTestSupport.publication(() -> true)));
        }
        AtomicReference<Thread> cleanupThread = new AtomicReference<>();
        when(prefill.releaseRequest(item))
                .thenAnswer(
                        invocation -> {
                            assertFalse(Thread.holdsLock(context));
                            cleanupThread.set(Thread.currentThread());
                            return true;
                        });
        RequestContext requestContext = lifecycle.requests.findActive("708");
        lifecycle.onDeadline(
                RequestProtocolTestSupport.<ExpirationTimer.Token>field(
                        requestContext, "inactivityDeadline"),
                Long.MAX_VALUE);
        assertFalse(future.get(5, TimeUnit.SECONDS).isSuccess());
        SchedulerTestSupport.awaitRequestWork(
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(lifecycle));
        assertNotNull(cleanupThread.get());
        assertNotEquals(Thread.currentThread(), cleanupThread.get());
        verify(prefill).releaseRequest(item);
        assertEquals(
                RequestState.Phase.TIMED_OUT, RequestState.from(requestContext.snapshot()).state());
    }

    @Test
    void terminalRecordDropsUnconsumedExactOwnerFactReferences() {
        RequestContext context = context(709L);
        CompletableFuture<Response> future =
                RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context);
        PrefillEndpoint prefill = mock(PrefillEndpoint.class);
        context.data().setFuture(future);
        RouteExecution item =
                SchedulingTestConfig.createRoute(
                        context,
                        new Response(),
                        null,
                        null,
                        prefill,
                        null,
                        null,
                        System.currentTimeMillis());
        try (PlacementOperation admission = SchedulerTestSupport.claim(lifecycle, "709", future);
                var admissionCompletion9 = RequestProtocolTestSupport.finishOnExit(admission)) {
            assertNotNull(admission);
            assertEquals(
                    PlacementResult.Status.SUCCESS,
                    RequestProtocolTestSupport.commitRoute(
                            lifecycle, item, RequestProtocolTestSupport.publication(() -> true)));
        }
        RequestContext requestContext = lifecycle.requests.findActive("709");
        lifecycle.onDeadline(
                RequestProtocolTestSupport.<ExpirationTimer.Token>field(
                        requestContext, "inactivityDeadline"),
                Long.MAX_VALUE);
        try {
            assertTrue(RequestProtocolTestSupport.closeAdmissionAndAwaitMutations(lifecycle));
            lifecycle.closeOutstandingAndTerminalize();
            SchedulerTestSupport.awaitRequestWork(
                    org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(lifecycle));
            assertEquals(Stage.FINISHED, SchedulerTestSupport.stage(requestContext));
            assertNull(requestContext.snapshot().route());
            assertNull(requestContext.snapshot().route());
            assertNull(RequestProtocolTestSupport.field(requestContext, "requestDeadline"));
            assertNull(RequestProtocolTestSupport.field(requestContext, "decisionDeadline"));
            assertNull(
                    RequestProtocolTestSupport.<ExpirationTimer.Token>field(
                            requestContext, "inactivityDeadline"));
            assertNull(lifecycle.requests.findActive("709"));
            RequestState terminal =
                    org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle)
                            .getRequestState("709", 0L);
            assertTrue(terminal.state().isTerminal());
            assertSame(
                    terminal,
                    org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle)
                            .getRequestState("709", 0L));
        } finally {
            SchedulerTestSupport.timer(
                            org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(lifecycle))
                    .close();
            SchedulerTestSupport.closeRequestExecutors(
                    org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(lifecycle));
        }
    }

    @Test
    void endpointStopCannotReplaceAnAcceptedTimeoutDuringCleanup() throws Exception {
        RequestContext context = context(710L);
        CompletableFuture<Response> future =
                RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context);
        PrefillEndpoint prefill = mock(PrefillEndpoint.class);
        context.data().setFuture(future);
        RouteExecution item =
                SchedulingTestConfig.createRoute(
                        context,
                        new Response(),
                        null,
                        null,
                        prefill,
                        null,
                        null,
                        System.currentTimeMillis());
        try (PlacementOperation admission = SchedulerTestSupport.claim(lifecycle, "710", future);
                var admissionCompletion10 = RequestProtocolTestSupport.finishOnExit(admission)) {
            assertNotNull(admission);
            assertEquals(
                    PlacementResult.Status.SUCCESS,
                    RequestProtocolTestSupport.commitRoute(
                            lifecycle, item, RequestProtocolTestSupport.publication(() -> true)));
        }
        CountDownLatch entered = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        when(prefill.releaseRequest(item))
                .thenAnswer(
                        call -> {
                            assertFalse(Thread.holdsLock(context));
                            entered.countDown();
                            assertTrue(release.await(5, TimeUnit.SECONDS));
                            return true;
                        });
        try {
            lifecycle.onDeadline(
                    RequestProtocolTestSupport.<ExpirationTimer.Token>field(
                            context, "inactivityDeadline"),
                    Long.MAX_VALUE);
            assertTrue(entered.await(2, TimeUnit.SECONDS));
            lifecycle.rejectRoute(item, new IllegalStateException("endpoint stopped"));
            assertEquals(
                    RequestState.Phase.TIMED_OUT, RequestState.from(context.snapshot()).state());
        } finally {
            release.countDown();
        }
        assertFalse(future.get(5, TimeUnit.SECONDS).isSuccess());
        SchedulerTestSupport.awaitRequestWork(
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(lifecycle));
        verify(prefill).releaseRequest(item);
        assertEquals(RequestState.Phase.TIMED_OUT, RequestState.from(context.snapshot()).state());
    }

    @Test
    void admissionHandleDefersCancellationUntilItsExactCapabilityCloses() {
        CompletableFuture<Response> future =
                RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context(301L));
        PlacementOperation scope = SchedulerTestSupport.claim(lifecycle, "301", future);
        assertNotNull(scope);

        RequestState requested = lifecycle.cancel("301", 0L, CancelReason.CLIENT_CANCELLED);

        assertEquals(RequestState.Phase.CANCEL_REQUESTED, requested.state());
        assertFalse(
                future.isDone(), "the admission mutation still owns rollback and terminal cleanup");

        AbstractRequestScheduler.finishPlacement(scope);

        assertEquals(StrategyErrorType.REQUEST_CANCELLED.getErrorCode(), future.join().getCode());
        assertEquals(
                RequestState.Phase.CANCELLED,
                org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle)
                        .getRequestState("301", 0L)
                        .state());
    }

    @Test
    void repeatedCancellationDuringAdmissionKeepsTheFirstCause() throws Exception {
        CompletableFuture<Response> future =
                RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context(303L));
        PlacementOperation admission = SchedulerTestSupport.claim(lifecycle, "303", future);
        assertNotNull(admission);

        lifecycle.cancel("303", 0L, CancelReason.CLIENT_CANCELLED);
        lifecycle.cancel("303", 0L, CancelReason.DEADLINE_EXCEEDED);
        assertFalse(future.isDone());

        AbstractRequestScheduler.finishPlacement(admission);

        assertEquals(
                StrategyErrorType.REQUEST_CANCELLED.getErrorCode(),
                future.get(5, TimeUnit.SECONDS).getCode());
        assertEquals(
                RequestState.Phase.CANCELLED,
                org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle)
                        .getRequestState("303", 0L)
                        .state());
        assertEquals(
                0,
                org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle)
                        .liveRequestCount());
    }

    @Test
    void retirementFromAnotherEndpointLeavesCurrentRouteUnchanged() {
        Registered registered = registerItem(7112L);
        assertEquals(PlacementResult.Status.SUCCESS, commitRoute(lifecycle, registered));
        var route = registered.item();
        var context = route.ctx();
        var before = context.snapshot();
        lifecycle.onEndpointRetired(
                context,
                new ContextResult.EndpointRetirement(
                        route,
                        route.decodeReservation(),
                        RoleType.DECODE,
                        mock(DecodeEndpoint.class),
                        "old endpoint",
                        123L));
        assertSame(before, context.snapshot());
        assertFalse(registered.future().isDone());
    }

    @Test
    void endpointRetirementDoesNotConfirmWorkerExecution() {
        org.mockito.Mockito.doReturn(true)
                .when((QueuedRequestScheduler) SchedulerTestServices.scheduler(lifecycle))
                .enqueue(org.mockito.ArgumentMatchers.any());
        Registered registered = registerItem(7113L);
        var route = registered.item();
        var context = route.ctx();
        try (var admission = lifecycle.claimPlacement(context);
                var finish = RequestProtocolTestSupport.finishOnExit(admission)) {
            assertNotNull(admission);
            assertEquals(
                    PlacementResult.Status.SUCCESS,
                    RequestProtocolTestSupport.commitRoute(
                            lifecycle, route, RequestProtocolTestSupport.publication(() -> true)));
            var before = context.snapshot().engine();
            lifecycle.onEndpointRetired(
                    context,
                    new ContextResult.EndpointRetirement(
                            route,
                            route.decodeReservation(),
                            RoleType.DECODE,
                            route.decodeEndpoint(),
                            "endpoint retired",
                            System.currentTimeMillis()));
            assertSame(before, context.snapshot().engine());
            assertFalse(context.snapshot().engine().decodeAccepted());
            assertNotNull(context.snapshot().routeFailure());
            assertFalse(registered.future().isDone());
        }
    }

    @Test
    void workerTerminalDuringAdmissionWinsOverRetirementAndAdmissionFailure() throws Exception {
        Registered registered = registerItem(7111L);
        RouteExecution original = registered.item();
        RequestContext context = original.ctx();
        PrefillEndpoint prefill = mock(PrefillEndpoint.class);
        when(prefill.getStatus())
                .thenReturn(
                        org.flexlb.dao.master.WorkerStatus.createDiscovered(
                                RoleType.PREFILL, "test", "127.0.0.1", 8000, 8001, "test"));
        RouteExecution route =
                SchedulingTestConfig.createRoute(
                        context,
                        SchedulingTestConfig.routeResponse(original),
                        null,
                        null,
                        prefill,
                        original.decodeEndpoint(),
                        original.decodeReservation(),
                        original.enqueuedAtMs());
        var callbacks = new java.util.concurrent.atomic.AtomicInteger();
        var published =
                registered
                        .future()
                        .thenAccept(
                                response -> {
                                    assertFalse(Thread.holdsLock(context));
                                    callbacks.incrementAndGet();
                                });
        try (var admission =
                        SchedulerTestSupport.claim(
                                lifecycle, route.requestId(), registered.future());
                var finish = RequestProtocolTestSupport.finishOnExit(admission)) {
            assertNotNull(admission);
            assertEquals(
                    PlacementResult.Status.SUCCESS,
                    RequestProtocolTestSupport.commitRoute(
                            lifecycle, route, RequestProtocolTestSupport.publication(() -> true)));
            lifecycle.onEndpointRetired(
                    route.ctx(),
                    new ContextResult.EndpointRetirement(
                            route,
                            route.decodeReservation(),
                            RoleType.PREFILL,
                            route.prefillEndpoint(),
                            "Prefill endpoint generation retired",
                            System.currentTimeMillis()));
            SchedulerTestSupport.awaitRequestWork(
                    org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(lifecycle));
            RequestProtocolTestSupport.applyPrefillStatus(
                    lifecycle,
                    context,
                    prefill,
                    RoleType.PREFILL,
                    new PrefillEndpoint.PrefillRequestStatus(
                            route,
                            org.flexlb.balance.endpoint.PrefillEndpoint.PrefillRequestStatus.Kind
                                    .FAILED,
                            42L));
            assertFalse(
                    registered.future().isDone(),
                    "the routing owner still retains all terminal evidence");
            AbstractRequestScheduler.failPlacement(
                    admission, Response.error(StrategyErrorType.RESOURCE_EXHAUSTED));
            AbstractRequestScheduler.finishPlacement(admission);
        }
        assertEquals(
                StrategyErrorType.WORKER_EXECUTION_FAILED.getErrorCode(),
                registered.future().get(5, TimeUnit.SECONDS).getCode());
        published.get(5, TimeUnit.SECONDS);
        SchedulerTestSupport.awaitRequestWork(
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(lifecycle));
        assertEquals(1, callbacks.get());
        assertEquals(Stage.FINISHED, SchedulerTestSupport.stage(context));
        assertEquals(RequestState.Phase.FAILED, RequestState.from(context.snapshot()).state());
        assertEquals(0, SchedulerTestSupport.repository(lifecycle).liveRequestCount());
        verify(route.decodeEndpoint())
                .release(
                        route.decodeReservation(),
                        DecodeResources.ReleaseReason.COUNTERPART_FINISHED);
        org.mockito.Mockito.verify(prefill, org.mockito.Mockito.never()).releaseRequest(route);
    }

    @Test
    void admissionFailurePreservesAnEarlierCancellation() throws Exception {
        CompletableFuture<Response> future =
                RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context(304L));
        PlacementOperation admission = SchedulerTestSupport.claim(lifecycle, "304", future);
        assertNotNull(admission);

        lifecycle.cancel("304", 0L, CancelReason.CLIENT_CANCELLED);
        AbstractRequestScheduler.failPlacement(
                admission, Response.error(StrategyErrorType.RESOURCE_EXHAUSTED));
        AbstractRequestScheduler.finishPlacement(admission);

        assertEquals(
                StrategyErrorType.REQUEST_CANCELLED.getErrorCode(),
                future.get(5, TimeUnit.SECONDS).getCode());
        assertEquals(
                RequestState.Phase.CANCELLED,
                org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle)
                        .getRequestState("304", 0L)
                        .state());
        assertEquals(
                0,
                org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle)
                        .liveRequestCount());
    }

    @Test
    void shutdownIntentSurvivesAdmissionFailureWithoutAnItem() throws Exception {
        CompletableFuture<Response> future =
                RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context(305L));
        PlacementOperation admission = SchedulerTestSupport.claim(lifecycle, "305", future);
        assertNotNull(admission);
        var requestContext = lifecycle.requests.findActive("305");
        synchronized (requestContext) {
            var ending =
                    requestContext.beginClose(
                            new ContextResult.EndEvidence(
                                    null,
                                    ContextResult.EndKind.SHUTDOWN,
                                    null,
                                    false,
                                    0L,
                                    StrategyErrorType.DISPATCH_FAILED,
                                    "request scheduler is shutting down",
                                    false,
                                    false),
                            System.currentTimeMillis());
            assertNull(ending.outcome());
            assertNotNull(ending.placement());
        }
        AbstractRequestScheduler.failPlacement(
                admission, Response.error(StrategyErrorType.RESOURCE_EXHAUSTED));
        assertEquals(
                StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(),
                future.get(5, TimeUnit.SECONDS).getCode());
        assertEquals(
                RequestState.Phase.FAILED,
                org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle)
                        .getRequestState("305", 0L)
                        .state());
    }

    @Test
    void queueDecisionResponsePublishesOutsideTheDecisionCaller() throws Exception {
        CompletableFuture<Response> future =
                RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context(302L));
        CountDownLatch published = new CountDownLatch(1);
        AtomicReference<String> callbackThread = new AtomicReference<>();
        future.thenAccept(
                response -> {
                    callbackThread.set(Thread.currentThread().getName());
                    published.countDown();
                });

        Response rejection = Response.error(StrategyErrorType.RESOURCE_EXHAUSTED);
        assertTrue(lifecycle.publish(lifecycle.requests.findActive("302"), rejection, false));
        assertTrue(published.await(5, TimeUnit.SECONDS));
        assertNotEquals(Thread.currentThread().getName(), callbackThread.get());
        assertEquals(
                StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(),
                future.get(5, TimeUnit.SECONDS).getCode());
    }

    @Test
    void shutdownGateWaitsForTheExactPlacementOperationAndRejectsNewWork() throws Exception {
        CompletableFuture<Response> heldFuture =
                RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context(401L));
        PlacementOperation held = SchedulerTestSupport.claim(lifecycle, "401", heldFuture);
        assertNotNull(held);
        ExecutorService executor = Executors.newSingleThreadExecutor();
        try {
            Future<Boolean> shutdownOwner =
                    executor.submit(
                            () ->
                                    RequestProtocolTestSupport.closeAdmissionAndAwaitMutations(
                                            lifecycle));
            awaitCondition(
                    org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle)
                            ::isClosed);
            assertFalse(
                    shutdownOwner.isDone(),
                    "shutdown must not overtake an exact admission mutation");

            CompletableFuture<Response> rejected =
                    RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context(402L));
            assertEquals(
                    StrategyErrorType.DISPATCH_FAILED.getErrorCode(), rejected.join().getCode());

            AbstractRequestScheduler.finishPlacement(held);
            assertTrue(shutdownOwner.get(5, TimeUnit.SECONDS));
            lifecycle.closeOutstandingAndTerminalize();
            SchedulerTestSupport.timer(
                            org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(lifecycle))
                    .close();
            SchedulerTestSupport.closeRequestExecutors(
                    org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(lifecycle));
            assertEquals(
                    StrategyErrorType.DISPATCH_FAILED.getErrorCode(),
                    heldFuture.get(5, TimeUnit.SECONDS).getCode());
        } finally {
            executor.shutdownNow();
        }
    }

    @Test
    void cancelRequiresTheExpectedBatchGenerationAndUnknownIdsStayAbsent() {
        CompletableFuture<Response> future =
                RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context(501L));

        assertNull(lifecycle.cancel("999", 0L, CancelReason.CLIENT_CANCELLED));
        assertNull(lifecycle.cancel("501", 91L, CancelReason.CLIENT_CANCELLED));
        assertFalse(future.isDone());

        RequestState exact = lifecycle.cancel("501", 0L, CancelReason.CLIENT_CANCELLED);
        assertNotNull(exact);
        assertEquals(StrategyErrorType.REQUEST_CANCELLED.getErrorCode(), future.join().getCode());
    }

    @Test
    void publishedQueueDeadlineReleasesLocalReservationWithoutEngineCancel() {
        Registered registered = registerItem(602L);
        assertEquals(PlacementResult.Status.SUCCESS, commitRoute(lifecycle, registered));
        RequestContext requestContext = lifecycle.requests.findActive("602");
        synchronized (requestContext) {
            RequestProtocolTestSupport.applyPrefillStatus(
                    lifecycle,
                    requestContext,
                    registered.item().prefillEndpoint(),
                    RoleType.PREFILL,
                    new PrefillEndpoint.PrefillRequestStatus(
                            registered.item(),
                            PrefillEndpoint.PrefillRequestStatus.Kind.ACTIVE,
                            0L));
        }
        lifecycle.cancel("602", 0L, CancelReason.DEADLINE_EXCEEDED);
        SchedulerTestSupport.awaitRequestWork(
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(lifecycle));
        assertEquals(
                RequestState.Phase.TIMED_OUT,
                org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle)
                        .getRequestState("602", 0L)
                        .state());
        assertEquals(0, lifecycle.requests.liveRequestCount());
        verify(registered.item().prefillEndpoint()).releaseRequest(registered.item());
        verify(registered.item().decodeEndpoint())
                .release(
                        registered.item().decodeReservation(),
                        DecodeResources.ReleaseReason.COUNTERPART_FINISHED);
    }

    @Test
    void expiredArrivalDoesNotDisturbTheWaitingRequests() throws Exception {
        var low = RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context(1));
        var expired = context(2);
        expired.data()
                .setSchedulingMetadata(
                        SchedulingMetadata.explicit(90, System.currentTimeMillis() - 1L));
        // QUEUE expiry before placement is admission capacity exhaustion (8431).
        assertEquals(
                StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(),
                RequestProtocolTestSupport.submitForManualPlacement(lifecycle, expired)
                        .get(5, TimeUnit.SECONDS)
                        .getCode());
        assertFalse(low.isDone());
        assertEquals(
                1,
                org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle)
                        .liveRequestCount());
    }

    @Test
    void concurrentGlobalAdmissionRetainsEveryUniqueRequest() throws Exception {
        try (var executor = Executors.newFixedThreadPool(8)) {
            List<Future<CompletableFuture<Response>>> futures = new ArrayList<>();
            for (long id = 1; id <= 128; id++) {
                long requestId = id;
                futures.add(executor.submit(() -> lifecycle.submit(context(requestId))));
            }
            for (var future : futures) {
                assertFalse(future.get(5, TimeUnit.SECONDS).isDone());
            }
            assertEquals(
                    128,
                    org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle)
                            .liveRequestCount());
        }
    }

    @Test
    void oldDeliveryAndPreemptionCapabilitiesCannotReachAReusedRequestId() throws Exception {
        Registered registered = registerItemWithCancelTarget(703L);
        assertEquals(PlacementResult.Status.SUCCESS, commitRoute(lifecycle, registered));
        RequestContext old = lifecycle.requests.findActive("703");
        Delivery.Member delivery =
                RequestProtocolTestSupport.claimBatchWithoutPrediction(
                        lifecycle, registered.item(), 17L, () -> true);
        assertNotNull(delivery);
        PreemptionRegistration preemption =
                lifecycle
                        .tryClaim(
                                new DecodeResources.RequestReservation(
                                        1L, String.valueOf(703L), 1L),
                                19L)
                        .orElseThrow();
        RequestProtocolTestSupport.expireInactiveRequest(
                lifecycle,
                old,
                RequestProtocolTestSupport.<Long>inspect(
                        lifecycle, old, "inactivityExpiresAtMsLocked"));
        registered.future().join();
        assertTrue(lifecycle.requests.isCurrent(old), "sender still owns prepared delivery");
        delivery.route()
                .ctx()
                .data()
                .getRequestScheduler()
                .onDeliveryResult(
                        new ContextResult.TransportResult(
                                delivery,
                                org.flexlb.balance.scheduler.Delivery.Result.notSent(
                                        new IllegalStateException("expired before send")),
                                System.currentTimeMillis()));
        SchedulerTestSupport.awaitRequestWork(
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(lifecycle));
        assertTrue(
                org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle)
                        .removeExactTerminal(
                                org.flexlb.balance.scheduler.SchedulerTestSupport.terminalRecord(
                                        lifecycle,
                                        org.flexlb.balance.scheduler.SchedulerTestSupport
                                                .repository(lifecycle)
                                                .getRequestState("703", 0L)),
                                Long.MAX_VALUE));
        CompletableFuture<Response> replacement =
                RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context(703L));
        assertSame(registered.future(), registered.item().ctx().data().getFuture());
        assertNotSame(replacement, registered.item().ctx().data().getFuture());
        assertThrows(IllegalStateException.class, () -> old.data().setFuture(replacement));
        assertSame(registered.future(), registered.item().ctx().data().getFuture());
        var stale =
                org.flexlb.balance.scheduler.RequestChangeAssertions.record(
                        old,
                        observedRequest ->
                                observedRequest.recordDeliveryResult(
                                        new ContextResult.TransportResult(
                                                delivery,
                                                org.flexlb.balance.scheduler.Delivery.Result
                                                        .delivered(),
                                                System.currentTimeMillis())));
        assertEquals(RequestChangeAssertions.UpdateKind.STALE, stale.kind());
        assertSame(stale.before(), stale.after());
        assertFalse(
                lifecycle.onPreemption(
                        preemption,
                        new AbstractRequestScheduler.PreemptionEvent(
                                null,
                                false,
                                registered.item().decodeEndpoint(),
                                registered.item().decodeReservation(),
                                "late engine cancellation")));
        assertFalse(
                lifecycle.onPreemption(
                        preemption,
                        new AbstractRequestScheduler.PreemptionEvent(
                                null, true, null, null, null)));
        lifecycle.cancel(old, 0L, CancelReason.CLIENT_CANCELLED);
        lifecycle.cancel(
                registered.item().ctx(),
                0L,
                org.flexlb.balance.scheduler.CancelReason.DEADLINE_EXCEEDED);
        lifecycle.rejectRoute(registered.item(), new IllegalStateException("late queue failure"));
        lifecycle.rejectRoute(registered.item(), null);
        RequestContext current = lifecycle.requests.findActive("703");
        ContextResult.Snapshot oldState = old.snapshot();
        ContextResult.Snapshot currentState = current.snapshot();
        RequestProtocolTestSupport.applyPrefillStatus(
                lifecycle,
                old,
                registered.item().prefillEndpoint(),
                RoleType.PREFILL,
                new PrefillEndpoint.PrefillRequestStatus(
                        registered.item(), PrefillEndpoint.PrefillRequestStatus.Kind.ACTIVE, 0L));
        RequestProtocolTestSupport.applyPrefillStatus(
                lifecycle,
                old,
                registered.item().prefillEndpoint(),
                RoleType.PREFILL,
                new PrefillEndpoint.PrefillRequestStatus(
                        registered.item(),
                        org.flexlb.balance.endpoint.PrefillEndpoint.PrefillRequestStatus.Kind
                                .COMPLETED,
                        0L));
        RequestProtocolTestSupport.applyDecodeStatus(
                lifecycle,
                old,
                registered.item().decodeEndpoint(),
                DecodeResources.DecodeRequestStatus.active(registered.item().decodeReservation()));
        RequestProtocolTestSupport.applyDecodeStatus(
                lifecycle,
                old,
                registered.item().decodeEndpoint(),
                DecodeResources.DecodeRequestStatus.terminal(
                        registered.item().decodeReservation(), 0L));
        SchedulerTestSupport.awaitRequestWork(
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(lifecycle));
        assertEquals(
                oldState,
                old.snapshot(),
                "late Worker status must not update the archived Context");
        assertEquals(
                currentState,
                current.snapshot(),
                "late Worker status must not update the reused ID");
        assertFalse(replacement.isDone());
        assertEquals(
                RequestState.Phase.QUEUED,
                org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle)
                        .getRequestState("703", 0L)
                        .state());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void decodeAllocationReconcilesOnlyNotFoundProtocol(boolean acceptedCancel) throws Exception {
        Registered registered = registerItemWithCancelTarget(706L);
        assertEquals(PlacementResult.Status.SUCCESS, commitRoute(lifecycle, registered));
        var delivery =
                RequestProtocolTestSupport.claimBatch(
                        lifecycle, registered.item(), 17L, () -> true);
        assertNotNull(delivery);
        delivery.route()
                .ctx()
                .data()
                .getRequestScheduler()
                .onDeliveryResult(
                        new ContextResult.TransportResult(
                                delivery,
                                org.flexlb.balance.scheduler.Delivery.Result.delivered(),
                                System.currentTimeMillis()));
        SchedulerTestSupport.awaitRequestWork(
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(lifecycle));
        assertTrue(
                registered.future().get(5, TimeUnit.SECONDS).isSuccess(),
                "this case starts after ACK publication; deferred ACK reconciliation is a separate"
                        + " contract");
        var claim =
                lifecycle
                        .tryClaim(
                                new DecodeResources.RequestReservation(
                                        1L, String.valueOf(706L), 1L),
                                20L)
                        .orElseThrow();
        assertTrue(
                lifecycle.onPreemption(
                        claim,
                        new AbstractRequestScheduler.PreemptionEvent(
                                org.flexlb.balance.preemption.PreemptionCancelPhase
                                        .CANCEL_IN_FLIGHT,
                                false,
                                null,
                                null,
                                null)));
        assertTrue(
                lifecycle.onPreemption(
                        claim,
                        new AbstractRequestScheduler.PreemptionEvent(
                                acceptedCancel
                                        ? org.flexlb.balance.preemption.PreemptionCancelPhase
                                                .CANCEL_REQUESTED
                                        : org.flexlb.balance.preemption.PreemptionCancelPhase
                                                .NOT_FOUND_STALE,
                                false,
                                null,
                                null,
                                null)));
        var source = registered.item().decodeEndpoint();
        verify(source, org.mockito.Mockito.never())
                .updatePreemption(
                        org.mockito.ArgumentMatchers.anyLong(), org.mockito.ArgumentMatchers.any());
        when(source.updatePreemption(
                        org.mockito.ArgumentMatchers.eq(20L), org.mockito.ArgumentMatchers.any()))
                .thenReturn(
                        org.flexlb.balance.endpoint.DecodeResources.ResourceChange.of(
                                org.flexlb.balance.endpoint.DecodeResources.MutationStatus.APPLIED,
                                false));
        org.mockito.Mockito.doAnswer(
                        call -> {
                            assertFalse(Thread.holdsLock(registered.item().ctx()));
                            return null;
                        })
                .when(source)
                .publishCapacity(org.flexlb.balance.endpoint.DecodeEndpoint.CapacityEdge.BOTH);
        RequestProtocolTestSupport.applyDecodeStatus(
                lifecycle,
                registered.item().ctx(),
                source,
                DecodeResources.DecodeRequestStatus.allocated(
                        registered.item().decodeReservation()));
        SchedulerTestSupport.awaitRequestWork(
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(lifecycle));
        assertSame(
                acceptedCancel ? claim : null,
                SchedulerTestSupport.preemption(registered.item().ctx()));
        if (acceptedCancel) {
            verify(source, org.mockito.Mockito.never())
                    .updatePreemption(
                            org.mockito.ArgumentMatchers.anyLong(),
                            org.mockito.ArgumentMatchers.any());
        } else {
            verify(source)
                    .updatePreemption(
                            20L,
                            DecodeResources.PreemptionUpdate.active(
                                    registered.item().decodeReservation()));
            verify(source)
                    .publishCapacity(org.flexlb.balance.endpoint.DecodeEndpoint.CapacityEdge.BOTH);
        }
    }

    @Test
    void decodeAllocationSettlesCleanupAfterRetainedTerminalAndNotFound() {
        Registered registered = registerItemWithCancelTarget(707L);
        assertEquals(PlacementResult.Status.SUCCESS, commitRoute(lifecycle, registered));
        var delivery =
                RequestProtocolTestSupport.claimBatch(
                        lifecycle, registered.item(), 17L, () -> true);
        delivery.route()
                .ctx()
                .data()
                .getRequestScheduler()
                .onDeliveryResult(
                        new ContextResult.TransportResult(
                                delivery,
                                org.flexlb.balance.scheduler.Delivery.Result.delivered(),
                                System.currentTimeMillis()));
        SchedulerTestSupport.awaitRequestWork(
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(lifecycle));
        var context = registered.item().ctx();
        var source = registered.item().decodeEndpoint();
        when(source.ownership(registered.item().decodeReservation()))
                .thenReturn(org.flexlb.balance.endpoint.DecodeResources.Ownership.HELD);
        when(source.updatePreemption(
                        org.mockito.ArgumentMatchers.eq(21L), org.mockito.ArgumentMatchers.any()))
                .thenReturn(
                        org.flexlb.balance.endpoint.DecodeResources.ResourceChange.of(
                                org.flexlb.balance.endpoint.DecodeResources.MutationStatus.APPLIED,
                                false));
        var claim =
                lifecycle
                        .tryClaim(
                                new DecodeResources.RequestReservation(
                                        1L, String.valueOf(707L), 1L),
                                21L)
                        .orElseThrow();
        assertTrue(
                lifecycle.onPreemption(
                        claim,
                        new AbstractRequestScheduler.PreemptionEvent(
                                org.flexlb.balance.preemption.PreemptionCancelPhase
                                        .CANCEL_IN_FLIGHT,
                                false,
                                null,
                                null,
                                null)));
        context.beginClose(
                new ContextResult.EndEvidence(
                        registered.item(),
                        ContextResult.EndKind.WORKER,
                        RoleType.PREFILL,
                        true,
                        0L,
                        StrategyErrorType.WORKER_EXECUTION_FAILED,
                        "worker completed",
                        true,
                        false),
                System.currentTimeMillis());
        lifecycle.onDeliveryResult(
                new ContextResult.PreparationFailure(
                        delivery,
                        registered.item(),
                        "Prefill rejected delivery while cancellation was pending",
                        System.currentTimeMillis()));
        SchedulerTestSupport.awaitRequestWork(
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(lifecycle));
        assertTrue(
                lifecycle.onPreemption(
                        claim,
                        new AbstractRequestScheduler.PreemptionEvent(
                                org.flexlb.balance.preemption.PreemptionCancelPhase.NOT_FOUND_STALE,
                                false,
                                null,
                                null,
                                null)));
        assertEquals(Stage.FINALIZING, SchedulerTestSupport.stage(context));
        assertFalse(
                (context.snapshot().preemption() == null
                        || context.snapshot().preemption().finished()),
                "NOT_FOUND is not independent Decode release proof");
        assertSame(context, lifecycle.requests.findActive("707"));
        verify(source, org.mockito.Mockito.never())
                .release(org.mockito.ArgumentMatchers.any(), org.mockito.ArgumentMatchers.any());

        RequestProtocolTestSupport.applyDecodeStatus(
                lifecycle,
                registered.item().ctx(),
                source,
                DecodeResources.DecodeRequestStatus.allocated(
                        registered.item().decodeReservation()));
        SchedulerTestSupport.awaitRequestWork(
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(lifecycle));

        verify(source)
                .updatePreemption(
                        21L,
                        DecodeResources.PreemptionUpdate.finished(
                                registered.item().decodeReservation()));
        assertTrue(
                (context.snapshot().preemption() == null
                        || context.snapshot().preemption().finished()));
        assertEquals(Stage.FINISHED, SchedulerTestSupport.stage(context));
        assertTrue(claim.completion().toCompletableFuture().isDone());
        assertNull(lifecycle.requests.findActive("707"));
        assertNull(SchedulerTestSupport.preemption(context));
    }

    @Test
    void decodeAllocationCarriesSettledResourcesIntoTheFirstTerminalCleanup() {
        Registered registered = registerItemWithCancelTarget(711L);
        assertEquals(PlacementResult.Status.SUCCESS, commitRoute(lifecycle, registered));
        var delivery =
                RequestProtocolTestSupport.claimBatch(
                        lifecycle, registered.item(), 17L, () -> true);
        delivery.route()
                .ctx()
                .data()
                .getRequestScheduler()
                .onDeliveryResult(
                        new ContextResult.TransportResult(
                                delivery,
                                org.flexlb.balance.scheduler.Delivery.Result.delivered(),
                                System.currentTimeMillis()));
        SchedulerTestSupport.awaitRequestWork(
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(lifecycle));
        var context = registered.item().ctx();
        var claim =
                lifecycle
                        .tryClaim(
                                new DecodeResources.RequestReservation(
                                        1L, String.valueOf(711L), 1L),
                                22L)
                        .orElseThrow();
        assertTrue(
                lifecycle.onPreemption(
                        claim,
                        new AbstractRequestScheduler.PreemptionEvent(
                                org.flexlb.balance.preemption.PreemptionCancelPhase
                                        .CANCEL_IN_FLIGHT,
                                false,
                                null,
                                null,
                                null)));
        assertTrue(
                lifecycle.onPreemption(
                        claim,
                        new AbstractRequestScheduler.PreemptionEvent(
                                org.flexlb.balance.preemption.PreemptionCancelPhase.NOT_FOUND_STALE,
                                false,
                                null,
                                null,
                                null)));
        synchronized (context) {
            assertFalse(context.snapshot().phase() == ContextResult.Phase.ENDING);
            context.beginClose(
                    new ContextResult.EndEvidence(
                            registered.item(),
                            ContextResult.EndKind.WORKER,
                            RoleType.PREFILL,
                            true,
                            0L,
                            StrategyErrorType.WORKER_EXECUTION_FAILED,
                            "worker completed",
                            true,
                            false),
                    System.currentTimeMillis());
        }
        var source = registered.item().decodeEndpoint();
        when(source.updatePreemption(
                        22L,
                        DecodeResources.PreemptionUpdate.finished(
                                registered.item().decodeReservation())))
                .thenReturn(
                        org.flexlb.balance.endpoint.DecodeResources.ResourceChange.of(
                                org.flexlb.balance.endpoint.DecodeResources.MutationStatus.APPLIED,
                                false));
        org.mockito.Mockito.doAnswer(
                        call -> {
                            assertFalse(
                                    Thread.holdsLock(context),
                                    "capacity notification must follow the resource CAS outside the"
                                            + " request lock");
                            return null;
                        })
                .when(source)
                .publishCapacity(org.flexlb.balance.endpoint.DecodeEndpoint.CapacityEdge.BOTH);

        RequestProtocolTestSupport.applyDecodeStatus(
                lifecycle,
                registered.item().ctx(),
                source,
                DecodeResources.DecodeRequestStatus.allocated(
                        registered.item().decodeReservation()));
        SchedulerTestSupport.awaitRequestWork(
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(lifecycle));

        verify(source)
                .updatePreemption(
                        22L,
                        DecodeResources.PreemptionUpdate.finished(
                                registered.item().decodeReservation()));
        verify(source)
                .publishCapacity(org.flexlb.balance.endpoint.DecodeEndpoint.CapacityEdge.BOTH);
        verify(source, org.mockito.Mockito.never())
                .release(org.mockito.ArgumentMatchers.any(), org.mockito.ArgumentMatchers.any());
        assertTrue(
                (context.snapshot().preemption() == null
                        || context.snapshot().preemption().finished()));
        assertTrue(claim.completion().toCompletableFuture().isDone());
        assertEquals(Stage.FINISHED, SchedulerTestSupport.stage(context));
        assertEquals(RequestState.Phase.COMPLETED, RequestState.from(context.snapshot()).state());
        assertNull(lifecycle.requests.findActive("711"));
    }

    @Test
    void deliveryCannotConsumeAnotherRoutesHandoffReceipt() {
        Registered registered = registerItem(705L);
        assertEquals(PlacementResult.Status.SUCCESS, commitRoute(lifecycle, registered));
        Registered other = registerItem(706L);
        var wrongReceipt =
                mock(org.flexlb.balance.endpoint.DecodeEndpoint.EngineDispatchPermit.class);
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        DeliveryDispatcher.commitDelivery(
                                lifecycle,
                                new org.flexlb.balance.scheduler.Delivery.Member(
                                        registered.item(), wrongReceipt),
                                Delivery.SendMode.BATCH_ENQUEUE,
                                7L));
        assertNull(SchedulerTestSupport.delivery(registered.item().ctx()));
        assertEquals(Stage.READY_TO_DELIVER, SchedulerTestSupport.stage(registered.item().ctx()));
        assertFalse(registered.future().isDone());
    }

    @Test
    void invalidBatchIdentityCannotTransferEndpointOwnership() {
        Registered registered = registerItem(704L);
        assertEquals(PlacementResult.Status.SUCCESS, commitRoute(lifecycle, registered));
        var transfer = mock(java.util.function.BooleanSupplier.class);

        assertThrows(
                IllegalArgumentException.class,
                () ->
                        DeliveryDispatcher.commitDelivery(
                                lifecycle,
                                new org.flexlb.balance.scheduler.Delivery.Member(
                                        registered.item(),
                                        RequestProtocolTestSupport.handoff(transfer)),
                                Delivery.SendMode.BATCH_ENQUEUE,
                                0L));

        org.mockito.Mockito.verify(transfer, org.mockito.Mockito.never()).getAsBoolean();
        assertEquals(
                RequestState.Phase.QUEUED,
                org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle)
                        .getRequestState("704", 0L)
                        .state());
        assertFalse(registered.future().isDone());
    }

    @Test
    void invalidResultDoesNotConsumeDeliveryAndDuplicateResultCannotChangeItsOutcome()
            throws Exception {
        Registered registered = registerItem(705L);
        assertEquals(PlacementResult.Status.SUCCESS, commitRoute(lifecycle, registered));
        Delivery.Member claim =
                RequestProtocolTestSupport.claimBatch(
                        lifecycle, registered.item(), 23L, () -> true);
        assertNotNull(claim);

        assertThrows(
                NullPointerException.class,
                () ->
                        claim.route()
                                .ctx()
                                .data()
                                .getRequestScheduler()
                                .onDeliveryResult(
                                        new ContextResult.TransportResult(
                                                claim, null, System.currentTimeMillis())));
        assertEquals(
                RequestState.Phase.DISPATCHING,
                org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle)
                        .getRequestState("705", 23L)
                        .state());
        claim.route()
                .ctx()
                .data()
                .getRequestScheduler()
                .onDeliveryResult(
                        new ContextResult.TransportResult(
                                claim,
                                org.flexlb.balance.scheduler.Delivery.Result.delivered(),
                                System.currentTimeMillis()));
        assertTrue(registered.future().get(5, TimeUnit.SECONDS).isSuccess());
        assertThrows(
                IllegalStateException.class,
                () ->
                        claim.route()
                                .ctx()
                                .data()
                                .getRequestScheduler()
                                .onDeliveryResult(
                                        new ContextResult.TransportResult(
                                                claim,
                                                org.flexlb.balance.scheduler.Delivery.Result
                                                        .notSent(
                                                                new IllegalStateException(
                                                                        "duplicate failure")),
                                                System.currentTimeMillis())));

        assertEquals(
                RequestState.Phase.ACKNOWLEDGED,
                org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle)
                        .getRequestState("705", 23L)
                        .state());
        org.mockito.Mockito.verify(registered.item().decodeEndpoint(), org.mockito.Mockito.never())
                .release(
                        registered.item().decodeReservation(),
                        DecodeResources.ReleaseReason.NOT_SENT);
    }

    private RequestContext context(long requestId) {
        return RequestProtocolTestSupport.context(config, requestId);
    }

    @Test
    void preemptionClaimCapturesTheExactCancelTargetAndRejectsStaleReservation() {
        Registered registered = registerItemWithCancelTarget(709L);
        assertEquals(PlacementResult.Status.SUCCESS, commitRoute(lifecycle, registered));
        assertTrue(
                lifecycle
                        .tryClaim(
                                new DecodeResources.RequestReservation(
                                        1L, String.valueOf(709L), 2L),
                                23L)
                        .isEmpty());
        PreemptionRegistration claim =
                lifecycle
                        .tryClaim(
                                new DecodeResources.RequestReservation(
                                        1L, String.valueOf(709L), 1L),
                                24L)
                        .orElseThrow();
        assertEquals(
                new org.flexlb.balance.preemption.CancelTarget("127.0.0.1", 8090),
                claim.cancelTarget());
        var exported = registered.item().prefill();
        exported.setServerIp("changed-after-claim");
        exported.setGrpcPort(0);
        assertEquals("127.0.0.1", claim.cancelTarget().prefillIp());
        assertTrue(
                lifecycle
                        .tryClaim(
                                new DecodeResources.RequestReservation(
                                        1L, String.valueOf(709L), 1L),
                                26L)
                        .isEmpty());
        lifecycle.onPreemption(
                claim, new AbstractRequestScheduler.PreemptionEvent(null, true, null, null, null));
    }

    @Test
    void exactPreemptionRouteWithUnroutableCancelTargetIsAControlFailureAndDoesNotInstallClaim() {
        Registered registered = registerItem(710L, 0);
        assertEquals(PlacementResult.Status.SUCCESS, commitRoute(lifecycle, registered));
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        lifecycle.tryClaim(
                                new DecodeResources.RequestReservation(
                                        1L, String.valueOf(710L), 1L),
                                25L));
        assertNull(SchedulerTestSupport.preemption(registered.item().ctx()));
    }

    @Test
    void preemptionClaimRejectsAnotherDecodeGenerationWithTheSameRequestAndToken() {
        Registered registered = registerItemWithCancelTarget(711L);
        assertEquals(PlacementResult.Status.SUCCESS, commitRoute(lifecycle, registered));
        var current = registered.item().decodeReservation();
        var stale =
                new DecodeResources.RequestReservation(
                        current.endpointGenerationId() + 1L,
                        String.valueOf(current.requestId()),
                        current.reservationToken());
        assertTrue(lifecycle.tryClaim(stale, 27L).isEmpty());
        assertNull(SchedulerTestSupport.preemption(registered.item().ctx()));
        PreemptionRegistration valid = lifecycle.tryClaim(current, 28L).orElseThrow();
        assertEquals(
                new org.flexlb.balance.preemption.CancelTarget("127.0.0.1", 8090),
                valid.cancelTarget());
        lifecycle.onPreemption(
                valid, new AbstractRequestScheduler.PreemptionEvent(null, true, null, null, null));
    }

    private Registered registerItemWithCancelTarget(long requestId) {
        Registered registered = registerItem(requestId);
        var prefill = new org.flexlb.dao.loadbalance.ServerStatus();
        prefill.setServerIp("127.0.0.1");
        prefill.setGrpcPort(8090);
        RouteExecution original = registered.item();
        return new Registered(
                SchedulingTestConfig.createRoute(
                        original.ctx(),
                        SchedulingTestConfig.routeResponse(original),
                        prefill,
                        original.decode(),
                        original.prefillEndpoint(),
                        original.decodeEndpoint(),
                        original.decodeReservation(),
                        original.enqueuedAtMs()),
                registered.future());
    }

    private Registered registerItem(long requestId) {
        return registerItem(requestId, 8090);
    }

    private Registered registerItem(long requestId, int grpcPort) {
        RequestContext context = context(requestId);
        CompletableFuture<Response> future =
                RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context);
        DecodeEndpoint decode = RequestProtocolTestSupport.decodeEndpoint();
        DecodeResources.RequestReservation reservation =
                new DecodeResources.RequestReservation(1L, String.valueOf(requestId), 1L);
        context.data().setFuture(future);
        var selectedPrefill = new org.flexlb.dao.loadbalance.ServerStatus();
        selectedPrefill.setServerIp("127.0.0.1");
        selectedPrefill.setGrpcPort(grpcPort);
        return new Registered(
                SchedulingTestConfig.createRoute(
                        context,
                        new Response(),
                        selectedPrefill,
                        null,
                        null,
                        decode,
                        reservation,
                        System.currentTimeMillis()),
                future);
    }
}
