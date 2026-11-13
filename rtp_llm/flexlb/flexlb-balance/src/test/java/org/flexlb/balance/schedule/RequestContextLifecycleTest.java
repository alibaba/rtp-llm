package org.flexlb.balance.schedule;

import org.flexlb.balance.metrics.DeliveryMetricsReporter;
import org.flexlb.balance.metrics.RequestSchedulerReporter;
import org.flexlb.balance.queue.GlobalQueueEntry;
import org.flexlb.balance.queue.OrderedRequestQueue;
import org.flexlb.balance.requestlog.TheoryCacheHitReporter;
import org.flexlb.balance.schedule.RequestContext.SelectionAttempt;
import org.flexlb.balance.schedule.RequestProtocolTestSupport.Registered;
import org.flexlb.balance.schedule.ResponseCompletionExecutor.RequestFuture;
import org.flexlb.balance.schedule.SchedulerTestSupport.Stage;
import org.flexlb.balance.strategy.RoutingResult;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.service.RoleType;
import org.flexlb.worker.endpoint.DecodeEndpoint;
import org.flexlb.worker.endpoint.PrefillEndpoint;
import org.flexlb.worker.endpoint.Response;
import org.flexlb.worker.endpoint.StrategyErrorType;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import static org.flexlb.balance.schedule.RequestProtocolTestSupport.awaitCondition;
import static org.flexlb.balance.schedule.RequestProtocolTestSupport.commitRoute;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
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
        config = SchedulingTestConfig.newConfig();
        SchedulingTestConfig.usePriorityQueue(config);
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        lifecycle =
                org.flexlb.balance.schedule.SchedulerTestSupport.create(
                        configService,
                        mock(DeliveryMetricsReporter.class),
                        mock(RequestSchedulerReporter.class),
                        mock(TheoryCacheHitReporter.class));
    }

    @AfterEach
    void tearDown() {
        if (RequestProtocolTestSupport.closeRegistrationAndAwaitScheduling(lifecycle)) {
            lifecycle.closeOutstandingAndTerminalize();
            SchedulerTestSupport.timer(
                            org.flexlb.balance.schedule.SchedulerTestSupport.runtime(lifecycle))
                    .close();
            SchedulerTestSupport.closeRequestExecutors(
                    org.flexlb.balance.schedule.SchedulerTestSupport.runtime(lifecycle));
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
        var admission = SchedulerTestSupport.claim(lifecycle, "895", future);
        try (var completion = RequestProtocolTestSupport.finishOnExit(admission)) {
            assertNotNull(lifecycle.currentSelectionAttempt(route));
            context.registerSelectedRoute(admission, route);
            var original = context.snapshot().deadlines().inactivity();
            lifecycle.onRequestEvent(
                    context,
                    new RequestContext.RequestEvent(
                            route,
                            null,
                            org.flexlb.config.service.RoleType.PDFUSION,
                            null,
                            org.flexlb.worker.endpoint.RequestEventType.COMPLETED,
                            0L,
                            "worker complete",
                            original.atMs() - 1));
            assertNull(context.snapshot().requestEndResult());
            lifecycle.onDeadline(original, original.atMs());
            awaitCondition(() -> context.snapshot().deadlines().inactivity() != original);
            awaitCondition(
                    () ->
                            !SchedulerTestSupport.runtime(lifecycle)
                                    .deadlines()
                                    .needsUpdate(context, context.snapshot().deadlines()));
            assertNull(context.snapshot().requestEndResult());
            assertFalse(future.isDone());
        }
    }

    @Test
    void deadlineInstallationDoesNotReadUnboundRequestForDiagnostics() {
        RequestContext context = new RequestContext(config);
        context.setFuture(new CompletableFuture<>());
        var timer =
                SchedulerTestSupport.timer(
                        org.flexlb.balance.schedule.SchedulerTestSupport.runtime(lifecycle));
        var before = context.snapshot();
        org.junit.jupiter.api.Assertions.assertDoesNotThrow(
                () -> timer.update(context, before.deadlines()));
        assertEquals(before, context.snapshot());
        assertNull(context.snapshot().requestId());
    }

    @Test
    void registrationCapturesCurrentIdentityAndKeepsItAfterFutureCompletion() {
        RequestContext context = context(890L);

        context.setRequest(context.getRequest().toBuilder().requestId("891").build());
        var future = RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context);
        assertSame(future, context.getFuture());
        assertEquals("891", context.getRequestId());
        RequestContext other = context(893L);
        assertThrows(IllegalStateException.class, () -> other.setFuture(future));
        assertEquals("893", other.getRequestId());
        var otherFuture = RequestProtocolTestSupport.submitForManualPlacement(lifecycle, other);
        assertFalse(otherFuture.isDone());
        assertSame(otherFuture, other.getFuture());
        assertEquals("893", other.getRequestId());
        assertThrows(
                IllegalStateException.class,
                () ->
                        context.setRequest(
                                context.getRequest().toBuilder().requestId("892").build()));
        assertEquals("891", context.getRequestId());
        assertTrue(future.cancel(false));
        assertEquals("891", context.getRequestId());
        assertThrows(
                IllegalStateException.class, () -> context.setFuture(new CompletableFuture<>()));
        assertTrue(
                RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context).isDone());
        assertSame(future, context.getFuture());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void registrationFreezesPriorityAndDeadline(boolean suppliedMetadata) {
        RequestContext context = context(806L);
        context.setRequest(context.getRequest().toBuilder().rawPriority(73).build());
        context.setSchedulingMetadata(
                suppliedMetadata
                        ? SchedulingMetadata.explicit(64, Long.MAX_VALUE)
                        : SchedulingMetadata.explicit(
                                73, config.resolveExpiresAtMs(context.getTiming().startedAtMs())));
        int priority = context.getPriority();
        long expiresAt = context.getRequestExpiresAtMs();
        RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context);
        assertNotNull(context.getRequest());
        var metadata = context.getSchedulingMetadata();
        assertNotNull(metadata);
        context.setSchedulingMetadata(metadata);
        assertThrows(IllegalStateException.class, () -> context.setSchedulingMetadata(null));
        assertThrows(
                IllegalStateException.class,
                () -> context.setSchedulingMetadata(SchedulingMetadata.explicit(1, 1L)));
        assertThrows(
                IllegalStateException.class,
                () -> context.setRequest(context.getRequest().toBuilder().rawPriority(1).build()));
        org.springframework.test.util.ReflectionTestUtils.setField(
                context, "timing", context.getTiming().withStartedAtMs(1L));
        assertEquals(priority, context.getPriority());
        assertEquals(expiresAt, context.getRequestExpiresAtMs());
        assertEquals(priority, metadata.priority());
    }

    @ParameterizedTest
    @ValueSource(ints = {0, -1, 101, 1, 73, 100})
    void queueReadsFrozenPriorityAndPreservesInternalDefaulting(int priority) {
        var context = context(808L);
        context.setSchedulingMetadata(
                SchedulingMetadata.explicit(
                        priority, config.resolveExpiresAtMs(context.getTiming().startedAtMs())));
        context.setRequest(context.getRequest().toBuilder().rawPriority(priority).build());
        RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context);
        assertEquals(priority, context.getSchedulingMetadata().priority());
        int queuePriority = priority >= 1 && priority <= 100 ? priority : 50;
        var entry = new GlobalQueueEntry(context, null);
        for (boolean priorityOrdering : new boolean[] {false, true}) {
            var queue = new OrderedRequestQueue(priorityOrdering);
            queue.add(entry);
            assertThrows(
                    IllegalStateException.class,
                    () ->
                            context.setRequest(
                                    context.getRequest().toBuilder().rawPriority(99).build()));
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
                    context.getFuture() instanceof RequestFuture,
                    "registration must wait for the Context monitor");
            context.setFuture(placeholder);
        }
        registrar.join(5_000L);
        assertFalse(registrar.isAlive());
        assertNotNull(registered.get());
        assertSame(registered.get(), context.getFuture());
        assertTrue(context.getFuture() instanceof RequestFuture);
        assertFalse(context.getFuture().isDone());
    }

    @Test
    void duplicateSubmitPreservesCanonicalFutureAndRequestId() {
        RequestContext context = context(800L);
        CompletableFuture<Response> canonical =
                RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context);
        assertThrows(
                IllegalStateException.class,
                () ->
                        context.setRequest(
                                context.getRequest().toBuilder().requestId("801").build()));
        assertEquals(
                StrategyErrorType.INVALID_REQUEST.getErrorCode(),
                RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context)
                        .join()
                        .getCode());
        assertSame(canonical, context.getFuture());
        assertEquals("800", context.getRequestId());
        assertEquals("800", context.getRequest().getRequestId());
        assertThrows(
                IllegalStateException.class, () -> context.setFuture(new CompletableFuture<>()));
        lifecycle.cancel("800", CancelReason.CLIENT_CANCELLED);
        assertEquals(
                StrategyErrorType.REQUEST_CANCELLED.getErrorCode(), canonical.join().getCode());
        assertNull(lifecycle.repository.findActive("800"));
        assertEquals(
                RequestResult.State.CANCELLED,
                org.flexlb.balance.schedule.SchedulerTestSupport.repository(lifecycle)
                        .getRequestState("800")
                        .state());
        assertNull(
                org.flexlb.balance.schedule.SchedulerTestSupport.repository(lifecycle)
                        .getRequestState("801"));
    }

    @Test
    void rejectedDuplicateSubmitCannotReplaceTheCanonicalCancellation() {
        var queue =
                (QueuedRequestScheduler)
                        org.flexlb.balance.schedule.SchedulerTestServices.scheduler(lifecycle);
        org.mockito.Mockito.doNothing()
                .when(queue)
                .removeRequest(org.mockito.ArgumentMatchers.any());
        RequestContext context = context(804L);
        CompletableFuture<Response> canonical =
                RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context);
        lifecycle.cancel("804", CancelReason.CLIENT_CANCELLED);

        assertEquals(
                StrategyErrorType.INVALID_REQUEST.getErrorCode(),
                SchedulerTestServices.entry(queue).submit(context).join().getCode());
        assertEquals(RequestContext.Phase.CLOSED, context.snapshot().phase());
        assertNull(lifecycle.repository.findActive("804"));

        assertEquals(
                StrategyErrorType.REQUEST_CANCELLED.getErrorCode(), canonical.join().getCode());
    }

    @Test
    void expiredTerminalAllowsNewContextButCannotReuseOldContext() {
        RequestContext old = context(802L);
        CompletableFuture<Response> original =
                RequestProtocolTestSupport.submitForManualPlacement(lifecycle, old);
        lifecycle.cancel("802", CancelReason.CLIENT_CANCELLED);
        original.join();
        assertTrue(
                org.flexlb.balance.schedule.SchedulerTestSupport.repository(lifecycle)
                        .removeExactTerminal(
                                org.flexlb.balance.schedule.SchedulerTestSupport
                                        .requestRecordForArchive(
                                                lifecycle,
                                                org.flexlb.balance.schedule.SchedulerTestSupport
                                                        .repository(lifecycle)
                                                        .getRequestState("802")),
                                Long.MAX_VALUE));
        assertEquals(
                StrategyErrorType.INVALID_REQUEST.getErrorCode(),
                RequestProtocolTestSupport.submitForManualPlacement(lifecycle, old)
                        .join()
                        .getCode());
        assertSame(original, old.getFuture());
        CompletableFuture<Response> replacement =
                RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context(802L));
        assertFalse(replacement.isDone());
        assertSame(replacement, lifecycle.repository.findActive("802").getFuture());
        var queue =
                (QueuedRequestScheduler)
                        org.flexlb.balance.schedule.SchedulerTestServices.scheduler(lifecycle);
        queue.removeRequest(old);
        assertNull(SchedulerTestServices.lifecycle(queue).cancel(old, CancelReason.SHUTDOWN));
        assertSame(replacement, lifecycle.repository.findActive("802").getFuture());
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
        assertSame(canonical, lifecycle.repository.findActive("101").getFuture());
        assertEquals(
                1,
                org.flexlb.balance.schedule.SchedulerTestSupport.repository(lifecycle)
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
                RequestResult.State.QUEUED,
                org.flexlb.balance.schedule.SchedulerTestSupport.repository(lifecycle)
                        .getRequestState("10001")
                        .state());
        assertEquals(
                1,
                org.flexlb.balance.schedule.SchedulerTestSupport.repository(lifecycle)
                        .liveRequestCount());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void futureCancelDuringAdmissionIsSynchronousAndTheAdmissionOwnerSettlesIt(
            boolean mayInterruptIfRunning) {
        QueuedRequestScheduler queue =
                (QueuedRequestScheduler)
                        org.flexlb.balance.schedule.SchedulerTestServices.scheduler(lifecycle);
        CompletableFuture<Response> future =
                RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context(10003L));
        RequestContext requestContext = lifecycle.repository.findActive("10003");
        queue.scheduleRequest(requestContext);
        SelectionAttempt admission = SchedulerTestSupport.claim(lifecycle, "10003", future);
        assertNotNull(admission);
        assertTrue(future.cancel(mayInterruptIfRunning));
        assertTrue(future.isCancelled());
        assertEquals(
                RequestResult.State.CANCEL_REQUESTED,
                RequestResult.from(requestContext.snapshot()).state());
        admission
                .requestContext()
                .getRequestScheduler()
                .finishSelectionWithoutCommit(admission, null);
        assertEquals(0, RequestProtocolTestSupport.queuedCount(queue));
        assertEquals(
                RequestResult.State.CANCELLED,
                RequestResult.from(requestContext.snapshot()).state());
        assertEquals(
                0,
                org.flexlb.balance.schedule.SchedulerTestSupport.repository(lifecycle)
                        .liveRequestCount());
        assertEquals(0, RequestProtocolTestSupport.queuedCount(queue));
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
        var admission = SchedulerTestSupport.claim(lifecycle, context.getRequestId(), future);
        try (var admissionCompletion3 = RequestProtocolTestSupport.finishOnExit(admission)) {
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
                        RoutingResult.Status.BLOCKED,
                        RequestProtocolTestSupport.commitRoute(
                                lifecycle,
                                item,
                                RequestProtocolTestSupport.publication(() -> false)));
            }
            assertNull(context.snapshot().route());
            assertEquals(Stage.ROUTING, SchedulerTestSupport.stage(context));
        }
        assertEquals(Stage.QUEUED, SchedulerTestSupport.stage(context));
    }

    @Test
    void globalCloseConsumesAnAcceptedInactivityFactBeforeShutdownFailure() throws Exception {
        CompletableFuture<Response> future =
                RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context(10002L));
        RequestContext context = lifecycle.repository.findActive("10002");
        lifecycle.onDeadline(
                RequestProtocolTestSupport.<ExpirationTimer.Token>field(
                        context, "inactivityDeadline"),
                Long.MAX_VALUE);
        lifecycle.closeRequest(
                context,
                new RequestContext.RequestEndEvent(
                        context.snapshot().route(),
                        RequestContext.EndKind.SHUTDOWN,
                        false,
                        0L,
                        StrategyErrorType.DISPATCH_FAILED,
                        "shutdown"));
        assertEquals(RequestResult.State.TIMED_OUT, RequestResult.from(context.snapshot()).state());
        assertFalse(future.get(5, TimeUnit.SECONDS).isSuccess());
    }

    @Test
    void terminalRecordPreservesIdentityWithoutRetainingRequestContext() throws Exception {
        RequestContext context = context(103L);
        CompletableFuture<Response> future =
                RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context);
        lifecycle.cancel("103", CancelReason.CLIENT_CANCELLED);
        assertEquals(
                StrategyErrorType.REQUEST_CANCELLED.getErrorCode(),
                future.get(5, TimeUnit.SECONDS).getCode());
        awaitCondition(
                () ->
                        org.springframework.test.util.ReflectionTestUtils.getField(future, "target")
                                == null);
        RequestResult terminal =
                org.flexlb.balance.schedule.SchedulerTestSupport.repository(lifecycle)
                        .getRequestState("103");
        assertEquals(RequestResult.State.CANCELLED, terminal.state());
        assertNull(lifecycle.repository.findActive("103"));
        assertSame(
                terminal,
                org.flexlb.balance.schedule.SchedulerTestSupport.repository(lifecycle)
                        .getRequestState("103"));
        assertEquals(
                StrategyErrorType.INVALID_REQUEST.getErrorCode(),
                RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context(103L))
                        .join()
                        .getCode());
    }

    @Test
    void globalWaitingRequestsHaveNoQuantityAdmissionLimit() {
        var low = context(1L);
        low.setSchedulingMetadata(SchedulingMetadata.explicit(10, Long.MAX_VALUE));
        var waiting = RequestProtocolTestSupport.submitForManualPlacement(lifecycle, low);
        for (long id = 2; id <= 1001; id++) {
            var high = context(id);
            high.setSchedulingMetadata(SchedulingMetadata.explicit(90, Long.MAX_VALUE));
            assertFalse(
                    RequestProtocolTestSupport.submitForManualPlacement(lifecycle, high).isDone());
        }
        assertFalse(waiting.isDone(), "higher priority arrivals must not evict waiting requests");
        assertEquals(
                1001,
                org.flexlb.balance.schedule.SchedulerTestSupport.repository(lifecycle)
                        .liveRequestCount());
        assertEquals(
                StrategyErrorType.INVALID_REQUEST.getErrorCode(),
                RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context(1L))
                        .join()
                        .getCode());
        for (long id = 1; id <= 1001; id++) {
            lifecycle.cancel(String.valueOf(id), CancelReason.CLIENT_CANCELLED);
        }
        assertEquals(
                0,
                org.flexlb.balance.schedule.SchedulerTestSupport.repository(lifecycle)
                        .liveRequestCount());
    }

    @Test
    void globalWaitingCancellationDoesNotWaitForDecisionOwnerTicket() throws Exception {
        QueuedRequestScheduler queue =
                (QueuedRequestScheduler)
                        org.flexlb.balance.schedule.SchedulerTestServices.scheduler(lifecycle);
        org.mockito.Mockito.doNothing()
                .when(queue)
                .removeRequest(org.mockito.ArgumentMatchers.any());
        RequestContext context = context(1002L);
        CompletableFuture<Response> future =
                RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context);

        assertEquals(
                RequestResult.State.CANCELLED,
                lifecycle.cancel("1002", CancelReason.CLIENT_CANCELLED).state());
        assertNull(lifecycle.repository.findActive("1002"));
        verify(queue).removeRequest(context);

        assertEquals(
                StrategyErrorType.REQUEST_CANCELLED.getErrorCode(),
                future.get(5, TimeUnit.SECONDS).getCode());
        assertEquals(
                RequestResult.State.CANCELLED,
                org.flexlb.balance.schedule.SchedulerTestSupport.repository(lifecycle)
                        .getRequestState("1002")
                        .state());
    }

    @Test
    void publicQueriesOwnTheirLockAndPrivateDecisionsStillRequireIt() {
        RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context(102L));
        RequestContext requestContext = lifecycle.repository.findActive("102");
        assertNull(requestContext.snapshot().route());
        assertTrue(SchedulerTestSupport.isOpen(requestContext));
        assertTrue(requestContext.snapshot().phase() != RequestContext.Phase.CLOSED);
        var before = requestContext.snapshot();
        assertFalse(
                requestContext.recordEvent(
                        new RequestContext.RequestEvent(
                                null,
                                null,
                                RoleType.PREFILL,
                                null,
                                org.flexlb.worker.endpoint.RequestEventType.ACTIVE,
                                0L,
                                null,
                                System.currentTimeMillis())));
        assertEquals(before, requestContext.snapshot());
        assertEquals(
                RequestResult.State.QUEUED, RequestResult.from(requestContext.snapshot()).state());
    }

    @Test
    void deliveredRequestsHaveNoExtraGlobalQuantityGate() {
        for (long id = 201; id <= 401; id++) {
            Registered registered = registerItem(id);
            assertEquals(RoutingResult.Status.SUCCESS, commitRoute(lifecycle, registered));
            assertNotNull(
                    RequestProtocolTestSupport.claimRoute(
                            lifecycle, registered.item(), () -> true));
        }
        assertEquals(
                201,
                org.flexlb.balance.schedule.SchedulerTestSupport.repository(lifecycle)
                        .liveRequestCount());
    }

    @Test
    void failedPublicationKeepsSchedulingStageAndAllowsExactRetry() {
        Registered registered = registerItem(706L);
        RequestContext requestContext = lifecycle.repository.findActive("706");
        SelectionAttempt admission =
                SchedulerTestSupport.claim(lifecycle, "706", registered.future());
        try (var admissionCompletion6 = RequestProtocolTestSupport.finishOnExit(admission)) {
            assertNotNull(admission);
            assertEquals(
                    RoutingResult.Status.BLOCKED,
                    RequestProtocolTestSupport.commitRoute(
                            lifecycle,
                            registered.item(),
                            RequestProtocolTestSupport.publication(() -> false)));
            assertEquals(Stage.ROUTING, SchedulerTestSupport.stage(requestContext));
            assertNull(requestContext.snapshot().route());
            assertEquals(
                    RequestResult.State.QUEUED,
                    RequestResult.from(requestContext.snapshot()).state());
            assertEquals(
                    RoutingResult.Status.SUCCESS,
                    RequestProtocolTestSupport.commitRoute(
                            lifecycle,
                            registered.item(),
                            RequestProtocolTestSupport.publication(() -> true)));
            assertEquals(Stage.RESULT_PENDING, SchedulerTestSupport.stage(requestContext));
        }
    }

    @Test
    void stoppedLocalOwnerFallsBackToSharedContinuationInsteadOfTimerThread() throws Exception {
        RequestContext context = context(708L);
        CompletableFuture<Response> future =
                RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context);
        PrefillEndpoint prefill = mock(PrefillEndpoint.class);
        context.setFuture(future);
        RequestResult item =
                SchedulingTestConfig.createRoute(
                        context,
                        new Response(),
                        null,
                        null,
                        prefill,
                        null,
                        null,
                        System.currentTimeMillis());
        SelectionAttempt admission = SchedulerTestSupport.claim(lifecycle, "708", future);
        try (var admissionCompletion8 = RequestProtocolTestSupport.finishOnExit(admission)) {
            assertNotNull(admission);
            assertEquals(
                    RoutingResult.Status.SUCCESS,
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
        RequestContext requestContext = lifecycle.repository.findActive("708");
        lifecycle.onDeadline(
                RequestProtocolTestSupport.<ExpirationTimer.Token>field(
                        requestContext, "inactivityDeadline"),
                Long.MAX_VALUE);
        assertTrue(future.get(5, TimeUnit.SECONDS).isSuccess());
        SchedulerTestSupport.awaitRequestWork(
                org.flexlb.balance.schedule.SchedulerTestSupport.runtime(lifecycle));
        assertNotNull(cleanupThread.get());
        assertNotEquals(Thread.currentThread(), cleanupThread.get());
        verify(prefill).releaseRequest(item);
        assertEquals(
                RequestResult.State.TIMED_OUT,
                RequestResult.from(requestContext.snapshot()).state());
    }

    @Test
    void terminalRecordDropsUnconsumedExactOwnerFactReferences() {
        RequestContext context = context(709L);
        CompletableFuture<Response> future =
                RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context);
        PrefillEndpoint prefill = mock(PrefillEndpoint.class);
        context.setFuture(future);
        RequestResult item =
                SchedulingTestConfig.createRoute(
                        context,
                        new Response(),
                        null,
                        null,
                        prefill,
                        null,
                        null,
                        System.currentTimeMillis());
        SelectionAttempt admission = SchedulerTestSupport.claim(lifecycle, "709", future);
        try (var admissionCompletion9 = RequestProtocolTestSupport.finishOnExit(admission)) {
            assertNotNull(admission);
            assertEquals(
                    RoutingResult.Status.SUCCESS,
                    RequestProtocolTestSupport.commitRoute(
                            lifecycle, item, RequestProtocolTestSupport.publication(() -> true)));
        }
        RequestContext requestContext = lifecycle.repository.findActive("709");
        lifecycle.onDeadline(
                RequestProtocolTestSupport.<ExpirationTimer.Token>field(
                        requestContext, "inactivityDeadline"),
                Long.MAX_VALUE);
        try {
            assertTrue(RequestProtocolTestSupport.closeRegistrationAndAwaitScheduling(lifecycle));
            lifecycle.closeOutstandingAndTerminalize();
            SchedulerTestSupport.awaitRequestWork(
                    org.flexlb.balance.schedule.SchedulerTestSupport.runtime(lifecycle));
            assertEquals(Stage.FINISHED, SchedulerTestSupport.stage(requestContext));
            assertNull(requestContext.snapshot().route());
            assertNull(requestContext.snapshot().route());
            assertNull(RequestProtocolTestSupport.field(requestContext, "requestDeadline"));
            assertNull(RequestProtocolTestSupport.field(requestContext, "decisionDeadline"));
            assertNull(
                    RequestProtocolTestSupport.<ExpirationTimer.Token>field(
                            requestContext, "inactivityDeadline"));
            assertNull(lifecycle.repository.findActive("709"));
            RequestResult terminal =
                    org.flexlb.balance.schedule.SchedulerTestSupport.repository(lifecycle)
                            .getRequestState("709");
            assertTrue(terminal.state().isTerminal());
            assertSame(
                    terminal,
                    org.flexlb.balance.schedule.SchedulerTestSupport.repository(lifecycle)
                            .getRequestState("709"));
        } finally {
            SchedulerTestSupport.timer(
                            org.flexlb.balance.schedule.SchedulerTestSupport.runtime(lifecycle))
                    .close();
            SchedulerTestSupport.closeRequestExecutors(
                    org.flexlb.balance.schedule.SchedulerTestSupport.runtime(lifecycle));
        }
    }

    @Test
    void admissionHandleDefersCancellationUntilItsExactCapabilityCloses() {
        CompletableFuture<Response> future =
                RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context(301L));
        SelectionAttempt scope = SchedulerTestSupport.claim(lifecycle, "301", future);
        assertNotNull(scope);

        RequestResult requested = lifecycle.cancel("301", CancelReason.CLIENT_CANCELLED);

        assertEquals(RequestResult.State.CANCEL_REQUESTED, requested.state());
        assertFalse(
                future.isDone(), "the admission mutation still owns rollback and terminal cleanup");

        scope.requestContext().getRequestScheduler().finishSelectionWithoutCommit(scope, null);

        assertEquals(StrategyErrorType.REQUEST_CANCELLED.getErrorCode(), future.join().getCode());
        assertEquals(
                RequestResult.State.CANCELLED,
                org.flexlb.balance.schedule.SchedulerTestSupport.repository(lifecycle)
                        .getRequestState("301")
                        .state());
    }

    @Test
    void repeatedCancellationDuringAdmissionKeepsTheFirstCause() throws Exception {
        CompletableFuture<Response> future =
                RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context(303L));
        SelectionAttempt admission = SchedulerTestSupport.claim(lifecycle, "303", future);
        assertNotNull(admission);

        lifecycle.cancel("303", CancelReason.CLIENT_CANCELLED);
        lifecycle.cancel("303", CancelReason.DEADLINE_EXCEEDED);
        assertFalse(future.isDone());

        admission
                .requestContext()
                .getRequestScheduler()
                .finishSelectionWithoutCommit(admission, null);

        assertEquals(
                StrategyErrorType.REQUEST_CANCELLED.getErrorCode(),
                future.get(5, TimeUnit.SECONDS).getCode());
        assertEquals(
                RequestResult.State.CANCELLED,
                org.flexlb.balance.schedule.SchedulerTestSupport.repository(lifecycle)
                        .getRequestState("303")
                        .state());
        assertEquals(
                0,
                org.flexlb.balance.schedule.SchedulerTestSupport.repository(lifecycle)
                        .liveRequestCount());
    }

    @Test
    void retirementFromAnotherEndpointLeavesCurrentRouteUnchanged() {
        Registered registered = registerItem(7112L);
        assertEquals(RoutingResult.Status.SUCCESS, commitRoute(lifecycle, registered));
        var route = registered.item();
        var context = route.ctx();
        var before = context.snapshot();
        lifecycle.onRequestEvent(
                context,
                new RequestContext.RequestEvent(
                        route,
                        route.decodeRegistration(),
                        RoleType.DECODE,
                        mock(DecodeEndpoint.class),
                        org.flexlb.worker.endpoint.RequestEventType.ENDPOINT_CLOSED,
                        0L,
                        "old endpoint",
                        123L));
        assertEquals(before, context.snapshot());
        assertTrue(registered.future().join().isSuccess());
    }

    @Test
    void workerTerminalDuringAdmissionWinsOverRetirementAndAdmissionFailure() throws Exception {
        Registered registered = registerItem(7111L);
        RequestResult original = registered.item();
        RequestContext context = original.ctx();
        PrefillEndpoint prefill = mock(PrefillEndpoint.class);
        when(prefill.getStatus())
                .thenReturn(
                        org.flexlb.worker.status.WorkerStatus.createDiscovered(
                                RoleType.PREFILL, "test", "127.0.0.1", 8000, 8001, "test"));
        RequestResult route =
                SchedulingTestConfig.createRoute(
                        context,
                        SchedulingTestConfig.routeResponse(original),
                        null,
                        null,
                        prefill,
                        original.decodeEndpoint(),
                        original.decodeRegistration(),
                        original.ctx().getTiming().startedAtMs());
        var callbacks = new java.util.concurrent.atomic.AtomicInteger();
        var published =
                registered
                        .future()
                        .thenAccept(
                                response -> {
                                    assertFalse(Thread.holdsLock(context));
                                    callbacks.incrementAndGet();
                                });
        var admission =
                SchedulerTestSupport.claim(lifecycle, route.requestId(), registered.future());
        try (var finish = RequestProtocolTestSupport.finishOnExit(admission)) {
            assertNotNull(admission);
            context.registerSelectedRoute(admission, route);
            lifecycle.onRequestEvent(
                    route.ctx(),
                    new RequestContext.RequestEvent(
                            route,
                            route.decodeRegistration(),
                            RoleType.PREFILL,
                            route.prefillEndpoint(),
                            org.flexlb.worker.endpoint.RequestEventType.ENDPOINT_CLOSED,
                            0L,
                            "Prefill endpoint generation retired",
                            System.currentTimeMillis()));
            SchedulerTestSupport.awaitRequestWork(
                    org.flexlb.balance.schedule.SchedulerTestSupport.runtime(lifecycle));
            RequestProtocolTestSupport.applyPrefillStatus(
                    lifecycle,
                    context,
                    prefill,
                    RoleType.PREFILL,
                    new PrefillEndpoint.PrefillRequestStatus(
                            route, org.flexlb.worker.endpoint.RequestEventType.FAILED, 42L));
            assertFalse(
                    registered.future().isDone(),
                    "the routing owner still retains all terminal evidence");
            admission
                    .requestContext()
                    .getRequestScheduler()
                    .finishSelectionWithoutCommit(
                            admission, Response.error(StrategyErrorType.RESOURCE_EXHAUSTED));
            admission
                    .requestContext()
                    .getRequestScheduler()
                    .finishSelectionWithoutCommit(admission, null);
        }
        assertEquals(
                StrategyErrorType.WORKER_EXECUTION_FAILED.getErrorCode(),
                registered.future().get(5, TimeUnit.SECONDS).getCode());
        published.get(5, TimeUnit.SECONDS);
        SchedulerTestSupport.awaitRequestWork(
                org.flexlb.balance.schedule.SchedulerTestSupport.runtime(lifecycle));
        assertEquals(1, callbacks.get());
        assertEquals(Stage.FINISHED, SchedulerTestSupport.stage(context));
        assertEquals(RequestResult.State.FAILED, RequestResult.from(context.snapshot()).state());
        assertEquals(0, SchedulerTestSupport.repository(lifecycle).liveRequestCount());
        verify(route.decodeEndpoint()).releaseRequest(route.decodeRegistration());
        org.mockito.Mockito.verify(prefill, org.mockito.Mockito.never()).releaseRequest(route);
    }

    @Test
    void admissionFailurePreservesAnEarlierCancellation() throws Exception {
        CompletableFuture<Response> future =
                RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context(304L));
        SelectionAttempt admission = SchedulerTestSupport.claim(lifecycle, "304", future);
        assertNotNull(admission);

        lifecycle.cancel("304", CancelReason.CLIENT_CANCELLED);
        admission
                .requestContext()
                .getRequestScheduler()
                .finishSelectionWithoutCommit(
                        admission, Response.error(StrategyErrorType.RESOURCE_EXHAUSTED));
        admission
                .requestContext()
                .getRequestScheduler()
                .finishSelectionWithoutCommit(admission, null);

        assertEquals(
                StrategyErrorType.REQUEST_CANCELLED.getErrorCode(),
                future.get(5, TimeUnit.SECONDS).getCode());
        assertEquals(
                RequestResult.State.CANCELLED,
                org.flexlb.balance.schedule.SchedulerTestSupport.repository(lifecycle)
                        .getRequestState("304")
                        .state());
        assertEquals(
                0,
                org.flexlb.balance.schedule.SchedulerTestSupport.repository(lifecycle)
                        .liveRequestCount());
    }

    @Test
    void shutdownIntentSurvivesAdmissionFailureWithoutAnItem() throws Exception {
        CompletableFuture<Response> future =
                RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context(305L));
        SelectionAttempt admission = SchedulerTestSupport.claim(lifecycle, "305", future);
        assertNotNull(admission);
        var requestContext = lifecycle.repository.findActive("305");
        synchronized (requestContext) {
            var ending =
                    requestContext.endRequest(
                            new RequestContext.RequestEndEvent(
                                    null,
                                    RequestContext.EndKind.SHUTDOWN,
                                    false,
                                    0L,
                                    StrategyErrorType.DISPATCH_FAILED,
                                    "request scheduler is shutting down"),
                            System.currentTimeMillis(),
                            null);
            assertNull(ending.requestEndResult());
            assertNotNull(ending.selectionId());
        }
        admission
                .requestContext()
                .getRequestScheduler()
                .finishSelectionWithoutCommit(
                        admission, Response.error(StrategyErrorType.RESOURCE_EXHAUSTED));
        assertEquals(
                StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(),
                future.get(5, TimeUnit.SECONDS).getCode());
        assertEquals(
                RequestResult.State.FAILED,
                org.flexlb.balance.schedule.SchedulerTestSupport.repository(lifecycle)
                        .getRequestState("305")
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
        assertTrue(lifecycle.publish(lifecycle.repository.findActive("302"), rejection, false));
        assertTrue(published.await(5, TimeUnit.SECONDS));
        assertNotEquals(Thread.currentThread().getName(), callbackThread.get());
        assertEquals(
                StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(),
                future.get(5, TimeUnit.SECONDS).getCode());
    }

    @Test
    void shutdownWaitsForPlacementTaskAndRejectsNewWork() throws Exception {
        var context = context(401L);
        CompletableFuture<Response> heldFuture =
                RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context);
        var runtime = SchedulerTestSupport.runtime(lifecycle);
        var entered = new CountDownLatch(1);
        var release = new CountDownLatch(1);
        runtime.tasks()
                .execute(
                        () -> {
                            var operation = lifecycle.tryBeginRouteSelection(context);
                            assertNotNull(operation);
                            try {
                                entered.countDown();
                                assertTrue(release.await(5, TimeUnit.SECONDS));
                            } catch (InterruptedException interrupted) {
                                throw new AssertionError(interrupted);
                            } finally {
                                operation
                                        .requestContext()
                                        .getRequestScheduler()
                                        .finishSelectionWithoutCommit(operation, null);
                            }
                        });
        assertTrue(entered.await(5, TimeUnit.SECONDS));
        var shutdown = CompletableFuture.runAsync(() -> SchedulerTestSupport.shutdown(runtime));
        try {
            awaitCondition(runtime.requests()::isClosed);
            assertThrows(
                    java.util.concurrent.TimeoutException.class,
                    () -> shutdown.get(100, TimeUnit.MILLISECONDS));

            CompletableFuture<Response> rejected =
                    RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context(402L));
            assertEquals(
                    StrategyErrorType.SCHEDULER_UNAVAILABLE.getErrorCode(),
                    rejected.join().getCode());
        } finally {
            release.countDown();
            shutdown.get(5, TimeUnit.SECONDS);
        }
        assertFalse(heldFuture.get(5, TimeUnit.SECONDS).isSuccess());
        assertNull(context.currentSelectionAttempt());
        assertEquals(0, runtime.requests().liveRequestCount());
    }

    @Test
    void cancelUsesRequestIdentityAndUnknownIdsStayAbsent() {
        CompletableFuture<Response> future =
                RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context(501L));

        assertNull(lifecycle.cancel("999", CancelReason.CLIENT_CANCELLED));
        RequestResult exact = lifecycle.cancel("501", CancelReason.CLIENT_CANCELLED);
        assertNotNull(exact);
        assertEquals(StrategyErrorType.REQUEST_CANCELLED.getErrorCode(), future.join().getCode());
    }

    @Test
    void expiredArrivalDoesNotDisturbTheWaitingRequests() throws Exception {
        var low = RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context(1));
        var expired = context(2);
        expired.setSchedulingMetadata(
                SchedulingMetadata.explicit(90, System.currentTimeMillis() - 1L));
        // Expiry before selection establishes a queue timeout, not a capacity rejection.
        assertEquals(
                StrategyErrorType.QUEUE_TIMEOUT.getErrorCode(),
                RequestProtocolTestSupport.submitForManualPlacement(lifecycle, expired)
                        .get(5, TimeUnit.SECONDS)
                        .getCode());
        assertFalse(low.isDone());
        assertEquals(
                1,
                org.flexlb.balance.schedule.SchedulerTestSupport.repository(lifecycle)
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
                    org.flexlb.balance.schedule.SchedulerTestSupport.repository(lifecycle)
                            .liveRequestCount());
        }
    }

    private Registered registerItem(long requestId) {
        return registerItem(requestId, 8090);
    }

    private Registered registerItem(long requestId, int grpcPort) {
        RequestContext context = context(requestId);
        CompletableFuture<Response> future =
                RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context);
        DecodeEndpoint decode = RequestProtocolTestSupport.decodeEndpoint();
        DecodeEndpoint.RequestRegistration reservation =
                new DecodeEndpoint.RequestRegistration(1L, String.valueOf(requestId));
        context.setFuture(future);
        var selectedPrefill = new org.flexlb.worker.status.ServerStatus();
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

    private RequestContext context(long requestId) {
        return RequestProtocolTestSupport.context(config, requestId);
    }
}
