package org.flexlb.balance.schedule;

import org.flexlb.balance.metrics.DeliveryMetricsReporter;
import org.flexlb.balance.metrics.RequestSchedulerReporter;
import org.flexlb.balance.requestlog.TheoryCacheHitReporter;
import org.flexlb.balance.schedule.RequestContext.SelectionAttempt;
import org.flexlb.balance.schedule.RequestProtocolTestSupport.Registered;
import org.flexlb.balance.strategy.RoutingResult;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.worker.endpoint.DecodeEndpoint;
import org.flexlb.worker.endpoint.Response;
import org.flexlb.worker.endpoint.StrategyErrorType;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.timeout;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Real endpoint reservations remain exact across competing local terminal paths.
 */
class RequestAdmissionResourceLeakTest {
    private FlexlbConfig config;
    private AbstractRequestScheduler lifecycle;

    @BeforeEach
    void setUp() {
        config = SchedulingTestConfig.newConfig();
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        lifecycle =
                org.flexlb.balance.schedule.SchedulerTestSupport.create(
                        service,
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
    void declinedPublicationLeavesNoCanonicalItem() {
        Registered registered = registerItem(1L);
        SelectionAttempt admission =
                SchedulerTestSupport.claim(lifecycle, "1", registered.future());
        try (var admissionCompletion1 = RequestProtocolTestSupport.finishOnExit(admission)) {
            assertNotNull(admission);
            assertEquals(
                    RoutingResult.Status.BLOCKED,
                    RequestProtocolTestSupport.commitRoute(
                            lifecycle,
                            registered.item(),
                            RequestProtocolTestSupport.publication(() -> false)));
            assertNull(activeRoute(1L));
            assertTrue(lifecycle.isAdmissionOpen(registered.item().ctx()));
            verify(registered.item().decodeEndpoint())
                    .rollbackSelection(registered.item().decodeRegistration());
        }
    }

    @Test
    void throwingPublicationLeavesTheExactRegistrationRetryable() {
        Registered registered = registerItem(2L);
        SelectionAttempt admission =
                SchedulerTestSupport.claim(lifecycle, "2", registered.future());
        try (var admissionCompletion2 = RequestProtocolTestSupport.finishOnExit(admission)) {
            assertNotNull(admission);
            assertThrows(
                    IllegalStateException.class,
                    () ->
                            RequestProtocolTestSupport.commitRoute(
                                    lifecycle,
                                    registered.item(),
                                    RequestProtocolTestSupport.publication(
                                            () -> {
                                                throw new IllegalStateException(
                                                        "publication failed");
                                            })));
            assertNull(activeRoute(2L));
            assertTrue(lifecycle.isAdmissionOpen(registered.item().ctx()));
        }
    }

    @Test
    void duplicatePublicationDoesNotReleaseCanonicalDecodeReservation() {
        Registered registered = registerItem(3L);
        RequestProtocolTestSupport.beginRouteResult(lifecycle, registered);
        registered
                .item()
                .ctx()
                .recordEvent(
                        new RequestContext.RequestEvent(
                                registered.item(),
                                null,
                                null,
                                null,
                                org.flexlb.worker.endpoint.RequestEventType.REMOTE_CLEANED,
                                0L,
                                null,
                                System.currentTimeMillis()));
        assertEquals(
                RoutingResult.Status.CLOSED,
                RequestProtocolTestSupport.commitRoute(
                        lifecycle,
                        registered.item(),
                        RequestProtocolTestSupport.publication(() -> true)));
        assertSame(registered.item(), activeRoute(3L));
        verify(registered.item().decodeEndpoint(), never()).releaseRequest(any());
        lifecycle.cancel("3", CancelReason.CLIENT_CANCELLED);
        assertTrue(registered.future().join().isSuccess());
        SchedulerTestSupport.awaitRequestWork(
                org.flexlb.balance.schedule.SchedulerTestSupport.runtime(lifecycle));
        verify(registered.item().decodeEndpoint(), times(1))
                .releaseRequest(registered.item().decodeRegistration());
    }

    @Test
    void cancellationWaitsForPublicationMutationBeforeReleasingReservation() {
        Registered registered = registerItem(4L);
        SelectionAttempt admission =
                SchedulerTestSupport.claim(lifecycle, "4", registered.future());
        assertNotNull(admission);
        try {
            registered.item().ctx().registerSelectedRoute(admission, registered.item());
            assertEquals(
                    RequestResult.State.CANCEL_REQUESTED,
                    lifecycle.cancel("4", CancelReason.CLIENT_CANCELLED).state());
            verify(registered.item().decodeEndpoint(), never()).releaseRequest(any());
        } finally {
            admission
                    .requestContext()
                    .getRequestScheduler()
                    .finishSelectionWithoutCommit(admission, null);
        }
        registered.future().join();
        SchedulerTestSupport.awaitRequestWork(
                org.flexlb.balance.schedule.SchedulerTestSupport.runtime(lifecycle));
        verify(registered.item().decodeEndpoint(), times(1))
                .releaseRequest(registered.item().decodeRegistration());
    }

    @Test
    void competingLocalTerminalPathsReleaseTheExactReservationOnce() throws Exception {
        try (var executor = Executors.newFixedThreadPool(2)) {
            for (long id = 10; id < 42; id++) {
                Registered registered = registerItem(id);
                RequestProtocolTestSupport.beginRouteResult(lifecycle, registered);
                registered
                        .item()
                        .ctx()
                        .recordEvent(
                                new RequestContext.RequestEvent(
                                        registered.item(),
                                        null,
                                        null,
                                        null,
                                        org.flexlb.worker.endpoint.RequestEventType.REMOTE_CLEANED,
                                        0L,
                                        null,
                                        System.currentTimeMillis()));
                CountDownLatch start = new CountDownLatch(1);
                long requestId = id;
                var canceled =
                        executor.submit(
                                () -> {
                                    RequestProtocolTestSupport.await(start);
                                    lifecycle.cancel(
                                            String.valueOf(requestId),
                                            CancelReason.CLIENT_CANCELLED);
                                });
                var completed =
                        executor.submit(
                                () -> {
                                    RequestProtocolTestSupport.await(start);
                                    registered
                                            .future()
                                            .complete(
                                                    Response.error(
                                                            StrategyErrorType.INVALID_REQUEST));
                                });
                start.countDown();
                canceled.get(5, TimeUnit.SECONDS);
                completed.get(5, TimeUnit.SECONDS);
                verify(registered.item().decodeEndpoint(), timeout(1000).times(1))
                        .releaseRequest(registered.item().decodeRegistration());
            }
        }
    }

    @Test
    void shutdownReleasesQueuedReservationsOnce() {
        Registered registered = registerItem(51L);
        RequestProtocolTestSupport.beginRouteResult(lifecycle, registered);
        registered
                .item()
                .ctx()
                .recordEvent(
                        new RequestContext.RequestEvent(
                                registered.item(),
                                null,
                                null,
                                null,
                                org.flexlb.worker.endpoint.RequestEventType.REMOTE_CLEANED,
                                0L,
                                null,
                                System.currentTimeMillis()));
        assertTrue(RequestProtocolTestSupport.closeRegistrationAndAwaitScheduling(lifecycle));
        lifecycle.closeOutstandingAndTerminalize();
        verify(registered.item().decodeEndpoint(), times(1))
                .releaseRequest(registered.item().decodeRegistration());
        lifecycle.closeOutstandingAndTerminalize();
        verify(registered.item().decodeEndpoint(), times(1))
                .releaseRequest(registered.item().decodeRegistration());
        SchedulerTestSupport.timer(
                        org.flexlb.balance.schedule.SchedulerTestSupport.runtime(lifecycle))
                .close();
        SchedulerTestSupport.closeRequestExecutors(
                org.flexlb.balance.schedule.SchedulerTestSupport.runtime(lifecycle));
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.MethodSource("publicationCancellationCases")
    void cancellationAtEveryPublicationOutcomeSettlesTheOriginalRequest(
            String publicationOutcome, CancelReason reason, boolean inactivityExpired)
            throws Exception {
        Registered registered = registerItem(71L);
        var context = registered.item().ctx();
        SelectionAttempt admission =
                SchedulerTestSupport.claim(lifecycle, "71", registered.future());
        assertNotNull(admission);
        try {
            var publication =
                    RequestProtocolTestSupport.publication(
                            () -> {
                                org.junit.jupiter.api.Assertions.assertFalse(
                                        Thread.holdsLock(context));
                                if (inactivityExpired) {
                                    RequestProtocolTestSupport.expireInactiveRequest(
                                            lifecycle, context, Long.MAX_VALUE);
                                } else {
                                    lifecycle.cancel("71", reason);
                                }
                                org.junit.jupiter.api.Assertions.assertFalse(
                                        registered.future().isDone(),
                                        "publication owner must settle before terminalization");
                                verify(registered.item().decodeEndpoint(), never())
                                        .releaseRequest(any());
                                return switch (publicationOutcome) {
                                    case "SUCCESS" -> true;
                                    case "BLOCKED" -> false;
                                    case "THROW" ->
                                            throw new IllegalStateException("publication failed");
                                    default -> throw new AssertionError(publicationOutcome);
                                };
                            });
            if (publicationOutcome.equals("THROW")) {
                assertThrows(
                        IllegalStateException.class,
                        () ->
                                RequestProtocolTestSupport.commitRoute(
                                        lifecycle, registered.item(), publication));
            } else {
                assertEquals(
                        RoutingResult.Status.CLOSED,
                        RequestProtocolTestSupport.commitRoute(
                                lifecycle, registered.item(), publication));
            }
        } finally {
            admission
                    .requestContext()
                    .getRequestScheduler()
                    .finishSelectionWithoutCommit(admission, null);
        }
        var response = registered.future().get(5, TimeUnit.SECONDS);
        org.junit.jupiter.api.Assertions.assertFalse(response.isSuccess());
        assertEquals(
                reason == CancelReason.DEADLINE_EXCEEDED
                        ? StrategyErrorType.QUEUE_TIMEOUT.getErrorCode()
                        : StrategyErrorType.REQUEST_CANCELLED.getErrorCode(),
                response.getCode());
        SchedulerTestSupport.awaitRequestWork(
                org.flexlb.balance.schedule.SchedulerTestSupport.runtime(lifecycle));
        assertEquals(0, lifecycle.repository.liveRequestCount());
        org.junit.jupiter.api.Assertions.assertFalse(
                lifecycle.isAdmissionOpen(registered.item().ctx()));
        verify(registered.item().decodeEndpoint())
                .rollbackSelection(registered.item().decodeRegistration());
        lifecycle.cancel("71", CancelReason.SHUTDOWN);
        assertSame(response, registered.future().join());
    }

    static java.util.stream.Stream<org.junit.jupiter.params.provider.Arguments>
            publicationCancellationCases() {
        return java.util.stream.Stream.of("SUCCESS", "BLOCKED", "THROW")
                .flatMap(
                        outcome ->
                                java.util.stream.Stream.concat(
                                        java.util.stream.Stream.of(
                                                        CancelReason.CLIENT_CANCELLED,
                                                        CancelReason.SHUTDOWN,
                                                        CancelReason.DEADLINE_EXCEEDED)
                                                .map(
                                                        reason ->
                                                                org.junit.jupiter.params.provider
                                                                        .Arguments.of(
                                                                        outcome, reason, false)),
                                        java.util.stream.Stream.of(
                                                org.junit.jupiter.params.provider.Arguments.of(
                                                        outcome,
                                                        CancelReason.DEADLINE_EXCEEDED,
                                                        true))));
    }

    private RequestResult activeRoute(long requestId) {
        RequestContext requestContext = lifecycle.repository.findActive(String.valueOf(requestId));
        synchronized (requestContext) {
            return requestContext.snapshot().route();
        }
    }

    private Registered registerItem(long requestId) {
        RequestContext context = RequestProtocolTestSupport.context(config, requestId);
        var future = RequestProtocolTestSupport.submitForManualPlacement(lifecycle, context);
        DecodeEndpoint decode = RequestProtocolTestSupport.decodeEndpoint();
        var reservation = new DecodeEndpoint.RequestRegistration(1L, String.valueOf(requestId));
        context.setFuture(future);
        return new Registered(
                org.flexlb.balance.schedule.SchedulingTestConfig.createRoute(
                        context,
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
