package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.scheduler.RequestProtocolTestSupport.Registered;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.service.TheoryCacheHitReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
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
import static org.mockito.ArgumentMatchers.eq;
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
        config = SchedulingTestConfig.batchConfig();
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        lifecycle =
                org.flexlb.balance.scheduler.SchedulerTestSupport.create(
                        service,
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
    void declinedPublicationLeavesNoCanonicalItem() {
        Registered registered = registerItem("1");
        try (AdmissionHandle admission = lifecycle.claimAdmissionHandle("1", registered.future());
                var admissionCompletion1 = RequestProtocolTestSupport.finishOnExit(admission)) {
            assertNotNull(admission);
            assertEquals(
                    PlacementResult.Status.BLOCKED,
                    lifecycle.commitRoute(
                            registered.item(),
                            RequestProtocolTestSupport.publication(() -> false)));
            rollbackUnpublished(admission, registered);
            assertNull(activeRoute("1"));
            assertTrue(lifecycle.isAdmissionOpen("1", registered.future()));
            verify(registered.item().decodeEp(), never())
                    .release(any(), eq(DecodeResources.ReleaseReason.COUNTERPART_FINISHED));
        }
    }

    @Test
    void throwingPublicationLeavesTheExactRegistrationRetryable() {
        Registered registered = registerItem("2");
        try (AdmissionHandle admission = lifecycle.claimAdmissionHandle("2", registered.future());
                var admissionCompletion2 = RequestProtocolTestSupport.finishOnExit(admission)) {
            assertNotNull(admission);
            assertThrows(
                    IllegalStateException.class,
                    () ->
                            lifecycle.commitRoute(
                                    registered.item(),
                                    RequestProtocolTestSupport.publication(
                                            () -> {
                                                throw new IllegalStateException(
                                                        "publication failed");
                                            })));
            rollbackUnpublished(admission, registered);
            assertNull(activeRoute("2"));
            assertTrue(lifecycle.isAdmissionOpen("2", registered.future()));
        }
    }

    @Test
    void duplicatePublicationDoesNotReleaseCanonicalDecodeReservation() {
        Registered registered = registerItem("3");
        RequestProtocolTestSupport.beginRoutePublication(lifecycle, registered);
        assertEquals(
                PlacementResult.Status.CLOSED,
                lifecycle.commitRoute(
                        registered.item(), RequestProtocolTestSupport.publication(() -> true)));
        assertSame(registered.item(), activeRoute("3"));
        verify(registered.item().decodeEp(), never())
                .release(any(), eq(DecodeResources.ReleaseReason.COUNTERPART_FINISHED));
        lifecycle.cancel("3", 0L, CancelReason.CLIENT_CANCELLED);
        assertEquals(
                StrategyErrorType.REQUEST_CANCELLED.getErrorCode(),
                registered.future().join().getCode());
        SchedulerTestSupport.awaitRequestWork(lifecycle.runtime);
        verify(registered.item().decodeEp(), times(1))
                .release(
                        registered.item().decodeReservation(),
                        DecodeResources.ReleaseReason.COUNTERPART_FINISHED);
    }

    @Test
    void cancellationWaitsForPublicationMutationBeforeReleasingReservation() {
        Registered registered = registerItem("4");
        AdmissionHandle admission = lifecycle.claimAdmissionHandle("4", registered.future());
        assertNotNull(admission);
        try {
            assertEquals(
                    PlacementResult.Status.SUCCESS,
                    lifecycle.commitRoute(
                            registered.item(), RequestProtocolTestSupport.publication(() -> true)));
            assertEquals(
                    RequestState.Phase.CANCEL_REQUESTED,
                    lifecycle.cancel("4", 0L, CancelReason.CLIENT_CANCELLED).state());
            verify(registered.item().decodeEp(), never())
                    .release(any(), eq(DecodeResources.ReleaseReason.COUNTERPART_FINISHED));
        } finally {
            admission.finish();
        }
        registered.future().join();
        SchedulerTestSupport.awaitRequestWork(lifecycle.runtime);
        verify(registered.item().decodeEp(), times(1))
                .release(
                        registered.item().decodeReservation(),
                        DecodeResources.ReleaseReason.COUNTERPART_FINISHED);
    }

    @Test
    void competingLocalTerminalPathsReleaseTheExactReservationOnce() throws Exception {
        try (var executor = Executors.newFixedThreadPool(2)) {
            for (long id = 10; id < 42; id++) {
                Registered registered = registerItem(String.valueOf(id));
                RequestProtocolTestSupport.beginRoutePublication(lifecycle, registered);
                CountDownLatch start = new CountDownLatch(1);
                long requestId = id;
                var canceled =
                        executor.submit(
                                () -> {
                                    RequestProtocolTestSupport.await(start);
                                    lifecycle.cancel(
                                            String.valueOf(requestId),
                                            0L,
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
                verify(registered.item().decodeEp(), timeout(1000).times(1))
                        .release(
                                registered.item().decodeReservation(),
                                DecodeResources.ReleaseReason.COUNTERPART_FINISHED);
            }
        }
    }

    @Test
    void shutdownReleasesQueuedReservationsOnce() {
        Registered registered = registerItem("51");
        RequestProtocolTestSupport.beginRoutePublication(lifecycle, registered);
        assertTrue(RequestProtocolTestSupport.closeAdmissionAndAwaitMutations(lifecycle));
        lifecycle.closeOutstandingAndTerminalize();
        verify(registered.item().decodeEp(), times(1))
                .release(
                        registered.item().decodeReservation(),
                        DecodeResources.ReleaseReason.COUNTERPART_FINISHED);
        lifecycle.closeOutstandingAndTerminalize();
        verify(registered.item().decodeEp(), times(1))
                .release(
                        registered.item().decodeReservation(),
                        DecodeResources.ReleaseReason.COUNTERPART_FINISHED);
        SchedulerTestSupport.timer(
                        org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(lifecycle))
                .close();
        SchedulerTestSupport.closeRequestExecutors(
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(lifecycle));
    }

    @Test
    void queuedPrefillPreemptionReturnsPreemptedAndReleasesDecodeOnce() throws Exception {
        Registered victim = registerItem("61");
        Registered incoming = registerItem("62");
        RequestProtocolTestSupport.beginRoutePublication(lifecycle, victim);
        lifecycle.onQueueEvent(
                victim.item(), AbstractRequestScheduler.QueueKind.PREEMPTED, incoming.item(), null);
        Response response = victim.future().get(5, TimeUnit.SECONDS);
        assertEquals(StrategyErrorType.PRIORITY_PREEMPTED.getErrorCode(), response.getCode());
        assertEquals("preempted by higher-priority request 62", response.getErrorMessage());
        lifecycle.onQueueEvent(
                victim.item(), AbstractRequestScheduler.QueueKind.PREEMPTED, incoming.item(), null);
        lifecycle.cancel("61", 0L, CancelReason.CLIENT_CANCELLED);
        verify(victim.item().decodeEp(), times(1))
                .release(
                        victim.item().decodeReservation(),
                        DecodeResources.ReleaseReason.COUNTERPART_FINISHED);
        assertTrue(lifecycle.isAdmissionOpen("62", incoming.future()));
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.MethodSource("publicationCancellationCases")
    void cancellationAtEveryPublicationOutcomeSettlesTheOriginalRequest(
            String publicationOutcome, CancelReason reason, boolean inactivityExpired)
            throws Exception {
        Registered registered = registerItem("71");
        var context = registered.item().ctx();
        AdmissionHandle admission = lifecycle.claimAdmissionHandle("71", registered.future());
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
                                    lifecycle.cancel("71", 0L, reason);
                                }
                                org.junit.jupiter.api.Assertions.assertFalse(
                                        registered.future().isDone(),
                                        "publication owner must settle before terminalization");
                                verify(registered.item().decodeEp(), never()).release(any(), any());
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
                        () -> lifecycle.commitRoute(registered.item(), publication));
            } else {
                assertEquals(
                        publicationOutcome.equals("SUCCESS")
                                ? PlacementResult.Status.SUCCESS
                                : PlacementResult.Status.BLOCKED,
                        lifecycle.commitRoute(registered.item(), publication));
            }
        } finally {
            if (!publicationOutcome.equals("SUCCESS")) {
                rollbackUnpublished(admission, registered);
            }
            admission.finish();
        }
        var response = registered.future().get(5, TimeUnit.SECONDS);
        org.junit.jupiter.api.Assertions.assertFalse(response.isSuccess());
        assertEquals(
                reason == CancelReason.DEADLINE_EXCEEDED
                        ? StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode()
                        : StrategyErrorType.REQUEST_CANCELLED.getErrorCode(),
                response.getCode());
        SchedulerTestSupport.awaitRequestWork(lifecycle.runtime);
        assertEquals(0, lifecycle.requests.liveRequestCount());
        org.junit.jupiter.api.Assertions.assertFalse(
                lifecycle.isAdmissionOpen("71", registered.future()));
        // A scheduling deadline is canceled by the published Prefill queue owner;
        // an actual inactivity expiry retains separate EXPIRED cleanup proof.
        verify(registered.item().decodeEp(), times(publicationOutcome.equals("SUCCESS") ? 1 : 0))
                .release(
                        registered.item().decodeReservation(),
                        inactivityExpired
                                ? DecodeResources.ReleaseReason.EXPIRED
                                : DecodeResources.ReleaseReason.COUNTERPART_FINISHED);
        lifecycle.cancel("71", 0L, CancelReason.SHUTDOWN);
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

    private void rollbackUnpublished(AdmissionHandle admission, Registered registered) {
        var released =
                registered
                        .item()
                        .decodeEp()
                        .release(
                                registered.item().decodeReservation(),
                                DecodeResources.ReleaseReason.LOCAL_ROLLBACK)
                        .status();
        assertTrue(
                released == DecodeResources.ReservationReleaseResult.RELEASED
                        || released == DecodeResources.ReservationReleaseResult.STALE);
        registered
                .item()
                .ctx()
                .bindPlacement(
                        admission.token,
                        registered.item(),
                        ContextResult.PublicationResult.ROLLBACK_COMPLETED);
    }

    private RequestRoute activeRoute(String requestId) {
        RequestContext requestContext = lifecycle.requests.findActive(requestId);
        synchronized (requestContext) {
            return requestContext.snapshot().route();
        }
    }

    private Registered registerItem(String requestId) {
        RequestContext context = RequestProtocolTestSupport.context(config, requestId);
        var future = RequestProtocolTestSupport.register(lifecycle, context);
        DecodeEndpoint decode = RequestProtocolTestSupport.decodeEndpoint();
        var reservation = new DecodeResources.ReservationHandle(1L, requestId, 1L);
        context.data().setFuture(future);
        return new Registered(
                org.flexlb.balance.scheduler.SchedulingTestConfig.createRoute(
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
