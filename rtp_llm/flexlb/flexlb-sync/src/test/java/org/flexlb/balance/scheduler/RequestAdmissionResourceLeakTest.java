package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.scheduler.RequestContext.AdmissionHandle;
import org.flexlb.balance.scheduler.RequestProtocolTestSupport.Registered;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.service.RecentCacheKeyTraceReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.flexlb.balance.scheduler.SchedulingTestConfig.freezeInputs;
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

/** Real endpoint reservations remain exact across competing local terminal paths. */
class RequestAdmissionResourceLeakTest {
    private FlexlbConfig config;
    private AbstractRequestScheduler lifecycle;

    @BeforeEach
    void setUp() {
        config = SchedulingTestConfig.batchConfig();
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        lifecycle = org.flexlb.balance.scheduler.SchedulerTestSupport.create(service, mock(DeliveryMetricsReporter.class),
                mock(RequestSchedulerReporter.class),
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
    void declinedPublicationLeavesNoCanonicalItem() {
        Registered registered = registerItem(1L);
        try (AdmissionHandle admission = lifecycle.claimAdmissionHandle(1L, registered.future()); var admissionCompletion1 = RequestProtocolTestSupport.finishOnExit(admission)) {
            assertNotNull(admission);
            assertEquals(PlacementResult.Status.BLOCKED,
                    lifecycle.commitRoute(registered.item(), RequestProtocolTestSupport.publication(() -> false)));
            assertNull(activeRoute(1L));
            assertTrue(lifecycle.isAdmissionOpen(1L, registered.future()));
            verify(registered.item().decodeEp(), never()).release(any(), eq(DecodeResources.ReleaseReason.COUNTERPART_FINISHED));
        }
    }

    @Test
    void throwingPublicationLeavesTheExactRegistrationRetryable() {
        Registered registered = registerItem(2L);
        try (AdmissionHandle admission = lifecycle.claimAdmissionHandle(2L, registered.future()); var admissionCompletion2 = RequestProtocolTestSupport.finishOnExit(admission)) {
            assertNotNull(admission);
            assertThrows(IllegalStateException.class, () -> lifecycle.commitRoute(
                    registered.item(), RequestProtocolTestSupport.publication(() -> { throw new IllegalStateException("publication failed"); })));
            assertNull(activeRoute(2L));
            assertTrue(lifecycle.isAdmissionOpen(2L, registered.future()));
        }
    }

    @Test
    void duplicatePublicationDoesNotReleaseCanonicalDecodeReservation() {
        Registered registered = registerItem(3L);
        RequestProtocolTestSupport.bindRoute(lifecycle, registered);
        assertEquals(PlacementResult.Status.CLOSED,
                lifecycle.commitRoute(registered.item(), RequestProtocolTestSupport.publication(() -> true)));
        assertSame(registered.item(), activeRoute(3L));
        verify(registered.item().decodeEp(), never()).release(any(), eq(DecodeResources.ReleaseReason.COUNTERPART_FINISHED));
        lifecycle.cancel(3L, 0L, CancelReason.CLIENT_CANCELLED);
        assertEquals(StrategyErrorType.REQUEST_CANCELLED.getErrorCode(), registered.future().join().getCode());
        verify(registered.item().decodeEp(), times(1)).release(
                registered.item().decodeReservation(),
                DecodeResources.ReleaseReason.COUNTERPART_FINISHED);
    }

    @Test
    void cancellationWaitsForPublicationMutationBeforeReleasingReservation() {
        Registered registered = registerItem(4L);
        AdmissionHandle admission = lifecycle.claimAdmissionHandle(4L, registered.future());
        assertNotNull(admission);
        try {
            assertEquals(PlacementResult.Status.SUCCESS,
                    lifecycle.commitRoute(registered.item(), RequestProtocolTestSupport.publication(() -> true)));
            assertEquals(RequestState.Phase.CANCEL_REQUESTED,
                    lifecycle.cancel(4L, 0L, CancelReason.CLIENT_CANCELLED).state());
            verify(registered.item().decodeEp(), never()).release(any(), eq(DecodeResources.ReleaseReason.COUNTERPART_FINISHED));
        } finally {
            admission.finish();
        }
        registered.future().join();
        verify(registered.item().decodeEp(), times(1)).release(
                registered.item().decodeReservation(),
                DecodeResources.ReleaseReason.COUNTERPART_FINISHED);
    }

    @Test
    void competingLocalTerminalPathsReleaseTheExactReservationOnce() throws Exception {
        try (var executor = Executors.newFixedThreadPool(2)) {
            for (long id = 10; id < 42; id++) {
                Registered registered = registerItem(id);
                RequestProtocolTestSupport.bindRoute(lifecycle, registered);
                CountDownLatch start = new CountDownLatch(1);
                long requestId = id;
                var canceled = executor.submit(() -> {
                    RequestProtocolTestSupport.await(start);
                    lifecycle.cancel(requestId, 0L, CancelReason.CLIENT_CANCELLED);
                });
                var completed = executor.submit(() -> {
                    RequestProtocolTestSupport.await(start);
                    registered.future().complete(Response.error(StrategyErrorType.INVALID_REQUEST));
                });
                start.countDown();
                canceled.get(5, TimeUnit.SECONDS);
                completed.get(5, TimeUnit.SECONDS);
                verify(registered.item().decodeEp(), timeout(1000).times(1))
                        .release(registered.item().decodeReservation(), DecodeResources.ReleaseReason.COUNTERPART_FINISHED);
            }
        }
    }

    @Test
    void shutdownReleasesQueuedReservationsOnce() {
        Registered registered = registerItem(51L);
        RequestProtocolTestSupport.bindRoute(lifecycle, registered);
        assertTrue(RequestProtocolTestSupport.closeAdmissionAndAwaitMutations(lifecycle));
        lifecycle.closeOutstandingAndTerminalize();
        verify(registered.item().decodeEp(), times(1)).release(
                registered.item().decodeReservation(),
                DecodeResources.ReleaseReason.COUNTERPART_FINISHED);
        lifecycle.closeOutstandingAndTerminalize();
        verify(registered.item().decodeEp(), times(1)).release(
                registered.item().decodeReservation(),
                DecodeResources.ReleaseReason.COUNTERPART_FINISHED);
        org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(lifecycle).timer().close();
        org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(lifecycle).closeRequestExecutors();
    }

    @Test
    void queuedPrefillPreemptionReturnsPreemptedAndReleasesDecodeOnce() throws Exception {
        Registered victim = registerItem(61L);
        Registered incoming = registerItem(62L);
        RequestProtocolTestSupport.bindRoute(lifecycle, victim);
        lifecycle.onQueuedItemPreempted(victim.item(), incoming.item());
        Response response = victim.future().get(5, TimeUnit.SECONDS);
        assertEquals(StrategyErrorType.PRIORITY_PREEMPTED.getErrorCode(), response.getCode());
        assertEquals("preempted by higher-priority request 62", response.getErrorMessage());
        lifecycle.onQueuedItemPreempted(victim.item(), incoming.item());
        lifecycle.cancel(61L, 0L, CancelReason.CLIENT_CANCELLED);
        verify(victim.item().decodeEp(), times(1)).release(
                victim.item().decodeReservation(),
                DecodeResources.ReleaseReason.COUNTERPART_FINISHED);
        assertTrue(lifecycle.isAdmissionOpen(62L, incoming.future()));
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.MethodSource("publicationCancellationCases")
    void cancellationAtEveryPublicationOutcomeSettlesTheOriginalRequest(
            String publicationOutcome, CancelReason reason, boolean inactivityExpired) throws Exception {
        Registered registered = registerItem(71L);
        var context = registered.item().ctx();
        AdmissionHandle admission = lifecycle.claimAdmissionHandle(71L, registered.future());
        assertNotNull(admission);
        try {
            var publication = RequestProtocolTestSupport.publication(() -> {
                org.junit.jupiter.api.Assertions.assertFalse(Thread.holdsLock(context));
                if (inactivityExpired) {
                    RequestProtocolTestSupport.expireInactiveRequest(lifecycle, context, Long.MAX_VALUE);
                } else {
                    lifecycle.cancel(71L, 0L, reason);
                }
                org.junit.jupiter.api.Assertions.assertFalse(registered.future().isDone(), "publication owner must settle before terminalization");
                verify(registered.item().decodeEp(), never()).release(any(), any());
                return switch (publicationOutcome) {
                    case "SUCCESS" -> true;
                    case "BLOCKED" -> false;
                    case "THROW" -> throw new IllegalStateException("publication failed");
                    default -> throw new AssertionError(publicationOutcome);
                };
            });
            if (publicationOutcome.equals("THROW")) {
                assertThrows(IllegalStateException.class, () -> lifecycle.commitRoute(registered.item(), publication));
            } else {
                assertEquals(publicationOutcome.equals("SUCCESS") ? PlacementResult.Status.SUCCESS : PlacementResult.Status.BLOCKED,
                        lifecycle.commitRoute(registered.item(), publication));
            }
        } finally {
            admission.finish();
        }
        var response = registered.future().get(5, TimeUnit.SECONDS);
        org.junit.jupiter.api.Assertions.assertFalse(response.isSuccess());
        assertEquals(reason == CancelReason.DEADLINE_EXCEEDED ? StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode()
                : StrategyErrorType.REQUEST_CANCELLED.getErrorCode(), response.getCode());
        lifecycle.runtime.continuations().awaitIdle();
        assertEquals(0, lifecycle.requests.liveRequestCount());
        org.junit.jupiter.api.Assertions.assertFalse(lifecycle.isAdmissionOpen(71L, registered.future()));
        // A scheduling deadline is canceled by the published Prefill queue owner;
        // an actual inactivity expiry retains separate EXPIRED cleanup proof.
        verify(registered.item().decodeEp(), times(publicationOutcome.equals("SUCCESS") ? 1 : 0)).release(
                registered.item().decodeReservation(), inactivityExpired
                        ? DecodeResources.ReleaseReason.EXPIRED : DecodeResources.ReleaseReason.COUNTERPART_FINISHED);
        lifecycle.cancel(71L, 0L, CancelReason.SHUTDOWN);
        assertSame(response, registered.future().join());
    }

    static java.util.stream.Stream<org.junit.jupiter.params.provider.Arguments> publicationCancellationCases() {
        return java.util.stream.Stream.of("SUCCESS", "BLOCKED", "THROW").flatMap(outcome ->
                java.util.stream.Stream.concat(
                        java.util.stream.Stream.of(CancelReason.CLIENT_CANCELLED, CancelReason.SHUTDOWN, CancelReason.DEADLINE_EXCEEDED)
                                .map(reason -> org.junit.jupiter.params.provider.Arguments.of(outcome, reason, false)),
                        java.util.stream.Stream.of(org.junit.jupiter.params.provider.Arguments.of(
                                outcome, CancelReason.DEADLINE_EXCEEDED, true))));
    }

    private RequestRoute activeRoute(long requestId) {
        RequestContext requestContext = lifecycle.findRequestContext(requestId);
        synchronized (requestContext) {
            return requestContext.activeRoute();
        }
    }

    private Registered registerItem(long requestId) {
        RequestContext context = RequestProtocolTestSupport.context(config, requestId);
        var future = RequestProtocolTestSupport.register(lifecycle, context);
        DecodeEndpoint decode = RequestProtocolTestSupport.decodeEndpoint();
        var reservation = new DecodeResources.ReservationHandle(1L, requestId, 1L);
        context.setFuture(future);
        return new Registered(org.flexlb.balance.scheduler.SchedulingTestConfig.createRoute(freezeInputs(context), new Response(), null, null,
                null, decode, reservation, System.currentTimeMillis()), future);
    }
}
