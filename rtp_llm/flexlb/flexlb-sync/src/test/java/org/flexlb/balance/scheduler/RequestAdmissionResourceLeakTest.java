package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.scheduler.RequestLifecycleTestSupport.Registered;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
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
    private RequestRegistry lifecycle;

    @BeforeEach
    void setUp() {
        config = SchedulingTestConfig.batchConfig();
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        lifecycle = new RequestRegistry(service, mock(BatchSchedulerReporter.class),
                mock(RequestSchedulerReporter.class));
    }

    @AfterEach
    void tearDown() {
        if (lifecycle.closeAdmissionAndAwaitMutations()) {
            lifecycle.closeOutstandingAndTerminalize();
            lifecycle.closeExpiration();
            lifecycle.closePublisher();
        }
    }

    @Test
    void declinedPublicationLeavesNoCanonicalItem() {
        Registered registered = registerItem(1L);
        try (AdmissionMutation admission = lifecycle.claimAdmissionMutation(1L, registered.future())) {
            assertNotNull(admission);
            assertEquals(PlacementResult.Status.BLOCKED,
                    lifecycle.commitRoute(registered.item(), () -> false));
            assertNull(activeItem(1L));
            assertTrue(lifecycle.isAdmissionOpen(1L, registered.future()));
            verify(registered.item().decodeEp(), never()).releaseReservationExact(any());
        }
    }

    @Test
    void throwingPublicationLeavesTheExactRegistrationRetryable() {
        Registered registered = registerItem(2L);
        try (AdmissionMutation admission = lifecycle.claimAdmissionMutation(2L, registered.future())) {
            assertNotNull(admission);
            assertThrows(IllegalStateException.class, () -> lifecycle.commitRoute(
                    registered.item(), () -> { throw new IllegalStateException("publication failed"); }));
            assertNull(activeItem(2L));
            assertTrue(lifecycle.isAdmissionOpen(2L, registered.future()));
        }
    }

    @Test
    void duplicatePublicationDoesNotReleaseCanonicalDecodeReservation() {
        Registered registered = registerItem(3L);
        RequestLifecycleTestSupport.bindRoute(lifecycle, registered);
        assertEquals(PlacementResult.Status.CLOSED,
                lifecycle.commitRoute(registered.item(), () -> true));
        assertSame(registered.item(), activeItem(3L));
        verify(registered.item().decodeEp(), never()).releaseReservationExact(any());
        lifecycle.cancelRequest(3L, 0L, CancelReason.CLIENT_CANCELLED);
        assertEquals(StrategyErrorType.REQUEST_CANCELLED.getErrorCode(), registered.future().join().getCode());
        verify(registered.item().decodeEp(), times(1)).releaseReservationExact(registered.item().decodeReservation());
    }

    @Test
    void cancellationWaitsForPublicationMutationBeforeReleasingReservation() {
        Registered registered = registerItem(4L);
        AdmissionMutation admission = lifecycle.claimAdmissionMutation(4L, registered.future());
        assertNotNull(admission);
        assertEquals(PlacementResult.Status.SUCCESS,
                lifecycle.commitRoute(registered.item(), () -> true));
        assertEquals(RequestState.Phase.CANCEL_REQUESTED,
                lifecycle.cancelRequest(4L, 0L, CancelReason.CLIENT_CANCELLED).state());
        verify(registered.item().decodeEp(), never()).releaseReservationExact(any());
        admission.close();
        registered.future().join();
        verify(registered.item().decodeEp(), times(1)).releaseReservationExact(registered.item().decodeReservation());
    }

    @Test
    void competingLocalTerminalPathsReleaseTheExactReservationOnce() throws Exception {
        try (var executor = Executors.newFixedThreadPool(2)) {
            for (long id = 10; id < 42; id++) {
                Registered registered = registerItem(id);
                RequestLifecycleTestSupport.bindRoute(lifecycle, registered);
                CountDownLatch start = new CountDownLatch(1);
                long requestId = id;
                var canceled = executor.submit(() -> {
                    RequestLifecycleTestSupport.await(start);
                    lifecycle.cancelRequest(requestId, 0L, CancelReason.CLIENT_CANCELLED);
                });
                var completed = executor.submit(() -> {
                    RequestLifecycleTestSupport.await(start);
                    registered.future().complete(Response.error(StrategyErrorType.INVALID_REQUEST));
                });
                start.countDown();
                canceled.get(5, TimeUnit.SECONDS);
                completed.get(5, TimeUnit.SECONDS);
                verify(registered.item().decodeEp(), timeout(1000).times(1))
                        .releaseReservationExact(registered.item().decodeReservation());
            }
        }
    }

    @Test
    void shutdownReleasesQueuedReservationsOnce() {
        Registered registered = registerItem(51L);
        RequestLifecycleTestSupport.bindRoute(lifecycle, registered);
        assertTrue(lifecycle.closeAdmissionAndAwaitMutations());
        lifecycle.closeOutstandingAndTerminalize();
        verify(registered.item().decodeEp(), times(1)).releaseReservationExact(registered.item().decodeReservation());
        lifecycle.closeOutstandingAndTerminalize();
        verify(registered.item().decodeEp(), times(1)).releaseReservationExact(registered.item().decodeReservation());
        lifecycle.closeExpiration();
        lifecycle.closePublisher();
    }

    @Test
    void queuedPrefillPreemptionReturnsRetryableFailureAndReleasesDecodeOnce() throws Exception {
        Registered victim = registerItem(61L);
        Registered incoming = registerItem(62L);
        RequestLifecycleTestSupport.bindRoute(lifecycle, victim);
        lifecycle.onQueuedItemPreempted(victim.item(), incoming.item());
        assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(),
                victim.future().get(5, TimeUnit.SECONDS).getCode());
        lifecycle.onQueuedItemPreempted(victim.item(), incoming.item());
        lifecycle.cancelRequest(61L, 0L, CancelReason.CLIENT_CANCELLED);
        verify(victim.item().decodeEp(), times(1)).releaseReservationExact(victim.item().decodeReservation());
        assertTrue(lifecycle.isAdmissionOpen(62L, incoming.future()));
    }

    private ScheduledRequest activeItem(long requestId) {
        RequestSlot slot = lifecycle.requestSlot(requestId);
        synchronized (slot) {
            return slot.activeItem();
        }
    }

    private Registered registerItem(long requestId) {
        BalanceContext context = RequestLifecycleTestSupport.context(config, requestId);
        var future = lifecycle.register(context);
        DecodeEndpoint decode = mock(DecodeEndpoint.class);
        var reservation = new DecodeEndpoint.ReservationHandle(1L, requestId, 1L);
        return new Registered(new ScheduledRequest(context, future, new Response(), null, null,
                null, decode, reservation, System.currentTimeMillis()), future);
    }
}
