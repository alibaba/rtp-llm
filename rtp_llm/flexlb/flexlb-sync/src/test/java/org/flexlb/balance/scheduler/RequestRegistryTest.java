package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.scheduler.RequestLifecycleTestSupport.Registered;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import static org.flexlb.balance.scheduler.RequestLifecycleTestSupport.awaitCondition;
import static org.flexlb.balance.scheduler.RequestLifecycleTestSupport.commitRoute;
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

/** Canonical request-generation ownership tests, independent of the facade. */
class RequestRegistryTest {

    private FlexlbConfig config;
    private RequestRegistry lifecycle;

    @BeforeEach
    void setUp() {
        config = SchedulingTestConfig.batchConfig();
        SchedulingTestConfig.usePriorityQueue(config);
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        lifecycle = new RequestRegistry(
                configService,
                mock(BatchSchedulerReporter.class),
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
    void duplicateRegistrationCannotReplaceTheCanonicalExactGeneration() {
        BalanceContext context = context(101L);
        CompletableFuture<Response> canonical = lifecycle.register(context);

        CompletableFuture<Response> duplicate = lifecycle.register(context(101L));

        assertFalse(canonical.isDone());
        assertTrue(duplicate.isDone());
        assertEquals(StrategyErrorType.INVALID_REQUEST.getErrorCode(),
                duplicate.join().getCode());
        assertSame(canonical, lifecycle.requestSlot(101L).future());
        assertEquals(1, lifecycle.liveRequestCount());
    }

    @Test
    void globalWaitingRequestsHaveNoQuantityAdmissionLimit() {
        var low = context(1L);
        low.setSchedulingMetadata(SchedulingMetadata.explicit(10, Long.MAX_VALUE));
        var waiting = lifecycle.register(low);
        for (long id = 2; id <= 1001; id++) {
            var high = context(id);
            high.setSchedulingMetadata(SchedulingMetadata.explicit(90, Long.MAX_VALUE));
            assertFalse(lifecycle.register(high).isDone());
        }
        assertFalse(waiting.isDone(), "higher priority arrivals must not evict waiting requests");
        assertEquals(1001, lifecycle.liveRequestCount());
        assertEquals(StrategyErrorType.INVALID_REQUEST.getErrorCode(),
                lifecycle.register(context(1L)).join().getCode());
        for (long id = 1; id <= 1001; id++) {
            lifecycle.cancelRequest(id, 0L, CancelReason.CLIENT_CANCELLED);
        }
        assertEquals(0, lifecycle.liveRequestCount());
    }

    @Test
    void slotLockContractIsEnforcedWithoutJvmAssertions() {
        lifecycle.register(context(102L));
        RequestSlot slot = lifecycle.requestSlot(102L);

        IllegalStateException failure = assertThrows(
                IllegalStateException.class, slot::activeItem);

        assertTrue(failure.getMessage().contains("requires slot lock"));
    }

    @Test
    void deliveredRequestsHaveNoExtraGlobalQuantityGate() {
        for (long id = 201; id <= 401; id++) {
            Registered registered = registerItem(id);
            assertEquals(PlacementResult.Status.SUCCESS,
                    commitRoute(lifecycle, registered));
            assertNotNull(RequestLifecycleTestSupport.claimRoute(
                    lifecycle, registered.item(), () -> true));
        }
        assertEquals(201, lifecycle.liveRequestCount());
    }

    @Test
    void admissionMutationDefersCancellationUntilItsExactCapabilityCloses() {
        CompletableFuture<Response> future = lifecycle.register(context(301L));
        AdmissionMutation scope =
                lifecycle.claimAdmissionMutation(301L, future);
        assertNotNull(scope);

        RequestState requested = lifecycle.cancelRequest(
                301L, 0L, CancelReason.CLIENT_CANCELLED);

        assertEquals(RequestState.Phase.CANCEL_REQUESTED, requested.state());
        assertFalse(future.isDone(),
                "the admission mutation still owns rollback and terminal cleanup");

        scope.close();

        assertEquals(StrategyErrorType.REQUEST_CANCELLED.getErrorCode(),
                future.join().getCode());
        assertEquals(RequestState.Phase.CANCELLED,
                lifecycle.getRequestState(301L, 0L).state());
    }

    @Test
    void queueDecisionResponsePublishesOutsideTheDecisionCaller() throws Exception {
        CompletableFuture<Response> future = lifecycle.register(context(302L));
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
    void shutdownGateWaitsForTheExactAdmissionMutationAndRejectsNewWork()
            throws Exception {
        CompletableFuture<Response> heldFuture =
                lifecycle.register(context(401L));
        AdmissionMutation held =
                lifecycle.claimAdmissionMutation(401L, heldFuture);
        assertNotNull(held);
        ExecutorService executor = Executors.newSingleThreadExecutor();
        try {
            Future<Boolean> shutdownOwner =
                    executor.submit(lifecycle::closeAdmissionAndAwaitMutations);
            awaitCondition(lifecycle::isShuttingDown);
            assertFalse(shutdownOwner.isDone(),
                    "shutdown must not overtake an exact admission mutation");

            CompletableFuture<Response> rejected =
                    lifecycle.register(context(402L));
            assertEquals(StrategyErrorType.DISPATCH_FAILED.getErrorCode(),
                    rejected.join().getCode());

            held.close();
            assertTrue(shutdownOwner.get(5, TimeUnit.SECONDS));
            lifecycle.closeOutstandingAndTerminalize();
            lifecycle.closeExpiration();
            lifecycle.closePublisher();
            assertEquals(StrategyErrorType.DISPATCH_FAILED.getErrorCode(),
                    heldFuture.get(5, TimeUnit.SECONDS).getCode());
        } finally {
            executor.shutdownNow();
        }
    }

    @Test
    void cancelRequiresTheExpectedBatchGenerationAndUnknownIdsStayAbsent() {
        CompletableFuture<Response> future = lifecycle.register(context(501L));

        assertNull(lifecycle.cancelRequest(
                999L, 0L, CancelReason.CLIENT_CANCELLED));
        assertNull(lifecycle.cancelRequest(
                501L, 91L, CancelReason.CLIENT_CANCELLED));
        assertFalse(future.isDone());

        RequestState exact = lifecycle.cancelRequest(
                501L, 0L, CancelReason.CLIENT_CANCELLED);
        assertNotNull(exact);
        assertEquals(StrategyErrorType.REQUEST_CANCELLED.getErrorCode(),
                future.join().getCode());
    }

    @Test
    void publishedQueueDeadlineReleasesLocalReservationWithoutEngineCancel() {
        Registered registered = registerItem(602L);
        assertEquals(PlacementResult.Status.SUCCESS,
                commitRoute(lifecycle, registered));
        RequestSlot slot = lifecycle.requestSlot(602L);
        synchronized (slot) {
            slot.observePrefillFact(registered.item().prefillEp(), org.flexlb.dao.route.RoleType.PREFILL,
                    org.flexlb.balance.endpoint.PrefillState.WorkerStatusFact.active(registered.item()), System.currentTimeMillis());
        }
        lifecycle.cancelRequest(602L, 0L, CancelReason.DEADLINE_EXCEEDED);
        assertEquals(RequestState.Phase.TIMED_OUT,
                lifecycle.getRequestState(602L, 0L).state());
        verify(registered.item().decodeEp()).releaseReservationExact(
                registered.item().decodeReservation());
    }

    @Test
    void expiredArrivalDoesNotDisturbTheWaitingRequests() throws Exception {
        var low = lifecycle.register(context(1));
        var expired = context(2);
        expired.setSchedulingMetadata(SchedulingMetadata.explicit(90, System.currentTimeMillis() - 1L));
        assertEquals(StrategyErrorType.BATCH_SLO_EXPIRED.getErrorCode(),
                lifecycle.register(expired).get(5, TimeUnit.SECONDS).getCode());
        assertFalse(low.isDone());
        assertEquals(1, lifecycle.liveRequestCount());
    }

    @Test
    void concurrentGlobalAdmissionRetainsEveryUniqueRequest() throws Exception {
        try (var executor = Executors.newFixedThreadPool(8)) {
            List<Future<CompletableFuture<Response>>> futures = new ArrayList<>();
            for (long id = 1; id <= 128; id++) {
                long requestId = id;
                futures.add(executor.submit(() -> lifecycle.register(context(requestId))));
            }
            for (var future : futures) {
                assertFalse(future.get(5, TimeUnit.SECONDS).isDone());
            }
            assertEquals(128, lifecycle.liveRequestCount());
        }
    }

    private BalanceContext context(long requestId) {
        return RequestLifecycleTestSupport.context(config, requestId);
    }

    private Registered registerItem(long requestId) {
        BalanceContext context = context(requestId);
        CompletableFuture<Response> future = lifecycle.register(context);
        DecodeEndpoint decode = mock(DecodeEndpoint.class);
        DecodeEndpoint.ReservationHandle reservation =
                new DecodeEndpoint.ReservationHandle(1L, requestId, 1L);
        return new Registered(
                new ScheduledRequest(
                        context,
                        future,
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
