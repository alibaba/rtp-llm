package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;

import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.eviction.EngineCancelChannel;
import org.flexlb.balance.eviction.EvictionManager;
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

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.flexlb.balance.scheduler.RequestLifecycleTestSupport.await;
import static org.flexlb.balance.scheduler.RequestLifecycleTestSupport.awaitCondition;
import static org.flexlb.balance.scheduler.RequestLifecycleTestSupport.bind;
import static org.flexlb.balance.scheduler.RequestLifecycleTestSupport.bindRoute;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/** Regression coverage for exact ownership of Decode-admission resources. */
class RequestAdmissionResourceLeakTest {

    private FlexlbConfig config;
    private ConfigService configService;
    private BatchSchedulerReporter batchReporter;
    private RequestRegistry lifecycle;

    @BeforeEach
    void setUp() {
        config = SchedulingTestConfig.batchConfig();
        SchedulingTestConfig.usePriorityQueue(config);
        configService = mock(ConfigService.class);
        batchReporter = mock(BatchSchedulerReporter.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        lifecycle = new RequestRegistry(
                configService,
                batchReporter,
                mock(RequestSchedulerReporter.class),
                mock(EngineCancelChannel.class));
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
    void normalSubmitRejectionNeverInstallsAnAcceptanceGuard() {
        BalanceContext context = context(51L);
        DefaultRouter router = mock(DefaultRouter.class);
        PlacementResult<QueueRouteAdmission, PlacementKey> rejected =
                PlacementResult.rejected(
                        RequestRegistry.buildErrorResponse(
                                StrategyErrorType.NO_PREFILL_WORKER, null));
        when(router.routeForQueue(context)).thenAnswer(invocation -> {
            assertEquals(0, lifecycle.decodeAcceptanceCount(),
                    "unpublished placement must not own an acceptance guard");
            return rejected;
        });
        RequestScheduler scheduler = new RequestScheduler(
                configService,
                router,
                mock(EndpointRegistry.class),
                batchReporter,
                mock(EvictionManager.class),
                lifecycle);

        Response response = scheduler.submit(context).join();

        assertEquals(StrategyErrorType.NO_PREFILL_WORKER.getErrorCode(),
                response.getCode());
        assertEquals(0, lifecycle.decodeAcceptanceCount(),
                "route rejection cannot leak an unbound guard");
        scheduler.closePlacement();
    }

    @Test
    void publishedRouteBindsAcceptanceGuardExactlyOnce() {
        Registered registered = registerItem(61L);

        try (AdmissionMutation admission =
                     lifecycle.claimAdmissionMutation(61L, registered.future())) {
            assertNotNull(admission);
            assertEquals(PlacementResult.Status.SUCCESS,
                    lifecycle.commitRoute(
                            registered.item(), false, 1, 30_000L,
                            () -> true));
        }

        assertEquals(1, lifecycle.decodeAcceptanceCount());
        lifecycle.cancelRequest(61L, 0L, CancelReason.CLIENT_CANCELLED);
        assertEquals(0, lifecycle.decodeAcceptanceCount());
    }

    @Test
    void declinedRoutePublicationReleasesGuardAndItem() {
        Registered registered = registerItem(62L);

        try (AdmissionMutation admission =
                     lifecycle.claimAdmissionMutation(62L, registered.future())) {
            assertNotNull(admission);
            assertEquals(
                    PlacementResult.Status.BLOCKED,
                    lifecycle.commitRoute(
                            registered.item(), false, 1, 30_000L,
                            () -> false));
            assertEquals(0, lifecycle.decodeAcceptanceCount());
            assertTrue(lifecycle.isAdmissionOpen(
                    62L, registered.future()));
            assertNull(activeItem(62L));
        }
    }

    @Test
    void throwingRoutePublicationReleasesGuardAndItem() {
        Registered registered = registerItem(63L);

        try (AdmissionMutation admission =
                     lifecycle.claimAdmissionMutation(63L, registered.future())) {
            assertNotNull(admission);
            assertThrows(IllegalStateException.class,
                    () -> lifecycle.commitRoute(
                            registered.item(), false, 1, 30_000L,
                            () -> {
                                throw new IllegalStateException("publish failed");
                            }));
            assertEquals(0, lifecycle.decodeAcceptanceCount());
            assertTrue(lifecycle.isAdmissionOpen(
                    63L, registered.future()));
            assertNull(activeItem(63L));
        }
    }

    @Test
    void duplicateRouteCommitReleasesOnlyItsNewPermit() {
        Registered registered = registerItem(101L);
        bindRoute(lifecycle, registered, 0, 30_000L);

        assertEquals(PlacementResult.Status.CLOSED,
                lifecycle.commitRoute(
                        registered.item(), false, 0, 30_000L,
                        () -> true));
        assertEquals(1, lifecycle.decodeAcceptanceCount(),
                "a duplicate commit may release only its own permit");

        lifecycle.cancelRequest(101L, 0L, CancelReason.CLIENT_CANCELLED);
        assertEquals(0, lifecycle.decodeAcceptanceCount());
    }

    @Test
    void deferredCancellationReleasesGuardWhenAdmissionMutationCloses() {
        Registered registered = registerItem(201L);
        CompletableFuture<Response> future = registered.future();
        AdmissionMutation admission =
                lifecycle.claimAdmissionMutation(201L, future);
        assertNotNull(admission);
        assertEquals(PlacementResult.Status.SUCCESS,
                lifecycle.commitRoute(
                        registered.item(), false, 1, 30_000L,
                        () -> true));

        RequestState requested = lifecycle.cancelRequest(
                201L, 0L, CancelReason.CLIENT_CANCELLED);
        assertEquals(RequestState.Phase.CANCEL_REQUESTED,
                requested.state());
        assertEquals(1, lifecycle.decodeAcceptanceCount(),
                "the open admission mutation still owns terminal cleanup");

        admission.close();

        assertEquals(0, lifecycle.decodeAcceptanceCount());
        assertEquals(StrategyErrorType.REQUEST_CANCELLED.getErrorCode(),
                future.join().getCode());
    }

    @Test
    void duplicateCommitAfterDecodeOwnershipCannotInstallAnotherGuard() {
        Registered registered = registerItem(301L);
        bind(lifecycle, registered);
        publishDecodeAccepted(registered);
        assertDecodeOwned(registered.item().requestId());

        assertEquals(PlacementResult.Status.CLOSED,
                lifecycle.commitRoute(
                        registered.item(), false, 1, 30_000L,
                        () -> true));
        assertEquals(0, lifecycle.decodeAcceptanceCount(),
                "Decode already owns the request, so no second guard remains");
    }

    @Test
    void decodeAcceptanceReleasesGuardBeforeItsTimeout() throws Exception {
        Registered registered = registerItem(401L);
        bindRoute(lifecycle, registered, 1, 30_000L);
        acknowledge(registered);
        assertEquals(1, lifecycle.decodeAcceptanceCount());

        publishDecodeAccepted(registered);

        assertDecodeOwned(registered.item().requestId());
        awaitCondition(() -> lifecycle.decodeAcceptanceCount() == 0);
        assertEquals(RequestState.Phase.ACKNOWLEDGED,
                lifecycle.getRequestState(401L, 0L).state());
    }

    @Test
    void acceptanceTimeoutAndLateDecodeFactReleaseResourceExactlyOnce()
            throws Exception {
        Registered registered = registerItem(501L);
        AtomicInteger releases = bindCountingResource(
                registered.item().requestId(), 10L);
        bindAndAcknowledge(registered);

        awaitCondition(() -> releases.get() == 1);
        publishDecodeAccepted(registered);

        assertDecodeOwned(registered.item().requestId());
        assertEquals(1, releases.get(),
                "a late Decode fact must not release an expired resource twice");
    }

    @Test
    void competingTerminalPathsReleaseResourceExactlyOnce() throws Exception {
        ExecutorService contenders = Executors.newFixedThreadPool(2);
        try {
            for (int iteration = 0; iteration < 64; iteration++) {
                long requestId = 1_000L + iteration;
                CompletableFuture<Response> future =
                        lifecycle.register(context(requestId), 4);
                AtomicInteger releases =
                        bindCountingResource(requestId, 30_000L);
                CountDownLatch start = new CountDownLatch(1);

                Future<?> externalCompletion = contenders.submit(() -> {
                    await(start);
                    future.complete(Response.error(
                            StrategyErrorType.INVALID_REQUEST));
                });
                Future<?> cancellation = contenders.submit(() -> {
                    await(start);
                    lifecycle.cancelRequest(
                            requestId, 0L, CancelReason.CLIENT_CANCELLED);
                });
                start.countDown();
                externalCompletion.get(5, TimeUnit.SECONDS);
                cancellation.get(5, TimeUnit.SECONDS);
                future.get(5, TimeUnit.SECONDS);

                assertEquals(1, releases.get(),
                        "terminal race leaked or double-released request "
                                + requestId);
            }
        } finally {
            contenders.shutdownNow();
        }
    }

    @Test
    void shutdownReleasesResourceExactlyOnce() {
        CompletableFuture<Response> future =
                lifecycle.register(context(601L), 4);
        AtomicInteger releases = bindCountingResource(601L, 30_000L);

        assertTrue(lifecycle.closeAdmissionAndAwaitMutations());
        lifecycle.closeOutstandingAndTerminalize();
        lifecycle.closeExpiration();
        lifecycle.closePublisher();

        assertEquals(1, releases.get());
        assertTrue(future.isDone());
    }

    private AtomicInteger bindCountingResource(
            long requestId, long acceptanceTimeoutMs) {
        RequestSlot slot = lifecycle.requestSlot(requestId);
        assertNotNull(slot);
        AtomicInteger releases = new AtomicInteger();
        synchronized (slot) {
            assertNull(slot.bindAdmissionResources(
                    releases::incrementAndGet, acceptanceTimeoutMs));
        }
        return releases;
    }

    private ScheduledRequest activeItem(long requestId) {
        RequestSlot slot = lifecycle.requestSlot(requestId);
        assertNotNull(slot);
        synchronized (slot) {
            return slot.activeItem();
        }
    }

    private void bindAndAcknowledge(Registered registered) {
        bind(lifecycle, registered);
        acknowledge(registered);
    }

    private void acknowledge(Registered registered) {
        RequestRegistry.DeliveryClaim claim = lifecycle.tryClaimRouteDelivery(
                registered.item(),
                () -> true);
        assertNotNull(claim);
        lifecycle.complete(
                claim, DeliveryResult.delivered());
        assertTrue(registered.future().join().isSuccess());
    }

    private void publishDecodeAccepted(Registered registered) {
        new EndpointEventProjector(lifecycle).onDecodeStatus(
                registered.item().decodeEp(),
                List.of(DecodeEndpoint.WorkerStatusFact.accepted(
                        registered.item().decodeReservation())));
    }

    private void assertDecodeOwned(long requestId) {
        RequestSlot slot = lifecycle.requestSlot(requestId);
        assertNotNull(slot);
        synchronized (slot) {
            assertTrue(slot.decodeOwnsRequest());
        }
    }

    private Registered registerItem(long requestId) {
        BalanceContext context = context(requestId);
        CompletableFuture<Response> future = lifecycle.register(context, 4);
        DecodeEndpoint decode = mock(DecodeEndpoint.class);
        DecodeEndpoint.ReservationHandle reservation =
                new DecodeEndpoint.ReservationHandle(1L, requestId, 1L);
        ScheduledRequest item = new ScheduledRequest(
                context,
                future,
                new Response(),
                null,
                null,
                null,
                decode,
                reservation,
                System.currentTimeMillis());
        return new Registered(item, future);
    }

    private BalanceContext context(long requestId) {
        return RequestLifecycleTestSupport.context(config, requestId);
    }
}
