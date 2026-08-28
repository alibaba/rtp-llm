package org.flexlb.balance.scheduler;

import org.flexlb.balance.admission.AdmissionFallback;
import org.flexlb.balance.delivery.SlotDeliveryPort;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointEvent;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.eviction.EngineCancelChannel;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
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
import java.util.function.BooleanSupplier;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
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
    private RequestLifecycleCoordinator lifecycle;

    @BeforeEach
    void setUp() {
        config = SchedulingTestConfig.batchConfig();
        SchedulingTestConfig.usePriorityQueue(config);
        configService = mock(ConfigService.class);
        batchReporter = mock(BatchSchedulerReporter.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        lifecycle = new RequestLifecycleCoordinator(
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
        Router router = mock(Router.class);
        QueueRoutingResult.Rejected rejected =
                new QueueRoutingResult.Rejected(
                        RequestLifecycleCoordinator.buildErrorResponse(
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
                mock(AdmissionFallback.class),
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

        try (RequestLifecycleCoordinator.AdmissionScope admission =
                     lifecycle.beginAdmission(61L, registered.future())) {
            assertNotNull(admission);
            assertEquals(InflightCommitPort.RouteCommitResult.PUBLISHED,
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

        try (RequestLifecycleCoordinator.AdmissionScope admission =
                     lifecycle.beginAdmission(62L, registered.future())) {
            assertNotNull(admission);
            assertEquals(
                    InflightCommitPort.RouteCommitResult.PUBLICATION_REJECTED,
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

        try (RequestLifecycleCoordinator.AdmissionScope admission =
                     lifecycle.beginAdmission(63L, registered.future())) {
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
        bindRoute(registered, 0, 30_000L);

        assertEquals(InflightCommitPort.RouteCommitResult.REQUEST_CLOSED,
                lifecycle.commitRoute(
                        registered.item(), false, 0, 30_000L,
                        () -> true));
        assertEquals(1, lifecycle.decodeAcceptanceCount(),
                "a duplicate commit may release only its own permit");

        lifecycle.cancelRequest(101L, 0L, CancelReason.CLIENT_CANCELLED);
        assertEquals(0, lifecycle.decodeAcceptanceCount());
    }

    @Test
    void deferredCancellationReleasesGuardWhenAdmissionScopeCloses() {
        Registered registered = registerItem(201L);
        CompletableFuture<Response> future = registered.future();
        RequestLifecycleCoordinator.AdmissionScope admission =
                lifecycle.beginAdmission(201L, future);
        assertNotNull(admission);
        assertEquals(InflightCommitPort.RouteCommitResult.PUBLISHED,
                lifecycle.commitRoute(
                        registered.item(), false, 1, 30_000L,
                        () -> true));

        RequestLifecycleSnapshot requested = lifecycle.cancelRequest(
                201L, 0L, CancelReason.CLIENT_CANCELLED);
        assertEquals(RequestLifecycleState.CANCEL_REQUESTED,
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
        bind(registered);
        publishDecodeAccepted(registered);
        assertDecodeOwned(registered.item().requestId());

        assertEquals(InflightCommitPort.RouteCommitResult.REQUEST_CLOSED,
                lifecycle.commitRoute(
                        registered.item(), false, 1, 30_000L,
                        () -> true));
        assertEquals(0, lifecycle.decodeAcceptanceCount(),
                "Decode already owns the request, so no second guard remains");
    }

    @Test
    void decodeAcceptanceReleasesGuardBeforeItsTimeout() throws Exception {
        Registered registered = registerItem(401L);
        bindRoute(registered, 1, 30_000L);
        acknowledge(registered);
        assertEquals(1, lifecycle.decodeAcceptanceCount());

        publishDecodeAccepted(registered);

        assertDecodeOwned(registered.item().requestId());
        awaitCondition(() -> lifecycle.decodeAcceptanceCount() == 0);
        assertEquals(RequestLifecycleState.ACKNOWLEDGED,
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

    private BatchItem activeItem(long requestId) {
        RequestSlot slot = lifecycle.requestSlot(requestId);
        assertNotNull(slot);
        synchronized (slot) {
            return slot.activeItem();
        }
    }

    private void bind(Registered registered) {
        try (RequestLifecycleCoordinator.AdmissionScope admission =
                     lifecycle.beginAdmission(
                             registered.item().requestId(),
                             registered.future())) {
            assertNotNull(admission);
            assertTrue(lifecycle.commitItemForPublication(
                    registered.item(), false, () -> true));
        }
    }

    private void bindRoute(
            Registered registered,
            int limit,
            long acceptanceTimeoutMs) {
        try (RequestLifecycleCoordinator.AdmissionScope admission =
                     lifecycle.beginAdmission(
                             registered.item().requestId(),
                             registered.future())) {
            assertNotNull(admission);
            assertEquals(InflightCommitPort.RouteCommitResult.PUBLISHED,
                    lifecycle.commitRoute(
                            registered.item(), false, limit,
                            acceptanceTimeoutMs, () -> true));
        }
    }

    private void bindAndAcknowledge(Registered registered) {
        bind(registered);
        acknowledge(registered);
    }

    private void acknowledge(Registered registered) {
        SlotDeliveryPort.Claim claim = lifecycle.tryClaimForDelivery(
                registered.item(),
                SlotDeliveryPort.Identity.commitConfirmation(),
                () -> true);
        assertNotNull(claim);
        lifecycle.complete(
                claim, SlotDeliveryPort.Completion.Delivered.INSTANCE);
        assertTrue(registered.future().join().isSuccess());
    }

    private void publishDecodeAccepted(Registered registered) {
        lifecycle.onEndpointEvent(new EndpointEvent.StatusReduced(
                new DecodeEndpoint.StatusReduction(
                        registered.item().decodeEp(),
                        List.of(new DecodeEndpoint.AcceptedWorkerStatusFact(
                                registered.item().decodeReservation())))));
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
        BatchItem item = new BatchItem(
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
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(16L);
        BalanceContext context = new BalanceContext();
        context.setRequest(request);
        context.setConfig(config);
        context.setSchedulingMetadata(SchedulingMetadata.explicit(
                50,
                System.currentTimeMillis()
                        + TimeUnit.MINUTES.toMillis(1)));
        return context;
    }

    private static void await(CountDownLatch latch) {
        try {
            if (!latch.await(5, TimeUnit.SECONDS)) {
                throw new AssertionError("latch was not released");
            }
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
            throw new AssertionError(
                    "interrupted while awaiting latch", interrupted);
        }
    }

    private static void awaitCondition(BooleanSupplier condition)
            throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
        while (!condition.getAsBoolean() && System.nanoTime() < deadline) {
            Thread.sleep(1L);
        }
        assertTrue(condition.getAsBoolean(), "condition did not become true");
    }

    private record Registered(
            BatchItem item,
            CompletableFuture<Response> future) {
    }
}
