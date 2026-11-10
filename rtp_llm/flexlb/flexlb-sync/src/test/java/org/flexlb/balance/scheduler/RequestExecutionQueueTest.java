package org.flexlb.balance.scheduler;

import org.flexlb.balance.RoutingResult;
import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.master.WorkerStatus;
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
import java.util.OptionalLong;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.flexlb.balance.endpoint.EndpointTestSupport.withStateLock;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Canonical endpoint-queue contract exposed by {@link PrefillQueueScheduler}.
 */
class RequestExecutionQueueTest {

    private final List<PrefillQueueScheduler> runtimes = new ArrayList<>();
    private FlexlbConfig config;
    private PrefillEndpoint prefillEndpoint;
    private BlockingDispatcher deliveryStrategy;

    @BeforeEach
    void setUp() {
        config = SchedulingTestConfig.newConfig();
        SchedulingTestConfig.usePriorityQueue(config);
        SchedulingTestConfig.useSingleDecision(config);
        prefillEndpoint = stablePrefillEndpoint();
        deliveryStrategy = new BlockingDispatcher();
    }

    @AfterEach
    void stopRuntimes() {
        for (PrefillQueueScheduler runtime : runtimes) {
            assertNull(RequestExecutionTestSupport.stopAndAwait(runtime));
        }
    }

    @Test
    @org.junit.jupiter.api.Timeout(15)
    void removedCallbackFailureIsRecordedWithoutStoppingLaterQueueWork() throws Exception {
        var service = DeliveryTestSupport.service(config);
        AbstractRequestScheduler owner =
                SchedulerTestSupport.create(
                        service,
                        mock(DeliveryMetricsReporter.class),
                        mock(RequestSchedulerReporter.class),
                        mock(TheoryCacheHitReporter.class));
        RuntimeException observerFailure =
                new IllegalStateException("queue-removal observer failed");
        RuntimeException preparationFailure =
                new IllegalStateException("injected preparation failure");
        var callbackReturned = new CountDownLatch(1);
        var laterSelected = new CountDownLatch(1);
        PrefillQueueScheduler worker = null;
        try {
            prefillEndpoint =
                    org.flexlb.balance.endpoint.EndpointTestSupport.unstartedPrefill(
                            config, prefillEndpoint.getStatus(), deliveryStrategy, owner);
            worker =
                    org.flexlb.balance.endpoint.EndpointTestSupport.requestExecution(
                            prefillEndpoint);
            org.springframework.test.util.ReflectionTestUtils.setField(
                    deliveryStrategy,
                    "execution",
                    org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(owner).deliveries());
            var activeWorker = worker;
            var decode =
                    org.flexlb.balance.endpoint.EndpointTestSupport.decode(
                            WorkerStatus.createDiscovered(
                                    RoleType.DECODE, null, "127.0.0.1", 8190, 8191, null),
                            owner.repository);
            doAnswer(
                            call -> {
                                List<RouteResult> selected = call.getArgument(0);
                                if (selected.getFirst().requestId().equals("930")) {
                                    return CapacityBoundary.failed(preparationFailure);
                                }
                                laterSelected.countDown();
                                return CapacityBoundary.unavailable(
                                        CapacityBoundary.Resource.PREFILL_BATCH);
                            })
                    .when(worker)
                    .processSelection(
                            org.mockito.ArgumentMatchers.any(),
                            org.mockito.ArgumentMatchers.any(),
                            org.mockito.ArgumentMatchers.any(),
                            org.mockito.ArgumentMatchers.any());
            doAnswer(
                            call -> {
                                try {
                                    call.callRealMethod();
                                } finally {
                                    callbackReturned.countDown();
                                }
                                throw observerFailure;
                            })
                    .when(owner)
                    .recordPreparationFailure(
                            org.mockito.ArgumentMatchers.any(
                                    ContextResult.PreparationFailure.class));
            RequestExecutionTestSupport.start(worker);
            for (long id : new long[] {930L, 931L}) {
                var context = RequestProtocolTestSupport.context(config, id);
                var future = RequestProtocolTestSupport.submitForManualPlacement(owner, context);
                org.flexlb.balance.endpoint.DecodeResources.RequestReservation reservation;
                {
                    var selectedEndpoint = decode;

                    reservation =
                            org.flexlb.balance.endpoint.EndpointTestSupport.registerDecode(
                                    decode, id, 1L, 2L, 50);
                }
                var route =
                        org.flexlb.balance.scheduler.RequestExecutionTestSupport.mockProtocolRoute(
                                SchedulingTestConfig.createRoute(
                                        context,
                                        new Response(),
                                        null,
                                        null,
                                        prefillEndpoint,
                                        decode,
                                        reservation,
                                        System.currentTimeMillis()));
                RoutingContext admission =
                        SchedulerTestSupport.claim(owner, String.valueOf(id), future);
                try (var completed = RequestProtocolTestSupport.finishOnExit(admission)) {
                    assertNotNull(admission);
                    assertEquals(
                            RoutingResult.Status.SUCCESS,
                            RequestProtocolTestSupport.commitRoute(
                                    owner,
                                    route,
                                    RequestProtocolTestSupport.publication(
                                            () ->
                                                    RequestExecutionTestSupport.offer(
                                                            activeWorker, route))));
                }
                if (id == 930L) {
                    assertTrue(callbackReturned.await(5L, TimeUnit.SECONDS));
                    RequestProtocolTestSupport.awaitCondition(
                            () -> SchedulerTestSupport.failure(owner) != null);
                    assertSame(observerFailure, SchedulerTestSupport.failure(owner));
                    assertFalse(future.get(5L, TimeUnit.SECONDS).isSuccess());
                    RequestProtocolTestSupport.awaitCondition(
                            () -> !owner.repository.isCurrent(context));
                    assertEquals(0L, prefillEndpoint.admissionSummary(0, 0L).occupiedRequests());
                    assertFalse(decode.ownership(reservation).owned());
                    assertEquals(0, decode.routingView().totalLoad());
                } else {
                    assertTrue(
                            laterSelected.await(5L, TimeUnit.SECONDS),
                            "an observer failure must not stop the worker from planning the later request");
                    owner.cancel(String.valueOf(id), 0L, CancelReason.CLIENT_CANCELLED);
                    assertFalse(future.get(5L, TimeUnit.SECONDS).isSuccess());
                    RequestProtocolTestSupport.awaitCondition(
                            () -> !owner.repository.isCurrent(context));
                    assertEquals(0L, prefillEndpoint.admissionSummary(0, 0L).occupiedRequests());
                    assertFalse(decode.ownership(reservation).owned());
                    assertEquals(0, decode.routingView().totalLoad());
                }
            }
        } finally {
            if (worker != null) {
                assertNull(RequestExecutionTestSupport.stopAndAwait(worker));
            }
            if (RequestProtocolTestSupport.closeRegistrationAndAwaitScheduling(owner)) {
                owner.closeOutstandingAndTerminalize();
                SchedulerTestSupport.timer(
                                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(owner))
                        .close();
                SchedulerTestSupport.closeRequestExecutors(
                        org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(owner));
            }
        }
    }

    @Test
    void batchModeKeepsAcceptanceOrderWithoutAnExtraRequestCountLimit() {
        SchedulingTestConfig.useBatchDispatcher(config).setMaxInflightPerPrefillWorker(1);
        PrefillQueueScheduler runtime = runningRuntime();
        RouteResult low = item(901L, 10, Long.MAX_VALUE, 1, 128);
        RouteResult high = item(902L, 90, Long.MAX_VALUE, 2, 128);
        assertTrue(RequestExecutionTestSupport.offer(runtime, low));
        assertTrue(RequestExecutionTestSupport.offer(runtime, high));
        assertEquals(List.of(low, high), RequestExecutionTestSupport.capture(runtime).items());
    }

    @Test
    void acceptedRequestsKeepOfferOrderRegardlessOfPriority() {
        PrefillQueueScheduler runtime = runningRuntime();
        long now = System.currentTimeMillis();

        assertTrue(RequestExecutionTestSupport.offer(runtime, item(1, 50, now + 5_000, now, 128)));
        assertTrue(
                RequestExecutionTestSupport.offer(
                        runtime, item(2, 70, now + 9_000, now + 100, 128)));
        assertTrue(
                RequestExecutionTestSupport.offer(
                        runtime, item(3, 50, now + 1_000, now + 200, 128)));
        assertTrue(
                RequestExecutionTestSupport.offer(
                        runtime, item(4, 50, now + 5_000, now - 100, 128)));

        var snapshot = RequestExecutionTestSupport.capture(runtime);
        assertEquals(List.of(1L, 2L, 3L, 4L), requestIds(snapshot.items()));
        assertEquals(4, snapshot.items().size());
    }

    @Test
    void priorityOrderUsesUniqueOfferSequenceBeforeRequestIdOrCallerTimestamp() {
        PrefillQueueScheduler runtime = runningRuntime();
        long now = System.currentTimeMillis();

        assertTrue(RequestExecutionTestSupport.offer(runtime, item(1, 50, now + 9_000, now, 128)));
        assertTrue(RequestExecutionTestSupport.offer(runtime, item(2, 50, now + 1_000, now, 128)));
        assertTrue(RequestExecutionTestSupport.offer(runtime, item(4, 50, now + 9_000, now, 128)));
        assertTrue(RequestExecutionTestSupport.offer(runtime, item(3, 50, now + 9_000, now, 128)));

        assertEquals(
                List.of(1L, 2L, 4L, 3L),
                requestIds(RequestExecutionTestSupport.capture(runtime).items()));
    }

    @Test
    void fifoOrderIgnoresPriorityAndCallerTimestamp() {
        SchedulingTestConfig.useFifoQueue(config);
        PrefillQueueScheduler runtime = runningRuntime();
        long now = System.currentTimeMillis();

        assertTrue(RequestExecutionTestSupport.offer(runtime, item(1, 30, now + 1_000, now, 128)));
        assertTrue(
                RequestExecutionTestSupport.offer(runtime, item(2, 50, now + 500, now + 100, 128)));
        assertTrue(
                RequestExecutionTestSupport.offer(runtime, item(3, 70, now + 100, now + 200, 128)));

        assertEquals(
                List.of(1L, 2L, 3L),
                requestIds(RequestExecutionTestSupport.capture(runtime).items()));
    }

    @Test
    void life002AdmissionBoundaryAndFifoContract() {
        PrefillQueueScheduler runtime = runningRuntime();
        long now = System.currentTimeMillis();
        var requests =
                List.of(
                        item(1001L, 40, Long.MAX_VALUE, now, 128),
                        item(1002L, 40, Long.MAX_VALUE, now + 1, 256),
                        item(1003L, 40, Long.MAX_VALUE, now + 2, 512));

        requests.forEach(route -> assertTrue(RequestExecutionTestSupport.offer(runtime, route)));
        assertEquals(
                List.of(1001L, 1002L, 1003L),
                requestIds(RequestExecutionTestSupport.capture(runtime).items()));

        assertTrue(runtime.prefillEndpoint().releaseQueuedRequest(requests.get(0)));
        assertEquals(
                List.of(1002L, 1003L),
                requestIds(RequestExecutionTestSupport.capture(runtime).items()));
        assertTrue(
                RequestExecutionTestSupport.offer(
                        runtime, item(1004L, 40, Long.MAX_VALUE, now + 3, 128)));
        assertEquals(
                List.of(1002L, 1003L, 1004L),
                requestIds(RequestExecutionTestSupport.capture(runtime).items()),
                "released capacity must re-enter at the tail of the same-priority queue");
    }

    @Test
    void prefillQueueDoesNotReapplyGlobalPriorityOrdering() {
        PrefillQueueScheduler runtime = runningRuntime();
        long now = System.currentTimeMillis();
        assertTrue(
                RequestExecutionTestSupport.offer(
                        runtime, item(1101L, 0, Long.MAX_VALUE, now, 128)));
        assertTrue(
                RequestExecutionTestSupport.offer(
                        runtime, item(1102L, 50, Long.MAX_VALUE, now + 1, 128)));
        assertTrue(
                RequestExecutionTestSupport.offer(
                        runtime, item(1103L, 100, Long.MAX_VALUE, now + 2, 128)));
        assertTrue(
                RequestExecutionTestSupport.offer(
                        runtime, item(1104L, 50, Long.MAX_VALUE, now + 3, 1024)));

        assertEquals(
                List.of(1101L, 1102L, 1103L, 1104L),
                requestIds(RequestExecutionTestSupport.capture(runtime).items()),
                "accepted order is preserved regardless of request priority or length");
    }

    @Test
    void snapshotsExposeExactCanonicalIdentitiesAndMonotonicMutationVersion() {
        PrefillQueueScheduler runtime = runningRuntime();
        RouteResult first = item(11, 70, Long.MAX_VALUE, 1, 128);
        RouteResult second = item(12, 50, Long.MAX_VALUE, 2, 128);

        var empty = RequestExecutionTestSupport.capture(runtime);
        long emptyVersion =
                RequestExecutionTestSupport.state(runtime).captureQueue(1).queueVersion();
        assertTrue(RequestExecutionTestSupport.offer(runtime, first));
        assertTrue(RequestExecutionTestSupport.offer(runtime, second));
        var offered = RequestExecutionTestSupport.capture(runtime);

        long offeredVersion =
                RequestExecutionTestSupport.state(runtime).captureQueue(1).queueVersion();
        assertTrue(offeredVersion > emptyVersion);
        assertTrue(empty.items().isEmpty());
        assertSame(first, offered.items().get(0));
        assertSame(second, offered.items().get(1));
        assertEquals(2, RequestExecutionTestSupport.state(runtime).queueDepth());
        int[] counts = RequestExecutionTestSupport.state(runtime).queueCounters().byPriority();
        int[] expectedCounts = new int[counts.length];
        expectedCounts[70] = expectedCounts[50] = 1;
        org.junit.jupiter.api.Assertions.assertArrayEquals(expectedCounts, counts);

        assertTrue(runtime.prefillEndpoint().releaseQueuedRequest(first));
        var removed = RequestExecutionTestSupport.capture(runtime);
        assertTrue(
                RequestExecutionTestSupport.state(runtime).captureQueue(1).queueVersion()
                        > offeredVersion);
        assertEquals(List.of(first, second), offered.items());
        assertEquals(List.of(second), removed.items());
    }

    @Test
    void removalRequiresTheExactQueuedIdentityEvenForSameRequestId() {
        PrefillQueueScheduler runtime = runningRuntime();
        RouteResult canonical = item(21, 50, Long.MAX_VALUE, 1, 128);
        RouteResult lookalike = item(21, 50, Long.MAX_VALUE, 1, 128);
        assertTrue(RequestExecutionTestSupport.offer(runtime, canonical));

        assertFalse(runtime.prefillEndpoint().releaseQueuedRequest(lookalike));
        assertEquals(List.of(canonical), RequestExecutionTestSupport.capture(runtime).items());
        assertTrue(runtime.prefillEndpoint().releaseQueuedRequest(canonical));
    }

    @Test
    void activeQueueMutationsReuseImmutableCommittedWork() {
        prefillEndpoint =
                org.flexlb.balance.endpoint.EndpointTestSupport.unstartedPrefill(
                        config,
                        prefillEndpoint.getStatus(),
                        deliveryStrategy,
                        mock(AbstractRequestScheduler.class));
        PrefillQueueScheduler runtime =
                org.flexlb.balance.endpoint.EndpointTestSupport.requestExecution(prefillEndpoint);
        runtimes.add(runtime);
        RequestExecutionTestSupport.start(runtime);
        var before = prefillEndpoint.captureRouteProjectionInputs();
        RouteResult queued = item(905L, 50, Long.MAX_VALUE, 1L, 128L);
        assertTrue(RequestExecutionTestSupport.offer(runtime, queued));
        var after = prefillEndpoint.captureRouteProjectionInputs();
        assertTrue(after.ownershipVersion() > before.ownershipVersion());
        assertSame(before.work(), after.work());
        assertTrue(runtime.prefillEndpoint().releaseQueuedRequest(queued));
        assertSame(before.work(), prefillEndpoint.captureRouteProjectionInputs().work());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void queuedCancellationSettlesOutsideContextAndQueueLocks(boolean futureCancellation)
            throws Exception {
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        AbstractRequestScheduler registry =
                SchedulerTestSupport.create(
                        service,
                        mock(DeliveryMetricsReporter.class),
                        mock(RequestSchedulerReporter.class),
                        mock(TheoryCacheHitReporter.class));
        try {
            PrefillQueueScheduler runtime =
                    RequestExecutionTestSupport.create(
                            "test-worker", prefillEndpoint, config, deliveryStrategy, registry);
            runtimes.add(runtime);
            RequestContext context = RequestProtocolTestSupport.context(config, 907L);
            CompletableFuture<Response> future =
                    RequestProtocolTestSupport.submitForManualPlacement(registry, context);
            context.data().setFuture(future);
            RouteResult queued =
                    org.flexlb.balance.scheduler.RequestExecutionTestSupport.mockProtocolRoute(
                            SchedulingTestConfig.createRoute(
                                    context,
                                    new Response(),
                                    null,
                                    null,
                                    prefillEndpoint,
                                    null,
                                    null,
                                    System.currentTimeMillis()));
            AtomicReference<Thread> cleanupThread = new AtomicReference<>();
            AtomicInteger releases = new AtomicInteger();
            when(prefillEndpoint.releaseRequest(queued))
                    .thenAnswer(
                            call -> {
                                var state = RequestExecutionTestSupport.state(runtime);
                                assertFalse(
                                        Thread.holdsLock(context),
                                        "request cleanup must run outside Context's monitor");
                                org.junit.jupiter.api.Assertions.assertFalse(
                                        org.flexlb.balance.endpoint.EndpointTestSupport.stateLock(
                                                        state)
                                                .isHeldByCurrentThread(),
                                        "request cleanup must enter the resource ledger outside the queue lock");
                                cleanupThread.set(Thread.currentThread());
                                boolean released =
                                        org.flexlb.balance.endpoint.EndpointTestSupport
                                                .releaseRequest(state, queued);
                                if (released) {
                                    releases.incrementAndGet();
                                    org.junit.jupiter.api.Assertions.assertFalse(
                                            org.flexlb.balance.endpoint.EndpointTestSupport
                                                    .stateLock(state)
                                                    .isHeldByCurrentThread(),
                                            "capacity and queue notifications must run after resource release unlocks");
                                    runtime.capacityAvailableSignal.run();
                                }
                                return released;
                            });
            RoutingContext admission = SchedulerTestSupport.claim(registry, "907", future);
            try (var admissionCompletion1 = RequestProtocolTestSupport.finishOnExit(admission)) {
                assertEquals(
                        RoutingResult.Status.SUCCESS,
                        RequestProtocolTestSupport.commitRoute(
                                registry,
                                queued,
                                RequestProtocolTestSupport.publication(
                                        () -> {
                                            assertTrue(
                                                    SelectionTestSupport.registerPrefill(
                                                                    RequestExecutionTestSupport
                                                                            .state(runtime),
                                                                    queued,
                                                                    queued.prefillWorkMs(),
                                                                    true)
                                                            .accepted());
                                            assertTrue(prefillEndpoint.tryEnqueueRoute(queued));
                                            return true;
                                        })));
            }
            RequestExecutionTestSupport.start(runtime);
            RequestExecutionTestSupport.endpointChanged(
                    prefillEndpoint, PrefillEndpoint.Signal.INPUT);
            assertEquals(List.of(queued), RequestExecutionTestSupport.capture(runtime).items());
            RequestProtocolTestSupport.awaitCondition(
                    () -> {
                        var lock =
                                org.flexlb.balance.endpoint.EndpointTestSupport.stateLock(
                                        prefillEndpoint);
                        return withStateLock(
                                lock,
                                () -> {
                                    return org.springframework.test.util.ReflectionTestUtils
                                                    .getField(runtime, "waiting")
                                            != null;
                                });
                    });

            if (futureCancellation) {
                assertTrue(future.cancel(false));
                assertTrue(future.isCancelled());
                RequestProtocolTestSupport.awaitCondition(
                        () ->
                                SchedulerTestSupport.repository(registry)
                                                .getRequestState("907", 0L)
                                                .state()
                                        == RequestState.Phase.CANCELLED);
            } else {
                RequestState cancelled = registry.cancel("907", 0L, CancelReason.CLIENT_CANCELLED);
                assertEquals(RequestState.Phase.CANCELLED, cancelled.state());
                assertFalse(future.get(5, TimeUnit.SECONDS).isSuccess());
            }
            RequestProtocolTestSupport.awaitCondition(
                    () ->
                            SchedulerTestSupport.repository(registry)
                                                    .getRequestState("907", 0L)
                                                    .state()
                                            == RequestState.Phase.CANCELLED
                                    && RequestExecutionTestSupport.state(runtime).queueDepth()
                                            == 0);
            assertEquals(0, RequestExecutionTestSupport.state(runtime).queueDepth());
            assertEquals(
                    0L,
                    RequestExecutionTestSupport.state(runtime)
                            .admissionSummary(0, 0L)
                            .occupiedRequests());
            assertEquals(1, releases.get(), "the exact waiting resource seat is released once");
            assertFalse(
                    org.flexlb.balance.endpoint.EndpointTestSupport.releaseRequest(
                            RequestExecutionTestSupport.state(runtime), queued));
            assertEquals(
                    RequestState.Phase.CANCELLED,
                    SchedulerTestSupport.repository(registry).getRequestState("907", 0L).state());
            assertNotNull(cleanupThread.get());
            org.mockito.Mockito.verify(prefillEndpoint).releaseRequest(queued);
            org.mockito.Mockito.verify(prefillEndpoint).signal(PrefillEndpoint.Signal.PLACEMENT);
        } finally {
            if (RequestProtocolTestSupport.closeRegistrationAndAwaitScheduling(registry)) {
                registry.closeOutstandingAndTerminalize();
                SchedulerTestSupport.timer(SchedulerTestSupport.runtime(registry)).close();
                SchedulerTestSupport.closeRequestExecutors(SchedulerTestSupport.runtime(registry));
            }
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.CsvSource({
        "false,false",
        "false,true",
        "true,false",
        "true,true"
    })
    @org.junit.jupiter.api.Timeout(15)
    void failedQueueRejectsEveryRequestOutsideEndpointLock(boolean batch, boolean callbackFails) {
        if (batch) SchedulingTestConfig.useBatchDispatcher(config);
        else config.getDispatcher().setType(org.flexlb.config.DispatcherConfig.Type.NON_BATCH);
        var events = mock(AbstractRequestScheduler.class);
        var runtime =
                RequestExecutionTestSupport.create(
                        "failed-queue", prefillEndpoint, config, deliveryStrategy, events);
        runtimes.add(runtime);
        var cause = new IllegalStateException("scheduling failed");
        RuntimeException cleanupFailure =
                callbackFails ? new IllegalStateException("rejection callback failed") : null;
        var callbacks = new AtomicInteger();
        var queued = item(920L, 50, Long.MAX_VALUE, System.currentTimeMillis(), 128L);
        doAnswer(
                        call -> {
                            callbacks.incrementAndGet();
                            assertSame(cause, call.getArgument(1));
                            assertTrue(runtime.stopping());
                            assertFalse(
                                    org.flexlb.balance.endpoint.EndpointTestSupport.stateLock(
                                                    prefillEndpoint)
                                            .isHeldByCurrentThread());
                            assertTrue(
                                    RequestExecutionTestSupport.capture(runtime).items().isEmpty());
                            if (call.getArgument(0) == queued && cleanupFailure != null)
                                throw cleanupFailure;
                            RequestExecutionTestSupport.state(runtime)
                                    .releaseRequest(call.getArgument(0));
                            return null;
                        })
                .when(events)
                .rejectRoute(
                        org.mockito.ArgumentMatchers.any(), org.mockito.ArgumentMatchers.any());
        RequestExecutionTestSupport.start(runtime);
        assertTrue(RequestExecutionTestSupport.offer(runtime, queued));
        assertTrue(
                RequestExecutionTestSupport.offer(
                        runtime, item(922L, 50, Long.MAX_VALUE, System.currentTimeMillis(), 128L)));
        assertSame(cleanupFailure, runtime.failQueuedRequests(cause));
        assertEquals(2, callbacks.get(), "a failed callback must not skip later requests");
        assertFalse(
                RequestExecutionTestSupport.offer(
                        runtime, item(921L, 50, Long.MAX_VALUE, System.currentTimeMillis(), 128L)));
        var retirement =
                org.flexlb.balance.endpoint.EndpointTestSupport.retireResources(
                        RequestExecutionTestSupport.state(runtime));
        assertEquals(
                callbackFails ? List.of(queued) : List.of(),
                retirement.ownedItems(),
                "failed rejection retains its resource for endpoint retirement");
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void explicitStopDrainsRequestsWithoutAnExecutionThread(boolean fails) throws Exception {
        var owner = mock(AbstractRequestScheduler.class);
        var runtime =
                RequestExecutionTestSupport.create(
                        "custom-exit", prefillEndpoint, config, deliveryStrategy, owner);
        runtimes.add(runtime);
        var release = new CountDownLatch(1);
        var rejected = new CountDownLatch(1);
        var failure = new IllegalStateException("custom scheduling failed");
        var observed = new AtomicReference<Throwable>();
        doAnswer(
                        call -> {
                            observed.set(call.getArgument(1));
                            rejected.countDown();
                            return null;
                        })
                .when(owner)
                .rejectRoute(
                        org.mockito.ArgumentMatchers.any(), org.mockito.ArgumentMatchers.any());
        RequestExecutionTestSupport.start(runtime);
        try {
            assertTrue(
                    RequestExecutionTestSupport.offer(
                            runtime, item(999L, 50, Long.MAX_VALUE, 1L, 128L)));
            if (fails) runtime.failQueuedRequests(failure);
            else RequestExecutionTestSupport.stopAndAwait(runtime);
            assertTrue(rejected.await(5, TimeUnit.SECONDS));
            assertTrue(runtime.stopping());
            if (fails) assertSame(failure, observed.get());
            else assertTrue(observed.get() instanceof java.util.concurrent.CancellationException);
            assertNull(RequestExecutionTestSupport.stopAndAwait(runtime));
            assertTrue(RequestExecutionTestSupport.capture(runtime).items().isEmpty());
            assertFalse(
                    RequestExecutionTestSupport.offer(
                            runtime, item(1000L, 50, Long.MAX_VALUE, 2L, 128L)));
        } finally {
            release.countDown();
        }
    }

    @Test
    void constructingWorkerStateDoesNotChooseOrDispatchRequests() throws Exception {
        var owner = mock(AbstractRequestScheduler.class);
        var state =
                org.flexlb.balance.endpoint.EndpointTestSupport.prefillEndpoint(
                        new java.util.concurrent.locks.ReentrantLock(),
                        org.flexlb.balance.endpoint.OrderedPrefillRequestQueue.fifo(4),
                        System::currentTimeMillis);
        var dispatcher = mock(DeliveryCoordinator.class);
        var execution =
                org.flexlb.balance.endpoint.EndpointTestSupport.attach(
                        org.flexlb.balance.endpoint.EndpointTestSupport.worker(
                                "manual",
                                prefillEndpoint,
                                QueueExecutionSettings.capture(
                                        config,
                                        org.flexlb.balance.planner.BuiltinGroupPlanner.from(
                                                config)),
                                dispatcher,
                                state,
                                SchedulerTestSupport.queueDependencies(owner, config)));
        var delivered = new CountDownLatch(1);
        org.springframework.test.util.ReflectionTestUtils.setField(
                dispatcher, "dispatcher", mock(DeliveryDispatcher.class));
        doAnswer(
                        call -> {
                            delivered.countDown();
                            return null;
                        })
                .when(org.flexlb.balance.scheduler.DeliveryTestSupport.transport(dispatcher))
                .dispatch(org.mockito.ArgumentMatchers.any());
        try {
            var route = item(9999L, 50, Long.MAX_VALUE, 1L, 128L);
            SchedulerTestSupport.bindOwner(route.ctx(), owner);
            assertTrue(
                    SelectionTestSupport.registerPrefill(state, route, route.prefillWorkMs(), true)
                            .accepted());
            assertTrue(execution.prefillEndpoint().tryEnqueueRoute(route));
            assertFalse(delivered.await(100, TimeUnit.MILLISECONDS));
            assertEquals(List.of(route), execution.captureQueue(10).items());
        } finally {
            RequestExecutionTestSupport.stopAndAwait(execution);
        }
    }

    private PrefillQueueScheduler runningRuntime() {
        PrefillQueueScheduler runtime =
                RequestExecutionTestSupport.create(
                        "test-worker",
                        prefillEndpoint,
                        config,
                        deliveryStrategy,
                        mock(AbstractRequestScheduler.class));
        runtimes.add(runtime);
        RequestExecutionTestSupport.start(runtime);
        return runtime;
    }

    private RouteResult item(
            long requestId, int priority, long expiresAtMs, long enqueuedAtMs, long seqLen) {
        Request request =
                Request.builder()
                        .requestId(String.valueOf(requestId))
                        .seqLen(seqLen)
                        .rawPriority(priority)
                        .build();
        RequestContext context = new RequestContext(config);
        context.data().setRequest(request);
        context.data().setSchedulingMetadata(SchedulingMetadata.explicit(priority, expiresAtMs));
        context.data().setFuture(new CompletableFuture<Response>());
        return org.flexlb.balance.scheduler.RequestExecutionTestSupport.mockProtocolRoute(
                SchedulingTestConfig.createRoute(
                        context, null, null, null, prefillEndpoint, null, null, enqueuedAtMs));
    }

    private static List<Long> requestIds(List<RouteResult> items) {
        return items.stream().map(item -> Long.parseLong(item.requestId())).toList();
    }

    private static PrefillEndpoint stablePrefillEndpoint() {
        PrefillTimePredictor.Evaluator evaluator = mock(PrefillTimePredictor.Evaluator.class);
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        when(predictor.evaluator()).thenReturn(evaluator);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        org.springframework.test.util.ReflectionTestUtils.setField(
                endpoint, "predictor", predictor);
        WorkerStatus status =
                WorkerStatus.createDiscovered(
                        RoleType.PREFILL, "test", "127.0.0.1", 8080, 8090, "test-site");
        when(endpoint.getStatus()).thenReturn(status);
        return endpoint;
    }

    /**
     * Holds every exact head ACTIVE so snapshots can exercise live ordering.
     */
    private static final class BlockingDispatcher extends DeliveryTestSupport.StubDispatcher {

        private final AtomicInteger attempts = new AtomicInteger();

        @Override
        public CapacityBoundary selectBoundary(
                List<RouteResult> candidates,
                PrefillTimePredictor.Evaluator evaluator,
                OptionalLong plannedPrediction) {
            attempts.incrementAndGet();
            return CapacityBoundary.unavailable(CapacityBoundary.Resource.PREFILL_BATCH);
        }
    }
}
