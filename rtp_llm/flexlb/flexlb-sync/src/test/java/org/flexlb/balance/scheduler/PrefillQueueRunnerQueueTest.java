package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.RequestTimeEstimation;
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
 * Canonical endpoint-queue contract exposed by {@link PrefillQueueRunner}.
 */
class PrefillQueueRunnerQueueTest {

    private final List<PrefillQueueRunner> runtimes = new ArrayList<>();
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
        for (PrefillQueueRunner runtime : runtimes) {
            assertNull(runtime.stopAndAwait());
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
        PrefillQueueRunner worker = null;
        try {
            prefillEndpoint =
                    org.flexlb.balance.endpoint.EndpointTestSupport.unstartedPrefill(
                            config, prefillEndpoint.getStatus(), deliveryStrategy, owner);
            worker = org.flexlb.balance.endpoint.EndpointTestSupport.queueRunner(prefillEndpoint);
            var activeWorker = worker;
            var decode =
                    org.flexlb.balance.endpoint.EndpointTestSupport.decode(
                            WorkerStatus.createDiscovered(
                                    RoleType.DECODE, null, "127.0.0.1", 8190, 8191, null),
                            owner.requests);
            doAnswer(
                            call -> {
                                List<RequestRoute> selected = call.getArgument(0);
                                if (selected.getFirst().requestId().equals("930")) {
                                    return CapacityBoundary.failed(preparationFailure);
                                }
                                laterSelected.countDown();
                                return CapacityBoundary.deliveryUnavailable(
                                        CapacityBoundary.Resource.PREFILL_BATCH);
                            })
                    .when(worker)
                    .processSelection(
                            org.mockito.ArgumentMatchers.any(),
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
                    .onDeliveryEvidence(
                            org.mockito.ArgumentMatchers.isNull(),
                            org.mockito.ArgumentMatchers.isA(
                                    ContextResult.PreparationFailure.class));
            worker.start();
            for (long id : new long[] {930L, 931L}) {
                var context = RequestProtocolTestSupport.context(config, String.valueOf(id));
                var future = RequestProtocolTestSupport.register(owner, context);
                org.flexlb.balance.endpoint.DecodeResources.ReservationHandle reservation;
                try (var pin = decode.tryPinGeneration()) {
                    reservation =
                            org.flexlb.balance.endpoint.EndpointTestSupport.reserveUnqueuedDecode(
                                    decode, pin, String.valueOf(id), 1L, 2L, 50);
                }
                var route =
                        SchedulingTestConfig.createRoute(
                                context,
                                new Response(),
                                null,
                                null,
                                prefillEndpoint,
                                decode,
                                reservation,
                                System.currentTimeMillis());
                try (AdmissionHandle admission =
                                owner.claimAdmissionHandle(String.valueOf(id), future);
                        var completed = RequestProtocolTestSupport.finishOnExit(admission)) {
                    assertNotNull(admission);
                    assertEquals(
                            PlacementResult.Status.SUCCESS,
                            owner.commitRoute(
                                    route,
                                    RequestProtocolTestSupport.publication(
                                            () -> activeWorker.offer(route))));
                }
                if (id == 930L) {
                    assertTrue(callbackReturned.await(5L, TimeUnit.SECONDS));
                    RequestProtocolTestSupport.awaitCondition(
                            () -> SchedulerTestSupport.failure(owner) != null);
                    assertSame(observerFailure, SchedulerTestSupport.failure(owner));
                    assertFalse(future.get(5L, TimeUnit.SECONDS).isSuccess());
                    RequestProtocolTestSupport.awaitCondition(
                            () -> !owner.requests.isCurrent(context));
                    assertEquals(
                            0L,
                            prefillEndpoint.ownership.admissionSummary(0, 0L).occupiedRequests());
                    assertFalse(decode.ownership(reservation).owned());
                    assertEquals(0, decode.routingView().totalLoad());
                } else {
                    assertTrue(
                            laterSelected.await(5L, TimeUnit.SECONDS),
                            "an observer failure must not stop the worker from planning the later request");
                    owner.cancel(String.valueOf(id), 0L, CancelReason.CLIENT_CANCELLED);
                    assertFalse(future.get(5L, TimeUnit.SECONDS).isSuccess());
                    RequestProtocolTestSupport.awaitCondition(
                            () -> !owner.requests.isCurrent(context));
                    assertEquals(
                            0L,
                            prefillEndpoint.ownership.admissionSummary(0, 0L).occupiedRequests());
                    assertFalse(decode.ownership(reservation).owned());
                    assertEquals(0, decode.routingView().totalLoad());
                }
            }
        } finally {
            if (worker != null) {
                assertNull(worker.stopAndAwait());
            }
            if (RequestProtocolTestSupport.closeAdmissionAndAwaitMutations(owner)) {
                owner.closeOutstandingAndTerminalize();
                SchedulerTestSupport.timer(owner.runtime).close();
                SchedulerTestSupport.closeRequestExecutors(owner.runtime);
            }
        }
    }

    @Test
    void batchModeKeepsPriorityQueueWithoutAnExtraRequestCountLimit() {
        SchedulingTestConfig.useBatchDispatcher(config).setMaxInflightPerPrefillWorker(1);
        PrefillQueueRunner runtime = runningRuntime();
        RequestRoute low = item(901L, 10, Long.MAX_VALUE, 1, 128);
        RequestRoute high = item(902L, 90, Long.MAX_VALUE, 2, 128);
        assertTrue(runtime.offer(low));
        assertTrue(runtime.offer(high));
        assertEquals(List.of(high, low), PrefillQueueRunnerTestSupport.capture(runtime).items());
    }

    @Test
    void priorityOrderIsPriorityDescendingThenActualOfferFifo() {
        PrefillQueueRunner runtime = runningRuntime();
        long now = System.currentTimeMillis();

        assertTrue(runtime.offer(item(1, 50, now + 5_000, now, 128)));
        assertTrue(runtime.offer(item(2, 70, now + 9_000, now + 100, 128)));
        assertTrue(runtime.offer(item(3, 50, now + 1_000, now + 200, 128)));
        assertTrue(runtime.offer(item(4, 50, now + 5_000, now - 100, 128)));

        var snapshot = PrefillQueueRunnerTestSupport.capture(runtime);
        assertEquals(List.of("2", "1", "3", "4"), requestIds(snapshot.items()));
        assertEquals(4, snapshot.items().size());
    }

    @Test
    void priorityOrderUsesUniqueOfferSequenceBeforeRequestIdOrCallerTimestamp() {
        PrefillQueueRunner runtime = runningRuntime();
        long now = System.currentTimeMillis();

        assertTrue(runtime.offer(item(1, 50, now + 9_000, now, 128)));
        assertTrue(runtime.offer(item(2, 50, now + 1_000, now, 128)));
        assertTrue(runtime.offer(item(4, 50, now + 9_000, now, 128)));
        assertTrue(runtime.offer(item(3, 50, now + 9_000, now, 128)));

        assertEquals(
                List.of("1", "2", "4", "3"),
                requestIds(PrefillQueueRunnerTestSupport.capture(runtime).items()));
    }

    @Test
    void fifoOrderIgnoresPriorityAndCallerTimestamp() {
        SchedulingTestConfig.useFifoQueue(config);
        PrefillQueueRunner runtime = runningRuntime();
        long now = System.currentTimeMillis();

        assertTrue(runtime.offer(item(1, 30, now + 1_000, now, 128)));
        assertTrue(runtime.offer(item(2, 50, now + 500, now + 100, 128)));
        assertTrue(runtime.offer(item(3, 70, now + 100, now + 200, 128)));

        assertEquals(
                List.of("1", "2", "3"),
                requestIds(PrefillQueueRunnerTestSupport.capture(runtime).items()));
    }

    @Test
    void life002AdmissionBoundaryAndFifoContract() {
        PrefillQueueRunner runtime = runningRuntime();
        long now = System.currentTimeMillis();
        var requests =
                List.of(
                        item(1001L, 40, Long.MAX_VALUE, now, 128),
                        item(1002L, 40, Long.MAX_VALUE, now + 1, 256),
                        item(1003L, 40, Long.MAX_VALUE, now + 2, 512));

        requests.forEach(route -> assertTrue(runtime.offer(route)));
        assertEquals(
                List.of("1001", "1002", "1003"),
                requestIds(PrefillQueueRunnerTestSupport.capture(runtime).items()));

        assertTrue(runtime.control(requests.get(0), PrefillQueueRunner.Control.REMOVE));
        assertEquals(
                List.of("1002", "1003"),
                requestIds(PrefillQueueRunnerTestSupport.capture(runtime).items()));
        assertTrue(runtime.offer(item(1004L, 40, Long.MAX_VALUE, now + 3, 128)));
        assertEquals(
                List.of("1002", "1003", "1004"),
                requestIds(PrefillQueueRunnerTestSupport.capture(runtime).items()),
                "released capacity must re-enter at the tail of the same-priority queue");
    }

    @Test
    void sched001PriorityAndSamePriorityFairnessContract() {
        PrefillQueueRunner runtime = runningRuntime();
        long now = System.currentTimeMillis();
        assertTrue(runtime.offer(item(1101L, 0, Long.MAX_VALUE, now, 128)));
        assertTrue(runtime.offer(item(1102L, 50, Long.MAX_VALUE, now + 1, 128)));
        assertTrue(runtime.offer(item(1103L, 100, Long.MAX_VALUE, now + 2, 128)));
        assertTrue(runtime.offer(item(1104L, 50, Long.MAX_VALUE, now + 3, 1024)));

        assertEquals(
                List.of("1103", "1102", "1104", "1101"),
                requestIds(PrefillQueueRunnerTestSupport.capture(runtime).items()),
                "priority must dominate length while equal priority remains FIFO");
    }

    @Test
    void snapshotsExposeExactCanonicalIdentitiesAndMonotonicMutationVersion() {
        PrefillQueueRunner runtime = runningRuntime();
        RequestRoute first = item(11, 70, Long.MAX_VALUE, 1, 128);
        RequestRoute second = item(12, 50, Long.MAX_VALUE, 2, 128);

        var empty = PrefillQueueRunnerTestSupport.capture(runtime);
        long emptyVersion =
                PrefillQueueRunnerTestSupport.state(runtime).captureQueue(1).queueVersion();
        assertTrue(runtime.offer(first));
        assertTrue(runtime.offer(second));
        var offered = PrefillQueueRunnerTestSupport.capture(runtime);

        long offeredVersion =
                PrefillQueueRunnerTestSupport.state(runtime).captureQueue(1).queueVersion();
        assertTrue(offeredVersion > emptyVersion);
        assertTrue(empty.items().isEmpty());
        assertSame(first, offered.items().get(0));
        assertSame(second, offered.items().get(1));
        assertEquals(2, PrefillQueueRunnerTestSupport.state(runtime).progress().queueDepth());
        assertEquals(java.util.Map.of(70, 1, 50, 1), runtime.queueSizeByPriority());

        assertTrue(runtime.control(first, PrefillQueueRunner.Control.REMOVE));
        var removed = PrefillQueueRunnerTestSupport.capture(runtime);
        assertTrue(
                PrefillQueueRunnerTestSupport.state(runtime).captureQueue(1).queueVersion()
                        > offeredVersion);
        assertEquals(List.of(first, second), offered.items());
        assertEquals(List.of(second), removed.items());
    }

    @Test
    void removalRequiresTheExactQueuedIdentityEvenForSameRequestId() {
        PrefillQueueRunner runtime = runningRuntime();
        RequestRoute canonical = item(21, 50, Long.MAX_VALUE, 1, 128);
        RequestRoute lookalike = item(21, 50, Long.MAX_VALUE, 1, 128);
        assertTrue(runtime.offer(canonical));

        assertFalse(runtime.control(lookalike, PrefillQueueRunner.Control.REMOVE));
        assertEquals(List.of(canonical), PrefillQueueRunnerTestSupport.capture(runtime).items());
        assertTrue(runtime.control(canonical, PrefillQueueRunner.Control.REMOVE));
    }

    @Test
    void activeQueueMutationsReuseImmutableCommittedWork() {
        prefillEndpoint =
                org.flexlb.balance.endpoint.EndpointTestSupport.unstartedPrefill(
                        config,
                        prefillEndpoint.getStatus(),
                        deliveryStrategy,
                        mock(AbstractRequestScheduler.class));
        PrefillQueueRunner runtime =
                org.flexlb.balance.endpoint.EndpointTestSupport.queueRunner(prefillEndpoint);
        runtimes.add(runtime);
        runtime.start();
        var before = prefillEndpoint.captureRouteProjectionInputs();
        RequestRoute queued = item(905L, 50, Long.MAX_VALUE, 1L, 128L);
        assertTrue(runtime.offer(queued));
        var after = prefillEndpoint.captureRouteProjectionInputs();
        assertTrue(after.ownershipVersion() > before.ownershipVersion());
        assertSame(before.work(), after.work());
        assertTrue(runtime.control(queued, PrefillQueueRunner.Control.REMOVE));
        assertSame(before.work(), prefillEndpoint.captureRouteProjectionInputs().work());
    }

    @Test
    void controlTicketRunsOnWorkerOutsideQueueLock() throws Exception {
        AbstractRequestScheduler events = mock(AbstractRequestScheduler.class);
        PrefillQueueRunner runtime =
                PrefillQueueRunnerTestSupport.create(
                        "test-worker", prefillEndpoint, config, deliveryStrategy, events);
        runtimes.add(runtime);
        runtime.start();
        RequestRoute queued = item(906L, 50, Long.MAX_VALUE, 1L, 128L);
        assertTrue(runtime.offer(queued));
        CountDownLatch delivered = new CountDownLatch(1);
        AtomicReference<Thread> callbackThread = new AtomicReference<>();
        AtomicReference<Boolean> callbackHeldLock = new AtomicReference<>();
        doAnswer(
                        invocation -> {
                            callbackThread.set(Thread.currentThread());
                            var lock = prefillEndpoint.ownership.queueGuard();
                            callbackHeldLock.set(lock.isHeldByCurrentThread());
                            delivered.countDown();
                            return null;
                        })
                .when(events)
                .onQueueEvent(queued, AbstractRequestScheduler.QueueKind.CONTROL, null, null);

        runtime.control(queued, PrefillQueueRunner.Control.WAKE);

        assertTrue(delivered.await(5, TimeUnit.SECONDS));
        assertFalse(callbackHeldLock.get());
        assertFalse(callbackThread.get() == Thread.currentThread());
        assertEquals(List.of(queued), PrefillQueueRunnerTestSupport.capture(runtime).items());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void queuedCancellationIsSettledByLocalWorker(boolean futureCancellation) throws Exception {
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        AbstractRequestScheduler registry =
                SchedulerTestSupport.create(
                        service,
                        mock(DeliveryMetricsReporter.class),
                        mock(RequestSchedulerReporter.class),
                        mock(TheoryCacheHitReporter.class));
        try {
            PrefillQueueRunner runtime =
                    PrefillQueueRunnerTestSupport.create(
                            "test-worker", prefillEndpoint, config, deliveryStrategy, registry);
            runtimes.add(runtime);
            runtime.start();
            RequestContext context = RequestProtocolTestSupport.context(config, "907");
            CompletableFuture<Response> future =
                    RequestProtocolTestSupport.register(registry, context);
            context.data().setFuture(future);
            RequestRoute queued =
                    SchedulingTestConfig.createRoute(
                            context,
                            new Response(),
                            null,
                            null,
                            prefillEndpoint,
                            null,
                            null,
                            System.currentTimeMillis());
            doAnswer(
                            call -> {
                                runtime.signal(false);
                                return null;
                            })
                    .when(prefillEndpoint)
                    .signal(PrefillEndpoint.Signal.INPUT);
            when(prefillEndpoint.control(queued, PrefillQueueRunner.Control.WAKE))
                    .thenAnswer(call -> runtime.control(queued, PrefillQueueRunner.Control.WAKE));
            AtomicReference<Thread> cleanupThread = new AtomicReference<>();
            AtomicInteger releases = new AtomicInteger();
            when(prefillEndpoint.release(null, queued))
                    .thenAnswer(
                            call -> {
                                var state = PrefillQueueRunnerTestSupport.state(runtime);
                                assertFalse(
                                        Thread.holdsLock(context),
                                        "request cleanup must run outside Context's monitor");
                                assertFalse(
                                        state.queueGuard().isHeldByCurrentThread(),
                                        "request cleanup must enter the resource ledger outside the queue lock");
                                cleanupThread.set(Thread.currentThread());
                                boolean released =
                                        org.flexlb.balance.endpoint.EndpointTestSupport
                                                .releaseRequest(state, queued);
                                if (released) {
                                    releases.incrementAndGet();
                                    assertFalse(
                                            state.queueGuard().isHeldByCurrentThread(),
                                            "capacity and queue notifications must run after resource release unlocks");
                                    runtime.signal(false);
                                    runtime.capacityAvailableSignal.run();
                                }
                                return released;
                            });
            try (AdmissionHandle admission = registry.claimAdmissionHandle("907", future);
                    var admissionCompletion1 = RequestProtocolTestSupport.finishOnExit(admission)) {
                assertEquals(
                        PlacementResult.Status.SUCCESS,
                        registry.commitRoute(
                                queued,
                                RequestProtocolTestSupport.publication(
                                        () -> {
                                            assertTrue(runtime.offer(queued));
                                            org.junit.jupiter.api.Assertions.assertDoesNotThrow(
                                                    () ->
                                                            RequestProtocolTestSupport
                                                                    .awaitCondition(
                                                                            () ->
                                                                                    runtime.getLatestQueueWaitSnapshot()
                                                                                            .containsValue(
                                                                                                    "Route commit in progress")));
                                            return true;
                                        })));
            }
            assertEquals(List.of(queued), PrefillQueueRunnerTestSupport.capture(runtime).items());
            RequestProtocolTestSupport.awaitCondition(
                    () -> {
                        var lock = prefillEndpoint.ownership.queueGuard();
                        lock.lock();
                        try {
                            return org.springframework.test.util.ReflectionTestUtils.getField(
                                            runtime, "capacityBlockedHead")
                                    != null;
                        } finally {
                            lock.unlock();
                        }
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
                assertEquals(RequestState.Phase.CANCEL_REQUESTED, cancelled.state());
                assertFalse(future.get(5, TimeUnit.SECONDS).isSuccess());
            }
            RequestProtocolTestSupport.awaitCondition(
                    () ->
                            SchedulerTestSupport.repository(registry)
                                                    .getRequestState("907", 0L)
                                                    .state()
                                            == RequestState.Phase.CANCELLED
                                    && PrefillQueueRunnerTestSupport.state(runtime)
                                                    .progress()
                                                    .queueDepth()
                                            == 0);
            assertEquals(0, PrefillQueueRunnerTestSupport.state(runtime).progress().queueDepth());
            assertEquals(
                    0L,
                    PrefillQueueRunnerTestSupport.state(runtime)
                            .admissionSummary(0, 0L)
                            .occupiedRequests());
            assertEquals(1, releases.get(), "the exact waiting resource seat is released once");
            assertFalse(
                    org.flexlb.balance.endpoint.EndpointTestSupport.releaseRequest(
                            PrefillQueueRunnerTestSupport.state(runtime), queued));
            assertEquals(
                    RequestState.Phase.CANCELLED,
                    SchedulerTestSupport.repository(registry).getRequestState("907", 0L).state());
            assertNotNull(cleanupThread.get());
            assertFalse(cleanupThread.get() == Thread.currentThread());
            org.mockito.Mockito.verify(prefillEndpoint).release(null, queued);
            org.mockito.Mockito.verify(prefillEndpoint).signal(PrefillEndpoint.Signal.PLACEMENT);
        } finally {
            if (RequestProtocolTestSupport.closeAdmissionAndAwaitMutations(registry)) {
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
    void concurrentStopSharesDrainAndPreservesInterruptedWaiter(
            boolean batch, boolean callbackFails) throws Exception {
        if (batch) {
            SchedulingTestConfig.useBatchDispatcher(config);
        } else {
            config.getDispatcher().setType(org.flexlb.config.DispatcherConfig.Type.NON_BATCH);
        }
        AbstractRequestScheduler events = mock(AbstractRequestScheduler.class);
        PrefillQueueRunner runtime =
                PrefillQueueRunnerTestSupport.create(
                        "stop-test", prefillEndpoint, config, deliveryStrategy, events);
        if (!callbackFails) {
            runtimes.add(runtime);
        }
        RuntimeException cleanupFailure =
                callbackFails ? new IllegalStateException("stop callback failed") : null;
        CountDownLatch callbackEntered = new CountDownLatch(1);
        CountDownLatch releaseCallback = new CountDownLatch(1);
        AtomicInteger callbacks = new AtomicInteger();
        RequestRoute queued = item(920L, 50, Long.MAX_VALUE, System.currentTimeMillis(), 128L);
        doAnswer(
                        invocation -> {
                            callbacks.incrementAndGet();
                            org.junit.jupiter.api.Assertions.assertThrows(
                                    IllegalStateException.class, runtime::stopAndAwait);
                            if (invocation.getArgument(0) == queued) {
                                callbackEntered.countDown();
                                RequestProtocolTestSupport.await(releaseCallback);
                                if (cleanupFailure != null) {
                                    throw cleanupFailure;
                                }
                            }
                            return null;
                        })
                .when(events)
                .onQueueEvent(
                        org.mockito.ArgumentMatchers.any(),
                        org.mockito.ArgumentMatchers.eq(
                                AbstractRequestScheduler.QueueKind.REJECTED),
                        org.mockito.ArgumentMatchers.isNull(),
                        org.mockito.ArgumentMatchers.any());
        runtime.start();
        assertTrue(runtime.offer(queued));
        assertTrue(runtime.offer(item(922L, 50, Long.MAX_VALUE, System.currentTimeMillis(), 128L)));
        var stops = java.util.concurrent.Executors.newFixedThreadPool(2);
        AtomicReference<Thread> waiter = new AtomicReference<>();
        var interruptPreserved = new java.util.concurrent.atomic.AtomicBoolean();
        try {
            var first = stops.submit(runtime::stopAndAwait);
            assertTrue(callbackEntered.await(5L, TimeUnit.SECONDS));
            var second =
                    stops.submit(
                            () -> {
                                waiter.set(Thread.currentThread());
                                Thread.currentThread().interrupt();
                                Throwable result = runtime.stopAndAwait();
                                interruptPreserved.set(Thread.currentThread().isInterrupted());
                                return result;
                            });
            RequestProtocolTestSupport.awaitCondition(
                    () -> waiter.get() != null && waiter.get().getState() == Thread.State.WAITING);
            assertFalse(second.isDone());
            assertFalse(
                    runtime.offer(
                            item(921L, 50, Long.MAX_VALUE, System.currentTimeMillis(), 128L)));
            releaseCallback.countDown();
            assertSame(cleanupFailure, first.get(5L, TimeUnit.SECONDS));
            assertSame(cleanupFailure, second.get(5L, TimeUnit.SECONDS));
            assertTrue(interruptPreserved.get());
            assertEquals(2, callbacks.get(), "later items must drain even after a callback fails");
            assertEquals(0, PrefillQueueRunnerTestSupport.state(runtime).progress().queueDepth());
            assertSame(cleanupFailure, runtime.stopAndAwait());
            if (callbackFails) {
                var retirement =
                        PrefillQueueRunnerTestSupport.state(runtime).retireGenerationOwnership();
                assertNull(retirement.invariantFailure());
                assertEquals(
                        List.of(queued),
                        retirement.ownedItems(),
                        "failed stop callback must retain the exact owner for generation retirement");
            }
        } finally {
            releaseCallback.countDown();
            stops.shutdownNow();
            assertTrue(stops.awaitTermination(5L, TimeUnit.SECONDS));
            assertSame(cleanupFailure, runtime.stopAndAwait());
        }
    }

    private PrefillQueueRunner runningRuntime() {
        PrefillQueueRunner runtime =
                PrefillQueueRunnerTestSupport.create(
                        "test-worker",
                        prefillEndpoint,
                        config,
                        deliveryStrategy,
                        mock(AbstractRequestScheduler.class));
        runtimes.add(runtime);
        runtime.start();
        return runtime;
    }

    private RequestRoute item(
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
        return SchedulingTestConfig.createRoute(
                context, null, null, null, prefillEndpoint, null, null, enqueuedAtMs);
    }

    private static List<String> requestIds(List<RequestRoute> items) {
        return items.stream().map(RequestRoute::requestId).toList();
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
                List<RequestRoute> candidates,
                PrefillTimePredictor.Evaluator evaluator,
                OptionalLong plannedPrediction) {
            attempts.incrementAndGet();
            return CapacityBoundary.unavailable(
                    CapacityBoundary.Resource.PREFILL_BATCH,
                    new RequestTimeEstimation.AdmissionBlockSemantics(
                            "TEST_QUEUE_BLOCK",
                            RequestTimeEstimation.AfterProbeAdmission.BLOCKED,
                            "TEST_QUEUE_BLOCK",
                            RoleType.PREFILL));
        }
    }
}
