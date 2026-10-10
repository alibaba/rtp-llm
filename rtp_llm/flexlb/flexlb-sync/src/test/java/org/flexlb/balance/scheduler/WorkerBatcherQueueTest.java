package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.RecentCacheKeyTraceReporter;
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
import java.util.concurrent.locks.ReentrantLock;

import static org.flexlb.balance.scheduler.SchedulingTestConfig.freezeInputs;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/** Canonical endpoint-queue contract exposed by {@link WorkerBatcher}. */
class WorkerBatcherQueueTest {

    private final List<WorkerBatcher> runtimes = new ArrayList<>();
    private FlexlbConfig config;
    private PrefillEndpoint prefillEndpoint;
    private BlockingDeliveryStrategy deliveryStrategy;

    @BeforeEach
    void setUp() {
        config = org.flexlb.balance.scheduler.SchedulingTestConfig.newConfig();
        SchedulingTestConfig.usePriorityQueue(config);
        SchedulingTestConfig.useSingleDecision(config);
        prefillEndpoint = stablePrefillEndpoint();
        deliveryStrategy = new BlockingDeliveryStrategy();
    }

    @AfterEach
    void stopRuntimes() {
        for (WorkerBatcher runtime : runtimes) {
            assertNull(runtime.stopAndAwait());
        }
    }

    @Test
    void batchModeKeepsPriorityQueueWithoutAnExtraRequestCountLimit() {
        SchedulingTestConfig.useBatchDispatcher(config).setMaxInflightPerPrefillWorker(1);
        WorkerBatcher runtime = runningRuntime();
        RequestRoute low = item(901L, 10, Long.MAX_VALUE, 1, 128);
        RequestRoute high = item(902L, 90, Long.MAX_VALUE, 2, 128);
        assertTrue(runtime.offer(low));
        assertTrue(runtime.offer(high));
        assertEquals(List.of(high, low), WorkerBatcherTestSupport.capture(runtime).items());
    }

    @Test
    void priorityOrderIsPriorityDescendingThenActualOfferFifo() {
        WorkerBatcher runtime = runningRuntime();
        long now = System.currentTimeMillis();

        assertTrue(runtime.offer(item(1, 50, now + 5_000, now, 128)));
        assertTrue(runtime.offer(item(2, 70, now + 9_000, now + 100, 128)));
        assertTrue(runtime.offer(item(3, 50, now + 1_000, now + 200, 128)));
        assertTrue(runtime.offer(item(4, 50, now + 5_000, now - 100, 128)));

        var snapshot =
                WorkerBatcherTestSupport.capture(runtime);
        assertEquals(List.of(2L, 1L, 3L, 4L), requestIds(snapshot.items()));
        assertEquals(4, snapshot.items().size());
    }

    @Test
    void priorityOrderUsesUniqueOfferSequenceBeforeRequestIdOrCallerTimestamp() {
        WorkerBatcher runtime = runningRuntime();
        long now = System.currentTimeMillis();

        assertTrue(runtime.offer(item(1, 50, now + 9_000, now, 128)));
        assertTrue(runtime.offer(item(2, 50, now + 1_000, now, 128)));
        assertTrue(runtime.offer(item(4, 50, now + 9_000, now, 128)));
        assertTrue(runtime.offer(item(3, 50, now + 9_000, now, 128)));

        assertEquals(List.of(1L, 2L, 4L, 3L), requestIds(
                WorkerBatcherTestSupport.capture(runtime).items()));
    }

    @Test
    void fifoOrderIgnoresPriorityAndCallerTimestamp() {
        SchedulingTestConfig.useFifoQueue(config);
        WorkerBatcher runtime = runningRuntime();
        long now = System.currentTimeMillis();

        assertTrue(runtime.offer(item(1, 30, now + 1_000, now, 128)));
        assertTrue(runtime.offer(item(2, 50, now + 500, now + 100, 128)));
        assertTrue(runtime.offer(item(3, 70, now + 100, now + 200, 128)));

        assertEquals(List.of(1L, 2L, 3L), requestIds(
                WorkerBatcherTestSupport.capture(runtime).items()));
    }

    @Test
    void snapshotsExposeExactCanonicalIdentitiesAndMonotonicMutationVersion() {
        WorkerBatcher runtime = runningRuntime();
        RequestRoute first = item(11, 70, Long.MAX_VALUE, 1, 128);
        RequestRoute second = item(12, 50, Long.MAX_VALUE, 2, 128);

        var empty =
                WorkerBatcherTestSupport.capture(runtime);
        long emptyVersion = WorkerBatcherTestSupport.state(runtime).captureQueue(1).queueVersion();
        assertTrue(runtime.offer(first));
        assertTrue(runtime.offer(second));
        var offered =
                WorkerBatcherTestSupport.capture(runtime);

        long offeredVersion = WorkerBatcherTestSupport.state(runtime).captureQueue(1).queueVersion();
        assertTrue(offeredVersion > emptyVersion);
        assertTrue(empty.items().isEmpty());
        assertSame(first, offered.items().get(0));
        assertSame(second, offered.items().get(1));
        assertEquals(2, WorkerBatcherTestSupport.state(runtime).queueDepth());
        assertEquals(java.util.Map.of(70, 1, 50, 1),
                runtime.queueSizeByPriority());

        assertTrue(runtime.removeQueued(first, "test exact removal"));
        var removed =
                WorkerBatcherTestSupport.capture(runtime);
        assertTrue(WorkerBatcherTestSupport.state(runtime).captureQueue(1).queueVersion() > offeredVersion);
        assertEquals(List.of(first, second), offered.items());
        assertEquals(List.of(second), removed.items());
    }

    @Test
    void removalRequiresTheExactQueuedIdentityEvenForSameRequestId() {
        WorkerBatcher runtime = runningRuntime();
        RequestRoute canonical = item(21, 50, Long.MAX_VALUE, 1, 128);
        RequestRoute lookalike = item(21, 50, Long.MAX_VALUE, 1, 128);
        assertTrue(runtime.offer(canonical));

        assertFalse(runtime.removeQueued(lookalike, "stale identity"));
        assertEquals(List.of(canonical),
                WorkerBatcherTestSupport.capture(runtime).items());
        assertTrue(runtime.removeQueued(canonical, "canonical identity"));
    }

    @Test
    void activeQueueMutationsReuseImmutableCommittedWork() {
        prefillEndpoint = org.flexlb.balance.endpoint.EndpointTestSupport.unstartedPrefill(config,
                prefillEndpoint.getStatus(), deliveryStrategy, mock(AbstractRequestScheduler.class));
        WorkerBatcher runtime = org.flexlb.balance.endpoint.EndpointTestSupport.batcher(prefillEndpoint);
        runtimes.add(runtime);
        runtime.start();
        var before = prefillEndpoint.captureRouteProjectionInputs();
        RequestRoute queued = item(905L, 50, Long.MAX_VALUE, 1L, 128L);
        assertTrue(runtime.offer(queued));
        var after = prefillEndpoint.captureRouteProjectionInputs();
        assertTrue(after.ownershipVersion() > before.ownershipVersion());
        assertSame(before.work(), after.work());
        assertTrue(runtime.removeQueued(queued, "test cleanup"));
        assertSame(before.work(), prefillEndpoint.captureRouteProjectionInputs().work());
    }

    @Test
    void controlTicketRunsOnWorkerOutsideQueueLock() throws Exception {
        AbstractRequestScheduler events = mock(AbstractRequestScheduler.class);
        WorkerBatcher runtime = WorkerBatcherTestSupport.create(
                "test-worker", prefillEndpoint, config, deliveryStrategy, events);
        runtimes.add(runtime);
        runtime.start();
        RequestRoute queued = item(906L, 50, Long.MAX_VALUE, 1L, 128L);
        assertTrue(runtime.offer(queued));
        CountDownLatch delivered = new CountDownLatch(1);
        AtomicReference<Thread> callbackThread = new AtomicReference<>();
        AtomicReference<Boolean> callbackHeldLock = new AtomicReference<>();
        doAnswer(invocation -> {
            callbackThread.set(Thread.currentThread());
            ReentrantLock lock = (ReentrantLock) org.springframework.test.util.ReflectionTestUtils
                    .getField(runtime, "queueLock");
            callbackHeldLock.set(lock.isHeldByCurrentThread());
            delivered.countDown();
            return null;
        }).when(events).onQueuedItemControl(queued);

        runtime.signalControl(queued);

        assertTrue(delivered.await(5, TimeUnit.SECONDS));
        assertFalse(callbackHeldLock.get());
        assertFalse(callbackThread.get() == Thread.currentThread());
        assertEquals(List.of(queued), WorkerBatcherTestSupport.capture(runtime).items());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void queuedCancellationIsSettledByLocalWorker(boolean futureCancellation) throws Exception {
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        AbstractRequestScheduler registry = org.flexlb.balance.scheduler.SchedulerTestSupport.create(service,
                mock(DeliveryMetricsReporter.class), mock(RequestSchedulerReporter.class),
                mock(RecentCacheKeyTraceReporter.class));
        try {
            WorkerBatcher runtime = WorkerBatcherTestSupport.create("test-worker", prefillEndpoint,
                    config, deliveryStrategy, registry);
            runtimes.add(runtime);
            runtime.start();
            RequestContext context = RequestProtocolTestSupport.context(config, 907L);
            CompletableFuture<Response> future = RequestProtocolTestSupport.register(registry, context);
            context.setFuture(future);
            RequestRoute queued = org.flexlb.balance.scheduler.SchedulingTestConfig.createRoute(freezeInputs(context), new Response(),
                    null, null, prefillEndpoint, null, null, System.currentTimeMillis());
            doAnswer(call -> {
                runtime.signalSchedulingInputsChanged();
                return null;
            }).when(prefillEndpoint).signalRouteReady();
            when(prefillEndpoint.signalQueuedControl(queued)).thenAnswer(call -> runtime.signalControl(queued));
            AtomicReference<Thread> cleanupThread = new AtomicReference<>();
            AtomicInteger releases = new AtomicInteger();
            when(prefillEndpoint.releaseRequest(queued)).thenAnswer(call -> {
                var state = WorkerBatcherTestSupport.state(runtime);
                assertFalse(Thread.holdsLock(context), "request cleanup must run outside Context's monitor");
                assertFalse(state.ownershipLock().isHeldByCurrentThread(),
                        "request cleanup must enter the resource ledger outside the queue lock");
                cleanupThread.set(Thread.currentThread());
                boolean released = org.flexlb.balance.endpoint.EndpointTestSupport.releaseRequest(state, queued);
                if (released) {
                    releases.incrementAndGet();
                    assertFalse(state.ownershipLock().isHeldByCurrentThread(),
                            "capacity and queue notifications must run after resource release unlocks");
                    runtime.signalSchedulingInputsChanged();
                    runtime.capacityAvailableSignal().run();
                }
                return released;
            });
            try (RequestContext.AdmissionHandle admission = registry.claimAdmissionHandle(907L, future); var admissionCompletion1 = RequestProtocolTestSupport.finishOnExit(admission)) {
                assertEquals(PlacementResult.Status.SUCCESS,
                        registry.commitRoute(queued, RequestProtocolTestSupport.publication(() -> {
                            assertTrue(runtime.offer(queued));
                            org.junit.jupiter.api.Assertions.assertDoesNotThrow(() ->
                                    RequestProtocolTestSupport.awaitCondition(() ->
                                            runtime.getLatestQueueWaitSnapshot().containsValue("Route commit in progress")));
                            return true;
                        })));
            }
            assertEquals(List.of(queued), WorkerBatcherTestSupport.capture(runtime).items());
            RequestProtocolTestSupport.awaitCondition(() -> {
                ReentrantLock lock = (ReentrantLock) org.springframework.test.util.ReflectionTestUtils
                        .getField(runtime, "queueLock");
                lock.lock();
                try {
                    return org.springframework.test.util.ReflectionTestUtils
                            .getField(runtime, "capacityBlockedHead") != null;
                } finally {
                    lock.unlock();
                }
            });

            if (futureCancellation) {
                assertTrue(future.cancel(false));
                assertTrue(future.isCancelled());
                RequestProtocolTestSupport.awaitCondition(() ->
                        org.flexlb.balance.scheduler.SchedulerTestSupport.repository(registry).getRequestState(907L, 0L).state() == RequestState.Phase.CANCELLED);
            } else {
                RequestState cancelled = registry.cancel(907L, 0L, CancelReason.CLIENT_CANCELLED);
                assertEquals(RequestState.Phase.CANCEL_REQUESTED, cancelled.state());
                assertFalse(future.get(5, TimeUnit.SECONDS).isSuccess());
            }
            RequestProtocolTestSupport.awaitCondition(() ->
                    SchedulerTestSupport.repository(registry).getRequestState(907L, 0L).state()
                            == RequestState.Phase.CANCELLED
                            && WorkerBatcherTestSupport.state(runtime).queueDepth() == 0);
            assertEquals(0, WorkerBatcherTestSupport.state(runtime).queueDepth());
            assertEquals(0L, WorkerBatcherTestSupport.state(runtime).admissionSummary(0, 0L).occupiedRequests());
            assertEquals(1, releases.get(), "the exact waiting resource seat is released once");
            assertFalse(org.flexlb.balance.endpoint.EndpointTestSupport.releaseRequest(
                    WorkerBatcherTestSupport.state(runtime), queued));
            assertEquals(RequestState.Phase.CANCELLED,
                    SchedulerTestSupport.repository(registry).getRequestState(907L, 0L).state());
            assertNotNull(cleanupThread.get());
            assertFalse(cleanupThread.get() == Thread.currentThread());
            org.mockito.Mockito.verify(prefillEndpoint).releaseRequest(queued);
            org.mockito.Mockito.verify(prefillEndpoint).signalPlacementCapacityChanged();
        } finally {
            if (RequestProtocolTestSupport.closeAdmissionAndAwaitMutations(registry)) {
                registry.closeOutstandingAndTerminalize();
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry).timer().close();
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry).closeRequestExecutors();
            }
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.CsvSource({"false,false", "false,true", "true,false", "true,true"})
    @org.junit.jupiter.api.Timeout(15)
    void concurrentStopSharesDrainAndPreservesInterruptedWaiter(boolean batch, boolean callbackFails) throws Exception {
        if (batch) { SchedulingTestConfig.useBatchDispatcher(config); }
        else { config.getDispatcher().setType(org.flexlb.config.DispatcherConfig.Type.NON_BATCH); }
        AbstractRequestScheduler events = mock(AbstractRequestScheduler.class);
        WorkerBatcher runtime = WorkerBatcherTestSupport.create("stop-test", prefillEndpoint, config,
                deliveryStrategy, events);
        if (!callbackFails) { runtimes.add(runtime); }
        RuntimeException cleanupFailure = callbackFails ? new IllegalStateException("stop callback failed") : null;
        CountDownLatch callbackEntered = new CountDownLatch(1);
        CountDownLatch releaseCallback = new CountDownLatch(1);
        AtomicInteger callbacks = new AtomicInteger();
        RequestRoute queued = item(920L, 50, Long.MAX_VALUE, System.currentTimeMillis(), 128L);
        doAnswer(invocation -> {
            callbacks.incrementAndGet();
            org.junit.jupiter.api.Assertions.assertThrows(IllegalStateException.class, runtime::stopAndAwait);
            if (invocation.getArgument(0) == queued) {
                callbackEntered.countDown();
                RequestProtocolTestSupport.await(releaseCallback);
                if (cleanupFailure != null) { throw cleanupFailure; }
            }
            return null;
        }).when(events).onQueueOfferFailure(org.mockito.ArgumentMatchers.any(), org.mockito.ArgumentMatchers.any());
        runtime.start();
        assertTrue(runtime.offer(queued));
        assertTrue(runtime.offer(item(922L, 50, Long.MAX_VALUE, System.currentTimeMillis(), 128L)));
        var stops = java.util.concurrent.Executors.newFixedThreadPool(2);
        AtomicReference<Thread> waiter = new AtomicReference<>();
        var interruptPreserved = new java.util.concurrent.atomic.AtomicBoolean();
        try {
            var first = stops.submit(runtime::stopAndAwait);
            assertTrue(callbackEntered.await(5L, TimeUnit.SECONDS));
            var second = stops.submit(() -> {
                waiter.set(Thread.currentThread());
                Thread.currentThread().interrupt();
                Throwable result = runtime.stopAndAwait();
                interruptPreserved.set(Thread.currentThread().isInterrupted());
                return result;
            });
            RequestProtocolTestSupport.awaitCondition(() -> waiter.get() != null
                    && waiter.get().getState() == Thread.State.WAITING);
            assertFalse(second.isDone());
            assertFalse(runtime.offer(item(921L, 50, Long.MAX_VALUE, System.currentTimeMillis(), 128L)));
            releaseCallback.countDown();
            assertSame(cleanupFailure, first.get(5L, TimeUnit.SECONDS));
            assertSame(cleanupFailure, second.get(5L, TimeUnit.SECONDS));
            assertTrue(interruptPreserved.get());
            assertEquals(2, callbacks.get(), "later items must drain even after a callback fails");
            assertEquals(0, WorkerBatcherTestSupport.state(runtime).queueDepth());
            assertSame(cleanupFailure, runtime.stopAndAwait());
            if (callbackFails) {
                var retirement = WorkerBatcherTestSupport.state(runtime).retireGenerationOwnership();
                assertNull(retirement.invariantFailure());
                assertEquals(List.of(queued), retirement.ownedItems(),
                        "failed stop callback must retain the exact owner for generation retirement");
            }
        } finally {
            releaseCallback.countDown();
            stops.shutdownNow();
            assertTrue(stops.awaitTermination(5L, TimeUnit.SECONDS));
            assertSame(cleanupFailure, runtime.stopAndAwait());
        }
    }

    private WorkerBatcher runningRuntime() {
        WorkerBatcher runtime = WorkerBatcherTestSupport.create(
                "test-worker",
                prefillEndpoint,
                config,
                deliveryStrategy,
                mock(AbstractRequestScheduler.class));
        runtimes.add(runtime);
        runtime.start();
        return runtime;
    }

    private RequestRoute item(long requestId, int priority, long expiresAtMs,
                           long enqueuedAtMs, long seqLen) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(seqLen);
        request.setPriority(priority);
        RequestContext context = new RequestContext(config);
        context.setRequest(request);
        context.setSchedulingMetadata(
                SchedulingMetadata.explicit(priority, expiresAtMs));
        context.setFuture(new CompletableFuture<Response>());
        return org.flexlb.balance.scheduler.SchedulingTestConfig.createRoute(freezeInputs(context),
                null,
                null,
                null,
                prefillEndpoint,
                null,
                null,
                enqueuedAtMs);
    }

    private static List<Long> requestIds(List<RequestRoute> items) {
        return items.stream().map(RequestRoute::requestId).toList();
    }

    private static PrefillEndpoint stablePrefillEndpoint() {
        PrefillTimePredictor.Evaluator evaluator =
                mock(PrefillTimePredictor.Evaluator.class);
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        when(predictor.evaluator()).thenReturn(evaluator);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getPredictor()).thenReturn(predictor);
        WorkerStatus status = WorkerStatus.createDiscovered(
                RoleType.PREFILL,
                "test",
                "127.0.0.1",
                8080,
                8090,
                "test-site");
        when(endpoint.getStatus()).thenReturn(status);
        return endpoint;
    }

    /** Holds every exact head ACTIVE so snapshots can exercise live ordering. */
    private static final class BlockingDeliveryStrategy
            implements DeliveryStrategy {
        @Override
        public void deliver(DeliveryTransaction transaction, String reason, int queueDepth,
                org.flexlb.balance.projection.WorkSnapshot precedingWork, PrefillTimePredictor.Evaluator evaluator) {
            throw new AssertionError("boundary-only strategy cannot deliver");
        }


        private final AtomicInteger attempts = new AtomicInteger();
        private final CapacityBoundary.Availability availability =
                new CapacityBoundary.Availability() {
                    @Override
                    public boolean isAvailable() {
                        return false;
                    }

                    @Override
                    public void addListener(Runnable listener) {
                    }

                    @Override
                    public void removeListener(Runnable listener) {
                    }
                };

        @Override
        public DeliveryTransaction prepare(
                List<RequestRoute> candidates,
                PrefillTimePredictor.Evaluator evaluator,
                OptionalLong plannedPrediction) {
            attempts.incrementAndGet();
            return WorkerBatcherTestSupport.boundaryOnly(
                    candidates.getFirst(),
                    CapacityBoundary.unavailable(
                            availability,
                            new RouteProjection.AdmissionBlockSemantics(
                                    "TEST_QUEUE_BLOCK",
                                    RouteProjection.AfterProbeAdmission.BLOCKED,
                                    "TEST_QUEUE_BLOCK",
                                    RoleType.PREFILL)));
        }

        @Override
        public GroupPlanner.PrefixPrediction<RequestRoute> newGroupPredictor(
                PrefillTimePredictor.Evaluator evaluator) {
            return (added, items) -> {
                return 0.0;
            };
        }

        @Override
        public RouteProjection.DeliveryProjection projectionPolicy() {
            return mock(RouteProjection.DeliveryProjection.class);
        }
    }
}
