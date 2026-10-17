package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.balance.projection.WorkSnapshot;
import org.flexlb.config.DecisionPolicyConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.RecentCacheKeyTraceReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.OptionalLong;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Concurrency contracts at the final, threaded Prefill scheduling boundary.
 */
class WorkerBatcherSchedulingTest {

    private final List<WorkerBatcher> runtimes = new ArrayList<>();

    @AfterEach
    void stopRuntimes() {
        for (WorkerBatcher runtime : runtimes) {
            assertNull(runtime.stopAndAwait());
        }
    }

    @Test
    void projectionConstraintsRequireTheSharedOwnershipLock() {
        WorkerBatcher runtime = runningRuntime(singleConfig(), mock(PrefillEndpoint.class), SchedulerTestSupport.dispatcher());
        assertThrows(IllegalStateException.class, runtime::planningViewLocked);
        var lock = WorkerBatcherTestSupport.state(runtime).lock;
        lock.lock();
        try {
            assertEquals(1, runtime.planningViewLocked().constraints().maxRequests());
        } finally {
            lock.unlock();
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = { false, true })
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    void routeCommitWakesAWorkerThatAlreadyObservedRouting(boolean fixedWindow) throws Exception {
        FlexlbConfig config = fixedWindow ? fixedConfig() : singleConfig();
        if (fixedWindow) { SchedulingTestConfig.useFixedWindowDecision(config).setMaxRequests(1); }
        var service = mock(org.flexlb.config.ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        var scheduler = SchedulerTestSupport.create(service, mock(org.flexlb.service.monitor.DeliveryMetricsReporter.class), mock(org.flexlb.service.monitor.RequestSchedulerReporter.class),
                mock(RecentCacheKeyTraceReporter.class));
        PrefillEndpoint endpoint = stableEndpoint(stableStatus());
        EventDrivenBlock delivery = new EventDrivenBlock();
        WorkerBatcher runtime = runningRuntime(config, endpoint, delivery);
        doAnswer(call -> {
            runtime.signal(false);
            return null;
        }).when(endpoint).signal(PrefillEndpoint.Signal.INPUT);
        RequestContext context = item(config, endpoint, 901L, 50, System.currentTimeMillis()).ctx();
        context.data().setGenerateInputPb(com.google.protobuf.ByteString.copyFromUtf8("input"));
        var future = RequestProtocolTestSupport.register(scheduler, context);
        context.data().setFuture(future);
        var request = SchedulingTestConfig.createRoute(context, null, null, null, endpoint, null, null, System.currentTimeMillis());
        CountDownLatch published = new CountDownLatch(1);
        CountDownLatch allowCommit = new CountDownLatch(1);
        try (var executor = java.util.concurrent.Executors.newSingleThreadExecutor();
            var routing = scheduler.claimAdmissionHandle(901L, future); var admissionCompletion1 = RequestProtocolTestSupport.finishOnExit(routing)) {
            assertNotNull(routing);
            var committed = executor.submit(() -> scheduler.commitRoute(request, RequestProtocolTestSupport.publication(() -> {
                assertTrue(runtime.offer(request));
                published.countDown();
                await(allowCommit);
                return true;
            })));
            try {
                await(published);
                RequestProtocolTestSupport.awaitCondition(() -> runtime.getLatestQueueWaitSnapshot().containsValue("Route commit in progress"));
                assertEquals(SchedulerTestSupport.Stage.ROUTING, SchedulerTestSupport.stage(context));
                assertEquals(0, delivery.attempts.get());
                // The wait is unbounded even when a request deadline elapses during publication.
                var decision = org.springframework.test.util.ReflectionTestUtils.invokeMethod(runtime, "processQueue");
                assertEquals(Long.MAX_VALUE, org.springframework.test.util.ReflectionTestUtils.getField(decision, "wakeAtMs"));
                allowCommit.countDown();
                assertEquals(org.flexlb.balance.PlacementResult.Status.SUCCESS, committed.get(2, TimeUnit.SECONDS));
                await(delivery.firstAttempt);
                assertEquals(SchedulerTestSupport.Stage.READY_TO_DELIVER, SchedulerTestSupport.stage(context));
                assertEquals(1, delivery.attempts.get());
            } finally {
                allowCommit.countDown();
            }
        } finally {
            runtime.stopAndAwait();
            RequestProtocolTestSupport.closeAdmissionAndAwaitMutations(scheduler);
            scheduler.closeOutstandingAndTerminalize();
            SchedulerTestSupport.timer(SchedulerTestSupport.runtime(scheduler)).close();
            SchedulerTestSupport.closeRequestExecutors(SchedulerTestSupport.runtime(scheduler));
        }
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    void routeCommitFinishingAfterSnapshotCannotMixPlacementFacts() throws Exception {
        FlexlbConfig config = singleConfig();
        PrefillEndpoint endpoint = stableEndpoint(stableStatus());
        WorkerBatcher worker = WorkerBatcherTestSupport.create("commit-snapshot-race", endpoint, config,
                SchedulerTestSupport.dispatcher(), mock(AbstractRequestScheduler.class));
        RequestRoute route = spy(item(config, endpoint, 902L, 50, System.currentTimeMillis()));
        RequestContext context = route.ctx();
        context.register(902L, System.currentTimeMillis(), Long.MAX_VALUE, 60_000L);
        var operation = ((ContextResult.Granted<ContextResult.PlacementToken>) context.beginPlacement(null)).value();
        context.bindPlacement(operation, route, ContextResult.PublicationResult.PREPARED);
        CountDownLatch sampled = new CountDownLatch(1);
        CountDownLatch committed = new CountDownLatch(1);
        AtomicInteger reads = new AtomicInteger();
        // This probe releases the real Context monitor after sampling, so an actual
        // placement completion can run on another thread before Worker uses the sample.
        RequestContext observation = mock(RequestContext.class, org.mockito.AdditionalAnswers.delegatesTo(context));
        doAnswer(call -> {
            var state = context.snapshot();
            if (reads.incrementAndGet() == 1) {
                assertNotNull(state.placement());
                sampled.countDown();
                await(committed);
            }
            return state;
        }).when(observation).snapshot();
        org.mockito.Mockito.doReturn(observation).when(route).ctx();
        var queue = WorkerBatcherTestSupport.state(worker);
        queue.lock.lock();
        try { assertSame(route, queue.updateQueue(PrefillState.QueueChange.ENQUEUE, route, 16L, System.currentTimeMillis())); }
        finally { queue.lock.unlock(); }
        try (var executor = java.util.concurrent.Executors.newSingleThreadExecutor()) {
            var completion = executor.submit(() -> {
                await(sampled);
                try {
                    assertEquals(ContextResult.UpdateKind.APPLIED, context.completePlacement(operation,
                            ContextResult.PublicationResult.PUBLISHED, System.currentTimeMillis()).kind());
                } finally { committed.countDown(); }
            });
            var decision = org.springframework.test.util.ReflectionTestUtils.invokeMethod(worker, "processQueue");
            completion.get(2, TimeUnit.SECONDS);
            assertEquals(Long.MAX_VALUE, org.springframework.test.util.ReflectionTestUtils.getField(decision, "wakeAtMs"));
            assertEquals(1, reads.get(), "one queue decision must use one immutable request snapshot");
            assertNull(context.snapshot().placement());
            assertSame(route, context.snapshot().route());
        } finally {
            committed.countDown();
            queue.lock.lock();
            try { assertSame(route, queue.updateQueue(PrefillState.QueueChange.REMOVE, route, 0L, System.currentTimeMillis())); }
            finally { queue.lock.unlock(); }
        }
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    void blockedExactHeadWaitsForItsCapacityEventWithoutPolling()
            throws Exception {
        FlexlbConfig config = singleConfig();
        PrefillEndpoint endpoint = stableEndpoint(stableStatus());
        EventDrivenBlock delivery = new EventDrivenBlock();
        WorkerBatcher runtime = runningRuntime(config, endpoint, delivery);
        RequestRoute head = item(
                config, endpoint, 1L, 50, System.currentTimeMillis());

        assertTrue(runtime.offer(head));
        await(delivery.firstAttempt);
        await(delivery.firstCapacity.subscribed);

        var firstWait = runtime.getLatestQueueWaitSnapshot();
        assertEquals(1, firstWait.get("queueDepth"));
        assertEquals(Map.of(50, 1), firstWait.get("priorityCounts"));
        assertThrows(UnsupportedOperationException.class,
                () -> firstWait.put("cause", "modified"));
        Map<?, ?> firstCounts = (Map<?, ?>) firstWait.get("priorityCounts");
        assertThrows(UnsupportedOperationException.class, firstCounts::clear);
        assertTrue(runtime.offer(item(config, endpoint, 2L, 30, System.currentTimeMillis())));

        TimeUnit.MILLISECONDS.sleep(100L);
        assertEquals(1, delivery.attempts.get(),
                "a capacity miss must not be polled");
        assertSame(head, WorkerBatcherTestSupport.capture(runtime).items().getFirst());

        delivery.firstCapacity.release();
        await(delivery.secondAttempt);
        await(delivery.parkedCapacity.subscribed);
        assertEquals(2, runtime.getLatestQueueWaitSnapshot().get("queueDepth"));
        assertEquals(Map.of(50, 1, 30, 1), runtime.getLatestQueueWaitSnapshot().get("priorityCounts"));
        assertEquals(1, firstWait.get("queueDepth"), "later decisions must not mutate an earlier snapshot");
        assertTrue(delivery.firstCapacity.listeners.isEmpty());
        assertEquals(1, delivery.parkedCapacity.listeners.size());

        TimeUnit.MILLISECONDS.sleep(100L);
        assertEquals(2, delivery.attempts.get(),
                "one capacity signal must trigger exactly one retry");
        assertSame(head, WorkerBatcherTestSupport.capture(runtime).items().getFirst());
        assertNull(runtime.stopAndAwait());
        assertTrue(delivery.parkedCapacity.listeners.isEmpty());
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    void failedListenerRemovalStillReleasesQueueLock() throws Exception {
        FlexlbConfig config = singleConfig();
        PrefillEndpoint endpoint = stableEndpoint(stableStatus());
        EventDrivenBlock delivery = new EventDrivenBlock();
        WorkerBatcher runtime = runningRuntime(config, endpoint, delivery);
        assertTrue(runtime.offer(item(
                config, endpoint, 12L, 50, System.currentTimeMillis())));
        await(delivery.firstCapacity.subscribed);
        delivery.firstCapacity.removeFailure =
                new IllegalStateException("listener removal failed");

        // Release on this thread; removal runs on the scheduling thread.
        delivery.firstCapacity.release();
        await(delivery.firstCapacity.removed);
        CompletableFuture.supplyAsync(() -> WorkerBatcherTestSupport.capture(runtime))
                .get(2, TimeUnit.SECONDS);
        assertNull(runtime.stopAndAwait());
        assertTrue(WorkerBatcherTestSupport.capture(runtime).items().isEmpty());
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    void stopDrainsQueueAndUnlocksAfterListenerRemovalFails() throws Exception {
        FlexlbConfig config = singleConfig();
        PrefillEndpoint endpoint = stableEndpoint(stableStatus());
        EventDrivenBlock delivery = new EventDrivenBlock();
        WorkerBatcher runtime = boundaryRuntime("stop-failure", endpoint, config, delivery,
                mock(AbstractRequestScheduler.class));
        runtime.start();
        try {
            assertTrue(runtime.offer(item(config, endpoint, 13L, 50, System.currentTimeMillis())));
            await(delivery.firstCapacity.subscribed);
            RuntimeException failure = new IllegalStateException("listener removal failed during stop");
            delivery.firstCapacity.removeFailure = failure;

            assertSame(failure, runtime.stopAndAwait());
            assertTrue(delivery.firstCapacity.listeners.isEmpty());
            assertTrue(CompletableFuture.supplyAsync(() -> WorkerBatcherTestSupport.capture(runtime))
                    .get(2, TimeUnit.SECONDS).items().isEmpty(), "stop must drain and release the queue lock");
            assertTrue(WorkerBatcherTestSupport.state(runtime).retireGenerationOwnership().ownedItems().isEmpty(),
                    "successful stop callbacks must acknowledge every detached owner");
            assertSame(failure, runtime.stopAndAwait(), "later callers share the completed cleanup result");
        } finally {
            runtime.stopAndAwait();
        }
    }

    @Test
    void stopStillTerminatesEveryRequestWhenCapacityNotificationFails() {
        FlexlbConfig config = singleConfig();
        PrefillEndpoint endpoint = stableEndpoint(stableStatus());
        AbstractRequestScheduler events = mock(AbstractRequestScheduler.class);
        WorkerBatcher runtime = boundaryRuntime("stop-notification-failure", endpoint, config,
                new EventDrivenBlock(), events);
        RequestRoute request = item(config, endpoint, 915L, 50, System.currentTimeMillis());
        SchedulerTestSupport.bindOwner(request.ctx(), events);
        var state = WorkerBatcherTestSupport.state(runtime);
        state.lock.lock();
        try { assertSame(request, state.updateQueue(PrefillState.QueueChange.ENQUEUE, request, 0L, System.currentTimeMillis())); }
        finally { state.lock.unlock(); }
        var failure = new IllegalStateException("capacity notification failed");
        doThrow(failure).when(endpoint).signal(PrefillEndpoint.Signal.PLACEMENT);
        assertSame(failure, runtime.stopAndAwait());
        verify(events).onQueueEvent(org.mockito.ArgumentMatchers.same(request),
                org.mockito.ArgumentMatchers.eq(AbstractRequestScheduler.QueueKind.REJECTED),
                org.mockito.ArgumentMatchers.isNull(), org.mockito.ArgumentMatchers.any());
        assertTrue(state.retireGenerationOwnership().ownedItems().isEmpty());
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = { false, true })
    void removedRequestIsNotifiedEvenWhenCapacitySignalFails(boolean expired) {
        FlexlbConfig config = singleConfig();
        SchedulingTestConfig.useBatchDispatcher(config);
        PrefillEndpoint endpoint = stableEndpoint(stableStatus());
        AbstractRequestScheduler events = mock(AbstractRequestScheduler.class);
        WorkerBatcher runtime = boundaryRuntime("removal-failure", endpoint, config,
                new EventDrivenBlock(), events);
        RequestRoute request = item(config, endpoint, 914L, 50, System.currentTimeMillis());
        SchedulerTestSupport.bindOwner(request.ctx(), events);
        var state = WorkerBatcherTestSupport.state(runtime);
        state.lock.lock();
        try {
            assertSame(request, state.updateQueue(PrefillState.QueueChange.ENQUEUE, request, 0L, System.currentTimeMillis()));
        } finally {
            state.lock.unlock();
        }
        RuntimeException signalFailure = new IllegalStateException("capacity signal failed");
        RuntimeException admissionFailure = new IllegalStateException("admission failed");
        doThrow(signalFailure).when(endpoint).signal(PrefillEndpoint.Signal.PLACEMENT);
        try {
            assertSame(signalFailure, assertThrows(IllegalStateException.class, () -> {
                if (expired) {
                    org.springframework.test.util.ReflectionTestUtils.invokeMethod(runtime, "removeAndNotify", request, null, false);
                } else {
                    org.springframework.test.util.ReflectionTestUtils.invokeMethod(runtime, "commitBoundary",
                            request, CapacityBoundary.failed(admissionFailure));
                }
            }));
            if (expired) { verify(events).onQueueEvent(request, AbstractRequestScheduler.QueueKind.EXPIRED, null, null); }
            else { verify(events).onDeliveryEvidence(org.mockito.ArgumentMatchers.isNull(),
                    org.mockito.ArgumentMatchers.argThat(e -> e instanceof ContextResult.PreparationFailure failed
                            && failed.route() == request && failed.detail().equals(admissionFailure.getMessage()))); }
            assertEquals(0, state.progress().queueDepth());
            assertTrue(state.retireGenerationOwnership().ownedItems().isEmpty());
        } finally {
            assertNull(runtime.stopAndAwait());
        }
    }

    @Test
    void failedAdmissionPublishesOneCapacityChangePerRemoval() {
        FlexlbConfig config = singleConfig();
        PrefillEndpoint endpoint = stableEndpoint(stableStatus());
        AbstractRequestScheduler events = mock(AbstractRequestScheduler.class);
        RuntimeException failure = new IllegalStateException("admission failed");
        BoundaryDelivery delivery = new BoundaryDelivery() {
            @Override
            CapacityBoundary selectBoundary(List<RequestRoute> candidates,
                                       PrefillTimePredictor.Evaluator evaluator,
                                       OptionalLong prediction) {
                return CapacityBoundary.failed(failure);
            }
        };
        WorkerBatcher runtime = boundaryRuntime("removal-notification", endpoint, config,
                delivery, events);
        RequestRoute request = item(config, endpoint, 916L, 50, System.currentTimeMillis());
        SchedulerTestSupport.bindOwner(request.ctx(), events);
        var state = WorkerBatcherTestSupport.state(runtime);
        state.lock.lock();
        try { assertSame(request, state.updateQueue(PrefillState.QueueChange.ENQUEUE, request, 0L, System.currentTimeMillis())); }
        finally { state.lock.unlock(); }
        try {
            org.springframework.test.util.ReflectionTestUtils.invokeMethod(runtime, "processQueue");
            verify(events).onDeliveryEvidence(org.mockito.ArgumentMatchers.isNull(),
                    org.mockito.ArgumentMatchers.argThat(e -> e instanceof ContextResult.PreparationFailure failed
                            && failed.route() == request && failed.detail().equals(failure.getMessage())));
            verify(endpoint, times(1)).signal(PrefillEndpoint.Signal.PLACEMENT);
            assertEquals(0, state.progress().queueDepth());
        } finally {
            assertNull(runtime.stopAndAwait());
        }
    }

    // Actual committed-before-handoff and executor rejection failures are covered with
    // real endpoint ledgers in QueuedBatchDeliveryTest, rather than a mocked PreparedGroup.

    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    void projectionCacheTracksCapacityBlockAndWake() {
        FlexlbConfig config = singleConfig();
        ProjectionCacheBlock delivery = new ProjectionCacheBlock();
        PrefillEndpoint endpoint = org.flexlb.balance.endpoint.EndpointTestSupport.unstartedPrefill(
                config, WorkerStatus.createDiscovered(RoleType.PREFILL, "test",
                        "127.0.0.1", 8080, 9090, null), SchedulerTestSupport.dispatcher(), mock(AbstractRequestScheduler.class));
        WorkerBatcher runtime = org.flexlb.balance.endpoint.EndpointTestSupport.batcher(endpoint);
        delivery.install(runtime);
        runtimes.add(runtime);
        runtime.start();
        RequestRoute head = item(
                config, endpoint, 11L, 50, System.currentTimeMillis());

        try {
            assertTrue(runtime.offer(head));
            await(delivery.firstPrepareEntered);
            assertNull(endpoint.captureRouteProjectionInputs()
                    .queue().admissionBlock(),
                    "cache is warmed before the delivery miss is published");

            delivery.allowFirstPrepare.countDown();
            await(delivery.firstCapacity.subscribed);
            assertNotNull(endpoint.captureRouteProjectionInputs()
                    .queue().admissionBlock(),
                    "publishing the block must invalidate the warm cache");

            delivery.firstCapacity.release();
            await(delivery.secondPrepareEntered);
            assertNull(endpoint.captureRouteProjectionInputs()
                    .queue().admissionBlock(),
                    "capacity wake must invalidate the cached block");
        } finally {
            delivery.allowFirstPrepare.countDown();
            delivery.allowSecondPrepare.countDown();
        }
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    void deliveryOnlyWaitKeepsBacklogSelectable() throws Exception {
        FlexlbConfig config = singleConfig();
        EventDrivenBlock delivery = new EventDrivenBlock(true);
        PrefillEndpoint endpoint = org.flexlb.balance.endpoint.EndpointTestSupport.unstartedPrefill(
                config, WorkerStatus.createDiscovered(RoleType.PREFILL, "test",
                        "127.0.0.1", 8080, 9090, null), SchedulerTestSupport.dispatcher(), mock(AbstractRequestScheduler.class));
        WorkerBatcher runtime = org.flexlb.balance.endpoint.EndpointTestSupport.batcher(endpoint);
        delivery.install(runtime);
        runtimes.add(runtime);
        runtime.start();
        long now = System.currentTimeMillis();
        RequestRoute head = item(config, endpoint, 21L, 50, now);
        assertTrue(runtime.offer(head));
        await(delivery.firstCapacity.subscribed);
        RouteProjection.Inputs inputs = endpoint.captureRouteProjectionInputs();
        assertNotNull(inputs.queue().admissionBlock());
        assertNull(inputs.queue().admissionBlock().semantics(), "batch capacity waits must not impose publication restrictions");
        RouteProjection.DeliveryProjection projection = RouteProjection.forDelivery(org.flexlb.config.DispatcherConfig.Type.BATCH);
        RouteProjection.Candidate candidate = RouteProjectionTestSupport.project(inputs, new RouteProjectionTestSupport.Probe(22L, 50, now + 1L, Long.MAX_VALUE, 10L, 0L, 0L), endpoint.predictor.evaluator(), projection, now);
        assertTrue(candidate.selectable(), "a delivery-only wait must leave incoming backlog selectable");
        RequestRoute backlog = item(config, endpoint, 22L, 50, now + 1L);
        assertTrue(runtime.offer(backlog));
        TimeUnit.MILLISECONDS.sleep(100L);
        assertEquals(1, delivery.attempts.get(), "publishing backlog must not retry the capacity-blocked head");
        assertEquals(List.of(head, backlog), WorkerBatcherTestSupport.capture(runtime).items());
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    void concurrentReadersShareOneVersionWithoutHoldingQueueLock() throws Exception {
        FlexlbConfig config = singleConfig();
        ProjectionCacheBlock delivery = new ProjectionCacheBlock();
        PrefillEndpoint endpoint = org.flexlb.balance.endpoint.EndpointTestSupport.unstartedPrefill(
                config, WorkerStatus.createDiscovered(RoleType.PREFILL, "test",
                        "127.0.0.1", 8080, 9090, null), SchedulerTestSupport.dispatcher(), mock(AbstractRequestScheduler.class));
        WorkerBatcher runtime = org.flexlb.balance.endpoint.EndpointTestSupport.batcher(endpoint);
        delivery.install(runtime);
        runtimes.add(runtime);
        runtime.start();
        RequestRoute head = spy(item(config, endpoint, 13L, 50, System.currentTimeMillis()));
        CountDownLatch materializing = new CountDownLatch(1);
        CountDownLatch finish = new CountDownLatch(1);
        doAnswer(invocation -> {
            if (Thread.currentThread().getName().startsWith("projection-reader")) {
                materializing.countDown();
                await(finish);
            }
            return invocation.callRealMethod();
        }).when(head).seqLen();
        try (var readers = java.util.concurrent.Executors.newFixedThreadPool(
                16, Thread.ofPlatform().name("projection-reader-", 0).factory())) {
            assertTrue(runtime.offer(head));
            await(delivery.firstPrepareEntered);
            var first = readers.submit(endpoint::captureRouteProjectionInputs);
            await(materializing);
            var started = new CountDownLatch(15);
            var followers = new java.util.ArrayList<java.util.concurrent.Future<RouteProjection.Inputs>>();
            for (int i = 0; i < 15; i++) {
                followers.add(readers.submit(() -> {
                    started.countDown();
                    return endpoint.captureRouteProjectionInputs();
                }));
            }
            await(started);
            // A normal queue reader must progress while materialization is blocked.
            CompletableFuture.supplyAsync(() -> WorkerBatcherTestSupport.capture(runtime)).get(2, TimeUnit.SECONDS);
            finish.countDown();
            var shared = first.get(2, TimeUnit.SECONDS);
            for (var follower : followers) {
                org.junit.jupiter.api.Assertions.assertSame(shared, follower.get(2, TimeUnit.SECONDS));
            }
        } finally {
            finish.countDown();
            delivery.allowFirstPrepare.countDown();
            delivery.allowSecondPrepare.countDown();
        }
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    void capacityBlockInvalidatesAnUnfinishedProjectionCapture() throws Exception {
        FlexlbConfig config = singleConfig();
        ProjectionCacheBlock delivery = new ProjectionCacheBlock();
        PrefillEndpoint endpoint = org.flexlb.balance.endpoint.EndpointTestSupport.unstartedPrefill(
                config, WorkerStatus.createDiscovered(RoleType.PREFILL, "test",
                        "127.0.0.1", 8080, 9090, null), SchedulerTestSupport.dispatcher(), mock(AbstractRequestScheduler.class));
        WorkerBatcher runtime = org.flexlb.balance.endpoint.EndpointTestSupport.batcher(endpoint);
        delivery.install(runtime);
        runtimes.add(runtime);
        runtime.start();
        RequestRoute head = spy(item(
                config, endpoint, 12L, 50, System.currentTimeMillis()));
        AtomicReference<Thread> capturingThread = new AtomicReference<>();
        CountDownLatch materializingSnapshot = new CountDownLatch(1);
        CountDownLatch finishSnapshot = new CountDownLatch(1);
        doAnswer(invocation -> {
            if (Thread.currentThread() == capturingThread.get()) {
                materializingSnapshot.countDown();
                await(finishSnapshot);
            }
            return invocation.callRealMethod();
        }).when(head).seqLen();

        try {
            assertTrue(runtime.offer(head));
            await(delivery.firstPrepareEntered);
            CompletableFuture<RouteProjection.Inputs> oldCapture =
                    CompletableFuture.supplyAsync(() -> {
                        capturingThread.set(Thread.currentThread());
                        return endpoint.captureRouteProjectionInputs();
                    });
            await(materializingSnapshot);
            delivery.allowFirstPrepare.countDown();
            await(delivery.firstCapacity.subscribed);
            finishSnapshot.countDown();
            assertNull(oldCapture.get(2, TimeUnit.SECONDS).queue().admissionBlock(),
                    "the snapshot was taken before the block was published");
            assertNotNull(endpoint.captureRouteProjectionInputs().queue().admissionBlock(),
                    "a late snapshot must not overwrite capacity-block invalidation");
        } finally {
            finishSnapshot.countDown();
            delivery.allowFirstPrepare.countDown();
            delivery.allowSecondPrepare.countDown();
        }
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    void removingCapturedMemberDuringPredictionInvalidatesWholeSelection()
            throws Exception {
        FlexlbConfig config = fixedConfig();
        CountDownLatch firstStatusRead = new CountDownLatch(1);
        CountDownLatch allowFirstStatusRead = new CountDownLatch(1);
        WorkerStatus status = gatedFirstStatusRead(
                firstStatusRead, allowFirstStatusRead);
        PrefillEndpoint endpoint = stableEndpoint(status);
        PredictionGate delivery = new PredictionGate();
        when(endpoint.predictor.evaluator().newBatchPrediction()).thenAnswer(call -> delivery.newBatchPrediction());
        var scheduler = SchedulerTestSupport.create(SchedulingTestConfig.configService(config),
                mock(DeliveryMetricsReporter.class), mock(org.flexlb.service.monitor.RequestSchedulerReporter.class),
                mock(RecentCacheKeyTraceReporter.class));
        var execution = spy(scheduler.runtime.deliveries());
        org.springframework.test.util.ReflectionTestUtils.setField(scheduler.runtime, "deliveries", execution);
        doAnswer(call -> {
            delivery.prepareCalls.incrementAndGet();
            return call.callRealMethod();
        }).when(execution).tryReserveDelivery();
        WorkerBatcher runtime = WorkerBatcherTestSupport.create("revoked-selection", endpoint, config,
                SchedulerTestSupport.dispatcher(), scheduler);
        runtimes.add(runtime);
        var ownership = WorkerBatcherTestSupport.state(runtime);
        doAnswer(call -> { runtime.signal(false); return null; }).when(endpoint).signal(PrefillEndpoint.Signal.INPUT);
        when(endpoint.release(org.mockito.ArgumentMatchers.isNull(), any(RequestRoute.class))).thenAnswer(call ->
                ownership.release(null, call.getArgument(1)).released());
        runtime.start();
        long now = System.currentTimeMillis();
        RequestRoute first = item(config, endpoint, 1L, 50, now);
        RequestRoute revoked = item(config, endpoint, 2L, 50, now + 1L);
        for (RequestRoute route : List.of(first, revoked)) {
            route.ctx().data().setGenerateInputPb(com.google.protobuf.ByteString.copyFromUtf8("input"));
            assertSame(RequestProtocolTestSupport.register(scheduler, route.ctx()), route.future());
            assertTrue(scheduler.requests.isCurrent(route.ctx()));
        }

        try {
            try (var admission = scheduler.claimAdmissionHandle(first.requestId(), first.future())) {
                assertNotNull(admission);
                assertEquals(org.flexlb.balance.PlacementResult.Status.SUCCESS,
                        scheduler.commitRoute(first, RequestProtocolTestSupport.publication(() -> runtime.offer(first))));
                admission.finish();
            }
            await(firstStatusRead);
            try (var admission = scheduler.claimAdmissionHandle(revoked.requestId(), revoked.future())) {
                assertNotNull(admission);
                assertEquals(org.flexlb.balance.PlacementResult.Status.SUCCESS,
                        scheduler.commitRoute(revoked, RequestProtocolTestSupport.publication(() -> runtime.offer(revoked))));
                admission.finish();
            }
            allowFirstStatusRead.countDown();
            await(delivery.groupPredictionEntered);

            assertTrue(runtime.control(revoked, WorkerBatcher.Control.REMOVE));
            delivery.allowGroupPrediction.countDown();
            await(delivery.nextDecisionStarted);

            assertEquals(0, delivery.prepareCalls.get(),
                    "a selection containing a revoked identity cannot prepare");
            List<RequestRoute> remaining =
                    WorkerBatcherTestSupport.capture(runtime).items();
            assertEquals(1, remaining.size());
            assertSame(first, remaining.getFirst());
        } finally {
            allowFirstStatusRead.countDown();
            delivery.allowGroupPrediction.countDown();
            assertNull(runtime.stopAndAwait());
            scheduler.runtime.shutdown();
        }
    }

    private WorkerBatcher runningRuntime(FlexlbConfig config, PrefillEndpoint endpoint, BoundaryDelivery boundary) {
        WorkerBatcher worker = boundaryRuntime("scheduling-test", endpoint, config, boundary,
                mock(AbstractRequestScheduler.class));
        runtimes.add(worker);
        worker.start();
        return worker;
    }

    private static WorkerBatcher boundaryRuntime(String key, PrefillEndpoint endpoint, FlexlbConfig config,
            BoundaryDelivery boundary, AbstractRequestScheduler scheduler) {
        WorkerBatcher worker = WorkerBatcherTestSupport.create(key, endpoint, config,
                SchedulerTestSupport.dispatcher(), scheduler);
        boundary.install(worker);
        return worker;
    }

    private WorkerBatcher runningRuntime(
            FlexlbConfig config,
            PrefillEndpoint endpoint,
            DeliveryDispatcher delivery) {
        WorkerBatcher runtime = WorkerBatcherTestSupport.create(
                "scheduling-test", endpoint, config, delivery,
                mock(AbstractRequestScheduler.class));
        runtimes.add(runtime);
        runtime.start();
        return runtime;
    }

    private static FlexlbConfig singleConfig() {
        FlexlbConfig config = SchedulingTestConfig.newConfig();
        SchedulingTestConfig.useFifoQueue(config);
        SchedulingTestConfig.useSingleDecision(config);
        SchedulingTestConfig.useBatchDispatcher(config);
        return config;
    }

    private static FlexlbConfig fixedConfig() {
        FlexlbConfig config = SchedulingTestConfig.newConfig();
        SchedulingTestConfig.useFifoQueue(config);
        DecisionPolicyConfig decision =
                SchedulingTestConfig.useFixedWindowDecision(config);
        decision.setMaxRequests(2);
        decision.setMaxCollectionWaitMs(60_000L);
        decision.setMaxPredictedExecutionMs(500L);
        SchedulingTestConfig.useBatchDispatcher(config);
        return config;
    }

    private static RequestRoute item(
            FlexlbConfig config,
            PrefillEndpoint endpoint,
            long requestId,
            int priority,
            long enqueuedAtMs) {
        Request request = Request.builder()
                .requestId(requestId)
                .rawPriority(priority)
                .seqLen(10L).build();
        RequestContext context = new RequestContext(config);
        context.data().setRequest(request);
        context.data().setSchedulingMetadata(
                SchedulingMetadata.explicit(priority, Long.MAX_VALUE));
        context.data().setFuture(new CompletableFuture<Response>());
        return SchedulingTestConfig.createRoute(context,
                null,
                null,
                null,
                endpoint,
                null,
                null,
                enqueuedAtMs);
    }

    private static PrefillEndpoint stableEndpoint(WorkerStatus status) {
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        var evaluator = mock(PrefillTimePredictor.Evaluator.class);
        when(evaluator.newBatchPrediction()).thenAnswer(call -> (PrefillTimePredictor.BatchPrediction) (seqLen, hitCache) -> 0.0);
        when(predictor.evaluator()).thenReturn(evaluator);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getStatus()).thenReturn(status);
        org.springframework.test.util.ReflectionTestUtils.setField(endpoint, "predictor", predictor);
        return endpoint;
    }

    private static WorkerStatus stableStatus() {
        WorkerStatus status = mock(WorkerStatus.class);
        when(status.committedEngineObservation()).thenReturn(capacity());
        return status;
    }

    private static WorkerStatus gatedFirstStatusRead(
            CountDownLatch entered,
            CountDownLatch proceed) {
        WorkerStatus status = mock(WorkerStatus.class);
        AtomicBoolean first = new AtomicBoolean(true);
        when(status.committedEngineObservation()).thenAnswer(ignored -> {
            if (first.compareAndSet(true, false)) {
                entered.countDown();
                await(proceed);
            }
            return capacity();
        });
        return status;
    }

    private static WorkerStatus.EngineObservation capacity() {
        return new WorkerStatus.EngineObservation(
                RoleType.PREFILL,
                null,
                0L,
                0L,
                Map.of(),
                0.0,
                0L,
                0L,
                0L,
                0L,
                0L,
                1_000_000L,
                0L,
                0L);
    }

    private static void await(CountDownLatch latch) {
        try {
            assertTrue(latch.await(5, TimeUnit.SECONDS),
                    "worker did not reach the expected boundary");
        } catch (InterruptedException interruption) {
            Thread.currentThread().interrupt();
            throw new AssertionError(
                    "interrupted while awaiting worker boundary",
                    interruption);
        }
    }

    /** Queue scheduling fixture: stubs Worker boundaries, never Dispatcher business APIs. */
    private abstract static class BoundaryDelivery {
        abstract CapacityBoundary selectBoundary(List<RequestRoute> candidates,
                PrefillTimePredictor.Evaluator evaluator, OptionalLong plannedPrediction);

        void install(WorkerBatcher worker) {
            doAnswer(call -> selectBoundary(call.getArgument(0), call.getArgument(2), call.getArgument(3)))
                    .when(worker).processSelection(any(), anyString(), any(), any());
            doAnswer(call -> capacityAvailable(call.getArgument(0), call.getArgument(1)))
                    .when(worker).capacityAvailable(any(), any());
            doAnswer(call -> watchCapacity(call.getArgument(0), call.getArgument(1), call.getArgument(2)))
                    .when(worker).watchCapacity(any(), any(), any());
        }
        private final java.util.Map<CapacityBoundary, TestAvailability> bindings = new java.util.IdentityHashMap<>();

        CapacityBoundary unavailable(TestAvailability availability) { return boundary(availability, false); }

        CapacityBoundary boundary(TestAvailability availability, boolean deliveryOnly) {
            var boundary = deliveryOnly ? CapacityBoundary.deliveryUnavailable(CapacityBoundary.Resource.PREFILL_BATCH)
                    : CapacityBoundary.unavailable(CapacityBoundary.Resource.DISPATCH_EXECUTOR,
                            new RouteProjection.AdmissionBlockSemantics("TEST_CAPACITY", RouteProjection.AfterProbeAdmission.BLOCKED,
                                    "TEST_CAPACITY", RoleType.PREFILL));
            bindings.put(boundary, availability);
            return boundary;
        }

        boolean capacityAvailable(RequestRoute item, CapacityBoundary boundary) {
            return bindings.get(boundary).isAvailable();
        }
        Runnable watchCapacity(RequestRoute item, CapacityBoundary boundary, Runnable wake) {
            var source = bindings.get(boundary);
            source.addListener(wake);
            return () -> source.removeListener(wake);
        }
    }

    private static final class EventDrivenBlock extends BoundaryDelivery {

        private final boolean deliveryOnly;

        private final TestAvailability firstCapacity = new TestAvailability();

        private final TestAvailability parkedCapacity = new TestAvailability();

        private final AtomicInteger attempts = new AtomicInteger();

        private final CountDownLatch firstAttempt = new CountDownLatch(1);

        private final CountDownLatch secondAttempt = new CountDownLatch(1);

        private EventDrivenBlock() {
            this(false);
        }

        private EventDrivenBlock(boolean deliveryOnly) {
            this.deliveryOnly = deliveryOnly;
        }

        private CapacityBoundary boundary(TestAvailability availability) {
            return boundary(availability, deliveryOnly);
        }

        @Override
        CapacityBoundary selectBoundary(
                List<RequestRoute> candidates,
                PrefillTimePredictor.Evaluator evaluator,
                OptionalLong plannedPrediction) {
            int attempt = attempts.incrementAndGet();
            if (attempt == 1) {
                firstAttempt.countDown();
                return boundary(firstCapacity);
            }
            secondAttempt.countDown();
            return boundary(parkedCapacity);
        }
    }

    private static final class ProjectionCacheBlock extends BoundaryDelivery {

        private final TestAvailability firstCapacity = new TestAvailability();

        private final TestAvailability parkedCapacity = new TestAvailability();

        private final CountDownLatch firstPrepareEntered = new CountDownLatch(1);

        private final CountDownLatch allowFirstPrepare = new CountDownLatch(1);

        private final CountDownLatch secondPrepareEntered = new CountDownLatch(1);

        private final CountDownLatch allowSecondPrepare = new CountDownLatch(1);

        private final AtomicInteger attempts = new AtomicInteger();

        @Override
        CapacityBoundary selectBoundary(
                List<RequestRoute> candidates,
                PrefillTimePredictor.Evaluator evaluator,
                OptionalLong plannedPrediction) {
            if (attempts.incrementAndGet() == 1) {
                firstPrepareEntered.countDown();
                await(allowFirstPrepare);
                return unavailable(firstCapacity);
            }
            secondPrepareEntered.countDown();
            await(allowSecondPrepare);
            return unavailable(parkedCapacity);
        }
    }

    private static final class PredictionGate {

        private final CountDownLatch groupPredictionEntered =
                new CountDownLatch(1);

        private final CountDownLatch allowGroupPrediction =
                new CountDownLatch(1);

        private final CountDownLatch nextDecisionStarted =
                new CountDownLatch(1);

        private final AtomicBoolean groupPredictionReturned =
                new AtomicBoolean();

        private final AtomicInteger prepareCalls = new AtomicInteger();

        PrefillTimePredictor.BatchPrediction newBatchPrediction() {
            var size = new AtomicInteger();
            return (seqLen, hitCache) -> {
                if (groupPredictionReturned.get()) {
                    nextDecisionStarted.countDown();
                }
                if (size.incrementAndGet() == 2) {
                    groupPredictionEntered.countDown();
                    await(allowGroupPrediction);
                    groupPredictionReturned.set(true);
                }
                return 100.0;
            };
        }
    }

    private static final class TestAvailability {

        private final AtomicBoolean available = new AtomicBoolean();

        private final CopyOnWriteArrayList<Runnable> listeners =
                new CopyOnWriteArrayList<>();

        private final CountDownLatch subscribed = new CountDownLatch(1);

        private final CountDownLatch removed = new CountDownLatch(1);

        private volatile RuntimeException removeFailure;

        public boolean isAvailable() {
            return available.get();
        }

        public void addListener(Runnable listener) {
            listeners.add(listener);
            subscribed.countDown();
        }

        public void removeListener(Runnable listener) {
            listeners.remove(listener);
            removed.countDown();
            if (removeFailure != null) {
                throw removeFailure;
            }
        }

        void release() {
            available.set(true);
            listeners.forEach(Runnable::run);
        }
    }
}
