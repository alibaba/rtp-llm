package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.planner.GroupPlanner;
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

import static org.flexlb.balance.scheduler.SchedulingTestConfig.freezeInputs;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertFalse;
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
        WorkerBatcher runtime = runningRuntime(singleConfig(), mock(PrefillEndpoint.class), mock(DeliveryStrategy.class));
        assertThrows(IllegalStateException.class, runtime::projectionConstraintsLocked);
        var lock = WorkerBatcherTestSupport.state(runtime).ownershipLock();
        lock.lock();
        try {
            assertEquals(1, runtime.projectionConstraintsLocked().maxRequests());
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
        var scheduler = org.flexlb.balance.scheduler.SchedulerTestSupport.create(service, mock(org.flexlb.service.monitor.DeliveryMetricsReporter.class), mock(org.flexlb.service.monitor.RequestSchedulerReporter.class),
                mock(RecentCacheKeyTraceReporter.class));
        PrefillEndpoint endpoint = stableEndpoint(stableStatus());
        EventDrivenBlock delivery = new EventDrivenBlock();
        WorkerBatcher runtime = runningRuntime(config, endpoint, delivery);
        doAnswer(call -> {
            runtime.signalSchedulingInputsChanged();
            return null;
        }).when(endpoint).signalRouteReady();
        RequestContext context = item(config, endpoint, 901L, 50, System.currentTimeMillis()).ctx();
        context.setGenerateInputPb(com.google.protobuf.ByteString.copyFromUtf8("input"));
        var future = RequestProtocolTestSupport.register(scheduler, context);
        context.setFuture(future);
        var request = org.flexlb.balance.scheduler.SchedulingTestConfig.createRoute(freezeInputs(context), null, null, null, endpoint, null, null, System.currentTimeMillis());
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
                assertEquals(RequestContext.RequestStage.ROUTING, context.stage());
                assertEquals(0, delivery.attempts.get());
                // The wait is unbounded even when a request deadline elapses during publication.
                var decision = org.springframework.test.util.ReflectionTestUtils.invokeMethod(runtime, "processQueue");
                assertEquals(Long.MAX_VALUE, org.springframework.test.util.ReflectionTestUtils.getField(decision, "wakeAtMs"));
                allowCommit.countDown();
                assertEquals(org.flexlb.balance.PlacementResult.Status.SUCCESS, committed.get(2, TimeUnit.SECONDS));
                await(delivery.firstAttempt);
                assertEquals(RequestContext.RequestStage.READY_TO_DELIVER, context.stage());
                assertEquals(1, delivery.attempts.get());
            } finally {
                allowCommit.countDown();
            }
        } finally {
            runtime.stopAndAwait();
            RequestProtocolTestSupport.closeAdmissionAndAwaitMutations(scheduler);
            scheduler.closeOutstandingAndTerminalize();
            org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(scheduler).timer().close();
            org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(scheduler).closeRequestExecutors();
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
        WorkerBatcher runtime = WorkerBatcherTestSupport.create("stop-failure", endpoint, config, delivery,
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
        WorkerBatcher runtime = WorkerBatcherTestSupport.create("stop-notification-failure", endpoint, config,
                new EventDrivenBlock(), events);
        RequestRoute request = item(config, endpoint, 915L, 50, System.currentTimeMillis());
        request.ctx().bindScheduler(events);
        var state = WorkerBatcherTestSupport.state(runtime);
        state.ownershipLock().lock();
        try { assertTrue(state.enqueueActiveLocked(request, 0L)); }
        finally { state.ownershipLock().unlock(); }
        var failure = new IllegalStateException("capacity notification failed");
        doThrow(failure).when(endpoint).signalPlacementCapacityChanged();
        assertSame(failure, runtime.stopAndAwait());
        verify(events).onQueueOfferFailure(org.mockito.ArgumentMatchers.same(request), org.mockito.ArgumentMatchers.any());
        assertTrue(state.retireGenerationOwnership().ownedItems().isEmpty());
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = { false, true })
    void removedRequestIsNotifiedEvenWhenCapacitySignalFails(boolean expired) {
        FlexlbConfig config = singleConfig();
        SchedulingTestConfig.useBatchDispatcher(config);
        PrefillEndpoint endpoint = stableEndpoint(stableStatus());
        AbstractRequestScheduler events = mock(AbstractRequestScheduler.class);
        WorkerBatcher runtime = WorkerBatcherTestSupport.create("removal-failure", endpoint, config,
                new EventDrivenBlock(), events);
        RequestRoute request = item(config, endpoint, 914L, 50, System.currentTimeMillis());
        request.ctx().bindScheduler(events);
        var state = WorkerBatcherTestSupport.state(runtime);
        state.ownershipLock().lock();
        try {
            assertTrue(state.enqueueActiveLocked(request, 0L));
        } finally {
            state.ownershipLock().unlock();
        }
        RuntimeException signalFailure = new IllegalStateException("capacity signal failed");
        RuntimeException admissionFailure = new IllegalStateException("admission failed");
        doThrow(signalFailure).when(endpoint).signalPlacementCapacityChanged();
        try {
            assertSame(signalFailure, assertThrows(IllegalStateException.class, () -> {
                if (expired) {
                    org.springframework.test.util.ReflectionTestUtils.invokeMethod(runtime, "dropHead", request);
                } else {
                    org.springframework.test.util.ReflectionTestUtils.invokeMethod(runtime, "commitBoundary",
                            request, CapacityBoundary.failed(admissionFailure));
                }
            }));
            if (expired) { verify(events).onQueuedItemExpired(request); }
            else { verify(events).failDeliveryPreparation(request, admissionFailure); }
            assertEquals(0, state.queueDepth());
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
        DeliveryStrategy delivery = new BoundaryDelivery() {
            @Override
            public DeliveryTransaction prepare(List<RequestRoute> candidates,
                                       PrefillTimePredictor.Evaluator evaluator,
                                       OptionalLong prediction) {
                return WorkerBatcherTestSupport.boundaryOnly(candidates.get(0), CapacityBoundary.failed(failure));
            }
        };
        WorkerBatcher runtime = WorkerBatcherTestSupport.create("removal-notification", endpoint, config,
                delivery, events);
        RequestRoute request = item(config, endpoint, 916L, 50, System.currentTimeMillis());
        request.ctx().bindScheduler(events);
        var state = WorkerBatcherTestSupport.state(runtime);
        state.ownershipLock().lock();
        try { assertTrue(state.enqueueActiveLocked(request, 0L)); }
        finally { state.ownershipLock().unlock(); }
        try {
            org.springframework.test.util.ReflectionTestUtils.invokeMethod(runtime, "runOneCycle");
            verify(events).failDeliveryPreparation(request, failure);
            verify(endpoint, times(1)).signalPlacementCapacityChanged();
            assertEquals(0, state.queueDepth());
        } finally {
            assertNull(runtime.stopAndAwait());
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.CsvSource({ "true,false", "false,false", "true,true" })
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    void committedFailureAbortsOnceAndPreservesCause(boolean beforeHandoff, boolean materializationFailure) {
        FlexlbConfig config = singleConfig();
        PrefillEndpoint endpoint = stableEndpoint(stableStatus());
        DeliveryStrategy strategy = mock(DeliveryStrategy.class);
        DeliveryTransaction transaction = mock(DeliveryTransaction.class);
        WorkerBatcher runtime = runningRuntime(config, endpoint, strategy);
        RequestRoute selected = item(config, endpoint, 14L, 50, System.currentTimeMillis());
        // Park the real worker while the test drives its prepare/commit/handoff transaction.
        org.springframework.test.util.ReflectionTestUtils.setField(selected.ctx(), "stage", RequestContext.RequestStage.ROUTING);
        assertTrue(runtime.offer(selected));
        when(strategy.prepare(any(), any(), any())).thenReturn(transaction);
        when(transaction.items()).thenReturn(List.of(selected));
        var capture = mock(org.flexlb.balance.endpoint.PrefillState.WorkCapture.class);
        var committed = mock(org.flexlb.balance.endpoint.PrefillState.CommittedHandoff.class);
        when(committed.precedingWork()).thenReturn(capture);
        when(transaction.commitSelectionLocked(org.mockito.ArgumentMatchers.anyLong())).thenReturn(committed);
        RuntimeException failure = new IllegalStateException("committed delivery failure");
        RuntimeException abortFailure = new IllegalStateException("abort failure");
        when(capture.materialize()).thenAnswer(ignored -> {
            assertFalse(WorkerBatcherTestSupport.state(runtime).ownershipLock().isHeldByCurrentThread(),
                    "materialization must not hold the endpoint ownership lock");
            if (materializationFailure) { throw failure; }
            return new WorkSnapshot(0L, List.of(), List.of(), 0L);
        });
        if (beforeHandoff && !materializationFailure) {
            // The boundary is consumed after ownership commit; a failure must abort committed resources.
            when(transaction.blockedResult()).thenThrow(failure);
        } else {
            doThrow(failure).when(strategy).deliver(org.mockito.ArgumentMatchers.eq(transaction), anyString(), anyInt(), any(), any());
        }
        when(transaction.tryReclaimUnsent(false)).thenReturn(true);
        when(transaction.finishDelivery()).thenReturn(abortFailure);
        var evaluator = endpoint.getPredictor().evaluator();
        Runnable deliver = () -> org.springframework.test.util.ReflectionTestUtils.invokeMethod(runtime,
                "admitAndDeliverCapacityFeasiblePrefix", List.of(selected), "test",
                evaluator, OptionalLong.empty());

        if (beforeHandoff) {
            assertSame(failure, assertThrows(IllegalStateException.class, deliver::run));
        } else {
            deliver.run();
        }
        verify(transaction, times(1)).commitSelectionLocked(org.mockito.ArgumentMatchers.anyLong());
        verify(strategy).prepare(any(), org.mockito.ArgumentMatchers.same(evaluator), any());
        verify(strategy, beforeHandoff ? never() : times(1)).deliver(org.mockito.ArgumentMatchers.eq(transaction), anyString(), anyInt(), any(), org.mockito.ArgumentMatchers.same(evaluator));
        verify(transaction, times(1)).tryReclaimUnsent(false);
        verify(transaction, times(1)).finishDelivery();
        verify(transaction, times(1)).close();
        assertEquals(List.of(abortFailure), List.of(failure.getSuppressed()));
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    void projectionCacheTracksCapacityBlockAndWake() {
        FlexlbConfig config = singleConfig();
        ProjectionCacheBlock delivery = new ProjectionCacheBlock();
        PrefillEndpoint endpoint = org.flexlb.balance.endpoint.EndpointTestSupport.unstartedPrefill(
                config, WorkerStatus.createDiscovered(org.flexlb.dao.route.RoleType.PREFILL, "test",
                        "127.0.0.1", 8080, 9090, null), delivery, mock(AbstractRequestScheduler.class));
        WorkerBatcher runtime = org.flexlb.balance.endpoint.EndpointTestSupport.batcher(endpoint);
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
                config, WorkerStatus.createDiscovered(org.flexlb.dao.route.RoleType.PREFILL, "test",
                        "127.0.0.1", 8080, 9090, null), delivery, mock(AbstractRequestScheduler.class));
        WorkerBatcher runtime = org.flexlb.balance.endpoint.EndpointTestSupport.batcher(endpoint);
        runtimes.add(runtime);
        runtime.start();
        long now = System.currentTimeMillis();
        RequestRoute head = item(config, endpoint, 21L, 50, now);
        assertTrue(runtime.offer(head));
        await(delivery.firstCapacity.subscribed);
        RouteProjection.Inputs inputs = endpoint.captureRouteProjectionInputs();
        assertNotNull(inputs.queue().admissionBlock());
        assertNull(inputs.queue().admissionBlock().semantics(), "batch capacity waits must not impose publication restrictions");
        RouteProjection.DeliveryProjection projection = new BatchDeliveryStrategy(() -> {
            throw new AssertionError("projection cannot prepare delivery");
        }, () -> 0L, mock(DeliveryMetricsReporter.class)).projectionPolicy();
        RouteProjection.Candidate candidate = RouteProjectionTestSupport.project(inputs, new RouteProjectionTestSupport.Probe(22L, 50, now + 1L, Long.MAX_VALUE, 10L, 0L, 0L), endpoint.getPredictor().evaluator(), projection, now);
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
                config, WorkerStatus.createDiscovered(org.flexlb.dao.route.RoleType.PREFILL, "test",
                        "127.0.0.1", 8080, 9090, null), delivery, mock(AbstractRequestScheduler.class));
        WorkerBatcher runtime = org.flexlb.balance.endpoint.EndpointTestSupport.batcher(endpoint);
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
                config, WorkerStatus.createDiscovered(org.flexlb.dao.route.RoleType.PREFILL, "test",
                        "127.0.0.1", 8080, 9090, null), delivery, mock(AbstractRequestScheduler.class));
        WorkerBatcher runtime = org.flexlb.balance.endpoint.EndpointTestSupport.batcher(endpoint);
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
        WorkerBatcher runtime = runningRuntime(config, endpoint, delivery);
        long now = System.currentTimeMillis();
        RequestRoute first = item(config, endpoint, 1L, 50, now);
        RequestRoute revoked = item(config, endpoint, 2L, 50, now + 1L);

        try {
            assertTrue(runtime.offer(first));
            await(firstStatusRead);
            assertTrue(runtime.offer(revoked));
            allowFirstStatusRead.countDown();
            await(delivery.groupPredictionEntered);

            assertTrue(runtime.removeQueued(
                    revoked, "test revoke during prediction"));
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
        }
    }

    private WorkerBatcher runningRuntime(
            FlexlbConfig config,
            PrefillEndpoint endpoint,
            DeliveryStrategy delivery) {
        WorkerBatcher runtime = WorkerBatcherTestSupport.create(
                "scheduling-test", endpoint, config, delivery,
                mock(AbstractRequestScheduler.class));
        runtimes.add(runtime);
        runtime.start();
        return runtime;
    }

    private static FlexlbConfig singleConfig() {
        FlexlbConfig config = org.flexlb.balance.scheduler.SchedulingTestConfig.newConfig();
        SchedulingTestConfig.useFifoQueue(config);
        SchedulingTestConfig.useSingleDecision(config);
        SchedulingTestConfig.useBatchDispatcher(config);
        return config;
    }

    private static FlexlbConfig fixedConfig() {
        FlexlbConfig config = org.flexlb.balance.scheduler.SchedulingTestConfig.newConfig();
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
        Request request = new Request();
        request.setRequestId(requestId);
        request.setPriority(priority);
        request.setSeqLen(10L);
        RequestContext context = new RequestContext(config);
        context.setRequest(request);
        context.setSchedulingMetadata(
                SchedulingMetadata.explicit(priority, Long.MAX_VALUE));
        context.setFuture(new CompletableFuture<Response>());
        return org.flexlb.balance.scheduler.SchedulingTestConfig.createRoute(freezeInputs(context),
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
        when(predictor.evaluator())
                .thenReturn(mock(PrefillTimePredictor.Evaluator.class));
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getStatus()).thenReturn(status);
        when(endpoint.getPredictor()).thenReturn(predictor);
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

    private static CapacityBoundary unavailable(
            CapacityBoundary.Availability availability) {
        return CapacityBoundary.unavailable(
                availability,
                new RouteProjection.AdmissionBlockSemantics(
                        "TEST_CAPACITY",
                        RouteProjection.AfterProbeAdmission.BLOCKED,
                        "TEST_CAPACITY",
                        RoleType.PREFILL));
    }

    private abstract static class BoundaryDelivery implements DeliveryStrategy {
        @Override
        public void deliver(DeliveryTransaction transaction, String reason, int queueDepth,
                org.flexlb.balance.projection.WorkSnapshot precedingWork, PrefillTimePredictor.Evaluator evaluator) {
            throw new AssertionError("boundary-only strategy cannot deliver");
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
            return deliveryOnly
                    ? CapacityBoundary.deliveryUnavailable(availability)
                    : unavailable(availability);
        }

        @Override
        public DeliveryTransaction prepare(
                List<RequestRoute> candidates,
                PrefillTimePredictor.Evaluator evaluator,
                OptionalLong plannedPrediction) {
            int attempt = attempts.incrementAndGet();
            if (attempt == 1) {
                firstAttempt.countDown();
                return WorkerBatcherTestSupport.boundaryOnly(
                        candidates.getFirst(), boundary(firstCapacity));
            }
            secondAttempt.countDown();
            return WorkerBatcherTestSupport.boundaryOnly(
                    candidates.getFirst(), boundary(parkedCapacity));
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
        public DeliveryTransaction prepare(
                List<RequestRoute> candidates,
                PrefillTimePredictor.Evaluator evaluator,
                OptionalLong plannedPrediction) {
            if (attempts.incrementAndGet() == 1) {
                firstPrepareEntered.countDown();
                await(allowFirstPrepare);
                return WorkerBatcherTestSupport.boundaryOnly(
                        candidates.getFirst(), unavailable(firstCapacity));
            }
            secondPrepareEntered.countDown();
            await(allowSecondPrepare);
            return WorkerBatcherTestSupport.boundaryOnly(
                    candidates.getFirst(), unavailable(parkedCapacity));
        }
    }

    private static final class PredictionGate extends BoundaryDelivery {

        private final CountDownLatch groupPredictionEntered =
                new CountDownLatch(1);

        private final CountDownLatch allowGroupPrediction =
                new CountDownLatch(1);

        private final CountDownLatch nextDecisionStarted =
                new CountDownLatch(1);

        private final AtomicBoolean groupPredictionReturned =
                new AtomicBoolean();

        private final AtomicInteger prepareCalls = new AtomicInteger();

        @Override
        public DeliveryTransaction prepare(
                List<RequestRoute> candidates,
                PrefillTimePredictor.Evaluator evaluator,
                OptionalLong plannedPrediction) {
            prepareCalls.incrementAndGet();
            return WorkerBatcherTestSupport.boundaryOnly(
                    candidates.getFirst(),
                    unavailable(new TestAvailability()));
        }

        @Override
        public GroupPlanner.PrefixPrediction<RequestRoute> newGroupPredictor(
                PrefillTimePredictor.Evaluator evaluator) {
            return (added, items) -> {
                if (groupPredictionReturned.get()) {
                    nextDecisionStarted.countDown();
                }
                if (items.size() == 2) {
                    groupPredictionEntered.countDown();
                    await(allowGroupPrediction);
                    groupPredictionReturned.set(true);
                }
                return 100.0;
            };
        }
    }

    private static final class TestAvailability implements CapacityBoundary.Availability {

        private final AtomicBoolean available = new AtomicBoolean();

        private final CopyOnWriteArrayList<Runnable> listeners =
                new CopyOnWriteArrayList<>();

        private final CountDownLatch subscribed = new CountDownLatch(1);

        private final CountDownLatch removed = new CountDownLatch(1);

        private volatile RuntimeException removeFailure;

        @Override
        public boolean isAvailable() {
            return available.get();
        }

        @Override
        public void addListener(Runnable listener) {
            listeners.add(listener);
            subscribed.countDown();
        }

        @Override
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
