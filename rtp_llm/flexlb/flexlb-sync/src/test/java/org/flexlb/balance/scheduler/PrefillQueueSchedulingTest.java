package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.DeliveryBlockReason;
import org.flexlb.balance.delivery.DeliveryGroup;
import org.flexlb.balance.delivery.DeliveryStrategy;
import org.flexlb.balance.delivery.DeliveryTransaction;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.estimate.BatchTtftEstimator;
import org.flexlb.balance.estimate.PrefillTtftEstimate;
import org.flexlb.balance.estimate.PrefillTtftEstimator;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.config.DecisionPolicyConfig;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.flexlb.enums.PrefillQueueEnqueueResult.ACCEPTED;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.when;

/** Concurrency contracts at the final, threaded Prefill scheduling boundary. */
class PrefillQueueSchedulingTest {

    private final List<PrefillQueue> runtimes = new ArrayList<>();

    @AfterEach
    void stopRuntimes() {
        for (PrefillQueue runtime : runtimes) {
            assertNull(runtime.stopAndAwait());
        }
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    void blockedExactHeadWaitsForItsCapacityEventWithoutPolling()
            throws Exception {
        FlexlbConfig config = singleConfig();
        PrefillEndpoint endpoint = stableEndpoint(stableStatus());
        EventDrivenBlock delivery = new EventDrivenBlock();
        PrefillQueue runtime = runningRuntime(config, endpoint, delivery);
        ScheduledRequest head = item(
                config, endpoint, 1L, 50, System.currentTimeMillis());

        assertEquals(ACCEPTED, runtime.offerWithResult(head));
        await(delivery.firstAttempt);
        await(delivery.firstCapacity.subscribed);

        var firstWait = runtime.waitDiagnostics();
        assertEquals(1, firstWait.get("queueDepth"));
        assertEquals(Map.of(50, 1), firstWait.get("priorityCounts"));
        assertThrows(UnsupportedOperationException.class,
                () -> firstWait.put("cause", "modified"));
        Map<?, ?> firstCounts = (Map<?, ?>) firstWait.get("priorityCounts");
        assertThrows(UnsupportedOperationException.class, firstCounts::clear);
        assertEquals(ACCEPTED, runtime.offerWithResult(item(config, endpoint, 2L, 30, System.currentTimeMillis())));

        TimeUnit.MILLISECONDS.sleep(100L);
        assertEquals(1, delivery.attempts.get(),
                "a capacity miss must not be polled");
        assertSame(head, runtime.captureQueueSnapshot().items().getFirst());

        delivery.firstCapacity.release();
        await(delivery.secondAttempt);
        await(delivery.parkedCapacity.subscribed);
        assertEquals(2, runtime.waitDiagnostics().get("queueDepth"));
        assertEquals(Map.of(50, 1, 30, 1), runtime.waitDiagnostics().get("priorityCounts"));
        assertEquals(1, firstWait.get("queueDepth"), "later decisions must not mutate an earlier snapshot");
        assertTrue(delivery.firstCapacity.listeners.isEmpty());
        assertEquals(1, delivery.parkedCapacity.listeners.size());

        TimeUnit.MILLISECONDS.sleep(100L);
        assertEquals(2, delivery.attempts.get(),
                "one capacity signal must trigger exactly one retry");
        assertSame(head, runtime.captureQueueSnapshot().items().getFirst());
        assertNull(runtime.stopAndAwait());
        assertTrue(delivery.parkedCapacity.listeners.isEmpty());
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    void failedListenerRemovalStillReleasesQueueLock() throws Exception {
        FlexlbConfig config = singleConfig();
        PrefillEndpoint endpoint = stableEndpoint(stableStatus());
        EventDrivenBlock delivery = new EventDrivenBlock();
        PrefillQueue runtime = runningRuntime(config, endpoint, delivery);
        assertEquals(ACCEPTED, runtime.offerWithResult(item(
                config, endpoint, 12L, 50, System.currentTimeMillis())));
        await(delivery.firstCapacity.subscribed);
        delivery.firstCapacity.removeFailure =
                new IllegalStateException("listener removal failed");

        // Release on this thread; removal runs on the scheduling thread.
        delivery.firstCapacity.release();
        await(delivery.firstCapacity.removed);
        CompletableFuture.supplyAsync(runtime::captureQueueSnapshot)
                .get(2, TimeUnit.SECONDS);
        assertNull(runtime.stopAndAwait());
        assertTrue(runtime.captureQueueSnapshot().items().isEmpty());
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    void projectionCacheTracksCapacityBlockAndWake() {
        FlexlbConfig config = singleConfig();
        PrefillEndpoint endpoint = stableEndpoint(stableStatus());
        EstimateCacheBlock delivery = new EstimateCacheBlock();
        PrefillQueue runtime = runningRuntime(config, endpoint, delivery);
        ScheduledRequest head = item(
                config, endpoint, 11L, 50, System.currentTimeMillis());

        try {
            assertEquals(ACCEPTED, runtime.offerWithResult(head));
            await(delivery.firstPrepareEntered);
            assertNull(runtime.captureLoadView()
                    .queue().admissionBlock(),
                    "cache is warmed before the delivery miss is published");

            delivery.allowFirstPrepare.countDown();
            await(delivery.firstCapacity.subscribed);
            assertNotNull(runtime.captureLoadView()
                    .queue().admissionBlock(),
                    "publishing the block must invalidate the warm cache");

            delivery.firstCapacity.release();
            await(delivery.secondPrepareEntered);
            assertNull(runtime.captureLoadView()
                    .queue().admissionBlock(),
                    "capacity wake must invalidate the cached block");
        } finally {
            delivery.allowFirstPrepare.countDown();
            delivery.allowSecondPrepare.countDown();
        }
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    void nonBatchQueueKeepsPrefillVisibleWhileItsHeadWaitsForDelivery() {
        FlexlbConfig config = singleConfig();
        config.setDispatcher(DispatcherConfig.nonBatch());
        PrefillEndpoint endpoint = stableEndpoint(stableStatus());
        when(endpoint.isEngineSubQueueMode()).thenReturn(true);
        when(endpoint.tryReserveEngineSubQueueSlot(any())).thenReturn(true);
        when(endpoint.reserveRouteOwnership(any(), anyLong())).thenReturn(
                new PrefillState.ReservationResult<>(PrefillState.CapacityStatus.ACQUIRED,
                        mock(PrefillState.RouteReservation.class)));
        EventDrivenBlock delivery = new EventDrivenBlock();
        PrefillQueue runtime = runningRuntime(config, endpoint, delivery);
        long now = System.currentTimeMillis();

        assertEquals(ACCEPTED, runtime.offerWithResult(item(config, endpoint, 31L, 50, now)));
        await(delivery.firstCapacity.subscribed);

        PrefillTtftEstimate.LoadView inputs = runtime.captureLoadView();
        assertEquals(1, inputs.queue().activeItems().size());
        assertNotNull(inputs.queue().admissionBlock());
        assertEquals(Long.MAX_VALUE, inputs.queue().constraints().batchKvCapacity());
    }

    @ParameterizedTest
    @CsvSource({"false,0,0,0", "false,0,1,0", "false,0,0,1", "true,0,0,0", "true,0,1,0", "true,0,0,1",
            "false,10,1,1", "true,10,1,1"})
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    void kvGateChecksEngineQueuesOnlyWhenCapacityIsInsufficient(boolean fixedWindow,
                                                                long available,
                                                                long running,
                                                                long waiting) throws Exception {
        FlexlbConfig config = singleConfig();
        if (fixedWindow) {
            DecisionPolicyConfig decision = SchedulingTestConfig.useFixedWindowDecision(config);
            decision.setMaxRequests(1);
            decision.setMaxCollectionWaitMs(0L);
            decision.setMaxPredictedExecutionMs(0L);
        }
        WorkerStatus status = stableStatus();
        when(status.committedEngineObservation()).thenReturn(new WorkerStatus.EngineObservation(
                RoleType.PREFILL, null, available, 100L, Map.of(), 0.0,
                0L, 0L, 0L, 0L, 1_000_000L, 1_000_000L, running, waiting));
        PrefillEndpoint endpoint = stableEndpoint(status);
        EventDrivenBlock delivery = new EventDrivenBlock();
        PrefillQueue runtime = runningRuntime(config, endpoint, delivery);
        assertEquals(ACCEPTED, runtime.offerWithResult(
                item(config, endpoint, 91L, 50, System.currentTimeMillis())));
        if (available < 10 && (running != 0 || waiting != 0)) {
            long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
            while (runtime.waitDiagnostics() == null && System.nanoTime() < deadline) {
                TimeUnit.MILLISECONDS.sleep(10L);
            }
            assertNotNull(runtime.waitDiagnostics());
            assertEquals(0, delivery.attempts.get());
            when(status.committedEngineObservation()).thenReturn(new WorkerStatus.EngineObservation(
                    RoleType.PREFILL, null, available, 100L, Map.of(), 0.0,
                    0L, 0L, 0L, 0L, 1_000_000L, 1_000_000L, 0L, 0L));
            runtime.signalSchedulingInputsChanged();
        }
        await(delivery.firstCapacity.subscribed);
        assertEquals(1, delivery.attempts.get());
    }

    @Test
    void batchSelectionDoesNotUseCurrentPrefillKvAsACandidateLimit() {
        FlexlbConfig config = singleConfig();
        WorkerStatus status = stableStatus();
        when(status.committedEngineObservation()).thenReturn(new WorkerStatus.EngineObservation(
                RoleType.PREFILL, null, 0L, 100L, Map.of(), 0.0,
                0L, 0L, 0L, 0L, 1_000_000L, 1_000_000L, 0L, 0L));
        PrefillEndpoint endpoint = stableEndpoint(status);
        PrefillQueue runtime = runningRuntime(config, endpoint, new EventDrivenBlock());

        assertEquals(Long.MAX_VALUE, runtime.captureLoadView()
                .queue().constraints().batchKvCapacity());
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    void deliveryOnlyWaitKeepsBacklogSelectable() throws Exception {
        FlexlbConfig config = singleConfig();
        PrefillEndpoint endpoint = stableEndpoint(stableStatus());
        EventDrivenBlock delivery = new EventDrivenBlock(true);
        PrefillQueue runtime = runningRuntime(config, endpoint, delivery);
        long now = System.currentTimeMillis();
        ScheduledRequest head = item(config, endpoint, 21L, 50, now);

        assertEquals(ACCEPTED, runtime.offerWithResult(head));
        await(delivery.firstCapacity.subscribed);
        PrefillTtftEstimate.LoadView inputs = runtime.captureLoadView();
        assertNotNull(inputs.queue().admissionBlock());
        assertNull(inputs.queue().admissionBlock().semantics(),
                "batch capacity waits must not impose publication restrictions");
        PrefillTtftEstimate.TtftEstimationRule rule = new BatchTtftEstimator().ttftEstimationRule();
        PrefillTtftEstimate.Candidate candidate = PrefillTtftEstimate.estimate(
                inputs, new PrefillTtftEstimate.Probe("22", 50, now + 1L, Long.MAX_VALUE, 10L, 0L, 0L),
                endpoint.getPredictor().evaluator(), rule, now);
        assertTrue(candidate.selectable(),
                "a delivery-only wait must leave incoming backlog selectable");

        ScheduledRequest backlog = item(config, endpoint, 22L, 50, now + 1L);
        assertEquals(ACCEPTED, runtime.offerWithResult(backlog));
        TimeUnit.MILLISECONDS.sleep(100L);
        assertEquals(1, delivery.attempts.get(),
                "publishing backlog must not retry the capacity-blocked head");
        assertEquals(List.of(head, backlog), runtime.captureQueueSnapshot().items());
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    void concurrentReadersShareOneVersionWithoutHoldingQueueLock() throws Exception {
        FlexlbConfig config = singleConfig();
        PrefillEndpoint endpoint = stableEndpoint(stableStatus());
        EstimateCacheBlock delivery = new EstimateCacheBlock();
        PrefillQueue runtime = runningRuntime(config, endpoint, delivery);
        ScheduledRequest head = spy(item(config, endpoint, 13L, 50, System.currentTimeMillis()));
        CountDownLatch materializing = new CountDownLatch(1);
        CountDownLatch finish = new CountDownLatch(1);
        doAnswer(invocation -> {
            if (Thread.currentThread().getName().startsWith("estimate-reader")) {
                materializing.countDown();
                await(finish);
            }
            return invocation.callRealMethod();
        }).when(head).seqLen();
        try (var readers = java.util.concurrent.Executors.newFixedThreadPool(
                16, Thread.ofPlatform().name("estimate-reader-", 0).factory())) {
            assertEquals(ACCEPTED, runtime.offerWithResult(head));
            await(delivery.firstPrepareEntered);
            var first = readers.submit(runtime::captureLoadView);
            await(materializing);
            var started = new CountDownLatch(15);
            var followers = new java.util.ArrayList<java.util.concurrent.Future<PrefillTtftEstimate.LoadView>>();
            for (int i = 0; i < 15; i++) {
                followers.add(readers.submit(() -> {
                    started.countDown();
                    return runtime.captureLoadView();
                }));
            }
            await(started);
            // A normal queue reader must progress while materialization is blocked.
            CompletableFuture.supplyAsync(runtime::captureQueueSnapshot).get(2, TimeUnit.SECONDS);
            finish.countDown();
            var shared = first.get(2, TimeUnit.SECONDS);
            for (var follower : followers) {
                assertSame(shared, follower.get(2, TimeUnit.SECONDS));
            }
        } finally {
            finish.countDown();
            delivery.allowFirstPrepare.countDown();
            delivery.allowSecondPrepare.countDown();
        }
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    void capacityBlockInvalidatesAnUnfinishedEstimateCapture() throws Exception {
        FlexlbConfig config = singleConfig();
        PrefillEndpoint endpoint = stableEndpoint(stableStatus());
        EstimateCacheBlock delivery = new EstimateCacheBlock();
        PrefillQueue runtime = runningRuntime(config, endpoint, delivery);
        ScheduledRequest head = spy(item(
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
            assertEquals(ACCEPTED, runtime.offerWithResult(head));
            await(delivery.firstPrepareEntered);
            CompletableFuture<PrefillTtftEstimate.LoadView> oldCapture =
                    CompletableFuture.supplyAsync(() -> {
                        capturingThread.set(Thread.currentThread());
                        return runtime.captureLoadView();
                    });
            await(materializingSnapshot);
            delivery.allowFirstPrepare.countDown();
            await(delivery.firstCapacity.subscribed);
            finishSnapshot.countDown();
            assertNull(oldCapture.get(2, TimeUnit.SECONDS).queue().admissionBlock(),
                    "the snapshot was taken before the block was published");
            assertNotNull(runtime.captureLoadView().queue().admissionBlock(),
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
        PrefillQueue runtime = runningRuntime(config, endpoint, delivery);
        long now = System.currentTimeMillis();
        ScheduledRequest first = item(config, endpoint, 1L, 50, now);
        ScheduledRequest revoked = item(config, endpoint, 2L, 50, now + 1L);

        try {
            assertEquals(ACCEPTED, runtime.offerWithResult(first));
            await(firstStatusRead);
            assertEquals(ACCEPTED, runtime.offerWithResult(revoked));
            allowFirstStatusRead.countDown();
            await(delivery.groupPredictionEntered);

            assertTrue(runtime.removeQueued(
                    revoked, "test revoke during prediction"));
            delivery.allowGroupPrediction.countDown();
            await(delivery.nextDecisionStarted);

            assertEquals(0, delivery.prepareCalls.get(),
                    "a selection containing a revoked identity cannot prepare");
            List<ScheduledRequest> remaining =
                    runtime.captureQueueSnapshot().items();
            assertEquals(1, remaining.size());
            assertSame(first, remaining.getFirst());
        } finally {
            allowFirstStatusRead.countDown();
            delivery.allowGroupPrediction.countDown();
        }
    }

    @ParameterizedTest
    @CsvSource({"NON_BATCH,SINGLE,1,0,single_request", "BATCH,SINGLE,1,0,single_request",
            "NON_BATCH,FIXED_WINDOW,2,60000,batch_full", "BATCH,FIXED_WINDOW,2,60000,batch_full",
            "NON_BATCH,FIXED_WINDOW,1,0,fixed_window_timeout", "BATCH,FIXED_WINDOW,1,0,fixed_window_timeout"})
    void committedGroupIsVisibleBeforeDeliveryPublication(String dispatcher,
                                                          String policy,
                                                          int count,
                                                          long windowMs,
                                                          String expectedReason)
            throws Exception {
        FlexlbConfig config = singleConfig();
        if (dispatcher.equals("NON_BATCH")) {
            config.setDispatcher(org.flexlb.config.DispatcherConfig.nonBatch());
        }
        if (policy.equals("FIXED_WINDOW")) {
            var decision = SchedulingTestConfig.useFixedWindowDecision(config);
            decision.setMaxRequests(2);
            decision.setMaxCollectionWaitMs(windowMs);
        }
        PrefillEndpoint endpoint = stableEndpoint(stableStatus());
        when(endpoint.reserveRouteOwnership(
                org.mockito.ArgumentMatchers.any(),
                org.mockito.ArgumentMatchers.anyLong()))
                .thenAnswer(ignored -> new org.flexlb.balance.endpoint.PrefillState.ReservationResult<>(
                        org.flexlb.balance.endpoint.PrefillState.CapacityStatus.ACQUIRED,
                        mock(org.flexlb.balance.endpoint.PrefillState.RouteReservation.class)));
        DeliveryStrategy delivery = mock(DeliveryStrategy.class);
        List<org.flexlb.dao.pv.DecisionGroup> published = new CopyOnWriteArrayList<>();
        when(delivery.prepare(org.mockito.ArgumentMatchers.any())).thenAnswer(invocation -> {
            List<ScheduledRequest> members = List.copyOf(((DeliveryGroup) invocation.getArgument(0)).requests());
                    var transaction = mock(DeliveryTransaction.class);
                    when(transaction.requests()).thenReturn(members);
                    org.mockito.Mockito.doAnswer(ignored -> {
                        for (ScheduledRequest member : members) {
                            published.add(member.ctx().getDecisionGroup());
                            member.future().complete(new Response());
                        }
                        return null;
                    })
                            .when(transaction)
                            .deliver(
                    org.mockito.ArgumentMatchers.anyString(),
                    org.mockito.ArgumentMatchers.anyInt(),
                    org.mockito.ArgumentMatchers.any());
            return transaction;
        });
        PrefillQueue runtime = runningRuntime(config, endpoint, delivery, mock(PrefillTtftEstimator.class));
        List<ScheduledRequest> requests = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            ScheduledRequest request = item(config, endpoint, 100L + i, 50, System.currentTimeMillis());
            requests.add(request);
            assertEquals(ACCEPTED, runtime.offerWithResult(request));
        }
        for (ScheduledRequest request : requests) {
            request.future().get(5, TimeUnit.SECONDS);
        }
        assertEquals(count, published.size());
        String id = published.getFirst().id();
        assertNotNull(id);
        for (var group : published) {
            assertEquals(id, group.id());
            assertEquals(count, group.committedSize());
            assertEquals(expectedReason, group.reason());
            assertEquals(policy, group.policy());
            assertEquals(dispatcher, group.dispatcher());
            assertEquals("10.0.0.1", group.worker());
            assertTrue(group.requestWaitMs() >= 0L);
        }
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    void prefillRetiredDuringCommitFailsHeadAsPreparationFailureAndKeepsLoopRunning() throws Exception {
        FlexlbConfig config = singleConfig();
        PrefillEndpoint endpoint = stableEndpoint(stableStatus());
        ScheduledRequest retiredItem = item(config, endpoint, 200L, 50, System.currentTimeMillis());
        ScheduledRequest laterItem = item(config, endpoint, 201L, 50, System.currentTimeMillis());
        DeliveryStrategy delivery = mock(DeliveryStrategy.class);
        when(delivery.prepare(org.mockito.ArgumentMatchers.any())).thenAnswer(invocation -> {
            List<ScheduledRequest> members = List.copyOf(((DeliveryGroup) invocation.getArgument(0)).requests());
                    var transaction = mock(DeliveryTransaction.class);
                    when(transaction.requests()).thenReturn(members);
            if (members.get(0) == retiredItem) {
                        when(transaction.commit()).thenThrow(new EndpointRetiredException("Prefill", retiredItem));
            } else {
                        org.mockito.Mockito.doAnswer(ignored -> {
                            members.forEach(member -> member.future().complete(new Response()));
                            return null;
                        })
                                .when(transaction)
                                .deliver(
                        org.mockito.ArgumentMatchers.anyString(),
                        org.mockito.ArgumentMatchers.anyInt(),
                        org.mockito.ArgumentMatchers.any());
            }
            return transaction;
        });
        EndpointEventHandler eventHandler = mock(EndpointEventHandler.class);
        PrefillQueue runtime = new PrefillQueue(
                "retired-commit", endpoint, () -> config, delivery, mock(PrefillTtftEstimator.class), eventHandler, new QueueCapacity());
        runtimes.add(runtime);
        runtime.start();

        assertEquals(ACCEPTED, runtime.offerWithResult(retiredItem));
        org.mockito.Mockito.verify(eventHandler, org.mockito.Mockito.timeout(5000)).onPreparedDeliveryFailure(
                org.mockito.ArgumentMatchers.eq(retiredItem),
                org.mockito.ArgumentMatchers.isA(EndpointRetiredException.class));

        assertEquals(ACCEPTED, runtime.offerWithResult(laterItem));
        laterItem.future().get(5, TimeUnit.SECONDS);
    }

    @Test
    void windowUpdateInvalidatesEstimateWithoutQueueMutation() {
        FlexlbConfig initial = fixedConfig();
        initial.decisionPolicy().setMaxCollectionWaitMs(100L);
        AtomicReference<FlexlbConfig> current = new AtomicReference<>(initial);
        PrefillQueue runtime = new PrefillQueue("hot-window-estimate", stableEndpoint(stableStatus()),
                current::get, mock(DeliveryStrategy.class), mock(PrefillTtftEstimator.class),
                mock(EndpointEventHandler.class), new QueueCapacity());
        runtimes.add(runtime);
        runtime.start();

        var before = runtime.captureLoadView();
        assertEquals(100L, before.queue().constraints().collectionWindowMs());
        assertSame(before, runtime.captureLoadView());
        FlexlbConfig updated = fixedConfig();
        updated.decisionPolicy().setMaxCollectionWaitMs(500L);
        current.set(updated);
        var after = runtime.captureLoadView();
        assertEquals(500L, after.queue().constraints().collectionWindowMs());
        assertSame(after, runtime.captureLoadView());
        assertEquals(before.queue().activeItems(), after.queue().activeItems());
        assertEquals(100L, before.queue().constraints().collectionWindowMs());

        current.set(initial);
        assertEquals(100L, runtime.captureLoadView().queue().constraints().collectionWindowMs());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    void existingRuntimeUsesUpdatedWindowOnNextDecision(boolean increaseWindow) throws Exception {
        FlexlbConfig initial = fixedConfig();
        initial.decisionPolicy().setMaxPredictedExecutionMs(null);
        initial.decisionPolicy().setMaxCollectionWaitMs(increaseWindow ? 0L : 60_000L);
        AtomicReference<FlexlbConfig> current = new AtomicReference<>(initial);
        PrefillEndpoint endpoint = stableEndpoint(stableStatus());
        EventDrivenBlock delivery = new EventDrivenBlock();
        PrefillQueue runtime = new PrefillQueue("hot-window-delivery", endpoint, current::get, delivery, delivery,
                mock(EndpointEventHandler.class), new QueueCapacity());
        runtimes.add(runtime);
        runtime.start();

        FlexlbConfig updated = fixedConfig();
        updated.decisionPolicy().setMaxPredictedExecutionMs(null);
        updated.decisionPolicy().setMaxCollectionWaitMs(increaseWindow ? 60_000L : 0L);
        current.set(updated);
        assertEquals(increaseWindow ? 60_000L : 0L,
                runtime.captureLoadView().queue().constraints().collectionWindowMs());
        assertEquals(ACCEPTED, runtime.offerWithResult(
                item(initial, endpoint, 77L, 50, System.currentTimeMillis())));
        if (increaseWindow) {
            current.set(initial);
            assertEquals(0L, runtime.captureLoadView().queue().constraints().collectionWindowMs());
            runtime.signalSchedulingInputsChanged();
        }
        await(delivery.firstAttempt);
    }

    private PrefillQueue runningRuntime(FlexlbConfig config, PrefillEndpoint endpoint, BoundaryDelivery delivery) {
        return runningRuntime(config, endpoint, delivery, delivery);
    }

    private PrefillQueue runningRuntime(FlexlbConfig config, PrefillEndpoint endpoint, DeliveryStrategy delivery,
                                         PrefillTtftEstimator estimator) {
        PrefillQueue runtime = new PrefillQueue(
                "scheduling-test", endpoint, () -> config, delivery, estimator,
                mock(EndpointEventHandler.class), new QueueCapacity());
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

    private static ScheduledRequest item(
            FlexlbConfig config,
            PrefillEndpoint endpoint,
            long requestId,
            int priority,
            long enqueuedAtMs) {
        Request request = new Request();
        request.setRequestId(Long.toString(requestId));
        request.setPriority(priority);
        request.setSeqLen(10L);
        BalanceContext context = new BalanceContext(config);
        context.setRequest(request);
        context.setSchedulingMetadata(
                SchedulingMetadata.explicit(priority, Long.MAX_VALUE));
        return new ScheduledRequest(
                context,
                new CompletableFuture<Response>(),
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
        when(status.getRole()).thenReturn(RoleType.PREFILL);
        when(endpoint.getPredictor()).thenReturn(predictor);
        return endpoint;
    }

    private static WorkerStatus stableStatus() {
        WorkerStatus status = mock(WorkerStatus.class);
        when(status.committedEngineObservation()).thenReturn(capacity());
        when(status.getMetricIpPort()).thenReturn("10.0.0.1");
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

    private static DeliveryBlockReason unavailable(
            DeliveryBlockReason.Availability availability) {
        return DeliveryBlockReason.unavailable(
                availability,
                new PrefillTtftEstimate.AdmissionBlockSemantics(
                        "TEST_CAPACITY",
                        PrefillTtftEstimate.AfterProbeAdmission.BLOCKED,
                        "TEST_CAPACITY",
                        RoleType.PREFILL));
    }

    private abstract static class BoundaryDelivery implements DeliveryStrategy, PrefillTtftEstimator {

        @Override
        public GroupPlanner.PrefixPrediction<ScheduledRequest> newGroupPredictor(
                PrefillTimePredictor.Evaluator evaluator) {
            return (added, items) -> 0.0;
        }

        @Override
        public PrefillTtftEstimate.TtftEstimationRule ttftEstimationRule() {
            return mock(PrefillTtftEstimate.TtftEstimationRule.class);
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

        private DeliveryBlockReason boundary(TestAvailability availability) {
            return deliveryOnly
                    ? DeliveryBlockReason.deliveryUnavailable(availability)
                    : unavailable(availability);
        }

        @Override
        public DeliveryTransaction prepare(DeliveryGroup group) {
            List<ScheduledRequest> candidates = group.requests();
            int attempt = attempts.incrementAndGet();
            if (attempt == 1) {
                firstAttempt.countDown();
                return PrefillQueueTestSupport.boundaryOnly(
                        candidates.getFirst(), boundary(firstCapacity));
            }
            secondAttempt.countDown();
            return PrefillQueueTestSupport.boundaryOnly(
                    candidates.getFirst(), boundary(parkedCapacity));
        }
    }

    private static final class EstimateCacheBlock extends BoundaryDelivery {

        private final TestAvailability firstCapacity = new TestAvailability();
        private final TestAvailability parkedCapacity = new TestAvailability();
        private final CountDownLatch firstPrepareEntered = new CountDownLatch(1);
        private final CountDownLatch allowFirstPrepare = new CountDownLatch(1);
        private final CountDownLatch secondPrepareEntered = new CountDownLatch(1);
        private final CountDownLatch allowSecondPrepare = new CountDownLatch(1);
        private final AtomicInteger attempts = new AtomicInteger();

        @Override
        public DeliveryTransaction prepare(DeliveryGroup group) {
            List<ScheduledRequest> candidates = group.requests();
            if (attempts.incrementAndGet() == 1) {
                firstPrepareEntered.countDown();
                await(allowFirstPrepare);
                return PrefillQueueTestSupport.boundaryOnly(
                        candidates.getFirst(), unavailable(firstCapacity));
            }
            secondPrepareEntered.countDown();
            await(allowSecondPrepare);
            return PrefillQueueTestSupport.boundaryOnly(
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
        public DeliveryTransaction prepare(DeliveryGroup group) {
            List<ScheduledRequest> candidates = group.requests();
            prepareCalls.incrementAndGet();
            return PrefillQueueTestSupport.boundaryOnly(
                    candidates.getFirst(),
                    unavailable(new TestAvailability()));
        }

        @Override
        public GroupPlanner.PrefixPrediction<ScheduledRequest> newGroupPredictor(
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

    private static final class TestAvailability
            implements DeliveryBlockReason.Availability {

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
