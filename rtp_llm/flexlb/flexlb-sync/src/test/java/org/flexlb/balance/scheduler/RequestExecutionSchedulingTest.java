package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.RequestTimeEstimation;
import org.flexlb.balance.scheduler.QueuedRequestScheduler.PrefillQueueScheduler;
import org.flexlb.balance.scheduler.RequestTimeEstimationTestSupport.Candidate;
import org.flexlb.config.DecisionPolicyConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.TheoryCacheHitReporter;
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

import static org.flexlb.balance.endpoint.EndpointTestSupport.runWithStateLock;
import static org.flexlb.balance.endpoint.EndpointTestSupport.withStateLock;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Concurrency contracts at the final, threaded Prefill scheduling boundary.
 */
class RequestExecutionSchedulingTest {

    private final List<PrefillQueueScheduler> runtimes = new ArrayList<>();

    @AfterEach
    void stopRuntimes() {
        for (PrefillQueueScheduler runtime : runtimes) {
            assertNull(runtime.stopAndAwait());
        }
    }

    @Test
    @Timeout(10)
    void firstEndpointNotificationDoesNotHoldSchedulerLockWhileBindingQueueStatistics()
            throws Exception {
        var config = fixedConfig();
        var runtime =
                SchedulerTestSupport.create(
                        SchedulingTestConfig.configService(config),
                        mock(DeliveryMetricsReporter.class),
                        mock(org.flexlb.service.monitor.RequestSchedulerReporter.class),
                        mock(TheoryCacheHitReporter.class));
        var scheduler = (QueuedRequestScheduler) SchedulerTestServices.scheduler(runtime);
        var status = stableStatus();
        when(status.getRole()).thenReturn(RoleType.PREFILL);
        when(status.getLogicalIpPort()).thenReturn("prefill:8080");
        when(status.getLogicalIpPort()).thenReturn("prefill:8080");
        when(status.topologySnapshot()).thenReturn(mock(WorkerStatus.TopologySnapshot.class));
        var endpoint =
                spy(
                        org.flexlb.balance.endpoint.EndpointTestSupport.realPrefill(
                                status,
                                config,
                                SchedulerTestSupport.dispatcher(
                                        SchedulerTestSupport.runtime(runtime)),
                                SchedulerTestSupport.repository(
                                        SchedulerTestServices.lifecycle(scheduler)),
                                mock(DeliveryMetricsReporter.class)));
        endpoint.enableQueue(
                QueueExecutionSettings.capture(
                        config, org.flexlb.balance.planner.BuiltinGroupPlanner.from(config)));
        var binding = new CountDownLatch(1);
        doAnswer(
                        call -> {
                            binding.countDown();
                            return call.callRealMethod();
                        })
                .when(endpoint)
                .observeQueueWait(any(), any());
        var guard = org.flexlb.balance.endpoint.EndpointTestSupport.stateLock(endpoint);
        try (var threads = java.util.concurrent.Executors.newFixedThreadPool(2)) {
            var notification =
                    withStateLock(
                            guard,
                            () -> {
                                var pending =
                                        CompletableFuture.runAsync(
                                                () -> scheduler.onEndpointChanged(endpoint),
                                                threads);
                                if (!com.google.common.util.concurrent.Uninterruptibles
                                        .awaitUninterruptibly(binding, 2, TimeUnit.SECONDS)) {
                                    pending.join();
                                    org.junit.jupiter.api.Assertions.fail(
                                            "queue statistics were not bound");
                                }
                                // A notification from the thread owning Endpoint must still enter
                                // Scheduler.
                                scheduler.onEndpointChanged(endpoint);
                                return pending;
                            });
            notification.get(2, TimeUnit.SECONDS);
        } finally {
            scheduler.close();
            endpoint.close();
            SchedulerTestSupport.closeRequestExecutors(SchedulerTestSupport.runtime(runtime));
        }
    }

    @Test
    void endpointChangeBeforeTaskSubmissionCannotStartTwoConcurrentDecisions() throws Exception {
        var config = fixedConfig();
        var owner =
                SchedulerTestSupport.create(
                        SchedulingTestConfig.configService(config),
                        mock(DeliveryMetricsReporter.class),
                        mock(org.flexlb.service.monitor.RequestSchedulerReporter.class),
                        mock(TheoryCacheHitReporter.class));
        var scheduler = (QueuedRequestScheduler) SchedulerTestServices.scheduler(owner);
        var status =
                WorkerStatus.createDiscovered(
                        RoleType.PREFILL, null, "127.0.0.1", 8080, 8081, null);
        var endpoint = stableEndpoint(status);
        when(endpoint.ipPort()).thenReturn(status.getLogicalIpPort());
        var execution = mock(PrefillQueueScheduler.class);
        ((java.util.Map<org.flexlb.balance.endpoint.PrefillEndpoint, PrefillQueueScheduler>)
                        org.springframework.test.util.ReflectionTestUtils.getField(
                                scheduler, "prefillQueueSchedulers"))
                .put(endpoint, execution);
        when(endpoint.queueEnabled()).thenReturn(true);
        when(execution.settings())
                .thenReturn(
                        QueueExecutionSettings.capture(
                                config,
                                org.flexlb.balance.planner.BuiltinGroupPlanner.from(config)));
        var entered = new CountDownLatch(1);
        var release = new CountDownLatch(1);
        var duplicate = new CountDownLatch(1);
        var attempts = new AtomicInteger();
        when(execution.captureQueue(org.mockito.ArgumentMatchers.anyInt()))
                .thenAnswer(
                        call -> {
                            if (attempts.incrementAndGet() > 1) duplicate.countDown();
                            entered.countDown();
                            assertTrue(release.await(3, TimeUnit.SECONDS));
                            return new PrefillEndpoint.QueueSnapshot(0L, 0L, List.of(), null);
                        });
        var changed = new AtomicBoolean();
        doAnswer(
                        call -> {
                            if (changed.compareAndSet(false, true))
                                scheduler.onEndpointChanged(endpoint);
                            return null;
                        })
                .when(execution)
                .observeWait(null);
        try {
            scheduler.onEndpointChanged(endpoint);
            scheduler.schedule();
            assertTrue(entered.await(2, TimeUnit.SECONDS));
            scheduler.schedule();
            org.junit.jupiter.api.Assertions.assertFalse(
                    duplicate.await(100, TimeUnit.MILLISECONDS),
                    "one endpoint must have at most one active scheduling decision");
            assertEquals(1, attempts.get());
        } finally {
            release.countDown();
            SchedulerTestSupport.shutdown(SchedulerTestSupport.runtime(owner));
        }
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    void endpointChangeWhileFinishingWaitIsNotLost() throws Exception {
        var config = fixedConfig();
        var owner =
                SchedulerTestSupport.create(
                        SchedulingTestConfig.configService(config),
                        mock(DeliveryMetricsReporter.class),
                        mock(org.flexlb.service.monitor.RequestSchedulerReporter.class),
                        mock(TheoryCacheHitReporter.class));
        var scheduler = (QueuedRequestScheduler) SchedulerTestServices.scheduler(owner);
        var status =
                WorkerStatus.createDiscovered(
                        RoleType.PREFILL, null, "127.0.0.1", 8080, 8081, null);
        var endpoint = stableEndpoint(status);
        when(endpoint.ipPort()).thenReturn(status.getLogicalIpPort());
        var execution = mock(PrefillQueueScheduler.class);
        ((java.util.Map<org.flexlb.balance.endpoint.PrefillEndpoint, PrefillQueueScheduler>)
                        org.springframework.test.util.ReflectionTestUtils.getField(
                                scheduler, "prefillQueueSchedulers"))
                .put(endpoint, execution);
        when(endpoint.queueEnabled()).thenReturn(true);
        when(execution.settings())
                .thenReturn(
                        QueueExecutionSettings.capture(
                                config,
                                org.flexlb.balance.planner.BuiltinGroupPlanner.from(config)));
        var retried = new CountDownLatch(1);
        var notified = new CountDownLatch(1);
        var attempts = new AtomicInteger();
        when(execution.captureQueue(org.mockito.ArgumentMatchers.anyInt()))
                .thenAnswer(
                        call -> {
                            if (attempts.incrementAndGet() == 2) retried.countDown();
                            return new PrefillEndpoint.QueueSnapshot(0L, 0L, List.of(), null);
                        });
        var checks = new AtomicInteger();
        when(execution.waitCurrent(any(), org.mockito.ArgumentMatchers.anyLong()))
                .thenAnswer(
                        call -> {
                            int check = checks.incrementAndGet();
                            if (check == 1) {
                                // Return the facts read before this event; finishing the wait must
                                // retain it.
                                scheduler.onEndpointChanged(endpoint);
                                notified.countDown();
                                return true;
                            }
                            return check != 2;
                        });
        try {
            scheduler.onEndpointChanged(endpoint);
            scheduler.schedule();
            assertTrue(notified.await(2, TimeUnit.SECONDS));
            long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(2);
            while (retried.getCount() != 0 && System.nanoTime() < deadline) {
                scheduler.schedule();
                Thread.sleep(1);
            }
            assertEquals(
                    0L,
                    retried.getCount(),
                    "the event must cause another decision without a new notification");
        } finally {
            SchedulerTestSupport.shutdown(SchedulerTestSupport.runtime(owner));
        }
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    void closeClearsWaitInstalledByAnInFlightDecision() throws Exception {
        var config = fixedConfig();
        var owner =
                SchedulerTestSupport.create(
                        SchedulingTestConfig.configService(config),
                        mock(DeliveryMetricsReporter.class),
                        mock(org.flexlb.service.monitor.RequestSchedulerReporter.class),
                        mock(TheoryCacheHitReporter.class));
        var scheduler = (QueuedRequestScheduler) SchedulerTestServices.scheduler(owner);
        var status =
                WorkerStatus.createDiscovered(
                        RoleType.PREFILL, null, "127.0.0.1", 8080, 8081, null);
        var endpoint = stableEndpoint(status);
        when(endpoint.ipPort()).thenReturn(status.getLogicalIpPort());
        var execution = mock(PrefillQueueScheduler.class);
        ((java.util.Map<org.flexlb.balance.endpoint.PrefillEndpoint, PrefillQueueScheduler>)
                        org.springframework.test.util.ReflectionTestUtils.getField(
                                scheduler, "prefillQueueSchedulers"))
                .put(endpoint, execution);
        when(endpoint.queueEnabled()).thenReturn(true);
        when(execution.settings())
                .thenReturn(
                        QueueExecutionSettings.capture(
                                config,
                                org.flexlb.balance.planner.BuiltinGroupPlanner.from(config)));
        when(execution.captureQueue(org.mockito.ArgumentMatchers.anyInt()))
                .thenReturn(new PrefillEndpoint.QueueSnapshot(0L, 0L, List.of(), null));
        when(execution.waitCurrent(any(), org.mockito.ArgumentMatchers.anyLong())).thenReturn(true);
        var installingWait = new CountDownLatch(1);
        var release = new CountDownLatch(1);
        doAnswer(
                        call -> {
                            if (call.getArgument(0) != null) {
                                installingWait.countDown();
                                assertTrue(release.await(3, TimeUnit.SECONDS));
                            }
                            return null;
                        })
                .when(execution)
                .observeWait(any());
        Thread closer = null;
        try {
            scheduler.onEndpointChanged(endpoint);
            scheduler.schedule();
            assertTrue(installingWait.await(2, TimeUnit.SECONDS));
            closer = Thread.ofVirtual().start(scheduler::close);
            var planners =
                    (java.util.concurrent.ExecutorService)
                            org.springframework.test.util.ReflectionTestUtils.getField(
                                    scheduler, "planners");
            long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(2);
            while (!planners.isShutdown() && System.nanoTime() < deadline) Thread.sleep(1);
            assertTrue(planners.isShutdown());
            release.countDown();
            closer.join(2000);
            org.junit.jupiter.api.Assertions.assertFalse(closer.isAlive());
            verify(execution, times(2)).observeWait(null);
        } finally {
            release.countDown();
            if (closer != null) closer.join(2000);
            SchedulerTestSupport.shutdown(SchedulerTestSupport.runtime(owner));
        }
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    void endpointWaitCheckDoesNotBlockSchedulingOtherRequests() throws Exception {
        var config = fixedConfig();
        var owner =
                SchedulerTestSupport.create(
                        SchedulingTestConfig.configService(config),
                        mock(DeliveryMetricsReporter.class),
                        mock(org.flexlb.service.monitor.RequestSchedulerReporter.class),
                        mock(TheoryCacheHitReporter.class));
        var scheduler = (QueuedRequestScheduler) SchedulerTestServices.scheduler(owner);
        var status =
                WorkerStatus.createDiscovered(
                        RoleType.PREFILL, null, "127.0.0.1", 8080, 8081, null);
        var endpoint = stableEndpoint(status);
        when(endpoint.ipPort()).thenReturn(status.getLogicalIpPort());
        var execution = mock(PrefillQueueScheduler.class);
        ((java.util.Map<org.flexlb.balance.endpoint.PrefillEndpoint, PrefillQueueScheduler>)
                        org.springframework.test.util.ReflectionTestUtils.getField(
                                scheduler, "prefillQueueSchedulers"))
                .put(endpoint, execution);
        when(endpoint.queueEnabled()).thenReturn(true);
        when(execution.settings())
                .thenReturn(
                        QueueExecutionSettings.capture(
                                config,
                                org.flexlb.balance.planner.BuiltinGroupPlanner.from(config)));
        when(execution.captureQueue(org.mockito.ArgumentMatchers.anyInt()))
                .thenReturn(new PrefillEndpoint.QueueSnapshot(0L, 0L, List.of(), null));
        when(execution.waitCurrent(any(), org.mockito.ArgumentMatchers.anyLong())).thenReturn(true);
        var planners =
                (java.util.concurrent.ThreadPoolExecutor)
                        org.springframework.test.util.ReflectionTestUtils.getField(
                                scheduler, "planners");
        try {
            scheduler.onEndpointChanged(endpoint);
            scheduler.schedule();
            long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(2);
            while (planners.getCompletedTaskCount() < 1 && System.nanoTime() < deadline)
                Thread.sleep(1);
            assertEquals(1L, planners.getCompletedTaskCount());
            var entered = new CountDownLatch(1);
            var release = new CountDownLatch(1);
            when(execution.waitCurrent(any(), org.mockito.ArgumentMatchers.anyLong()))
                    .thenAnswer(
                            call -> {
                                entered.countDown();
                                if (!release.await(5, TimeUnit.SECONDS))
                                    throw new AssertionError("endpoint check was not released");
                                return true;
                            });
            try {
                scheduler.onEndpointChanged(endpoint);
                CompletableFuture.runAsync(() -> scheduler.schedule()).get(1, TimeUnit.SECONDS);
                assertTrue(entered.await(1, TimeUnit.SECONDS));
            } finally {
                release.countDown();
            }
        } finally {
            SchedulerTestSupport.shutdown(SchedulerTestSupport.runtime(owner));
        }
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    void replacementSchedulerControlsWhenRequestsExecute() {
        var config = fixedConfig();
        var endpoint = stableEndpoint(stableStatus());
        var permitted = new AtomicBoolean();
        var waiting = new CountDownLatch(1);
        java.util.function.Function<PrefillQueueScheduler, PrefillQueueScheduler.Decision>
                scheduler = mock(java.util.function.Function.class);
        when(scheduler.apply(any()))
                .thenAnswer(
                        call -> {
                            PrefillQueueScheduler execution = call.getArgument(0);
                            if (permitted.get()) {
                                return RequestExecutionTestSupport.schedule(execution);
                            }
                            var snapshot = execution.captureQueue(1);
                            waiting.countDown();
                            return PrefillQueueScheduler.Decision.awaitingSchedulingChange(
                                    snapshot.head(),
                                    snapshot.queueVersion(),
                                    snapshot.schedulingInputVersion(),
                                    Long.MAX_VALUE,
                                    "custom scheduler waiting");
                        });
        var runtime =
                RequestExecutionTestSupport.create(
                        "custom-scheduler",
                        endpoint,
                        config,
                        SchedulerTestSupport.dispatcher(),
                        mock(AbstractRequestScheduler.class),
                        new org.flexlb.balance.planner.FirstRequestPlanner(),
                        scheduler::apply);
        var boundary = new EventDrivenBlock();
        boundary.install(runtime);
        runtimes.add(runtime);
        RequestExecutionTestSupport.start(runtime);
        assertTrue(
                RequestExecutionTestSupport.offer(
                        runtime, item(config, endpoint, 99101L, 50, System.currentTimeMillis())));
        await(waiting);
        verify(runtime, org.mockito.Mockito.never())
                .processSelection(
                        any(),
                        any(),
                        any(),
                        any(),
                        any(),
                        org.mockito.ArgumentMatchers.any(),
                        org.mockito.ArgumentMatchers.any());
        permitted.set(true);
        runtime.signal(false);
        await(boundary.firstAttempt);
        verify(scheduler, org.mockito.Mockito.atLeastOnce()).apply(runtime);
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    void customPlannerControlsTheGroupAndDispatchWithoutTheBuiltinWindow() {
        var config = fixedConfig();
        var endpoint = stableEndpoint(stableStatus());
        var planner = new org.flexlb.balance.planner.FirstRequestPlanner();
        var runtime =
                RequestExecutionTestSupport.create(
                        "custom-planner",
                        endpoint,
                        config,
                        SchedulerTestSupport.dispatcher(),
                        mock(AbstractRequestScheduler.class),
                        planner);
        var boundary = new EventDrivenBlock();
        boundary.install(runtime);
        runtimes.add(runtime);
        var first = item(config, endpoint, 101L, 50, System.currentTimeMillis());
        RequestExecutionTestSupport.start(runtime);
        var lock =
                org.flexlb.balance.endpoint.EndpointTestSupport.stateLock(
                        RequestExecutionTestSupport.state(runtime));
        runWithStateLock(
                lock,
                () -> {
                    assertTrue(RequestExecutionTestSupport.offer(runtime, first));
                    assertTrue(
                            RequestExecutionTestSupport.offer(
                                    runtime,
                                    item(config, endpoint, 102L, 50, System.currentTimeMillis())));
                });
        await(boundary.firstAttempt);
        verify(runtime)
                .processSelection(
                        eq(List.of(first)),
                        eq("first_request"),
                        any(),
                        any(),
                        any(),
                        org.mockito.ArgumentMatchers.any(),
                        org.mockito.ArgumentMatchers.any());
        assertTrue(planner.selections.get() > 0);
    }

    @Test
    void admissionBlockCanBeReadWithoutBorrowingTheEndpointLock() {
        PrefillQueueScheduler runtime =
                runningRuntime(
                        singleConfig(),
                        mock(PrefillEndpoint.class),
                        SchedulerTestSupport.dispatcher());
        assertNull(runtime.admissionBlock());
        var lock =
                org.flexlb.balance.endpoint.EndpointTestSupport.stateLock(
                        RequestExecutionTestSupport.state(runtime));
        runWithStateLock(
                lock,
                () -> {
                    assertNull(runtime.admissionBlock());
                });
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    void routeCommitWakesAWorkerThatAlreadyObservedRouting(boolean fixedWindow) throws Exception {
        FlexlbConfig config = fixedWindow ? fixedConfig() : singleConfig();
        if (fixedWindow) {
            SchedulingTestConfig.useFixedWindowDecision(config).setMaxRequests(1);
        }
        var service = mock(org.flexlb.config.ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        var scheduler =
                SchedulerTestSupport.create(
                        service,
                        mock(org.flexlb.service.monitor.DeliveryMetricsReporter.class),
                        mock(org.flexlb.service.monitor.RequestSchedulerReporter.class),
                        mock(TheoryCacheHitReporter.class));
        PrefillEndpoint endpoint = stableEndpoint(stableStatus());
        EventDrivenBlock delivery = new EventDrivenBlock();
        PrefillQueueScheduler runtime = runningRuntime(config, endpoint, delivery);
        doAnswer(
                        call -> {
                            runtime.signal(false);
                            return null;
                        })
                .when(endpoint)
                .signal(PrefillEndpoint.Signal.INPUT);
        RequestContext context = item(config, endpoint, 901L, 50, System.currentTimeMillis()).ctx();
        context.data().setGenerateInputPb(com.google.protobuf.ByteString.copyFromUtf8("input"));
        var future = RequestProtocolTestSupport.submitForManualPlacement(scheduler, context);
        context.data().setFuture(future);
        var request =
                SchedulingTestConfig.createRoute(
                        context,
                        null,
                        null,
                        null,
                        endpoint,
                        null,
                        null,
                        System.currentTimeMillis());
        CountDownLatch published = new CountDownLatch(1);
        CountDownLatch allowCommit = new CountDownLatch(1);
        try (var executor = java.util.concurrent.Executors.newSingleThreadExecutor();
                var routing = SchedulerTestSupport.claim(scheduler, "901", future);
                var admissionCompletion1 = RequestProtocolTestSupport.finishOnExit(routing)) {
            assertNotNull(routing);
            var committed =
                    executor.submit(
                            () ->
                                    RequestProtocolTestSupport.commitRoute(
                                            scheduler,
                                            request,
                                            RequestProtocolTestSupport.publication(
                                                    () -> {
                                                        assertTrue(
                                                                RequestExecutionTestSupport.offer(
                                                                        runtime, request));
                                                        published.countDown();
                                                        await(allowCommit);
                                                        return true;
                                                    })));
            try {
                await(published);
                RequestProtocolTestSupport.awaitCondition(
                        () ->
                                runtime.getLatestQueueWaitSnapshot()
                                        .containsValue("Route commit in progress"));
                assertEquals(
                        SchedulerTestSupport.Stage.ROUTING, SchedulerTestSupport.stage(context));
                assertEquals(0, delivery.attempts.get());
                // The wait is unbounded even when a request deadline elapses during publication.
                var decision = RequestExecutionTestSupport.schedule(runtime);
                assertEquals(
                        Long.MAX_VALUE,
                        org.springframework.test.util.ReflectionTestUtils.getField(
                                decision, "wakeAtMs"));
                allowCommit.countDown();
                assertEquals(
                        org.flexlb.balance.PlacementResult.Status.SUCCESS,
                        committed.get(2, TimeUnit.SECONDS));
                await(delivery.firstAttempt);
                assertEquals(
                        SchedulerTestSupport.Stage.READY_TO_DELIVER,
                        SchedulerTestSupport.stage(context));
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
        PrefillQueueScheduler worker =
                RequestExecutionTestSupport.create(
                        "commit-snapshot-race",
                        endpoint,
                        config,
                        SchedulerTestSupport.dispatcher(),
                        mock(AbstractRequestScheduler.class));
        RouteExecution route = spy(item(config, endpoint, 902L, 50, System.currentTimeMillis()));
        RequestContext context = route.ctx();
        context.register("902", System.currentTimeMillis(), Long.MAX_VALUE, 60_000L);
        var operation = context.beginPlacement(null);
        context.bindPlacement(operation, route, ContextResult.PublicationResult.PREPARED);
        CountDownLatch sampled = new CountDownLatch(1);
        CountDownLatch committed = new CountDownLatch(1);
        AtomicInteger reads = new AtomicInteger();
        // This probe releases the real Context monitor after sampling, so an actual
        // placement completion can run on another thread before Worker uses the sample.
        RequestContext observation =
                mock(RequestContext.class, org.mockito.AdditionalAnswers.delegatesTo(context));
        doAnswer(
                        call -> {
                            var state = context.snapshot();
                            if (reads.incrementAndGet() == 1) {
                                assertNotNull(state.placement());
                                sampled.countDown();
                                await(committed);
                            }
                            return state;
                        })
                .when(observation)
                .snapshot();
        org.mockito.Mockito.doReturn(observation).when(route).ctx();
        var queue = RequestExecutionTestSupport.state(worker);
        runWithStateLock(
                org.flexlb.balance.endpoint.EndpointTestSupport.stateLock(queue),
                () -> {
                    assertSame(
                            route,
                            org.flexlb.balance.endpoint.EndpointTestSupport.enqueue(queue, route));
                });
        try (var executor = java.util.concurrent.Executors.newSingleThreadExecutor()) {
            var completion =
                    executor.submit(
                            () -> {
                                await(sampled);
                                try {
                                    assertEquals(
                                            RequestChangeAssertions.UpdateKind.APPLIED,
                                            org.flexlb.balance.scheduler.RequestChangeAssertions
                                                    .record(
                                                            context,
                                                            observedRequest ->
                                                                    observedRequest
                                                                            .completePlacement(
                                                                                    operation,
                                                                                    ContextResult
                                                                                            .PublicationResult
                                                                                            .PUBLISHED,
                                                                                    System
                                                                                            .currentTimeMillis()))
                                                    .kind());
                                } finally {
                                    committed.countDown();
                                }
                            });
            var decision = RequestExecutionTestSupport.schedule(worker);
            completion.get(2, TimeUnit.SECONDS);
            assertEquals(
                    Long.MAX_VALUE,
                    org.springframework.test.util.ReflectionTestUtils.getField(
                            decision, "wakeAtMs"));
            assertEquals(
                    1, reads.get(), "one queue decision must use one immutable request snapshot");
            assertNull(context.snapshot().placement());
            assertSame(route, context.snapshot().route());
        } finally {
            committed.countDown();
            runWithStateLock(
                    org.flexlb.balance.endpoint.EndpointTestSupport.stateLock(queue),
                    () -> {
                        assertSame(
                                route,
                                org.flexlb.balance.endpoint.EndpointTestSupport.removeQueued(
                                        queue, route));
                    });
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.CsvSource({
        "PREFILL_REQUEST,Prefill request capacity exhausted",
        "PREFILL_BATCH,Prefill batch capacity exhausted",
        "DISPATCH_EXECUTOR,DELIVERY_CAPACITY_BATCH_ADMISSION",
        "DECODE_ENGINE,DELIVERY_CAPACITY_DECODE_ENGINE"
    })
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    void blockedExactHeadWaitsForItsCapacityEventWithoutPolling(
            CapacityBoundary.Resource resource, String cause) throws Exception {
        FlexlbConfig config = singleConfig();
        if (resource == CapacityBoundary.Resource.PREFILL_REQUEST) {
            SchedulingTestConfig.useNonBatchDispatcher(config);
        }
        PrefillEndpoint endpoint = stableEndpoint(stableStatus());
        EventDrivenBlock delivery = new EventDrivenBlock(resource);
        PrefillQueueScheduler runtime = runningRuntime(config, endpoint, delivery);
        RouteExecution head = item(config, endpoint, 1L, 50, System.currentTimeMillis());

        assertTrue(RequestExecutionTestSupport.offer(runtime, head));
        await(delivery.firstAttempt);
        await(delivery.firstCapacity.subscribed);

        var firstWait = runtime.getLatestQueueWaitSnapshot();
        assertEquals(cause, firstWait.get("cause"));
        assertEquals(1, firstWait.get("queueDepth"));
        assertEquals(Map.of(50, 1), firstWait.get("priorityCounts"));
        assertThrows(UnsupportedOperationException.class, () -> firstWait.put("cause", "modified"));
        Map<?, ?> firstCounts = (Map<?, ?>) firstWait.get("priorityCounts");
        assertThrows(UnsupportedOperationException.class, firstCounts::clear);
        assertTrue(
                RequestExecutionTestSupport.offer(
                        runtime, item(config, endpoint, 2L, 30, System.currentTimeMillis())));

        TimeUnit.MILLISECONDS.sleep(100L);
        assertEquals(1, delivery.attempts.get(), "a capacity miss must not be polled");
        assertSame(head, RequestExecutionTestSupport.capture(runtime).items().getFirst());

        delivery.firstCapacity.release();
        await(delivery.secondAttempt);
        await(delivery.parkedCapacity.subscribed);
        assertEquals(2, runtime.getLatestQueueWaitSnapshot().get("queueDepth"));
        assertEquals(
                Map.of(50, 1, 30, 1), runtime.getLatestQueueWaitSnapshot().get("priorityCounts"));
        assertEquals(
                1,
                firstWait.get("queueDepth"),
                "later decisions must not mutate an earlier snapshot");
        assertTrue(delivery.firstCapacity.listeners.isEmpty());
        assertEquals(1, delivery.parkedCapacity.listeners.size());

        TimeUnit.MILLISECONDS.sleep(100L);
        assertEquals(
                2, delivery.attempts.get(), "one capacity signal must trigger exactly one retry");
        assertSame(head, RequestExecutionTestSupport.capture(runtime).items().getFirst());
        assertNull(runtime.stopAndAwait());
        assertTrue(delivery.parkedCapacity.listeners.isEmpty());
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    void slowCapacitySubscriptionDoesNotHoldEndpointLockAndStopRemovesItsListener()
            throws Exception {
        FlexlbConfig config = singleConfig();
        PrefillEndpoint endpoint = stableEndpoint(stableStatus());
        EventDrivenBlock delivery = new EventDrivenBlock();
        PrefillQueueScheduler runtime =
                boundaryRuntime(
                        "subscription-stop",
                        endpoint,
                        config,
                        delivery,
                        mock(AbstractRequestScheduler.class));
        CountDownLatch subscribing = new CountDownLatch(1);
        CountDownLatch finishSubscription = new CountDownLatch(1);
        doAnswer(
                        call -> {
                            assertFalse(
                                    org.flexlb.balance.endpoint.EndpointTestSupport.stateLock(
                                                    endpoint)
                                            .isHeldByCurrentThread());
                            Runnable remove =
                                    delivery.watchCapacity(
                                            call.getArgument(0),
                                            call.getArgument(1),
                                            call.getArgument(2));
                            subscribing.countDown();
                            await(finishSubscription);
                            return remove;
                        })
                .when(runtime)
                .watchCapacity(any(), any(), any());
        RequestExecutionTestSupport.start(runtime);
        try {
            RouteExecution head = item(config, endpoint, 71L, 50, System.currentTimeMillis());
            assertTrue(RequestExecutionTestSupport.offer(runtime, head));
            await(subscribing);
            assertEquals(
                    1,
                    CompletableFuture.supplyAsync(() -> endpoint.captureQueue(10))
                            .get(2, TimeUnit.SECONDS)
                            .items()
                            .size());
            var stopping = CompletableFuture.supplyAsync(runtime::stopAndAwait);
            finishSubscription.countDown();
            assertNull(stopping.get(2, TimeUnit.SECONDS));
            assertTrue(delivery.firstCapacity.listeners.isEmpty());
            runtime.observeWait(
                    PrefillQueueScheduler.Decision.capacityBlocked(
                            head,
                            CapacityBoundary.unavailable(
                                    CapacityBoundary.Resource.DISPATCH_EXECUTOR),
                            0L));
            assertTrue(delivery.firstCapacity.listeners.isEmpty());
            verify(runtime, times(1)).watchCapacity(any(), any(), any());
            assertTrue(endpoint.captureQueue(10).items().isEmpty());
        } finally {
            finishSubscription.countDown();
            runtime.stopAndAwait();
        }
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    void capacityReleaseDuringSubscriptionIsNotLost() throws Exception {
        FlexlbConfig config = singleConfig();
        PrefillEndpoint endpoint = stableEndpoint(stableStatus());
        EventDrivenBlock delivery = new EventDrivenBlock();
        PrefillQueueScheduler runtime =
                boundaryRuntime(
                        "subscription-wakeup",
                        endpoint,
                        config,
                        delivery,
                        mock(AbstractRequestScheduler.class));
        AtomicBoolean first = new AtomicBoolean(true);
        doAnswer(
                        call -> {
                            assertFalse(
                                    org.flexlb.balance.endpoint.EndpointTestSupport.stateLock(
                                                    endpoint)
                                            .isHeldByCurrentThread());
                            Runnable remove =
                                    delivery.watchCapacity(
                                            call.getArgument(0),
                                            call.getArgument(1),
                                            call.getArgument(2));
                            if (first.compareAndSet(true, false)) delivery.firstCapacity.release();
                            return remove;
                        })
                .when(runtime)
                .watchCapacity(any(), any(), any());
        RequestExecutionTestSupport.start(runtime);
        try {
            assertTrue(
                    RequestExecutionTestSupport.offer(
                            runtime, item(config, endpoint, 72L, 50, System.currentTimeMillis())));
            await(delivery.secondAttempt);
            await(delivery.parkedCapacity.subscribed);
            assertEquals(2, delivery.attempts.get());
            assertTrue(delivery.firstCapacity.listeners.isEmpty());
            assertEquals(1, delivery.parkedCapacity.listeners.size());
        } finally {
            runtime.stopAndAwait();
        }
        assertTrue(delivery.parkedCapacity.listeners.isEmpty());
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    void failedListenerRemovalStillReleasesQueueLock() throws Exception {
        FlexlbConfig config = singleConfig();
        PrefillEndpoint endpoint = stableEndpoint(stableStatus());
        EventDrivenBlock delivery = new EventDrivenBlock();
        PrefillQueueScheduler runtime = runningRuntime(config, endpoint, delivery);
        assertTrue(
                RequestExecutionTestSupport.offer(
                        runtime, item(config, endpoint, 12L, 50, System.currentTimeMillis())));
        await(delivery.firstCapacity.subscribed);
        delivery.firstCapacity.removeFailure = new IllegalStateException("listener removal failed");

        // Release on this thread; removal runs on the scheduling thread.
        delivery.firstCapacity.release();
        await(delivery.firstCapacity.removed);
        CompletableFuture.supplyAsync(() -> RequestExecutionTestSupport.capture(runtime))
                .get(2, TimeUnit.SECONDS);
        assertNull(runtime.stopAndAwait());
        assertTrue(RequestExecutionTestSupport.capture(runtime).items().isEmpty());
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    void stopDrainsQueueAndUnlocksAfterListenerRemovalFails() throws Exception {
        FlexlbConfig config = singleConfig();
        PrefillEndpoint endpoint = stableEndpoint(stableStatus());
        EventDrivenBlock delivery = new EventDrivenBlock();
        PrefillQueueScheduler runtime =
                boundaryRuntime(
                        "stop-failure",
                        endpoint,
                        config,
                        delivery,
                        mock(AbstractRequestScheduler.class));
        RequestExecutionTestSupport.start(runtime);
        try {
            assertTrue(
                    RequestExecutionTestSupport.offer(
                            runtime, item(config, endpoint, 13L, 50, System.currentTimeMillis())));
            await(delivery.firstCapacity.subscribed);
            RuntimeException failure =
                    new IllegalStateException("listener removal failed during stop");
            delivery.firstCapacity.removeFailure = failure;

            assertSame(failure, runtime.stopAndAwait());
            assertTrue(delivery.firstCapacity.listeners.isEmpty());
            assertTrue(
                    CompletableFuture.supplyAsync(
                                    () -> RequestExecutionTestSupport.capture(runtime))
                            .get(2, TimeUnit.SECONDS)
                            .items()
                            .isEmpty(),
                    "stop must drain and release the queue lock");
            assertEquals(
                    1,
                    org.flexlb.balance.endpoint.EndpointTestSupport.retireResources(
                                    RequestExecutionTestSupport.state(runtime))
                            .ownedItems()
                            .size(),
                    "queue drain cannot settle resources on behalf of the mocked lifecycle");
            assertSame(
                    failure,
                    runtime.stopAndAwait(),
                    "later callers share the completed cleanup result");
        } finally {
            runtime.stopAndAwait();
        }
    }

    @Test
    void stopStillTerminatesEveryRequestWhenCapacityNotificationFails() {
        FlexlbConfig config = singleConfig();
        PrefillEndpoint endpoint = stableEndpoint(stableStatus());
        AbstractRequestScheduler events = mock(AbstractRequestScheduler.class);
        PrefillQueueScheduler runtime =
                boundaryRuntime(
                        "stop-notification-failure",
                        endpoint,
                        config,
                        new EventDrivenBlock(),
                        events);
        RouteExecution request = item(config, endpoint, 915L, 50, System.currentTimeMillis());
        SchedulerTestSupport.bindOwner(request.ctx(), events);
        var state = RequestExecutionTestSupport.state(runtime);
        runWithStateLock(
                org.flexlb.balance.endpoint.EndpointTestSupport.stateLock(state),
                () -> {
                    assertSame(
                            request,
                            org.flexlb.balance.endpoint.EndpointTestSupport.enqueue(
                                    state, request));
                });
        var failure = new IllegalStateException("capacity notification failed");
        doThrow(failure).when(endpoint).signal(PrefillEndpoint.Signal.CAPACITY);
        assertSame(failure, runtime.stopAndAwait());
        verify(events)
                .rejectRoute(
                        org.mockito.ArgumentMatchers.same(request),
                        org.mockito.ArgumentMatchers.any());
        assertEquals(
                List.of(request),
                org.flexlb.balance.endpoint.EndpointTestSupport.retireResources(state).ownedItems(),
                "queue drain leaves resource settlement to the lifecycle");
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
    void removedRequestIsNotifiedEvenWhenCapacitySignalFails(boolean expired) {
        FlexlbConfig config = singleConfig();
        SchedulingTestConfig.useBatchDispatcher(config);
        PrefillEndpoint endpoint = stableEndpoint(stableStatus());
        AbstractRequestScheduler events = mock(AbstractRequestScheduler.class);
        PrefillQueueScheduler runtime =
                boundaryRuntime(
                        "removal-failure", endpoint, config, new EventDrivenBlock(), events);
        RouteExecution request = item(config, endpoint, 914L, 50, System.currentTimeMillis());
        SchedulerTestSupport.bindOwner(request.ctx(), events);
        var state = RequestExecutionTestSupport.state(runtime);
        runWithStateLock(
                org.flexlb.balance.endpoint.EndpointTestSupport.stateLock(state),
                () -> {
                    assertSame(
                            request,
                            org.flexlb.balance.endpoint.EndpointTestSupport.enqueue(
                                    state, request));
                });
        RuntimeException signalFailure = new IllegalStateException("capacity signal failed");
        RuntimeException admissionFailure = new IllegalStateException("admission failed");
        doThrow(signalFailure).when(endpoint).signal(PrefillEndpoint.Signal.PLACEMENT);
        try {
            assertSame(
                    signalFailure,
                    assertThrows(
                            IllegalStateException.class,
                            () -> {
                                if (expired) {
                                    org.springframework.test.util.ReflectionTestUtils.invokeMethod(
                                            runtime, "removeAndNotify", request, null, false);
                                } else {
                                    org.springframework.test.util.ReflectionTestUtils.invokeMethod(
                                            runtime,
                                            "commitBoundary",
                                            request,
                                            CapacityBoundary.failed(admissionFailure),
                                            0L);
                                }
                            }));
            if (expired) {
                verify(events)
                        .cancel(
                                request.ctx(),
                                0L,
                                org.flexlb.balance.scheduler.CancelReason.DEADLINE_EXCEEDED);
            } else {
                verify(events)
                        .onDeliveryResult(
                                org.mockito.ArgumentMatchers.any(
                                        ContextResult.DeliveryEvidence.class));
            }
            assertEquals(0, state.progress().queueDepth());
            assertEquals(
                    List.of(request),
                    org.flexlb.balance.endpoint.EndpointTestSupport.retireResources(state)
                            .ownedItems(),
                    "queue drain leaves resource settlement to the lifecycle");
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
        BoundaryDelivery delivery =
                new BoundaryDelivery() {
                    @Override
                    CapacityBoundary selectBoundary(
                            List<RouteExecution> candidates,
                            PrefillTimePredictor.Evaluator evaluator,
                            OptionalLong prediction) {
                        return CapacityBoundary.failed(failure);
                    }
                };
        PrefillQueueScheduler runtime =
                boundaryRuntime("removal-notification", endpoint, config, delivery, events);
        RouteExecution request = item(config, endpoint, 916L, 50, System.currentTimeMillis());
        SchedulerTestSupport.bindOwner(request.ctx(), events);
        var state = RequestExecutionTestSupport.state(runtime);
        runWithStateLock(
                org.flexlb.balance.endpoint.EndpointTestSupport.stateLock(state),
                () -> {
                    assertSame(
                            request,
                            org.flexlb.balance.endpoint.EndpointTestSupport.enqueue(
                                    state, request));
                });
        try {
            RequestExecutionTestSupport.schedule(runtime);
            verify(events)
                    .onDeliveryResult(
                            org.mockito.ArgumentMatchers.any(ContextResult.DeliveryEvidence.class));
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
        PrefillEndpoint endpoint =
                org.flexlb.balance.endpoint.EndpointTestSupport.unstartedPrefill(
                        config,
                        WorkerStatus.createDiscovered(
                                RoleType.PREFILL, "test", "127.0.0.1", 8080, 9090, null),
                        SchedulerTestSupport.dispatcher(),
                        mock(AbstractRequestScheduler.class));
        PrefillQueueScheduler runtime =
                org.flexlb.balance.endpoint.EndpointTestSupport.requestExecution(endpoint);
        delivery.install(runtime);
        runtimes.add(runtime);
        RequestExecutionTestSupport.start(runtime);
        RouteExecution head = item(config, endpoint, 11L, 50, System.currentTimeMillis());

        try {
            assertTrue(RequestExecutionTestSupport.offer(runtime, head));
            await(delivery.firstPrepareEntered);
            assertNull(
                    endpoint.captureRouteProjectionInputs().queue().admissionBlock(),
                    "cache is warmed before the delivery miss is published");

            delivery.allowFirstPrepare.countDown();
            await(delivery.firstCapacity.subscribed);
            assertNotNull(
                    endpoint.captureRouteProjectionInputs().queue().admissionBlock(),
                    "publishing the block must invalidate the warm cache");

            delivery.firstCapacity.release();
            await(delivery.secondPrepareEntered);
            assertNull(
                    endpoint.captureRouteProjectionInputs().queue().admissionBlock(),
                    "capacity wake must invalidate the cached block");
        } finally {
            delivery.allowFirstPrepare.countDown();
            delivery.allowSecondPrepare.countDown();
        }
    }

    @Test
    @Timeout(10)
    void nonBatchSlotReleaseWakesTheWaitingHeadWithoutPolling() throws Exception {
        FlexlbConfig config = singleConfig();
        config.setDispatcher(org.flexlb.config.ConfigTestSupport.nonBatch());
        config.getDispatcher().setMaxInflightPerPrefillWorker(1);
        var endpoint =
                org.flexlb.balance.endpoint.EndpointTestSupport.unstartedPrefill(
                        config,
                        WorkerStatus.createDiscovered(
                                RoleType.PREFILL, null, "127.0.0.1", 8080, 9090, null),
                        SchedulerTestSupport.dispatcher(),
                        mock(AbstractRequestScheduler.class));
        PrefillQueueScheduler worker =
                org.flexlb.balance.endpoint.EndpointTestSupport.requestExecution(endpoint);
        runtimes.add(worker);
        var occupied = item(config, endpoint, 31L, 50, System.currentTimeMillis());
        var reservation = endpoint.reserveUnqueuedRoute(occupied, 1L, 1L).value();
        assertNotNull(reservation);
        var waiting = new CountDownLatch(1);
        var resumed = new CountDownLatch(1);
        var attempts = new AtomicInteger();
        doAnswer(
                        call -> {
                            attempts.incrementAndGet();
                            if (!endpoint.requestCapacityAvailable(1, 1)) {
                                waiting.countDown();
                                return CapacityBoundary.unavailable(
                                        CapacityBoundary.Resource.PREFILL_REQUEST);
                            }
                            resumed.countDown();
                            return CapacityBoundary.OWNERSHIP_LOST;
                        })
                .when(worker)
                .processSelection(
                        any(),
                        anyString(),
                        any(),
                        any(),
                        any(),
                        org.mockito.ArgumentMatchers.any(),
                        org.mockito.ArgumentMatchers.any());
        RequestExecutionTestSupport.start(worker);
        assertTrue(
                RequestExecutionTestSupport.offer(
                        worker, item(config, endpoint, 32L, 50, System.currentTimeMillis())));
        await(waiting);
        assertTrue(
                RequestExecutionTestSupport.offer(
                        worker, item(config, endpoint, 33L, 50, System.currentTimeMillis())));
        TimeUnit.MILLISECONDS.sleep(100L);
        assertEquals(1, attempts.get(), "backlog publication must not poll a full delivery slot");
        assertTrue(endpoint.rollback(reservation));
        await(resumed);
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    void deliveryOnlyWaitKeepsBacklogSelectable() throws Exception {
        FlexlbConfig config = singleConfig();
        EventDrivenBlock delivery = new EventDrivenBlock(CapacityBoundary.Resource.PREFILL_BATCH);
        PrefillEndpoint endpoint =
                org.flexlb.balance.endpoint.EndpointTestSupport.unstartedPrefill(
                        config,
                        WorkerStatus.createDiscovered(
                                RoleType.PREFILL, "test", "127.0.0.1", 8080, 9090, null),
                        SchedulerTestSupport.dispatcher(),
                        mock(AbstractRequestScheduler.class));
        PrefillQueueScheduler runtime =
                org.flexlb.balance.endpoint.EndpointTestSupport.requestExecution(endpoint);
        delivery.install(runtime);
        runtimes.add(runtime);
        RequestExecutionTestSupport.start(runtime);
        long now = System.currentTimeMillis();
        RouteExecution head = item(config, endpoint, 21L, 50, now);
        assertTrue(RequestExecutionTestSupport.offer(runtime, head));
        await(delivery.firstCapacity.subscribed);
        RequestTimeEstimation.Inputs inputs = endpoint.captureRouteProjectionInputs();
        assertNotNull(inputs.queue().admissionBlock());
        assertNull(
                inputs.queue().admissionBlock().semantics(),
                "batch capacity waits must not impose publication restrictions");
        Candidate candidate =
                RequestTimeEstimationTestSupport.project(
                        inputs,
                        new RequestTimeEstimationTestSupport.Probe(
                                22L, 50, now + 1L, Long.MAX_VALUE, 10L, 0L, 0L),
                        now);
        assertTrue(
                candidate.selectable(),
                "a delivery-only wait must leave incoming backlog selectable");
        RouteExecution backlog = item(config, endpoint, 22L, 50, now + 1L);
        assertTrue(RequestExecutionTestSupport.offer(runtime, backlog));
        TimeUnit.MILLISECONDS.sleep(100L);
        assertEquals(
                1,
                delivery.attempts.get(),
                "publishing backlog must not retry the capacity-blocked head");
        assertEquals(List.of(head, backlog), RequestExecutionTestSupport.capture(runtime).items());
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    void concurrentReadersShareOneVersionWithoutHoldingQueueLock() throws Exception {
        FlexlbConfig config = singleConfig();
        ProjectionCacheBlock delivery = new ProjectionCacheBlock();
        PrefillEndpoint endpoint =
                org.flexlb.balance.endpoint.EndpointTestSupport.unstartedPrefill(
                        config,
                        WorkerStatus.createDiscovered(
                                RoleType.PREFILL, "test", "127.0.0.1", 8080, 9090, null),
                        SchedulerTestSupport.dispatcher(),
                        mock(AbstractRequestScheduler.class));
        PrefillQueueScheduler runtime =
                org.flexlb.balance.endpoint.EndpointTestSupport.requestExecution(endpoint);
        delivery.install(runtime);
        runtimes.add(runtime);
        RequestExecutionTestSupport.start(runtime);
        RouteExecution head = spy(item(config, endpoint, 13L, 50, System.currentTimeMillis()));
        CountDownLatch materializing = new CountDownLatch(1);
        CountDownLatch finish = new CountDownLatch(1);
        doAnswer(
                        invocation -> {
                            if (Thread.currentThread().getName().startsWith("projection-reader")) {
                                materializing.countDown();
                                await(finish);
                            }
                            return invocation.callRealMethod();
                        })
                .when(head)
                .seqLen();
        try (var readers =
                java.util.concurrent.Executors.newFixedThreadPool(
                        16, Thread.ofPlatform().name("projection-reader-", 0).factory())) {
            assertTrue(RequestExecutionTestSupport.offer(runtime, head));
            await(delivery.firstPrepareEntered);
            var first = readers.submit(endpoint::captureRouteProjectionInputs);
            await(materializing);
            var started = new CountDownLatch(15);
            var followers =
                    new java.util.ArrayList<
                            java.util.concurrent.Future<RequestTimeEstimation.Inputs>>();
            for (int i = 0; i < 15; i++) {
                followers.add(
                        readers.submit(
                                () -> {
                                    started.countDown();
                                    return endpoint.captureRouteProjectionInputs();
                                }));
            }
            await(started);
            // A normal queue reader must progress while materialization is blocked.
            CompletableFuture.supplyAsync(() -> RequestExecutionTestSupport.capture(runtime))
                    .get(2, TimeUnit.SECONDS);
            finish.countDown();
            var shared = first.get(2, TimeUnit.SECONDS);
            for (var follower : followers) {
                org.junit.jupiter.api.Assertions.assertSame(
                        shared, follower.get(2, TimeUnit.SECONDS));
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
        PrefillEndpoint endpoint =
                org.flexlb.balance.endpoint.EndpointTestSupport.unstartedPrefill(
                        config,
                        WorkerStatus.createDiscovered(
                                RoleType.PREFILL, "test", "127.0.0.1", 8080, 9090, null),
                        SchedulerTestSupport.dispatcher(),
                        mock(AbstractRequestScheduler.class));
        PrefillQueueScheduler runtime =
                org.flexlb.balance.endpoint.EndpointTestSupport.requestExecution(endpoint);
        delivery.install(runtime);
        runtimes.add(runtime);
        RequestExecutionTestSupport.start(runtime);
        RouteExecution head = spy(item(config, endpoint, 12L, 50, System.currentTimeMillis()));
        AtomicReference<Thread> capturingThread = new AtomicReference<>();
        CountDownLatch materializingSnapshot = new CountDownLatch(1);
        CountDownLatch finishSnapshot = new CountDownLatch(1);
        doAnswer(
                        invocation -> {
                            if (Thread.currentThread() == capturingThread.get()) {
                                materializingSnapshot.countDown();
                                await(finishSnapshot);
                            }
                            return invocation.callRealMethod();
                        })
                .when(head)
                .seqLen();

        try {
            assertTrue(RequestExecutionTestSupport.offer(runtime, head));
            await(delivery.firstPrepareEntered);
            CompletableFuture<RequestTimeEstimation.Inputs> oldCapture =
                    CompletableFuture.supplyAsync(
                            () -> {
                                capturingThread.set(Thread.currentThread());
                                return endpoint.captureRouteProjectionInputs();
                            });
            await(materializingSnapshot);
            delivery.allowFirstPrepare.countDown();
            await(delivery.firstCapacity.subscribed);
            finishSnapshot.countDown();
            assertNull(
                    oldCapture.get(2, TimeUnit.SECONDS).queue().admissionBlock(),
                    "the snapshot was taken before the block was published");
            assertNotNull(
                    endpoint.captureRouteProjectionInputs().queue().admissionBlock(),
                    "a late snapshot must not overwrite capacity-block invalidation");
        } finally {
            finishSnapshot.countDown();
            delivery.allowFirstPrepare.countDown();
            delivery.allowSecondPrepare.countDown();
        }
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    void removingCapturedMemberDuringPredictionInvalidatesWholeSelection() throws Exception {
        FlexlbConfig config = fixedConfig();
        CountDownLatch firstStatusRead = new CountDownLatch(1);
        CountDownLatch allowFirstStatusRead = new CountDownLatch(1);
        WorkerStatus status = gatedFirstStatusRead(firstStatusRead, allowFirstStatusRead);
        PrefillEndpoint endpoint = stableEndpoint(status);
        PredictionGate delivery = new PredictionGate();
        when(endpoint.predictor.evaluator().newBatchPrediction())
                .thenAnswer(call -> delivery.newBatchPrediction());
        var scheduler =
                SchedulerTestSupport.create(
                        SchedulingTestConfig.configService(config),
                        mock(DeliveryMetricsReporter.class),
                        mock(org.flexlb.service.monitor.RequestSchedulerReporter.class),
                        mock(TheoryCacheHitReporter.class));
        var execution =
                spy(
                        org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(scheduler)
                                .deliveries());
        org.flexlb.balance.scheduler.SchedulerTestServices.setDependency(
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(scheduler),
                "deliveries",
                execution);
        doAnswer(
                        call -> {
                            delivery.prepareCalls.incrementAndGet();
                            return call.callRealMethod();
                        })
                .when(execution)
                .tryReserveDelivery();
        PrefillQueueScheduler runtime =
                RequestExecutionTestSupport.create(
                        "revoked-selection",
                        endpoint,
                        config,
                        SchedulerTestSupport.dispatcher(),
                        scheduler);
        runtimes.add(runtime);
        var ownership = RequestExecutionTestSupport.state(runtime);
        doAnswer(
                        call -> {
                            runtime.signal(false);
                            return null;
                        })
                .when(endpoint)
                .signal(PrefillEndpoint.Signal.INPUT);
        when(endpoint.releaseRequest(any(RouteExecution.class)))
                .thenAnswer(call -> ownership.releaseRequest(call.getArgument(0)));
        RequestExecutionTestSupport.start(runtime);
        long now = System.currentTimeMillis();
        RouteExecution first = item(config, endpoint, 1L, 50, now);
        RouteExecution revoked = item(config, endpoint, 2L, 50, now + 1L);
        for (RouteExecution route : List.of(first, revoked)) {
            route.ctx()
                    .data()
                    .setGenerateInputPb(com.google.protobuf.ByteString.copyFromUtf8("input"));
            assertSame(
                    RequestProtocolTestSupport.submitForManualPlacement(scheduler, route.ctx()),
                    route.ctx().data().getFuture());
            assertTrue(scheduler.requests.isCurrent(route.ctx()));
        }

        try {
            try (var admission =
                    SchedulerTestSupport.claim(
                            scheduler, first.requestId(), first.ctx().data().getFuture())) {
                assertNotNull(admission);
                try {
                    assertEquals(
                            org.flexlb.balance.PlacementResult.Status.SUCCESS,
                            RequestProtocolTestSupport.commitRoute(
                                    scheduler,
                                    first,
                                    RequestProtocolTestSupport.publication(
                                            () ->
                                                    RequestExecutionTestSupport.offer(
                                                            runtime, first))));
                } finally {
                    AbstractRequestScheduler.finishPlacement(admission);
                }
            }
            await(firstStatusRead);
            try (var admission =
                    SchedulerTestSupport.claim(
                            scheduler, revoked.requestId(), revoked.ctx().data().getFuture())) {
                assertNotNull(admission);
                try {
                    assertEquals(
                            org.flexlb.balance.PlacementResult.Status.SUCCESS,
                            RequestProtocolTestSupport.commitRoute(
                                    scheduler,
                                    revoked,
                                    RequestProtocolTestSupport.publication(
                                            () ->
                                                    RequestExecutionTestSupport.offer(
                                                            runtime, revoked))));
                } finally {
                    AbstractRequestScheduler.finishPlacement(admission);
                }
            }
            allowFirstStatusRead.countDown();
            await(delivery.groupPredictionEntered);

            assertTrue(runtime.prefillEndpoint().removeQueued(revoked));
            delivery.allowGroupPrediction.countDown();
            await(delivery.nextDecisionStarted);

            assertEquals(
                    0,
                    delivery.prepareCalls.get(),
                    "a selection containing a revoked identity cannot prepare");
            List<RouteExecution> remaining = RequestExecutionTestSupport.capture(runtime).items();
            assertEquals(1, remaining.size());
            assertSame(first, remaining.getFirst());
        } finally {
            allowFirstStatusRead.countDown();
            delivery.allowGroupPrediction.countDown();
            assertNull(runtime.stopAndAwait());
            SchedulerTestSupport.shutdown(
                    org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(scheduler));
        }
    }

    private PrefillQueueScheduler runningRuntime(
            FlexlbConfig config, PrefillEndpoint endpoint, BoundaryDelivery boundary) {
        PrefillQueueScheduler worker =
                boundaryRuntime(
                        "scheduling-test",
                        endpoint,
                        config,
                        boundary,
                        mock(AbstractRequestScheduler.class));
        runtimes.add(worker);
        RequestExecutionTestSupport.start(worker);
        return worker;
    }

    private static PrefillQueueScheduler boundaryRuntime(
            String key,
            PrefillEndpoint endpoint,
            FlexlbConfig config,
            BoundaryDelivery boundary,
            AbstractRequestScheduler scheduler) {
        PrefillQueueScheduler worker =
                RequestExecutionTestSupport.create(
                        key, endpoint, config, SchedulerTestSupport.dispatcher(), scheduler);
        boundary.install(worker);
        return worker;
    }

    private PrefillQueueScheduler runningRuntime(
            FlexlbConfig config, PrefillEndpoint endpoint, DeliveryDispatcher delivery) {
        PrefillQueueScheduler runtime =
                RequestExecutionTestSupport.create(
                        "scheduling-test",
                        endpoint,
                        config,
                        delivery,
                        mock(AbstractRequestScheduler.class));
        runtimes.add(runtime);
        RequestExecutionTestSupport.start(runtime);
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
        DecisionPolicyConfig decision = SchedulingTestConfig.useFixedWindowDecision(config);
        decision.setMaxRequests(2);
        decision.setMaxCollectionWaitMs(60_000L);
        decision.setMaxPredictedExecutionMs(500L);
        SchedulingTestConfig.useBatchDispatcher(config);
        return config;
    }

    private static RouteExecution item(
            FlexlbConfig config,
            PrefillEndpoint endpoint,
            long requestId,
            int priority,
            long enqueuedAtMs) {
        Request request =
                Request.builder()
                        .requestId(String.valueOf(requestId))
                        .rawPriority(priority)
                        .seqLen(10L)
                        .build();
        RequestContext context = new RequestContext(config);
        context.data().setRequest(request);
        context.data().setSchedulingMetadata(SchedulingMetadata.explicit(priority, Long.MAX_VALUE));
        context.data().setFuture(new CompletableFuture<Response>());
        return SchedulingTestConfig.createRoute(
                context, null, null, null, endpoint, null, null, enqueuedAtMs);
    }

    private static PrefillEndpoint stableEndpoint(WorkerStatus status) {
        if (org.mockito.Mockito.mockingDetails(status).isMock())
            when(status.isAvailableForScheduling()).thenReturn(true);
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        var evaluator = mock(PrefillTimePredictor.Evaluator.class);
        when(evaluator.newBatchPrediction())
                .thenAnswer(
                        call -> (PrefillTimePredictor.BatchPrediction) (seqLen, hitCache) -> 0.0);
        when(predictor.evaluator()).thenReturn(evaluator);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getStatus()).thenReturn(status);
        when(endpoint.acceptsNewRequests())
                .thenAnswer(call -> !endpoint.isRetired() && status.isAvailableForScheduling());
        org.springframework.test.util.ReflectionTestUtils.setField(
                endpoint, "predictor", predictor);
        return endpoint;
    }

    private static WorkerStatus stableStatus() {
        WorkerStatus status = mock(WorkerStatus.class);
        when(status.committedEngineObservation()).thenReturn(capacity());
        return status;
    }

    private static WorkerStatus gatedFirstStatusRead(
            CountDownLatch entered, CountDownLatch proceed) {
        WorkerStatus status = mock(WorkerStatus.class);
        AtomicBoolean first = new AtomicBoolean(true);
        when(status.committedEngineObservation())
                .thenAnswer(
                        ignored -> {
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
            assertTrue(
                    latch.await(5, TimeUnit.SECONDS), "worker did not reach the expected boundary");
        } catch (InterruptedException interruption) {
            Thread.currentThread().interrupt();
            throw new AssertionError("interrupted while awaiting worker boundary", interruption);
        }
    }

    /** Queue scheduling fixture: stubs Worker boundaries, never Dispatcher business APIs. */
    private abstract static class BoundaryDelivery {
        abstract CapacityBoundary selectBoundary(
                List<RouteExecution> candidates,
                PrefillTimePredictor.Evaluator evaluator,
                OptionalLong plannedPrediction);

        void install(PrefillQueueScheduler worker) {
            doAnswer(
                            call ->
                                    selectBoundary(
                                            call.getArgument(0),
                                            call.getArgument(2),
                                            call.getArgument(3)))
                    .when(worker)
                    .processSelection(
                            any(),
                            anyString(),
                            any(),
                            any(),
                            any(),
                            org.mockito.ArgumentMatchers.any(),
                            org.mockito.ArgumentMatchers.any());
            doAnswer(call -> capacityAvailable(call.getArgument(0), call.getArgument(1)))
                    .when(worker)
                    .capacityAvailable(any(), any());
            doAnswer(
                            call ->
                                    watchCapacity(
                                            call.getArgument(0),
                                            call.getArgument(1),
                                            call.getArgument(2)))
                    .when(worker)
                    .watchCapacity(any(), any(), any());
        }

        private final java.util.Map<CapacityBoundary, TestAvailability> bindings =
                new java.util.IdentityHashMap<>();

        CapacityBoundary unavailable(TestAvailability availability) {
            return boundary(availability, CapacityBoundary.Resource.DISPATCH_EXECUTOR);
        }

        CapacityBoundary boundary(
                TestAvailability availability, CapacityBoundary.Resource resource) {
            var boundary =
                    resource == CapacityBoundary.Resource.PREFILL_REQUEST
                                    || resource == CapacityBoundary.Resource.PREFILL_BATCH
                            ? CapacityBoundary.unavailable(resource)
                            : CapacityBoundary.unavailable(resource);
            bindings.put(boundary, availability);
            return boundary;
        }

        boolean capacityAvailable(RouteExecution item, CapacityBoundary boundary) {
            return bindings.get(boundary).isAvailable();
        }

        Runnable watchCapacity(RouteExecution item, CapacityBoundary boundary, Runnable wake) {
            var source = bindings.get(boundary);
            source.addListener(wake);
            return () -> source.removeListener(wake);
        }
    }

    @lombok.RequiredArgsConstructor
    private static final class EventDrivenBlock extends BoundaryDelivery {

        private final CapacityBoundary.Resource resource;

        private final TestAvailability firstCapacity = new TestAvailability();

        private final TestAvailability parkedCapacity = new TestAvailability();

        private final AtomicInteger attempts = new AtomicInteger();

        private final CountDownLatch firstAttempt = new CountDownLatch(1);

        private final CountDownLatch secondAttempt = new CountDownLatch(1);

        private EventDrivenBlock() {
            this(CapacityBoundary.Resource.DISPATCH_EXECUTOR);
        }

        @Override
        CapacityBoundary selectBoundary(
                List<RouteExecution> candidates,
                PrefillTimePredictor.Evaluator evaluator,
                OptionalLong plannedPrediction) {
            int attempt = attempts.incrementAndGet();
            if (attempt == 1) {
                firstAttempt.countDown();
                return boundary(firstCapacity, resource);
            }
            secondAttempt.countDown();
            return boundary(parkedCapacity, resource);
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
                List<RouteExecution> candidates,
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

        private final CountDownLatch groupPredictionEntered = new CountDownLatch(1);

        private final CountDownLatch allowGroupPrediction = new CountDownLatch(1);

        private final CountDownLatch nextDecisionStarted = new CountDownLatch(1);

        private final AtomicBoolean groupPredictionReturned = new AtomicBoolean();

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

        private final CopyOnWriteArrayList<Runnable> listeners = new CopyOnWriteArrayList<>();

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
