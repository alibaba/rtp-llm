package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.service.TheoryCacheHitReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.any;
import static org.mockito.Mockito.isNull;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

/** A scheduler implemented without the default scheduler's inheritance or decision thread. */
class CustomRequestSchedulerContractTest {
    @Test
    void customBeanReplacesDefaultAssemblyWithoutCreatingItsDependencies() {
        try (ManualScheduler scheduler = new ManualScheduler(true)) {
            var config = mock(ConfigService.class);
            when(config.loadBalanceConfig()).thenReturn(scheduler.config);
            new org.springframework.boot.test.context.runner.ApplicationContextRunner()
                    .withUserConfiguration(PlacementConfiguration.class)
                    .withBean(
                            "environmentConfigSource",
                            org.flexlb.service.config.ConfigSource.class,
                            () -> mock(org.flexlb.service.config.ConfigSource.class))
                    .withBean(
                            "nacosConfigSource",
                            org.flexlb.service.config.ConfigSource.class,
                            () -> mock(org.flexlb.service.config.ConfigSource.class))
                    .withBean(
                            "uniConfigConfigSource",
                            org.flexlb.service.config.ConfigSource.class,
                            () -> mock(org.flexlb.service.config.ConfigSource.class))
                    .withBean(ConfigService.class, () -> config)
                    .withBean(RequestScheduler.class, () -> scheduler)
                    .run(
                            context -> {
                                org.assertj.core.api.Assertions.assertThat(context)
                                        .hasSingleBean(RequestScheduler.class)
                                        .doesNotHaveBean(QueuedRequestScheduler.class);
                                org.junit.jupiter.api.Assertions.assertSame(
                                        scheduler, context.getBean(RequestScheduler.class));
                            });
        }
    }

    @Test
    void cancellationFencesAnAdmissionAlreadyStartedByACustomScheduler() throws Exception {
        try (ManualScheduler scheduler = new ManualScheduler(true)) {
            var context = RequestProtocolTestSupport.context(scheduler.config, 81004L);
            var result = scheduler.entry.submit(context);
            var admission = scheduler.lifecycle.claimAdmissionHandle("81004", result);
            assertNotNull(admission);
            try {
                scheduler.entry.cancel("81004", 0L, CancelReason.CLIENT_CANCELLED);
                assertFalse(scheduler.lifecycle.isAdmissionOpen("81004", result));
                assertFalse(scheduler.advance());
            } finally {
                admission.finish();
            }
            assertFalse(result.get(2, TimeUnit.SECONDS).isSuccess());
            RequestProtocolTestSupport.awaitCondition(
                    () -> scheduler.runtime.requests().liveRequestCount() == 0);
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void queuePreservesScanBudgetRetryAndExactRestore(boolean alternative) {
        PlacementQueue queue =
                alternative ? new ListPlacementQueue() : new OrderedRequestQueue(false);
        var config = SchedulingTestConfig.batchConfig();
        var first = new GlobalQueueEntry(RequestProtocolTestSupport.context(config, 81005L), null);
        var second = new GlobalQueueEntry(RequestProtocolTestSupport.context(config, 81006L), null);
        queue.add(first);
        queue.add(second);
        assertTrue(queue.scanForPlanningCandidates(2, 1, entry -> false).isEmpty());
        assertEquals(
                java.util.List.of(second), queue.scanForPlanningCandidates(2, 1, entry -> true));
        queue.markRequestReadyForRetry(first);
        assertEquals(
                java.util.List.of(first), queue.scanForPlanningCandidates(2, 1, entry -> true));
        long sequence = first.sequence();
        assertTrue(queue.remove(first));
        assertFalse(queue.remove(first));
        queue.restore(first);
        assertEquals(sequence, first.sequence());
        assertTrue(queue.order().compare(first, second) < 0);
        assertEquals(2, queue.drain().size());
        assertEquals(0, queue.size());
        assertFalse(queue.hasUnscannedRequests());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void callerControlsProgressButCancellationUsesSharedLifecycle(boolean alternative)
            throws Exception {
        try (ManualScheduler scheduler = new ManualScheduler(alternative)) {
            var context = RequestProtocolTestSupport.context(scheduler.config, 81001L);
            var result = scheduler.entry.submit(context);
            assertFalse(result.isDone());
            scheduler.entry.cancel("81001", 0L, CancelReason.CLIENT_CANCELLED);
            assertEquals(
                    StrategyErrorType.REQUEST_CANCELLED.getErrorCode(),
                    result.get(2, TimeUnit.SECONDS).getCode());
            assertEquals(0, scheduler.queue.size());
            assertFalse(scheduler.advance());
            verifyNoInteractions(scheduler.selector);
            RequestProtocolTestSupport.awaitCondition(
                    () -> scheduler.runtime.requests().liveRequestCount() == 0);
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void deadlineUsesSameInvalidationWithoutWaitingForSchedulingTurn(boolean alternative)
            throws Exception {
        try (ManualScheduler scheduler = new ManualScheduler(alternative)) {
            var context = RequestProtocolTestSupport.context(scheduler.config, 81002L);
            var result = scheduler.entry.submit(context);
            var deadline = context.snapshot().deadlines().request();
            scheduler.lifecycle.onDeadline(deadline, deadline.atMs());
            assertFalse(result.get(2, TimeUnit.SECONDS).isSuccess());
            RequestProtocolTestSupport.awaitCondition(() -> scheduler.queue.size() == 0);
            assertEquals(
                    CancelReason.DEADLINE_EXCEEDED, context.snapshot().cancellation().reason());
            assertFalse(scheduler.advance());
            verifyNoInteractions(scheduler.selector);
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void customSchedulingTurnCanCompleteARegisteredRequest(boolean alternative) throws Exception {
        try (ManualScheduler scheduler = new ManualScheduler(alternative)) {
            var context = RequestProtocolTestSupport.context(scheduler.config, 81003L);
            var result = scheduler.entry.submit(context);
            assertFalse(result.isDone());
            assertTrue(scheduler.advance());
            assertEquals(
                    StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(),
                    result.get(2, TimeUnit.SECONDS).getCode());
            verify(scheduler.selector).select(context, null);
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void customSchedulerOwnsDeadlineForBothRequestModes(boolean queued) throws Exception {
        var due = new java.util.concurrent.atomic.AtomicLong(Long.MAX_VALUE);
        var dispatched = new java.util.concurrent.atomic.AtomicLong();
        var scheduled = new java.util.concurrent.CountDownLatch(1);
        try (var scheduler =
                new ManualScheduler(true, queued) {
                    @Override
                    public void onRequest(RequestContext request) {
                        super.onRequest(request);
                        due.set(System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(80));
                        runtime.taskTimer().schedule(this::schedule, 80L);
                    }

                    @Override
                    public void schedule() {
                        long now = System.nanoTime();
                        if (now < due.get()) throw new AssertionError("timer ran before deadline");
                        if (advance()) dispatched.set(now);
                        scheduled.countDown();
                    }
                }) {
            scheduler.start();
            var future =
                    scheduler.entry.submit(
                            RequestProtocolTestSupport.context(scheduler.config, 81006L));
            assertFalse(future.isDone(), "admission must not bypass the scheduler's deadline");
            assertFalse(future.get(2, TimeUnit.SECONDS).isSuccess());
            assertTrue(scheduled.await(2, TimeUnit.SECONDS));
            assertTrue(dispatched.get() >= due.get());
            verify(scheduler.selector).select(any(), isNull());
        }
    }

    @Test
    void customSchedulerOwnsEndpointSubscriptions() throws Exception {
        var availability = new PlacementAvailability();
        var changed = new java.util.concurrent.atomic.AtomicBoolean();
        var scheduled = new java.util.concurrent.CountDownLatch(1);
        try (var scheduler =
                new ManualScheduler(true) {
                    private final PlacementAvailability.Listener listener =
                            key -> {
                                changed.set(true);
                                schedule();
                            };

                    @Override
                    public void start() {
                        availability.addListener(listener);
                    }

                    @Override
                    public void stop() {
                        availability.removeListener(listener);
                    }

                    @Override
                    public void schedule() {
                        if (changed.get()) scheduled.countDown();
                    }
                }) {
            scheduler.start();
            var endpoint = mock(org.flexlb.balance.endpoint.PrefillEndpoint.class);
            when(endpoint.getStatus())
                    .thenReturn(
                            org.flexlb.dao.master.WorkerStatus.createDiscovered(
                                    org.flexlb.dao.route.RoleType.PREFILL,
                                    null,
                                    "127.0.0.1",
                                    8100,
                                    8101,
                                    null));
            when(endpoint.ipPort()).thenReturn("127.0.0.1:8100");
            availability.changed(endpoint);
            assertTrue(scheduled.await(2, TimeUnit.SECONDS));
        }
    }

    private static class ManualScheduler implements RequestScheduler, AutoCloseable {
        final FlexlbConfig config = SchedulingTestConfig.batchConfig();
        final SchedulerTestServices runtime;
        final RequestLifecycle lifecycle;
        final RequestLifecycle entry;
        final PlacementQueue queue;
        final RequestWorkerSelector selector = mock(RequestWorkerSelector.class);

        ManualScheduler(boolean alternative) {
            this(alternative, true);
        }

        ManualScheduler(boolean alternative, boolean queued) {
            if (!queued) {
                config.setScheduler(org.flexlb.config.ConfigTestSupport.direct());
                config.getDispatcher().setType(org.flexlb.config.DispatcherConfig.Type.NON_BATCH);
            }
            queue = alternative ? new ListPlacementQueue() : new OrderedRequestQueue(false);
            when(selector.select(any(), isNull()))
                    .thenReturn(
                            org.flexlb.balance.PlacementResult.rejected(
                                    Response.error(StrategyErrorType.RESOURCE_EXHAUSTED)));
            var service = mock(ConfigService.class);
            when(service.loadBalanceConfig()).thenReturn(config);
            runtime =
                    new SchedulerTestServices(
                            new RequestRepository(),
                            mock(EndpointRegistry.class),
                            mock(DeliveryMetricsReporter.class),
                            mock(RequestSchedulerReporter.class),
                            SchedulerTestSupport.dispatcher(),
                            service,
                            mock(EngineGrpcClient.class));
            lifecycle =
                    runtime.boundLifecycle(
                            this,
                            config,
                            SchedulingTestConfig.capacity(config),
                            mock(TheoryCacheHitReporter.class));
            entry = lifecycle;
            runtime.registerScheduler(this);
        }

        @Override
        public void schedule() {}

        public void start() {}

        public void stop() {}

        public boolean awaitStopped(long deadlineNanos) {
            return true;
        }

        @Override
        public void onEndpointChanged(org.flexlb.balance.endpoint.WorkerEndpoint endpoint) {}

        @Override
        public void onRequest(RequestContext request) {
            synchronized (queue) {
                if (!request.data().getFuture().isDone()
                        && request.snapshot().cancellation() == null) {
                    queue.add(new GlobalQueueEntry(request, null));
                }
            }
        }

        @Override
        public void onRequestChanged(RequestContext request) {
            synchronized (queue) {
                for (var entry : queue.drain()) {
                    if (entry.context != request) queue.restore(entry);
                }
            }
        }

        boolean advance() {
            GlobalQueueEntry selected;
            synchronized (queue) {
                var candidates = queue.scanForPlanningCandidates(1, 1, entry -> true);
                if (candidates.isEmpty()) return false;
                selected = candidates.getFirst();
                queue.remove(selected);
            }
            var result = selector.select(selected.context, selected.routingGroup);
            lifecycle.publish(selected.context, result.failure(), false);
            return true;
        }

        @Override
        public void close() {
            SchedulerTestSupport.shutdown(runtime);
        }
    }
}
