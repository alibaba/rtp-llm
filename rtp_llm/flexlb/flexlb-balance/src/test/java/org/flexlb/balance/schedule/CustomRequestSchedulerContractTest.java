package org.flexlb.balance.schedule;

import org.flexlb.balance.metrics.DeliveryMetricsReporter;
import org.flexlb.balance.metrics.RequestSchedulerReporter;
import org.flexlb.balance.queue.GlobalQueueEntry;
import org.flexlb.balance.queue.ListSchedulingQueue;
import org.flexlb.balance.queue.OrderedRequestQueue;
import org.flexlb.balance.queue.SchedulingQueue;
import org.flexlb.balance.requestlog.TheoryCacheHitReporter;
import org.flexlb.balance.strategy.RequestRouteSelector;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.worker.endpoint.CapacityNotifier;
import org.flexlb.worker.endpoint.EndpointRegistry;
import org.flexlb.worker.endpoint.PrefillEndpoint;
import org.flexlb.worker.endpoint.Response;
import org.flexlb.worker.endpoint.StrategyErrorType;
import org.flexlb.worker.endpoint.WorkerEndpoint;
import org.flexlb.worker.rpc.EngineGrpcClient;
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

/**
 * A custom scheduler owns its decisions and triggers while inheriting shared request management.
 */
class CustomRequestSchedulerContractTest {
    @Test
    void missingSchedulingMetadataIsRejectedWithoutCreatingDefaults() {
        try (var scheduler = new ManualScheduler(true)) {
            var context = RequestProtocolTestSupport.context(scheduler.config, 81904L);
            context.setSchedulingMetadata(null);
            assertEquals(
                    StrategyErrorType.INVALID_REQUEST.getErrorCode(),
                    scheduler.submit(context).join().getCode());
            org.junit.jupiter.api.Assertions.assertNull(context.getSchedulingMetadata());
            org.junit.jupiter.api.Assertions.assertNull(context.getRequestScheduler());
            assertEquals(0, scheduler.runtime.requests().liveRequestCount());
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void schedulerChoosesTimeoutErrorForEveryTimeoutEntry(boolean queued) throws Exception {
        try (var scheduler =
                new ManualScheduler(true, queued) {
                    @Override
                    protected StrategyErrorType schedulingTimeoutError() {
                        return StrategyErrorType.ADMISSION_UNAVAILABLE;
                    }
                }) {
            var expired = RequestProtocolTestSupport.context(scheduler.config, 81901L);
            expired.setSchedulingMetadata(
                    org.flexlb.balance.schedule.SchedulingMetadata.explicit(
                            50, System.currentTimeMillis() - 1));
            assertEquals(
                    StrategyErrorType.ADMISSION_UNAVAILABLE.getErrorCode(),
                    scheduler.submit(expired).get(2, TimeUnit.SECONDS).getCode());

            var timed = RequestProtocolTestSupport.context(scheduler.config, 81902L);
            var timedResult = scheduler.submit(timed);
            var deadline = timed.snapshot().deadlines().request();
            scheduler.onDeadline(deadline, deadline.atMs());
            assertEquals(
                    StrategyErrorType.ADMISSION_UNAVAILABLE.getErrorCode(),
                    timedResult.get(2, TimeUnit.SECONDS).getCode());

            var cancelled = RequestProtocolTestSupport.context(scheduler.config, 81903L);
            var cancelledResult = scheduler.submit(cancelled);
            scheduler.cancel("81903", CancelReason.DEADLINE_EXCEEDED);
            assertEquals(
                    StrategyErrorType.ADMISSION_UNAVAILABLE.getErrorCode(),
                    cancelledResult.get(2, TimeUnit.SECONDS).getCode());
        }
    }

    @Test
    void customBeanReplacesDefaultAssemblyWithoutCreatingItsDependencies() {
        try (ManualScheduler scheduler = new ManualScheduler(true)) {
            var config = mock(ConfigService.class);
            when(config.loadBalanceConfig()).thenReturn(scheduler.config);
            new org.springframework.boot.test.context.runner.ApplicationContextRunner()
                    .withUserConfiguration(SchedulerConfiguration.class)
                    .withBean(
                            "environmentConfigSource",
                            org.flexlb.config.loading.source.ConfigSource.class,
                            () -> mock(org.flexlb.config.loading.source.ConfigSource.class))
                    .withBean(
                            "nacosConfigSource",
                            org.flexlb.config.loading.source.ConfigSource.class,
                            () -> mock(org.flexlb.config.loading.source.ConfigSource.class))
                    .withBean(
                            "uniConfigConfigSource",
                            org.flexlb.config.loading.source.ConfigSource.class,
                            () -> mock(org.flexlb.config.loading.source.ConfigSource.class))
                    .withBean(ConfigService.class, () -> config)
                    .withBean(RequestScheduler.class, () -> scheduler)
                    .run(
                            context -> {
                                org.assertj.core.api.Assertions.assertThat(context)
                                        .hasSingleBean(RequestScheduler.class)
                                        .doesNotHaveBean(QueuedRequestScheduler.class)
                                        .doesNotHaveBean(EndpointRegistry.class);
                                org.junit.jupiter.api.Assertions.assertSame(
                                        scheduler, context.getBean(RequestScheduler.class));
                            });
        }
    }

    @Test
    void cancellationFencesAnAdmissionAlreadyStartedByACustomScheduler() throws Exception {
        try (ManualScheduler scheduler = new ManualScheduler(true)) {
            var context = RequestProtocolTestSupport.context(scheduler.config, 81004L);
            var result = scheduler.submit(context);
            var admission = SchedulerTestSupport.claim(scheduler, "81004", result);
            assertNotNull(admission);
            try {
                scheduler.cancel("81004", CancelReason.CLIENT_CANCELLED);
                assertFalse(scheduler.isAdmissionOpen(context));
                scheduler.advance();
                org.mockito.Mockito.verifyNoInteractions(scheduler.selector);
            } finally {
                admission
                        .requestContext()
                        .getRequestScheduler()
                        .finishSelectionWithoutCommit(admission, null);
            }
            assertFalse(result.get(2, TimeUnit.SECONDS).isSuccess());
            RequestProtocolTestSupport.awaitCondition(
                    () -> scheduler.runtime.requests().liveRequestCount() == 0);
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void queuePreservesScanBudgetRetryAndExactRestore(boolean alternative) {
        SchedulingQueue queue =
                alternative ? new ListSchedulingQueue() : new OrderedRequestQueue(false);
        var config = SchedulingTestConfig.newConfig();
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
            var result = scheduler.submit(context);
            assertFalse(result.isDone());
            scheduler.cancel("81001", CancelReason.CLIENT_CANCELLED);
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
            var result = scheduler.submit(context);
            var deadline = context.snapshot().deadlines().request();
            scheduler.onDeadline(deadline, deadline.atMs());
            assertFalse(result.get(2, TimeUnit.SECONDS).isSuccess());
            RequestProtocolTestSupport.awaitCondition(() -> scheduler.queue.size() == 0);
            assertEquals(CancelReason.DEADLINE_EXCEEDED, context.snapshot().cancellationReason());
            assertFalse(scheduler.advance());
            verifyNoInteractions(scheduler.selector);
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void customSchedulingTurnCanCompleteARegisteredRequest(boolean alternative) throws Exception {
        try (ManualScheduler scheduler = new ManualScheduler(alternative)) {
            var context = RequestProtocolTestSupport.context(scheduler.config, 81003L);
            var result = scheduler.submit(context);
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
                    public void scheduleRequest(RequestContext request) {
                        super.scheduleRequest(request);
                        due.set(System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(80));
                        runtime.taskTimer().schedule(this::schedule, 80L);
                    }

                    public void schedule() {
                        long now = System.nanoTime();
                        if (now < due.get()) throw new AssertionError("timer ran before deadline");
                        if (advance()) dispatched.set(now);
                        scheduled.countDown();
                    }
                }) {
            scheduler.start();
            var future =
                    scheduler.submit(RequestProtocolTestSupport.context(scheduler.config, 81006L));
            assertFalse(future.isDone(), "admission must not bypass the scheduler's deadline");
            assertFalse(future.get(2, TimeUnit.SECONDS).isSuccess());
            assertTrue(scheduled.await(2, TimeUnit.SECONDS));
            assertTrue(dispatched.get() >= due.get());
            verify(scheduler.selector).select(any(), isNull());
        }
    }

    @Test
    void customSchedulerOwnsEndpointSubscriptions() throws Exception {
        var availability = new CapacityNotifier();
        var changed = new java.util.concurrent.atomic.AtomicBoolean();
        var scheduled = new java.util.concurrent.CountDownLatch(1);
        try (var scheduler =
                new ManualScheduler(true) {
                    private final CapacityNotifier.Listener listener =
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

                    public void schedule() {
                        if (changed.get()) scheduled.countDown();
                    }
                }) {
            scheduler.start();
            var endpoint = mock(PrefillEndpoint.class);
            when(endpoint.getStatus())
                    .thenReturn(
                            org.flexlb.worker.status.WorkerStatus.createDiscovered(
                                    org.flexlb.config.service.RoleType.PREFILL,
                                    null,
                                    "127.0.0.1",
                                    8100,
                                    8101,
                                    null));
            when(endpoint.ipPort()).thenReturn("127.0.0.1:8100");
            availability.notifyChanged(endpoint);
            assertTrue(scheduled.await(2, TimeUnit.SECONDS));
        }
    }

    /**
     * Exposes manual scheduling turns while retaining the shared request lifecycle and shutdown facilities.
     */
    private static class ManualScheduler extends AbstractRequestScheduler {
        final FlexlbConfig config;
        final SchedulerTestServices runtime;
        final SchedulingQueue queue;
        final RequestRouteSelector selector = mock(RequestRouteSelector.class);

        ManualScheduler(boolean alternative) {
            this(alternative, true);
        }

        ManualScheduler(boolean alternative, boolean queued) {
            this(alternative, configuration(queued));
        }

        private ManualScheduler(boolean alternative, FlexlbConfig config) {
            this(alternative, config, mock(EndpointRegistry.class));
        }

        private ManualScheduler(
                boolean alternative, FlexlbConfig config, EndpointRegistry endpoints) {
            this(alternative, config, endpoints, facilities(config, endpoints));
        }

        private ManualScheduler(
                boolean alternative,
                FlexlbConfig config,
                EndpointRegistry endpoints,
                SchedulerTestServices runtime) {
            super(
                    runtime.requests(),
                    runtime.tasks,
                    runtime.deadlines(),
                    runtime.dispatcher,
                    runtime.responseCompletions(),
                    runtime.deliveryReporter,
                    mock(TheoryCacheHitReporter.class),
                    endpoints);
            this.config = config;
            this.runtime = runtime;
            queue = alternative ? new ListSchedulingQueue() : new OrderedRequestQueue(false);
            when(selector.select(any(), isNull()))
                    .thenReturn(
                            RequestResult.rejected(
                                    Response.error(StrategyErrorType.RESOURCE_EXHAUSTED)));
            SchedulerTestServices.bind(this, runtime);
            runtime.registerScheduler(this);
        }

        public void start() {}

        public void stop() {}

        public boolean awaitStopped(long deadlineNanos) {
            return true;
        }

        @Override
        public void onEndpointChanged(WorkerEndpoint endpoint) {}

        @Override
        public void scheduleRequest(RequestContext request) {
            synchronized (queue) {
                if (!request.getFuture().isDone()
                        && request.snapshot().cancellationReason() == null) {
                    queue.add(new GlobalQueueEntry(request, null));
                }
            }
        }

        @Override
        public void removeRequest(RequestContext request) {
            synchronized (queue) {
                for (var entry : queue.drain()) {
                    if (entry.context() != request) queue.restore(entry);
                }
            }
        }

        @Override
        public void close() {
            SchedulerTestSupport.shutdown(runtime);
        }

        @Override
        protected StrategyErrorType schedulingTimeoutError() {
            return config.isQueue()
                    ? StrategyErrorType.QUEUE_TIMEOUT
                    : StrategyErrorType.SCHEDULING_DEADLINE_EXCEEDED;
        }

        boolean advance() {
            GlobalQueueEntry selected;
            synchronized (queue) {
                var candidates = queue.scanForPlanningCandidates(1, 1, entry -> true);
                if (candidates.isEmpty()) return false;
                selected = candidates.getFirst();
                queue.remove(selected);
            }
            var routing = tryBeginRouteSelection(selected.context());
            if (routing == null) return true;
            org.flexlb.balance.schedule.RequestResult result;
            try {
                result = selector.select(selected.context(), selected.routingGroup());
            } finally {
                finishSelectionWithoutCommit(routing, null);
            }
            publish(selected.context(), result.failure(), false);
            return true;
        }

        private static FlexlbConfig configuration(boolean queued) {
            var config = SchedulingTestConfig.newConfig();
            if (!queued) {
                config.setScheduler(org.flexlb.config.ConfigTestSupport.direct());
            }
            return config;
        }

        private static SchedulerTestServices facilities(
                FlexlbConfig config, EndpointRegistry endpoints) {
            var service = mock(ConfigService.class);
            when(service.loadBalanceConfig()).thenReturn(config);
            var runtime =
                    new SchedulerTestServices(
                            new RequestRepository(),
                            endpoints,
                            mock(DeliveryMetricsReporter.class),
                            mock(RequestSchedulerReporter.class),
                            SchedulerTestSupport.dispatcher(),
                            service,
                            mock(EngineGrpcClient.class));
            return runtime;
        }
    }
}
