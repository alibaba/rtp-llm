package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.service.RecentCacheKeyTraceReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.concurrent.CompletableFuture;
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
            var result = scheduler.submit(context);
            var admission = scheduler.lifecycle.claimAdmissionHandle(81004L, result);
            assertNotNull(admission);
            try {
                scheduler.cancel(81004L, 0L, CancelReason.CLIENT_CANCELLED);
                assertFalse(scheduler.lifecycle.isAdmissionOpen(81004L, result));
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
            var result = scheduler.submit(context);
            assertFalse(result.isDone());
            scheduler.cancel(81001L, 0L, CancelReason.CLIENT_CANCELLED);
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
            var result = scheduler.submit(context);
            assertFalse(result.isDone());
            assertTrue(scheduler.advance());
            assertEquals(
                    StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(),
                    result.get(2, TimeUnit.SECONDS).getCode());
            verify(scheduler.selector).select(context, null);
        }
    }

    private static final class ManualScheduler
            implements RequestScheduler, RequestLifecycle.Scheduling, AutoCloseable {
        final FlexlbConfig config = SchedulingTestConfig.batchConfig();
        final SchedulerTestServices runtime;
        final RequestLifecycle lifecycle;
        final PlacementQueue queue;
        final RequestWorkerSelector selector = mock(RequestWorkerSelector.class);

        ManualScheduler(boolean alternative) {
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
                            this,
                            config,
                            SchedulingTestConfig.capacity(config),
                            mock(RecentCacheKeyTraceReporter.class));
            runtime.registerScheduler(this);
        }

        @Override
        public RequestLifecycle lifecycle() {
            return lifecycle;
        }

        @Override
        public CompletableFuture<Response> submit(RequestContext request) {
            var future = lifecycle.register(request, () -> {}, StrategyErrorType.BATCH_SLO_EXPIRED);
            synchronized (queue) {
                if (!future.isDone() && request.snapshot().cancellation() == null) {
                    queue.add(new GlobalQueueEntry(request, null));
                }
            }
            return future;
        }

        @Override
        public void remove(RequestContext request) {
            synchronized (queue) {
                for (var entry : queue.drain()) {
                    if (entry.context != request) queue.restore(entry);
                }
            }
        }

        @Override
        public boolean reschedule(RequestRoute previousRoute) {
            return false;
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
        public void start() {}

        @Override
        public boolean awaitSchedulingCompletion(long deadlineNanos) {
            synchronized (queue) {
                return queue.size() == 0;
            }
        }

        @Override
        public void close() {
            SchedulerTestSupport.shutdown(runtime);
        }
    }
}
