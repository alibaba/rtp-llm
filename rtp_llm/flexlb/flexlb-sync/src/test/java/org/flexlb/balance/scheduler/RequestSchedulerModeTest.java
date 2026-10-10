package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.eviction.DecodeCapacityAcquirer;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.SchedulerConfig;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.service.RecentCacheKeyTraceReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/** Public submission and cancellation contracts against both scheduler modes. */
class RequestSchedulerModeTest {
    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void acceptedSelectionFailureSettlesOriginalFutureWithoutRetry(boolean queued) throws Exception {
        try (Fixture f = new Fixture(queued)) {
            when(f.router.select(f.context, null)).thenThrow(new IllegalStateException("selection failed"));
            var original = f.scheduler.submit(f.context);
            assertSame(original, f.context.getFuture());
            assertEquals(StrategyErrorType.DISPATCH_FAILED.getErrorCode(), original.get(3, TimeUnit.SECONDS).getCode());
            f.requests.awaitAdmissionMutations();
            verify(f.router, times(1)).select(f.context, null);
            assertEquals(0, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(f.requests).liveRequestCount());
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void policyResolutionFailureIsOwnedWithoutAQueueEntryOrFallback(boolean queued) throws Exception {
        try (Fixture f = new Fixture(queued)) {
            when(f.router.resolvePolicyGroup(f.context)).thenThrow(new IllegalStateException("policy unavailable"));
            var original = f.scheduler.submit(f.context);
            assertSame(original, f.context.getFuture());
            assertEquals(StrategyErrorType.DISPATCH_FAILED.getErrorCode(), original.get(3, TimeUnit.SECONDS).getCode());
            f.requests.awaitAdmissionMutations();
            assertEquals(0, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(f.requests).liveRequestCount());
            if (f.queue != null) { assertEquals(0, RequestProtocolTestSupport.queuedCount(f.queue)); }
            verify(f.router, never()).select(f.context, null);
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void cancellationDuringSelectionWinsOverLateFailure(boolean queued) throws Exception {
        CountDownLatch selecting = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        try (Fixture f = new Fixture(queued); var executor = Executors.newSingleThreadExecutor()) {
            when(f.router.select(f.context, null)).thenAnswer(invocation -> {
                selecting.countDown();
                assertTrue(release.await(3, TimeUnit.SECONDS));
                throw new IllegalStateException("late selection failure");
            });
            var submitted = executor.submit(() -> f.scheduler.submit(f.context));
            try {
                assertTrue(selecting.await(3, TimeUnit.SECONDS));
                f.scheduler.cancel(f.context.getRequestId(), 0, CancelReason.CLIENT_CANCELLED);
            } finally {
                release.countDown();
            }
            var original = submitted.get(3, TimeUnit.SECONDS);
            assertSame(original, f.context.getFuture());
            assertEquals(StrategyErrorType.REQUEST_CANCELLED.getErrorCode(), original.get(3, TimeUnit.SECONDS).getCode());
            f.requests.awaitAdmissionMutations();
            verify(f.router, times(1)).select(f.context, null);
        } finally {
            release.countDown();
        }
    }

    @Test
    void fixedModeRejectsMismatchedInputAndAnotherSchedulerCanShareLifecycle() throws Exception {
        try (Fixture direct = new Fixture(false)) {
            var queueConfig = SchedulingTestConfig.batchConfig();
            var laterQueue = RequestProtocolTestSupport.context(queueConfig, 8762L);
            laterQueue.setGenerateInputPb(com.google.protobuf.ByteString.copyFromUtf8("input"));
            assertEquals(StrategyErrorType.DISPATCH_FAILED.getErrorCode(), direct.scheduler.submit(laterQueue).join().getCode());
            assertEquals(0, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(direct.requests).liveRequestCount());
        }
        try (Fixture queued = new Fixture(true)) {
            FlexlbConfig directConfig = new FlexlbConfig();
            directConfig.setScheduler(SchedulerConfig.direct());
            SchedulingTestConfig.useNonBatchDispatcher(directConfig);
            var laterDirect = RequestProtocolTestSupport.context(directConfig, 8763L);
            when(queued.router.select(laterDirect, null)).thenReturn(
                    PlacementResult.rejected(Response.error(StrategyErrorType.NO_PREFILL_WORKER)));
            var future = new DirectRequestScheduler(queued.router, org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(queued.requests), directConfig).submit(laterDirect);
            assertSame(future, laterDirect.getFuture());
            assertEquals(StrategyErrorType.NO_PREFILL_WORKER.getErrorCode(), future.get(3, TimeUnit.SECONDS).getCode());
            verify(queued.router, times(1)).select(laterDirect, null);
            assertEquals(0, RequestProtocolTestSupport.queuedCount(queued.queue));
        }
    }

    @ParameterizedTest
    @org.junit.jupiter.params.provider.MethodSource("entryFailureCases")
    void entryFailureMatrixPreservesOwnershipAndDrainsEveryConfiguration(boolean queued, boolean priority,
            boolean fixedWindow, boolean batch, String outcome) throws Exception {
        try (Fixture f = new Fixture(queued, priority, fixedWindow, batch)) {
            when(f.router.select(f.context, null)).thenReturn(PlacementResult.rejected(Response.error(StrategyErrorType.NO_PREFILL_WORKER)));
            if (outcome.equals("THROW")) {
                when(f.router.select(f.context, null)).thenThrow(new IllegalStateException("selection failed"));
            }
            if (outcome.equals("EXPIRED")) {
                f.context.setSchedulingMetadata(org.flexlb.dao.SchedulingMetadata.explicit(50, System.currentTimeMillis() - 1L));
            }
            if (outcome.equals("STOPPED")) { SchedulerTestSupport.runtime(f.scheduler).stopAccepting(); }
            var future = f.scheduler.submit(f.context);
            StrategyErrorType expected = switch (outcome) {
                case "REJECTED" -> StrategyErrorType.NO_PREFILL_WORKER;
                case "THROW", "STOPPED" -> StrategyErrorType.DISPATCH_FAILED;
                case "EXPIRED" -> queued ? StrategyErrorType.RESOURCE_EXHAUSTED : StrategyErrorType.BATCH_SLO_EXPIRED;
                default -> throw new AssertionError(outcome);
            };
            assertEquals(expected.getErrorCode(), future.get(5, TimeUnit.SECONDS).getCode());
            if (!outcome.equals("STOPPED")) { assertSame(f.context.getFuture(), future); }
            verify(f.router, times(outcome.equals("REJECTED") || outcome.equals("THROW") ? 1 : 0)).select(f.context, null);
            f.requests.awaitAdmissionMutations();
            assertEquals(0, f.requests.requests.liveRequestCount());
            SchedulerTestSupport.runtime(f.scheduler).stopAccepting();
            SchedulerTestSupport.runtime(f.scheduler).shutdown();

            if (f.queue != null) { assertEquals(0, RequestProtocolTestSupport.queuedCount(f.queue)); }
        }
    }

    static java.util.stream.Stream<org.junit.jupiter.params.provider.Arguments> entryFailureCases() {
        var queueCases = java.util.stream.IntStream.range(0, 8).boxed().flatMap(bits ->
                java.util.stream.Stream.of("REJECTED", "THROW", "EXPIRED", "STOPPED").map(outcome ->
                        org.junit.jupiter.params.provider.Arguments.of(true, (bits & 1) != 0, (bits & 2) != 0, (bits & 4) != 0, outcome)));
        var directCases = java.util.stream.Stream.of("REJECTED", "THROW", "EXPIRED", "STOPPED").map(outcome ->
                org.junit.jupiter.params.provider.Arguments.of(false, false, false, false, outcome));
        return java.util.stream.Stream.concat(queueCases, directCases);
    }

    private static final class Fixture implements AutoCloseable {
        final RequestWorkerSelector router = mock(RequestWorkerSelector.class);
        final AbstractRequestScheduler requests;
        final RequestContext context;
        final QueuedRequestScheduler queue;
        final RequestScheduler scheduler;

        Fixture(boolean queued) {
            this(queued, false, true, queued);
        }

        Fixture(boolean queued, boolean priority, boolean fixedWindow, boolean batch) {
            FlexlbConfig config = SchedulingTestConfig.batchConfig();
            if (queued) {
                if (priority) { SchedulingTestConfig.usePriorityQueue(config); }
                else { SchedulingTestConfig.useFifoQueue(config); }
                if (fixedWindow) { SchedulingTestConfig.useFixedWindowDecision(config); }
                else { SchedulingTestConfig.useSingleDecision(config); }
                if (!batch) { SchedulingTestConfig.useNonBatchDispatcher(config); }
            }
            if (!queued) {
                config.setScheduler(SchedulerConfig.direct());
                SchedulingTestConfig.useNonBatchDispatcher(config);
            }
            ConfigService service = mock(ConfigService.class);
            when(service.loadBalanceConfig()).thenReturn(config);
            var reporter = mock(DeliveryMetricsReporter.class);
            requests = org.flexlb.balance.scheduler.SchedulerTestSupport.create(service, reporter, mock(RequestSchedulerReporter.class),
                    mock(RecentCacheKeyTraceReporter.class));
            scheduler = org.flexlb.balance.scheduler.SchedulerTestSupport.configure(requests, config, router, reporter,
                    mock(DecodeCapacityAcquirer.class), new PlacementAvailability());
            queue = scheduler instanceof QueuedRequestScheduler queuedScheduler ? queuedScheduler : null;
            context = RequestProtocolTestSupport.context(config, 8761L);
        }

        @Override
        public void close() {
            SchedulerTestSupport.runtime(scheduler).stopAccepting();
            org.flexlb.balance.scheduler.SchedulerTestSupport.repository(requests).closeRegistration();
            if (queue != null) { queue.close(); }
            requests.awaitAdmissionMutations();
            org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(requests).timer().close();
            requests.runtime.continuations().awaitIdle();
            requests.closeOutstandingAndTerminalize();
            org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(requests).closeRequestExecutors();
        }
    }
}
