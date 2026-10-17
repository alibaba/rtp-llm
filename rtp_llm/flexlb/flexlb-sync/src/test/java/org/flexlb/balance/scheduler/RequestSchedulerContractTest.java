package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.eviction.DecodeCapacityAcquirer;
import org.flexlb.config.ConfigService;
import org.flexlb.config.ConfigTestSupport;
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
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.any;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class RequestSchedulerContractTest {
    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void cancellationRejectsForeignActiveAndTerminalRequestsInTheSharedRepository(boolean queued) throws Exception {
        try (Fixture f = new Fixture(queued)) {
            var other = new DirectRequestScheduler(f.router, f.requests.runtime, f.config, SchedulingTestConfig.capacity(f.config), org.mockito.Mockito.mock(org.flexlb.service.RecentCacheKeyTraceReporter.class));
            var ownContext = f.context(942L);
            var ownFuture = f.requests.register(ownContext, () -> { }, StrategyErrorType.BATCH_SLO_EXPIRED);
            var foreignContext = f.context(943L);
            var foreignFuture = other.register(foreignContext, () -> { }, StrategyErrorType.BATCH_SLO_EXPIRED);
            try {
                assertNull(other.cancel(942L, 0L, CancelReason.CLIENT_CANCELLED));
                assertNull(f.scheduler.cancel(943L, 0L, CancelReason.CLIENT_CANCELLED));
                assertFalse(ownFuture.isDone());
                assertFalse(foreignFuture.isDone());
                assertNull(SchedulerTestSupport.cancellationReason(ownContext));
                assertNull(SchedulerTestSupport.cancellationReason(foreignContext));

                assertNotNull(other.cancel(943L, 0L, CancelReason.CLIENT_CANCELLED));
                foreignFuture.get(3, TimeUnit.SECONDS);
                assertNull(f.scheduler.cancel(943L, 0L, CancelReason.CLIENT_CANCELLED));
                assertNotNull(other.cancel(943L, 0L, CancelReason.CLIENT_CANCELLED));
                assertNull(other.cancel(943L, 17L, CancelReason.CLIENT_CANCELLED));

                assertNotNull(f.scheduler.cancel(942L, 0L, CancelReason.CLIENT_CANCELLED));
                // Direct registration does not create a global queue entry; consume its control ticket explicitly.
                if (queued) { ((QueuedRequestScheduler) f.requests).queueControl(ownContext); }
                ownFuture.get(3, TimeUnit.SECONDS);
                assertNull(other.cancel(942L, 0L, CancelReason.CLIENT_CANCELLED));
                assertNotNull(f.scheduler.cancel(942L, 0L, CancelReason.CLIENT_CANCELLED));
            } finally {
                other.cancel(943L, 0L, CancelReason.CLIENT_CANCELLED);
            }
        }
    }

    @Test
    void cancellationResolvesTheNewOwnerAfterRequestIdReuse() throws Exception {
        try (Fixture f = new Fixture(false)) {
            var other = new DirectRequestScheduler(f.router, f.requests.runtime, f.config, SchedulingTestConfig.capacity(f.config), org.mockito.Mockito.mock(org.flexlb.service.RecentCacheKeyTraceReporter.class));
            var oldContext = f.context(944L);
            var oldFuture = other.register(oldContext, () -> { }, StrategyErrorType.BATCH_SLO_EXPIRED);
            try {
                other.cancel(944L, 0L, CancelReason.CLIENT_CANCELLED);
                oldFuture.get(3, TimeUnit.SECONDS);
                var repository = f.requests.requests;
                assertTrue(repository.removeExactTerminal(repository.findTerminal(944L), Long.MAX_VALUE));

                var replacement = f.context(944L);
                var future = f.requests.register(replacement, () -> { }, StrategyErrorType.BATCH_SLO_EXPIRED);
                assertNull(other.cancel(944L, 0L, CancelReason.CLIENT_CANCELLED));
                other.cancel(oldContext, 0L, CancelReason.CLIENT_CANCELLED);
                assertFalse(future.isDone());
                assertNull(SchedulerTestSupport.cancellationReason(replacement));
                assertNotNull(f.scheduler.cancel(944L, 0L, CancelReason.CLIENT_CANCELLED));
                future.get(3, TimeUnit.SECONDS);
            } finally {
                other.cancel(944L, 0L, CancelReason.CLIENT_CANCELLED);
            }
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void missingOwnerCannotActivateARequestOrPreventLaterRegistration(boolean queued) throws Exception {
        try (Fixture f = new Fixture(queued)) {
            var request = f.context(900);
            var originalFuture = request.data().getFuture();
            assertThrows(NullPointerException.class,
                    () -> f.requests.requests.register(request, null, new RequestFuture((a, b, c) -> false)));
            assertSame(originalFuture, request.data().getFuture());
            assertNull(request.data().getScheduler());
            assertNull(f.requests.requests.findActive(900));

            var future = f.requests.register(request, () -> { }, StrategyErrorType.BATCH_SLO_EXPIRED);
            try (var admission = f.requests.claimAdmissionHandle(900, future); var admissionCompletion1 = RequestProtocolTestSupport.finishOnExit(admission)) {
                assertNotNull(admission);
                f.scheduler.cancel(900, 0, CancelReason.CLIENT_CANCELLED);
            }
            future.get(3, TimeUnit.SECONDS);
            SchedulerTestSupport.runtime(f.scheduler).requests().stopAccepting();
            SchedulerTestSupport.runtime(f.scheduler).shutdown();

        }
    }

    @Test
    void staleTimerCannotReopenADrainedGeneration() throws Exception {
        try (Fixture f = new Fixture(false)) {
            var request = f.context(901);
            var future = f.requests.register(request, () -> { }, StrategyErrorType.BATCH_SLO_EXPIRED);
            f.requests.runtime.deadlines().update(request, request.snapshot().deadlines());
            ExpirationTimer.Token deadline = RequestProtocolTestSupport.field(request, "inactivityDeadline");
            assertNotNull(deadline);
            f.scheduler.cancel(901, 0, CancelReason.CLIENT_CANCELLED);
            future.get(3, TimeUnit.SECONDS);
            SchedulerTestSupport.runtime(f.scheduler).requests().stopAccepting();
            SchedulerTestSupport.runtime(f.scheduler).shutdown();

            f.requests.onDeadline(deadline, Long.MAX_VALUE);
            SchedulerTestSupport.awaitRequestWork(f.requests.runtime);
        }
    }

    @Test
    void queuedTerminationIncludesItsOwnWorkers() throws Exception {
        try (Fixture f = new Fixture(true)) {
            SchedulerTestSupport.runtime(f.scheduler).requests().stopAccepting();
            SchedulerTestSupport.runtime(f.scheduler).shutdown();

            var decision = (Thread) org.springframework.test.util.ReflectionTestUtils.getField(f.scheduler, "decisionThread");
            var planners = (java.util.concurrent.ExecutorService)
                    org.springframework.test.util.ReflectionTestUtils.getField(f.scheduler, "planners");
            assertFalse(decision.isAlive());
            assertTrue(planners.isTerminated());
        }
    }

    @Test
    void runtimeShutdownClosesWorkersEvenAfterEarlierCleanupFailure() throws Exception {
        try (Fixture f = new Fixture(true)) {
            var runtime = f.requests.runtime;
            var planners = (java.util.concurrent.ExecutorService)
                    org.springframework.test.util.ReflectionTestUtils.getField(f.scheduler, "planners");
            var decision = (Thread) org.springframework.test.util.ReflectionTestUtils.getField(f.scheduler, "decisionThread");
            var failure = new IllegalStateException("delivery cleanup failed");
            runtime.recordFailure(failure);
            runtime.requests().stopAccepting();
            assertFalse(planners.isShutdown(), "stopping intake does not execute shutdown");
            assertSame(failure, assertThrows(IllegalStateException.class, runtime::shutdown));
            assertTrue(planners.isTerminated());
            assertFalse(decision.isAlive());
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void stoppingFromPlacementDoesNotWaitForItsOwnSubmission(boolean queued) throws Exception {
        try (Fixture f = new Fixture(queued); var executor = Executors.newVirtualThreadPerTaskExecutor()) {
            var request = f.context(98);
            when(f.router.select(request, null)).thenAnswer(call -> {
                SchedulerTestSupport.runtime(f.scheduler).requests().stopAccepting();
                SchedulerTestSupport.runtime(f.scheduler).requests().stopAccepting();
                return rejection();
            });
            var submission = executor.submit(() -> f.scheduler.submit(request));
            assertFalse(submission.get(3, TimeUnit.SECONDS).get(3, TimeUnit.SECONDS).isSuccess());
            SchedulerTestSupport.runtime(f.scheduler).shutdown();

        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void acceptedSubmissionsCanEnterConcurrently(boolean queued) throws Exception {
        CountDownLatch entered = new CountDownLatch(2);
        CountDownLatch resume = new CountDownLatch(1);
        try (Fixture f = new Fixture(queued); var executor = Executors.newVirtualThreadPerTaskExecutor()) {
            Runnable pause = () -> {
                entered.countDown();
                RequestProtocolTestSupport.await(resume);
            };
            when(f.router.select(any(), any())).thenAnswer(call -> {
                if (!queued) { pause.run(); }
                return rejection();
            });
            var first = executor.submit(() -> queued
                    ? f.scheduler.submit(f.context(96), pause) : f.scheduler.submit(f.context(96)));
            var second = executor.submit(() -> queued
                    ? f.scheduler.submit(f.context(97), pause) : f.scheduler.submit(f.context(97)));
            try {
                assertTrue(entered.await(3, TimeUnit.SECONDS), "submissions must not serialize on an exclusive lock");
                SchedulerTestSupport.runtime(f.scheduler).requests().stopAccepting();

                assertFalse(f.scheduler.submit(f.context(95)).get(3, TimeUnit.SECONDS).isSuccess());
            } finally {
                resume.countDown();
            }
            first.get(3, TimeUnit.SECONDS).get(3, TimeUnit.SECONDS);
            second.get(3, TimeUnit.SECONDS).get(3, TimeUnit.SECONDS);
            SchedulerTestSupport.runtime(f.scheduler).shutdown();

        } finally {
            resume.countDown();
        }
    }

    @Test
    void fixedSchedulerPreservesRequestOwnershipAndAcceptance() throws Exception {
        try (Fixture f = new Fixture(false)) {
            var service = mock(ConfigService.class);
            when(service.loadBalanceConfig()).thenReturn(f.config);
            var reporter = mock(DeliveryMetricsReporter.class);
            var runtime = new SchedulerRuntime(new RequestRepository(), mock(EndpointRegistry.class),
                    reporter, mock(RequestSchedulerReporter.class), SchedulerTestSupport.dispatcher(),
                    service,
                    mock(EngineGrpcClient.class));
            runtime.registerScheduler(PlacementConfiguration.create(runtime, f.config,
                    f.router, reporter, mock(DecodeCapacityAcquirer.class), new PlacementAvailability(), SchedulingTestConfig.capacity(f.config), org.mockito.Mockito.mock(org.flexlb.service.RecentCacheKeyTraceReporter.class)));
            try {
                var owner = (AbstractRequestScheduler) SchedulerTestSupport.initializedScheduler(runtime);
                var request = f.context(100);
                var future = owner.register(request, () -> { }, StrategyErrorType.BATCH_SLO_EXPIRED);

                assertSame(owner, request.data().getScheduler());
                assertSame(owner, runtime.requests().ownerOf(100L));

                var next = f.context(101);
                var nextFuture = owner.register(next, () -> { }, StrategyErrorType.BATCH_SLO_EXPIRED);
                assertSame(owner, next.data().getScheduler());
                assertSame(owner, runtime.requests().ownerOf(101L));
                owner.cancel(100, 0, CancelReason.CLIENT_CANCELLED);
                owner.cancel(101, 0, CancelReason.CLIENT_CANCELLED);
                future.get(3, TimeUnit.SECONDS);
                nextFuture.get(3, TimeUnit.SECONDS);
                RequestProtocolTestSupport.awaitCondition(() -> runtime.requests().findTerminal(100L) != null
                        && runtime.requests().findTerminal(101L) != null);
                assertSame(owner, runtime.requests().findTerminal(100L).owner(), "terminal records preserve ownership too");
                assertSame(owner, runtime.requests().findTerminal(101L).owner());
            } finally {
                runtime.shutdown();
            }
        }
    }

    @Test
    void stoppingIntakeIsIdempotentAndRejectsNewRequests() throws Exception {
        try (Fixture f = new Fixture(false)) {
            RequestScheduler api = f.scheduler;

            var request = f.context(1);
            when(f.router.select(request, null)).thenReturn(rejection());
            assertEquals(StrategyErrorType.NO_PREFILL_WORKER.getErrorCode(),
                    api.submit(request).get(3, TimeUnit.SECONDS).getCode());
            SchedulerTestSupport.runtime(api).requests().stopAccepting();
            SchedulerTestSupport.runtime(api).requests().stopAccepting();
            SchedulerTestSupport.runtime(api).shutdown();

            assertEquals(StrategyErrorType.DISPATCH_FAILED.getErrorCode(),
                    api.submit(f.context(2)).join().getCode());
            verify(f.router, times(1)).select(any(), any());
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void stopRejectsNewRequestsButLetsAcceptedPlacementFinish(boolean queued) throws Exception {
        CountDownLatch selecting = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        try (Fixture f = new Fixture(queued); var executor = Executors.newVirtualThreadPerTaskExecutor()) {
            RequestScheduler api = f.scheduler;
            var request = f.context(10);
            when(f.router.select(request, null)).thenAnswer(invocation -> {
                selecting.countDown();
                RequestProtocolTestSupport.await(release);
                return rejection();
            });
            var submission = executor.submit(() -> api.submit(request));
            try {
                assertTrue(selecting.await(3, TimeUnit.SECONDS));
                SchedulerTestSupport.runtime(api).requests().stopAccepting();

                assertFalse(SchedulerTestSupport.repository(f.requests).isClosed(), "draining must not close accepted admission");
                assertEquals(StrategyErrorType.DISPATCH_FAILED.getErrorCode(), api.submit(f.context(11)).join().getCode());
            } finally {
                release.countDown();
            }
            var future = submission.get(3, TimeUnit.SECONDS);
            assertSame(request.data().getFuture(), future);
            assertEquals(StrategyErrorType.NO_PREFILL_WORKER.getErrorCode(), future.get(3, TimeUnit.SECONDS).getCode());
            SchedulerTestSupport.runtime(api).shutdown();

            assertEquals(0, SchedulerTestSupport.repository(f.requests).liveRequestCount());
        } finally {
            release.countDown();
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void registeredWorkCanStartAdmissionAfterStopAndCancellationStillWorks(boolean queued) throws Exception {
        try (Fixture f = new Fixture(queued)) {
            RequestScheduler api = f.scheduler;
            var request = f.context(20);
            var future = f.requests.register(request, () -> { }, StrategyErrorType.BATCH_SLO_EXPIRED);
            SchedulerTestSupport.runtime(api).requests().stopAccepting();
            try (var admission = f.requests.claimAdmissionHandle(20, future); var admissionCompletion3 = RequestProtocolTestSupport.finishOnExit(admission)) {
                assertNotNull(admission);
                assertNotNull(api.cancel(20, 0, CancelReason.CLIENT_CANCELLED));

            }
            assertEquals(StrategyErrorType.REQUEST_CANCELLED.getErrorCode(), future.get(3, TimeUnit.SECONDS).getCode());
            SchedulerTestSupport.runtime(api).shutdown();

            assertNotNull(api.cancel(20, 0, CancelReason.CLIENT_CANCELLED));
            assertNull(api.cancel(999, 0, CancelReason.CLIENT_CANCELLED));
        }
    }

    @Test
    void cancelledFutureDoesNotHideAnOutstandingAdmission() throws Exception {
        try (Fixture f = new Fixture(true)) {
            RequestScheduler api = f.scheduler;
            var request = f.context(30);
            var future = f.requests.register(request, () -> { }, StrategyErrorType.BATCH_SLO_EXPIRED);
            try (var admission = f.requests.claimAdmissionHandle(30, future); var admissionCompletion4 = RequestProtocolTestSupport.finishOnExit(admission)) {
                assertNotNull(admission);
                assertTrue(future.cancel(true));
                SchedulerTestSupport.runtime(api).requests().stopAccepting();
                assertTrue(future.isDone());

            }
            SchedulerTestSupport.runtime(api).shutdown();

            assertEquals(0, SchedulerTestSupport.repository(f.requests).liveRequestCount());
        }
    }

    @Test
    void stopFromPublicationCallbackDoesNotDeadlockOrFinishBeforePublicationDrains() throws Exception {
        CountDownLatch publishing = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        try (Fixture f = new Fixture(false)) {
            RequestScheduler api = f.scheduler;
            var request = f.context(40);
            var future = f.requests.register(request, () -> { }, StrategyErrorType.BATCH_SLO_EXPIRED);
            CompletableFuture<Response> callback = future.whenComplete((result, failure) -> {
                SchedulerTestSupport.runtime(api).requests().stopAccepting();
                publishing.countDown();
                RequestProtocolTestSupport.await(release);
            });
            CompletableFuture<Void> shutdown = null;
            try {
                assertTrue(f.requests.publish(request, Response.error(StrategyErrorType.NO_PREFILL_WORKER), false));
                assertTrue(publishing.await(3, TimeUnit.SECONDS));
                assertTrue(future.isDone());
                assertEquals(0, SchedulerTestSupport.repository(f.requests).liveRequestCount());
                var runtime = SchedulerTestSupport.runtime(api);
                shutdown = CompletableFuture.runAsync(runtime::shutdown);
                RequestProtocolTestSupport.awaitCondition(() -> runtime.requests().isClosed());
                var closing = shutdown;
                assertThrows(java.util.concurrent.TimeoutException.class,
                        () -> closing.get(100, TimeUnit.MILLISECONDS), "shutdown must await the active publication");

            } finally {
                release.countDown();
            }
            callback.get(3, TimeUnit.SECONDS);
            shutdown.get(3, TimeUnit.SECONDS);

        } finally {
            release.countDown();
        }
    }

    private static PlacementResult<RequestRoute, PlacementKey> rejection() {
        return PlacementResult.rejected(Response.error(StrategyErrorType.NO_PREFILL_WORKER));
    }

    @Test
    void cleanupFailureIsRetainedForShutdownReporting() throws Exception {
        try (Fixture f = new Fixture(false)) {
            RequestScheduler api = f.scheduler;
            var request = f.context(50);
            var future = f.requests.register(request, () -> { }, StrategyErrorType.BATCH_SLO_EXPIRED);
            var endpoint = mock(PrefillEndpoint.class);
            var route = SchedulingTestConfig.createRoute(request, new Response(),
                    null, null, endpoint, null, null, System.currentTimeMillis());
            try (var admission = f.requests.claimAdmissionHandle(50, future); var admissionCompletion5 = RequestProtocolTestSupport.finishOnExit(admission)) {
                assertNotNull(admission);
                assertEquals(PlacementResult.Status.SUCCESS, f.requests.commitRoute(route, RequestProtocolTestSupport.publication(() -> true)));
            }
            var failure = new IllegalStateException("endpoint release failed");
            doThrow(failure).when(endpoint).release(null, route);
            SchedulerTestSupport.runtime(api).requests().stopAccepting();
            assertTrue(future.completeExceptionally(new IllegalStateException("request failed")));
            SchedulerTestSupport.awaitRequestWork(f.requests.runtime);
            assertSame(failure, SchedulerTestSupport.failure(api));
            assertTrue(future.isCompletedExceptionally(), "cleanup failure must not revoke the selected response");
            assertEquals(SchedulerTestSupport.Stage.FINALIZING, SchedulerTestSupport.stage(request));
            verify(endpoint, times(3)).release(null, route);
        }
    }

    @Test
    void admissionFailureIsRecordedAndReleasesDrainGate() throws Exception {
        try (Fixture f = new Fixture(false)) {
            var context = f.context(940L);
            var future = f.requests.register(context, () -> { }, StrategyErrorType.BATCH_SLO_EXPIRED);
            var handle = f.requests.claimAdmissionHandle(940L, future);
            assertNotNull(handle);
            var originalTimer = f.requests.runtime.deadlines();
            var timer = org.mockito.Mockito.spy(originalTimer);
            var failure = new IllegalStateException("expiry attachment failed");
            doThrow(failure).when(timer).update(org.mockito.ArgumentMatchers.eq(context), org.mockito.ArgumentMatchers.any());
            org.springframework.test.util.ReflectionTestUtils.setField(f.requests.runtime, "timer", timer);
            try {
                assertSame(failure, assertThrows(IllegalStateException.class, handle::finish));
                assertSame(failure, SchedulerTestSupport.failure(f.requests));
                f.requests.runtime.awaitAdmissions();
                assertEquals(0L, org.springframework.test.util.ReflectionTestUtils.getField(
                        f.requests.runtime, "admissions"));
            } finally {
                org.springframework.test.util.ReflectionTestUtils.setField(f.requests.runtime, "timer", originalTimer);
                f.scheduler.cancel(940L, 0L, CancelReason.CLIENT_CANCELLED);
            }
        }
    }

    private static final class Fixture implements AutoCloseable {
        final FlexlbConfig config = SchedulingTestConfig.newConfig();
        final RequestWorkerSelector router = mock(RequestWorkerSelector.class);
        final AbstractRequestScheduler requests;
        final RequestScheduler scheduler;

        Fixture(boolean queued) {
            if (queued) { SchedulingTestConfig.useFifoQueue(config); } else { config.setScheduler(ConfigTestSupport.direct()); }
            SchedulingTestConfig.useNonBatchDispatcher(config);
            ConfigService service = mock(ConfigService.class);
            when(service.loadBalanceConfig()).thenReturn(config);
            var reporter = mock(DeliveryMetricsReporter.class);
            requests = SchedulerTestSupport.create(service, reporter, mock(RequestSchedulerReporter.class),
                    mock(RecentCacheKeyTraceReporter.class));
            scheduler = SchedulerTestSupport.configure(requests, service.loadBalanceConfig(), router, reporter, mock(DecodeCapacityAcquirer.class), new PlacementAvailability());
        }

        RequestContext context(long id) { return RequestProtocolTestSupport.context(config, id); }

        @Override
        public void close() {
            SchedulerTestSupport.repository(requests).closeRegistration();
            RequestProtocolTestSupport.close(scheduler);
            requests.runtime.awaitAdmissions();
            SchedulerTestSupport.timer(SchedulerTestSupport.runtime(requests)).close();
            SchedulerTestSupport.awaitRequestWork(requests.runtime);
            requests.closeOutstandingAndTerminalize();
            SchedulerTestSupport.closeRequestExecutors(SchedulerTestSupport.runtime(requests));
        }
    }
}
