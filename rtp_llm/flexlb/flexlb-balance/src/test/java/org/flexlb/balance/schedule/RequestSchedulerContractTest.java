package org.flexlb.balance.schedule;

import org.flexlb.balance.metrics.DeliveryMetricsReporter;
import org.flexlb.balance.metrics.RequestSchedulerReporter;
import org.flexlb.balance.requestlog.TheoryCacheHitReporter;
import org.flexlb.balance.schedule.ResponseCompletionExecutor.RequestFuture;
import org.flexlb.balance.strategy.RequestRouteSelector;
import org.flexlb.balance.strategy.RoutingResult;
import org.flexlb.config.ConfigService;
import org.flexlb.config.ConfigTestSupport;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.service.RoleType;
import org.flexlb.worker.endpoint.CapacityKey;
import org.flexlb.worker.endpoint.CapacityNotifier;
import org.flexlb.worker.endpoint.EndpointRegistry;
import org.flexlb.worker.endpoint.PrefillEndpoint;
import org.flexlb.worker.endpoint.Response;
import org.flexlb.worker.endpoint.StrategyErrorType;
import org.flexlb.worker.rpc.EngineGrpcClient;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
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

/**
 * Exercises request ownership, cancellation, admission, and shutdown through both direct and queued schedulers.
 */
class RequestSchedulerContractTest {
    @Test
    void waitingCountIncludesRegisteredRequestsBeforeTheFirstDecision() throws Exception {
        try (Fixture f = new Fixture(true, false)) {
            var context = f.context(945L);
            var future = f.scheduler.submit(context);
            assertFalse(future.isDone());
            assertEquals(1, f.scheduler.blockedRequestCount());
            f.scheduler.cancel(context, CancelReason.CLIENT_CANCELLED);
            RequestProtocolTestSupport.awaitCondition(() -> f.scheduler.blockedRequestCount() == 0);
        }
    }

    @Test
    void remoteCleanupDoesNotCompareRequestSnapshots() {
        var scheduler =
                mock(AbstractRequestScheduler.class, org.mockito.Mockito.CALLS_REAL_METHODS);
        var context = mock(RequestContext.class);
        var route = mock(RequestResult.class);
        when(route.ctx()).thenReturn(context);
        when(context.getRequestScheduler()).thenReturn(scheduler);
        var cleanup =
                new RequestContext.RequestEvent(
                        route,
                        null,
                        null,
                        null,
                        org.flexlb.worker.endpoint.RequestEventType.REMOTE_CLEANED,
                        0L,
                        null,
                        1000L);

        scheduler.onRequestEvent(context, cleanup);

        verify(context).recordEvent(cleanup);
        verify(context, org.mockito.Mockito.never()).snapshot();
    }

    @ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(strings = {"cancel", "timeout", "failure"})
    void endingAWaitingRequestRemovesItFromScheduling(String ending) throws Exception {
        try (Fixture f = new Fixture(true)) {
            when(f.router.select(any(), any()))
                    .thenReturn(RequestResult.blocked(CapacityKey.anyGroup(RoleType.PREFILL)));
            var context = f.context(944L);
            var future = f.scheduler.submit(context);
            RequestProtocolTestSupport.awaitCondition(() -> f.scheduler.blockedRequestCount() == 1);

            switch (ending) {
                case "cancel" -> f.scheduler.cancel(context, CancelReason.CLIENT_CANCELLED);
                case "timeout" -> {
                    var deadline =
                            context.snapshot().deadlines().token(ExpirationTimer.Kind.REQUEST);
                    assertNotNull(deadline);
                    f.scheduler.onDeadline(deadline, deadline.atMs());
                }
                case "failure" ->
                        assertTrue(
                                future.completeExceptionally(
                                        new IllegalStateException("frontend failed")));
                default -> throw new AssertionError(ending);
            }

            RequestProtocolTestSupport.awaitCondition(
                    () ->
                            future.isDone()
                                    && f.scheduler.blockedRequestCount() == 0
                                    && SchedulerTestSupport.repository(f.scheduler)
                                                    .liveRequestCount()
                                            == 0);
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void cancellationRejectsForeignActiveAndTerminalRequestsInTheSharedRepository(boolean queued)
            throws Exception {
        try (Fixture f = new Fixture(queued)) {
            var other =
                    SchedulerTestServices.direct(
                            f.router,
                            org.flexlb.balance.schedule.SchedulerTestSupport.runtime(f.requests)
                                    .newLifecycle(
                                            f.config,
                                            SchedulingTestConfig.capacity(f.config),
                                            org.mockito.Mockito.mock(
                                                    org.flexlb.balance.requestlog
                                                            .TheoryCacheHitReporter.class)));
            var ownContext = f.context(942L);
            org.mockito.Mockito.doNothing().when(f.requests).scheduleRequest(ownContext);
            var ownFuture = f.requests.submit(ownContext, () -> {});
            var foreignConfig = SchedulingTestConfig.newConfig();
            foreignConfig.setScheduler(ConfigTestSupport.direct());
            var foreignContext = RequestProtocolTestSupport.context(foreignConfig, 943L);
            var foreignFuture =
                    SchedulerTestServices.lifecycle(other).submit(foreignContext, () -> {});
            try {
                assertNull(
                        SchedulerTestServices.lifecycle(other)
                                .cancel("942", CancelReason.CLIENT_CANCELLED));
                assertNull(
                        SchedulerTestServices.entry(f.scheduler)
                                .cancel("943", CancelReason.CLIENT_CANCELLED));
                assertFalse(ownFuture.isDone());
                assertFalse(foreignFuture.isDone());
                assertNull(SchedulerTestSupport.cancellationReason(ownContext));
                assertNull(SchedulerTestSupport.cancellationReason(foreignContext));

                assertNotNull(
                        SchedulerTestServices.lifecycle(other)
                                .cancel("943", CancelReason.CLIENT_CANCELLED));
                foreignFuture.get(3, TimeUnit.SECONDS);
                assertNull(
                        SchedulerTestServices.entry(f.scheduler)
                                .cancel("943", CancelReason.CLIENT_CANCELLED));
                assertNotNull(
                        SchedulerTestServices.lifecycle(other)
                                .cancel("943", CancelReason.CLIENT_CANCELLED));

                assertNotNull(
                        SchedulerTestServices.entry(f.scheduler)
                                .cancel("942", CancelReason.CLIENT_CANCELLED));
                // Direct registration does not create a global queue entry; consume its control
                // ticket explicitly.
                if (queued) {
                    ((QueuedRequestScheduler)
                                    org.flexlb.balance.schedule.SchedulerTestServices.scheduler(
                                            f.requests))
                            .removeRequest(ownContext);
                }
                ownFuture.get(3, TimeUnit.SECONDS);
                assertNull(
                        SchedulerTestServices.lifecycle(other)
                                .cancel("942", CancelReason.CLIENT_CANCELLED));
                assertNotNull(
                        SchedulerTestServices.entry(f.scheduler)
                                .cancel("942", CancelReason.CLIENT_CANCELLED));
            } finally {
                SchedulerTestServices.lifecycle(other).cancel("943", CancelReason.CLIENT_CANCELLED);
            }
        }
    }

    @Test
    void cancellationResolvesTheNewOwnerAfterRequestIdReuse() throws Exception {
        try (Fixture f = new Fixture(false)) {
            var other =
                    SchedulerTestServices.direct(
                            f.router,
                            org.flexlb.balance.schedule.SchedulerTestSupport.runtime(f.requests)
                                    .newLifecycle(
                                            f.config,
                                            SchedulingTestConfig.capacity(f.config),
                                            org.mockito.Mockito.mock(
                                                    org.flexlb.balance.requestlog
                                                            .TheoryCacheHitReporter.class)));
            var oldContext = f.context(944L);
            var oldFuture = SchedulerTestServices.lifecycle(other).submit(oldContext, () -> {});
            try {
                SchedulerTestServices.lifecycle(other).cancel("944", CancelReason.CLIENT_CANCELLED);
                oldFuture.get(3, TimeUnit.SECONDS);
                var repository = f.requests.repository;
                assertTrue(
                        repository.removeExactTerminal(
                                repository.findTerminal("944"), Long.MAX_VALUE));

                var replacement = f.context(944L);
                var future = f.requests.submit(replacement, () -> {});
                assertNull(
                        SchedulerTestServices.lifecycle(other)
                                .cancel("944", CancelReason.CLIENT_CANCELLED));
                SchedulerTestServices.lifecycle(other)
                        .cancel(oldContext, CancelReason.CLIENT_CANCELLED);
                assertFalse(future.isDone());
                assertNull(SchedulerTestSupport.cancellationReason(replacement));
                assertNotNull(
                        SchedulerTestServices.entry(f.scheduler)
                                .cancel("944", CancelReason.CLIENT_CANCELLED));
                future.get(3, TimeUnit.SECONDS);
            } finally {
                SchedulerTestServices.lifecycle(other).cancel("944", CancelReason.CLIENT_CANCELLED);
            }
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void missingOwnerCannotActivateARequestOrPreventLaterRegistration(boolean queued)
            throws Exception {
        try (Fixture f = new Fixture(queued)) {
            var request = f.context(900);
            var originalFuture = request.getFuture();
            assertThrows(
                    NullPointerException.class,
                    () ->
                            f.requests.repository.register(
                                    request,
                                    null,
                                    new RequestFuture((a, b, c) -> false),
                                    60_000L,
                                    org.flexlb.worker.endpoint.StrategyErrorType
                                            .SCHEDULING_DEADLINE_EXCEEDED));
            assertSame(originalFuture, request.getFuture());
            assertNull(request.getRequestScheduler());
            assertNull(f.requests.repository.findActive("900"));

            var future = f.requests.submit(request, () -> {});
            var admission = SchedulerTestSupport.claim(f.requests, "900", future);
            try (var admissionCompletion1 = RequestProtocolTestSupport.finishOnExit(admission)) {
                assertNotNull(admission);
                SchedulerTestServices.entry(f.scheduler)
                        .cancel("900", CancelReason.CLIENT_CANCELLED);
            }
            future.get(3, TimeUnit.SECONDS);
            SchedulerTestSupport.runtime(f.scheduler).requests().stopAccepting();
            SchedulerTestSupport.shutdown(SchedulerTestSupport.runtime(f.scheduler));
        }
    }

    @Test
    void staleTimerCannotReopenADrainedGeneration() throws Exception {
        try (Fixture f = new Fixture(false)) {
            var request = f.context(901);
            var future = f.requests.submit(request, () -> {});
            org.flexlb.balance.schedule.SchedulerTestSupport.runtime(f.requests)
                    .deadlines()
                    .update(request, request.snapshot().deadlines());
            ExpirationTimer.Token deadline =
                    RequestProtocolTestSupport.field(request, "inactivityDeadline");
            assertNotNull(deadline);
            SchedulerTestServices.entry(f.scheduler).cancel("901", CancelReason.CLIENT_CANCELLED);
            future.get(3, TimeUnit.SECONDS);
            SchedulerTestSupport.runtime(f.scheduler).requests().stopAccepting();
            SchedulerTestSupport.shutdown(SchedulerTestSupport.runtime(f.scheduler));

            f.requests.onDeadline(deadline, Long.MAX_VALUE);
            SchedulerTestSupport.awaitRequestWork(
                    org.flexlb.balance.schedule.SchedulerTestSupport.runtime(f.requests));
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void stopWaitsForAcceptedSelectionWork(boolean queued) throws Exception {
        var entered = new CountDownLatch(1);
        var resume = new CountDownLatch(1);
        try (Fixture f = new Fixture(queued)) {
            when(f.router.select(any(), any()))
                    .thenAnswer(
                            call -> {
                                entered.countDown();
                                assertTrue(resume.await(3, TimeUnit.SECONDS));
                                return rejection();
                            });
            var response = f.requests.submit(f.context(945L));
            try {
                assertTrue(entered.await(3, TimeUnit.SECONDS));
                f.scheduler.stop();
                assertFalse(
                        f.scheduler.awaitStopped(
                                System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(20)));
            } finally {
                resume.countDown();
            }
            assertTrue(f.scheduler.awaitStopped(System.nanoTime() + TimeUnit.SECONDS.toNanos(3)));
            f.requests.cancel("945", CancelReason.SHUTDOWN);
            assertFalse(response.get(3, TimeUnit.SECONDS).isSuccess());
        } finally {
            resume.countDown();
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void acceptedSelectionCannotWaitForItself(boolean queued) throws Exception {
        try (Fixture f = new Fixture(queued)) {
            var selfWait = new CompletableFuture<Boolean>();
            when(f.router.select(any(), any()))
                    .thenAnswer(
                            call -> {
                                f.scheduler.stop();
                                selfWait.complete(
                                        f.scheduler.awaitStopped(
                                                System.nanoTime() + TimeUnit.SECONDS.toNanos(2)));
                                return rejection();
                            });
            var response = f.requests.submit(f.context(946L));
            assertFalse(selfWait.get(500, TimeUnit.MILLISECONDS));
            f.requests.cancel("946", CancelReason.SHUTDOWN);
            assertFalse(response.get(3, TimeUnit.SECONDS).isSuccess());
            assertTrue(f.scheduler.awaitStopped(System.nanoTime() + TimeUnit.SECONDS.toNanos(3)));
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void closeFromAcceptedSelectionCannotClaimToHaveDrained(boolean queued) throws Exception {
        try (Fixture f = new Fixture(queued)) {
            var selfClose = new CompletableFuture<Throwable>();
            when(f.router.select(any(), any()))
                    .thenAnswer(
                            call -> {
                                try {
                                    f.scheduler.close();
                                    selfClose.complete(null);
                                } catch (Throwable failure) {
                                    selfClose.complete(failure);
                                }
                                return rejection();
                            });
            var response = f.requests.submit(f.context(947L));
            org.junit.jupiter.api.Assertions.assertInstanceOf(
                    IllegalStateException.class, selfClose.get(500, TimeUnit.MILLISECONDS));
            f.requests.cancel("947", CancelReason.SHUTDOWN);
            assertFalse(response.get(3, TimeUnit.SECONDS).isSuccess());
            assertTrue(f.scheduler.awaitStopped(System.nanoTime() + TimeUnit.SECONDS.toNanos(3)));
        }
    }

    @Test
    void queuedTerminationIncludesItsOwnWorkers() throws Exception {
        try (Fixture f = new Fixture(true)) {
            SchedulerTestSupport.runtime(f.scheduler).requests().stopAccepting();
            SchedulerTestSupport.shutdown(SchedulerTestSupport.runtime(f.scheduler));

            var planners =
                    (java.util.concurrent.ExecutorService)
                            org.springframework.test.util.ReflectionTestUtils.getField(
                                    f.scheduler, "planners");
            assertTrue(f.scheduler.awaitStopped(System.nanoTime()));
            assertTrue(planners.isTerminated());
        }
    }

    @Test
    void runtimeShutdownClosesWorkersEvenAfterEarlierCleanupFailure() throws Exception {
        try (Fixture f = new Fixture(true)) {
            var runtime = org.flexlb.balance.schedule.SchedulerTestSupport.runtime(f.requests);
            var planners =
                    (java.util.concurrent.ExecutorService)
                            org.springframework.test.util.ReflectionTestUtils.getField(
                                    f.scheduler, "planners");
            var failure = new IllegalStateException("delivery cleanup failed");
            runtime.recordFailure(failure);
            runtime.requests().stopAccepting();
            assertFalse(planners.isShutdown(), "stopping intake does not execute shutdown");
            assertSame(
                    failure,
                    assertThrows(
                            IllegalStateException.class,
                            () -> SchedulerTestSupport.shutdown(runtime)));
            assertTrue(planners.isTerminated());
            assertTrue(f.scheduler.awaitStopped(System.nanoTime()));
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void stoppingFromPlacementDoesNotWaitForItsOwnSubmission(boolean queued) throws Exception {
        try (Fixture f = new Fixture(queued);
                var executor = Executors.newVirtualThreadPerTaskExecutor()) {
            var request = f.context(98);
            when(f.router.select(request, null))
                    .thenAnswer(
                            call -> {
                                SchedulerTestSupport.runtime(f.scheduler)
                                        .requests()
                                        .stopAccepting();
                                SchedulerTestSupport.runtime(f.scheduler)
                                        .requests()
                                        .stopAccepting();
                                return rejection();
                            });
            var submission =
                    executor.submit(() -> SchedulerTestServices.entry(f.scheduler).submit(request));
            assertFalse(submission.get(3, TimeUnit.SECONDS).get(3, TimeUnit.SECONDS).isSuccess());
            SchedulerTestSupport.shutdown(SchedulerTestSupport.runtime(f.scheduler));
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void acceptedSubmissionsCanEnterConcurrently(boolean queued) throws Exception {
        CountDownLatch entered = new CountDownLatch(2);
        CountDownLatch resume = new CountDownLatch(1);
        try (Fixture f = new Fixture(queued);
                var executor = Executors.newVirtualThreadPerTaskExecutor()) {
            Runnable pause =
                    () -> {
                        entered.countDown();
                        RequestProtocolTestSupport.await(resume);
                    };
            when(f.router.select(any(), any()))
                    .thenAnswer(
                            call -> {
                                if (!queued) {
                                    pause.run();
                                }
                                return rejection();
                            });
            var first =
                    executor.submit(
                            () ->
                                    queued
                                            ? SchedulerTestServices.entry(f.scheduler)
                                                    .submit(f.context(96), pause)
                                            : SchedulerTestServices.entry(f.scheduler)
                                                    .submit(f.context(96)));
            var second =
                    executor.submit(
                            () ->
                                    queued
                                            ? SchedulerTestServices.entry(f.scheduler)
                                                    .submit(f.context(97), pause)
                                            : SchedulerTestServices.entry(f.scheduler)
                                                    .submit(f.context(97)));
            try {
                assertTrue(
                        entered.await(3, TimeUnit.SECONDS),
                        "submissions must not serialize on an exclusive lock");
                SchedulerTestSupport.runtime(f.scheduler).requests().stopAccepting();

                assertFalse(
                        SchedulerTestServices.entry(f.scheduler)
                                .submit(f.context(95))
                                .get(3, TimeUnit.SECONDS)
                                .isSuccess());
            } finally {
                resume.countDown();
            }
            first.get(3, TimeUnit.SECONDS).get(3, TimeUnit.SECONDS);
            second.get(3, TimeUnit.SECONDS).get(3, TimeUnit.SECONDS);
            SchedulerTestSupport.shutdown(SchedulerTestSupport.runtime(f.scheduler));

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
            var runtime =
                    new SchedulerTestServices(
                            new RequestRepository(),
                            mock(EndpointRegistry.class),
                            reporter,
                            mock(RequestSchedulerReporter.class),
                            SchedulerTestSupport.dispatcher(),
                            service,
                            mock(EngineGrpcClient.class));
            runtime.registerScheduler(
                    runtime.newLifecycle(
                            f.config,
                            new org.flexlb.worker.endpoint.DecodeResources.AdmissionCapacity(
                                    0, 100),
                            mock(org.flexlb.balance.requestlog.TheoryCacheHitReporter.class)));
            try {
                var owner =
                        SchedulerTestSupport.lifecycle(
                                SchedulerTestSupport.initializedScheduler(runtime));
                var request = f.context(100);
                var future = owner.submit(request, () -> {});

                assertSame(owner, request.getRequestScheduler());
                assertSame(owner, runtime.requests().ownerOf("100"));

                var next = f.context(101);
                var nextFuture = owner.submit(next, () -> {});
                assertSame(owner, next.getRequestScheduler());
                assertSame(owner, runtime.requests().ownerOf("101"));
                owner.cancel("100", CancelReason.CLIENT_CANCELLED);
                owner.cancel("101", CancelReason.CLIENT_CANCELLED);
                future.get(3, TimeUnit.SECONDS);
                nextFuture.get(3, TimeUnit.SECONDS);
                RequestProtocolTestSupport.awaitCondition(
                        () ->
                                runtime.requests().findTerminal("100") != null
                                        && runtime.requests().findTerminal("101") != null);
                assertSame(
                        owner,
                        runtime.requests().findTerminal("100").owner(),
                        "terminal records preserve ownership too");
                assertSame(owner, runtime.requests().findTerminal("101").owner());
            } finally {
                SchedulerTestSupport.shutdown(runtime);
            }
        }
    }

    @Test
    void stoppingIntakeIsIdempotentAndRejectsNewRequests() throws Exception {
        try (Fixture f = new Fixture(false)) {
            AbstractRequestScheduler api = f.scheduler;

            var request = f.context(1);
            when(f.router.select(request, null)).thenReturn(rejection());
            assertEquals(
                    StrategyErrorType.NO_PREFILL_WORKER.getErrorCode(),
                    SchedulerTestServices.entry(api)
                            .submit(request)
                            .get(3, TimeUnit.SECONDS)
                            .getCode());
            SchedulerTestSupport.runtime(api).requests().stopAccepting();
            SchedulerTestSupport.runtime(api).requests().stopAccepting();
            SchedulerTestSupport.shutdown(SchedulerTestSupport.runtime(api));

            assertEquals(
                    StrategyErrorType.SCHEDULER_UNAVAILABLE.getErrorCode(),
                    SchedulerTestServices.entry(api).submit(f.context(2)).join().getCode());
            verify(f.router, times(1)).select(any(), any());
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void stopRejectsNewRequestsButLetsAcceptedPlacementFinish(boolean queued) throws Exception {
        CountDownLatch selecting = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        try (Fixture f = new Fixture(queued);
                var executor = Executors.newVirtualThreadPerTaskExecutor()) {
            AbstractRequestScheduler api = f.scheduler;
            var request = f.context(10);
            when(f.router.select(request, null))
                    .thenAnswer(
                            invocation -> {
                                selecting.countDown();
                                RequestProtocolTestSupport.await(release);
                                return rejection();
                            });
            var submission =
                    executor.submit(() -> SchedulerTestServices.entry(api).submit(request));
            try {
                assertTrue(selecting.await(3, TimeUnit.SECONDS));
                SchedulerTestSupport.runtime(api).requests().stopAccepting();

                assertFalse(
                        SchedulerTestSupport.repository(f.requests).isClosed(),
                        "draining must not close accepted admission");
                assertEquals(
                        StrategyErrorType.SCHEDULER_UNAVAILABLE.getErrorCode(),
                        SchedulerTestServices.entry(api).submit(f.context(11)).join().getCode());
            } finally {
                release.countDown();
            }
            var future = submission.get(3, TimeUnit.SECONDS);
            assertSame(request.getFuture(), future);
            assertEquals(
                    StrategyErrorType.NO_PREFILL_WORKER.getErrorCode(),
                    future.get(3, TimeUnit.SECONDS).getCode());
            SchedulerTestSupport.shutdown(SchedulerTestSupport.runtime(api));

            assertEquals(0, SchedulerTestSupport.repository(f.requests).liveRequestCount());
        } finally {
            release.countDown();
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void registeredWorkCanStartAdmissionAfterStopAndCancellationStillWorks(boolean queued)
            throws Exception {
        try (Fixture f = new Fixture(queued)) {
            AbstractRequestScheduler api = f.scheduler;
            var request = f.context(20);
            var future = f.requests.submit(request, () -> {});
            SchedulerTestSupport.runtime(api).requests().stopAccepting();
            var admission = SchedulerTestSupport.claim(f.requests, "20", future);
            try (var admissionCompletion3 = RequestProtocolTestSupport.finishOnExit(admission)) {
                assertNotNull(admission);
                assertNotNull(
                        SchedulerTestServices.entry(api)
                                .cancel("20", CancelReason.CLIENT_CANCELLED));
            }
            assertEquals(
                    StrategyErrorType.REQUEST_CANCELLED.getErrorCode(),
                    future.get(3, TimeUnit.SECONDS).getCode());
            SchedulerTestSupport.shutdown(SchedulerTestSupport.runtime(api));

            assertNotNull(
                    SchedulerTestServices.entry(api).cancel("20", CancelReason.CLIENT_CANCELLED));
            assertNull(
                    SchedulerTestServices.entry(api).cancel("999", CancelReason.CLIENT_CANCELLED));
        }
    }

    @Test
    void cancelledFutureDoesNotHideAnOutstandingAdmission() throws Exception {
        try (Fixture f = new Fixture(true)) {
            AbstractRequestScheduler api = f.scheduler;
            var request = f.context(30);
            var future = f.requests.submit(request, () -> {});
            var admission = SchedulerTestSupport.claim(f.requests, "30", future);
            try (var admissionCompletion4 = RequestProtocolTestSupport.finishOnExit(admission)) {
                assertNotNull(admission);
                assertTrue(future.cancel(true));
                SchedulerTestSupport.runtime(api).requests().stopAccepting();
                assertTrue(future.isDone());
            }
            SchedulerTestSupport.shutdown(SchedulerTestSupport.runtime(api));

            assertEquals(0, SchedulerTestSupport.repository(f.requests).liveRequestCount());
        }
    }

    @Test
    void stopFromPublicationCallbackDoesNotDeadlockOrFinishBeforePublicationDrains()
            throws Exception {
        CountDownLatch publishing = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        try (Fixture f = new Fixture(false)) {
            AbstractRequestScheduler api = f.scheduler;
            var request = f.context(40);
            org.mockito.Mockito.doNothing().when(f.requests).scheduleRequest(request);
            var future = f.requests.submit(request, () -> {});
            CompletableFuture<Response> callback =
                    future.whenComplete(
                            (result, failure) -> {
                                SchedulerTestSupport.runtime(api).requests().stopAccepting();
                                publishing.countDown();
                                RequestProtocolTestSupport.await(release);
                            });
            CompletableFuture<Void> shutdown = null;
            try {
                assertTrue(
                        f.requests.publish(
                                request,
                                Response.error(StrategyErrorType.NO_PREFILL_WORKER),
                                false));
                assertTrue(publishing.await(3, TimeUnit.SECONDS));
                assertTrue(future.isDone());
                assertEquals(0, SchedulerTestSupport.repository(f.requests).liveRequestCount());
                var runtime = SchedulerTestSupport.runtime(api);
                shutdown = CompletableFuture.runAsync(() -> SchedulerTestSupport.shutdown(runtime));
                RequestProtocolTestSupport.awaitCondition(() -> runtime.requests().isClosed());
                var closing = shutdown;
                assertThrows(
                        java.util.concurrent.TimeoutException.class,
                        () -> closing.get(100, TimeUnit.MILLISECONDS),
                        "shutdown must await the active publication");

            } finally {
                release.countDown();
            }
            callback.get(3, TimeUnit.SECONDS);
            shutdown.get(3, TimeUnit.SECONDS);

        } finally {
            release.countDown();
        }
    }

    @Test
    void cleanupFailureIsRetainedForShutdownReporting() throws Exception {
        try (Fixture f = new Fixture(false)) {
            AbstractRequestScheduler api = f.scheduler;
            var request = f.context(50);
            org.mockito.Mockito.doNothing().when(f.requests).scheduleRequest(request);
            var future = f.requests.submit(request, () -> {});
            var endpoint = mock(PrefillEndpoint.class);
            var route =
                    SchedulingTestConfig.createRoute(
                            request,
                            new Response(),
                            null,
                            null,
                            endpoint,
                            null,
                            null,
                            System.currentTimeMillis());
            var admission = SchedulerTestSupport.claim(f.requests, "50", future);
            try (var admissionCompletion5 = RequestProtocolTestSupport.finishOnExit(admission)) {
                assertNotNull(admission);
                assertEquals(
                        RoutingResult.Status.SUCCESS,
                        RequestProtocolTestSupport.commitRoute(
                                f.requests,
                                route,
                                RequestProtocolTestSupport.publication(() -> true)));
            }
            var failure = new IllegalStateException("endpoint release failed");
            doThrow(failure).when(endpoint).releaseRequest(route);
            SchedulerTestSupport.runtime(api).requests().stopAccepting();
            assertTrue(future.join().isSuccess());
            assertSame(
                    failure,
                    assertThrows(
                            IllegalStateException.class,
                            () ->
                                    f.requests.cancel(
                                            request.getRequestId(),
                                            CancelReason.CLIENT_CANCELLED)));
            SchedulerTestSupport.awaitRequestWork(
                    org.flexlb.balance.schedule.SchedulerTestSupport.runtime(f.requests));
            assertSame(failure, SchedulerTestSupport.failure(api));
            assertTrue(
                    future.join().isSuccess(),
                    "cleanup failure must not revoke the selected response");
            assertEquals(
                    SchedulerTestSupport.Stage.FINALIZING, SchedulerTestSupport.stage(request));
            verify(endpoint, times(3)).releaseRequest(route);
            org.mockito.Mockito.doReturn(true).when(endpoint).releaseRequest(route);
            f.requests.closeRequest(request, null);
            RequestProtocolTestSupport.awaitCondition(
                    () -> !f.requests.repository.isCurrent(request));
            SchedulerTestSupport.awaitRequestWork(
                    org.flexlb.balance.schedule.SchedulerTestSupport.runtime(f.requests));
            verify(endpoint, times(4)).releaseRequest(route);
            assertSame(failure, SchedulerTestSupport.failure(api));
            assertSame(
                    failure,
                    assertThrows(
                            IllegalStateException.class,
                            () ->
                                    SchedulerTestSupport.shutdown(
                                            org.flexlb.balance.schedule.SchedulerTestSupport
                                                    .runtime(f.requests))));
        }
    }

    @Test
    void placementCompletionFailureIsRecordedAndFinishesPlacement() throws Exception {
        try (Fixture f = new Fixture(false)) {
            var context = f.context(940L);
            var future = f.requests.submit(context, () -> {});
            var handle = SchedulerTestSupport.claim(f.requests, "940", future);
            assertNotNull(handle);
            var originalTimer =
                    org.flexlb.balance.schedule.SchedulerTestSupport.runtime(f.requests)
                            .deadlines();
            originalTimer.remove(context);
            var timer = org.mockito.Mockito.spy(originalTimer);
            var failure = new IllegalStateException("expiry attachment failed");
            doThrow(failure)
                    .when(timer)
                    .update(
                            org.mockito.ArgumentMatchers.eq(context),
                            org.mockito.ArgumentMatchers.any());
            org.flexlb.balance.schedule.SchedulerTestServices.setDependency(
                    org.flexlb.balance.schedule.SchedulerTestSupport.runtime(f.requests),
                    "timer",
                    timer);
            try {
                assertDoesNotThrow(
                        () ->
                                handle.requestContext()
                                        .getRequestScheduler()
                                        .finishSelectionWithoutCommit(handle, null));
                assertSame(failure, SchedulerTestSupport.failure(f.requests));
                assertNull(context.currentSelectionAttempt());
                assertDoesNotThrow(
                        () ->
                                handle.requestContext()
                                        .getRequestScheduler()
                                        .finishSelectionWithoutCommit(handle, null));
                verify(timer, times(3)).update(org.mockito.ArgumentMatchers.eq(context), any());
            } finally {
                org.flexlb.balance.schedule.SchedulerTestServices.setDependency(
                        org.flexlb.balance.schedule.SchedulerTestSupport.runtime(f.requests),
                        "timer",
                        originalTimer);
                SchedulerTestServices.entry(f.scheduler)
                        .cancel("940", CancelReason.CLIENT_CANCELLED);
            }
        }
    }

    private static RequestResult rejection() {
        return RequestResult.rejected(Response.error(StrategyErrorType.NO_PREFILL_WORKER));
    }

    /**
     * Builds either scheduler mode around a controlled selector and owns the shared runtime used by its requests.
     */
    private static final class Fixture implements AutoCloseable {
        final FlexlbConfig config = SchedulingTestConfig.newConfig();
        final RequestRouteSelector router = mock(RequestRouteSelector.class);
        final AbstractRequestScheduler requests;
        final AbstractRequestScheduler scheduler;

        Fixture(boolean queued) {
            this(queued, true);
        }

        Fixture(boolean queued, boolean startPlacement) {
            if (queued) {
                SchedulingTestConfig.useFifoQueue(config);
            } else {
                config.setScheduler(ConfigTestSupport.direct());
            }
            ConfigService service = mock(ConfigService.class);
            when(service.loadBalanceConfig()).thenReturn(config);
            var reporter = mock(DeliveryMetricsReporter.class);
            requests =
                    SchedulerTestSupport.create(
                            service,
                            reporter,
                            mock(RequestSchedulerReporter.class),
                            mock(TheoryCacheHitReporter.class),
                            router,
                            mock(DecodePreemption.class),
                            new CapacityNotifier(),
                            startPlacement);
            scheduler =
                    ((AbstractRequestScheduler)
                            org.flexlb.balance.schedule.SchedulerTestServices.scheduler(requests));
        }

        @Override
        public void close() {
            SchedulerTestSupport.repository(requests).closeRegistration();
            RequestProtocolTestSupport.close(scheduler);
            SchedulerTestSupport.timer(SchedulerTestSupport.runtime(requests)).close();
            SchedulerTestSupport.awaitRequestWork(
                    org.flexlb.balance.schedule.SchedulerTestSupport.runtime(requests));
            requests.closeOutstandingAndTerminalize();
            SchedulerTestSupport.closeRequestExecutors(SchedulerTestSupport.runtime(requests));
        }

        RequestContext context(long id) {
            return RequestProtocolTestSupport.context(config, id);
        }
    }
}
