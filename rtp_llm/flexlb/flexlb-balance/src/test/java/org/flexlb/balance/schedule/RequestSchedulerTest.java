package org.flexlb.balance.schedule;

import org.flexlb.balance.metrics.DeliveryMetricsReporter;
import org.flexlb.balance.metrics.RequestSchedulerReporter;
import org.flexlb.balance.requestlog.TheoryCacheHitReporter;
import org.flexlb.balance.strategy.RequestRouteSelector;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.service.RoleType;
import org.flexlb.worker.endpoint.CapacityKey;
import org.flexlb.worker.endpoint.CapacityNotifier;
import org.flexlb.worker.endpoint.CapacityNotifierTestSupport;
import org.flexlb.worker.endpoint.PrefillEndpoint;
import org.flexlb.worker.endpoint.Request;
import org.flexlb.worker.endpoint.Response;
import org.flexlb.worker.endpoint.StrategyErrorType;
import org.flexlb.worker.status.ServerStatus;
import org.flexlb.worker.status.WorkerStatus;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BiConsumer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.isNull;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.timeout;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Verifies queue admission, cancellation and retries using controlled placement collaborators.
 */
class RequestSchedulerTest {
    @Test
    void futureCancelBeforeFailedGlobalOfferSettlesWithoutQueueTicket() throws Exception {
        FlexlbConfig config = SchedulingTestConfig.newConfig();
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        RequestRouteSelector router = mockRouter();
        AbstractRequestScheduler lifecycle =
                org.flexlb.balance.schedule.SchedulerTestSupport.create(
                        service,
                        mock(DeliveryMetricsReporter.class),
                        mock(RequestSchedulerReporter.class),
                        mock(TheoryCacheHitReporter.class));
        RequestContext context = context(config, 990009L);
        when(router.resolvePolicyGroup(context))
                .thenAnswer(
                        invocation -> {
                            assertTrue(context.getFuture().cancel(false));
                            throw new IllegalStateException("policy group unavailable");
                        });
        AbstractRequestScheduler scheduler =
                RequestProtocolTestSupport.injectPlacementDependencies(
                        lifecycle,
                        service,
                        router,
                        mock(DeliveryMetricsReporter.class),
                        org.flexlb.balance.schedule.DecodePreemptionTestSupport.mock(),
                        new CapacityNotifier());
        try {
            CompletableFuture<Response> future =
                    SchedulerTestServices.entry(scheduler).submit(context);
            assertTrue(future.isCancelled());
            RequestProtocolTestSupport.awaitCondition(
                    () -> lifecycle.repository.liveRequestCount() == 0);
            assertEquals(
                    RequestResult.State.CANCELLED,
                    org.flexlb.balance.schedule.SchedulerTestSupport.repository(lifecycle)
                            .getRequestState("990009")
                            .state());
            assertEquals(
                    0,
                    org.flexlb.balance.schedule.SchedulerTestSupport.repository(lifecycle)
                            .liveRequestCount());
        } finally {
            RequestProtocolTestSupport.close(scheduler);
            if (RequestProtocolTestSupport.closeRegistrationAndAwaitScheduling(lifecycle)) {
                lifecycle.closeOutstandingAndTerminalize();
                SchedulerTestSupport.timer(
                                org.flexlb.balance.schedule.SchedulerTestSupport.runtime(lifecycle))
                        .close();
                SchedulerTestSupport.closeRequestExecutors(
                        org.flexlb.balance.schedule.SchedulerTestSupport.runtime(lifecycle));
            }
        }
    }

    @Test
    void closeSettlesInFlightAdmissionAfterLatePlanReleasesItsHandle() throws Exception {
        FlexlbConfig config = SchedulingTestConfig.newConfig();
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        RequestRouteSelector router = mockRouter();
        AbstractRequestScheduler lifecycle =
                org.flexlb.balance.schedule.SchedulerTestSupport.create(
                        service,
                        mock(DeliveryMetricsReporter.class),
                        mock(RequestSchedulerReporter.class),
                        mock(TheoryCacheHitReporter.class));
        RequestContext context = context(config, 990005L);
        CountDownLatch planning = new CountDownLatch(1);
        CountDownLatch releasePlan = new CountDownLatch(1);
        CountDownLatch closeSettled = new CountDownLatch(1);
        RequestResult route = RequestProtocolTestSupport.mockSelectedRoute(context);
        when(router.select(context, null))
                .thenAnswer(
                        invocation -> {
                            planning.countDown();
                            assertTrue(releasePlan.await(5, TimeUnit.SECONDS));
                            return route;
                        });
        doAnswer(
                        invocation -> {
                            Object result = invocation.callRealMethod();
                            closeSettled.countDown();
                            return result;
                        })
                .when(lifecycle)
                .closeRequest(
                        org.mockito.ArgumentMatchers.same(context),
                        any(RequestContext.RequestEndEvent.class));
        QueuedRequestScheduler coordinator =
                (QueuedRequestScheduler)
                        RequestProtocolTestSupport.injectPlacementDependencies(
                                lifecycle,
                                service,
                                router,
                                mock(DeliveryMetricsReporter.class),
                                org.flexlb.balance.schedule.DecodePreemptionTestSupport.mock(),
                                new CapacityNotifier());
        Thread closer = null;
        Thread secondCloser = null;
        try {
            CompletableFuture<Response> future = lifecycle.submit(context);
            assertTrue(planning.await(5, TimeUnit.SECONDS));
            closer = new Thread(coordinator::close);
            closer.start();
            assertTrue(closeSettled.await(5, TimeUnit.SECONDS));
            assertFalse(future.isDone(), "the in-flight admission still owns its handle");
            assertTrue(
                    closer.isAlive(), "close must wait for the late planner to release its handle");
            CountDownLatch secondEntered = new CountDownLatch(1);
            CountDownLatch secondReturned = new CountDownLatch(1);
            secondCloser =
                    new Thread(
                            () -> {
                                secondEntered.countDown();
                                try {
                                    coordinator.close();
                                } finally {
                                    secondReturned.countDown();
                                }
                            });
            secondCloser.start();
            assertTrue(secondEntered.await(5, TimeUnit.SECONDS));
            assertFalse(
                    secondReturned.await(200, TimeUnit.MILLISECONDS),
                    "a second close caller must observe the same drain");
            releasePlan.countDown();
            assertEquals(
                    StrategyErrorType.DISPATCH_FAILED.getErrorCode(),
                    future.get(5, TimeUnit.SECONDS).getCode());

        } finally {
            releasePlan.countDown();
            coordinator.close();
            if (closer != null) {
                closer.join(5_000);
                assertFalse(closer.isAlive());
            }
            if (secondCloser != null) {
                secondCloser.join(5_000);
                assertFalse(secondCloser.isAlive());
            }
            if (RequestProtocolTestSupport.closeRegistrationAndAwaitScheduling(lifecycle)) {
                lifecycle.closeOutstandingAndTerminalize();
                SchedulerTestSupport.timer(
                                org.flexlb.balance.schedule.SchedulerTestSupport.runtime(lifecycle))
                        .close();
                SchedulerTestSupport.closeRequestExecutors(
                        org.flexlb.balance.schedule.SchedulerTestSupport.runtime(lifecycle));
            }
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void controlTicketRemovesInFlightEntryBeforeLatePlanCanPublish(boolean deadline)
            throws Exception {
        FlexlbConfig config = SchedulingTestConfig.newConfig();
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        RequestRouteSelector router = mockRouter();
        AbstractRequestScheduler lifecycle =
                org.flexlb.balance.schedule.SchedulerTestSupport.create(
                        service,
                        mock(DeliveryMetricsReporter.class),
                        mock(RequestSchedulerReporter.class),
                        mock(TheoryCacheHitReporter.class));
        RequestContext context = context(config, 990001L);
        CountDownLatch planning = new CountDownLatch(1);
        CountDownLatch releasePlan = new CountDownLatch(1);
        CountDownLatch planClosed = new CountDownLatch(1);
        RequestResult route = RequestProtocolTestSupport.mockSelectedRoute(context);
        when(router.select(context, null))
                .thenAnswer(
                        invocation -> {
                            planning.countDown();
                            assertTrue(releasePlan.await(5, TimeUnit.SECONDS));
                            return route;
                        });
        doAnswer(
                        invocation -> {
                            try {
                                return invocation.callRealMethod();
                            } finally {
                                planClosed.countDown();
                            }
                        })
                .when(lifecycle)
                .finishSelectionWithoutCommit(any(), any());
        QueuedRequestScheduler coordinator =
                (QueuedRequestScheduler)
                        RequestProtocolTestSupport.injectPlacementDependencies(
                                lifecycle,
                                service,
                                router,
                                mock(DeliveryMetricsReporter.class),
                                org.flexlb.balance.schedule.DecodePreemptionTestSupport.mock(),
                                new CapacityNotifier());
        CompletableFuture<Response> future = lifecycle.submit(context);
        try {
            assertTrue(planning.await(5, TimeUnit.SECONDS));
            if (deadline) {
                RequestContext requestContext =
                        SchedulerTestSupport.repository(lifecycle).findActive("990001");
                var exact = requestContext.snapshot().deadlines().request();
                lifecycle.onDeadline(exact, exact.atMs());
            } else {
                lifecycle.cancel("990001", CancelReason.CLIENT_CANCELLED);
            }
            assertEquals(
                    RequestResult.State.CANCEL_REQUESTED,
                    org.flexlb.balance.schedule.SchedulerTestSupport.repository(lifecycle)
                            .getRequestState("990001")
                            .state());
            assertFalse(future.isDone(), "the admission claim still owns its route");
            releasePlan.countDown();
            assertTrue(planClosed.await(5, TimeUnit.SECONDS));
            assertEquals(0, RequestProtocolTestSupport.queuedCount(coordinator));
            assertEquals(
                    (deadline
                                    ? StrategyErrorType.QUEUE_TIMEOUT
                                    : StrategyErrorType.REQUEST_CANCELLED)
                            .getErrorCode(),
                    future.get(5, TimeUnit.SECONDS).getCode());
            RequestProtocolTestSupport.awaitCondition(
                    () -> {
                        var lock =
                                (java.util.concurrent.locks.ReentrantLock)
                                        org.springframework.test.util.ReflectionTestUtils.getField(
                                                coordinator, "lock");
                        lock.lock();
                        try {
                            var registered =
                                    (SchedulingRequests)
                                            org.springframework.test.util.ReflectionTestUtils
                                                    .getField(coordinator, "schedulingRequests");
                            return registered.find(context) == null;
                        } finally {
                            lock.unlock();
                        }
                    });
            verify(route, never()).successResponse();
        } finally {
            releasePlan.countDown();
            coordinator.close();
            if (RequestProtocolTestSupport.closeRegistrationAndAwaitScheduling(lifecycle)) {
                lifecycle.closeOutstandingAndTerminalize();
                SchedulerTestSupport.timer(
                                org.flexlb.balance.schedule.SchedulerTestSupport.runtime(lifecycle))
                        .close();
                SchedulerTestSupport.closeRequestExecutors(
                        org.flexlb.balance.schedule.SchedulerTestSupport.runtime(lifecycle));
            }
        }
    }

    @Test
    void higherPriorityRequestRunsAheadOfBlockedLowerPriorityFrontier() {
        FlexlbConfig config = SchedulingTestConfig.newConfig();
        SchedulingTestConfig.usePriorityQueue(config);
        SchedulingTestConfig.useSingleDecision(config);
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        RequestRouteSelector router = mockRouter();
        AbstractRequestScheduler lifecycle = RequestProtocolTestSupport.schedulerMock();
        DecodePreemption decodePreemption =
                org.flexlb.balance.schedule.DecodePreemptionTestSupport.mock();
        RequestContext lowPriority = context(config, 910L, 10);
        RequestContext highPriority = context(config, 911L, 90);
        CompletableFuture<Response> lowFuture = new CompletableFuture<>();
        CompletableFuture<Response> highFuture = new CompletableFuture<>();
        when(lifecycle.submit(
                        org.mockito.ArgumentMatchers.eq(lowPriority),
                        org.mockito.ArgumentMatchers.any(Runnable.class)))
                .thenAnswer(
                        invocation -> {
                            RequestContext registered = invocation.getArgument(0);
                            registered.setFuture(lowFuture);
                            if (!lowFuture.isDone())
                                ((AbstractRequestScheduler) invocation.getMock())
                                        .scheduleRequest(invocation.getArgument(0));
                            return lowFuture;
                        });
        when(lifecycle.submit(
                        org.mockito.ArgumentMatchers.eq(highPriority),
                        org.mockito.ArgumentMatchers.any(Runnable.class)))
                .thenAnswer(
                        invocation -> {
                            RequestContext registered = invocation.getArgument(0);
                            registered.setFuture(highFuture);
                            if (!highFuture.isDone())
                                ((AbstractRequestScheduler) invocation.getMock())
                                        .scheduleRequest(invocation.getArgument(0));
                            return highFuture;
                        });
        when(lifecycle.tryBeginRouteSelection(
                        org.mockito.ArgumentMatchers.argThat(
                                request -> request != null && request.getFuture() == lowFuture)))
                .thenAnswer(i -> RequestProtocolTestSupport.admissionHandle(i.getArgument(0)));
        when(lifecycle.tryBeginRouteSelection(
                        org.mockito.ArgumentMatchers.argThat(
                                request -> request != null && request.getFuture() == highFuture)))
                .thenAnswer(i -> RequestProtocolTestSupport.admissionHandle(i.getArgument(0)));
        when(router.select(lowPriority, null))
                .thenReturn(RequestResult.blocked(CapacityKey.anyGroup(RoleType.PREFILL)));
        when(router.select(highPriority, null))
                .thenReturn(
                        RequestResult.rejected(
                                Response.buildErrorResponse(
                                        StrategyErrorType.NO_PREFILL_WORKER, null)));
        AbstractRequestScheduler scheduler =
                RequestProtocolTestSupport.injectPlacementDependencies(
                        lifecycle,
                        configService,
                        router,
                        mock(DeliveryMetricsReporter.class),
                        decodePreemption,
                        new CapacityNotifier());
        try {
            SchedulerTestServices.entry(scheduler).submit(lowPriority);
            verify(router, timeout(1_000)).select(lowPriority, null);
            SchedulerTestServices.entry(scheduler).submit(highPriority);
            verify(router, timeout(1_000)).select(highPriority, null);
            assertFalse(lowFuture.isDone());
        } finally {
            RequestProtocolTestSupport.close(scheduler);
        }
    }

    @Test
    void capacityEventRetriesWaitersInOrderAndIngressDoesNotRetryFailures() throws Exception {
        try (CapacityFixture fixture = new CapacityFixture(0, false)) {
            fixture.submitBlockedRequests();
            assertEquals(List.of(), fixture.admitted);
            for (CapacityRequest request : fixture.requests) {
                assertEquals(
                        1,
                        request.attempts.get(),
                        "every request receives its own placement attempt");
            }
            fixture.releaseSlots(1);
            fixture.requests.get(0).future.get(5, TimeUnit.SECONDS);
            assertTrue(fixture.requests.get(1).retryBlockedReturn.await(5, TimeUnit.SECONDS));
            fixture.awaitIndependentCommit();
            RequestProtocolTestSupport.awaitGlobalCapacityWaiters(fixture.lifecycle, 2);
            assertEquals(List.of(820L), fixture.admitted);
            assertEquals(0, fixture.availableSlots.get());
            assertEquals(2, fixture.requests.get(0).attempts.get());
            assertEquals(2, fixture.requests.get(1).attempts.get());
            assertEquals(
                    1,
                    fixture.requests.get(2).attempts.get(),
                    "a confirming failure stops this capacity round");
            fixture.awaitIndependentCommit();
            assertEquals(
                    2,
                    fixture.requests.get(1).attempts.get(),
                    "unrelated ingress cannot retry failures");
            assertEquals(1, fixture.requests.get(2).attempts.get());
            fixture.releaseSlots(2);
            fixture.awaitAllPublished();
            assertEquals(List.of(820L, 821L, 822L), fixture.admitted);
            assertEquals(0, fixture.availableSlots.get());
        }
    }

    @Test
    void oneCapacityEventAdmitsAllReleasedSlotsInFifoOrder() throws Exception {
        assertReleasedCapacityAdmitsWaiters(3, 0);
    }

    @Test
    void capacityReleasedDuringPublicationAdmitsWaitersWithoutAnotherEvent() throws Exception {
        assertReleasedCapacityAdmitsWaiters(1, 2);
    }

    @Test
    void capacityReleasedBeforeBlockedReturnRetriesActiveRequestWithoutLosingWakeup()
            throws Exception {
        try (CapacityFixture fixture = new CapacityFixture(0, false)) {
            fixture.submitBlockedRequests();
            CapacityRequest head = fixture.requests.get(0);
            head.beforeBlockedReturn = () -> fixture.releaseSlots(1);

            // Activate the head while the endpoint is still full. Its failed
            // admission publishes a release after reading capacity but before park.
            CapacityNotifierTestSupport.notifyChanged(fixture.availability, fixture.key);
            head.future.get(5, TimeUnit.SECONDS);
            assertTrue(fixture.requests.get(1).retryBlockedReturn.await(5, TimeUnit.SECONDS));
            fixture.awaitIndependentCommit();
            RequestProtocolTestSupport.awaitGlobalCapacityWaiters(fixture.lifecycle, 2);

            assertEquals(List.of(820L), fixture.admitted);
            assertEquals(0, fixture.availableSlots.get());
            assertEquals(
                    3,
                    head.attempts.get(),
                    "the stale full observation must trigger a fresh successful admission");
            assertEquals(2, fixture.requests.get(1).attempts.get());
            assertEquals(1, fixture.requests.get(2).attempts.get());
            fixture.awaitIndependentCommit();
            assertEquals(
                    2,
                    fixture.requests.get(1).attempts.get(),
                    "each failed request waits for another capacity edge");
        }
    }

    @Test
    void capacityWakeLetsEveryReadyRequestChooseAnotherWorker() throws Exception {
        try (CapacityFixture fixture =
                new CapacityFixture(0, false, new CompletableFuture<>(), 1)) {
            fixture.submitBlockedRequests();
            PrefillEndpoint other = mockPrefillEndpoint("other-prefill", 8080);
            for (CapacityRequest request : fixture.requests) {
                RequestResult route = RequestProtocolTestSupport.mockSelectedRoute(request.context);
                ServerStatus status = new ServerStatus();
                status.setRole(RoleType.PREFILL);
                status.setRequestId(request.context.getRequestId());
                status.setSuccess(true);
                RequestResult item =
                        RequestProtocolTestSupport.mockSelectedRoute(
                                request.context,
                                List.of(SelectionTestSupport.metadata(other, status, 0L)));
                request.alternative = new AlternativeRoute(route, item);
            }
            CapacityNotifierTestSupport.notifyChanged(fixture.availability, fixture.key);
            fixture.awaitAllPublished();
            assertEquals(List.of(820L, 821L, 822L), fixture.admitted);
            assertEquals(0, fixture.availableSlots.get(), "the original worker stayed full");
            for (CapacityRequest request : fixture.requests) {
                assertEquals(1, request.attempts.get(), "no retry returned to the full worker");
            }
        }
    }

    @Test
    void cancellingActiveRetryHandsUnusedCapacityToNextWaiter() throws Exception {
        try (CapacityFixture fixture = new CapacityFixture(0, true)) {
            fixture.submitBlockedRequests();
            fixture.releaseSlots(1);
            assertTrue(fixture.headRetryStarted.await(5, TimeUnit.SECONDS));

            assertTrue(fixture.requests.get(0).future.cancel(false));
            fixture.allowHeadRetry.countDown();
            fixture.requests.get(1).future.get(5, TimeUnit.SECONDS);
            assertTrue(fixture.requests.get(2).retryBlockedReturn.await(5, TimeUnit.SECONDS));
            fixture.awaitIndependentCommit();
            RequestProtocolTestSupport.awaitGlobalCapacityWaiters(fixture.lifecycle, 1);

            assertEquals(List.of(821L), fixture.admitted);
            assertEquals(0, fixture.availableSlots.get());
            assertEquals(
                    1,
                    fixture.requests.get(0).attempts.get(),
                    "the cancelled active request must not consume the released slot");
            assertEquals(2, fixture.requests.get(2).attempts.get());
        }
    }

    @Test
    void completedActiveRetryHandsOffWithoutRegisteringACompletionCallback() throws Exception {
        ConcurrentLinkedQueue<Runnable> callbacks = new ConcurrentLinkedQueue<>();
        CompletableFuture<Response> headFuture =
                new CompletableFuture<>() {

                    @Override
                    public CompletableFuture<Response> whenComplete(
                            BiConsumer<? super Response, ? super Throwable> action) {
                        return super.whenCompleteAsync(action, callbacks::add);
                    }
                };
        try (CapacityFixture fixture = new CapacityFixture(0, false, headFuture, 1)) {
            fixture.submitBlockedRequests();
            PrefillEndpoint independent = mockPrefillEndpoint("independent", 8080);
            CapacityRequest first = fixture.createRequest(840L, independent);
            CountDownLatch publicationStarted = new CountDownLatch(1);
            CountDownLatch allowPublication = new CountDownLatch(1);
            first.beforePublication =
                    () -> {
                        publicationStarted.countDown();
                        assertTrue(allowPublication.await(5, TimeUnit.SECONDS));
                        return null;
                    };
            try {
                SchedulerTestServices.entry(fixture.scheduler).submit(first.context);
                assertTrue(publicationStarted.await(5, TimeUnit.SECONDS));
                fixture.releaseSlots(1);
                assertTrue(headFuture.complete(new Response()));
                assertEquals(
                        0,
                        callbacks.size(),
                        "the queue must not attach a second lifecycle callback");
                CapacityRequest second = fixture.createRequest(841L, independent);
                SchedulerTestServices.entry(fixture.scheduler).submit(second.context);
                allowPublication.countDown();
                second.future.get(5, TimeUnit.SECONDS);
                // Ready waiters progress without another event or completion callback.
                fixture.requests.get(1).future.get(5, TimeUnit.SECONDS);
                assertEquals(
                        0,
                        callbacks.size(),
                        "handoff must not rely on running completion callbacks");
                // Even a subsequent edge must not be swallowed by an active request
                // that was removed from the ordered queue during pruning.
                CapacityNotifierTestSupport.notifyChanged(fixture.availability, fixture.key);
                fixture.requests.get(1).future.get(5, TimeUnit.SECONDS);
                assertEquals(List.of(821L), fixture.admitted);
                assertEquals(0, fixture.availableSlots.get());
                assertEquals(1, fixture.requests.get(0).attempts.get());
            } finally {
                allowPublication.countDown();
            }
        }
    }

    @Test
    void submissionUsesRequestConfigWithoutReloading() throws Exception {
        Fixture fixture = new Fixture(true);
        try {
            when(fixture.configService.loadBalanceConfig())
                    .thenThrow(
                            new IllegalStateException(
                                    "configuration unavailable after initialization"));
            CompletableFuture<Response> future =
                    SchedulerTestServices.entry(fixture.scheduler).submit(fixture.context);

            assertSame(fixture.future, future);
            assertEquals(
                    StrategyErrorType.NO_PREFILL_WORKER.getErrorCode(),
                    future.get(5, TimeUnit.SECONDS).getCode());
            verify(fixture.router).select(fixture.context, null);
        } finally {
            doReturn(fixture.config).when(fixture.configService).loadBalanceConfig();
            RequestProtocolTestSupport.close(fixture.scheduler);
        }
    }

    @Test
    void terminalRouteRejectionDoesNotAcquireDecodeAcceptance() {
        Fixture fixture = new Fixture(true);

        Response response =
                SchedulerTestServices.entry(fixture.scheduler)
                        .submit(fixture.context)
                        .orTimeout(3, TimeUnit.SECONDS)
                        .join();

        assertEquals(StrategyErrorType.NO_PREFILL_WORKER.getErrorCode(), response.getCode());
        verify(fixture.router, timeout(1_000)).select(fixture.context, null);
        verify(fixture.lifecycle, never()).currentSelectionAttempt(any());
        RequestProtocolTestSupport.close(fixture.scheduler);
    }

    @Test
    void prioritySelectorMissWaitsWithoutInventingAnEvictionRoute() {
        Fixture fixture = new Fixture(true);
        when(fixture.router.select(fixture.context, null))
                .thenReturn(RequestResult.blocked(CapacityKey.anyGroup(RoleType.PREFILL)));
        CompletableFuture<Response> waiting =
                SchedulerTestServices.entry(fixture.scheduler).submit(fixture.context);

        verify(fixture.router, timeout(1_000)).select(fixture.context, null);
        verify(fixture.decodePreemption, never()).tryPreempt(any(), any());
        assertFalse(waiting.isDone());
        verify(fixture.lifecycle, never()).currentSelectionAttempt(any());
        RequestProtocolTestSupport.close(fixture.scheduler);
    }

    @Test
    void fifoTemporaryDecodeMissRemainsQueued() {
        Fixture fixture = new Fixture(false);
        when(fixture.router.select(fixture.context, null))
                .thenReturn(RequestResult.blocked(CapacityKey.anyGroup(RoleType.DECODE)));
        CompletableFuture<Response> waiting =
                SchedulerTestServices.entry(fixture.scheduler).submit(fixture.context);

        verify(fixture.router, timeout(1_000)).select(fixture.context, null);
        assertFalse(waiting.isDone());
        verify(fixture.decodePreemption, never()).tryPreempt(any(), any());
        verify(fixture.lifecycle, never()).currentSelectionAttempt(any());
        RequestProtocolTestSupport.close(fixture.scheduler);
    }

    @Test
    void initialPlacementFailureCompletesTheRegisteredGeneration() {
        Fixture fixture = new Fixture(false);
        when(fixture.router.select(fixture.context, null))
                .thenThrow(new IllegalStateException("selector failed"));

        CompletableFuture<Response> returned =
                SchedulerTestServices.entry(fixture.scheduler).submit(fixture.context);

        assertEquals(fixture.future, returned);
        assertEquals(StrategyErrorType.DISPATCH_FAILED.getErrorCode(), returned.join().getCode());
        RequestProtocolTestSupport.close(fixture.scheduler);
    }

    private static RequestRouteSelector mockRouter() {
        RequestRouteSelector router = mock(RequestRouteSelector.class);
        return router;
    }

    private void assertReleasedCapacityAdmitsWaiters(
            int initiallyReleasedSlots, int slotsReleasedDuringPublication) throws Exception {
        try (CapacityFixture fixture = new CapacityFixture(slotsReleasedDuringPublication, false)) {
            fixture.submitBlockedRequests();
            fixture.releaseSlots(initiallyReleasedSlots);
            fixture.awaitAllPublished();

            assertEquals(List.of(820L, 821L, 822L), fixture.admitted);
            assertEquals(0, fixture.availableSlots.get());
        }
    }

    private static PrefillEndpoint mockPrefillEndpoint(String ip, int port) {
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        WorkerStatus status =
                WorkerStatus.createDiscovered(RoleType.PREFILL, "g1", ip, port, port, "test");
        when(endpoint.getStatus()).thenReturn(status);
        when(endpoint.ipPort()).thenReturn(status.getLogicalIpPort());
        when(endpoint.getIp()).thenReturn(ip);
        return endpoint;
    }

    private static RequestContext context(FlexlbConfig config, long requestId) {
        return context(config, requestId, 50);
    }

    private static RequestContext context(FlexlbConfig config, long requestId, int priority) {
        RequestContext context = new RequestContext(config);
        Request request =
                Request.builder()
                        .requestId(String.valueOf(requestId))
                        .rawPriority(priority)
                        .build();
        context.setRequest(request);

        context.setSchedulingMetadata(
                SchedulingMetadata.explicit(priority, System.currentTimeMillis() + 60_000L));
        return context;
    }

    /**
     * Real ordered scheduler with an exact endpoint whose free slots are explicitly controlled.
     */
    private static final class CapacityFixture implements AutoCloseable {
        private final ConcurrentMap<Long, CapacityRequest> requestsById = new ConcurrentHashMap<>();
        private final RequestRouteSelector router = mockRouter();
        private final AbstractRequestScheduler lifecycle =
                RequestProtocolTestSupport.schedulerMock();
        private final CapacityNotifier availability = new CapacityNotifier();
        private final PrefillEndpoint endpoint = mockPrefillEndpoint("capacity-prefill", 8080);
        private final CapacityKey key =
                CapacityKey.exact(RoleType.PREFILL, "g1", "capacity-prefill:8080");
        private final AtomicInteger availableSlots = new AtomicInteger();
        private final List<Long> admitted = new CopyOnWriteArrayList<>();
        private final List<CapacityRequest> requests = new ArrayList<>();
        private final CountDownLatch allPublished = new CountDownLatch(3);
        private final CountDownLatch headRetryStarted = new CountDownLatch(1);
        private final CountDownLatch allowHeadRetry = new CountDownLatch(1);

        private final int slotsReleasedDuringPublication;
        private final boolean pauseHeadRetry;

        private final CompletableFuture<Response> headFuture;
        private final AbstractRequestScheduler scheduler;
        private final FlexlbConfig config;

        private long nextIndependentId = 830L;

        private CapacityFixture(int slotsReleasedDuringPublication, boolean pauseHeadRetry) {
            this(slotsReleasedDuringPublication, pauseHeadRetry, new CompletableFuture<>(), 0);
        }

        private CapacityFixture(
                int slotsReleasedDuringPublication,
                boolean pauseHeadRetry,
                CompletableFuture<Response> headFuture,
                int plannerThreads) {
            this.slotsReleasedDuringPublication = slotsReleasedDuringPublication;
            this.pauseHeadRetry = pauseHeadRetry;
            this.headFuture = headFuture;
            FlexlbConfig config = SchedulingTestConfig.newConfig();
            if (plannerThreads > 0) {
                config = spy(config);
                var runtime = spy(config.getInternalRuntime());
                when(runtime.getQueuePlannerThreads()).thenReturn(plannerThreads);
                when(config.getInternalRuntime()).thenReturn(runtime);
            }
            SchedulingTestConfig.useFifoQueue(config);
            SchedulingTestConfig.useSingleDecision(config);
            ConfigService configService = mock(ConfigService.class);
            when(configService.loadBalanceConfig()).thenReturn(config);
            this.config = config;
            // Configure shared mocks before starting the scheduler. Independent requests
            // may be added concurrently later, through data rather than new stubbings.
            when(lifecycle.submit(
                            any(RequestContext.class),
                            org.mockito.ArgumentMatchers.any(Runnable.class)))
                    .thenAnswer(
                            invocation -> {
                                RequestContext context = invocation.getArgument(0);
                                var future =
                                        requestsById.get(Long.parseLong(context.getRequestId()))
                                                .future;
                                context.setFuture(future);
                                if (!future.isDone())
                                    ((AbstractRequestScheduler) invocation.getMock())
                                            .scheduleRequest(invocation.getArgument(0));
                                return future;
                            });
            when(lifecycle.tryBeginRouteSelection(any(RequestContext.class)))
                    .thenAnswer(
                            invocation ->
                                    RequestProtocolTestSupport.admissionHandle(
                                            invocation.getArgument(0)));
            when(router.select(any(RequestContext.class), isNull()))
                    .thenAnswer(
                            invocation -> {
                                RequestContext context = invocation.getArgument(0);
                                CapacityRequest request =
                                        requestsById.get(Long.parseLong(context.getRequestId()));
                                if (request.plans.incrementAndGet() == 2
                                        && context.getRequestId().equals("820")
                                        && pauseHeadRetry) {
                                    headRetryStarted.countDown();
                                    assertTrue(
                                            allowHeadRetry.await(5, TimeUnit.SECONDS),
                                            "the cancelled active request's planning gate must be released");
                                }
                                AlternativeRoute alternative = request.alternative;
                                var selected =
                                        alternative == null ? request.route : alternative.route();
                                when(selected.ctx()).thenReturn(context);
                                return selected;
                            });
            org.mockito.Mockito.doAnswer(
                            invocation -> {
                                RequestResult selectedRoute = invocation.getArgument(0);
                                RequestContext context = selectedRoute.ctx();
                                long requestId = Long.parseLong(context.getRequestId());
                                CapacityRequest request = requestsById.get(requestId);
                                AlternativeRoute alternative = request.alternative;
                                if (alternative != null && selectedRoute == alternative.route()) {
                                    admitted.add(requestId);
                                    var published =
                                            RequestProtocolTestSupport.published(
                                                    selectedRoute, alternative.item());
                                    request.future.complete(new Response());
                                    if (request.primaryEndpoint) allPublished.countDown();
                                    return published;
                                }
                                assertSame(request.route, selectedRoute);
                                int attempt = request.attempts.incrementAndGet();
                                request.beforePublication.call();
                                if (request.primaryEndpoint) {
                                    int slots =
                                            availableSlots.getAndUpdate(
                                                    value -> value > 0 ? value - 1 : value);
                                    if (slots == 0) {
                                        Runnable hook = request.beforeBlockedReturn;
                                        request.beforeBlockedReturn = null;
                                        if (hook != null) {
                                            hook.run();
                                        }
                                        if (attempt > 1) {
                                            request.retryBlockedReturn.countDown();
                                        }
                                        return RequestResult.blocked(key);
                                    }
                                    admitted.add(requestId);
                                    if (requestId == 820L && slotsReleasedDuringPublication > 0) {
                                        // Status reconciliation releases capacity before
                                        // publication returns
                                        // and before the global queue can retire this active
                                        // request.
                                        releaseSlots(slotsReleasedDuringPublication);
                                    }
                                }
                                var published =
                                        RequestProtocolTestSupport.published(
                                                selectedRoute, request.item);
                                request.future.complete(new Response());
                                if (request.primaryEndpoint) allPublished.countDown();
                                return published;
                            })
                    .when(RequestProtocolTestSupport.publication(lifecycle))
                    .publishSelectedRoute(any(RequestResult.class));
            DeliveryMetricsReporter reporter = mock(DeliveryMetricsReporter.class);
            for (long requestId = 820L; requestId < 823L; requestId++) {
                requests.add(createRequest(requestId, endpoint));
            }
            scheduler =
                    RequestProtocolTestSupport.injectPlacementDependencies(
                            lifecycle,
                            configService,
                            router,
                            reporter,
                            org.flexlb.balance.schedule.DecodePreemptionTestSupport.mock(),
                            availability);
        }

        @Override
        public void close() {
            allowHeadRetry.countDown();
            RequestProtocolTestSupport.close(scheduler);
        }

        private void submitBlockedRequests() throws InterruptedException {
            for (CapacityRequest request : requests) {
                SchedulerTestServices.entry(scheduler).submit(request.context);
            }
            RequestProtocolTestSupport.awaitGlobalCapacityWaiters(scheduler, requests.size());
        }

        private void releaseSlots(int count) {
            availableSlots.addAndGet(count);
            CapacityNotifierTestSupport.notifyChanged(availability, key);
        }

        private void awaitAllPublished() throws InterruptedException {
            assertTrue(
                    allPublished.await(5, TimeUnit.SECONDS),
                    () ->
                            "released slots must be consumed without another event; admitted="
                                    + admitted
                                    + ", freeSlots="
                                    + availableSlots.get());
        }

        private void awaitIndependentCommit() throws Exception {
            long requestId = nextIndependentId++;
            PrefillEndpoint independent = mockPrefillEndpoint("independent-" + requestId, 8080);
            CapacityRequest request = createRequest(requestId, independent);
            SchedulerTestServices.entry(scheduler).submit(request.context);
            request.future.get(5, TimeUnit.SECONDS);
        }

        private CapacityRequest createRequest(long requestId, PrefillEndpoint target) {
            CapacityRequest request =
                    new CapacityRequest(
                            config,
                            requestId,
                            target,
                            target == endpoint,
                            requestId == 820L ? headFuture : new CompletableFuture<>());
            requestsById.put(requestId, request);
            return request;
        }
    }

    private record AlternativeRoute(RequestResult route, RequestResult item) {}

    /**
     * Holds one request and its mocked placement outcomes for controlled capacity retries.
     */
    private static final class CapacityRequest {

        private final AtomicInteger plans = new AtomicInteger();

        private final RequestContext context;

        private final CompletableFuture<Response> future;

        private final AtomicInteger attempts = new AtomicInteger();

        private final RequestResult route;

        private final RequestResult item;

        private final CountDownLatch retryBlockedReturn = new CountDownLatch(1);

        private final boolean primaryEndpoint;

        private Callable<Void> beforePublication = () -> null;

        private Runnable beforeBlockedReturn;

        private volatile AlternativeRoute alternative;

        private CapacityRequest(
                FlexlbConfig config,
                long requestId,
                PrefillEndpoint target,
                boolean primaryEndpoint,
                CompletableFuture<Response> future) {
            context = context(config, requestId);
            ServerStatus prefill = new ServerStatus();
            prefill.setRole(RoleType.PREFILL);
            prefill.setRequestId(context.getRequestId());
            prefill.setSuccess(true);
            var roles = List.of(SelectionTestSupport.metadata(target, prefill, 0L));
            route = RequestProtocolTestSupport.mockSelectedRoute(context, roles);
            item = RequestProtocolTestSupport.mockSelectedRoute(context, roles);
            this.primaryEndpoint = primaryEndpoint;
            this.future = future;
        }
    }

    /**
     * Wires request scheduling with controllable role selection and Decode preemption.
     */
    private static final class Fixture {

        private final long requestId = 701L;

        private final FlexlbConfig config = SchedulingTestConfig.newConfig();

        private final ConfigService configService = mock(ConfigService.class);

        private final RequestRouteSelector router = mockRouter();

        private final DecodePreemption decodePreemption =
                org.flexlb.balance.schedule.DecodePreemptionTestSupport.mock();

        private final AbstractRequestScheduler lifecycle =
                RequestProtocolTestSupport.schedulerMock();

        private final CapacityNotifier availability = new CapacityNotifier();

        private final RequestContext context = context(config, requestId);

        private final CompletableFuture<Response> future = new CompletableFuture<>();

        private final AbstractRequestScheduler scheduler;

        private Fixture(boolean priority) {
            if (priority) {
                SchedulingTestConfig.usePriorityQueue(config);
            } else {
                SchedulingTestConfig.useFifoQueue(config);
            }
            when(configService.loadBalanceConfig()).thenReturn(config);
            when(lifecycle.submit(
                            org.mockito.ArgumentMatchers.eq(context),
                            org.mockito.ArgumentMatchers.any(Runnable.class)))
                    .thenAnswer(
                            invocation -> {
                                RequestContext registered = invocation.getArgument(0);
                                registered.setFuture(future);
                                if (!future.isDone())
                                    ((AbstractRequestScheduler) invocation.getMock())
                                            .scheduleRequest(invocation.getArgument(0));
                                return future;
                            });
            when(lifecycle.tryBeginRouteSelection(
                            org.mockito.ArgumentMatchers.argThat(
                                    request -> request != null && request.getFuture() == future)))
                    .thenAnswer(i -> RequestProtocolTestSupport.admissionHandle(i.getArgument(0)));
            doAnswer(
                            invocation -> {
                                RequestContext published = invocation.getArgument(0);
                                return published.getFuture().complete(invocation.getArgument(1));
                            })
                    .when(lifecycle)
                    .publish(any(), any(), org.mockito.ArgumentMatchers.anyBoolean());
            when(router.select(context, null))
                    .thenReturn(
                            RequestResult.rejected(
                                    Response.buildErrorResponse(
                                            StrategyErrorType.NO_PREFILL_WORKER, null)));
            scheduler =
                    RequestProtocolTestSupport.injectPlacementDependencies(
                            lifecycle,
                            configService,
                            router,
                            mock(DeliveryMetricsReporter.class),
                            decodePreemption,
                            availability);
        }
    }
}
