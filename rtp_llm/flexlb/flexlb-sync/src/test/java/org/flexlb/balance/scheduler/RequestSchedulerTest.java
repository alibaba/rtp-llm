package org.flexlb.balance.scheduler;

import org.flexlb.balance.RoutingResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.preemption.DecodePreemption;
import org.flexlb.balance.preemption.DecodePreemption.PreemptionResult;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.PreemptionStage;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.TheoryCacheHitReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.NullSource;
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
import static org.mockito.Mockito.after;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.timeout;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Verifies queue admission, cancellation and retries using controlled placement collaborators.
 */
class RequestSchedulerTest {
    @Test
    void futureCancelBeforeFailedGlobalOfferSettlesWithoutQueueTicket() throws Exception {
        FlexlbConfig config = SchedulingTestConfig.batchConfig();
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        RequestRouteSelector router = mockRouter();
        AbstractRequestScheduler lifecycle =
                org.flexlb.balance.scheduler.SchedulerTestSupport.create(
                        service,
                        mock(DeliveryMetricsReporter.class),
                        mock(RequestSchedulerReporter.class),
                        mock(TheoryCacheHitReporter.class));
        RequestContext context = context(config, 990009L);
        when(router.resolvePolicyGroup(context))
                .thenAnswer(
                        invocation -> {
                            assertTrue(context.data().getFuture().cancel(false));
                            throw new IllegalStateException("policy group unavailable");
                        });
        AbstractRequestScheduler scheduler =
                RequestProtocolTestSupport.injectPlacementDependencies(
                        lifecycle,
                        service,
                        router,
                        mock(DeliveryMetricsReporter.class),
                        org.flexlb.balance.preemption.DecodePreemptionTestSupport.mock(),
                        new CapacityNotifier());
        try {
            CompletableFuture<Response> future =
                    SchedulerTestServices.entry(scheduler).submit(context);
            assertTrue(future.isCancelled());
            RequestProtocolTestSupport.awaitCondition(
                    () -> lifecycle.repository.liveRequestCount() == 0);
            assertEquals(
                    RequestState.Phase.CANCELLED,
                    org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle)
                            .getRequestState("990009", 0L)
                            .state());
            assertEquals(
                    0,
                    org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle)
                            .liveRequestCount());
        } finally {
            RequestProtocolTestSupport.close(scheduler);
            if (RequestProtocolTestSupport.closeRegistrationAndAwaitScheduling(lifecycle)) {
                lifecycle.closeOutstandingAndTerminalize();
                SchedulerTestSupport.timer(
                                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(
                                        lifecycle))
                        .close();
                SchedulerTestSupport.closeRequestExecutors(
                        org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(lifecycle));
            }
        }
    }

    @Test
    void closeSettlesInFlightAdmissionAfterLatePlanReleasesItsHandle() throws Exception {
        FlexlbConfig config = SchedulingTestConfig.batchConfig();
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        RequestRouteSelector router = mockRouter();
        AbstractRequestScheduler lifecycle =
                org.flexlb.balance.scheduler.SchedulerTestSupport.create(
                        service,
                        mock(DeliveryMetricsReporter.class),
                        mock(RequestSchedulerReporter.class),
                        mock(TheoryCacheHitReporter.class));
        RequestContext context = context(config, 990005L);
        CountDownLatch planning = new CountDownLatch(1);
        CountDownLatch releasePlan = new CountDownLatch(1);
        CountDownLatch closeSettled = new CountDownLatch(1);
        RouteResult route = RequestProtocolTestSupport.mockSelectedRoute(context);
        when(router.select(context, null))
                .thenAnswer(
                        invocation -> {
                            planning.countDown();
                            assertTrue(releasePlan.await(5, TimeUnit.SECONDS));
                            return RoutingResult.success(
                                    RequestProtocolTestSupport.selectionOf(route));
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
                        any(ContextResult.EndEvidence.class));
        QueuedRequestScheduler coordinator =
                (QueuedRequestScheduler)
                        RequestProtocolTestSupport.injectPlacementDependencies(
                                lifecycle,
                                service,
                                router,
                                mock(DeliveryMetricsReporter.class),
                                org.flexlb.balance.preemption.DecodePreemptionTestSupport.mock(),
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
                                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(
                                        lifecycle))
                        .close();
                SchedulerTestSupport.closeRequestExecutors(
                        org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(lifecycle));
            }
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void controlTicketRemovesInFlightEntryBeforeLatePlanCanPublish(boolean deadline)
            throws Exception {
        FlexlbConfig config = SchedulingTestConfig.batchConfig();
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        RequestRouteSelector router = mockRouter();
        AbstractRequestScheduler lifecycle =
                org.flexlb.balance.scheduler.SchedulerTestSupport.create(
                        service,
                        mock(DeliveryMetricsReporter.class),
                        mock(RequestSchedulerReporter.class),
                        mock(TheoryCacheHitReporter.class));
        RequestContext context = context(config, 990001L);
        CountDownLatch planning = new CountDownLatch(1);
        CountDownLatch releasePlan = new CountDownLatch(1);
        CountDownLatch planClosed = new CountDownLatch(1);
        RouteResult route = RequestProtocolTestSupport.mockSelectedRoute(context);
        when(router.select(context, null))
                .thenAnswer(
                        invocation -> {
                            planning.countDown();
                            assertTrue(releasePlan.await(5, TimeUnit.SECONDS));
                            return RoutingResult.success(
                                    RequestProtocolTestSupport.selectionOf(route));
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
                .completeRouting(any(), any(), any());
        QueuedRequestScheduler coordinator =
                (QueuedRequestScheduler)
                        RequestProtocolTestSupport.injectPlacementDependencies(
                                lifecycle,
                                service,
                                router,
                                mock(DeliveryMetricsReporter.class),
                                org.flexlb.balance.preemption.DecodePreemptionTestSupport.mock(),
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
                lifecycle.cancel("990001", 0L, CancelReason.CLIENT_CANCELLED);
            }
            assertEquals(
                    RequestState.Phase.CANCEL_REQUESTED,
                    org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle)
                            .getRequestState("990001", 0L)
                            .state());
            assertFalse(future.isDone(), "the admission claim still owns its route");
            RequestProtocolTestSupport.awaitCondition(
                    () -> RequestProtocolTestSupport.queuedCount(coordinator) == 0);
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
                            @SuppressWarnings("unchecked")
                            var registered =
                                    (java.util.Map<CompletableFuture<Response>, ?>)
                                            org.springframework.test.util.ReflectionTestUtils
                                                    .getField(coordinator, "queuedEntries");
                            return !registered.containsKey(context);
                        } finally {
                            lock.unlock();
                        }
                    });
            verify(route, never()).requestId();
        } finally {
            releasePlan.countDown();
            coordinator.close();
            if (RequestProtocolTestSupport.closeRegistrationAndAwaitScheduling(lifecycle)) {
                lifecycle.closeOutstandingAndTerminalize();
                SchedulerTestSupport.timer(
                                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(
                                        lifecycle))
                        .close();
                SchedulerTestSupport.closeRequestExecutors(
                        org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(lifecycle));
            }
        }
    }

    @ParameterizedTest
    @NullSource
    @ValueSource(strings = {"healthy-group", "unavailable-group"})
    void unavailableRequestMustNotStopAHealthyFollowingRequest(String healthyGroup) {
        FlexlbConfig config = SchedulingTestConfig.batchConfig();
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        RequestRouteSelector router = mockRouter();

        AbstractRequestScheduler lifecycle = RequestProtocolTestSupport.schedulerMock();
        RequestContext unavailable = RequestProtocolTestSupport.context(config, 910001L);
        RequestContext healthy = RequestProtocolTestSupport.context(config, 910002L);
        CompletableFuture<Response> first = new CompletableFuture<>();
        CompletableFuture<Response> second = new CompletableFuture<>();
        when(lifecycle.submit(
                        org.mockito.ArgumentMatchers.eq(unavailable),
                        org.mockito.ArgumentMatchers.any(Runnable.class)))
                .thenAnswer(
                        invocation -> {
                            RequestContext registered = invocation.getArgument(0);
                            registered.data().setFuture(first);
                            if (!first.isDone())
                                ((AbstractRequestScheduler) invocation.getMock())
                                        .scheduleRequest(invocation.getArgument(0));
                            return first;
                        });
        when(lifecycle.submit(
                        org.mockito.ArgumentMatchers.eq(healthy),
                        org.mockito.ArgumentMatchers.any(Runnable.class)))
                .thenAnswer(
                        invocation -> {
                            RequestContext registered = invocation.getArgument(0);
                            registered.data().setFuture(second);
                            if (!second.isDone())
                                ((AbstractRequestScheduler) invocation.getMock())
                                        .scheduleRequest(invocation.getArgument(0));
                            return second;
                        });
        when(lifecycle.tryBeginRouteSelection(
                        org.mockito.ArgumentMatchers.argThat(
                                request -> request != null && request.data().getFuture() == first)))
                .thenAnswer(i -> RequestProtocolTestSupport.admissionHandle(i.getArgument(0)));
        when(lifecycle.tryBeginRouteSelection(
                        org.mockito.ArgumentMatchers.argThat(
                                request ->
                                        request != null && request.data().getFuture() == second)))
                .thenAnswer(i -> RequestProtocolTestSupport.admissionHandle(i.getArgument(0)));
        when(router.resolvePolicyGroup(unavailable)).thenReturn("unavailable-group");
        when(router.resolvePolicyGroup(healthy)).thenReturn(healthyGroup);
        when(router.select(unavailable, "unavailable-group"))
                .thenReturn(
                        RoutingResult.blocked(
                                new CapacityKey(RoleType.DECODE, "unavailable-group", null)));
        RouteResult healthyRoute = RequestProtocolTestSupport.mockSelectedRoute(healthy);
        when(router.select(healthy, healthyGroup))
                .thenReturn(
                        RoutingResult.success(
                                RequestProtocolTestSupport.selectionOf(healthyRoute)));
        org.mockito.Mockito.doReturn(
                        RoutingResult.success(
                                RequestProtocolTestSupport.mockSelectedRoute(healthy)))
                .when(RequestProtocolTestSupport.publication(lifecycle))
                .enqueueRoute(org.mockito.ArgumentMatchers.same(healthyRoute));
        AbstractRequestScheduler scheduler =
                RequestProtocolTestSupport.injectPlacementDependencies(
                        lifecycle,
                        service,
                        router,
                        mock(DeliveryMetricsReporter.class),
                        org.flexlb.balance.preemption.DecodePreemptionTestSupport.mock(),
                        new CapacityNotifier());
        try {
            SchedulerTestServices.entry(scheduler).submit(unavailable);
            verify(router, timeout(500)).select(unavailable, "unavailable-group");
            SchedulerTestServices.entry(scheduler).submit(healthy);
            verify(router, timeout(500)).select(healthy, healthyGroup);
            verify(RequestProtocolTestSupport.publication(lifecycle), timeout(500))
                    .enqueueRoute(org.mockito.ArgumentMatchers.same(healthyRoute));
        } finally {
            first.complete(new Response());
            second.complete(new Response());
            RequestProtocolTestSupport.close(scheduler);
        }
    }

    @ParameterizedTest
    @CsvSource({
        "false,success",
        "true,success",
        "false,rejected",
        "false,error",
        "false,placement_blocked",
        "false,closed",
        "false,cancelled",
        "false,closed_rollback_failure"
    })
    void priorityRescueCompletesSelectionWithTheExactRegistration(
            boolean synchronous, String outcome) throws Exception {
        FlexlbConfig config = SchedulingTestConfig.batchConfig();
        SchedulingTestConfig.allowVictim(config, PreemptionStage.DECODE_RESERVED);
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        RequestRouteSelector router = mockRouter();
        EndpointRegistry endpointRegistry = mock(EndpointRegistry.class);
        AbstractRequestScheduler lifecycle = RequestProtocolTestSupport.schedulerMock();
        DecodePreemption eviction =
                org.flexlb.balance.preemption.DecodePreemptionTestSupport.mock();
        CapacityNotifier availability = new CapacityNotifier();
        RequestContext context = context(config, 897L, 90);
        CompletableFuture<Response> future = new CompletableFuture<>();
        when(lifecycle.submit(
                        org.mockito.ArgumentMatchers.eq(context),
                        org.mockito.ArgumentMatchers.any(Runnable.class)))
                .thenAnswer(
                        invocation -> {
                            RequestContext registered = invocation.getArgument(0);
                            registered.data().setFuture(future);
                            if (!future.isDone())
                                ((AbstractRequestScheduler) invocation.getMock())
                                        .scheduleRequest(invocation.getArgument(0));
                            return future;
                        });
        RoutingContext handle = RequestProtocolTestSupport.admissionHandle(context);
        org.mockito.Mockito.doNothing()
                .when(RequestProtocolTestSupport.placementScheduler(handle))
                .completeRouting(
                        org.mockito.ArgumentMatchers.same(handle),
                        org.mockito.ArgumentMatchers.eq(ContextResult.RoutingCompletion.FAILED),
                        any());
        org.mockito.Mockito.doReturn(handle)
                .when(lifecycle)
                .tryBeginRouteSelection(
                        org.mockito.ArgumentMatchers.argThat(
                                request ->
                                        request != null && request.data().getFuture() == future));
        DecodeEndpoint selectedEndpoint = RequestProtocolTestSupport.decodeEndpoint();
        when(selectedEndpoint.ipPort()).thenReturn("selected-decode:8080");
        RouteResult selectedRoute = RequestProtocolTestSupport.mockSelectedRoute(context);
        when(selectedRoute.ctx()).thenReturn(context);
        CapacityKey blocker = CapacityKey.exact(RoleType.DECODE, "g1", "selected-decode:8080");
        when(router.select(context, null)).thenReturn(RoutingResult.blocked(blocker));
        when(router.blockedEndpoint(blocker)).thenReturn(selectedEndpoint);
        RouteResult submitted = RequestProtocolTestSupport.mockSelectedRoute(context);
        var publicationThreads = new CopyOnWriteArrayList<Thread>();
        doAnswer(
                        invocation -> {
                            publicationThreads.add(Thread.currentThread());
                            return outcome.equals("placement_blocked")
                                    ? RoutingResult.blocked(blocker)
                                    : RequestProtocolTestSupport.published(
                                            invocation.getArgument(0), submitted);
                        })
                .when(RequestProtocolTestSupport.publication(lifecycle))
                .enqueueRoute(org.mockito.ArgumentMatchers.same(selectedRoute));

        when(selectedRoute.decodeEndpoint()).thenReturn(selectedEndpoint);
        when(selectedRoute.requestId()).thenReturn(String.valueOf(context.data().getRequestId()));
        var reservation = new DecodeResources.RequestReservation(1L, String.valueOf(897L), 7L);
        when(router.select(context, null, selectedEndpoint, reservation))
                .thenAnswer(
                        call -> {
                            when(selectedRoute.decodeReservation()).thenReturn(reservation);
                            return RoutingResult.success(selectedRoute);
                        });
        when(selectedEndpoint.ownership(org.mockito.ArgumentMatchers.eq(reservation)))
                .thenReturn(
                        outcome.equals("adopt_failed")
                                ? DecodeResources.Ownership.NONE
                                : DecodeResources.Ownership.WITHDRAWABLE);
        var rollbackFailure = new IllegalStateException("rollback failed");
        if (outcome.equals("closed_rollback_failure")) {
            org.mockito.Mockito.doThrow(rollbackFailure)
                    .when(selectedEndpoint)
                    .release(reservation, DecodeResources.ReleaseReason.LOCAL_ROLLBACK);
        }
        var result =
                new PreemptionResult(
                        outcome.equals("rejected") ? null : reservation, false, "result");
        var execution = new CompletableFuture<PreemptionResult>();
        if (synchronous) {
            execution.complete(result);
        }
        when(eviction.tryPreempt(context, selectedEndpoint)).thenReturn(execution);
        AbstractRequestScheduler scheduler =
                RequestProtocolTestSupport.injectPlacementDependencies(
                        lifecycle,
                        configService,
                        router,
                        mock(DeliveryMetricsReporter.class),
                        eviction,
                        availability);
        CompletableFuture<Void> closing = null;
        try {
            SchedulerTestServices.entry(scheduler).submit(context);
            verify(eviction, timeout(1_000)).tryPreempt(context, selectedEndpoint);
            if (outcome.equals("cancelled")) {
                assertTrue(future.cancel(false));
            }
            if (outcome.startsWith("closed")) {
                closing =
                        CompletableFuture.runAsync(
                                () -> RequestProtocolTestSupport.close(scheduler));
                RequestProtocolTestSupport.awaitCondition(
                        () ->
                                ((java.util.concurrent.atomic.AtomicBoolean)
                                                org.springframework.test.util.ReflectionTestUtils
                                                        .getField(scheduler, "closed"))
                                        .get());
                assertFalse(closing.isDone(), "close must drain the outstanding preemption result");
            }
            if (!synchronous) {
                verify(RequestProtocolTestSupport.placementScheduler(handle), never())
                        .completeRouting(
                                org.mockito.ArgumentMatchers.same(handle),
                                org.mockito.ArgumentMatchers.eq(
                                        ContextResult.RoutingCompletion.FINISHED),
                                org.mockito.ArgumentMatchers.isNull());

                if (outcome.equals("error")) {
                    execution.completeExceptionally(
                            new IllegalStateException("engine unavailable"));
                } else {
                    execution.complete(result);
                }
            }
            if (!outcome.equals("rejected") && !outcome.equals("error"))
                verify(RequestProtocolTestSupport.placementScheduler(handle), timeout(1_000))
                        .completeRouting(
                                org.mockito.ArgumentMatchers.same(handle),
                                org.mockito.ArgumentMatchers.eq(
                                        ContextResult.RoutingCompletion.FINISHED),
                                org.mockito.ArgumentMatchers.isNull());
            if (closing != null) {
                closing.join();
            }
            if (outcome.equals("placement_blocked")) {
                RequestProtocolTestSupport.awaitGlobalCapacityWaiters(scheduler, 1);
                assertFalse(future.isDone());
            }
            if (outcome.equals("success")
                    || outcome.equals("placement_blocked")
                    || outcome.startsWith("closed")
                    || outcome.equals("cancelled")) {
                verify(RequestProtocolTestSupport.placementScheduler(handle), never())
                        .completeRouting(
                                org.mockito.ArgumentMatchers.same(handle),
                                org.mockito.ArgumentMatchers.eq(
                                        ContextResult.RoutingCompletion.FAILED),
                                any());
            } else {
                var rejection = org.mockito.ArgumentCaptor.forClass(Response.class);
                verify(RequestProtocolTestSupport.placementScheduler(handle), timeout(1_000))
                        .completeRouting(
                                org.mockito.ArgumentMatchers.same(handle),
                                org.mockito.ArgumentMatchers.eq(
                                        ContextResult.RoutingCompletion.FAILED),
                                rejection.capture());
                assertEquals(
                        StrategyErrorType.DISPATCH_FAILED.getErrorCode(),
                        rejection.getValue().getCode());
                assertEquals(
                        "Decode preemption did not commit: "
                                + (outcome.equals("error")
                                        ? "IllegalStateException: engine unavailable"
                                        : "result"),
                        rejection.getValue().getErrorMessage());
            }
            verify(lifecycle, times(1))
                    .tryBeginRouteSelection(
                            org.mockito.ArgumentMatchers.argThat(
                                    request ->
                                            request != null
                                                    && request.data().getFuture() == future));
            boolean hasReservation =
                    !outcome.equals("rejected")
                            && !outcome.equals("error")
                            && !outcome.startsWith("closed")
                            && !outcome.equals("cancelled");
            if (outcome.startsWith("closed") || outcome.equals("cancelled")) {
                verify(selectedEndpoint)
                        .release(reservation, DecodeResources.ReleaseReason.LOCAL_ROLLBACK);
            }
            assertTrue(
                    publicationThreads.stream()
                            .allMatch(
                                    thread ->
                                            thread.getName().startsWith("flexlb-global-planner-")),
                    "publication after preemption resumes on a planner task");
            verify(
                            RequestProtocolTestSupport.publication(lifecycle),
                            times(hasReservation && !outcome.equals("adopt_failed") ? 1 : 0))
                    .enqueueRoute(org.mockito.ArgumentMatchers.same(selectedRoute));
            verify(router, times(hasReservation ? 1 : 0))
                    .select(context, null, selectedEndpoint, reservation);
            if (outcome.equals("closed_rollback_failure")) {
                org.junit.jupiter.api.Assertions.assertSame(
                        rollbackFailure, SchedulerTestSupport.failure(lifecycle));
            }

            verify(router, times(1)).select(context, null);
        } finally {
            execution.complete(result);
            if (closing != null) {
                closing.join();
            }
            future.complete(new Response());
            RequestProtocolTestSupport.close(scheduler);
        }
    }

    @Test
    void nonBatchWaitsWhenEveryEngineRequestSlotIsOccupied() {
        FlexlbConfig config = SchedulingTestConfig.batchConfig();
        SchedulingTestConfig.useFifoQueue(config);
        SchedulingTestConfig.useFixedWindowDecision(config);
        SchedulingTestConfig.useNonBatchDispatcher(config);
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        RequestRouteSelector router = mockRouter();
        EndpointRegistry endpointRegistry = mock(EndpointRegistry.class);
        when(endpointRegistry.counts(RoleType.PREFILL))
                .thenReturn(new EndpointRegistry.Counts(0, 1));
        AtomicInteger availableSlots = new AtomicInteger();
        AbstractRequestScheduler lifecycle = RequestProtocolTestSupport.schedulerMock();
        RequestContext context = context(config, 899L);
        CompletableFuture<Response> future = new CompletableFuture<>();
        when(lifecycle.submit(
                        org.mockito.ArgumentMatchers.eq(context),
                        org.mockito.ArgumentMatchers.any(Runnable.class)))
                .thenAnswer(
                        invocation -> {
                            RequestContext registered = invocation.getArgument(0);
                            registered.data().setFuture(future);
                            if (!future.isDone())
                                ((AbstractRequestScheduler) invocation.getMock())
                                        .scheduleRequest(invocation.getArgument(0));
                            return future;
                        });
        when(lifecycle.tryBeginRouteSelection(
                        org.mockito.ArgumentMatchers.argThat(
                                request ->
                                        request != null && request.data().getFuture() == future)))
                .thenAnswer(i -> RequestProtocolTestSupport.admissionHandle(i.getArgument(0)));
        RouteResult route = RequestProtocolTestSupport.mockSelectedRoute(context);
        PrefillEndpoint endpoint = mockPrefillEndpoint("127.0.0.1", 8000);
        org.mockito.Mockito.doReturn(
                        RoutingResult.blocked(
                                CapacityKey.exact(RoleType.PREFILL, "g1", "127.0.0.1:8000")))
                .when(RequestProtocolTestSupport.publication(lifecycle))
                .enqueueRoute(org.mockito.ArgumentMatchers.same(route));
        when(router.select(context, null))
                .thenReturn(
                        RoutingResult.success(RequestProtocolTestSupport.selectionOf(route)),
                        RoutingResult.rejected(
                                Response.buildErrorResponse(
                                        StrategyErrorType.NO_PREFILL_WORKER, null)));
        CapacityNotifier availability = new CapacityNotifier();
        AbstractRequestScheduler scheduler =
                RequestProtocolTestSupport.injectPlacementDependencies(
                        lifecycle,
                        configService,
                        router,
                        mock(DeliveryMetricsReporter.class),
                        org.flexlb.balance.preemption.DecodePreemptionTestSupport.mock(),
                        availability);
        try {
            SchedulerTestServices.entry(scheduler).submit(context);
            verify(RequestProtocolTestSupport.publication(lifecycle), timeout(1_000))
                    .enqueueRoute(org.mockito.ArgumentMatchers.same(route));
            verify(router, after(100).times(1)).select(context, null);
            availableSlots.set(1);
            CapacityNotifierTestSupport.notifyChanged(
                    availability, CapacityKey.exact(RoleType.PREFILL, "g1", "127.0.0.1:8000"));
            verify(router, timeout(1_000).times(2)).select(context, null);
        } finally {
            RequestProtocolTestSupport.close(scheduler);
        }
    }

    @Test
    void planningContinuesWithoutAnAvailablePrefillEndpoint() {
        FlexlbConfig config = SchedulingTestConfig.batchConfig();
        SchedulingTestConfig.useFifoQueue(config);
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxRequests(2);
        SchedulingTestConfig.useNonBatchDispatcher(config);
        config.queueScheduler().getDecision().setMaxCollectionWaitMs(0L);
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        RequestRouteSelector router = mockRouter();
        EndpointRegistry endpointRegistry = mock(EndpointRegistry.class);
        AbstractRequestScheduler lifecycle = RequestProtocolTestSupport.schedulerMock();
        RequestContext context = context(config, 900L);
        CompletableFuture<Response> future = new CompletableFuture<>();
        when(lifecycle.submit(
                        org.mockito.ArgumentMatchers.eq(context),
                        org.mockito.ArgumentMatchers.any(Runnable.class)))
                .thenAnswer(
                        invocation -> {
                            RequestContext registered = invocation.getArgument(0);
                            registered.data().setFuture(future);
                            if (!future.isDone())
                                ((AbstractRequestScheduler) invocation.getMock())
                                        .scheduleRequest(invocation.getArgument(0));
                            return future;
                        });
        when(lifecycle.tryBeginRouteSelection(
                        org.mockito.ArgumentMatchers.argThat(
                                request ->
                                        request != null && request.data().getFuture() == future)))
                .thenAnswer(i -> RequestProtocolTestSupport.admissionHandle(i.getArgument(0)));
        when(router.select(context, null))
                .thenReturn(
                        RoutingResult.rejected(
                                Response.buildErrorResponse(
                                        StrategyErrorType.NO_PREFILL_WORKER, null)));
        AbstractRequestScheduler scheduler =
                RequestProtocolTestSupport.injectPlacementDependencies(
                        lifecycle,
                        configService,
                        router,
                        mock(DeliveryMetricsReporter.class),
                        org.flexlb.balance.preemption.DecodePreemptionTestSupport.mock(),
                        new CapacityNotifier());
        SchedulerTestServices.entry(scheduler).submit(context);
        verify(router, timeout(1_000)).select(context, null);
        RequestProtocolTestSupport.close(scheduler);
    }

    @Test
    void singleDecisionDoesNotSerializeTheGlobalPlanningFrontier() throws Exception {
        FlexlbConfig config = SchedulingTestConfig.batchConfig();
        SchedulingTestConfig.useFifoQueue(config);
        SchedulingTestConfig.useSingleDecision(config);
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        RequestRouteSelector router = mockRouter();
        EndpointRegistry endpointRegistry = mock(EndpointRegistry.class);
        AbstractRequestScheduler lifecycle = RequestProtocolTestSupport.schedulerMock();
        DeliveryMetricsReporter reporter = mock(DeliveryMetricsReporter.class);
        CapacityNotifier availability = new CapacityNotifier();
        RequestContext gate = context(config, 900L);
        CompletableFuture<Response> gateFuture = new CompletableFuture<>();
        CountDownLatch gatePlanningStarted = new CountDownLatch(1);
        CountDownLatch releaseGatePlanning = new CountDownLatch(1);
        when(lifecycle.submit(
                        org.mockito.ArgumentMatchers.eq(gate),
                        org.mockito.ArgumentMatchers.any(Runnable.class)))
                .thenAnswer(
                        invocation -> {
                            RequestContext registered = invocation.getArgument(0);
                            registered.data().setFuture(gateFuture);
                            if (!gateFuture.isDone())
                                ((AbstractRequestScheduler) invocation.getMock())
                                        .scheduleRequest(invocation.getArgument(0));
                            return gateFuture;
                        });
        when(lifecycle.tryBeginRouteSelection(
                        org.mockito.ArgumentMatchers.argThat(
                                request ->
                                        request != null
                                                && request.data().getFuture() == gateFuture)))
                .thenAnswer(i -> RequestProtocolTestSupport.admissionHandle(i.getArgument(0)));
        when(router.select(gate, null))
                .thenAnswer(
                        invocation -> {
                            gatePlanningStarted.countDown();
                            releaseGatePlanning.await(5, TimeUnit.SECONDS);
                            return RoutingResult.rejected(
                                    Response.buildErrorResponse(
                                            StrategyErrorType.NO_PREFILL_WORKER, null));
                        });
        CountDownLatch aggregatePlansStarted = new CountDownLatch(6);
        List<RequestContext> contexts = new ArrayList<>();
        for (long requestId = 901L; requestId < 907L; requestId++) {
            RequestContext context = context(config, requestId);
            CompletableFuture<Response> future = new CompletableFuture<>();
            RouteResult route = RequestProtocolTestSupport.mockSelectedRoute(context);
            when(lifecycle.submit(
                            org.mockito.ArgumentMatchers.eq(context),
                            org.mockito.ArgumentMatchers.any(Runnable.class)))
                    .thenAnswer(
                            invocation -> {
                                RequestContext registered = invocation.getArgument(0);
                                registered.data().setFuture(future);
                                if (!future.isDone())
                                    ((AbstractRequestScheduler) invocation.getMock())
                                            .scheduleRequest(invocation.getArgument(0));
                                return future;
                            });
            when(lifecycle.tryBeginRouteSelection(
                            org.mockito.ArgumentMatchers.argThat(
                                    request ->
                                            request != null
                                                    && request.data().getFuture() == future)))
                    .thenAnswer(i -> RequestProtocolTestSupport.admissionHandle(i.getArgument(0)));
            when(router.select(context, null))
                    .thenAnswer(
                            invocation -> {
                                aggregatePlansStarted.countDown();
                                return RoutingResult.success(
                                        RequestProtocolTestSupport.selectionOf(route));
                            });
            RouteResult published = RequestProtocolTestSupport.mockSelectedRoute(context);
            when(published.prefillEndpoint()).thenReturn(mock(PrefillEndpoint.class));
            org.mockito.Mockito.doReturn(RoutingResult.success(published))
                    .when(RequestProtocolTestSupport.publication(lifecycle))
                    .enqueueRoute(org.mockito.ArgumentMatchers.same(route));
            contexts.add(context);
        }
        AbstractRequestScheduler scheduler =
                RequestProtocolTestSupport.injectPlacementDependencies(
                        lifecycle,
                        configService,
                        router,
                        reporter,
                        org.flexlb.balance.preemption.DecodePreemptionTestSupport.mock(),
                        availability);
        try {
            SchedulerTestServices.entry(scheduler).submit(gate);
            assertTrue(gatePlanningStarted.await(5, TimeUnit.SECONDS));
            for (RequestContext context : contexts) {
                SchedulerTestServices.entry(scheduler).submit(context);
            }
            releaseGatePlanning.countDown();
            assertTrue(
                    aggregatePlansStarted.await(5, TimeUnit.SECONDS),
                    "all aggregate slots must be submitted from one captured frontier");
        } finally {
            releaseGatePlanning.countDown();
            RequestProtocolTestSupport.close(scheduler);
        }
    }

    @Test
    void higherPriorityRequestRunsAheadOfBlockedLowerPriorityFrontier() {
        FlexlbConfig config = SchedulingTestConfig.batchConfig();
        SchedulingTestConfig.usePriorityQueue(config);
        SchedulingTestConfig.useSingleDecision(config);
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        RequestRouteSelector router = mockRouter();
        AbstractRequestScheduler lifecycle = RequestProtocolTestSupport.schedulerMock();
        DecodePreemption decodePreemption =
                org.flexlb.balance.preemption.DecodePreemptionTestSupport.mock();
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
                            registered.data().setFuture(lowFuture);
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
                            registered.data().setFuture(highFuture);
                            if (!highFuture.isDone())
                                ((AbstractRequestScheduler) invocation.getMock())
                                        .scheduleRequest(invocation.getArgument(0));
                            return highFuture;
                        });
        when(lifecycle.tryBeginRouteSelection(
                        org.mockito.ArgumentMatchers.argThat(
                                request ->
                                        request != null
                                                && request.data().getFuture() == lowFuture)))
                .thenAnswer(i -> RequestProtocolTestSupport.admissionHandle(i.getArgument(0)));
        when(lifecycle.tryBeginRouteSelection(
                        org.mockito.ArgumentMatchers.argThat(
                                request ->
                                        request != null
                                                && request.data().getFuture() == highFuture)))
                .thenAnswer(i -> RequestProtocolTestSupport.admissionHandle(i.getArgument(0)));
        when(router.select(lowPriority, null))
                .thenReturn(RoutingResult.blocked(CapacityKey.anyGroup(RoleType.PREFILL)));
        when(router.select(highPriority, null))
                .thenReturn(
                        RoutingResult.rejected(
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
    void expiredHeadDoesNotConsumeTheCapacityOpportunity() throws Exception {
        FlexlbConfig config = SchedulingTestConfig.batchConfig();
        SchedulingTestConfig.useFifoQueue(config);
        // This contract isolates the ordered-head retry semantics, so it asks
        // for a one-request planning frontier explicitly. A wider frontier may
        // speculatively prepare a bounded suffix, which is covered by the
        // planning-frontier tests instead.
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxRequests(1);
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        RequestRouteSelector router = mockRouter();
        AbstractRequestScheduler lifecycle = RequestProtocolTestSupport.schedulerMock();
        CapacityNotifier availability = new CapacityNotifier();
        CapacityKey blocker = CapacityKey.anyGroup(RoleType.PREFILL);
        RequestContext expired = context(config, 801L);
        RequestContext follower = context(config, 802L);
        CompletableFuture<Response> expiredFuture = new CompletableFuture<>();
        CompletableFuture<Response> followerFuture = new CompletableFuture<>();
        when(lifecycle.submit(
                        org.mockito.ArgumentMatchers.eq(expired),
                        org.mockito.ArgumentMatchers.any(Runnable.class)))
                .thenAnswer(
                        invocation -> {
                            RequestContext registered = invocation.getArgument(0);
                            registered.data().setFuture(expiredFuture);
                            if (!expiredFuture.isDone())
                                ((AbstractRequestScheduler) invocation.getMock())
                                        .scheduleRequest(invocation.getArgument(0));
                            return expiredFuture;
                        });
        when(lifecycle.submit(
                        org.mockito.ArgumentMatchers.eq(follower),
                        org.mockito.ArgumentMatchers.any(Runnable.class)))
                .thenAnswer(
                        invocation -> {
                            RequestContext registered = invocation.getArgument(0);
                            registered.data().setFuture(followerFuture);
                            if (!followerFuture.isDone())
                                ((AbstractRequestScheduler) invocation.getMock())
                                        .scheduleRequest(invocation.getArgument(0));
                            return followerFuture;
                        });
        when(lifecycle.tryBeginRouteSelection(
                        org.mockito.ArgumentMatchers.argThat(
                                request ->
                                        request != null
                                                && request.data().getFuture() == expiredFuture)))
                .thenAnswer(i -> RequestProtocolTestSupport.admissionHandle(i.getArgument(0)));
        when(lifecycle.tryBeginRouteSelection(
                        org.mockito.ArgumentMatchers.argThat(
                                request ->
                                        request != null
                                                && request.data().getFuture() == followerFuture)))
                .thenAnswer(i -> RequestProtocolTestSupport.admissionHandle(i.getArgument(0)));
        when(router.select(any(), any())).thenReturn(RoutingResult.blocked(blocker));
        AbstractRequestScheduler scheduler =
                RequestProtocolTestSupport.injectPlacementDependencies(
                        lifecycle,
                        configService,
                        router,
                        mock(DeliveryMetricsReporter.class),
                        org.flexlb.balance.preemption.DecodePreemptionTestSupport.mock(),
                        availability);
        SchedulerTestServices.entry(scheduler).submit(expired);
        SchedulerTestServices.entry(scheduler).submit(follower);
        // Both independent requests may plan before the capacity event. Wait for
        // actual park registration so the event represents exactly one retry.
        RequestProtocolTestSupport.awaitGlobalCapacityWaiters(scheduler, 2);
        expired.data()
                .setSchedulingMetadata(
                        SchedulingMetadata.explicit(50, System.currentTimeMillis() - 1L));
        CapacityNotifierTestSupport.notifyChanged(availability, blocker);
        verify(router, timeout(1_000).times(2)).select(follower, null);
        verify(router, times(1)).select(expired, null);
        RequestProtocolTestSupport.close(scheduler);
    }

    @Test
    void endpointConflictAllowsIndependentSuffixCommit() throws Exception {
        FlexlbConfig config = SchedulingTestConfig.batchConfig();
        SchedulingTestConfig.useFifoQueue(config);
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxRequests(2);
        config.queueScheduler().getDecision().setMaxCollectionWaitMs(0L);
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        RequestRouteSelector router = mockRouter();
        AbstractRequestScheduler lifecycle = RequestProtocolTestSupport.schedulerMock();
        CapacityNotifier availability = new CapacityNotifier();
        RequestContext blocked = context(config, 803L);
        RequestContext independent = context(config, 804L);
        CompletableFuture<Response> blockedFuture = new CompletableFuture<>();
        CompletableFuture<Response> independentFuture = new CompletableFuture<>();
        when(lifecycle.submit(
                        org.mockito.ArgumentMatchers.eq(blocked),
                        org.mockito.ArgumentMatchers.any(Runnable.class)))
                .thenAnswer(
                        invocation -> {
                            RequestContext registered = invocation.getArgument(0);
                            registered.data().setFuture(blockedFuture);
                            if (!blockedFuture.isDone())
                                ((AbstractRequestScheduler) invocation.getMock())
                                        .scheduleRequest(invocation.getArgument(0));
                            return blockedFuture;
                        });
        when(lifecycle.submit(
                        org.mockito.ArgumentMatchers.eq(independent),
                        org.mockito.ArgumentMatchers.any(Runnable.class)))
                .thenAnswer(
                        invocation -> {
                            RequestContext registered = invocation.getArgument(0);
                            registered.data().setFuture(independentFuture);
                            if (!independentFuture.isDone())
                                ((AbstractRequestScheduler) invocation.getMock())
                                        .scheduleRequest(invocation.getArgument(0));
                            return independentFuture;
                        });
        when(lifecycle.tryBeginRouteSelection(
                        org.mockito.ArgumentMatchers.argThat(
                                request ->
                                        request != null
                                                && request.data().getFuture() == blockedFuture)))
                .thenAnswer(i -> RequestProtocolTestSupport.admissionHandle(i.getArgument(0)));
        when(lifecycle.tryBeginRouteSelection(
                        org.mockito.ArgumentMatchers.argThat(
                                request ->
                                        request != null
                                                && request.data().getFuture()
                                                        == independentFuture)))
                .thenAnswer(i -> RequestProtocolTestSupport.admissionHandle(i.getArgument(0)));
        PrefillEndpoint fullEndpoint = mockPrefillEndpoint("full-prefill", 8080);
        PrefillEndpoint availableEndpoint = mock(PrefillEndpoint.class);
        RouteResult blockedRoute = RequestProtocolTestSupport.mockSelectedRoute(blocked);
        RouteResult independentRoute = RequestProtocolTestSupport.mockSelectedRoute(independent);
        RouteResult independentItem = RequestProtocolTestSupport.mockSelectedRoute(independent);
        CapacityKey exactBlocker = CapacityKey.exact(RoleType.PREFILL, "g1", "full-prefill:8080");
        CountDownLatch blockedRouteAttempts = new CountDownLatch(2);
        when(independentItem.prefillEndpoint()).thenReturn(availableEndpoint);
        when(router.select(blocked, null))
                .thenAnswer(
                        invocation -> {
                            blockedRouteAttempts.countDown();
                            return RoutingResult.success(
                                    RequestProtocolTestSupport.selectionOf(blockedRoute));
                        });
        when(router.select(independent, null))
                .thenReturn(
                        RoutingResult.success(
                                RequestProtocolTestSupport.selectionOf(independentRoute)));
        org.mockito.Mockito.doReturn(RoutingResult.blocked(exactBlocker))
                .when(RequestProtocolTestSupport.publication(lifecycle))
                .enqueueRoute(org.mockito.ArgumentMatchers.same(blockedRoute));
        when(independentRoute.prefillEndpoint()).thenReturn(availableEndpoint);
        org.mockito.Mockito.doReturn(RoutingResult.success(independentItem))
                .when(RequestProtocolTestSupport.publication(lifecycle))
                .enqueueRoute(org.mockito.ArgumentMatchers.same(independentRoute));
        AbstractRequestScheduler scheduler =
                RequestProtocolTestSupport.injectPlacementDependencies(
                        lifecycle,
                        configService,
                        router,
                        mock(DeliveryMetricsReporter.class),
                        org.flexlb.balance.preemption.DecodePreemptionTestSupport.mock(),
                        availability);
        try {
            SchedulerTestServices.entry(scheduler).submit(blocked);
            SchedulerTestServices.entry(scheduler).submit(independent);
            verify(RequestProtocolTestSupport.publication(lifecycle), timeout(1_000).times(1))
                    .enqueueRoute(org.mockito.ArgumentMatchers.same(blockedRoute));
            verify(RequestProtocolTestSupport.publication(lifecycle), timeout(1_000))
                    .enqueueRoute(org.mockito.ArgumentMatchers.same(independentRoute));
            assertFalse(blockedFuture.isDone());
            CapacityNotifierTestSupport.notifyChanged(
                    availability, new CapacityKey(RoleType.PREFILL, "g1", null));
            assertTrue(
                    blockedRouteAttempts.await(1, TimeUnit.SECONDS),
                    "group capacity changes must allow selection to consider another endpoint");
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
                RouteResult route = RequestProtocolTestSupport.mockSelectedRoute(request.context);
                ServerStatus status = new ServerStatus();
                status.setRole(RoleType.PREFILL);
                status.setRequestId(request.context.data().getRequestId());
                status.setSuccess(true);
                RouteResult item =
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
    void staleEndpointConflictReplansWithoutFixedRetryBudget() {
        FlexlbConfig config = SchedulingTestConfig.batchConfig();
        SchedulingTestConfig.useFifoQueue(config);
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        RequestRouteSelector router = mockRouter();
        AbstractRequestScheduler lifecycle = RequestProtocolTestSupport.schedulerMock();
        CapacityNotifier availability = new CapacityNotifier();
        RequestContext context = context(config, 807L);
        CompletableFuture<Response> future = new CompletableFuture<>();
        when(lifecycle.submit(
                        org.mockito.ArgumentMatchers.eq(context),
                        org.mockito.ArgumentMatchers.any(Runnable.class)))
                .thenAnswer(
                        invocation -> {
                            RequestContext registered = invocation.getArgument(0);
                            registered.data().setFuture(future);
                            if (!future.isDone())
                                ((AbstractRequestScheduler) invocation.getMock())
                                        .scheduleRequest(invocation.getArgument(0));
                            return future;
                        });
        when(lifecycle.tryBeginRouteSelection(
                        org.mockito.ArgumentMatchers.argThat(
                                request ->
                                        request != null && request.data().getFuture() == future)))
                .thenAnswer(i -> RequestProtocolTestSupport.admissionHandle(i.getArgument(0)));
        PrefillEndpoint staleEndpoint = mockPrefillEndpoint("stale-prefill", 8080);
        RouteResult staleRoute = RequestProtocolTestSupport.mockSelectedRoute(context);
        RouteResult freshRoute = RequestProtocolTestSupport.mockSelectedRoute(context);
        RouteResult committed = RequestProtocolTestSupport.mockSelectedRoute(context);
        when(committed.prefillEndpoint()).thenReturn(mock(PrefillEndpoint.class));
        CapacityKey exactBlocker = CapacityKey.exact(RoleType.PREFILL, "g1", "stale-prefill:8080");
        when(router.select(context, null))
                .thenReturn(
                        RoutingResult.success(RequestProtocolTestSupport.selectionOf(staleRoute)),
                        RoutingResult.success(RequestProtocolTestSupport.selectionOf(freshRoute)));
        org.mockito.Mockito.doReturn(RoutingResult.retry())
                .when(RequestProtocolTestSupport.publication(lifecycle))
                .enqueueRoute(org.mockito.ArgumentMatchers.same(staleRoute));
        org.mockito.Mockito.doReturn(RoutingResult.success(committed))
                .when(RequestProtocolTestSupport.publication(lifecycle))
                .enqueueRoute(org.mockito.ArgumentMatchers.same(freshRoute));
        AbstractRequestScheduler scheduler =
                RequestProtocolTestSupport.injectPlacementDependencies(
                        lifecycle,
                        configService,
                        router,
                        mock(DeliveryMetricsReporter.class),
                        org.flexlb.balance.preemption.DecodePreemptionTestSupport.mock(),
                        availability);
        try {
            SchedulerTestServices.entry(scheduler).submit(context);
            verify(RequestProtocolTestSupport.publication(lifecycle), timeout(1_000))
                    .enqueueRoute(org.mockito.ArgumentMatchers.same(freshRoute));
            verify(router, times(2)).select(context, null);
        } finally {
            RequestProtocolTestSupport.close(scheduler);
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
        verify(fixture.lifecycle, never()).currentRoutingContext(any());
        RequestProtocolTestSupport.close(fixture.scheduler);
    }

    @Test
    void prioritySelectorMissWaitsWithoutInventingAnEvictionRoute() {
        Fixture fixture = new Fixture(true);
        when(fixture.router.select(fixture.context, null))
                .thenReturn(RoutingResult.blocked(CapacityKey.anyGroup(RoleType.PREFILL)));
        CompletableFuture<Response> waiting =
                SchedulerTestServices.entry(fixture.scheduler).submit(fixture.context);

        verify(fixture.router, timeout(1_000)).select(fixture.context, null);
        verify(fixture.decodePreemption, never()).tryPreempt(any(), any());
        assertFalse(waiting.isDone());
        verify(fixture.lifecycle, never()).currentRoutingContext(any());
        RequestProtocolTestSupport.close(fixture.scheduler);
    }

    @Test
    void fifoTemporaryDecodeMissRemainsQueued() {
        Fixture fixture = new Fixture(false);
        when(fixture.router.select(fixture.context, null))
                .thenReturn(RoutingResult.blocked(CapacityKey.anyGroup(RoleType.DECODE)));
        CompletableFuture<Response> waiting =
                SchedulerTestServices.entry(fixture.scheduler).submit(fixture.context);

        verify(fixture.router, timeout(1_000)).select(fixture.context, null);
        assertFalse(waiting.isDone());
        verify(fixture.decodePreemption, never()).tryPreempt(any(), any());
        verify(fixture.lifecycle, never()).currentRoutingContext(any());
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
        context.data().setRequest(request);
        context.data()
                .setGenerateInputPb(com.google.protobuf.ByteString.copyFromUtf8("test-input"));
        context.data()
                .setSchedulingMetadata(
                        SchedulingMetadata.explicit(
                                priority, System.currentTimeMillis() + 60_000L));
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
        private final ConcurrentLinkedQueue<CapacityRequest> pendingReports =
                new ConcurrentLinkedQueue<>();
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
            FlexlbConfig config = SchedulingTestConfig.batchConfig();
            if (plannerThreads > 0) {
                config = spy(config);
                var runtime = spy(config.getInternalRuntime());
                when(runtime.getQueuePlannerThreads()).thenReturn(plannerThreads);
                when(config.getInternalRuntime()).thenReturn(runtime);
            }
            SchedulingTestConfig.useFifoQueue(config);
            SchedulingTestConfig.useSingleDecision(config);
            SchedulingTestConfig.useNonBatchDispatcher(config).setMaxInflightPerPrefillWorker(3);
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
                                        requestsById.get(
                                                        Long.parseLong(
                                                                context.data().getRequestId()))
                                                .future;
                                context.data().setFuture(future);
                                if (!future.isDone())
                                    ((AbstractRequestScheduler) invocation.getMock())
                                            .scheduleRequest(invocation.getArgument(0));
                                return future;
                            });
            when(lifecycle.tryBeginRouteSelection(any(RequestContext.class)))
                    .thenAnswer(
                            invocation -> {
                                var request =
                                        requestsById.get(
                                                Long.parseLong(
                                                        invocation
                                                                .getArgument(
                                                                        0, RequestContext.class)
                                                                .data()
                                                                .getRequestId()));
                                var handle =
                                        RequestProtocolTestSupport.admissionHandle(
                                                invocation.getArgument(0));
                                doAnswer(
                                                finished -> {
                                                    if (pendingReports.remove(request)) {
                                                        request.future.complete(new Response());
                                                        if (request.primaryEndpoint) {
                                                            allPublished.countDown();
                                                        }
                                                    }
                                                    return null;
                                                })
                                        .when(RequestProtocolTestSupport.placementScheduler(handle))
                                        .completeRouting(
                                                org.mockito.ArgumentMatchers.same(handle),
                                                org.mockito.ArgumentMatchers.eq(
                                                        ContextResult.RoutingCompletion.FINISHED),
                                                org.mockito.ArgumentMatchers.isNull());
                                return handle;
                            });
            when(router.select(any(RequestContext.class), isNull()))
                    .thenAnswer(
                            invocation -> {
                                RequestContext context = invocation.getArgument(0);
                                CapacityRequest request =
                                        requestsById.get(
                                                Long.parseLong(context.data().getRequestId()));
                                if (request.plans.incrementAndGet() == 2
                                        && context.data().getRequestId().equals("820")
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
                                return RoutingResult.success(
                                        RequestProtocolTestSupport.selectionOf(selected));
                            });
            org.mockito.Mockito.doAnswer(
                            invocation -> {
                                RouteResult selectedRoute = invocation.getArgument(0);
                                RequestContext context = selectedRoute.ctx();
                                long requestId = Long.parseLong(context.data().getRequestId());
                                CapacityRequest request = requestsById.get(requestId);
                                AlternativeRoute alternative = request.alternative;
                                if (alternative != null && selectedRoute == alternative.route()) {
                                    admitted.add(requestId);
                                    pendingReports.add(request);
                                    return RequestProtocolTestSupport.published(
                                            selectedRoute, alternative.item());
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
                                        return RoutingResult.blocked(key);
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
                                pendingReports.add(request);
                                return RequestProtocolTestSupport.published(
                                        selectedRoute, request.item);
                            })
                    .when(RequestProtocolTestSupport.publication(lifecycle))
                    .enqueueRoute(any(RouteResult.class));
            DeliveryMetricsReporter reporter = mock(DeliveryMetricsReporter.class);
            // The exact placement scope finishes after the coordinator retires the queue entry.
            // A delayed Future callback therefore cannot hide a lost capacity handoff.
            for (long requestId = 820L; requestId < 823L; requestId++) {
                requests.add(createRequest(requestId, endpoint));
            }
            scheduler =
                    RequestProtocolTestSupport.injectPlacementDependencies(
                            lifecycle,
                            configService,
                            router,
                            reporter,
                            org.flexlb.balance.preemption.DecodePreemptionTestSupport.mock(),
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

    private record AlternativeRoute(RouteResult route, RouteResult item) {}

    /**
     * Holds one request and its mocked placement outcomes for controlled capacity retries.
     */
    private static final class CapacityRequest {

        private final AtomicInteger plans = new AtomicInteger();

        private final RequestContext context;

        private final CompletableFuture<Response> future;

        private final AtomicInteger attempts = new AtomicInteger();

        private final RouteResult route;

        private final RouteResult item;

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
            prefill.setRequestId(context.data().getRequestId());
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

        private final FlexlbConfig config = SchedulingTestConfig.batchConfig();

        private final ConfigService configService = mock(ConfigService.class);

        private final RequestRouteSelector router = mockRouter();

        private final DecodePreemption decodePreemption =
                org.flexlb.balance.preemption.DecodePreemptionTestSupport.mock();

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
                                registered.data().setFuture(future);
                                if (!future.isDone())
                                    ((AbstractRequestScheduler) invocation.getMock())
                                            .scheduleRequest(invocation.getArgument(0));
                                return future;
                            });
            when(lifecycle.tryBeginRouteSelection(
                            org.mockito.ArgumentMatchers.argThat(
                                    request ->
                                            request != null
                                                    && request.data().getFuture() == future)))
                    .thenAnswer(i -> RequestProtocolTestSupport.admissionHandle(i.getArgument(0)));
            doAnswer(
                            invocation -> {
                                RequestContext published = invocation.getArgument(0);
                                return published
                                        .data()
                                        .getFuture()
                                        .complete(invocation.getArgument(1));
                            })
                    .when(lifecycle)
                    .publish(any(), any(), org.mockito.ArgumentMatchers.anyBoolean());
            when(router.select(context, null))
                    .thenReturn(
                            RoutingResult.rejected(
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
