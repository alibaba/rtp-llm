package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.eviction.DecodeCapacityAcquirer.PreemptionResult;
import org.flexlb.balance.eviction.DecodeCapacityAcquirer;
import org.flexlb.balance.scheduler.RequestContext.AdmissionHandle;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.VictimStage;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.RecentCacheKeyTraceReporter;
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
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.BiConsumer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
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

class RequestSchedulerTest {

    @Test
    void futureCancelBeforeFailedGlobalOfferSettlesWithoutQueueTicket() throws Exception {
        FlexlbConfig config = SchedulingTestConfig.batchConfig();
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        RequestWorkerSelector router = mockRouter();
        AbstractRequestScheduler lifecycle = org.flexlb.balance.scheduler.SchedulerTestSupport.create(service, mock(DeliveryMetricsReporter.class), mock(RequestSchedulerReporter.class),
                mock(RecentCacheKeyTraceReporter.class));
        RequestContext context = context(config, 990009L);
        when(router.resolvePolicyGroup(context)).thenAnswer(invocation -> {
            assertTrue(context.getFuture().cancel(false));
            throw new IllegalStateException("policy group unavailable");
        });
        RequestScheduler scheduler = RequestProtocolTestSupport.configure(lifecycle, service, router, mock(DeliveryMetricsReporter.class), mock(DecodeCapacityAcquirer.class), new PlacementAvailability());
        try {
            CompletableFuture<Response> future = scheduler.submit(context);
            assertTrue(future.isCancelled());
            assertEquals(RequestState.Phase.CANCELLED, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle).getRequestState(990009L, 0L).state());
            assertEquals(0, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle).liveRequestCount());
        } finally {
            RequestProtocolTestSupport.close(scheduler);
            if (RequestProtocolTestSupport.closeAdmissionAndAwaitMutations(lifecycle)) {
                lifecycle.closeOutstandingAndTerminalize();
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(lifecycle).timer().close();
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(lifecycle).closeRequestExecutors();
            }
        }
    }

    @Test
    void futureCancelBeforeGlobalOfferSettlesOnIngressHandoff() throws Exception {
        FlexlbConfig config = SchedulingTestConfig.batchConfig();
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        RequestWorkerSelector router = mockRouter();
        AbstractRequestScheduler lifecycle = org.flexlb.balance.scheduler.SchedulerTestSupport.create(service, mock(DeliveryMetricsReporter.class), mock(RequestSchedulerReporter.class),
                mock(RecentCacheKeyTraceReporter.class));
        RequestContext context = context(config, 990008L);
        doAnswer(invocation -> {
            @SuppressWarnings("unchecked")
            CompletableFuture<Response> future = (CompletableFuture<Response>) invocation.callRealMethod();
            assertTrue(future.cancel(false));
            return future;
        }).when(lifecycle).register(org.mockito.ArgumentMatchers.eq(context), any());
        RequestScheduler scheduler = RequestProtocolTestSupport.configure(lifecycle, service, router, mock(DeliveryMetricsReporter.class), mock(DecodeCapacityAcquirer.class), new PlacementAvailability());
        try {
            CompletableFuture<Response> future = scheduler.submit(context);
            assertTrue(future.isCancelled());
            assertEquals(RequestState.Phase.CANCELLED, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle).getRequestState(990008L, 0L).state());
            assertEquals(0, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle).liveRequestCount());
            verify(router, never()).select(any(), any());
        } finally {
            RequestProtocolTestSupport.close(scheduler);
            if (RequestProtocolTestSupport.closeAdmissionAndAwaitMutations(lifecycle)) {
                lifecycle.closeOutstandingAndTerminalize();
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(lifecycle).timer().close();
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(lifecycle).closeRequestExecutors();
            }
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = { false, true })
    void futureCancelCompletesSynchronouslyButGlobalOwnerSettlesResources(boolean priorBusinessCancel) throws Exception {
        FlexlbConfig config = SchedulingTestConfig.batchConfig();
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        AbstractRequestScheduler lifecycle = org.flexlb.balance.scheduler.SchedulerTestSupport.create(service, mock(DeliveryMetricsReporter.class), mock(RequestSchedulerReporter.class),
                mock(RecentCacheKeyTraceReporter.class));
        CountDownLatch controlStarted = new CountDownLatch(1);
        CountDownLatch releaseControl = new CountDownLatch(1);
        AtomicReference<Thread> targetOwner = new AtomicReference<>();
        doAnswer(invocation -> {
            long id = ((RequestContext) invocation.getArgument(0)).getRequestId();
            if (id == 990006L) {
                controlStarted.countDown();
                assertTrue(releaseControl.await(5, TimeUnit.SECONDS));
            } else if (id == 990007L) {
                targetOwner.set(Thread.currentThread());
            }
            return invocation.callRealMethod();
        }).when((QueuedRequestScheduler) lifecycle).onGlobalControl(any());
        QueuedRequestScheduler coordinator = RequestProtocolTestSupport.queue(service, mockRouter(), mock(DeliveryMetricsReporter.class), mock(DecodeCapacityAcquirer.class), lifecycle, new PlacementAvailability());
        try {
            RequestContext gate = context(config, 990006L);
            CompletableFuture<Response> gateFuture = RequestProtocolTestSupport.register(lifecycle, gate);
            lifecycle.cancel(990006L, 0L, CancelReason.CLIENT_CANCELLED);
            assertTrue(coordinator.trySubmitRegistered(gate));
            assertTrue(controlStarted.await(5, TimeUnit.SECONDS));
            RequestContext target = context(config, 990007L);
            CompletableFuture<Response> future = RequestProtocolTestSupport.register(lifecycle, target);
            assertTrue(coordinator.trySubmitRegistered(target));
            if (priorBusinessCancel) {
                lifecycle.cancel(990007L, 0L, CancelReason.CLIENT_CANCELLED);
                Response success = new Response();
                success.setSuccess(true);
                assertFalse(future.complete(success));
                assertFalse(future.completeExceptionally(new IllegalStateException("late completion")));
            }
            assertTrue(future.cancel(false));
            assertTrue(future.isCancelled());
            assertEquals(RequestState.Phase.CANCEL_REQUESTED, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle).getRequestState(990007L, 0L).state());
            assertEquals(1, RequestProtocolTestSupport.queuedCount(coordinator), "the decision owner has not consumed the ticket");
            releaseControl.countDown();
            RequestProtocolTestSupport.awaitCondition(() -> org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle).getRequestState(990007L, 0L).state() == RequestState.Phase.CANCELLED);
            assertThrows(java.util.concurrent.CancellationException.class, future::join);
            assertEquals("flexlb-global-decision", targetOwner.get().getName());
        } finally {
            releaseControl.countDown();
            coordinator.close();
            if (RequestProtocolTestSupport.closeAdmissionAndAwaitMutations(lifecycle)) {
                lifecycle.closeOutstandingAndTerminalize();
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(lifecycle).timer().close();
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(lifecycle).closeRequestExecutors();
            }
        }
    }

    @Test
    void closeSettlesInFlightAdmissionAfterLatePlanReleasesItsHandle() throws Exception {
        FlexlbConfig config = SchedulingTestConfig.batchConfig();
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        RequestWorkerSelector router = mockRouter();
        AbstractRequestScheduler lifecycle = org.flexlb.balance.scheduler.SchedulerTestSupport.create(service, mock(DeliveryMetricsReporter.class), mock(RequestSchedulerReporter.class),
                mock(RecentCacheKeyTraceReporter.class));
        RequestContext context = context(config, 990005L);
        CountDownLatch planning = new CountDownLatch(1);
        CountDownLatch releasePlan = new CountDownLatch(1);
        CountDownLatch closeSettled = new CountDownLatch(1);
        RequestRoute route = mock(RequestRoute.class);
        when(router.select(context, null)).thenAnswer(invocation -> {
            planning.countDown();
            assertTrue(releasePlan.await(5, TimeUnit.SECONDS));
            return PlacementResult.success(route);
        });
        doAnswer(invocation -> {
            Object result = invocation.callRealMethod();
            closeSettled.countDown();
            return result;
        }).when((QueuedRequestScheduler) lifecycle).settleGlobalQueueClose(org.mockito.ArgumentMatchers.same(context));
        QueuedRequestScheduler coordinator = RequestProtocolTestSupport.queue(service, router, mock(DeliveryMetricsReporter.class), mock(DecodeCapacityAcquirer.class), lifecycle, new PlacementAvailability());
        Thread closer = null;
        Thread secondCloser = null;
        try {
            CompletableFuture<Response> future = RequestProtocolTestSupport.register(lifecycle, context);
            assertTrue(coordinator.trySubmitRegistered(context));
            assertTrue(planning.await(5, TimeUnit.SECONDS));
            closer = new Thread(coordinator::close);
            closer.start();
            assertTrue(closeSettled.await(5, TimeUnit.SECONDS));
            assertFalse(future.isDone(), "the in-flight admission still owns its handle");
            assertTrue(closer.isAlive(), "close must wait for the late planner to release its handle");
            CountDownLatch secondEntered = new CountDownLatch(1);
            CountDownLatch secondReturned = new CountDownLatch(1);
            secondCloser = new Thread(() -> {
                secondEntered.countDown();
                try {
                    coordinator.close();
                } finally {
                    secondReturned.countDown();
                }
            });
            secondCloser.start();
            assertTrue(secondEntered.await(5, TimeUnit.SECONDS));
            assertFalse(secondReturned.await(200, TimeUnit.MILLISECONDS), "a second close caller must observe the same drain");
            releasePlan.countDown();
            assertEquals(StrategyErrorType.DISPATCH_FAILED.getErrorCode(), future.get(5, TimeUnit.SECONDS).getCode());
            verify(route, timeout(5_000)).close();
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
            if (RequestProtocolTestSupport.closeAdmissionAndAwaitMutations(lifecycle)) {
                lifecycle.closeOutstandingAndTerminalize();
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(lifecycle).timer().close();
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(lifecycle).closeRequestExecutors();
            }
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = { false, true })
    void closePreservesAcceptedCancellationBeforeGlobalControlRuns(boolean futureCancellation) throws Exception {
        FlexlbConfig config = SchedulingTestConfig.batchConfig();
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        AbstractRequestScheduler lifecycle = org.flexlb.balance.scheduler.SchedulerTestSupport.create(service, mock(DeliveryMetricsReporter.class), mock(RequestSchedulerReporter.class),
                mock(RecentCacheKeyTraceReporter.class));
        CountDownLatch controlStarted = new CountDownLatch(1);
        CountDownLatch releaseControl = new CountDownLatch(1);
        doAnswer(invocation -> {
            if (((RequestContext) invocation.getArgument(0)).getRequestId() == 990003L) {
                controlStarted.countDown();
                assertTrue(releaseControl.await(5, TimeUnit.SECONDS));
            }
            return invocation.callRealMethod();
        }).when((QueuedRequestScheduler) lifecycle).onGlobalControl(any());
        doAnswer(invocation -> {
            if (((RequestContext) invocation.getArgument(0)).getRequestId() == 990004L) {
                if (futureCancellation) {
                    CompletableFuture<Response> future = ((RequestContext) invocation.getArgument(0)).getFuture();
                    assertTrue(future.cancel(false));
                } else {
                    assertEquals(RequestState.Phase.CANCEL_REQUESTED, lifecycle.cancel(990004L, 0L, CancelReason.CLIENT_CANCELLED).state());
                }
            }
            return invocation.callRealMethod();
        }).when((QueuedRequestScheduler) lifecycle).settleGlobalQueueClose(any());
        QueuedRequestScheduler coordinator = RequestProtocolTestSupport.queue(service, mockRouter(), mock(DeliveryMetricsReporter.class), mock(DecodeCapacityAcquirer.class), lifecycle, new PlacementAvailability());
        Thread closer = null;
        try {
            RequestContext gate = context(config, 990003L);
            CompletableFuture<Response> gateFuture = RequestProtocolTestSupport.register(lifecycle, gate);
            lifecycle.cancel(990003L, 0L, CancelReason.CLIENT_CANCELLED);
            assertTrue(coordinator.trySubmitRegistered(gate));
            assertTrue(controlStarted.await(5, TimeUnit.SECONDS));
            RequestContext target = context(config, 990004L);
            CompletableFuture<Response> future = RequestProtocolTestSupport.register(lifecycle, target);
            assertTrue(coordinator.trySubmitRegistered(target));
            closer = new Thread(coordinator::close);
            closer.start();
            RequestProtocolTestSupport.awaitCondition(() -> {
                var closed = (java.util.concurrent.atomic.AtomicBoolean) org.springframework.test.util.ReflectionTestUtils.getField(coordinator, "closed");
                return closed.get();
            });
            releaseControl.countDown();
            if (futureCancellation) {
                RequestProtocolTestSupport.awaitCondition(() -> org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle).getRequestState(990004L, 0L).state() == RequestState.Phase.CANCELLED);
                assertThrows(java.util.concurrent.CancellationException.class, future::join);
            } else {
                assertEquals(StrategyErrorType.REQUEST_CANCELLED.getErrorCode(), future.get(5, TimeUnit.SECONDS).getCode());
            }
            verify((QueuedRequestScheduler) lifecycle).settleGlobalQueueClose(target);
            verify((QueuedRequestScheduler) lifecycle, never()).onGlobalControl(target);
        } finally {
            releaseControl.countDown();
            coordinator.close();
            if (closer != null) {
                closer.join(5_000);
                assertFalse(closer.isAlive());
            }
            if (RequestProtocolTestSupport.closeAdmissionAndAwaitMutations(lifecycle)) {
                lifecycle.closeOutstandingAndTerminalize();
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(lifecycle).timer().close();
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(lifecycle).closeRequestExecutors();
            }
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = { false, true })
    void cancellationBetweenRegistrationAndOfferGetsAnExactControlTicket(boolean futureCancellation) throws Exception {
        FlexlbConfig config = SchedulingTestConfig.batchConfig();
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        RequestWorkerSelector router = mockRouter();
        AbstractRequestScheduler lifecycle = org.flexlb.balance.scheduler.SchedulerTestSupport.create(service, mock(DeliveryMetricsReporter.class), mock(RequestSchedulerReporter.class),
                mock(RecentCacheKeyTraceReporter.class));
        QueuedRequestScheduler coordinator = RequestProtocolTestSupport.queue(service, router, mock(DeliveryMetricsReporter.class), mock(DecodeCapacityAcquirer.class), lifecycle, new PlacementAvailability());
        RequestContext context = context(config, 990002L);
        CompletableFuture<Response> future = RequestProtocolTestSupport.register(lifecycle, context);
        try {
            if (futureCancellation) {
                assertTrue(future.cancel(false));
                assertTrue(future.isCancelled());
            } else {
                assertEquals(RequestState.Phase.CANCEL_REQUESTED, lifecycle.cancel(990002L, 0L, CancelReason.CLIENT_CANCELLED).state());
                assertFalse(future.isDone());
            }
            assertTrue(coordinator.trySubmitRegistered(context));
            if (futureCancellation) {
                RequestProtocolTestSupport.awaitCondition(() -> org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle).getRequestState(990002L, 0L).state() == RequestState.Phase.CANCELLED);
                assertThrows(java.util.concurrent.CancellationException.class, future::join);
            } else {
                assertEquals(StrategyErrorType.REQUEST_CANCELLED.getErrorCode(), future.get(5, TimeUnit.SECONDS).getCode());
            }
            RequestProtocolTestSupport.awaitCondition(() -> RequestProtocolTestSupport.queuedCount(coordinator) == 0);
            verify(router, never()).select(any(), any());
        } finally {
            coordinator.close();
            if (RequestProtocolTestSupport.closeAdmissionAndAwaitMutations(lifecycle)) {
                lifecycle.closeOutstandingAndTerminalize();
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(lifecycle).timer().close();
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(lifecycle).closeRequestExecutors();
            }
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = { false, true })
    void controlTicketRemovesInFlightEntryBeforeLatePlanCanPublish(boolean deadline) throws Exception {
        FlexlbConfig config = SchedulingTestConfig.batchConfig();
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        RequestWorkerSelector router = mockRouter();
        AbstractRequestScheduler lifecycle = org.flexlb.balance.scheduler.SchedulerTestSupport.create(service, mock(DeliveryMetricsReporter.class), mock(RequestSchedulerReporter.class),
                mock(RecentCacheKeyTraceReporter.class));
        RequestContext context = context(config, 990001L);
        CountDownLatch planning = new CountDownLatch(1);
        CountDownLatch releasePlan = new CountDownLatch(1);
        CountDownLatch planClosed = new CountDownLatch(1);
        RequestRoute route = mock(RequestRoute.class);
        when(router.select(context, null)).thenAnswer(invocation -> {
            planning.countDown();
            assertTrue(releasePlan.await(5, TimeUnit.SECONDS));
            return PlacementResult.success(route);
        });
        doAnswer(invocation -> {
            planClosed.countDown();
            return null;
        }).when(route).close();
        QueuedRequestScheduler coordinator = RequestProtocolTestSupport.queue(service, router, mock(DeliveryMetricsReporter.class), mock(DecodeCapacityAcquirer.class), lifecycle, new PlacementAvailability());
        CompletableFuture<Response> future = RequestProtocolTestSupport.register(lifecycle, context);
        try {
            assertTrue(coordinator.trySubmitRegistered(context));
            assertTrue(planning.await(5, TimeUnit.SECONDS));
            if (deadline) {
                RequestContext requestContext = lifecycle.findRequestContext(990001L);
                var exact = (ExpirationTimer.RequestDeadline) org.springframework.test.util.ReflectionTestUtils.getField(requestContext, "requestDeadline");
                lifecycle.onSchedulingDeadline(requestContext, exact);
            } else {
                lifecycle.cancel(990001L, 0L, CancelReason.CLIENT_CANCELLED);
            }
            assertEquals(RequestState.Phase.CANCEL_REQUESTED, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(lifecycle).getRequestState(990001L, 0L).state());
            assertFalse(future.isDone(), "the admission claim still owns its route");
            RequestProtocolTestSupport.awaitCondition(() -> RequestProtocolTestSupport.queuedCount(coordinator) == 0);
            releasePlan.countDown();
            assertTrue(planClosed.await(5, TimeUnit.SECONDS));
            assertEquals(0, RequestProtocolTestSupport.queuedCount(coordinator));
            assertEquals((deadline ? StrategyErrorType.RESOURCE_EXHAUSTED : StrategyErrorType.REQUEST_CANCELLED).getErrorCode(), future.get(5, TimeUnit.SECONDS).getCode());
            RequestProtocolTestSupport.awaitCondition(() -> {
                var lock = (java.util.concurrent.locks.ReentrantLock) org.springframework.test.util.ReflectionTestUtils.getField(coordinator, "lock");
                lock.lock();
                try {
                    @SuppressWarnings("unchecked")
                    var registered = (java.util.Map<CompletableFuture<Response>, ?>) org.springframework.test.util.ReflectionTestUtils.getField(coordinator, "queuedEntries");
                    return !registered.containsKey(context);
                } finally {
                    lock.unlock();
                }
            });
            verify(route, never()).requestId();
        } finally {
            releasePlan.countDown();
            coordinator.close();
            if (RequestProtocolTestSupport.closeAdmissionAndAwaitMutations(lifecycle)) {
                lifecycle.closeOutstandingAndTerminalize();
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(lifecycle).timer().close();
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(lifecycle).closeRequestExecutors();
            }
        }
    }

    private static RequestWorkerSelector mockRouter() {
        RequestWorkerSelector router = mock(RequestWorkerSelector.class);
        return router;
    }

    @ParameterizedTest
    @NullSource
    @ValueSource(strings = { "healthy-group", "unavailable-group" })
    void unavailableRequestMustNotStopAHealthyFollowingRequest(String healthyGroup) {
        FlexlbConfig config = SchedulingTestConfig.batchConfig();
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        RequestWorkerSelector router = mockRouter();

        AbstractRequestScheduler lifecycle = RequestProtocolTestSupport.schedulerMock();
        RequestContext unavailable = RequestProtocolTestSupport.context(config, 910001L);
        RequestContext healthy = RequestProtocolTestSupport.context(config, 910002L);
        CompletableFuture<Response> first = new CompletableFuture<>();
        CompletableFuture<Response> second = new CompletableFuture<>();
        when(lifecycle.register(org.mockito.ArgumentMatchers.eq(unavailable), org.mockito.ArgumentMatchers.any())).thenAnswer(invocation -> { RequestContext registered = invocation.getArgument(0); registered.setFuture(first); return first; });
        when(lifecycle.register(org.mockito.ArgumentMatchers.eq(healthy), org.mockito.ArgumentMatchers.any())).thenAnswer(invocation -> { RequestContext registered = invocation.getArgument(0); registered.setFuture(second); return second; });
        when(lifecycle.claimAdmissionHandle(910001L, first)).thenReturn(mock(AdmissionHandle.class));
        when(lifecycle.claimAdmissionHandle(910002L, second)).thenReturn(mock(AdmissionHandle.class));
        when(router.resolvePolicyGroup(unavailable)).thenReturn("unavailable-group");
        when(router.resolvePolicyGroup(healthy)).thenReturn(healthyGroup);
        when(router.select(unavailable, "unavailable-group")).thenReturn(PlacementResult.blocked(new PlacementKey(RoleType.DECODE, "unavailable-group", null)));
        RequestRoute healthyRoute = mock(RequestRoute.class);
        when(router.select(healthy, healthyGroup)).thenReturn(PlacementResult.success(healthyRoute));
        org.mockito.Mockito.doReturn(PlacementResult.success(mock(RequestRoute.class))).when(RequestProtocolTestSupport.publication(lifecycle)).enqueueRoute(RequestProtocolTestSupport.matchingPlan(healthy, healthyRoute));
        RequestScheduler scheduler = RequestProtocolTestSupport.configure(lifecycle, service, router, mock(DeliveryMetricsReporter.class), mock(DecodeCapacityAcquirer.class), new PlacementAvailability());
        try {
            scheduler.submit(unavailable);
            verify(router, timeout(500)).select(unavailable, "unavailable-group");
            scheduler.submit(healthy);
            verify(router, timeout(500)).select(healthy, healthyGroup);
            verify(RequestProtocolTestSupport.publication(lifecycle), timeout(500)).enqueueRoute(RequestProtocolTestSupport.matchingPlan(healthy, healthyRoute));
        } finally {
            first.complete(new Response());
            second.complete(new Response());
            RequestProtocolTestSupport.close(scheduler);
        }
    }

    @ParameterizedTest
    @CsvSource({"false,success", "true,success", "false,rejected", "false,error",
            "false,adopt_failed", "false,placement_blocked", "false,closed", "false,cancelled", "false,closed_rollback_failure"})
    void priorityRescueConsumesTheOriginalExactRoute(boolean synchronous, String outcome) throws Exception {
        FlexlbConfig config = SchedulingTestConfig.batchConfig();
        SchedulingTestConfig.allowVictim(config, VictimStage.DECODE_RESERVED);
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        RequestWorkerSelector router = mockRouter();
        EndpointRegistry endpointRegistry = mock(EndpointRegistry.class);
        AbstractRequestScheduler lifecycle = RequestProtocolTestSupport.schedulerMock();
        DecodeCapacityAcquirer eviction = mock(DecodeCapacityAcquirer.class);
        PlacementAvailability availability = new PlacementAvailability();
        RequestContext context = context(config, 897L, 90);
        CompletableFuture<Response> future = new CompletableFuture<>();
        when(lifecycle.register(org.mockito.ArgumentMatchers.eq(context), org.mockito.ArgumentMatchers.any())).thenAnswer(invocation -> { RequestContext registered = invocation.getArgument(0); registered.setFuture(future); return future; });
        AdmissionHandle handle = mock(AdmissionHandle.class);
        when(lifecycle.claimAdmissionHandle(897L, future)).thenReturn(handle);
        DecodeEndpoint selectedEndpoint = RequestProtocolTestSupport.decodeEndpoint();
        when(selectedEndpoint.ipPort()).thenReturn("selected-decode:8080");
        RequestRoute selectedRoute = mock(RequestRoute.class);
        PlacementKey blocker = PlacementKey.exact(RoleType.DECODE, "g1", "selected-decode:8080");
        when(router.select(context, null)).thenReturn(PlacementResult.success(selectedRoute));
        RequestRoute submitted = mock(RequestRoute.class);
        var publicationThreads = new CopyOnWriteArrayList<Thread>();
        doAnswer(invocation -> {
            publicationThreads.add(Thread.currentThread());
            return publicationThreads.size() == 1 || outcome.equals("placement_blocked")
                    ? PlacementResult.blocked(blocker) : RequestProtocolTestSupport.published(invocation.getArgument(0), submitted);
        }).when(RequestProtocolTestSupport.publication(lifecycle)).enqueueRoute(RequestProtocolTestSupport.matchingPlan(context, selectedRoute));
        when(selectedRoute.blockedEndpointIfCurrent(any(PlacementKey.class))).thenReturn(selectedEndpoint);
        var binding = RequestRequirements.capture(context);
        org.springframework.test.util.ReflectionTestUtils.setField(context, "requirements", binding);
        when(selectedRoute.decodeEp()).thenReturn(selectedEndpoint);
        when(selectedRoute.requestId()).thenReturn(context.getRequestId());
        var reservation = new DecodeResources.ReservationHandle(1L, 897L, 7L);
        when(selectedEndpoint.markQueued(any(), org.mockito.ArgumentMatchers.eq(reservation)))
                .thenReturn(!outcome.equals("adopt_failed"));
        var rollbackFailure = new IllegalStateException("rollback failed");
        if (outcome.equals("closed_rollback_failure")) {
            org.mockito.Mockito.doThrow(rollbackFailure).when(selectedEndpoint)
                    .release(reservation, DecodeResources.ReleaseReason.LOCAL_ROLLBACK);
        }
        var result = new PreemptionResult(outcome.equals("rejected") ? null : reservation, false, "result");
        var execution = new CompletableFuture<PreemptionResult>();
        if (synchronous) { execution.complete(result); }
        when(eviction.tryReclaim(context, binding, selectedEndpoint)).thenReturn(execution);
        RequestScheduler scheduler = RequestProtocolTestSupport.configure(lifecycle, configService, router, mock(DeliveryMetricsReporter.class), eviction, availability);
        CompletableFuture<Void> closing = null;
        try {
            scheduler.submit(context);
            verify(eviction, timeout(1_000)).tryReclaim(context, binding, selectedEndpoint);
            if (outcome.equals("cancelled")) { assertTrue(future.cancel(false)); }
            if (outcome.startsWith("closed")) {
                closing = CompletableFuture.runAsync(() -> RequestProtocolTestSupport.close(scheduler));
                RequestProtocolTestSupport.awaitCondition(() -> ((java.util.concurrent.atomic.AtomicBoolean)
                        org.springframework.test.util.ReflectionTestUtils.getField(scheduler, "closed")).get());
                assertFalse(closing.isDone(), "close must drain the outstanding preemption result");
            }
            if (!synchronous) {
                verify(handle, never()).finish();
                verify(selectedRoute, never()).close();
                if (outcome.equals("error")) {
                    execution.completeExceptionally(new IllegalStateException("engine unavailable"));
                } else {
                    execution.complete(result);
                }
            }
            verify(handle, timeout(1_000)).finish();
            if (closing != null) { closing.join(); }
            if (outcome.equals("success") || outcome.startsWith("closed") || outcome.equals("cancelled")) {
                verify(handle, never()).terminate(any());
            } else {
                var rejection = org.mockito.ArgumentCaptor.forClass(Response.class);
                verify(handle).terminate(rejection.capture());
                assertEquals(StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(), rejection.getValue().getCode());
            }
            verify(lifecycle, times(1)).claimAdmissionHandle(897L, future);
            boolean hasReservation = !outcome.equals("rejected") && !outcome.equals("error") && !outcome.startsWith("closed") && !outcome.equals("cancelled");
            if (outcome.startsWith("closed") || outcome.equals("cancelled")) {
                verify(selectedEndpoint).release(reservation, DecodeResources.ReleaseReason.LOCAL_ROLLBACK);
            }
            Thread decision = (Thread) org.springframework.test.util.ReflectionTestUtils.getField(scheduler, "decisionThread");
            assertTrue(publicationThreads.stream().allMatch(thread -> thread == decision),
                    "both initial publication and preemption completion belong to the decision thread");
            verify(RequestProtocolTestSupport.publication(lifecycle), times(hasReservation && !outcome.equals("adopt_failed") ? 2 : 1)).enqueueRoute(RequestProtocolTestSupport.matchingPlan(context, selectedRoute));
            verify(selectedEndpoint, times(hasReservation ? 1 : 0))
                    .markQueued(any(), org.mockito.ArgumentMatchers.eq(reservation));
            if (outcome.equals("closed_rollback_failure")) { verify(lifecycle).recordFailure(rollbackFailure); }
            verify(selectedRoute).close();
            verify(router, times(1)).select(context, null);
        } finally {
            execution.complete(result);
            if (closing != null) { closing.join(); }
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
        RequestWorkerSelector router = mockRouter();
        EndpointRegistry endpointRegistry = mock(EndpointRegistry.class);
        when(endpointRegistry.getEndpointCount(RoleType.PREFILL)).thenReturn(1);
        AtomicInteger availableSlots = new AtomicInteger();
        AbstractRequestScheduler lifecycle = RequestProtocolTestSupport.schedulerMock();
        RequestContext context = context(config, 899L);
        CompletableFuture<Response> future = new CompletableFuture<>();
        when(lifecycle.register(org.mockito.ArgumentMatchers.eq(context), org.mockito.ArgumentMatchers.any())).thenAnswer(invocation -> { RequestContext registered = invocation.getArgument(0); registered.setFuture(future); return future; });
        when(lifecycle.claimAdmissionHandle(899L, future)).thenReturn(mock(AdmissionHandle.class));
        RequestRoute route = mock(RequestRoute.class);
        PrefillEndpoint endpoint = mockPrefillEndpoint("127.0.0.1", 8000);
        when(route.blockedEndpointIfCurrent(any(PlacementKey.class))).thenReturn(endpoint);
        org.mockito.Mockito.doReturn(PlacementResult.blocked(PlacementKey.exact(RoleType.PREFILL, "g1", "127.0.0.1:8000"))).when(RequestProtocolTestSupport.publication(lifecycle)).enqueueRoute(RequestProtocolTestSupport.matchingPlan(context, route));
        when(router.select(context, null)).thenReturn(PlacementResult.success(route), PlacementResult.rejected(Response.buildErrorResponse(StrategyErrorType.NO_PREFILL_WORKER, null)));
        PlacementAvailability availability = new PlacementAvailability();
        RequestScheduler scheduler = RequestProtocolTestSupport.configure(lifecycle, configService, router, mock(DeliveryMetricsReporter.class), mock(DecodeCapacityAcquirer.class), availability);
        try {
            scheduler.submit(context);
            verify(RequestProtocolTestSupport.publication(lifecycle), timeout(1_000)).enqueueRoute(RequestProtocolTestSupport.matchingPlan(context, route));
            verify(router, after(100).times(1)).select(context, null);
            availableSlots.set(1);
            availability.changed(PlacementKey.exact(RoleType.PREFILL, "g1", "127.0.0.1:8000"));
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
        RequestWorkerSelector router = mockRouter();
        EndpointRegistry endpointRegistry = mock(EndpointRegistry.class);
        AbstractRequestScheduler lifecycle = RequestProtocolTestSupport.schedulerMock();
        RequestContext context = context(config, 900L);
        CompletableFuture<Response> future = new CompletableFuture<>();
        when(lifecycle.register(org.mockito.ArgumentMatchers.eq(context), org.mockito.ArgumentMatchers.any())).thenAnswer(invocation -> { RequestContext registered = invocation.getArgument(0); registered.setFuture(future); return future; });
        when(lifecycle.claimAdmissionHandle(900L, future)).thenReturn(mock(AdmissionHandle.class));
        when(router.select(context, null)).thenReturn(PlacementResult.rejected(Response.buildErrorResponse(StrategyErrorType.NO_PREFILL_WORKER, null)));
        RequestScheduler scheduler = RequestProtocolTestSupport.configure(lifecycle, configService, router, mock(DeliveryMetricsReporter.class), mock(DecodeCapacityAcquirer.class), new PlacementAvailability());
        scheduler.submit(context);
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
        RequestWorkerSelector router = mockRouter();
        EndpointRegistry endpointRegistry = mock(EndpointRegistry.class);
        AbstractRequestScheduler lifecycle = RequestProtocolTestSupport.schedulerMock();
        DeliveryMetricsReporter reporter = mock(DeliveryMetricsReporter.class);
        PlacementAvailability availability = new PlacementAvailability();
        RequestContext gate = context(config, 900L);
        CompletableFuture<Response> gateFuture = new CompletableFuture<>();
        CountDownLatch gatePlanningStarted = new CountDownLatch(1);
        CountDownLatch releaseGatePlanning = new CountDownLatch(1);
        when(lifecycle.register(org.mockito.ArgumentMatchers.eq(gate), org.mockito.ArgumentMatchers.any())).thenAnswer(invocation -> { RequestContext registered = invocation.getArgument(0); registered.setFuture(gateFuture); return gateFuture; });
        when(lifecycle.claimAdmissionHandle(900L, gateFuture)).thenReturn(mock(AdmissionHandle.class));
        when(router.select(gate, null)).thenAnswer(invocation -> {
            gatePlanningStarted.countDown();
            releaseGatePlanning.await(5, TimeUnit.SECONDS);
            return PlacementResult.rejected(Response.buildErrorResponse(StrategyErrorType.NO_PREFILL_WORKER, null));
        });
        CountDownLatch aggregatePlansStarted = new CountDownLatch(6);
        List<RequestContext> contexts = new ArrayList<>();
        for (long requestId = 901L; requestId < 907L; requestId++) {
            RequestContext context = context(config, requestId);
            CompletableFuture<Response> future = new CompletableFuture<>();
            RequestRoute route = mock(RequestRoute.class);
            when(lifecycle.register(org.mockito.ArgumentMatchers.eq(context), org.mockito.ArgumentMatchers.any())).thenAnswer(invocation -> { RequestContext registered = invocation.getArgument(0); registered.setFuture(future); return future; });
            when(lifecycle.claimAdmissionHandle(requestId, future)).thenReturn(mock(AdmissionHandle.class));
            when(router.select(context, null)).thenAnswer(invocation -> {
                aggregatePlansStarted.countDown();
                return PlacementResult.success(route);
            });
            RequestRoute published = mock(RequestRoute.class);
            when(published.prefillEp()).thenReturn(mock(PrefillEndpoint.class));
            org.mockito.Mockito.doReturn(PlacementResult.success(published)).when(RequestProtocolTestSupport.publication(lifecycle)).enqueueRoute(RequestProtocolTestSupport.matchingPlan(context, route));
            contexts.add(context);
        }
        RequestScheduler scheduler = RequestProtocolTestSupport.configure(lifecycle, configService, router, reporter, mock(DecodeCapacityAcquirer.class), availability);
        try {
            scheduler.submit(gate);
            assertTrue(gatePlanningStarted.await(5, TimeUnit.SECONDS));
            for (RequestContext context : contexts) {
                scheduler.submit(context);
            }
            releaseGatePlanning.countDown();
            assertTrue(aggregatePlansStarted.await(5, TimeUnit.SECONDS), "all aggregate slots must be submitted from one captured frontier");
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
        RequestWorkerSelector router = mockRouter();
        AbstractRequestScheduler lifecycle = RequestProtocolTestSupport.schedulerMock();
        DecodeCapacityAcquirer decodeCapacity = mock(DecodeCapacityAcquirer.class);
        RequestContext lowPriority = context(config, 910L, 10);
        RequestContext highPriority = context(config, 911L, 90);
        CompletableFuture<Response> lowFuture = new CompletableFuture<>();
        CompletableFuture<Response> highFuture = new CompletableFuture<>();
        when(lifecycle.register(org.mockito.ArgumentMatchers.eq(lowPriority), org.mockito.ArgumentMatchers.any())).thenAnswer(invocation -> { RequestContext registered = invocation.getArgument(0); registered.setFuture(lowFuture); return lowFuture; });
        when(lifecycle.register(org.mockito.ArgumentMatchers.eq(highPriority), org.mockito.ArgumentMatchers.any())).thenAnswer(invocation -> { RequestContext registered = invocation.getArgument(0); registered.setFuture(highFuture); return highFuture; });
        when(lifecycle.claimAdmissionHandle(910L, lowFuture)).thenReturn(mock(AdmissionHandle.class));
        when(lifecycle.claimAdmissionHandle(911L, highFuture)).thenReturn(mock(AdmissionHandle.class));
        when(router.select(lowPriority, null)).thenReturn(PlacementResult.blocked(PlacementKey.anyGroup(RoleType.PREFILL)));
        when(router.select(highPriority, null)).thenReturn(PlacementResult.rejected(Response.buildErrorResponse(StrategyErrorType.NO_PREFILL_WORKER, null)));
        RequestScheduler scheduler = RequestProtocolTestSupport.configure(lifecycle, configService, router, mock(DeliveryMetricsReporter.class), decodeCapacity, new PlacementAvailability());
        try {
            scheduler.submit(lowPriority);
            verify(router, timeout(1_000)).select(lowPriority, null);
            scheduler.submit(highPriority);
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
        RequestWorkerSelector router = mockRouter();
        AbstractRequestScheduler lifecycle = RequestProtocolTestSupport.schedulerMock();
        PlacementAvailability availability = new PlacementAvailability();
        PlacementKey blocker = PlacementKey.anyGroup(RoleType.PREFILL);
        RequestContext expired = context(config, 801L);
        RequestContext follower = context(config, 802L);
        CompletableFuture<Response> expiredFuture = new CompletableFuture<>();
        CompletableFuture<Response> followerFuture = new CompletableFuture<>();
        when(lifecycle.register(org.mockito.ArgumentMatchers.eq(expired), org.mockito.ArgumentMatchers.any())).thenAnswer(invocation -> { RequestContext registered = invocation.getArgument(0); registered.setFuture(expiredFuture); return expiredFuture; });
        when(lifecycle.register(org.mockito.ArgumentMatchers.eq(follower), org.mockito.ArgumentMatchers.any())).thenAnswer(invocation -> { RequestContext registered = invocation.getArgument(0); registered.setFuture(followerFuture); return followerFuture; });
        when(lifecycle.claimAdmissionHandle(801L, expiredFuture)).thenReturn(mock(AdmissionHandle.class));
        when(lifecycle.claimAdmissionHandle(802L, followerFuture)).thenReturn(mock(AdmissionHandle.class));
        when(router.select(any(), any())).thenReturn(PlacementResult.blocked(blocker));
        RequestScheduler scheduler = RequestProtocolTestSupport.configure(lifecycle, configService, router, mock(DeliveryMetricsReporter.class), mock(DecodeCapacityAcquirer.class), availability);
        scheduler.submit(expired);
        scheduler.submit(follower);
        // Both independent requests may plan before the capacity event. Wait for
        // actual park registration so the event represents exactly one retry.
        RequestProtocolTestSupport.awaitGlobalCapacityWaiters(scheduler, 2);
        expired.setSchedulingMetadata(SchedulingMetadata.explicit(50, System.currentTimeMillis() - 1L));
        availability.changed(blocker);
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
        RequestWorkerSelector router = mockRouter();
        AbstractRequestScheduler lifecycle = RequestProtocolTestSupport.schedulerMock();
        PlacementAvailability availability = new PlacementAvailability();
        RequestContext blocked = context(config, 803L);
        RequestContext independent = context(config, 804L);
        CompletableFuture<Response> blockedFuture = new CompletableFuture<>();
        CompletableFuture<Response> independentFuture = new CompletableFuture<>();
        when(lifecycle.register(org.mockito.ArgumentMatchers.eq(blocked), org.mockito.ArgumentMatchers.any())).thenAnswer(invocation -> { RequestContext registered = invocation.getArgument(0); registered.setFuture(blockedFuture); return blockedFuture; });
        when(lifecycle.register(org.mockito.ArgumentMatchers.eq(independent), org.mockito.ArgumentMatchers.any())).thenAnswer(invocation -> { RequestContext registered = invocation.getArgument(0); registered.setFuture(independentFuture); return independentFuture; });
        when(lifecycle.claimAdmissionHandle(803L, blockedFuture)).thenReturn(mock(AdmissionHandle.class));
        when(lifecycle.claimAdmissionHandle(804L, independentFuture)).thenReturn(mock(AdmissionHandle.class));
        PrefillEndpoint fullEndpoint = mockPrefillEndpoint("full-prefill", 8080);
        PrefillEndpoint availableEndpoint = mock(PrefillEndpoint.class);
        RequestRoute blockedRoute = mock(RequestRoute.class);
        RequestRoute independentRoute = mock(RequestRoute.class);
        RequestRoute independentItem = mock(RequestRoute.class);
        PlacementKey exactBlocker = PlacementKey.exact(RoleType.PREFILL, "g1", "full-prefill:8080");
        CountDownLatch blockedRouteAttempts = new CountDownLatch(2);
        when(independentItem.prefillEp()).thenReturn(availableEndpoint);
        when(router.select(blocked, null)).thenAnswer(invocation -> {
            blockedRouteAttempts.countDown();
            return PlacementResult.success(blockedRoute);
        });
        when(router.select(independent, null)).thenReturn(PlacementResult.success(independentRoute));
        org.mockito.Mockito.doReturn(PlacementResult.blocked(exactBlocker)).when(RequestProtocolTestSupport.publication(lifecycle)).enqueueRoute(RequestProtocolTestSupport.matchingPlan(blocked, blockedRoute));
        when(blockedRoute.blockedEndpointIfCurrent(any(PlacementKey.class))).thenReturn(fullEndpoint);
        when(independentRoute.prefillEp()).thenReturn(availableEndpoint);
        org.mockito.Mockito.doReturn(PlacementResult.success(independentItem)).when(RequestProtocolTestSupport.publication(lifecycle)).enqueueRoute(RequestProtocolTestSupport.matchingPlan(independent, independentRoute));
        RequestScheduler scheduler = RequestProtocolTestSupport.configure(lifecycle, configService, router, mock(DeliveryMetricsReporter.class), mock(DecodeCapacityAcquirer.class), availability);
        try {
            scheduler.submit(blocked);
            scheduler.submit(independent);
            verify(RequestProtocolTestSupport.publication(lifecycle), timeout(1_000).times(1)).enqueueRoute(RequestProtocolTestSupport.matchingPlan(blocked, blockedRoute));
            verify(RequestProtocolTestSupport.publication(lifecycle), timeout(1_000)).enqueueRoute(RequestProtocolTestSupport.matchingPlan(independent, independentRoute));
            assertFalse(blockedFuture.isDone());
            availability.changed(new PlacementKey(RoleType.PREFILL, "g1", null));
            assertFalse(blockedRouteAttempts.await(100, TimeUnit.MILLISECONDS), "a group-wide edge must not release an exact endpoint blocker");
            availability.changed(exactBlocker);
            assertTrue(blockedRouteAttempts.await(1, TimeUnit.SECONDS));
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
                assertEquals(1, request.attempts.get(), "every request receives its own placement attempt");
            }
            fixture.releaseSlots(1);
            fixture.requests.get(0).future.get(5, TimeUnit.SECONDS);
            fixture.awaitIndependentCommit();
            assertEquals(List.of(820L), fixture.admitted);
            assertEquals(0, fixture.availableSlots.get());
            assertEquals(2, fixture.requests.get(0).attempts.get());
            assertEquals(2, fixture.requests.get(1).attempts.get());
            assertEquals(1, fixture.requests.get(2).attempts.get(), "a confirming failure stops this capacity round");
            fixture.awaitIndependentCommit();
            assertEquals(2, fixture.requests.get(1).attempts.get(), "unrelated ingress cannot retry failures");
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
    void capacityReleasedBeforeBlockedReturnRetriesActiveRequestWithoutLosingWakeup() throws Exception {
        try (CapacityFixture fixture = new CapacityFixture(0, false)) {
            fixture.submitBlockedRequests();
            CapacityRequest head = fixture.requests.get(0);
            head.beforeBlockedReturn = () -> fixture.releaseSlots(1);

            // Activate the head while the endpoint is still full. Its failed
            // admission publishes a release after reading capacity but before park.
            fixture.availability.changed(fixture.key);
            head.future.get(5, TimeUnit.SECONDS);
            assertTrue(fixture.requests.get(1).blockedAttempt.await(5, TimeUnit.SECONDS));
            fixture.awaitIndependentCommit();

            assertEquals(List.of(820L), fixture.admitted);
            assertEquals(0, fixture.availableSlots.get());
            assertEquals(3, head.attempts.get(),
                    "the stale full observation must trigger a fresh successful admission");
            assertEquals(2, fixture.requests.get(1).attempts.get());
            assertEquals(1, fixture.requests.get(2).attempts.get());
            fixture.awaitIndependentCommit();
            assertEquals(2, fixture.requests.get(1).attempts.get(),
                    "each failed request waits for another capacity edge");
        }
    }

    @Test
    void capacityWakeLetsEveryReadyRequestChooseAnotherWorker() throws Exception {
        try (CapacityFixture fixture = new CapacityFixture(0, false,
                new CompletableFuture<>(), 1)) {
            fixture.submitBlockedRequests();
            PrefillEndpoint other = mockPrefillEndpoint("other-prefill", 8080);
            for (CapacityRequest request : fixture.requests) {
                RequestRoute route = mock(RequestRoute.class);
                RequestRoute item = mock(RequestRoute.class);
                when(item.prefillEp()).thenReturn(other);
                ServerStatus status = new ServerStatus();
                status.setRole(RoleType.PREFILL);
                when(item.prefill()).thenReturn(status);
                request.alternative = new AlternativeRoute(route, item);
            }
            fixture.availability.changed(fixture.key);
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
            assertTrue(fixture.requests.get(2).blockedAttempt.await(5, TimeUnit.SECONDS));
            fixture.awaitIndependentCommit();

            assertEquals(List.of(821L), fixture.admitted);
            assertEquals(0, fixture.availableSlots.get());
            assertEquals(1, fixture.requests.get(0).attempts.get(),
                    "the cancelled active request must not consume the released slot");
            assertEquals(2, fixture.requests.get(2).attempts.get());
        }
    }

    @Test
    void completedActiveRetryHandsOffWithoutItsDelayedCompletionCallback() throws Exception {
        ConcurrentLinkedQueue<Runnable> callbacks = new ConcurrentLinkedQueue<>();
        CompletableFuture<Response> headFuture = new CompletableFuture<>() {

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
            first.beforePublication = () -> {
                publicationStarted.countDown();
                assertTrue(allowPublication.await(5, TimeUnit.SECONDS));
                return null;
            };
            try {
                fixture.scheduler.submit(first.context);
                assertTrue(publicationStarted.await(5, TimeUnit.SECONDS));
                fixture.releaseSlots(1);
                assertTrue(headFuture.complete(new Response()));
                assertEquals(1, callbacks.size(), "the completed active request's cleanup must still be deferred");
                CapacityRequest second = fixture.createRequest(841L, independent);
                fixture.scheduler.submit(second.context);
                allowPublication.countDown();
                second.future.get(5, TimeUnit.SECONDS);
                // Ready waiters progress without another event or completion callback.
                fixture.requests.get(1).future.get(5, TimeUnit.SECONDS);
                assertEquals(1, callbacks.size(), "handoff must not rely on running completion callbacks");
                Runnable callback;
                while ((callback = callbacks.poll()) != null) {
                    callback.run();
                }
                // Even a subsequent edge must not be swallowed by an active request
                // that was removed from the ordered queue during pruning.
                fixture.availability.changed(fixture.key);
                fixture.requests.get(1).future.get(5, TimeUnit.SECONDS);
                assertEquals(List.of(821L), fixture.admitted);
                assertEquals(0, fixture.availableSlots.get());
                assertEquals(1, fixture.requests.get(0).attempts.get());
            } finally {
                allowPublication.countDown();
            }
        }
    }

    private void assertReleasedCapacityAdmitsWaiters(
            int initiallyReleasedSlots,
            int slotsReleasedDuringPublication) throws Exception {
        try (CapacityFixture fixture = new CapacityFixture(slotsReleasedDuringPublication, false)) {
            fixture.submitBlockedRequests();
            fixture.releaseSlots(initiallyReleasedSlots);
            fixture.awaitAllPublished();

            assertEquals(List.of(820L, 821L, 822L), fixture.admitted);
            assertEquals(0, fixture.availableSlots.get());
        }
    }

    @Test
    void staleEndpointConflictReplansWithoutFixedRetryBudget() {
        FlexlbConfig config = SchedulingTestConfig.batchConfig();
        SchedulingTestConfig.useFifoQueue(config);
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        RequestWorkerSelector router = mockRouter();
        AbstractRequestScheduler lifecycle = RequestProtocolTestSupport.schedulerMock();
        PlacementAvailability availability = new PlacementAvailability();
        RequestContext context = context(config, 807L);
        CompletableFuture<Response> future = new CompletableFuture<>();
        when(lifecycle.register(org.mockito.ArgumentMatchers.eq(context), org.mockito.ArgumentMatchers.any())).thenAnswer(invocation -> { RequestContext registered = invocation.getArgument(0); registered.setFuture(future); return future; });
        when(lifecycle.claimAdmissionHandle(807L, future)).thenReturn(mock(AdmissionHandle.class), mock(AdmissionHandle.class));
        PrefillEndpoint staleEndpoint = mockPrefillEndpoint("stale-prefill", 8080);
        RequestRoute staleRoute = mock(RequestRoute.class);
        RequestRoute freshRoute = mock(RequestRoute.class);
        RequestRoute committed = mock(RequestRoute.class);
        when(committed.prefillEp()).thenReturn(mock(PrefillEndpoint.class));
        PlacementKey exactBlocker = PlacementKey.exact(RoleType.PREFILL, "g1", "stale-prefill:8080");
        when(router.select(context, null)).thenReturn(PlacementResult.success(staleRoute), PlacementResult.success(freshRoute));
        org.mockito.Mockito.doReturn(PlacementResult.blocked(exactBlocker)).when(RequestProtocolTestSupport.publication(lifecycle)).enqueueRoute(RequestProtocolTestSupport.matchingPlan(context, staleRoute));
        when(staleRoute.blockedEndpointIfCurrent(exactBlocker)).thenReturn(null);
        org.mockito.Mockito.doReturn(PlacementResult.success(committed)).when(RequestProtocolTestSupport.publication(lifecycle)).enqueueRoute(RequestProtocolTestSupport.matchingPlan(context, freshRoute));
        RequestScheduler scheduler = RequestProtocolTestSupport.configure(lifecycle, configService, router, mock(DeliveryMetricsReporter.class), mock(DecodeCapacityAcquirer.class), availability);
        try {
            scheduler.submit(context);
            verify(RequestProtocolTestSupport.publication(lifecycle), timeout(1_000)).enqueueRoute(RequestProtocolTestSupport.matchingPlan(context, freshRoute));
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
                    .thenThrow(new IllegalStateException("configuration unavailable after initialization"));
            CompletableFuture<Response> future = fixture.scheduler.submit(fixture.context);

            assertSame(fixture.future, future);
            assertEquals(StrategyErrorType.NO_PREFILL_WORKER.getErrorCode(),
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

        Response response = fixture.scheduler.submit(fixture.context).orTimeout(3, TimeUnit.SECONDS).join();

        assertEquals(StrategyErrorType.NO_PREFILL_WORKER.getErrorCode(),
                response.getCode());
        verify(fixture.router, timeout(1_000))
                .select(fixture.context, null);
        verify(fixture.lifecycle, never())
                .commitRoute(
                        any(), any());
        RequestProtocolTestSupport.close(fixture.scheduler);
    }

    @Test
    void prioritySelectorMissWaitsWithoutInventingAnEvictionRoute() {
        Fixture fixture = new Fixture(true);
        when(fixture.router.select(fixture.context, null)).thenReturn(
                PlacementResult.blocked(
                        PlacementKey.anyGroup(RoleType.PREFILL)));
        CompletableFuture<Response> waiting =
                fixture.scheduler.submit(fixture.context);

        verify(fixture.router, timeout(1_000))
                .select(fixture.context, null);
        verify(fixture.decodeCapacity, never())
                .tryReclaim(any(), any(), any());
        assertFalse(waiting.isDone());
        verify(fixture.lifecycle, never())
                .commitRoute(
                        any(), any());
        RequestProtocolTestSupport.close(fixture.scheduler);
    }

    @Test
    void fifoTemporaryDecodeMissRemainsQueued() {
        Fixture fixture = new Fixture(false);
        when(fixture.router.select(fixture.context, null)).thenReturn(
                PlacementResult.blocked(
                        PlacementKey.anyGroup(RoleType.DECODE)));
        CompletableFuture<Response> waiting =
                fixture.scheduler.submit(fixture.context);

        verify(fixture.router, timeout(1_000))
                .select(fixture.context, null);
        assertFalse(waiting.isDone());
        verify(fixture.decodeCapacity, never())
                .tryReclaim(any(), any(), any());
        verify(fixture.lifecycle, never())
                .commitRoute(
                        any(), any());
        RequestProtocolTestSupport.close(fixture.scheduler);
    }

    @Test
    void initialPlacementFailureCompletesTheRegisteredGeneration() {
        Fixture fixture = new Fixture(false);
        when(fixture.router.select(fixture.context, null))
                .thenThrow(new IllegalStateException("selector failed"));

        CompletableFuture<Response> returned =
                fixture.scheduler.submit(fixture.context);

        assertEquals(fixture.future, returned);
        assertEquals(StrategyErrorType.DISPATCH_FAILED.getErrorCode(),
                returned.join().getCode());
        RequestProtocolTestSupport.close(fixture.scheduler);
    }

    private static PrefillEndpoint mockPrefillEndpoint(String ip, int port) {
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        WorkerStatus status = WorkerStatus.createDiscovered(
                RoleType.PREFILL, "g1", ip, port, port, "test");
        when(endpoint.getStatus()).thenReturn(status);
        when(endpoint.ipPort()).thenReturn(status.getIpPort());
        when(endpoint.getIp()).thenReturn(ip);
        return endpoint;
    }

    private static RequestContext context(FlexlbConfig config, long requestId) {
        return context(config, requestId, 50);
    }

    private static RequestContext context(FlexlbConfig config, long requestId, int priority) {
        RequestContext context = new RequestContext(config);
        Request request = new Request();
        request.setRequestId(requestId);
        request.setPriority(priority);
        context.setRequest(request);
        context.setGenerateInputPb(com.google.protobuf.ByteString.copyFromUtf8("test-input"));
        context.setSchedulingMetadata(SchedulingMetadata.explicit(
                priority, System.currentTimeMillis() + 60_000L));
        return context;
    }

    /**
     * Real ordered scheduler with an exact endpoint whose free slots are explicitly controlled.
     */
    private static final class CapacityFixture implements AutoCloseable {

        private final ConcurrentMap<Long, CapacityRequest> requestsById = new ConcurrentHashMap<>();

        private final RequestWorkerSelector router = mockRouter();

        private final AbstractRequestScheduler lifecycle = RequestProtocolTestSupport.schedulerMock();

        private final PlacementAvailability availability = new PlacementAvailability();

        private final PrefillEndpoint endpoint = mockPrefillEndpoint("capacity-prefill", 8080);

        private final PlacementKey key = PlacementKey.exact(
                RoleType.PREFILL, "g1", "capacity-prefill:8080");

        private final AtomicInteger availableSlots = new AtomicInteger();

        private final List<Long> admitted = new CopyOnWriteArrayList<>();

        private final List<CapacityRequest> requests = new ArrayList<>();

        private final ConcurrentLinkedQueue<CapacityRequest> pendingReports = new ConcurrentLinkedQueue<>();

        private final CountDownLatch allPublished = new CountDownLatch(3);

        private final CountDownLatch headRetryStarted = new CountDownLatch(1);

        private final CountDownLatch allowHeadRetry = new CountDownLatch(1);

        private final int slotsReleasedDuringPublication;

        private final boolean pauseHeadRetry;

        private final CompletableFuture<Response> headFuture;

        private final RequestScheduler scheduler;

        private final FlexlbConfig config;

        private long nextIndependentId = 830L;

        private CapacityFixture(int slotsReleasedDuringPublication, boolean pauseHeadRetry) {
            this(slotsReleasedDuringPublication, pauseHeadRetry, new CompletableFuture<>(), 0);
        }

        private CapacityFixture(int slotsReleasedDuringPublication, boolean pauseHeadRetry, CompletableFuture<Response> headFuture, int plannerThreads) {
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
            when(lifecycle.register(any(RequestContext.class), org.mockito.ArgumentMatchers.any())).thenAnswer(invocation -> {
                RequestContext context = invocation.getArgument(0);
                var future = requestsById.get(context.getRequestId()).future;
                context.setFuture(future);
                return future;
            });
            when(lifecycle.claimAdmissionHandle(anyLong(), any())).thenReturn(mock(AdmissionHandle.class));
            when(router.select(any(RequestContext.class), isNull())).thenAnswer(invocation -> {
                RequestContext context = invocation.getArgument(0);
                CapacityRequest request = requestsById.get(context.getRequestId());
                if (request.plans.incrementAndGet() == 2 && context.getRequestId() == 820L && pauseHeadRetry) {
                    headRetryStarted.countDown();
                    assertTrue(allowHeadRetry.await(5, TimeUnit.SECONDS), "the cancelled active request's planning gate must be released");
                }
                AlternativeRoute alternative = request.alternative;
                return PlacementResult.success(alternative == null ? request.route : alternative.route());
            });
            org.mockito.Mockito.doAnswer(invocation -> {
                QueuedRequestScheduler.Plan plan = invocation.getArgument(0);
                RequestContext context = RequestProtocolTestSupport.planContext(plan);
                long requestId = context.getRequestId();
                CapacityRequest request = requestsById.get(requestId);
                AlternativeRoute alternative = request.alternative;
                if (alternative != null && RequestProtocolTestSupport.planRouting(plan) == alternative.route()) {
                    admitted.add(requestId);
                    pendingReports.add(request);
                    return RequestProtocolTestSupport.published(plan, alternative.item());
                }
                assertSame(request.route, RequestProtocolTestSupport.planRouting(plan));
                request.attempts.incrementAndGet();
                request.beforePublication.call();
                if (request.primaryEndpoint) {
                    int slots = availableSlots.getAndUpdate(value -> value > 0 ? value - 1 : value);
                    if (slots == 0) {
                        request.blockedAttempt.countDown();
                        Runnable hook = request.beforeBlockedReturn;
                        request.beforeBlockedReturn = null;
                        if (hook != null) {
                            hook.run();
                        }
                        return PlacementResult.blocked(key);
                    }
                    admitted.add(requestId);
                    if (requestId == 820L && slotsReleasedDuringPublication > 0) {
                        // Status reconciliation releases capacity before publication returns
                        // and before the global queue can retire this active request.
                        releaseSlots(slotsReleasedDuringPublication);
                    }
                }
                pendingReports.add(request);
                return RequestProtocolTestSupport.published(plan, request.item);
            }).when(RequestProtocolTestSupport.publication(lifecycle)).enqueueRoute(any(QueuedRequestScheduler.Plan.class));
            DeliveryMetricsReporter reporter = mock(DeliveryMetricsReporter.class);
            doAnswer(invocation -> {
                CapacityRequest request = pendingReports.remove();
                // The coordinator reports only after retiring the queue entry. Completing
                // here cannot hide a lost handoff with an earlier completion callback.
                request.future.complete(new Response());
                if (request.primaryEndpoint) {
                    allPublished.countDown();
                }
                return null;
            }).when(reporter).reportLatency(org.mockito.ArgumentMatchers.eq(DeliveryMetricsReporter.Latency.ROUTE_SUBMIT), any(), any(), anyLong());
            for (long requestId = 820L; requestId < 823L; requestId++) {
                requests.add(createRequest(requestId, endpoint));
            }
            scheduler = RequestProtocolTestSupport.configure(lifecycle, configService, router, reporter, mock(DecodeCapacityAcquirer.class), availability);
        }

        private CapacityRequest createRequest(long requestId, PrefillEndpoint target) {
            CapacityRequest request = new CapacityRequest(config, requestId, target == endpoint,
                    requestId == 820L ? headFuture : new CompletableFuture<>());
            RequestRoute route = request.route;
            RequestRoute item = request.item;
            ServerStatus prefill = mock(ServerStatus.class);
            when(prefill.getRole()).thenReturn(RoleType.PREFILL);
            when(item.prefill()).thenReturn(prefill);
            when(item.prefillEp()).thenReturn(target);
            when(route.prefillEp()).thenReturn(target);
            when(route.blockedEndpointIfCurrent(any(PlacementKey.class))).thenReturn(target);
            requestsById.put(requestId, request);
            return request;
        }

        private void submitBlockedRequests() throws InterruptedException {
            for (CapacityRequest request : requests) {
                scheduler.submit(request.context);
            }
            RequestProtocolTestSupport.awaitGlobalCapacityWaiters(scheduler, requests.size());
        }

        private void releaseSlots(int count) {
            availableSlots.addAndGet(count);
            availability.changed(key);
        }

        private void awaitAllPublished() throws InterruptedException {
            assertTrue(allPublished.await(5, TimeUnit.SECONDS),
                    () -> "released slots must be consumed without another event; admitted=" + admitted
                            + ", freeSlots=" + availableSlots.get());
        }

        private void awaitIndependentCommit() throws Exception {
            long requestId = nextIndependentId++;
            PrefillEndpoint independent = mockPrefillEndpoint("independent-" + requestId, 8080);
            CapacityRequest request = createRequest(requestId, independent);
            scheduler.submit(request.context);
            request.future.get(5, TimeUnit.SECONDS);
        }

        @Override
        public void close() {
            allowHeadRetry.countDown();
            RequestProtocolTestSupport.close(scheduler);
        }
    }

    private record AlternativeRoute(RequestRoute route, RequestRoute item) { }

    private static final class CapacityRequest {

        private final AtomicInteger plans = new AtomicInteger();

        private final RequestContext context;

        private final CompletableFuture<Response> future;

        private final AtomicInteger attempts = new AtomicInteger();

        private final RequestRoute route = mock(RequestRoute.class);

        private final RequestRoute item = mock(RequestRoute.class);

        private final CountDownLatch blockedAttempt = new CountDownLatch(1);

        private final boolean primaryEndpoint;

        private Callable<Void> beforePublication = () -> null;

        private Runnable beforeBlockedReturn;

        private volatile AlternativeRoute alternative;

        private CapacityRequest(FlexlbConfig config, long requestId, boolean primaryEndpoint, CompletableFuture<Response> future) {
            context = context(config, requestId);
            this.primaryEndpoint = primaryEndpoint;
            this.future = future;
        }
    }

    private static final class Fixture {

        private final long requestId = 701L;

        private final FlexlbConfig config = SchedulingTestConfig.batchConfig();

        private final ConfigService configService = mock(ConfigService.class);

        private final RequestWorkerSelector router = mockRouter();

        private final DecodeCapacityAcquirer decodeCapacity = mock(DecodeCapacityAcquirer.class);

        private final AbstractRequestScheduler lifecycle = RequestProtocolTestSupport.schedulerMock();

        private final PlacementAvailability availability = new PlacementAvailability();

        private final RequestContext context = mock(RequestContext.class);

        private final CompletableFuture<Response> future =
                new CompletableFuture<>();

        private final RequestScheduler scheduler;

        private Fixture(boolean priority) {
            if (priority) {
                SchedulingTestConfig.usePriorityQueue(config);
            } else {
                SchedulingTestConfig.useFifoQueue(config);
            }
            when(configService.loadBalanceConfig()).thenReturn(config);
            when(context.getRequest()).thenReturn(new Request());
            when(context.isOpen()).thenReturn(true);
            when(context.hasGenerateInput()).thenReturn(true);
            when(context.getConfig()).thenReturn(config);
            when(context.getRequestId()).thenReturn(requestId);
            when(context.getFuture()).thenReturn(future);
            when(lifecycle.register(org.mockito.ArgumentMatchers.eq(context), org.mockito.ArgumentMatchers.any())).thenAnswer(invocation -> { RequestContext registered = invocation.getArgument(0); registered.setFuture(future); return future; });
            when(lifecycle.claimAdmissionHandle(requestId, future)).thenReturn(mock(AdmissionHandle.class));
            when(lifecycle.publishDecisionResponseAsync(anyLong(), any(), any())).thenAnswer(invocation -> {
                @SuppressWarnings("unchecked")
                CompletableFuture<Response> responseFuture = (CompletableFuture<Response>) invocation.getArgument(1);
                responseFuture.complete(invocation.getArgument(2));
                return true;
            });
            when(router.select(context, null)).thenReturn(PlacementResult.rejected(Response.buildErrorResponse(StrategyErrorType.NO_PREFILL_WORKER, null)));
            scheduler = RequestProtocolTestSupport.configure(lifecycle, configService, router, mock(DeliveryMetricsReporter.class), decodeCapacity, availability);
        }
    }
}
