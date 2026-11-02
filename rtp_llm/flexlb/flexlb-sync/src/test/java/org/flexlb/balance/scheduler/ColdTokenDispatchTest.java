package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.eviction.DecodeCapacityAcquirer;
import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.balance.strategy.DecodeSelector;
import org.flexlb.balance.strategy.VitWorkerSelector;
import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.master.WorkerStatusTestSupport;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.RecentCacheKeyTraceReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.Test;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.flexlb.balance.scheduler.RequestProtocolTestSupport.awaitCondition;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.nullable;
import static org.mockito.Mockito.*;

class ColdTokenDispatchTest {
    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
    void coldRequestDoesNotWaitForWorkerCollectionWindow(boolean batch) throws Exception {
        try (var f = new Fixture(batch)) {
            var request = f.context(1);
            var response = f.scheduler.submit(request).get(2, TimeUnit.SECONDS);
            assertTrue(response.isSuccess(), response.getErrorMessage());
            assertEquals(batch, response.isEnqueuedByMaster());
            assertEquals(0, f.a.ownership.progress().queueDepth());
            assertEquals(0, f.b.ownership.progress().queueDepth());
            assertEquals(1000, f.a.ownership.tokenWorkload().pendingTokens() + f.b.ownership.tokenWorkload().pendingTokens());
            assertEquals(0, f.a.ownership.tokenWorkload().plannedTokens() + f.b.ownership.tokenWorkload().plannedTokens());
        }
    }

    @Test
    void capacityLostAfterSelectionRetriesAnotherWorkerWithoutBindingToTheOldQueue() throws Exception {
        try (var f = new Fixture()) {
            var selected = new CountDownLatch(1);
            var proceed = new CountDownLatch(1);
            var first = new AtomicBoolean(true);
            var selectedWorker = new java.util.concurrent.atomic.AtomicReference<PrefillEndpoint>();
            doAnswer(call -> {
                var result = (org.flexlb.balance.PlacementResult<RequestRoute, PlacementKey>) call.callRealMethod();
                if (result.value() != null && first.getAndSet(false)) {
                    selectedWorker.set(result.value().prefillEp());
                    selected.countDown();
                    assertTrue(proceed.await(3, TimeUnit.SECONDS));
                }
                return result;
            }).when(f.router).select(any(), nullable(String.class));
            var context = f.context(2);
            var response = f.scheduler.submit(context);
            assertTrue(selected.await(2, TimeUnit.SECONDS));
            var oldWorker = selectedWorker.get();
            var reservation = f.occupy(oldWorker, 99);
            proceed.countDown();
            try {
                assertTrue(response.get(2, TimeUnit.SECONDS).isSuccess());
                assertNotSame(oldWorker, context.snapshot().route().prefillEp());
                assertEquals(0, oldWorker.ownership.progress().queueDepth());
                assertEquals(0, oldWorker.ownership.tokenWorkload().plannedTokens());
            } finally { oldWorker.release(reservation, null); }
        }
    }

    @Test
    void waitingColdRequestHasNoWorkerAssignmentAndUsesTheWorkerThatBecomesFree() throws Exception {
        try (var f = new Fixture()) {
            var a = f.occupy(f.a, 90);
            var b = f.occupy(f.b, 91);
            var context = f.context(3);
            var response = f.scheduler.submit(context);
            try {
                awaitCondition(() -> !f.scheduler.getLatestQueueWaitSnapshot().isEmpty()
                        && f.scheduler.getLatestQueueWaitSnapshot().containsKey("role"));
                assertFalse(response.isDone());
                assertNull(context.snapshot().route());
                f.b.release(b, null);
                assertTrue(response.get(2, TimeUnit.SECONDS).isSuccess());
                assertSame(f.b, context.snapshot().route().prefillEp());
                assertEquals(0, f.a.ownership.progress().queueDepth());
                assertEquals(0, f.b.ownership.progress().queueDepth());
            } finally {
                f.a.release(a, null);
                f.b.release(b, null);
            }
        }
    }

    @Test
    void cancellationWhileWaitingLeavesNoTokenOrWorkerOwnership() throws Exception {
        try (var f = new Fixture()) {
            var a = f.occupy(f.a, 90);
            var b = f.occupy(f.b, 91);
            var context = f.context(4);
            var response = f.scheduler.submit(context);
            try {
                awaitCondition(() -> f.scheduler.getLatestQueueWaitSnapshot().containsKey("role"));
                assertTrue(response.cancel(false));
                f.a.release(a, null);
                f.b.release(b, null);
                awaitCondition(() -> f.scheduler.requests.findActive(4) == null);
                assertEquals(0, f.a.ownership.tokenWorkload().score());
                assertEquals(0, f.b.ownership.tokenWorkload().score());
            } finally {
                f.a.release(a, null);
                f.b.release(b, null);
            }
        }
    }

    @Test
    void batchCapacityReleaseWakesGlobalWaiterWithoutAWorkerQueue() throws Exception {
        try (var f = new Fixture(true)) {
            var one = f.context(10);
            var two = f.context(11);
            assertTrue(f.scheduler.submit(one).get(2, TimeUnit.SECONDS).isSuccess());
            assertTrue(f.scheduler.submit(two).get(2, TimeUnit.SECONDS).isSuccess());
            assertNotSame(one.snapshot().route().prefillEp(), two.snapshot().route().prefillEp());
            var three = f.context(12);
            var future = f.scheduler.submit(three);
            awaitCondition(() -> f.scheduler.getLatestQueueWaitSnapshot().containsKey("role"));
            assertFalse(future.isDone());
            assertNull(three.snapshot().route());
            assertEquals(0, f.a.ownership.progress().queueDepth() + f.b.ownership.progress().queueDepth());
            var released = one.snapshot().route().prefillEp();
            f.scheduler.cancel(10, one.snapshot().delivery().identity().batchId(), CancelReason.CLIENT_CANCELLED);
            assertTrue(future.get(2, TimeUnit.SECONDS).isSuccess());
            assertSame(released, three.snapshot().route().prefillEp());
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
    void decodeWaitAndAsyncPreemptionRebalancePrefillAtDispatch(boolean preempt) throws Exception {
        try (var f = new Fixture(false, true, preempt)) {
            org.flexlb.balance.endpoint.DecodeResources.ReservationHandle occupied;
            try (var pin = f.decode.tryPinGeneration()) {
                occupied = f.decode.tryReserveQueuedRequest(pin, 99, 1000, 1001, 10, null).handle();
            }
            var firstWorker = new java.util.concurrent.atomic.AtomicReference<PrefillEndpoint>();
            doAnswer(call -> {
                var result = (org.flexlb.balance.PlacementResult<RequestRoute, PlacementKey>) call.callRealMethod();
                if (result.value() != null) { firstWorker.compareAndSet(null, result.value().prefillEp()); }
                return result;
            }).when(f.router).select(any(), nullable(String.class));
            var reclaim = new java.util.concurrent.CompletableFuture<DecodeCapacityAcquirer.PreemptionResult>();
            var reclaimStarted = new CountDownLatch(1);
            when(f.eviction.tryReclaim(any(), any())).thenAnswer(call -> { reclaimStarted.countDown(); return reclaim; });
            var context = f.context(20);
            var future = f.scheduler.submit(context);
            if (preempt) { assertTrue(reclaimStarted.await(2, TimeUnit.SECONDS)); }
            else { awaitCondition(() -> f.scheduler.getLatestQueueWaitSnapshot().containsKey("role")); }
            assertFalse(future.isDone());
            assertNull(context.snapshot().route());
            assertEquals(0, f.a.ownership.tokenWorkload().pendingTokens() + f.b.ownership.tokenWorkload().pendingTokens());
            var old = firstWorker.get();
            var load = f.occupy(old, 98);
            try {
                f.decode.release(occupied, org.flexlb.balance.endpoint.DecodeResources.ReleaseReason.LOCAL_ROLLBACK).status();
                if (preempt) {
                    org.flexlb.balance.endpoint.DecodeResources.ReservationHandle incoming;
                    try (var pin = f.decode.tryPinGeneration()) {
                        incoming = f.decode.tryReserveQueuedRequest(pin, 20, 1000, 1001, 50, null).handle();
                    }
                    reclaim.complete(new DecodeCapacityAcquirer.PreemptionResult(incoming, false, "committed"));
                }
                assertTrue(future.get(2, TimeUnit.SECONDS).isSuccess());
                assertNotSame(old, context.snapshot().route().prefillEp());
                assertEquals(0, f.a.ownership.progress().queueDepth() + f.b.ownership.progress().queueDepth());
                assertEquals(0, f.a.ownership.tokenWorkload().plannedTokens() + f.b.ownership.tokenWorkload().plannedTokens());
                assertEquals(1, f.decode.routingView().engineCapacityUsed());
            } finally { old.release(load, null); }
        }
    }

    @Test
    void batchExecutorCapacityWaitDoesNotBindWorkerAndReleaseWakesIt() throws Exception {
        try (var f = new Fixture(true)) {
            var permits = new java.util.ArrayList<DeliveryExecution.DeliveryPermit>();
            try {
                while (true) {
                    var attempt = f.runtime.deliveries().tryReserveDelivery();
                    if (!attempt.accepted()) { break; }
                    permits.add(attempt.value());
                    assertTrue(permits.size() < 1024);
                }
                var request = f.context(30);
                var future = f.scheduler.submit(request);
                awaitCondition(() -> f.scheduler.getLatestQueueWaitSnapshot().containsKey("role"));
                assertFalse(future.isDone());
                assertNull(request.snapshot().route());
                assertEquals(0, f.a.ownership.tokenWorkload().pendingTokens() + f.b.ownership.tokenWorkload().pendingTokens());
                permits.removeLast().close();
                assertTrue(future.get(2, TimeUnit.SECONDS).isSuccess());
            } finally { permits.forEach(DeliveryExecution.DeliveryPermit::close); }
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
    void cancellationAfterSelectionRollsBackImmediateAdmission(boolean batch) throws Exception {
        try (var f = new Fixture(batch)) {
            var selected = new CountDownLatch(1);
            var proceed = new CountDownLatch(1);
            doAnswer(call -> {
                var result = call.callRealMethod();
                selected.countDown();
                assertTrue(proceed.await(3, TimeUnit.SECONDS));
                return result;
            }).when(f.router).select(any(), nullable(String.class));
            var context = f.context(31);
            var future = f.scheduler.submit(context);
            assertTrue(selected.await(2, TimeUnit.SECONDS));
            try { assertTrue(future.cancel(false)); }
            finally { proceed.countDown(); }
            awaitCondition(() -> f.scheduler.requests.findActive(31) == null);
            awaitCondition(() -> f.a.ownership.tokenWorkload().score() + f.b.ownership.tokenWorkload().score() == 0);
            assertEquals(0, f.a.ownership.stats().batchCount() + f.b.ownership.stats().batchCount());
        }
    }

    private static final class Fixture implements AutoCloseable {
        final FlexlbConfig config = SchedulingTestConfig.newConfig();
        final EndpointRegistry endpoints;
        final PrefillEndpoint a;
        final PrefillEndpoint b;
        final org.flexlb.balance.endpoint.DecodeEndpoint decode;
        final DecodeCapacityAcquirer eviction = mock(DecodeCapacityAcquirer.class);
        final RequestWorkerSelector router;
        final QueuedRequestScheduler scheduler;
        final SchedulerRuntime runtime;
        final CacheAwareService cache = mock(CacheAwareService.class);

        Fixture() { this(false); }

        Fixture(boolean batch) { this(batch, false, false); }

        Fixture(boolean batch, boolean separateDecode, boolean priority) {
            SchedulingTestConfig.useFifoQueue(config);
            if (priority) { SchedulingTestConfig.usePriorityQueue(config); }
            config.getRouter().getRoles().getDecode().getAvailability().setMaxEngineRequests(1L);
            SchedulingTestConfig.useNonBatchDispatcher(config);
            config.getDispatcher().setMaxInflightPerPrefillWorker(1);
            if (batch) { config.getDispatcher().setType(org.flexlb.config.DispatcherConfig.Type.BATCH); }
            config.queueScheduler().getDecision().setMaxRequests(8);
            config.queueScheduler().getDecision().setMaxCollectionWaitMs(60_000L);
            config.getRouter().getRoles().getPrefill().setTokenBalance(new RoutingConfig.TokenBalanceConfig());
            var service = SchedulingTestConfig.configService(config);
            var reporter = mock(DeliveryMetricsReporter.class);
            var grpc = mock(org.flexlb.engine.grpc.EngineGrpcClient.class);
            when(grpc.batchEnqueueAsync(anyString(), anyInt(), any())).thenAnswer(call ->
                    java.util.concurrent.CompletableFuture.completedFuture(LinearDeliveryFixture.ack(call.getArgument(2))));
            when(grpc.cancelAsync(anyString(), anyInt(), any(), anyLong())).thenReturn(
                    java.util.concurrent.CompletableFuture.completedFuture(org.flexlb.engine.grpc.EngineGrpcClient.CancelReply.REQUEST_CLEANED));
            var dispatcher = new DeliveryDispatcher(grpc, service, reporter);
            var repository = new RequestRepository();
            runtime = new SchedulerRuntime(repository, mock(EndpointRegistry.class), reporter,
                    mock(RequestSchedulerReporter.class), dispatcher, service, grpc);
            var availability = new PlacementAvailability();
            endpoints = new EndpointRegistry(service, repository, reporter, dispatcher, availability);
            ReflectionTestUtils.setField(runtime, "endpoints", endpoints);
            a = (PrefillEndpoint) publish(RoleType.PREFILL, "127.0.0.1");
            b = (PrefillEndpoint) publish(RoleType.PREFILL, "127.0.0.2");
            decode = separateDecode ? (org.flexlb.balance.endpoint.DecodeEndpoint) publish(RoleType.DECODE, "127.0.0.3") : null;
            when(cache.findMatchingEngines(any(), any(), any())).thenReturn(Map.of());
            var strategy = new CostBasedPrefillStrategy(endpoints, cache, mock(EngineHealthReporter.class),
                    mock(CacheMetricsReporter.class), service);
            var model = mock(ModelMetaConfig.class);
            when(model.requiredRoles()).thenReturn(separateDecode ? List.of(RoleType.PREFILL, RoleType.DECODE) : List.of(RoleType.PREFILL));
            router = spy(new RequestWorkerSelector(strategy, new DecodeSelector(endpoints, service, SchedulingTestConfig.capacity(config)), mock(VitWorkerSelector.class), model));
            scheduler = (QueuedRequestScheduler) PlacementConfiguration.create(runtime, config, router, reporter,
                    eviction, availability, SchedulingTestConfig.capacity(config),
                    mock(RecentCacheKeyTraceReporter.class));
            runtime.registerScheduler(scheduler);
        }

        RequestContext context(long id) {
            var context = RequestProtocolTestSupport.context(config, id);
            context.data().setRequest(context.data().getRequest().toBuilder().seqLen(1000L).maxNewTokens(1)
                    .blockCacheKeys(List.of(1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L)).cacheKeyBlockSize(100L).build());
            if (config.getDispatcher().requiresGenerateInput()) {
                context.data().setGenerateInputPb(org.flexlb.engine.grpc.EngineRpcService.GenerateInputPB.newBuilder()
                        .setRequestId(id).setGenerateConfig(org.flexlb.engine.grpc.EngineRpcService.GenerateConfigPB.newBuilder())
                        .build().toByteString());
            }
            return context;
        }

        PrefillState.RouteReservation occupy(PrefillEndpoint endpoint, long id) {
            var context = context(id);
            var route = SchedulingTestConfig.createRoute(context, null, null, null, endpoint, null, null,
                    System.currentTimeMillis());
            var result = endpoint.ownership.reserveUnqueuedRoute(route, 1000, 1);
            assertEquals(PrefillState.CapacityStatus.ACQUIRED, result.status());
            return result.reservation();
        }

        org.flexlb.balance.endpoint.WorkerEndpoint publish(RoleType role, String ip) {
            var worker = WorkerStatus.createDiscovered(role, "g1", ip, 8080, 8081, "test");
            worker.lock.lock();
            try {
                var status = new WorkerStatusResponse();
                status.setRole(role);
                status.setAlive(true);
                status.setStatusVersion(1L);
                status.setRunningTaskInfo(Map.of());
                status.setFinishedTaskInfo(Map.of());
                status.setTotalKvCacheTokens(1_000_000L);
                status.setAvailableKvCacheTokens(1_000_000L);
                return endpoints.publishPreparedEndpoint(worker.getIpPort(), worker,
                        worker.prepareNewStatus(WorkerStatusTestSupport.freezeStatusResponse(worker, status)));
            } finally { worker.lock.unlock(); }
        }

        @Override public void close() { runtime.shutdown(); }
    }
}
