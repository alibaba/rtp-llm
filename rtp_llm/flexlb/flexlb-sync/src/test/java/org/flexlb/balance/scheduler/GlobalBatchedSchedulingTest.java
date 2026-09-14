package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.DeliveryMetrics;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.eviction.DecodePreemptionCoordinator;
import org.flexlb.balance.eviction.EngineCancelChannel;
import org.flexlb.balance.eviction.EvictionManager;
import org.flexlb.balance.strategy.CostBasedBatchedPrefillStrategy;
import org.flexlb.balance.strategy.CostBasedDecodeStrategy;
import org.flexlb.balance.strategy.RandomStrategy;
import org.flexlb.cache.domain.CacheMatchQuery;
import org.flexlb.cache.domain.CacheMatchResult;
import org.flexlb.cache.domain.CacheMatchSource;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.cache.HostCacheMatch;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.Endpoint;
import org.flexlb.dao.route.GroupRoleEndPoint;
import org.flexlb.dao.route.RoleType;
import org.flexlb.dao.route.ServiceRoute;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.flexlb.sync.status.WorkerDirectory;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/** Public scheduler contracts with real routing, ownership and delivery. */
class GlobalBatchedSchedulingTest {
    @Test
    void incompleteGlobalWindowWaitsThenDeliversWithoutWorkerCollection() throws Exception {
        try (Fixture fixture = new Fixture(2, 500)) {
            var future = fixture.scheduler.submit(
                    RequestLifecycleTestSupport.context(fixture.config, 930001L));
            assertThrows(TimeoutException.class, () -> future.get(100, TimeUnit.MILLISECONDS));
            Response response = future.get(3, TimeUnit.SECONDS);
            assertTrue(response.isSuccess(), response.getErrorMessage());
        }
    }

    @Test
    void fullGlobalWindowDeliversBeforeItsDeadline() throws Exception {
        try (Fixture fixture = new Fixture(2, 3000)) {
            var first = fixture.scheduler.submit(
                    RequestLifecycleTestSupport.context(fixture.config, 930002L));
            assertThrows(TimeoutException.class, () -> first.get(100, TimeUnit.MILLISECONDS));
            var second = fixture.scheduler.submit(
                    RequestLifecycleTestSupport.context(fixture.config, 930003L));
            assertTrue(first.get(1, TimeUnit.SECONDS).isSuccess());
            assertTrue(second.get(1, TimeUnit.SECONDS).isSuccess());
        }
    }

    @Test
    void configurationBoundaryClosesTheCurrentWindowBeforeLimitChanges() throws Exception {
        try (Fixture fixture = new Fixture(2, 3000)) {
            var first = fixture.scheduler.submit(
                    RequestLifecycleTestSupport.context(fixture.config, 930014L));
            assertThrows(TimeoutException.class,
                    () -> first.get(100, TimeUnit.MILLISECONDS));

            FlexlbConfig updatedWindowConfig = ConfigService.parse("""
                    {"scheduler":{"type":"QUEUE","globalDecision":{
                      "type":"FIXED_WINDOW","maxRequests":1,"maxCollectionWaitMs":0},
                      "decision":{"type":"SINGLE"}},
                     "dispatcher":{"type":"NON_BATCH"},
                     "router":{"roles":{"prefill":{"candidateChoice":{
                       "type":"BEST_ONLY"}}}}}
                    """);
            var second = fixture.scheduler.submit(
                    RequestLifecycleTestSupport.context(updatedWindowConfig, 930015L));

            assertTrue(first.get(1, TimeUnit.SECONDS).isSuccess());
            assertTrue(second.get(1, TimeUnit.SECONDS).isSuccess());
        }
    }

    @Test
    void higherPriorityKeepsTheOnlyWorkerSlotDespiteArrivingSecond() throws Exception {
        try (Fixture fixture = new Fixture(2, 500, true)) {
            var low = RequestLifecycleTestSupport.context(fixture.config, 930004L);
            low.setSchedulingMetadata(SchedulingMetadata.explicit(
                    10, System.currentTimeMillis() + 60000));
            var high = RequestLifecycleTestSupport.context(fixture.config, 930005L);
            high.setSchedulingMetadata(SchedulingMetadata.explicit(
                    90, System.currentTimeMillis() + 60000));
            var lowFuture = fixture.scheduler.submit(low);
            assertThrows(TimeoutException.class, () -> lowFuture.get(100, TimeUnit.MILLISECONDS));
            var highFuture = fixture.scheduler.submit(high);
            assertTrue(highFuture.get(1, TimeUnit.SECONDS).isSuccess());
            assertThrows(TimeoutException.class, () -> lowFuture.get(100, TimeUnit.MILLISECONDS));
        }
    }

    @Test
    void higherPriorityArrivalClosesAnIncompleteLowPriorityWindow() throws Exception {
        try (Fixture fixture = new Fixture(3, 1000, true)) {
            var low = RequestLifecycleTestSupport.context(fixture.config, 930010L);
            low.setSchedulingMetadata(SchedulingMetadata.explicit(
                    10, System.currentTimeMillis() + 60000));
            var high = RequestLifecycleTestSupport.context(fixture.config, 930011L);
            high.setSchedulingMetadata(SchedulingMetadata.explicit(
                    90, System.currentTimeMillis() + 60000));
            var lowFuture = fixture.scheduler.submit(low);
            assertThrows(TimeoutException.class, () -> lowFuture.get(100, TimeUnit.MILLISECONDS));
            var highFuture = fixture.scheduler.submit(high);
            assertTrue(highFuture.get(500, TimeUnit.MILLISECONDS).isSuccess());
        }
    }

    @Test
    void lowerPriorityReplansAfterHigherPriorityConsumesWorkerCapacity() throws Exception {
        try (Fixture fixture = new Fixture(2, 3000, true, true)) {
            var low = RequestLifecycleTestSupport.context(fixture.config, 930012L);
            low.setSchedulingMetadata(SchedulingMetadata.explicit(
                    10, System.currentTimeMillis() + 60000));
            var high = RequestLifecycleTestSupport.context(fixture.config, 930013L);
            high.setSchedulingMetadata(SchedulingMetadata.explicit(
                    90, System.currentTimeMillis() + 60000));
            var lowFuture = fixture.scheduler.submit(low);
            assertThrows(TimeoutException.class, () -> lowFuture.get(100, TimeUnit.MILLISECONDS));
            var highFuture = fixture.scheduler.submit(high);
            Response highResponse = highFuture.get(1, TimeUnit.SECONDS);
            Response lowResponse = lowFuture.get(1, TimeUnit.SECONDS);
            assertTrue(highResponse.isSuccess(), highResponse.getErrorMessage());
            assertTrue(lowResponse.isSuccess(), lowResponse.getErrorMessage());
            assertFalse(highResponse.getServerStatus().getFirst().getServerIp()
                    .equals(lowResponse.getServerStatus().getFirst().getServerIp()));
        }
    }

    @Test
    void cancellationDuringCollectionDoesNotConsumeDeliveryCapacity() throws Exception {
        try (Fixture fixture = new Fixture(2, 500, true)) {
            var cancelled = fixture.scheduler.submit(
                    RequestLifecycleTestSupport.context(fixture.config, 930006L));
            assertThrows(TimeoutException.class, () -> cancelled.get(100, TimeUnit.MILLISECONDS));
            assertEquals(RequestState.Phase.CANCEL_REQUESTED,
                    fixture.scheduler.cancelRequest("930006", 0, CancelReason.CLIENT_CANCELLED).state());
            assertFalse(cancelled.get(1, TimeUnit.SECONDS).isSuccess());
            var next = fixture.scheduler.submit(
                    RequestLifecycleTestSupport.context(fixture.config, 930007L));
            assertTrue(next.get(2, TimeUnit.SECONDS).isSuccess());
            assertEquals(RequestState.Phase.CANCELLED,
                    fixture.scheduler.getRequestState("930006", 0).state());
        }
    }

    @Test
    void jointPlacementLeavesTheCachedWorkerForTheRequestThatNeedsIt() throws Exception {
        try (Fixture fixture = new Fixture(2, 3000, false, true)) {
            var flexible = RequestLifecycleTestSupport.context(fixture.config, 930008L);
            flexible.getRequest().setSeqLen(1000L);
            flexible.getRequest().setBlockCacheKeys(List.of());
            var cached = RequestLifecycleTestSupport.context(fixture.config, 930009L);
            cached.getRequest().setSeqLen(1000L);
            cached.getRequest().setBlockCacheKeys(List.of());
            var first = fixture.scheduler.submit(flexible);
            assertThrows(TimeoutException.class, () -> first.get(100, TimeUnit.MILLISECONDS));
            var second = fixture.scheduler.submit(cached);
            Response flexibleResponse = first.get(2, TimeUnit.SECONDS);
            Response cachedResponse = second.get(2, TimeUnit.SECONDS);
            assertTrue(flexibleResponse.isSuccess(), flexibleResponse.getErrorMessage());
            assertTrue(cachedResponse.isSuccess(), cachedResponse.getErrorMessage());
            assertEquals("127.0.0.2", flexibleResponse.getServerStatus().getFirst().getServerIp());
            assertEquals("127.0.0.1", cachedResponse.getServerStatus().getFirst().getServerIp());
            assertEquals(900L, cachedResponse.getServerStatus().getFirst().getDebugInfo().getHitCacheLen());
        }
    }

    private static final class Fixture implements AutoCloseable {
        final FlexlbConfig config;
        final RequestScheduler scheduler;
        final SchedulerRuntime runtime;
        final EvictionManager eviction;

        Fixture(int maximum, long waitMs) {
            this(maximum, waitMs, false);
        }

        Fixture(int maximum, long waitMs, boolean priority) {
            this(maximum, waitMs, priority, false);
        }

        Fixture(int maximum, long waitMs, boolean priority, boolean jointPlacement) {
            ConfigService service = new ConfigService();
            config = service.loadBalanceConfig();
            FlexlbConfig parsed = ConfigService.parse("""
                    {"scheduler":{"type":"QUEUE","globalDecision":{
                      "type":"FIXED_WINDOW","maxRequests":%d,"maxCollectionWaitMs":%d},
                      "decision":{"type":"SINGLE"}},
                     "dispatcher":{"type":"NON_BATCH"},
                     "router":{"roles":{"prefill":{"candidateChoice":{"type":"BEST_ONLY"}}}}}
                    """.formatted(maximum, waitMs));
            config.setScheduler(parsed.getScheduler());
            config.setDispatcher(parsed.getDispatcher());
            config.setRouter(parsed.getRouter());
            if (priority) {
                SchedulingTestConfig.usePriorityQueue(config);
                config.getDispatcher().setMaxInflightRequestsPerPrefillWorker(1);
            }
            if (jointPlacement) {
                config.getRouter().getRoles().getPrefill().getExecutionTimeEstimator()
                        .setExpression("sum(computeTokens)");
            }
            FlexMonitor metrics = mock(FlexMonitor.class);
            BatchSchedulerReporter reporter = new BatchSchedulerReporter(metrics);
            RequestSchedulerReporter requestReporter = new RequestSchedulerReporter(metrics);
            EngineCancelChannel cancel = mock(EngineCancelChannel.class);
            RequestRegistry lifecycle = new RequestRegistry(service, reporter, requestReporter, cancel);
            PlacementAvailability availability = new PlacementAvailability();
            EndpointRegistry endpoints = new EndpointRegistry(service, new EndpointEventProjector(lifecycle),
                    reporter, new RouteDeliveryStrategy(lifecycle, new DeliveryMetrics(reporter)), availability);
            publishWorker(endpoints, "127.0.0.1");
            if (jointPlacement) {
                publishWorker(endpoints, "127.0.0.2");
            }
            WorkerDirectory workers = new WorkerDirectory(endpoints);
            CacheAwareService cache = mock(CacheAwareService.class);
            when(cache.findMatchingEngines(any()))
                    .thenReturn(CacheMatchResult.empty(CacheMatchSource.LOCAL_SYNC));
            if (jointPlacement) {
                when(cache.findMatchingEngines(any())).thenAnswer(invocation -> {
                    CacheMatchQuery query = invocation.getArgument(0);
                    return query.requestId().equals("930009")
                            ? new CacheMatchResult(Map.of("127.0.0.1:8080@0", HostCacheMatch.local(9)),
                                    CacheMatchSource.LOCAL_SYNC, 0L, 100L)
                            : CacheMatchResult.empty(CacheMatchSource.LOCAL_SYNC);
                });
            }
            ModelMetaConfig model = new ModelMetaConfig();
            GroupRoleEndPoint group = new GroupRoleEndPoint();
            group.setGroup("g1");
            group.setPdFusionEndpoint(new Endpoint());
            ServiceRoute route = new ServiceRoute();
            route.setServiceId("test");
            route.setRoleEndpoints(List.of(group));
            model.putServiceRoute("test", route);
            DefaultRouter router = new DefaultRouter(
                    new CostBasedBatchedPrefillStrategy(
                            workers, cache, mock(EngineHealthReporter.class)),
                    new CostBasedDecodeStrategy(workers), new RandomStrategy(workers), service, model);
            eviction = new EvictionManager(requestReporter, cancel,
                    new DecodePreemptionCoordinator(cancel, lifecycle), lifecycle, reporter);
            scheduler = new RequestScheduler(service, router, endpoints, reporter, eviction, lifecycle, availability);
            runtime = new SchedulerRuntime(lifecycle, endpoints, reporter, requestReporter, scheduler);
        }

        private static void publishWorker(EndpointRegistry endpoints, String ip) {
            WorkerStatus worker = WorkerStatus.createDiscovered(
                    RoleType.PDFUSION, "g1", ip, 8080, 8081, "test");
            WorkerStatusResponse status = new WorkerStatusResponse();
            status.setRole(RoleType.PDFUSION);
            status.setAlive(true);
            status.setStatusVersion(1L);
            status.setLatestFinishedVersion(0L);
            status.setTotalKvCacheTokens(100000L);
            status.setAvailableKvCacheTokens(100000L);
            status.setMaxSeqLen(10000L);
            status.setMaxBatchTokensSize(10000L);
            worker.lock.lock();
            try {
                endpoints.publishPreparedEndpoint(worker.getLogicalIpPort(), worker,
                        worker.prepareNewStatus(worker.freezeStatusResponse(status)));
                worker.recordSuccessfulPoll(true);
            } finally {
                worker.lock.unlock();
            }
        }

        @Override
        public void close() {
            runtime.shutdown();
            eviction.shutdown();
        }
    }
}
