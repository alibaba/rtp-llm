package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.ConfigTestSupport;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.util.ArrayList;
import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

class TokenPlanWakeupTest {
    @ParameterizedTest
    @CsvSource({"NON_BATCH,false,false", "NON_BATCH,true,false", "BATCH,false,false", "BATCH,true,false",
            "NON_BATCH,false,true", "NON_BATCH,true,true", "BATCH,false,true", "BATCH,true,true"})
    void abandonedSelectionWakesWaiterWithoutAnEngineCompletion(DispatcherConfig.Type mode, boolean beforePark, boolean admitted) {
        var config = SchedulingTestConfig.newConfig();
        config.setDispatcher(mode == DispatcherConfig.Type.NON_BATCH
                ? ConfigTestSupport.nonBatch() : SchedulingTestConfig.batchConfig().getDispatcher());
        config.getRouter().getRoles().getPrefill().setTokenBalance(new RoutingConfig.TokenBalanceConfig());
        config.getRouter().getRoles().getPrefill().getTokenBalance().setMaxPendingComputeTokens(100L);
        var availability = new PlacementAvailability();
        var service = SchedulingTestConfig.configService(config);
        var registry = new org.flexlb.balance.endpoint.EndpointRegistry(service, new RequestRepository(),
                mock(DeliveryMetricsReporter.class), SchedulerTestSupport.dispatcher(), availability);
        try {
            var worker = WorkerStatus.createDiscovered(RoleType.PREFILL, "g", "127.0.0.1", 8000, 8001, "test");
            var response = new org.flexlb.dao.master.WorkerStatusResponse();
            response.setRole(RoleType.PREFILL); response.setAlive(true); response.setStatusVersion(1L);
            response.setAvailableKvCacheTokens(1_000_000L); response.setTotalKvCacheTokens(1_000_000L);
            PrefillEndpoint endpoint;
            worker.lock.lock();
            try {
                var prepared = worker.prepareNewStatus(org.flexlb.dao.master.WorkerStatusTestSupport.freezeStatusResponse(worker, response));
                endpoint = (PrefillEndpoint) registry.publishPreparedEndpoint(worker.getIpPort(), worker, prepared);
            } finally { worker.lock.unlock(); }
            var cache = mock(org.flexlb.cache.service.CacheAwareService.class);
            when(cache.findMatchingEngines(any(), any(), any())).thenReturn(java.util.Map.of());
            var strategy = new org.flexlb.balance.strategy.CostBasedPrefillStrategy(registry, cache,
                    mock(org.flexlb.service.monitor.EngineHealthReporter.class),
                    mock(org.flexlb.cache.monitor.CacheMetricsReporter.class), service);
            var queue = new PlacementCapacityWaitQueue(false, availability);
            availability.addListener(queue::capacityChanged);
            var context = new RequestContext(config);
            context.data().setRequest(Request.builder().requestId(1L).seqLen(100L).build());
            context.data().setFuture(new CompletableFuture<>());
            context.data().setSchedulingMetadata(org.flexlb.dao.SchedulingMetadata.explicit(50, Long.MAX_VALUE));
            var waiter = new GlobalQueueEntry(context, null);
            var key = PlacementKey.exact(RoleType.PREFILL, "g", endpoint.ipPort());
            long version = availability.sequence();
            var route = new RequestRoute(context);
            assertEquals(org.flexlb.balance.PlacementResult.Status.SUCCESS,
                    strategy.select(context.data().getRequest(), context.data().getSchedulingMetadata(), RoleType.PREFILL, "g", route).status());
            try (var blocked = new RequestRoute(context)) {
                assertEquals(org.flexlb.balance.PlacementResult.Status.BLOCKED,
                        strategy.select(context.data().getRequest(), context.data().getSchedulingMetadata(), RoleType.PREFILL, "g", blocked).status());
            }
            var canonical = RequestRoute.create(context, route.prepare(), null);
            if (!beforePark) { assertTrue(queue.park(waiter, key, version)); }
            var lease = admitted ? endpoint.ownership.reserveUnqueuedRoute(canonical, 1, 0).reservation() : null;
            canonical.close();
            route.close();
            canonical.close();
            if (admitted) {
                assertNotNull(lease);
                assertEquals(version, availability.sequence(), "transfer to owned work creates no capacity edge");
                assertEquals(100L, endpoint.ownership.tokenWorkload().pendingTokens());
                assertEquals(0L, endpoint.ownership.tokenWorkload().plannedTokens());
                endpoint.ownership.release(lease, null);
                return;
            }
            assertEquals(0L, endpoint.ownership.tokenWorkload().plannedTokens());
            if (beforePark) {
                assertFalse(queue.park(waiter, key, version), "release racing with parking must force a retry");
            } else {
                var resumed = new ArrayList<GlobalQueueEntry>();
                queue.resumeReady(1, resumed::add);
                assertEquals(java.util.List.of(waiter), resumed, "idle engine produces no later completion to rescue this waiter");
            }
            assertEquals(version + 1, availability.sequence(), "idempotent close must publish only one capacity edge");
        } finally { registry.close(); }
    }
}
