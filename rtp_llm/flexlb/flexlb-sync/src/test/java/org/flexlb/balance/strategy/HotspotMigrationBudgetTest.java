package org.flexlb.balance.strategy;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.scheduler.RequestContext;
import org.flexlb.balance.scheduler.RequestRoute;
import org.flexlb.balance.scheduler.SchedulingTestConfig;
import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.junit.jupiter.api.Test;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.atLeastOnce;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class HotspotMigrationBudgetTest {
    @Test
    void boundsQueueCaptureAndRotatesSourcesAcrossLargeFleet() {
        var config = SchedulingTestConfig.newConfig();
        var balance = new RoutingConfig.TokenBalanceConfig();
        balance.setMaxPendingComputeTokens(100L);
        balance.setHotspotRescue(new RoutingConfig.HotspotRescueConfig());
        config.getRouter().getRoles().getPrefill().setTokenBalance(balance);
        var registry = mock(EndpointRegistry.class);
        var fleet = new ArrayList<EndpointRegistry.PrefillRoutingEntry>();
        var sources = new ArrayList<PrefillEndpoint>();
        var checks = new AtomicInteger();
        for (int i = 0; i < 200; i++) {
            var endpoint = endpoint(i, i >= 100);
            var routes = new ArrayList<RequestRoute>();
            for (int j = 0; j < 64; j++) { var route = route(100, i * 64L + j + 1); when(route.prefillEp()).thenReturn(endpoint); routes.add(route); }
            when(endpoint.ownership.captureQueue(anyInt())).thenAnswer(call -> {
                int limit = call.getArgument(0);
                checks.addAndGet(limit);
                return new PrefillState.QueueSnapshot(1, 1, routes.subList(0, limit), routes.getFirst());
            });
            if (i < 100) { sources.add(endpoint); }
            fleet.add(new EndpointRegistry.PrefillRoutingEntry("worker-" + i, endpoint));
        }
        when(registry.prefillRoutingSnapshot(RoleType.PREFILL)).thenReturn(fleet);
        var strategy = new CostBasedPrefillStrategy(registry, mock(CacheAwareService.class),
                mock(EngineHealthReporter.class), mock(CacheMetricsReporter.class), SchedulingTestConfig.configService(config));
        for (int round = 0; round < 200; round++) {
            checks.set(0);
            assertTrue(strategy.hotspotMigrationCandidates(2).size() <= 2);
            assertTrue(checks.get() <= 128, "captured queue entries must be bounded independently of fleet and backlog size");
        }
        for (var source : sources) { verify(source.ownership, atLeastOnce()).captureQueue(anyInt()); }
    }

    @Test
    void returnsOnlyBestBoundedCandidatesFromExaminedSources() {
        var config = SchedulingTestConfig.newConfig();
        var balance = new RoutingConfig.TokenBalanceConfig();
        balance.setMaxPendingComputeTokens(100L);
        balance.setHotspotRescue(new RoutingConfig.HotspotRescueConfig());
        config.getRouter().getRoles().getPrefill().setTokenBalance(balance);
        var registry = mock(EndpointRegistry.class);
        var source = endpoint(0, false);
        var idle = endpoint(1, true);
        var routes = List.of(route(400, 1), route(100, 2), route(200, 3), route(100, 4));
        for (var route : routes) { when(route.prefillEp()).thenReturn(source); }
        when(source.ownership.captureQueue(anyInt())).thenReturn(new PrefillState.QueueSnapshot(1, 1, routes, routes.getFirst()));
        when(registry.prefillRoutingSnapshot(RoleType.PREFILL)).thenReturn(List.of(
                new EndpointRegistry.PrefillRoutingEntry("a", source), new EndpointRegistry.PrefillRoutingEntry("b", idle)));
        var strategy = new CostBasedPrefillStrategy(registry, mock(CacheAwareService.class),
                mock(EngineHealthReporter.class), mock(CacheMetricsReporter.class), SchedulingTestConfig.configService(config));
        assertEquals(List.of(routes.get(1), routes.get(3)), strategy.hotspotMigrationCandidates(2));
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.CsvSource({"9,0,false", "9,5,false", "9,9,true", "4,0,true", "4,2,true", "0,0,true"})
    void migrationPreservesHotCacheAffinityAndIgnoresColdCacheAffinity(int sourceHit, int idleHit, boolean allowed) {
        var config = SchedulingTestConfig.newConfig();
        var balance = new RoutingConfig.TokenBalanceConfig();
        balance.setMaxPendingComputeTokens(100L);
        balance.setHotspotRescue(new RoutingConfig.HotspotRescueConfig());
        config.getRouter().getRoles().getPrefill().setTokenBalance(balance);
        var registry = mock(EndpointRegistry.class);
        var source = endpoint(0, false);
        var idle = endpoint(1, true);
        var route = route(1000, 1);
        when(route.prefillEp()).thenReturn(source);
        when(source.ownership.captureQueue(anyInt())).thenReturn(new PrefillState.QueueSnapshot(1, 1, List.of(route), route));
        when(registry.prefillRoutingSnapshot(RoleType.PREFILL)).thenReturn(List.of(
                new EndpointRegistry.PrefillRoutingEntry("a", source), new EndpointRegistry.PrefillRoutingEntry("b", idle)));
        var cache = mock(CacheAwareService.class);
        when(cache.findMatchingEngines(org.mockito.ArgumentMatchers.any(), org.mockito.ArgumentMatchers.any(), org.mockito.ArgumentMatchers.any()))
                .thenReturn(java.util.Map.of("a", sourceHit, "b", idleHit));
        var strategy = new CostBasedPrefillStrategy(registry, cache, mock(EngineHealthReporter.class),
                mock(CacheMetricsReporter.class), SchedulingTestConfig.configService(config));
        assertEquals(allowed ? List.of(route) : List.of(), strategy.hotspotMigrationCandidates(1));
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.CsvSource({"0,0,0,true,false", "0,1,0,false,false", "50,20,30,true,false",
            "50,20,31,false,false", "0,0,0,false,true"})
    void targetWatermarkIncludesPendingAndPlansButNotCompletedHistory(long threshold, long pending,
            long planned, boolean allowed, boolean congested) {
        var config = SchedulingTestConfig.newConfig();
        var balance = new RoutingConfig.TokenBalanceConfig();
        balance.setMaxPendingComputeTokens(100L);
        var rescue = new RoutingConfig.HotspotRescueConfig();
        rescue.setMaxIdlePendingTokens(threshold);
        balance.setHotspotRescue(rescue);
        config.getRouter().getRoles().getPrefill().setTokenBalance(balance);
        var registry = mock(EndpointRegistry.class);
        var source = endpoint(0, false);
        var target = endpoint(1, true);
        when(target.engineQueueAdmissionBlocked()).thenReturn(congested);
        when(target.ownership.tokenWorkload()).thenReturn(new PrefillTokenWorkload.Snapshot(10_000, pending, planned));
        var route = route(1000, 1);
        when(route.prefillEp()).thenReturn(source);
        when(source.ownership.captureQueue(anyInt())).thenReturn(new PrefillState.QueueSnapshot(1, 1, List.of(route), route));
        when(registry.prefillRoutingSnapshot(RoleType.PREFILL)).thenReturn(List.of(
                new EndpointRegistry.PrefillRoutingEntry("a", source), new EndpointRegistry.PrefillRoutingEntry("b", target)));
        var cache = mock(CacheAwareService.class);
        when(cache.findMatchingEngines(org.mockito.ArgumentMatchers.any(), org.mockito.ArgumentMatchers.any(), org.mockito.ArgumentMatchers.any()))
                .thenReturn(java.util.Map.of("a", 9, "b", 9));
        var strategy = new CostBasedPrefillStrategy(registry, cache, mock(EngineHealthReporter.class),
                mock(CacheMetricsReporter.class), SchedulingTestConfig.configService(config));
        assertEquals(allowed ? List.of(route) : List.of(), strategy.hotspotMigrationCandidates(1));
        assertEquals(allowed, !strategy.hotspotRescueDomains().isEmpty());
    }

    @Test
    void shortestTotalInputWinsEvenWhenLongRequestHasLessUncachedWork() {
        var config = SchedulingTestConfig.newConfig();
        var balance = new RoutingConfig.TokenBalanceConfig();
        balance.setMaxPendingComputeTokens(100L);
        balance.setHotspotRescue(new RoutingConfig.HotspotRescueConfig());
        config.getRouter().getRoles().getPrefill().setTokenBalance(balance);
        var registry = mock(EndpointRegistry.class);
        var source = endpoint(0, false);
        var idle = endpoint(1, true);
        var shortRoute = route(900, 2); // 800 hit + 100 compute
        var longRoute = route(950, 1);  // 900 hit + 50 compute
        when(shortRoute.hitCache()).thenReturn(800L);
        when(longRoute.hitCache()).thenReturn(900L);
        for (var route : List.of(shortRoute, longRoute)) { when(route.prefillEp()).thenReturn(source); }
        when(source.ownership.captureQueue(anyInt())).thenReturn(
                new PrefillState.QueueSnapshot(1, 1, List.of(longRoute, shortRoute), longRoute));
        when(registry.prefillRoutingSnapshot(RoleType.PREFILL)).thenReturn(List.of(
                new EndpointRegistry.PrefillRoutingEntry("a", source), new EndpointRegistry.PrefillRoutingEntry("b", idle)));
        var cache = mock(CacheAwareService.class);
        when(cache.findMatchingEngines(org.mockito.ArgumentMatchers.any(), org.mockito.ArgumentMatchers.any(), org.mockito.ArgumentMatchers.any()))
                .thenReturn(java.util.Map.of("a", 8, "b", 8));
        var strategy = new CostBasedPrefillStrategy(registry, cache, mock(EngineHealthReporter.class),
                mock(CacheMetricsReporter.class), SchedulingTestConfig.configService(config));
        assertEquals(List.of(shortRoute), strategy.hotspotMigrationCandidates(1));
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.CsvSource({
            "30,-1,0,2100000,false", "30,0,256,2100000,false",
            "30,29,0,2100000,false", "30,30,0,2100000,false",
            "30,31,0,1280000,true", "30,256,0,1280000,true",
            "60,60,0,2100000,false", "60,61,0,1,true",
            "0,0,256,2100000,false", "0,1,0,1,true"
    })
    void sourceUsesEngineWaitingCountInsteadOfPendingTokens(long threshold, long waiting, long running,
            long pending, boolean allowed) {
        var config = SchedulingTestConfig.newConfig();
        var balance = new RoutingConfig.TokenBalanceConfig();
        var rescue = new RoutingConfig.HotspotRescueConfig();
        rescue.setEngineQueueMigrationThreshold(threshold);
        balance.setHotspotRescue(rescue);
        config.getRouter().getRoles().getPrefill().setTokenBalance(balance);
        var registry = mock(EndpointRegistry.class);
        var source = endpoint(0, false);
        var idle = endpoint(1, true);
        publishEngineQueue(source, waiting, running);
        when(source.ownership.tokenWorkload()).thenReturn(new PrefillTokenWorkload.Snapshot(pending));
        var route = route(100, 1);
        when(route.prefillEp()).thenReturn(source);
        when(source.ownership.captureQueue(anyInt())).thenReturn(
                new PrefillState.QueueSnapshot(1, 1, List.of(route), route));
        when(registry.prefillRoutingSnapshot(RoleType.PREFILL)).thenReturn(List.of(
                new EndpointRegistry.PrefillRoutingEntry("a", source), new EndpointRegistry.PrefillRoutingEntry("b", idle)));
        var strategy = new CostBasedPrefillStrategy(registry, mock(CacheAwareService.class),
                mock(EngineHealthReporter.class), mock(CacheMetricsReporter.class), SchedulingTestConfig.configService(config));
        assertEquals(allowed ? List.of(route) : List.of(), strategy.hotspotMigrationCandidates(1));
        if (!allowed) {
            verify(source.ownership, org.mockito.Mockito.never()).captureQueue(anyInt());
        }
        // A subsequent Engine report clears the source trigger, independently of local queue/token backlog.
        publishEngineQueue(source, 0L, 256L);
        assertTrue(strategy.hotspotMigrationCandidates(1).isEmpty());
    }

    private void publishEngineQueue(PrefillEndpoint endpoint, long waiting, long running) {
        var status = endpoint.getStatus();
        var response = StrategyTestSupport.response(RoleType.PREFILL, true, 10_000, 10_000,
                status.appliedStatusCursor().statusVersion() + 1L);
        response.setWaitingQueryLen(waiting);
        response.setRunningQueryLen(running);
        StrategyTestSupport.publish(status, response);
    }

    private PrefillEndpoint endpoint(int id, boolean idle) {
        var endpoint = mock(PrefillEndpoint.class);
        var ownership = mock(PrefillState.class);
        ReflectionTestUtils.setField(endpoint, "ownership", ownership);
        when(endpoint.getStatus()).thenReturn(StrategyTestSupport.workerStatus(RoleType.PREFILL, null,
                "10.0.0." + id, 8080, 8081, true, 10_000, 10_000));
        publishEngineQueue(endpoint, idle ? 0L : 31L, 0L);
        when(ownership.tokenWorkload()).thenReturn(new PrefillTokenWorkload.Snapshot(idle ? 0L : 1000L));
        when(ownership.progress()).thenReturn(mock(PrefillState.Progress.class));
        var routes = List.of(route(100, id + 1));
        when(ownership.captureQueue(anyInt())).thenReturn(new PrefillState.QueueSnapshot(1, 1, routes, routes.getFirst()));
        return endpoint;
    }

    private RequestRoute route(long tokens, long sequence) {
        var route = mock(RequestRoute.class);
        when(route.seqLen()).thenReturn(tokens);
        when(route.enqueueSeq()).thenReturn(sequence);
        when(route.decodeEp()).thenReturn(mock(DecodeEndpoint.class));
        var context = new RequestContext(SchedulingTestConfig.newConfig());
        context.data().setRequest(org.flexlb.dao.loadbalance.Request.builder().requestId(sequence).seqLen(tokens)
                .blockCacheKeys(List.of(1L,2L,3L,4L,5L,6L,7L,8L,9L)).cacheKeyBlockSize(100L).build());
        context.register(sequence, System.currentTimeMillis() - 6000, Long.MAX_VALUE, 60_000);
        when(route.ctx()).thenReturn(context);
        return route;
    }
}
