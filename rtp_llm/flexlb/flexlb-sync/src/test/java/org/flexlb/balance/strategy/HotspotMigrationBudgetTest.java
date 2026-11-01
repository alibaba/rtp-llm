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
            for (int j = 0; j < 64; j++) { routes.add(route(100, i * 64L + j + 1)); }
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
        when(source.ownership.captureQueue(anyInt())).thenReturn(new PrefillState.QueueSnapshot(1, 1, routes, routes.getFirst()));
        when(registry.prefillRoutingSnapshot(RoleType.PREFILL)).thenReturn(List.of(
                new EndpointRegistry.PrefillRoutingEntry("a", source), new EndpointRegistry.PrefillRoutingEntry("b", idle)));
        var strategy = new CostBasedPrefillStrategy(registry, mock(CacheAwareService.class),
                mock(EngineHealthReporter.class), mock(CacheMetricsReporter.class), SchedulingTestConfig.configService(config));
        assertEquals(List.of(routes.get(1), routes.get(3)), strategy.hotspotMigrationCandidates(2));
    }

    private PrefillEndpoint endpoint(int id, boolean idle) {
        var endpoint = mock(PrefillEndpoint.class);
        var ownership = mock(PrefillState.class);
        ReflectionTestUtils.setField(endpoint, "ownership", ownership);
        when(endpoint.getStatus()).thenReturn(StrategyTestSupport.workerStatus(RoleType.PREFILL, null,
                "10.0.0." + id, 8080, 8081, true, 10_000, 10_000));
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
        context.register(sequence, System.currentTimeMillis() - 6000, Long.MAX_VALUE, 60_000);
        when(route.ctx()).thenReturn(context);
        return route;
    }
}
