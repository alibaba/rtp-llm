package org.flexlb.balance.strategy;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.scheduler.SchedulingTestConfig;
import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.ConfigTestSupport;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class PrefillQueueEstimateTest {
    @Test void estimateUsesPoolCacheTokensAndConfiguredUncachedWorkFormula() {
        var config = StrategyTestSupport.config();
        config.setDispatcher(ConfigTestSupport.nonBatch());
        config.getRouter().getRoles().getPrefill().getExecutionTimeEstimator().setExpression("sum(computeTokens) / 70");
        var service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        EndpointRegistry registry = StrategyTestSupport.endpointRegistry(service);
        try {
            for (int i = 1; i <= 2; i++) {
                String ip = "10.0.0." + i;
                var status = StrategyTestSupport.workerStatus(RoleType.PREFILL, "a", ip, 8080, 8081,
                        true, 1_000_000L, 1_000_000L);
                StrategyTestSupport.publishEndpoint(registry, RoleType.PREFILL, ip + ":8080", status);
            }
            var cache = mock(CacheAwareService.class);
            when(cache.findMatchingEngines(any(), any(), any())).thenReturn(Map.of("10.0.0.2:8080", 900));
            var strategy = new CostBasedPrefillStrategy(registry, cache, mock(EngineHealthReporter.class),
                    mock(CacheMetricsReporter.class), SchedulingTestConfig.configService(config));
            var request = Request.builder().requestId(1L).seqLen(100_000L).cacheKeyBlockSize(100)
                    .blockCacheKeys(List.of()).build();
            var estimate = strategy.estimateQueuedWork(request, RoleType.PREFILL, "a");
            assertNotNull(estimate);
            assertEquals(2, estimate.workers());
            assertEquals(90_000L, estimate.hitTokens());
            assertTrue(estimate.workMs() >= 142 && estimate.workMs() <= 143);
            assertNull(strategy.estimateQueuedWork(request, RoleType.PREFILL, "b"));
            when(cache.findMatchingEngines(any(), any(), any())).thenReturn(Map.of("10.0.0.2:8080", 2000));
            assertEquals(99_900L, strategy.estimateQueuedWork(request, RoleType.PREFILL, "a").hitTokens(),
                    "the final block remains compute work even for a full cache match");
        } finally { registry.close(); }
    }
    @Test void benefitEstimateUsesEffectiveCacheAndIgnoresTimePrediction() {
        var config = StrategyTestSupport.config();
        config.setDispatcher(ConfigTestSupport.nonBatch());
        var balance = new RoutingConfig.TokenBalanceConfig();
        balance.setMinPrefixHitPercent(5);
        config.getRouter().getRoles().getPrefill().setTokenBalance(balance);
        config.getRouter().getRoles().getPrefill().getExecutionTimeEstimator().setExpression("sum(computeTokens) + 1000 * sum(hitCacheTokens)");
        var service = SchedulingTestConfig.configService(config);
        var registry = StrategyTestSupport.endpointRegistry(service);
        try {
            for (int i = 1; i <= 2; i++) {
                String ip = "10.0.0." + i;
                var status = StrategyTestSupport.workerStatus(RoleType.PREFILL, i == 1 ? "a" : "b",
                        ip, 8080, 8081, true, 1_000_000L, 1_000_000L);
                StrategyTestSupport.publishEndpoint(registry, RoleType.PREFILL, ip + ":8080", status);
            }
            var cache = mock(CacheAwareService.class);
            when(cache.findMatchingEngines(any(), any(), any())).thenReturn(
                    Map.of("10.0.0.1:8080", 900, "10.0.0.2:8080", 999));
            var strategy = new CostBasedPrefillStrategy(registry, cache, mock(EngineHealthReporter.class),
                    mock(CacheMetricsReporter.class), service);
            var request = Request.builder().requestId(2L).seqLen(100_000L).cacheKeyBlockSize(100)
                    .blockCacheKeys(List.of()).build();
            var estimate = strategy.estimateQueuedTokens(request, RoleType.PREFILL, "a");
            assertEquals(100_000L, estimate.inputTokens());
            assertEquals(90_000L, estimate.hitTokens());
            assertEquals(10_000L, estimate.computeTokens());
            assertNull(strategy.estimateQueuedTokens(request, RoleType.PREFILL, "missing"));
            assertEquals(99_900L, strategy.estimateQueuedTokens(request, RoleType.PREFILL, null).hitTokens());
            assertTrue(strategy.estimateQueuedWork(request, RoleType.PREFILL, "a").workMs() > 1_000_000L,
                    "time predictions and benefit token counts are independent");
            when(cache.findMatchingEngines(any(), any(), any())).thenReturn(Map.of("10.0.0.1:8080", 50));
            assertEquals(0L, strategy.estimateQueuedTokens(request, RoleType.PREFILL, "a").hitTokens(),
                    "exactly the affinity threshold is routed as cold");
            when(cache.findMatchingEngines(any(), any(), any())).thenReturn(Map.of("10.0.0.1:8080", 51));
            assertEquals(5100L, strategy.estimateQueuedTokens(request, RoleType.PREFILL, "a").hitTokens());
            when(cache.findMatchingEngines(any(), any(), any())).thenReturn(Map.of("10.0.0.1:8080", 2000));
            assertEquals(99_900L, strategy.estimateQueuedTokens(request, RoleType.PREFILL, "a").hitTokens(),
                    "the final block remains compute work for full matches");
        } finally { registry.close(); }
    }

}
