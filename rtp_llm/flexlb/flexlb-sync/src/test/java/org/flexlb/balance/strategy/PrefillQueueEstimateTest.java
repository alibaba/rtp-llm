package org.flexlb.balance.strategy;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.balance.scheduler.SchedulingTestConfig;
import org.flexlb.config.ConfigService;
import org.flexlb.config.ConfigTestSupport;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

class PrefillQueueEstimateTest {
    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(value = RoleType.class, names = {"PREFILL", "PDFUSION"})
    void capacityRefreshUsesTheModelsPrefillRoleAndRoutingGroup(RoleType role) {
        var config = StrategyTestSupport.config();
        var service = SchedulingTestConfig.configService(config);
        var registry = StrategyTestSupport.endpointRegistry(service);
        try {
            var strategy = new CostBasedPrefillStrategy(registry, mock(CacheAwareService.class),
                    mock(EngineHealthReporter.class), mock(CacheMetricsReporter.class), service);
            var meta = mock(org.flexlb.config.ModelMetaConfig.class);
            when(meta.requiredRoles()).thenReturn(List.of(role));
            var router = new org.flexlb.balance.scheduler.RequestWorkerSelector(strategy,
                    mock(DecodeSelector.class), mock(VitWorkerSelector.class), meta);
            for (int i = 1; i <= 4; i++) {
                String ip = "10.0.1." + i;
                StrategyTestSupport.publishEndpoint(registry, role, ip + ":8080",
                        StrategyTestSupport.workerStatus(role, "a", ip, 8080, 8081,
                                true, 1_000_000L, 1_000_000L));
                assertEquals(i, router.currentPrefillWorkerCount("a"));
                assertEquals(0, router.currentPrefillWorkerCount("b"));
            }
        } finally { registry.close(); }
    }

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
}
