package org.flexlb.balance.strategy;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.WorkerDirectory;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/** Fleet-level work and cache estimates for queue shedding. */
class PrefillCostEstimateTest {
    private FlexlbConfig config;
    private EndpointRegistry registry;
    private CacheAwareService cache;
    private CostBasedPrefillStrategy strategy;
    private BalanceContext context;

    @BeforeEach
    void setUp() {
        config = StrategyTestSupport.config();
        config.setDispatcher(DispatcherConfig.nonBatch());
        var balance = new RoutingConfig.TokenBalanceConfig();
        balance.setMinPrefixHitPercent(5);
        config.getRouter().getRoles().getPrefill().setTokenBalance(balance);
        config.getRouter().getRoles().getPrefill().getExecutionTimeEstimator()
                .setExpression("sum(computeTokens) + 1000 * sum(hitCacheTokens)");
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        registry = StrategyTestSupport.endpointRegistry(configService);
        cache = mock(CacheAwareService.class);
        strategy = new CostBasedPrefillStrategy(
                new WorkerDirectory(registry), cache, mock(EngineHealthReporter.class));
        publish("10.0.0.1", "a");
        publish("10.0.0.2", "b");
        Request request = new Request();
        request.setRequestId(2L);
        request.setSeqLen(100_000L);
        request.setPriority(50);
        request.setCacheKeyBlockSize(100);
        request.setBlockCacheKeys(List.of());
        context = new BalanceContext(config);
        context.setRequest(request);
        context.setSchedulingMetadata(SchedulingMetadata.explicit(
                50, System.currentTimeMillis() + 60_000L));
    }

    @AfterEach
    void tearDown() {
        registry.close();
    }

    @Test
    void queueSheddingEstimatesWorkAndReachableCacheWithinThePolicyGroup() {
        when(cache.findMatchingEngines(any(), any(), any())).thenReturn(
                Map.of("10.0.0.1:8080", 900, "10.0.0.2:8080", 999));
        var estimate = strategy.estimateQueuedWork(context, RoleType.PREFILL, "a");
        assertEquals(100_000L, estimate.inputTokens());
        assertEquals(90_000L, estimate.hitTokens());
        assertEquals(1, estimate.workers());
        assertTrue(estimate.workMs() > 1_000_000L);
        assertNull(strategy.estimateQueuedWork(context, RoleType.PREFILL, "missing"));
        assertEquals(99_900L, strategy.estimateQueuedWork(context, RoleType.PREFILL, null).hitTokens());
        when(cache.findMatchingEngines(any(), any(), any())).thenReturn(Map.of("10.0.0.1:8080", 2000));
        assertEquals(99_900L, strategy.estimateQueuedWork(context, RoleType.PREFILL, "a").hitTokens(),
                "reachable cache benefit is bounded by the request length");
    }

    private void publish(String ip, String group) {
        var status = StrategyTestSupport.workerStatus(
                RoleType.PREFILL, group, ip, 8080, 8081, true, 1_000_000L, 1_000_000L);
        StrategyTestSupport.publishEndpoint(registry, RoleType.PREFILL, ip + ":8080", status);
    }
}
