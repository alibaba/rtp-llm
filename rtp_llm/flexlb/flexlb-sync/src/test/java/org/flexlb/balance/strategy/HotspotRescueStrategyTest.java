package org.flexlb.balance.strategy;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.scheduler.PlacementKey;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.WorkerDirectory;
import org.flexlb.testfixture.WorkerStatusResponse;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/** Hotspot rescue pool facts and migrated-request placement rules. */
class HotspotRescueStrategyTest {
    private FlexlbConfig config;
    private RoutingConfig.TokenBalanceConfig balance;
    private EndpointRegistry registry;
    private CacheAwareService cache;
    private CostBasedPrefillStrategy strategy;
    private BalanceContext context;

    @BeforeEach
    void setUp() {
        config = StrategyTestSupport.config();
        config.setDispatcher(DispatcherConfig.nonBatch());
        config.getRouter().getRoles().getPrefill().getExecutionTimeEstimator()
                .setExpression("sum(computeTokens)");
        balance = new RoutingConfig.TokenBalanceConfig();
        config.getRouter().getRoles().getPrefill().setTokenBalance(balance);
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        registry = StrategyTestSupport.endpointRegistry(configService);
        cache = mock(CacheAwareService.class);
        when(cache.findMatchingEngines(any(), any(), any())).thenReturn(Map.of());
        strategy = new CostBasedPrefillStrategy(
                new WorkerDirectory(registry), cache, mock(EngineHealthReporter.class));
        context = hotContext(10_001L, 1_000L);
    }

    @AfterEach
    void tearDown() {
        registry.close();
    }

    @Test
    void rescueStaysInactiveWithoutHotspotRescueConfig() {
        publish("10.0.0.1", 31L);
        // No rescue config: no pool facts and no target search, even if a context
        // carries stale migration evidence.
        assertTrue(strategy.hotspotRescuePools(RoleType.PREFILL, null).isEmpty());
        context.setPrefillMigrationSource("10.0.0.1:8080");
        assertFalse(strategy.hasCacheEligibleRescueTarget(
                context, RoleType.PREFILL, null, endpoint("10.0.0.1")));
        // Ordinary selection is untouched: the recorded source stays selectable.
        try (SelectedRole selected = select()) {
            assertEquals("10.0.0.1", selected.serverStatus().getServerIp());
        }
    }

    @Test
    void poolsReportOnlyWorkersBeyondTheEngineWaitingThreshold() {
        enableRescue();
        publish("10.0.0.1", 31L);
        publish("10.0.0.2", 5L);
        publish("10.0.0.3", 256L);
        List<CostBasedPrefillStrategy.HotspotRescuePool> pools =
                strategy.hotspotRescuePools(RoleType.PREFILL, balance.getHotspotRescue());
        assertEquals(1, pools.size());
        List<String> sources = pools.getFirst().sources().stream()
                .map(PrefillEndpoint::ipPort).toList();
        assertEquals(List.of("10.0.0.1:8080", "10.0.0.3:8080"), sources);
        assertTrue(pools.getFirst().idleTarget());
    }

    @Test
    void idleTargetRequiresPendingAndPlannedWorkWithinTheWatermark() {
        enableRescue();
        publish("10.0.0.1", 31L);
        publish("10.0.0.2", 0L);
        PrefillEndpoint loaded = endpoint("10.0.0.1");
        PrefillEndpoint idle = endpoint("10.0.0.2");
        try (var ignored = loaded.planTokenWork(1_000L);
                var planned = idle.planTokenWork(1_000L)) {
            // 2000 outstanding > default watermark 0: no idle target in the pool.
            List<CostBasedPrefillStrategy.HotspotRescuePool> pools =
                    strategy.hotspotRescuePools(RoleType.PREFILL, balance.getHotspotRescue());
            assertEquals(1, pools.size());
            assertFalse(pools.getFirst().idleTarget());
        }
        // Planned work is advisory: releasing the reservation restores the target.
        List<CostBasedPrefillStrategy.HotspotRescuePool> pools =
                strategy.hotspotRescuePools(RoleType.PREFILL, balance.getHotspotRescue());
        assertTrue(pools.getFirst().idleTarget());
    }

    @Test
    void migratedColdRequestSelectsAnyIdleWorkerExceptItsSource() {
        enableRescue();
        publish("10.0.0.1", 31L);
        publish("10.0.0.2", 0L);
        context.setPrefillMigrationSource("10.0.0.1:8080");
        try (SelectedRole selected = select()) {
            assertEquals("10.0.0.2", selected.serverStatus().getServerIp());
        }
    }

    @Test
    void migratedRequestWithoutAnotherIdleWorkerParksInsteadOfReturning() {
        enableRescue();
        publish("10.0.0.1", 31L);
        context.setPrefillMigrationSource("10.0.0.1:8080");
        PlacementResult<SelectedRole, PlacementKey> result =
                strategy.select(context, RoleType.PREFILL, null);
        assertEquals(PlacementResult.Status.BLOCKED, result.status());
    }

    @Test
    void migratedHotRequestKeepsCacheEligibilityAndNeverDegradesTheHit() {
        enableRescue();
        publish("10.0.0.1", 31L);
        publish("10.0.0.2", 0L);
        context.getRequest().setBlockCacheKeys(List.of(1L, 2L, 3L, 4L, 5L));
        context.getRequest().setCacheKeyBlockSize(100L);
        // The source is the sole cache leader (600 of 1000 tokens); the other
        // worker only reaches 300, below the leader hit.
        when(cache.findMatchingEngines(any(), any(), any()))
                .thenReturn(Map.of("10.0.0.1:8080", 6, "10.0.0.2:8080", 3));
        context.setPrefillMigrationSource("10.0.0.1:8080");
        assertEquals(PlacementResult.Status.BLOCKED,
                strategy.select(context, RoleType.PREFILL, null).status(),
                "a rescued hot request parks rather than degrading its cache hit");
        // An equal-hit target is accepted and the search confirms it beforehand.
        when(cache.findMatchingEngines(any(), any(), any()))
                .thenReturn(Map.of("10.0.0.1:8080", 6, "10.0.0.2:8080", 6));
        assertTrue(strategy.hasCacheEligibleRescueTarget(
                context, RoleType.PREFILL, null, endpoint("10.0.0.1")));
        try (SelectedRole selected = select()) {
            assertEquals("10.0.0.2", selected.serverStatus().getServerIp());
        }
        // The source itself is never an eligible rescue target.
        when(cache.findMatchingEngines(any(), any(), any()))
                .thenReturn(Map.of("10.0.0.1:8080", 6));
        assertFalse(strategy.hasCacheEligibleRescueTarget(
                context, RoleType.PREFILL, null, endpoint("10.0.0.1")));
    }

    private void enableRescue() {
        balance.setHotspotRescue(new RoutingConfig.HotspotRescueConfig());
    }

    private SelectedRole select() {
        PlacementResult<SelectedRole, PlacementKey> result =
                strategy.select(context, RoleType.PREFILL, null);
        assertEquals(PlacementResult.Status.SUCCESS, result.status());
        return result.value();
    }

    private BalanceContext hotContext(long requestId, long seqLen) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(seqLen);
        request.setPriority(50);
        request.setBlockCacheKeys(List.of());
        BalanceContext hot = new BalanceContext(config);
        hot.setRequest(request);
        hot.setSchedulingMetadata(SchedulingMetadata.explicit(
                50, System.currentTimeMillis() + 60_000L));
        return hot;
    }

    private PrefillEndpoint endpoint(String ip) {
        return (PrefillEndpoint) registry.get(RoleType.PREFILL, ip + ":8080");
    }

    private PrefillEndpoint publish(String ip, long waitingQueryLen) {
        WorkerStatus status = WorkerStatus.createDiscovered(
                RoleType.PREFILL, null, ip, 8080, 8081, "test-site");
        WorkerStatusResponse response = StrategyTestSupport.response(
                RoleType.PREFILL, true, 1_000_000L, 1_000_000L,
                Math.max(1L, status.appliedStatusCursor().statusVersion() + 1L));
        response.setWaitingQueryLen(waitingQueryLen);
        StrategyTestSupport.publish(status, response);
        return (PrefillEndpoint) StrategyTestSupport.publishEndpoint(
                registry, RoleType.PREFILL, ip + ":8080", status);
    }
}
