package org.flexlb.balance.strategy;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.EndpointRegistry;
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
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * LEAST_RECENTLY_USED_IN_POOL must rotate across every endpoint its pool
 * contains. Three identical endpoints make the projected TTFT equal, so the
 * per-endpoint last-selected clock is the only tiebreaker left.
 */
class CostBasedPrefillLruRotationTest {
    private static final List<String> ADDRESSES =
            List.of("10.0.0.1", "10.0.0.2", "10.0.0.3");

    private FlexlbConfig config;
    private EndpointRegistry registry;
    private CostBasedPrefillStrategy strategy;
    private BalanceContext context;

    @BeforeEach
    void setUp() {
        config = new FlexlbConfig();
        config.setDispatcher(DispatcherConfig.nonBatch());
        config.getRouter().getRoles().getPrefill().getExecutionTimeEstimator()
                .setExpression("sum(computeTokens)");

        RoutingConfig.CandidateChoiceConfig candidateChoice = config.getRouter()
                .getRoles().getPrefill().getCandidateChoice();
        candidateChoice.setType(
                RoutingConfig.CandidateChoiceType.LEAST_RECENTLY_USED_IN_POOL);
        // FIXED/3 over three endpoints makes shortestTtftCandidateCount == size,
        // which is the only shape baselinePoolMask fills with every candidate.
        candidateChoice.getPool().setType(
                RoutingConfig.CandidatePoolType.FIXED);
        candidateChoice.getPool().setWorkers(ADDRESSES.size());

        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        registry = StrategyTestSupport.endpointRegistry(configService);
        for (String ip : ADDRESSES) {
            publish(ip, 8080);
        }

        CacheAwareService cache = mock(CacheAwareService.class);
        when(cache.findMatchingEngines(any(), any(), any())).thenReturn(Map.of());
        strategy = new CostBasedPrefillStrategy(
                new WorkerDirectory(registry), cache,
                mock(EngineHealthReporter.class));

        Request request = new Request();
        request.setSeqLen(1_000L);
        request.setPriority(50);
        request.setBlockCacheKeys(List.of());
        context = new BalanceContext();
        context.setRequest(request);
        context.setConfig(config);
        context.setSchedulingMetadata(SchedulingMetadata.explicit(
                50, System.currentTimeMillis() + 60_000L));
    }

    @AfterEach
    void tearDown() {
        registry.close();
    }

    @Test
    void lruSpreadsConsecutiveSelectionsOverEveryPoolMember() {
        List<String> picks = new ArrayList<>();
        for (int i = 0; i < 6; i++) {
            context.getRequest().setRequestId(30_001L + i);
            try (SelectedRole selected = select()) {
                picks.add(selected.serverStatus().getServerIp());
            }
        }

        Map<String, Integer> histogram = new LinkedHashMap<>();
        for (String ip : picks) {
            histogram.merge(ip, 1, Integer::sum);
        }
        assertEquals(
                Map.of("10.0.0.1", 2, "10.0.0.2", 2, "10.0.0.3", 2),
                histogram,
                "six selections over three equal endpoints must land twice each, "
                        + "actual order " + picks);
    }

    private SelectedRole select() {
        PlacementResult<SelectedRole, RoleType> result =
                strategy.selectForQueue(context, RoleType.PREFILL, null);
        assertEquals(PlacementResult.Status.SUCCESS, result.status());
        return result.value();
    }

    private void publish(String ip, int port) {
        WorkerStatus status = StrategyTestSupport.workerStatus(
                RoleType.PREFILL, null, ip, port, port + 1,
                true, 1_000_000L, 1_000_000L);
        StrategyTestSupport.publishEndpoint(
                registry, RoleType.PREFILL, ip + ":" + port, status);
    }
}
