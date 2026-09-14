package org.flexlb.balance.composition;

import org.flexlb.balance.strategy.CostBasedBatchedPrefillStrategy;
import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.balance.strategy.PrefillStrategy;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.WorkerDirectory;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class PrefillStrategyBindingConfigurationTest {

    @Test
    void global_fixed_window_selects_batched_strategy_at_startup() {
        ConfigService configService = configService("""
                {"scheduler":{"globalDecision":{"type":"FIXED_WINDOW"},
                              "decision":{"type":"SINGLE"}},
                 "router":{"roles":{"prefill":{
                   "candidateChoice":{"type":"BEST_ONLY"}}}}}
                """);

        PrefillStrategy strategy = new PrefillStrategyBindingConfiguration()
                .activePrefillStrategy(configService, mock(WorkerDirectory.class),
                        mock(CacheAwareService.class), mock(EngineHealthReporter.class));

        assertInstanceOf(CostBasedBatchedPrefillStrategy.class, strategy);
    }

    @Test
    void local_decision_selects_single_strategy_at_startup() {
        ConfigService configService = configService("{}");

        PrefillStrategy strategy = new PrefillStrategyBindingConfiguration()
                .activePrefillStrategy(configService, mock(WorkerDirectory.class),
                        mock(CacheAwareService.class), mock(EngineHealthReporter.class));

        assertInstanceOf(CostBasedPrefillStrategy.class, strategy);
    }

    private static ConfigService configService(String document) {
        ConfigService configService = mock(ConfigService.class);
        FlexlbConfig config = ConfigService.parse(document);
        when(configService.loadBalanceConfig()).thenReturn(config);
        return configService;
    }
}
