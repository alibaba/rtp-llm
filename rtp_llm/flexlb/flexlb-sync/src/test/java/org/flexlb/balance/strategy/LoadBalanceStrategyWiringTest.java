package org.flexlb.balance.strategy;

import org.flexlb.balance.resource.PrefillResourceMeasure;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.WorkerDirectory;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;

class LoadBalanceStrategyWiringTest {

    private final ApplicationContextRunner runner = new ApplicationContextRunner()
            .withBean(WorkerDirectory.class, () -> mock(WorkerDirectory.class))
            .withBean(CacheAwareService.class, () -> mock(CacheAwareService.class))
            .withBean(PrefillResourceMeasure.class, () -> mock(PrefillResourceMeasure.class))
            .withBean(EngineHealthReporter.class, () -> mock(EngineHealthReporter.class))
            .withBean("shortestTtftStrategy", ShortestTTFTStrategy.class)
            .withBean("costBasedPrefillStrategy", CostBasedPrefillStrategy.class)
            .withBean(ConfiguredLoadBalanceSelector.class);

    @Test
    void createsBothAffinityCapableBaselineStrategies() {
        runner.run(context -> {
            assertThat(context).hasNotFailed();
            assertThat(context).hasBean("shortestTtftStrategy");
            assertThat(context).hasBean("costBasedPrefillStrategy");
            assertThat(context).hasSingleBean(ConfiguredLoadBalanceSelector.class);

            ShortestTTFTStrategy shortest = context.getBean(
                    "shortestTtftStrategy", ShortestTTFTStrategy.class);
            CostBasedPrefillStrategy costBased = context.getBean(
                    "costBasedPrefillStrategy", CostBasedPrefillStrategy.class);
            RoutingConfig.EstimatedTtftSelectorConfig selector =
                    new RoutingConfig.EstimatedTtftSelectorConfig();

            assertThat(costBased.supports(RoleType.PREFILL, selector)).isTrue();
            assertThat(shortest.supports(RoleType.PREFILL, selector)).isFalse();

            selector.setCandidateChoice(
                    new RoutingConfig.LeastRecentlyUsedInPoolConfig());
            assertThat(shortest.supports(RoleType.PREFILL, selector)).isTrue();
            assertThat(costBased.supports(RoleType.PREFILL, selector)).isFalse();
        });
    }
}
