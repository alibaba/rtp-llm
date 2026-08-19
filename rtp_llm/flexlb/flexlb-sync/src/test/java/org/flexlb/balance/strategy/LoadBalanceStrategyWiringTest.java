package org.flexlb.balance.strategy;

import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;

class LoadBalanceStrategyWiringTest {

    private final ApplicationContextRunner runner = new ApplicationContextRunner()
            .withBean(EngineWorkerStatus.class, () -> mock(EngineWorkerStatus.class))
            .withBean(CacheAwareService.class, () -> mock(CacheAwareService.class))
            .withBean(ResourceMeasureFactory.class, () -> mock(ResourceMeasureFactory.class))
            .withBean(EngineHealthReporter.class, () -> mock(EngineHealthReporter.class))
            .withBean("shortestTtftStrategy", ShortestTTFTStrategy.class)
            .withBean("costBasedPrefillStrategy", CostBasedPrefillStrategy.class);

    @Test
    void createsBothAffinityCapableBaselineStrategies() {
        runner.run(context -> {
            assertThat(context).hasNotFailed();
            assertThat(context).hasBean("shortestTtftStrategy");
            assertThat(context).hasBean("costBasedPrefillStrategy");
            assertThat(LoadBalanceStrategyFactory.getLoadBalanceStrategy(
                    LoadBalanceStrategyEnum.SHORTEST_TTFT))
                    .isSameAs(context.getBean("shortestTtftStrategy"));
            assertThat(LoadBalanceStrategyFactory.getLoadBalanceStrategy(
                    LoadBalanceStrategyEnum.COST_BASED_PREFILL))
                    .isSameAs(context.getBean("costBasedPrefillStrategy"));
        });
    }
}
