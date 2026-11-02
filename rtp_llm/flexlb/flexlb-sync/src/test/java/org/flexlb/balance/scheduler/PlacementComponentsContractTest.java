package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillRequestQueue;
import org.flexlb.balance.eviction.DecodePreemption;
import org.flexlb.balance.eviction.DecodePreemptionPolicy;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.strategy.DecodeSelectionStrategy;
import org.flexlb.balance.strategy.PrefillSelectionStrategy;
import org.flexlb.balance.strategy.VitSelectionStrategy;
import org.flexlb.config.ConfigService;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class PlacementComponentsContractTest {
    @Test
    void suppliedComponentsReplaceDefaultsWithoutPrimaryOrDefaultDependencies() {
        var config = mock(ConfigService.class);
        when(config.loadBalanceConfig()).thenReturn(SchedulingTestConfig.batchConfig());
        var scheduler = mock(RequestScheduler.class);
        var queue = new ListPlacementQueue();
        var workers = mock(PrefillRequestQueue.Factory.class);
        var grouping = mock(GroupPlanner.class);
        PrefillSelectionStrategy prefill = mock(PrefillSelectionStrategy.class);
        DecodeSelectionStrategy decode =
                (request, metadata, group, endpoints, capacity) -> {
                    throw new AssertionError("assembly must not run selection");
                };
        VitSelectionStrategy vit = addresses -> 0;
        var preemption = mock(DecodePreemption.class);
        var victims = mock(DecodePreemptionPolicy.class);
        new ApplicationContextRunner()
                .withUserConfiguration(PlacementConfiguration.class)
                .withBean(ConfigService.class, () -> config)
                .withBean(RequestScheduler.class, () -> scheduler)
                .withBean(PlacementQueue.class, () -> queue)
                .withBean(PrefillRequestQueue.Factory.class, () -> workers)
                .withBean(GroupPlanner.class, () -> grouping)
                .withBean(PrefillSelectionStrategy.class, () -> prefill)
                .withBean(DecodeSelectionStrategy.class, () -> decode)
                .withBean(VitSelectionStrategy.class, () -> vit)
                .withBean(DecodePreemption.class, () -> preemption)
                .withBean(DecodePreemptionPolicy.class, () -> victims)
                .run(
                        context -> {
                            assertThat(context).hasNotFailed();
                            assertThat(context.getBeansOfType(RequestScheduler.class).values())
                                    .containsExactly(scheduler);
                            assertThat(context.getBeansOfType(PlacementQueue.class).values())
                                    .containsExactly(queue);
                            assertThat(
                                            context.getBeansOfType(
                                                            PrefillRequestQueue.Factory.class)
                                                    .values())
                                    .containsExactly(workers);
                            assertThat(context.getBeansOfType(GroupPlanner.class).values())
                                    .containsExactly(grouping);
                            assertThat(
                                            context.getBeansOfType(PrefillSelectionStrategy.class)
                                                    .values())
                                    .containsExactly(prefill);
                            assertThat(
                                            context.getBeansOfType(DecodeSelectionStrategy.class)
                                                    .values())
                                    .containsExactly(decode);
                            assertThat(context.getBeansOfType(VitSelectionStrategy.class).values())
                                    .containsExactly(vit);
                            assertThat(context.getBeansOfType(DecodePreemption.class).values())
                                    .containsExactly(preemption);
                            assertThat(
                                            context.getBeansOfType(DecodePreemptionPolicy.class)
                                                    .values())
                                    .containsExactly(victims);
                        });
    }
}
