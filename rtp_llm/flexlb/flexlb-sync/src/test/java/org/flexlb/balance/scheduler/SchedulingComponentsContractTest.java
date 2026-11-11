package org.flexlb.balance.scheduler;

import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.preemption.DecodePreemption;
import org.flexlb.balance.preemption.DecodePreemptionPolicy;
import org.flexlb.balance.strategy.VitSelectionStrategy;
import org.flexlb.config.ConfigService;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Checks that supplied Spring components replace defaults without instantiating their unused dependencies.
 */
class SchedulingComponentsContractTest {
    @Test
    void suppliedComponentsReplaceDefaultsWithoutPrimaryOrDefaultDependencies() {
        var config = mock(ConfigService.class);
        when(config.loadBalanceConfig()).thenReturn(SchedulingTestConfig.batchConfig());
        var scheduler = mock(RequestScheduler.class);
        var queue = new ListSchedulingQueue();
        var grouping = mock(GroupPlanner.class);
        org.flexlb.balance.strategy.PrefillSelectionStrategy prefill =
                mock(
                        org.flexlb.balance.strategy.PrefillSelectionStrategy.class,
                        org.mockito.Mockito.CALLS_REAL_METHODS);
        org.flexlb.balance.strategy.DecodeSelectionStrategy decode =
                (request, metadata, group, endpoints, capacity, refresh, register) -> {
                    throw new AssertionError("assembly must not run selection");
                };
        VitSelectionStrategy vit = addresses -> 0;
        var preemption = mock(DecodePreemption.class);
        var requestsToPreempt = mock(DecodePreemptionPolicy.class);
        new ApplicationContextRunner()
                .withUserConfiguration(SchedulerConfiguration.class)
                .withBean(
                        "environmentConfigSource",
                        org.flexlb.service.config.ConfigSource.class,
                        () -> mock(org.flexlb.service.config.ConfigSource.class))
                .withBean(
                        "nacosConfigSource",
                        org.flexlb.service.config.ConfigSource.class,
                        () -> mock(org.flexlb.service.config.ConfigSource.class))
                .withBean(
                        "uniConfigConfigSource",
                        org.flexlb.service.config.ConfigSource.class,
                        () -> mock(org.flexlb.service.config.ConfigSource.class))
                .withBean(ConfigService.class, () -> config)
                .withBean(RequestScheduler.class, () -> scheduler)
                .withBean(SchedulingQueue.class, () -> queue)
                .withBean(GroupPlanner.class, () -> grouping)
                .withBean(org.flexlb.balance.strategy.PrefillSelectionStrategy.class, () -> prefill)
                .withBean(org.flexlb.balance.strategy.DecodeSelectionStrategy.class, () -> decode)
                .withBean(VitSelectionStrategy.class, () -> vit)
                .withBean(DecodePreemption.class, () -> preemption)
                .withBean(DecodePreemptionPolicy.class, () -> requestsToPreempt)
                .run(
                        context -> {
                            assertThat(context).hasNotFailed();
                            assertThat(context.getBeansOfType(RequestScheduler.class).values())
                                    .containsExactly(scheduler);
                            assertThat(context.getBeansOfType(SchedulingQueue.class).values())
                                    .containsExactly(queue);
                            assertThat(context.getBeansOfType(GroupPlanner.class).values())
                                    .containsExactly(grouping);
                            assertThat(
                                            context.getBeansOfType(
                                                            org.flexlb.balance.strategy
                                                                    .PrefillSelectionStrategy.class)
                                                    .values())
                                    .containsExactly(prefill);
                            assertThat(
                                            context.getBeansOfType(
                                                            org.flexlb.balance.strategy
                                                                    .DecodeSelectionStrategy.class)
                                                    .values())
                                    .containsExactly(decode);
                            assertThat(context.getBeansOfType(VitSelectionStrategy.class).values())
                                    .containsExactly(vit);
                            assertThat(context.getBeansOfType(DecodePreemption.class).values())
                                    .containsExactly(preemption);
                            assertThat(
                                            context.getBeansOfType(DecodePreemptionPolicy.class)
                                                    .values())
                                    .containsExactly(requestsToPreempt);
                        });
    }
}
