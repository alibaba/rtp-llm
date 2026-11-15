package org.flexlb.balance.schedule;

import org.flexlb.config.ConfigService;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;

import static org.junit.jupiter.api.Assertions.assertSame;
import static org.mockito.Mockito.mock;

/**
 * Independent implementations exercise the public extension points and executor authority.
 */
class CustomDecodePreemptionContractTest {
    @Test
    void customServiceReplacesDefaultWithoutConstructingExecutorDependencies() {
        var config = SchedulingTestConfig.newConfig();
        RequestPreemption service = RequestPreemptionTestSupport.mock();
        new ApplicationContextRunner()
                .withUserConfiguration(SchedulerConfiguration.class)
                .withBean(
                        "environmentConfigSource",
                        org.flexlb.config.loading.source.ConfigSource.class,
                        () -> mock(org.flexlb.config.loading.source.ConfigSource.class))
                .withBean(
                        "nacosConfigSource",
                        org.flexlb.config.loading.source.ConfigSource.class,
                        () -> mock(org.flexlb.config.loading.source.ConfigSource.class))
                .withBean(
                        "uniConfigConfigSource",
                        org.flexlb.config.loading.source.ConfigSource.class,
                        () -> mock(org.flexlb.config.loading.source.ConfigSource.class))
                .withBean(ConfigService.class, () -> SchedulingTestConfig.configService(config))
                .withBean(RequestScheduler.class, () -> mock(RequestScheduler.class))
                .withBean(RequestPreemption.class, () -> service)
                .run(
                        context -> {
                            org.assertj.core.api.Assertions.assertThat(context)
                                    .hasSingleBean(RequestPreemption.class)
                                    .doesNotHaveBean(DefaultRequestPreemption.class);
                            assertSame(service, context.getBean(RequestPreemption.class));
                        });
    }
}
