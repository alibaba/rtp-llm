package org.flexlb.config;

import io.micrometer.core.instrument.simple.SimpleMeterRegistry;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.metric.MicrometerFlexMonitor;
import org.flexlb.metric.NoOpFlexMonitor;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;

import static org.assertj.core.api.Assertions.assertThat;

class MonitorClientConfigTest {

    private final ApplicationContextRunner contextRunner = new ApplicationContextRunner()
            .withUserConfiguration(MonitorClientConfig.class);

    @Test
    void providesMicrometerMonitorWhenRegistryIsAvailable() {
        contextRunner.withBean(SimpleMeterRegistry.class, SimpleMeterRegistry::new)
                .run(context -> {
                    assertThat(context).hasNotFailed();
                    assertThat(context).hasSingleBean(FlexMonitor.class);
                    assertThat(context.getBean(FlexMonitor.class))
                            .isInstanceOf(MicrometerFlexMonitor.class);
                });
    }

    @Test
    void backsOffWhenProviderMonitorAlreadyExists() {
        contextRunner.withBean(
                        "providerMonitor",
                        FlexMonitor.class,
                        NoOpFlexMonitor::getInstance)
                .run(context -> {
                    assertThat(context).hasNotFailed();
                    assertThat(context).hasSingleBean(FlexMonitor.class);
                    assertThat(context).hasBean("providerMonitor");
                    assertThat(context).doesNotHaveBean("micrometerFlexMonitor");
                    assertThat(context).doesNotHaveBean("flexMonitor");
                });
    }
}
