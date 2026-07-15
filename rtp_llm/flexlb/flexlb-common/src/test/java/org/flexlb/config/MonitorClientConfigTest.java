package org.flexlb.config;

import org.flexlb.metric.FlexMonitor;
import org.flexlb.metric.NoOpFlexMonitor;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;

import static org.assertj.core.api.Assertions.assertThat;

class MonitorClientConfigTest {

    private final ApplicationContextRunner contextRunner = new ApplicationContextRunner()
            .withUserConfiguration(MonitorClientConfig.class);

    @Test
    void alwaysProvidesNonPrimaryNoOpMonitor() {
        contextRunner.run(context -> {
            assertThat(context).hasNotFailed();
            assertThat(context).hasSingleBean(FlexMonitor.class);
            assertThat(context.getBean(FlexMonitor.class)).isSameAs(NoOpFlexMonitor.getInstance());
            assertThat(context.getBeanFactory().getBeanDefinition("flexMonitor").isPrimary()).isFalse();
        });
    }

    @Test
    void keepsDefaultNoOpAlongsideProviderMonitor() {
        contextRunner.withBean("providerMonitor", FlexMonitor.class, NoOpFlexMonitor::getInstance)
                .run(context -> {
                    assertThat(context).hasNotFailed();
                    assertThat(context).hasBean("flexMonitor");
                    assertThat(context).hasBean("providerMonitor");
                });
    }
}
