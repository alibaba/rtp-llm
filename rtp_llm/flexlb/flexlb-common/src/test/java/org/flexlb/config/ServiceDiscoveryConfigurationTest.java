package org.flexlb.config;

import org.flexlb.discovery.ServiceDiscovery;
import org.flexlb.discovery.StaticEnvironmentServiceDiscovery;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;

import static org.assertj.core.api.Assertions.assertThat;

class ServiceDiscoveryConfigurationTest {

    private final ApplicationContextRunner contextRunner = new ApplicationContextRunner()
            .withUserConfiguration(ServiceDiscoveryConfiguration.class);

    @Test
    void doesNotCreateDiscoveryWhenStrategyIsMissing() {
        contextRunner.run(context -> assertThat(context).doesNotHaveBean(ServiceDiscovery.class));
    }

    @Test
    void usesStaticEnvironmentDiscoveryWhenExplicitlySelected() {
        contextRunner
                .withPropertyValues("flexlb.discovery.type=static-env")
                .run(context -> assertThat(context).hasSingleBean(ServiceDiscovery.class));
    }

    @Test
    void doesNotSilentlyFallBackForAnotherStrategy() {
        contextRunner
                .withPropertyValues("flexlb.discovery.type=dashscope")
                .run(context -> assertThat(context).doesNotHaveBean(ServiceDiscovery.class));
    }
}
