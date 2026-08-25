package org.flexlb.config;

import org.flexlb.discovery.RoutingServiceDiscovery;
import org.flexlb.discovery.ServiceDiscovery;
import org.flexlb.discovery.StaticServiceDiscoveryProvider;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;

import static org.assertj.core.api.Assertions.assertThat;

class ServiceDiscoveryConfigurationTest {

    private final ApplicationContextRunner contextRunner = new ApplicationContextRunner()
            .withUserConfiguration(ServiceDiscoveryConfiguration.class);

    @Test
    void createsRouterAndStaticProviderWithoutGlobalStrategy() {
        contextRunner.run(context -> {
            assertThat(context).hasSingleBean(ServiceDiscovery.class);
            assertThat(context).hasSingleBean(RoutingServiceDiscovery.class);
            assertThat(context).hasSingleBean(StaticServiceDiscoveryProvider.class);
        });
    }
}
