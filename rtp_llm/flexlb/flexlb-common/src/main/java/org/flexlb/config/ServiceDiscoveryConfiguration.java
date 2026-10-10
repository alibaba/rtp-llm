package org.flexlb.config;

import org.flexlb.discovery.RoutingServiceDiscovery;
import org.flexlb.discovery.ServiceDiscoveryProvider;
import org.flexlb.discovery.StaticServiceDiscoveryProvider;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.util.List;

@Configuration
public class ServiceDiscoveryConfiguration {

    @Bean(destroyMethod = "")
    public ServiceDiscoveryProvider staticEnvironmentServiceDiscoveryProvider() {
        return new StaticServiceDiscoveryProvider();
    }

    @Bean(destroyMethod = "shutdown")
    public RoutingServiceDiscovery routingServiceDiscovery(
            List<ServiceDiscoveryProvider> providers,
            ObjectProvider<ConfigService> configServiceProvider) {
        ServiceDiscoveryRuntimeConfig defaults = new ServiceDiscoveryRuntimeConfig();
        return new RoutingServiceDiscovery(providers, () -> {
            ConfigService configService = configServiceProvider.getIfAvailable();
            return configService == null
                    ? defaults
                    : configService.loadBalanceConfig().getServiceDiscovery();
        });
    }
}
