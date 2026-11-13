package org.flexlb.config.service;

import org.flexlb.config.ConfigService;
import org.flexlb.util.discovery.DiscoveryClientSettings;
import org.flexlb.util.discovery.RoutingServiceDiscovery;
import org.flexlb.util.discovery.ServiceDiscoveryProvider;
import org.flexlb.util.discovery.StaticServiceDiscoveryProvider;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.util.List;

/**
 * Builds the discovery router and providers, leaving provider shutdown with their owning beans.
 */
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
        DiscoveryClientSettings defaults = new DiscoveryClientSettings();
        return new RoutingServiceDiscovery(
                providers,
                () -> {
                    ConfigService configService = configServiceProvider.getIfAvailable();
                    return configService == null
                            ? defaults
                            : configService.loadBalanceConfig().getServiceDiscovery();
                });
    }
}
