package org.flexlb.config;

import org.flexlb.discovery.RoutingServiceDiscovery;
import org.flexlb.discovery.ServiceDiscoveryProvider;
import org.flexlb.discovery.StaticServiceDiscoveryProvider;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.util.List;

/**
 * ServiceDiscoveryConfiguration - Service discovery default configuration
 *
 * @author saichen.sm
 */
@Configuration
public class ServiceDiscoveryConfiguration {

    @Bean(destroyMethod = "")
    public StaticServiceDiscoveryProvider staticServiceDiscoveryProvider() {
        return new StaticServiceDiscoveryProvider();
    }

    @Bean(destroyMethod = "shutdown")
    public RoutingServiceDiscovery routingServiceDiscovery(List<ServiceDiscoveryProvider> providers) {
        return new RoutingServiceDiscovery(providers);
    }
}
