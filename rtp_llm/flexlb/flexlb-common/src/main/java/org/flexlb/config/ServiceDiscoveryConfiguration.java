package org.flexlb.config;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.flexlb.discovery.FileServiceDiscovery;
import org.flexlb.discovery.RoutingServiceDiscovery;
import org.flexlb.discovery.ServiceDiscoveryProvider;
import org.flexlb.discovery.StaticServiceDiscoveryProvider;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.env.Environment;

import java.util.List;

/**
 * ServiceDiscoveryConfiguration - Service discovery default configuration
 *
 * @author saichen.sm
 */
@Slf4j
@Configuration
public class ServiceDiscoveryConfiguration {

    /**
     * Provides the implementation for {@code static-env} endpoints. A discovery
     * file replaces the embedded host list when FLEXLB_DISCOVERY_FILE is set;
     * all other endpoint discovery types continue through their own providers.
     */
    @Bean(destroyMethod = "")
    public ServiceDiscoveryProvider staticEnvironmentServiceDiscoveryProvider(Environment environment) {
        String discoveryFile = environment.getProperty("flexlb.discovery.file");
        if (StringUtils.isNotBlank(discoveryFile)) {
            log.info("Creating file-backed static-env discovery provider: {}", discoveryFile);
            return new FileServiceDiscovery(discoveryFile);
        }
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
