package org.flexlb.config;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.discovery.ServiceDiscovery;
import org.flexlb.discovery.StaticEnvironmentServiceDiscovery;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * ServiceDiscoveryConfiguration - Service discovery default configuration
 *
 * @author saichen.sm
 */
@Slf4j
@Configuration
public class ServiceDiscoveryConfiguration {
    /**
     * Create default service discovery backed by static environment variables.
     *
     * @return StaticEnvironmentServiceDiscovery instance
     */
    @Bean
    @ConditionalOnMissingBean(ServiceDiscovery.class)
    @ConditionalOnProperty(
            prefix = "flexlb.discovery",
            name = "type",
            havingValue = "static-env")
    public ServiceDiscovery staticEnvironmentServiceDiscovery() {
        log.info("Creating StaticEnvironmentServiceDiscovery (strategy=static-env)");
        return StaticEnvironmentServiceDiscovery.getInstance();
    }
}
