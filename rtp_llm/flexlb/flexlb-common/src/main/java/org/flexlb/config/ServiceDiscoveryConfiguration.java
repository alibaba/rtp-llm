package org.flexlb.config;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.discovery.NoOpServiceDiscovery;
import org.flexlb.discovery.ServiceDiscovery;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
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
     * Create default ServiceDiscovery Bean
     * Used when no other ServiceDiscovery implementation is available
     *
     * @return NoOpServiceDiscovery instance
     */
    @Bean
    @ConditionalOnMissingBean(ServiceDiscovery.class)
    public ServiceDiscovery serviceDiscovery() {
        log.info("Creating default NoOpServiceDiscovery (env-based discovery)");
        return NoOpServiceDiscovery.getInstance();
    }
}