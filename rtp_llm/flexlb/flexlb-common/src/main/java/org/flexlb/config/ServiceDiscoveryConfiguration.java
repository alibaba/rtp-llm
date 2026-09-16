package org.flexlb.config;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.flexlb.discovery.FileServiceDiscovery;
import org.flexlb.discovery.NoOpServiceDiscovery;
import org.flexlb.discovery.ServiceDiscovery;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.env.Environment;

/**
 * ServiceDiscoveryConfiguration - Service discovery default configuration
 *
 * @author saichen.sm
 */
@Slf4j
@Configuration
public class ServiceDiscoveryConfiguration {
    /**
     * Create the default ServiceDiscovery Bean
     * Used when no other ServiceDiscovery implementation is available
     *
     * <p>Selection inside the fallback (explicit, no bean-ordering tricks):
     * <ul>
     *   <li>{@code flexlb.discovery.file} set (env {@code FLEXLB_DISCOVERY_FILE})
     *       → {@link FileServiceDiscovery}, a small JSON file re-read on every
     *       lookup so the master's 20ms sync loop picks up dynamic scale
     *       changes without restart.</li>
     *   <li>Otherwise → {@link NoOpServiceDiscovery} (immutable env vars).</li>
     * </ul>
     *
     * @return ServiceDiscovery instance
     */
    @Bean
    @ConditionalOnMissingBean(ServiceDiscovery.class)
    public ServiceDiscovery serviceDiscovery(Environment environment) {
        String discoveryFile = environment.getProperty("flexlb.discovery.file");
        if (StringUtils.isNotBlank(discoveryFile)) {
            log.info("Creating FileServiceDiscovery (file-based dynamic discovery): {}", discoveryFile);
            return new FileServiceDiscovery(discoveryFile);
        }
        log.info("Creating default NoOpServiceDiscovery (env-based discovery)");
        return NoOpServiceDiscovery.getInstance();
    }
}