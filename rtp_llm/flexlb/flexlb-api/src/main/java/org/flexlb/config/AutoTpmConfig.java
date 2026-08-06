package org.flexlb.config;

import org.flexlb.autotpm.PriorityNormalizer;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * Auto-TPM bean configuration.
 */
@Configuration
public class AutoTpmConfig {

    @Bean
    public PriorityNormalizer priorityNormalizer(ConfigService configService) {
        return new PriorityNormalizer(configService.loadBalanceConfig());
    }
}
