package org.flexlb.cache.hash;

import org.flexlb.config.ConfigService;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class BlockHashStrategyConfiguration {

    @Bean
    public BlockHashStrategy blockHashStrategy(ConfigService configService) {
        return switch (configService.loadBalanceConfig().getBlockHashStrategy()) {
            case VLLM -> new VllmBlockHashStrategy();
            case SGLANG -> new SglangBlockHashStrategy();
        };
    }
}
