package org.flexlb.cache.hash;

import org.flexlb.config.BlockHashConfig;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.enums.BlockHashStrategyType;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class BlockHashStrategyConfiguration {

    @Bean
    public BlockHashStrategy blockHashStrategy(ConfigService configService) {
        return switch (blockHashStrategyType(configService.loadBalanceConfig())) {
            case VLLM -> new VllmBlockHashStrategy(configService);
            case SGLANG -> new SglangBlockHashStrategy();
        };
    }

    private BlockHashStrategyType blockHashStrategyType(FlexlbConfig config) {
        BlockHashConfig blockHashConfig = config.getBlockHashConfig();
        return blockHashConfig != null && blockHashConfig.getType() != null
                ? blockHashConfig.getType()
                : config.getBlockHashStrategy();
    }
}
