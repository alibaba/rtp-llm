package org.flexlb.cache.hash;

import org.flexlb.config.BlockHashConfig;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.enums.BlockHashStrategyType;
import org.flexlb.util.JsonUtils;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class BlockHashStrategyTest {

    @Test
    void prefersBlockHashConfigTypeAndFallsBackToDeprecatedStrategy() {
        BlockHashStrategyConfiguration configuration = new BlockHashStrategyConfiguration();
        ConfigService configService = mock(ConfigService.class);
        FlexlbConfig flexlbConfig = new FlexlbConfig();
        when(configService.loadBalanceConfig()).thenReturn(flexlbConfig);

        assertInstanceOf(
                VllmBlockHashStrategy.class,
                configuration.blockHashStrategy(configService));

        BlockHashConfig blockHashConfig = new BlockHashConfig();
        blockHashConfig.setType(BlockHashStrategyType.SGLANG);
        flexlbConfig.setBlockHashConfig(blockHashConfig);
        assertInstanceOf(
                SglangBlockHashStrategy.class,
                configuration.blockHashStrategy(configService));

        blockHashConfig.setType(null);
        flexlbConfig.setBlockHashStrategy(BlockHashStrategyType.SGLANG);
        assertInstanceOf(
                SglangBlockHashStrategy.class,
                configuration.blockHashStrategy(configService));
    }

    @Test
    void parsesBlockHashConfigFromFlexlbConfigJson() {
        FlexlbConfig config = JsonUtils.toObject(
                "{\"blockHashConfig\":{\"type\":\"SGLANG\",\"hashSeed\":\"configured-seed\"}}",
                FlexlbConfig.class);

        assertEquals(BlockHashStrategyType.SGLANG, config.getBlockHashConfig().getType());
        assertEquals("configured-seed", config.getBlockHashConfig().getHashSeed());
    }
}
