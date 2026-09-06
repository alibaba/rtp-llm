package org.flexlb.cache.hash;

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
    void defaultsToVllmAndSelectsSglangFromFlexlbConfig() {
        BlockHashStrategyConfiguration configuration = new BlockHashStrategyConfiguration();
        ConfigService configService = mock(ConfigService.class);
        FlexlbConfig flexlbConfig = new FlexlbConfig();
        when(configService.loadBalanceConfig()).thenReturn(flexlbConfig);

        assertInstanceOf(
                VllmBlockHashStrategy.class,
                configuration.blockHashStrategy(configService));

        flexlbConfig.setBlockHashStrategy(BlockHashStrategyType.SGLANG);
        assertInstanceOf(
                SglangBlockHashStrategy.class,
                configuration.blockHashStrategy(configService));
    }

    @Test
    void parsesSglangStrategyFromFlexlbConfigJson() {
        FlexlbConfig config = JsonUtils.toObject(
                "{\"blockHashStrategy\":\"SGLANG\"}",
                FlexlbConfig.class);

        assertEquals(BlockHashStrategyType.SGLANG, config.getBlockHashStrategy());
    }
}
