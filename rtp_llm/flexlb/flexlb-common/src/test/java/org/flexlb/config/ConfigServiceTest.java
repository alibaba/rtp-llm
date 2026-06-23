package org.flexlb.config;

import org.junit.jupiter.api.Test;
import uk.org.webcompere.systemstubs.environment.EnvironmentVariables;

import static org.junit.jupiter.api.Assertions.assertEquals;

class ConfigServiceTest {
    @Test
    void shouldUseTenMinuteDefaultChatStickyTtl() {
        FlexlbConfig config = new FlexlbConfig();

        assertEquals(600_000L, config.getChatStickyTtlMs());
    }

    @Test
    void shouldOverrideChatStickyTtlFromEnvironment() throws Exception {
        new EnvironmentVariables("CHAT_STICKY_TTL_MS", "120000")
                .execute(
                        () -> {
                            ConfigService configService = new ConfigService();

                            assertEquals(
                                    120_000L,
                                    configService.loadBalanceConfig().getChatStickyTtlMs());
                        });
    }
}
