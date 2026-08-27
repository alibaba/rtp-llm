package org.flexlb.service.config;

import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.enums.BlockHashStrategyType;
import org.flexlb.enums.LogLevel;
import org.junit.jupiter.api.Test;
import uk.org.webcompere.systemstubs.environment.EnvironmentVariables;

import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

class EnvironmentConfigSourceTest {

    @Test
    void registersStrictConfigAndAppliesSupportedCompatibilityOverrides() throws Exception {
        Map<String, String> environment = Map.of(
                "FLEXLB_CONFIG",
                """
                        {
                          "scheduler":{"type":"DIRECT"},
                          "dispatcher":{"type":"NON_BATCH"}
                        }
                        """,
                "FLEXLB_LOG_LEVEL", "DEBUG",
                "ENABLE_STDOUT_LOG", "true",
                "ENABLE_FALLBACK", "true",
                "BLOCK_HASH_STRATEGY", "SGLANG",
                "MODEL_SERVICE_CONFIG",
                "{\"service_id\":\"test-service\",\"role_endpoints\":[]}");
        ConfigService configService = new EnvironmentVariables(environment).execute(() -> {
            EnvironmentConfigSource source = new EnvironmentConfigSource();
            source.initialize();

            assertThat(source.name()).isEqualTo("environment");
            assertThat(source.priority()).isEqualTo(1);
            assertThat(source.load()).isNotBlank();
            return new ConfigService();
        });

        FlexlbConfig config = configService.loadBalanceConfig();
        assertThat(config.isDirect()).isTrue();
        assertThat(config.getBlockHashStrategy())
                .isEqualTo(BlockHashStrategyType.SGLANG);
        assertThat(config.getObservability().getLogging().getLevel())
                .isEqualTo(LogLevel.DEBUG);
        assertThat(config.getObservability().getLogging().isStdoutEnabled()).isTrue();
        assertThat(config.isEnableFallback()).isTrue();
        assertThat(configService.loadModelServiceConfig().getServiceId())
                .isEqualTo("test-service");
        configService.close();
    }
}
