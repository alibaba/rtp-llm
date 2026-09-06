package org.flexlb.service.config.source;

import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.enums.BlockHashStrategyType;
import org.flexlb.enums.LogLevel;
import org.flexlb.service.config.parser.StandardConfigDocumentParser;
import org.flexlb.service.config.parser.V0ConfigDocumentParser;
import org.junit.jupiter.api.Test;
import uk.org.webcompere.systemstubs.environment.EnvironmentVariables;

import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

class EnvironmentConfigSourceTest {

    @Test
    void registersStrictConfigAndIgnoresLegacyBehaviorEnvironmentVariables() throws Exception {
        Map<String, String> environment = Map.of(
                "FLEXLB_CONFIG",
                """
                        {
                          "schemaVersion":1,
                          "scheduler":{"type":"DIRECT"},
                          "dispatcher":{"type":"NON_BATCH"},
                          "observability":{"logging":{
                            "level":"debug","stdoutEnabled":true
                          }},
                          "enableFallback":true,
                          "blockHashStrategy":"SGLANG"
                        }
                        """,
                "FLEXLB_LOG_LEVEL", "ERROR",
                "ENABLE_STDOUT_LOG", "false",
                "ENABLE_FALLBACK", "false",
                "BLOCK_HASH_STRATEGY", "VLLM",
                "MODEL_SERVICE_CONFIG",
                "{\"service_id\":\"test-service\",\"role_endpoints\":[]}");
        ConfigService configService = new EnvironmentVariables(environment).execute(() -> {
            EnvironmentConfigSource source = new EnvironmentConfigSource();
            source.initialize();

            assertThat(source.name()).isEqualTo("environment");
            assertThat(source.priority()).isEqualTo(1);
            assertThat(source.load()).isNotBlank();
            return new ConfigService(List.of(new StandardConfigDocumentParser(), new V0ConfigDocumentParser()));
        });

        FlexlbConfig config = configService.loadBalanceConfig();
        assertThat(config.isDirect()).isTrue();
        assertThat(config.getBlockHashStrategy())
                .isEqualTo(BlockHashStrategyType.SGLANG);
        assertThat(config.getObservability().getLogging().getLevel())
                .isEqualTo(LogLevel.DEBUG);
        assertThat(config.getObservability().getLogging().isStdoutEnabled()).isTrue();
        assertThat(config.isEnableFallback()).isTrue();
        assertThat(configService.modelServiceConfig().getServiceId())
                .isEqualTo("test-service");
        configService.close();
    }
}
