package org.flexlb.service.config;

import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.enums.LogLevel;
import org.junit.jupiter.api.Test;
import uk.org.webcompere.systemstubs.environment.EnvironmentVariables;

import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

class EnvironmentConfigSourceTest {

    @Test
    void registersAndLoadsFullConfigThenAppliesIndividualEnvironmentOverrides() throws Exception {
        Map<String, String> environment = Map.of(
                "FLEXLB_CONFIG", "{\"maxRetryCount\":3,\"enableQueueing\":false}",
                "FLEXLB_LOG_LEVEL", "DEBUG",
                "FLEXLB_SYNC_CONSISTENCY_CONFIG",
                "{\"needConsistency\":true,\"zookeeperConfig\":{\"zkHost\":\"zk:2181\",\"zkTimeoutMs\":10000}}",
                "MAX_RETRY_COUNT", "4",
                "ENABLE_QUEUEING", "true");
        EnvironmentConfigSource source = new EnvironmentVariables(environment).execute(() -> {
            EnvironmentConfigSource configSource = new EnvironmentConfigSource();
            configSource.initialize();
            return configSource;
        });

        String initializedContent = source.load();
        ConfigService configService = new ConfigService();
        FlexlbConfig config = configService.loadBalanceConfig();

        assertThat(source.name()).isEqualTo("environment");
        assertThat(source.priority()).isEqualTo(1);
        assertThat(source.load()).isEqualTo(initializedContent);
        assertThat(config.getMaxRetryCount()).isEqualTo(4);
        assertThat(config.isEnableQueueing()).isTrue();
        assertThat(config.getFlexlbLogLevel()).isEqualTo(LogLevel.DEBUG);
        assertThat(config.getFlexlbSyncConsistencyConfig().isNeedConsistency()).isTrue();
        assertThat(config.getFlexlbSyncConsistencyConfig().getZookeeperConfig().getZkHost()).isEqualTo("zk:2181");
        assertThat(config.getFlexlbSyncConsistencyConfig().getZookeeperConfig().getZkTimeoutMs()).isEqualTo(10000);
        configService.close();
    }
}
