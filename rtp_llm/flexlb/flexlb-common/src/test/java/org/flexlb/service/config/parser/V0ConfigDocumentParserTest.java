package org.flexlb.service.config.parser;

import org.flexlb.config.ConfigSchemaVersion;
import org.flexlb.config.ConfigService;
import org.flexlb.config.KvcmCacheMatchingConfig;
import org.flexlb.config.NonBatchDispatcherConfig;
import org.flexlb.config.QueueSchedulerConfig;
import org.flexlb.config.ZookeeperConsistencyConfig;
import org.flexlb.service.config.ConfigSource;
import org.flexlb.service.config.NormalizedConfig;
import org.junit.jupiter.api.Test;
import uk.org.webcompere.systemstubs.environment.EnvironmentVariables;

import java.util.List;
import java.util.function.Consumer;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class V0ConfigDocumentParserTest {

    @Test
    void resolvesExactlyOneParserFromDocumentVersionBeforeEnvironmentVersion() throws Exception {
        new StandardConfigDocumentParser();
        new V0ConfigDocumentParser();
        assertThat(ConfigDocumentParserResolver.resolve("{\"schemaVersion\":0,\"enableQueueing\":true}").schemaVersion()).isEqualTo(ConfigSchemaVersion.V0_COMPATIBILITY);
        new EnvironmentVariables(ConfigDocumentParserResolver.CONFIG_SCHEMA_VERSION_ENV, "0").execute(() -> {
            assertThat(ConfigDocumentParserResolver.resolve("{\"schemaVersion\":1}").schemaVersion()).isEqualTo(ConfigSchemaVersion.STANDARD);
            assertThat(ConfigDocumentParserResolver.resolve("{\"enableQueueing\":true}").schemaVersion()).isEqualTo(ConfigSchemaVersion.V0_COMPATIBILITY);
        });
    }

    @Test
    void rejectsMissingV0CompatibilityDocument() {
        V0ConfigDocumentParser parser = new V0ConfigDocumentParser();
        assertThatThrownBy(() -> parser.parse(null, null)).isInstanceOf(IllegalArgumentException.class).hasMessage("V0 compatibility configuration document must not be null or blank");
        assertThatThrownBy(() -> parser.parse("   ", null)).isInstanceOf(IllegalArgumentException.class).hasMessage("V0 compatibility configuration document must not be null or blank");
    }

    @Test
    void convertsV0NacosDocumentIntoCurrentSplitContracts() {
        NormalizedConfig converted = new V0ConfigDocumentParser().parse("""
                {
                  "enableQueueing": true,
                  "cacheAffinityFirstMaxExtraWorkTokens": 8000,
                  "p2pHitDiscount": 0,
                  "flexlbSyncConsistencyConfig": {
                    "needConsistency": true,
                    "masterElectType": "ZOOKEEPER",
                    "zookeeperConfig": {"zkHost": "zk.example:2181", "zkTimeoutMs": 10000}
                  },
                  "syncRequestTimeoutMs": 500,
                  "flexlbLogLevel": "debug",
                  "fixedScheduleWorkerPermits": true,
                  "syncStatusInterval": 50,
                  "scheduleWorkerSize": 1,
                  "maxQueueSize": 200000,
                  "blockHashStrategy": "VLLM",
                  "cacheAffinityFirstOutstandingUncachedTokensThreshold": 50000,
                  "shortestTtftSimilarityThresholdRatio": 0.2,
                  "loadBalanceStrategy": "CACHE_AFFINITY_FIRST",
                  "modelServiceConfig": {
                    "service_id": "engine-service",
                    "kvcm": {
                      "p2p_host_count": 0,
                      "enabled": true,
                      "namespace": "test",
                      "local_standby": {
                        "capacity_multiplier": 1000,
                        "minimum_ttl_ms": 86400000,
                        "maximum_entries": 200000000,
                        "ttl_ms": 86400000,
                        "ttl_reduction_start_ratio": 0.99,
                        "block_size": 1152
                      },
                      "address": "kvcm.example",
                      "port": 6381,
                      "discovery": {"type": "dashscope"}
                    },
                    "optimizer": {
                      "enabled": true,
                      "address": "optimizer.example",
                      "port": 8082,
                      "discovery": {"type": "dashscope"}
                    },
                    "role_endpoints": [{
                      "group": "default",
                      "pd_fusion_endpoint": {
                        "worker_status_port": 18002,
                        "discovery": {"type": "dashscope"},
                        "protocol": "http",
                        "address": "engine.example",
                        "path": "/",
                        "multi_engine_num": 2
                      }
                    }]
                  },
                  "cacheAffinityFirstMinHitRate": 15,
                  "prefillQueueSizeThreshold": 1024,
                  "enableStdoutLog": true
                }
                """, null);

        ConfigService.register(new CompatibilitySource(converted));
        ConfigService service = new ConfigService(List.of(new StandardConfigDocumentParser(), new V0ConfigDocumentParser()));
        var behavior = service.loadBalanceConfig();
        var modelService = service.modelServiceConfig();

        assertThat(behavior.isQueue()).isTrue();
        assertThat(converted.sourceSchemaVersion()).isEqualTo(ConfigSchemaVersion.V0_COMPATIBILITY);
        assertThat(behavior.getDispatcher()).isInstanceOf(NonBatchDispatcherConfig.class);
        assertThat(((QueueSchedulerConfig) behavior.getScheduler()).getCapacity()
                .getMaxOutstandingRequestsGlobal()).isEqualTo(200000);
        assertThat(behavior.getCacheMatching()).isInstanceOf(KvcmCacheMatchingConfig.class);
        assertThat(behavior.getRouter().getRoles().getPrefill().getCacheAffinity()
                .getMaxOutstandingUncachedTokens()).isEqualTo(50000);
        assertThat(behavior.getConsistency()).isInstanceOf(ZookeeperConsistencyConfig.class);
        assertThat(((ZookeeperConsistencyConfig) behavior.getConsistency()).getConnectString())
                .isEqualTo("zk.example:2181");
        assertThat(modelService.getServiceId()).isEqualTo("engine-service");
        assertThat(modelService.getKvcm()).isNotNull();
        assertThat(modelService.getOptimizer()).isNotNull();
        service.close();
    }

    private static final class CompatibilitySource implements ConfigSource {
        private final NormalizedConfig converted;

        private CompatibilitySource(NormalizedConfig converted) {
            this.converted = converted;
        }

        @Override
        public String name() {
            return "V0 compatibility";
        }

        @Override
        public int priority() {
            return 2;
        }

        @Override
        public void setUpdateListener(Consumer<String> listener) {}

        @Override
        public String load() {
            return converted.flexlbConfig();
        }

        @Override
        public String loadModelServiceConfig() {
            return converted.modelServiceConfig();
        }

        @Override
        public NormalizedConfig loadConfig() {
            return converted;
        }
    }
}
