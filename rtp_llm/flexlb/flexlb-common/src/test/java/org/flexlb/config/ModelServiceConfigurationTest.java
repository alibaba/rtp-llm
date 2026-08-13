package org.flexlb.config;

import org.flexlb.dao.route.OptimizerConfig;
import org.flexlb.discovery.ServiceDiscoveryType;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;

import static org.assertj.core.api.Assertions.assertThat;

class ModelServiceConfigurationTest {

    private final ApplicationContextRunner contextRunner = new ApplicationContextRunner()
            .withUserConfiguration(
                    ServiceDiscoveryConfiguration.class,
                    ModelServiceConfiguration.class);

    @Test
    void loadsAndValidatesEndpointDiscoveryConfiguration() {
        contextRunner
                .withPropertyValues("MODEL_SERVICE_CONFIG=" + staticModelConfig())
                .run(context -> {
                    assertThat(context).hasNotFailed();
                    ModelMetaConfig config = context.getBean(ModelMetaConfig.class);
                    var endpoint = config.getServiceRoute("test-service").getAllEndpoints().getFirst();
                    assertThat(endpoint.getDiscovery().getType()).isEqualTo(ServiceDiscoveryType.STATIC_ENV);
                    assertThat(endpoint.getDiscovery().getHosts()).containsExactly("127.0.0.1:8080");
                    assertThat(endpoint.getWorkerStatusPort()).isEqualTo(18002);
                });
    }

    @Test
    void rejectsEndpointWithoutDiscovery() {
        String config = """
                {"service_id":"test-service","role_endpoints":[{"group":"default",
                "pd_fusion_endpoint":{"address":"service-a","protocol":"http","path":"/"}}]}
                """;

        contextRunner
                .withPropertyValues("MODEL_SERVICE_CONFIG=" + config)
                .run(context -> {
                    assertThat(context).hasFailed();
                    assertThat(context.getStartupFailure())
                            .hasRootCauseMessage("endpoint discovery must be configured for address: service-a");
                });
    }

    @Test
    void rejectsDiscoveryTypeWithoutProvider() {
        String config = """
                {"service_id":"test-service","role_endpoints":[{"group":"default",
                "pd_fusion_endpoint":{"address":"v-test","protocol":"http","path":"/",
                "discovery":{"type":"dashscope","base_url":"http://127.0.0.1:8880"}}}]}
                """;

        contextRunner
                .withPropertyValues("MODEL_SERVICE_CONFIG=" + config)
                .run(context -> {
                    assertThat(context).hasFailed();
                    assertThat(context.getStartupFailure())
                            .hasRootCauseMessage(
                                    "No service discovery provider available for type: dashscope, address: v-test");
                });
    }

    @Test
    void loadsEnabledKvcmConfiguration() {
        String config = """
                {"service_id":"test-service","kvcm":{"enabled":true,"address":"kvcm-service",
                "port":7381,
                "discovery":{"type":"static-env","hosts":["127.0.0.1:8080"]}},
                "role_endpoints":[{"group":"default",
                "pd_fusion_endpoint":{"address":"service-a","protocol":"http","path":"/",
                "discovery":{"type":"static-env","hosts":["127.0.0.1:8080"]}}}]}
                """;

        contextRunner
                .withPropertyValues("MODEL_SERVICE_CONFIG=" + config)
                .run(context -> {
                    assertThat(context).hasNotFailed();
                    var route = context.getBean(ModelMetaConfig.class).getServiceRoute("test-service");
                    assertThat(route.isKvcmEnabled()).isTrue();
                    assertThat(route.getKvcm().getPort()).isEqualTo(7381);
                    assertThat(route.getAllEndpoints()).hasSize(1);
                });
    }

    @Test
    void loadsEnabledOptimizerTraceConfiguration() {
        contextRunner
                .withPropertyValues("MODEL_SERVICE_CONFIG=" + modelConfig(validOptimizerConfig()))
                .run(context -> {
                    assertThat(context).hasNotFailed();
                    OptimizerConfig optimizer = context.getBean(ModelMetaConfig.class)
                            .getServiceRoute("test-service")
                            .getOptimizer();
                    assertThat(optimizer.isEnabled()).isTrue();
                    assertThat(optimizer.getAddress()).isEqualTo("optimizer-service");
                    assertThat(optimizer.getPath()).isEqualTo("/custom/optimizer");
                    assertThat(optimizer.toEndpoint().getProtocol()).isEqualTo("http");
                    assertThat(optimizer.toEndpoint().getDiscovery().getType())
                            .isEqualTo(ServiceDiscoveryType.STATIC_ENV);
                });
    }

    @Test
    void appliesOptimizerTraceDefaults() {
        contextRunner
                .withPropertyValues("MODEL_SERVICE_CONFIG=" + modelConfig(
                        validOptimizerConfig().replace(",\"path\":\"/custom/optimizer\"", "")))
                .run(context -> {
                    assertThat(context).hasNotFailed();
                    assertThat(context.getBean(ModelMetaConfig.class)
                            .getServiceRoute("test-service")
                            .getOptimizer()
                            .getPath())
                            .isEqualTo(OptimizerConfig.DEFAULT_PATH);
                });
    }

    @Test
    void ignoresOptimizerFieldsWhenDisabled() {
        contextRunner
                .withPropertyValues("MODEL_SERVICE_CONFIG=" + modelConfig("{\"enabled\":false}"))
                .run(context -> assertThat(context).hasNotFailed());
    }

    @Test
    void allowsEnabledOptimizerWithoutInstanceId() {
        contextRunner
                .withPropertyValues("MODEL_SERVICE_CONFIG=" + modelConfig(validOptimizerConfig()))
                .run(context -> assertThat(context).hasNotFailed());
    }

    @Test
    void rejectsEnabledOptimizerWithInvalidPathOrDiscovery() {
        assertOptimizerRejected(
                validOptimizerConfig().replace("\"path\":\"/custom/optimizer\"",
                        "\"path\":\"custom/optimizer\""),
                "MODEL_SERVICE_CONFIG online_optimizer.path must start with '/'");
        assertOptimizerRejected(
                validOptimizerConfig().replace("\"hosts\":[\"127.0.0.1:8082\"]",
                        "\"hosts\":[]"),
                "static-env discovery hosts must be configured for address: optimizer-service");
    }

    @Test
    void loadsLocalStandbyConfiguration() {
        String config = """
                {"service_id":"test-service","kvcm":{"enabled":true,"address":"kvcm-service",
                "heartbeat_failure_threshold":4,"query_failure_threshold":5,
                "max_query_retry_count":2,"recovery_success_threshold":2,
                "discovery":{"type":"static-env","hosts":["127.0.0.1:8080"]},
                "local_standby":{"auto_switch":true,"block_size":4096,
                "ttl_ms":300000,"minimum_ttl_ms":120000,
                "ttl_reduction_start_ratio":0.75,"maximum_entries":1000000,
                "capacity_multiplier":1.3,
                "async_queue_capacity":8192,"hash_thread_count":6,
                "hash_queue_capacity":2048}},
                "role_endpoints":[{"group":"default",
                "pd_fusion_endpoint":{"address":"service-a","protocol":"http","path":"/",
                "discovery":{"type":"static-env","hosts":["127.0.0.1:8080"]}}}]}
                """;

        contextRunner
                .withPropertyValues("MODEL_SERVICE_CONFIG=" + config)
                .run(context -> {
                    assertThat(context).hasNotFailed();
                    var kvcm = context.getBean(ModelMetaConfig.class)
                            .getServiceRoute("test-service")
                            .getKvcm();
                    var standby = kvcm.getLocalStandby();
                    assertThat(standby.isAutoSwitch()).isTrue();
                    assertThat(standby.getBlockSize()).isEqualTo(4096);
                    assertThat(standby.getMinimumTtlMs()).isEqualTo(120000);
                    assertThat(standby.getTtlReductionStartRatio()).isEqualTo(0.75);
                    assertThat(standby.getCapacityMultiplier()).isEqualTo(1.3);
                    assertThat(standby.getHashThreadCount()).isEqualTo(6);
                    assertThat(standby.getHashQueueCapacity()).isEqualTo(2048);
                    assertThat(kvcm.getHeartbeatFailureThreshold()).isEqualTo(4);
                    assertThat(kvcm.getQueryFailureThreshold()).isEqualTo(5);
                    assertThat(kvcm.getMaxQueryRetryCount()).isEqualTo(2);
                    assertThat(kvcm.getRecoverySuccessThreshold()).isEqualTo(2);
                });
    }

    @Test
    void rejectsInvalidDynamicTtlConfiguration() {
        String config = """
                {"service_id":"test-service","kvcm":{"enabled":true,"address":"kvcm-service",
                "discovery":{"type":"static-env","hosts":["127.0.0.1:8080"]},
                "local_standby":{"ttl_ms":1000,"minimum_ttl_ms":2000}},
                "role_endpoints":[{"group":"default",
                "pd_fusion_endpoint":{"address":"service-a","protocol":"http","path":"/",
                "discovery":{"type":"static-env","hosts":["127.0.0.1:8080"]}}}]}
                """;

        contextRunner
                .withPropertyValues("MODEL_SERVICE_CONFIG=" + config)
                .run(context -> {
                    assertThat(context).hasFailed();
                    assertThat(context.getStartupFailure())
                            .hasRootCauseMessage(
                                    "MODEL_SERVICE_CONFIG kvcm.local_standby contains invalid values");
                });
    }

    private String staticModelConfig() {
        return """
                {"service_id":"test-service","role_endpoints":[{"group":"default",
                "pd_fusion_endpoint":{"address":"service-a","protocol":"http","path":"/",
                "worker_status_port":18002,
                "discovery":{"type":"static-env","hosts":["127.0.0.1:8080"]}}}]}
                """;
    }

    private void assertOptimizerRejected(String optimizerConfig, String message) {
        contextRunner
                .withPropertyValues("MODEL_SERVICE_CONFIG=" + modelConfig(optimizerConfig))
                .run(context -> {
                    assertThat(context).hasFailed();
                    assertThat(context.getStartupFailure()).hasRootCauseMessage(message);
                });
    }

    private String modelConfig(String optimizerConfig) {
        return """
                {"service_id":"test-service","optimizer":%s,
                "role_endpoints":[{"group":"default",
                "pd_fusion_endpoint":{"address":"service-a","protocol":"http","path":"/",
                "discovery":{"type":"static-env","hosts":["127.0.0.1:8080"]}}}]}
                """.formatted(optimizerConfig);
    }

    private String validOptimizerConfig() {
        return """
                {"enabled":true,"address":"optimizer-service","path":"/custom/optimizer",
                "discovery":{"type":"static-env","hosts":["127.0.0.1:8082"]}}
                """;
    }
}
