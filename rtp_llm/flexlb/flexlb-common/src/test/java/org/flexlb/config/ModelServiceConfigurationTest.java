package org.flexlb.config;

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
    void loadsEnabledOnlineOptimizerConfiguration() {
        String config = """
                {"service_id":"test-service",
                "online_optimizer":{"enabled":true,"address":"optimizer-service",
                "path":"/custom/optimizer","instance_group":"test-group",
                "instance_id":"test-instance","register_timeout_ms":3000,
                "block_size":64,"linear_step":4,
                "discovery":{"type":"static-env","hosts":["127.0.0.1:8082"]},
                "location_spec_infos":[{"name":"full","size":4294967296},{"name":"linear","size":65536}],
                "location_spec_groups":[{"name":"full-group","spec_names":["full"]},
                {"name":"linear-group","spec_names":["linear"]}],
                "optimizer_state_info":{"full_location_spec_group_name":"full-group",
                "linear_location_spec_group_name":"linear-group"}},
                "role_endpoints":[{"group":"default",
                "pd_fusion_endpoint":{"address":"service-a","protocol":"http","path":"/",
                "discovery":{"type":"static-env","hosts":["127.0.0.1:8080"]}}}]}
                """;

        contextRunner
                .withPropertyValues("MODEL_SERVICE_CONFIG=" + config)
                .run(context -> {
                    assertThat(context).hasNotFailed();
                    var optimizer = context.getBean(ModelMetaConfig.class)
                            .getServiceRoute("test-service")
                            .getOnlineOptimizer();
                    assertThat(optimizer.isEnabled()).isTrue();
                    assertThat(optimizer.getAddress()).isEqualTo("optimizer-service");
                    assertThat(optimizer.getPath()).isEqualTo("/custom/optimizer");
                    assertThat(optimizer.getInstanceGroup()).isEqualTo("test-group");
                    assertThat(optimizer.getInstanceId()).isEqualTo("test-instance");
                    assertThat(optimizer.getRegisterTimeoutMs()).isEqualTo(3000);
                    assertThat(optimizer.getBlockSize()).isEqualTo(64);
                    assertThat(optimizer.getLinearStep()).isEqualTo(4);
                    assertThat(optimizer.getLocationSpecInfos().getFirst().getSize())
                            .isEqualTo(4_294_967_296L);
                    assertThat(optimizer.getLocationSpecGroups().get(1).getSpecNames())
                            .containsExactly("linear");
                    assertThat(optimizer.getOptimizerStateInfo().getFullLocationSpecGroupName())
                            .isEqualTo("full-group");
                    assertThat(optimizer.toEndpoint().getProtocol()).isEqualTo("http");
                    assertThat(optimizer.toEndpoint().getDiscovery().getType())
                            .isEqualTo(ServiceDiscoveryType.STATIC_ENV);
                });
    }

    @Test
    void appliesOnlineOptimizerDefaults() {
        String optimizerConfig = validOnlineOptimizerConfig()
                .replace(",\"path\":\"/custom/optimizer\"", "")
                .replace("\"register_timeout_ms\":3000,", "")
                .replace(",\"linear_step\":4", "");

        contextRunner
                .withPropertyValues("MODEL_SERVICE_CONFIG=" + modelConfig(optimizerConfig))
                .run(context -> {
                    assertThat(context).hasNotFailed();
                    var optimizer = context.getBean(ModelMetaConfig.class)
                            .getServiceRoute("test-service")
                            .getOnlineOptimizer();
                    assertThat(optimizer.getPath()).isEqualTo("/api/optimizer");
                    assertThat(optimizer.getRegisterTimeoutMs()).isEqualTo(5000);
                    assertThat(optimizer.getLinearStep()).isZero();
                });
    }

    @Test
    void ignoresInvalidOnlineOptimizerFieldsWhenDisabled() {
        contextRunner
                .withPropertyValues("MODEL_SERVICE_CONFIG=" + modelConfig("{\"enabled\":false}"))
                .run(context -> assertThat(context).hasNotFailed());
    }

    @Test
    void rejectsEnabledOnlineOptimizerWithoutExplicitInstanceIdentity() {
        assertOnlineOptimizerRejected(
                validOnlineOptimizerConfig().replace("\"instance_group\":\"test-group\",", ""),
                "MODEL_SERVICE_CONFIG online_optimizer.instance_group must not be blank");
        assertOnlineOptimizerRejected(
                validOnlineOptimizerConfig().replace("\"instance_id\":\"test-instance\",", ""),
                "MODEL_SERVICE_CONFIG online_optimizer.instance_id must not be blank");
    }

    @Test
    void rejectsEnabledOnlineOptimizerWithInvalidNumericFields() {
        assertOnlineOptimizerRejected(
                validOnlineOptimizerConfig().replace("\"register_timeout_ms\":3000", "\"register_timeout_ms\":0"),
                "MODEL_SERVICE_CONFIG online_optimizer.register_timeout_ms must be greater than zero");
        assertOnlineOptimizerRejected(
                validOnlineOptimizerConfig().replace("\"block_size\":64", "\"block_size\":0"),
                "MODEL_SERVICE_CONFIG online_optimizer.block_size must be greater than zero");
        assertOnlineOptimizerRejected(
                validOnlineOptimizerConfig().replace("\"linear_step\":4", "\"linear_step\":-1"),
                "MODEL_SERVICE_CONFIG online_optimizer.linear_step must not be negative");
    }

    @Test
    void rejectsEnabledOnlineOptimizerPathWithoutLeadingSlash() {
        assertOnlineOptimizerRejected(
                validOnlineOptimizerConfig().replace(
                        "\"path\":\"/custom/optimizer\"",
                        "\"path\":\"custom/optimizer\""),
                "MODEL_SERVICE_CONFIG online_optimizer.path must start with '/'");
    }

    @Test
    void rejectsEnabledOnlineOptimizerPathWithQueryOrFragment() {
        assertOnlineOptimizerRejected(
                validOnlineOptimizerConfig().replace(
                        "\"path\":\"/custom/optimizer\"",
                        "\"path\":\"/custom/optimizer?tenant=test\""),
                "MODEL_SERVICE_CONFIG online_optimizer.path must not contain query or fragment");
        assertOnlineOptimizerRejected(
                validOnlineOptimizerConfig().replace(
                        "\"path\":\"/custom/optimizer\"",
                        "\"path\":\"/custom/optimizer#fragment\""),
                "MODEL_SERVICE_CONFIG online_optimizer.path must not contain query or fragment");
    }

    @Test
    void rejectsEnabledOnlineOptimizerWithInvalidDiscoveryPollInterval() {
        assertOnlineOptimizerRejected(
                validOnlineOptimizerConfig().replace(
                        "\"type\":\"static-env\"",
                        "\"type\":\"static-env\",\"poll_interval_ms\":0"),
                "MODEL_SERVICE_CONFIG online_optimizer.discovery.poll_interval_ms "
                        + "must be greater than zero");
    }

    @Test
    void rejectsEnabledOnlineOptimizerWithoutLocationSpecs() {
        assertOnlineOptimizerRejected(
                validOnlineOptimizerConfig().replace(
                        "\"location_spec_infos\":[{\"name\":\"full\",\"size\":4294967296},"
                                + "{\"name\":\"linear\",\"size\":65536}]",
                        "\"location_spec_infos\":[]"),
                "MODEL_SERVICE_CONFIG online_optimizer.location_spec_infos must not be empty");
        assertOnlineOptimizerRejected(
                validOnlineOptimizerConfig().replaceAll(
                        "(?s)\\\"location_spec_groups\\\":\\[.*?\\],\\s*\\\"optimizer_state_info\\\"",
                        "\\\"location_spec_groups\\\":[],\\\"optimizer_state_info\\\""),
                "MODEL_SERVICE_CONFIG online_optimizer.location_spec_groups must not be empty");
    }

    @Test
    void rejectsEnabledOnlineOptimizerWithInvalidStateMapping() {
        assertOnlineOptimizerRejected(
                validOnlineOptimizerConfig().replace(
                        "\"full_location_spec_group_name\":\"full-group\"",
                        "\"full_location_spec_group_name\":\"missing-group\""),
                "MODEL_SERVICE_CONFIG online_optimizer full location spec group is not defined: missing-group");
        assertOnlineOptimizerRejected(
                validOnlineOptimizerConfig().replace(
                        "\"linear_location_spec_group_name\":\"linear-group\"",
                        "\"linear_location_spec_group_name\":\"full-group\""),
                "MODEL_SERVICE_CONFIG online_optimizer full and linear location spec groups must differ");
        assertOnlineOptimizerRejected(
                validOnlineOptimizerConfig().replace(
                        "\"name\":\"linear-group\",\"spec_names\":[\"linear\"]",
                        "\"name\":\"linear-group\",\"spec_names\":[\"full\",\"linear\"]"),
                "MODEL_SERVICE_CONFIG online_optimizer full and linear location spec groups "
                        + "must not share specs: full");
    }

    @Test
    void rejectsEnabledOnlineOptimizerWithInvalidLocationSpecEntries() {
        assertOnlineOptimizerRejected(
                validOnlineOptimizerConfig().replace(
                        "\"name\":\"full\",\"size\":4294967296",
                        "\"name\":\"\",\"size\":4294967296"),
                "MODEL_SERVICE_CONFIG online_optimizer.location_spec_infos entries require a name");
        assertOnlineOptimizerRejected(
                validOnlineOptimizerConfig().replace("\"size\":4294967296", "\"size\":0"),
                "MODEL_SERVICE_CONFIG online_optimizer.location_spec_infos sizes must be greater than zero");
        assertOnlineOptimizerRejected(
                validOnlineOptimizerConfig().replace(
                        "\"name\":\"full-group\",\"spec_names\":[\"full\"]",
                        "\"name\":\"\",\"spec_names\":[\"full\"]"),
                "MODEL_SERVICE_CONFIG online_optimizer.location_spec_groups entries require a name");
        assertOnlineOptimizerRejected(
                validOnlineOptimizerConfig().replace(
                        "\"name\":\"full-group\",\"spec_names\":[\"full\"]",
                        "\"name\":\"full-group\",\"spec_names\":[]"),
                "MODEL_SERVICE_CONFIG online_optimizer.location_spec_groups spec_names must not be empty");
    }

    @Test
    void rejectsEnabledOnlineOptimizerWithInconsistentLocationSpecs() {
        assertOnlineOptimizerRejected(
                validOnlineOptimizerConfig().replace(
                        "{\"name\":\"linear\",\"size\":65536}",
                        "{\"name\":\"full\",\"size\":65536}"),
                "MODEL_SERVICE_CONFIG online_optimizer.location_spec_infos names must be unique: full");
        assertOnlineOptimizerRejected(
                validOnlineOptimizerConfig().replace(
                        "{\"name\":\"linear-group\",\"spec_names\":[\"linear\"]}",
                        "{\"name\":\"full-group\",\"spec_names\":[\"linear\"]}"),
                "MODEL_SERVICE_CONFIG online_optimizer.location_spec_groups names must be unique: full-group");
        assertOnlineOptimizerRejected(
                validOnlineOptimizerConfig().replace(
                        "\"spec_names\":[\"linear\"]",
                        "\"spec_names\":[\"\"]"),
                "MODEL_SERVICE_CONFIG online_optimizer.location_spec_groups spec_names must not be blank");
        assertOnlineOptimizerRejected(
                validOnlineOptimizerConfig().replace(
                        "\"spec_names\":[\"linear\"]",
                        "\"spec_names\":[\"missing\"]"),
                "MODEL_SERVICE_CONFIG online_optimizer location spec is not defined: missing");
        assertOnlineOptimizerRejected(
                validOnlineOptimizerConfig().replace(
                        "\"spec_names\":[\"linear\"]",
                        "\"spec_names\":[\"linear\",\"linear\"]"),
                "MODEL_SERVICE_CONFIG online_optimizer location spec group contains duplicate spec: linear");
    }

    @Test
    void acceptsFullOnlyOnlineOptimizerStateMapping() {
        String optimizerConfig = validOnlineOptimizerConfig().replaceAll(
                ",\\s*\"linear_location_spec_group_name\":\"linear-group\"", "");

        contextRunner
                .withPropertyValues("MODEL_SERVICE_CONFIG=" + modelConfig(optimizerConfig))
                .run(context -> assertThat(context).hasNotFailed());
    }

    @Test
    void validatesEnabledOnlineOptimizerDiscovery() {
        assertOnlineOptimizerRejected(
                validOnlineOptimizerConfig().replace(
                        "\"hosts\":[\"127.0.0.1:8082\"]",
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

    private void assertOnlineOptimizerRejected(String optimizerConfig, String message) {
        contextRunner
                .withPropertyValues("MODEL_SERVICE_CONFIG=" + modelConfig(optimizerConfig))
                .run(context -> {
                    assertThat(context).hasFailed();
                    assertThat(context.getStartupFailure()).hasRootCauseMessage(message);
                });
    }

    private String modelConfig(String optimizerConfig) {
        return """
                {"service_id":"test-service","online_optimizer":%s,
                "role_endpoints":[{"group":"default",
                "pd_fusion_endpoint":{"address":"service-a","protocol":"http","path":"/",
                "discovery":{"type":"static-env","hosts":["127.0.0.1:8080"]}}}]}
                """.formatted(optimizerConfig);
    }

    private String validOnlineOptimizerConfig() {
        return """
                {"enabled":true,"address":"optimizer-service","path":"/custom/optimizer",
                "instance_group":"test-group","instance_id":"test-instance",
                "register_timeout_ms":3000,"block_size":64,"linear_step":4,
                "discovery":{"type":"static-env","hosts":["127.0.0.1:8082"]},
                "location_spec_infos":[{"name":"full","size":4294967296},{"name":"linear","size":65536}],
                "location_spec_groups":[{"name":"full-group","spec_names":["full"]},
                {"name":"linear-group","spec_names":["linear"]}],
                "optimizer_state_info":{"full_location_spec_group_name":"full-group",
                "linear_location_spec_group_name":"linear-group"}}
                """;
    }

}
