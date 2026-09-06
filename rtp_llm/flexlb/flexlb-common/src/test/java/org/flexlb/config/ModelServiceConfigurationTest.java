package org.flexlb.config;

import org.flexlb.dao.route.OptimizerConfig;
import org.flexlb.dao.route.ServiceRoute;
import org.flexlb.discovery.ServiceDiscoveryType;
import org.flexlb.util.JsonUtils;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class ModelServiceConfigurationTest {

    private final ApplicationContextRunner contextRunner = new ApplicationContextRunner()
            .withUserConfiguration(
                    ServiceDiscoveryConfiguration.class,
                    ModelServiceConfiguration.class)
            .withBean("environmentConfigSource", Object.class, Object::new)
            .withBean("nacosConfigSource", Object.class, Object::new);

    @Test
    void failsWhenModelServiceConfigIsMissing() {
        ConfigService configService = mock(ConfigService.class);
        when(configService.modelServiceConfig()).thenReturn(null);

        contextRunner
                .withBean(ConfigService.class, () -> configService)
                .run(context -> {
                    assertThat(context).hasFailed();
                    assertThat(context.getStartupFailure())
                            .hasRootCauseMessage("MODEL_SERVICE_CONFIG must not be blank");
                });
    }

    @Test
    void loadsAndValidatesEndpointDiscoveryTopology() {
        withModelConfig(staticModelConfig())
                .run(context -> {
                    assertThat(context).hasNotFailed();
                    ModelMetaConfig config = context.getBean(ModelMetaConfig.class);
                    var endpoint = config.getServiceRoute("test-service")
                            .getAllEndpoints().getFirst();
                    assertThat(endpoint.getDiscovery().getType())
                            .isEqualTo(ServiceDiscoveryType.STATIC_ENV);
                    assertThat(endpoint.getDiscovery().getHosts())
                            .containsExactly("127.0.0.1:8080");
                    assertThat(endpoint.getWorkerStatusPort()).isEqualTo(18002);
                });
    }

    @Test
    void rejectsEndpointWithoutDiscovery() {
        String config = """
                {"service_id":"test-service","role_endpoints":[{"group":"default",
                "pd_fusion_endpoint":{"address":"service-a","protocol":"http","path":"/"}}]}
                """;

        withModelConfig(config)
                .run(context -> {
                    assertThat(context).hasFailed();
                    assertThat(context.getStartupFailure())
                            .hasRootCauseMessage(
                                    "endpoint discovery must be configured for address: service-a");
                });
    }

    @Test
    void rejectsDiscoveryTypeWithoutProvider() {
        String config = """
                {"service_id":"test-service","role_endpoints":[{"group":"default",
                "pd_fusion_endpoint":{"address":"v-test","protocol":"http","path":"/",
                "discovery":{"type":"dashscope","base_url":"http://127.0.0.1:8880"}}}]}
                """;

        withModelConfig(config)
                .run(context -> {
                    assertThat(context).hasFailed();
                    assertThat(context.getStartupFailure())
                            .hasRootCauseMessage(
                                    "No service discovery provider available for type: "
                                            + "dashscope, address: v-test");
                });
    }

    @Test
    void loadsKvcmTopology() {
        String config = """
                {"service_id":"test-service","kvcm":{"address":"kvcm-service",
                "port":7381,
                "discovery":{"type":"static-env","hosts":["127.0.0.1:8080"]}},
                "role_endpoints":[{"group":"default",
                "pd_fusion_endpoint":{"address":"service-a","protocol":"http","path":"/",
                "discovery":{"type":"static-env","hosts":["127.0.0.1:8080"]}}}]}
                """;

        withModelConfig(config)
                .run(context -> {
                    assertThat(context).hasNotFailed();
                    var route = context.getBean(ModelMetaConfig.class)
                            .getServiceRoute("test-service");
                    assertThat(route.getKvcm()).isNotNull();
                    assertThat(route.getKvcm().getPort()).isEqualTo(7381);
                    assertThat(route.getAllEndpoints()).hasSize(1);
                });
    }

    @Test
    void loadsOptimizerTopology() {
        withModelConfig(modelConfig(validOptimizerConfig()))
                .run(context -> {
                    assertThat(context).hasNotFailed();
                    OptimizerConfig optimizer = context.getBean(ModelMetaConfig.class)
                            .getServiceRoute("test-service")
                            .getOptimizer();
                    assertThat(optimizer.getAddress()).isEqualTo("optimizer-service");
                    assertThat(optimizer.getPort()).isEqualTo(9090);
                    assertThat(optimizer.getPath()).isEqualTo("/custom/optimizer");
                    assertThat(optimizer.toEndpoint().getProtocol()).isEqualTo("http");
                    assertThat(optimizer.toEndpoint().getDiscovery().getType())
                            .isEqualTo(ServiceDiscoveryType.STATIC_ENV);
                });
    }

    @Test
    void appliesOptimizerTopologyDefaults() {
        withModelConfig(modelConfig(
                validOptimizerConfig().replace(
                        ",\"port\":9090,\"path\":\"/custom/optimizer\"", "")))
                .run(context -> {
                    assertThat(context).hasNotFailed();
                    OptimizerConfig optimizer = context.getBean(ModelMetaConfig.class)
                            .getServiceRoute("test-service")
                            .getOptimizer();
                    assertThat(optimizer.getPath()).isEqualTo(OptimizerConfig.DEFAULT_PATH);
                    assertThat(optimizer.getPort()).isEqualTo(OptimizerConfig.DEFAULT_PORT);
                });
    }

    @Test
    void rejectsOptimizerWithInvalidPathOrDiscovery() {
        assertOptimizerRejected(
                validOptimizerConfig().replace(
                        "\"path\":\"/custom/optimizer\"",
                        "\"path\":\"custom/optimizer\""),
                "MODEL_SERVICE_CONFIG online_optimizer.path must start with '/'");
        assertOptimizerRejected(
                validOptimizerConfig().replace(
                        "\"hosts\":[\"127.0.0.1:8082\"]",
                        "\"hosts\":[]"),
                "static-env discovery hosts must be configured for address: optimizer-service");
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
        withModelConfig(modelConfig(optimizerConfig))
                .run(context -> {
                    assertThat(context).hasFailed();
                    assertThat(context.getStartupFailure()).hasRootCauseMessage(message);
                });
    }

    private ApplicationContextRunner withModelConfig(String configJson) {
        ServiceRoute serviceRoute = JsonUtils.toObject(configJson, ServiceRoute.class);
        ConfigService configService = mock(ConfigService.class);
        when(configService.modelServiceConfig()).thenReturn(serviceRoute);
        return contextRunner.withBean(ConfigService.class, () -> configService);
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
                {"address":"optimizer-service","port":9090,"path":"/custom/optimizer",
                "discovery":{"type":"static-env","hosts":["127.0.0.1:8082"]}}
                """;
    }
}
