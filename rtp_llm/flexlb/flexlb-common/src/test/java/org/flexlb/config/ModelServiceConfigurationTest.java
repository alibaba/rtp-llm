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

    private String staticModelConfig() {
        return """
                {"service_id":"test-service","role_endpoints":[{"group":"default",
                "pd_fusion_endpoint":{"address":"service-a","protocol":"http","path":"/",
                "discovery":{"type":"static-env","hosts":["127.0.0.1:8080"]}}}]}
                """;
    }

}
