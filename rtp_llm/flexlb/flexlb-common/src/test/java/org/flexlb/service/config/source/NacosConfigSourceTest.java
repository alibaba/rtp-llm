package org.flexlb.service.config.source;

import com.alibaba.nacos.api.config.listener.Listener;
import org.flexlb.config.ConfigService;
import org.flexlb.config.DeploymentIdentity;
import org.flexlb.dao.nacos.NacosConfig;
import org.flexlb.service.config.parser.ConfigDocumentParserResolver;
import org.flexlb.service.config.parser.StandardConfigDocumentParser;
import org.flexlb.service.config.parser.V0ConfigDocumentParser;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.springframework.test.util.ReflectionTestUtils;
import uk.org.webcompere.systemstubs.environment.EnvironmentVariables;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.flexlb.constant.DeploymentIdentityConstants.HIPPO_ROLE;
import static org.flexlb.constant.DeploymentIdentityConstants.SPECTRUM_APPLICATION_NAME;
import static org.flexlb.constant.DeploymentIdentityConstants.SPECTRUM_DEPLOYMENT_NAME;
import static org.flexlb.constant.DeploymentIdentityConstants.SPECTRUM_WORKSPACE_ID;
import static org.flexlb.constant.NacosConfigConstants.DEFAULT_NACOS_GROUP;
import static org.flexlb.constant.NacosConfigConstants.NACOS_DATA_ID;
import static org.flexlb.constant.NacosConfigConstants.NACOS_GROUP;
import static org.flexlb.constant.NacosConfigConstants.NACOS_NAMESPACE;
import static org.flexlb.constant.NacosConfigConstants.NACOS_SERVER_ADDR;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class NacosConfigSourceTest {

    private static final String CONFIG_SCHEMA_VERSION_ENV = "FLEXLB_CONFIG_SCHEMA_VERSION";

    @Test
    void isDisabledWhenNacosAddressIsNotConfigured() throws Exception {
        NacosConfigSource source = new EnvironmentVariables(HIPPO_ROLE, "flexlb-test")
                .remove(NACOS_SERVER_ADDR)
                .execute(() -> new NacosConfigSource(new DeploymentIdentity()));

        source.initialize();
        ConfigService configService = new ConfigService(List.of(new StandardConfigDocumentParser(), new V0ConfigDocumentParser()));

        assertThat(source.priority()).isEqualTo(2);
        assertThat(configService.loadBalanceConfig().getRouter()
                .getAvailabilityHysteresisPercent()).isEqualTo(15);
        configService.close();
    }

    @Test
    void failsFastWhenDataIdCannotBeResolved() {
        EnvironmentVariables environment = new EnvironmentVariables(NACOS_SERVER_ADDR, "127.0.0.1:8848")
                .remove(NACOS_DATA_ID)
                .remove(HIPPO_ROLE)
                .remove(SPECTRUM_WORKSPACE_ID)
                .remove(SPECTRUM_APPLICATION_NAME)
                .remove(SPECTRUM_DEPLOYMENT_NAME);

        assertThatThrownBy(() -> environment.execute(() -> new NacosConfigSource(new DeploymentIdentity())))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining(HIPPO_ROLE);
    }

    @Test
    void usesHippoRoleWhenDataIdIsNotConfigured() throws Exception {
        NacosConfigSource source = new EnvironmentVariables(
                NACOS_SERVER_ADDR, "127.0.0.1:8848",
                HIPPO_ROLE, "flexlb-hongyi-test-v1-flexlb-standalone")
                .remove(NACOS_DATA_ID)
                .remove(SPECTRUM_WORKSPACE_ID)
                .remove(SPECTRUM_APPLICATION_NAME)
                .remove(SPECTRUM_DEPLOYMENT_NAME)
                .execute(() -> new NacosConfigSource(new DeploymentIdentity()));

        assertThat(source)
                .extracting("config")
                .isEqualTo(new NacosConfig(
                        "127.0.0.1:8848",
                        "flexlb-hongyi-test-v1-flexlb-standalone",
                        null,
                        null));
    }

    @Test
    void usesSpectrumIdentityWhenDataIdIsNotConfigured() throws Exception {
        NacosConfigSource source = new EnvironmentVariables(
                NACOS_SERVER_ADDR, "127.0.0.1:8848",
                SPECTRUM_WORKSPACE_ID, "df4a7748",
                SPECTRUM_APPLICATION_NAME, "flexlb-test",
                SPECTRUM_DEPLOYMENT_NAME, "flexlb-test-wlcb",
                HIPPO_ROLE, "legacy-role")
                .remove(NACOS_DATA_ID)
                .execute(() -> new NacosConfigSource(new DeploymentIdentity()));

        assertThat(source)
                .extracting("config")
                .isEqualTo(new NacosConfig(
                        "127.0.0.1:8848",
                        "spectrum:df4a7748:flexlb-test:flexlb-test-wlcb",
                        null,
                        null));
    }

    @Test
    void loadsV2CompatibilityThroughSchemaVersionZero() throws Exception {
        com.alibaba.nacos.api.config.ConfigService client =
                mock(com.alibaba.nacos.api.config.ConfigService.class);
        when(client.getConfig(
                org.mockito.ArgumentMatchers.eq("flexlb-test"),
                org.mockito.ArgumentMatchers.eq(DEFAULT_NACOS_GROUP),
                org.mockito.ArgumentMatchers.eq(3000L)))
                .thenReturn("{\"enableQueueing\":true}");
        new EnvironmentVariables(
                NACOS_SERVER_ADDR, "127.0.0.1:8848",
                NACOS_DATA_ID, "flexlb-test",
                HIPPO_ROLE, "flexlb-test",
                CONFIG_SCHEMA_VERSION_ENV, "0")
                .execute(() -> {
                    NacosConfigSource source = new NacosConfigSource(new DeploymentIdentity());
                    ReflectionTestUtils.setField(source, "client", client);
                    source.initialize();
                    assertThat(source.name()).isEqualTo("Nacos");
                    ConfigService configService = new ConfigService(List.of(new StandardConfigDocumentParser(), new V0ConfigDocumentParser()));
                    configService.close();
                });
    }

    @Test
    void rejectsUnknownNacosConfigCompatibilityModes() {
        EnvironmentVariables environment = new EnvironmentVariables(
                HIPPO_ROLE, "flexlb-test",
                CONFIG_SCHEMA_VERSION_ENV, "CURRENT");

        assertThatThrownBy(() -> environment.execute(() ->
                ConfigDocumentParserResolver.resolve("{}")))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining(CONFIG_SCHEMA_VERSION_ENV);
    }

    @Test
    void currentSchemaInNacosOverridesTheV2FallbackMode() throws Exception {
        com.alibaba.nacos.api.config.ConfigService client =
                mock(com.alibaba.nacos.api.config.ConfigService.class);
        ArgumentCaptor<Listener> listenerCaptor = ArgumentCaptor.forClass(Listener.class);
        when(client.getConfig(
                org.mockito.ArgumentMatchers.eq("flexlb-test"),
                org.mockito.ArgumentMatchers.eq("FLEXLB_GROUP"),
                org.mockito.ArgumentMatchers.eq(3000L)))
                .thenReturn("{\"enableQueueing\":true}");
        new EnvironmentVariables(
                NACOS_SERVER_ADDR, "127.0.0.1:8848",
                NACOS_DATA_ID, "flexlb-test",
                NACOS_GROUP, "FLEXLB_GROUP",
                HIPPO_ROLE, "flexlb-test",
                CONFIG_SCHEMA_VERSION_ENV, "0")
                .execute(() -> {
                    NacosConfigSource source = new NacosConfigSource(new DeploymentIdentity());
                    ReflectionTestUtils.setField(source, "client", client);
                    source.initialize();
                    assertThat(source.name()).isEqualTo("Nacos");
                    verify(client).addListener(
                            org.mockito.ArgumentMatchers.eq("flexlb-test"),
                            org.mockito.ArgumentMatchers.eq("FLEXLB_GROUP"),
                            listenerCaptor.capture());

                    listenerCaptor.getValue().receiveConfigInfo("""
                            {"schemaVersion":1,"scheduler":{"type":"QUEUE"},"dispatcher":{"type":"BATCH"}}
                            """);

                    assertThat(source.name()).isEqualTo("Nacos");
                    ConfigService configService = new ConfigService(List.of(new StandardConfigDocumentParser(), new V0ConfigDocumentParser()));
                    configService.close();
                });
    }

    @Test
    void loadsListensAndClosesNacosConfig() throws Exception {
        com.alibaba.nacos.api.config.ConfigService client =
                mock(com.alibaba.nacos.api.config.ConfigService.class);
        ArgumentCaptor<Listener> listenerCaptor = ArgumentCaptor.forClass(Listener.class);
        when(client.getConfig(
                org.mockito.ArgumentMatchers.eq("flexlb-test"),
                org.mockito.ArgumentMatchers.eq("FLEXLB_GROUP"),
                org.mockito.ArgumentMatchers.eq(3000L)))
                .thenReturn("{\"schemaVersion\":1,\"router\":{\"availabilityHysteresisPercent\":9}}");
        NacosConfigSource source = createSource(client, "test-namespace");

        source.initialize();
        verify(client).addListener(
                org.mockito.ArgumentMatchers.eq("flexlb-test"),
                org.mockito.ArgumentMatchers.eq("FLEXLB_GROUP"),
                listenerCaptor.capture());
        ConfigService configService = new ConfigService(List.of(new StandardConfigDocumentParser(), new V0ConfigDocumentParser()));

        assertThat(configService.loadBalanceConfig().getRouter()
                .getAvailabilityHysteresisPercent()).isEqualTo(9);
        listenerCaptor.getValue().receiveConfigInfo(
                "{\"schemaVersion\":1,\"router\":{\"availabilityHysteresisPercent\":10}}");
        configService.close();

        assertThat(configService.loadBalanceConfig().getRouter()
                .getAvailabilityHysteresisPercent()).isEqualTo(10);
        verify(client).removeListener(
                "flexlb-test",
                "FLEXLB_GROUP",
                listenerCaptor.getValue());
        verify(client).shutDown();
    }

    @Test
    void shutsDownClientWhenRemovingListenerFails() throws Exception {
        com.alibaba.nacos.api.config.ConfigService client =
                mock(com.alibaba.nacos.api.config.ConfigService.class);
        when(client.getConfig(
                org.mockito.ArgumentMatchers.eq("flexlb-test"),
                org.mockito.ArgumentMatchers.eq("FLEXLB_GROUP"),
                org.mockito.ArgumentMatchers.eq(3000L)))
                .thenReturn("{\"schemaVersion\":1,\"router\":{\"availabilityHysteresisPercent\":9}}");
        NacosConfigSource source = createSource(client, "");
        source.initialize();
        ConfigService configService = new ConfigService(List.of(new StandardConfigDocumentParser(), new V0ConfigDocumentParser()));
        doThrow(new RuntimeException("remove failed"))
                .when(client)
                .removeListener(
                        org.mockito.ArgumentMatchers.eq("flexlb-test"),
                        org.mockito.ArgumentMatchers.eq("FLEXLB_GROUP"),
                        org.mockito.ArgumentMatchers.any(Listener.class));

        assertThatThrownBy(source::close).hasMessage("remove failed");

        verify(client).shutDown();
        configService.close();
    }

    private NacosConfigSource createSource(com.alibaba.nacos.api.config.ConfigService client, String namespace) throws Exception {
        NacosConfigSource source = new EnvironmentVariables(
                NACOS_SERVER_ADDR, "127.0.0.1:8848",
                NACOS_DATA_ID, "flexlb-test",
                NACOS_GROUP, "FLEXLB_GROUP",
                NACOS_NAMESPACE, namespace,
                HIPPO_ROLE, "flexlb-test")
                .execute(() -> new NacosConfigSource(new DeploymentIdentity()));
        ReflectionTestUtils.setField(source, "client", client);
        return source;
    }
}
