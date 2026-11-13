package org.flexlb.config.loading.source;

import com.alibaba.nacos.api.config.listener.Listener;
import org.flexlb.config.ConfigService;
import org.flexlb.config.loading.parser.ConfigDocumentParserResolver;
import org.flexlb.config.loading.parser.StandardConfigDocumentParser;
import org.flexlb.config.loading.parser.V0ConfigDocumentParser;
import org.flexlb.config.service.DeploymentIdentity;
import org.flexlb.util.log.LogLevel;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.springframework.test.util.ReflectionTestUtils;
import uk.org.webcompere.systemstubs.environment.EnvironmentVariables;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.flexlb.config.loading.source.NacosConfigConstants.DEFAULT_NACOS_GROUP;
import static org.flexlb.config.loading.source.NacosConfigConstants.NACOS_DATA_ID;
import static org.flexlb.config.loading.source.NacosConfigConstants.NACOS_GROUP;
import static org.flexlb.config.loading.source.NacosConfigConstants.NACOS_NAMESPACE;
import static org.flexlb.config.loading.source.NacosConfigConstants.NACOS_SERVER_ADDR;
import static org.flexlb.config.service.DeploymentIdentityConstants.SPECTRUM_APPLICATION_NAME;
import static org.flexlb.config.service.DeploymentIdentityConstants.SPECTRUM_DEPLOYMENT_NAME;
import static org.flexlb.config.service.DeploymentIdentityConstants.SPECTRUM_WORKSPACE_ID;
import static org.flexlb.config.service.DeploymentIdentityConstants.WHALE_BIZ_NAME;
import static org.flexlb.config.service.DeploymentIdentityConstants.WHALE_DEPLOYMENT_NAME;
import static org.flexlb.config.service.DeploymentIdentityConstants.WHALE_ZONE_NAME;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Checks Nacos configuration subscription and update handling.
 */
class NacosConfigSourceTest {

    private static final String CONFIG_SCHEMA_VERSION_ENV = "FLEXLB_CONFIG_SCHEMA_VERSION";

    @Test
    void isDisabledWhenNacosAddressIsNotConfigured() throws Exception {
        NacosConfigSource source =
                new EnvironmentVariables(
                                WHALE_BIZ_NAME,
                                "dash_pd",
                                WHALE_DEPLOYMENT_NAME,
                                "flexlb-test",
                                WHALE_ZONE_NAME,
                                "master")
                        .remove(NACOS_SERVER_ADDR)
                        .execute(() -> new NacosConfigSource(new DeploymentIdentity()));

        source.initialize();
        ConfigService configService =
                new ConfigService(
                        List.of(new StandardConfigDocumentParser(), new V0ConfigDocumentParser()));

        assertThat(source.priority()).isEqualTo(2);
        assertThat(configService.loadBalanceConfig().getObservability().getLogging().getLevel())
                .isEqualTo(LogLevel.INFO);
        configService.close();
    }

    @Test
    void failsFastWhenDataIdCannotBeResolved() {
        EnvironmentVariables environment =
                new EnvironmentVariables(NACOS_SERVER_ADDR, "127.0.0.1:8848")
                        .remove(NACOS_DATA_ID)
                        .remove(WHALE_BIZ_NAME)
                        .remove(WHALE_DEPLOYMENT_NAME)
                        .remove(WHALE_ZONE_NAME)
                        .remove(SPECTRUM_WORKSPACE_ID)
                        .remove(SPECTRUM_APPLICATION_NAME)
                        .remove(SPECTRUM_DEPLOYMENT_NAME);

        assertThatThrownBy(
                        () ->
                                environment.execute(
                                        () -> new NacosConfigSource(new DeploymentIdentity())))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining(WHALE_BIZ_NAME);
    }

    @Test
    void usesBizDeploymentAndZoneWhenDataIdIsNotConfigured() throws Exception {
        NacosConfigSource source =
                new EnvironmentVariables(
                                NACOS_SERVER_ADDR,
                                "127.0.0.1:8848",
                                WHALE_BIZ_NAME,
                                "dash_pd",
                                WHALE_DEPLOYMENT_NAME,
                                "flexlb-hongyi-test-v1-flexlb-standalone",
                                WHALE_ZONE_NAME,
                                "master")
                        .remove(NACOS_DATA_ID)
                        .remove(SPECTRUM_WORKSPACE_ID)
                        .remove(SPECTRUM_APPLICATION_NAME)
                        .remove(SPECTRUM_DEPLOYMENT_NAME)
                        .execute(() -> new NacosConfigSource(new DeploymentIdentity()));

        assertThat(source)
                .extracting("config")
                .isEqualTo(
                        new NacosConfig(
                                "127.0.0.1:8848",
                                "dash_pd:flexlb-hongyi-test-v1-flexlb-standalone:master",
                                null,
                                null));
    }

    @Test
    void usesSpectrumIdentityWhenDataIdIsNotConfigured() throws Exception {
        NacosConfigSource source =
                new EnvironmentVariables(
                                NACOS_SERVER_ADDR,
                                "127.0.0.1:8848",
                                SPECTRUM_WORKSPACE_ID,
                                "df4a7748",
                                SPECTRUM_APPLICATION_NAME,
                                "flexlb-test",
                                SPECTRUM_DEPLOYMENT_NAME,
                                "flexlb-test-wlcb",
                                WHALE_BIZ_NAME,
                                "dash_pd",
                                WHALE_DEPLOYMENT_NAME,
                                "legacy-role",
                                WHALE_ZONE_NAME,
                                "master")
                        .remove(NACOS_DATA_ID)
                        .execute(() -> new NacosConfigSource(new DeploymentIdentity()));

        assertThat(source)
                .extracting("config")
                .isEqualTo(
                        new NacosConfig(
                                "127.0.0.1:8848",
                                "spectrum:df4a7748:flexlb-test:flexlb-test-wlcb",
                                null,
                                null));
    }

    @Test
    void loadsV0CompatibilityThroughSchemaVersionZero() throws Exception {
        com.alibaba.nacos.api.config.ConfigService client =
                mock(com.alibaba.nacos.api.config.ConfigService.class);
        when(client.getConfig(
                        org.mockito.ArgumentMatchers.eq("flexlb-test"),
                        org.mockito.ArgumentMatchers.eq(DEFAULT_NACOS_GROUP),
                        org.mockito.ArgumentMatchers.eq(3000L)))
                .thenReturn("{\"enableQueueing\":true}");
        new EnvironmentVariables(
                        NACOS_SERVER_ADDR,
                        "127.0.0.1:8848",
                        NACOS_DATA_ID,
                        "flexlb-test",
                        WHALE_BIZ_NAME,
                        "dash_pd",
                        WHALE_DEPLOYMENT_NAME,
                        "flexlb-test",
                        WHALE_ZONE_NAME,
                        "master",
                        CONFIG_SCHEMA_VERSION_ENV,
                        "0")
                .execute(
                        () -> {
                            NacosConfigSource source =
                                    new NacosConfigSource(new DeploymentIdentity());
                            ReflectionTestUtils.setField(source, "client", client);
                            source.initialize();
                            assertThat(source.name()).isEqualTo("Nacos");
                            assertThat(source.loadConfig().sourceSchemaVersion()).isEqualTo(0);
                            ConfigService configService =
                                    new ConfigService(
                                            List.of(
                                                    new StandardConfigDocumentParser(),
                                                    new V0ConfigDocumentParser()));
                            configService.close();
                        });
    }

    @Test
    void rejectsUnknownNacosConfigCompatibilityModes() {
        EnvironmentVariables environment =
                new EnvironmentVariables(
                        WHALE_BIZ_NAME,
                        "dash_pd",
                        WHALE_DEPLOYMENT_NAME,
                        "flexlb-test",
                        WHALE_ZONE_NAME,
                        "master",
                        CONFIG_SCHEMA_VERSION_ENV,
                        "CURRENT");

        assertThatThrownBy(
                        () -> environment.execute(() -> ConfigDocumentParserResolver.resolve("{}")))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining(CONFIG_SCHEMA_VERSION_ENV);
    }

    @Test
    void currentSchemaInNacosOverridesTheV0FallbackMode() throws Exception {
        com.alibaba.nacos.api.config.ConfigService client =
                mock(com.alibaba.nacos.api.config.ConfigService.class);
        ArgumentCaptor<Listener> listenerCaptor = ArgumentCaptor.forClass(Listener.class);
        when(client.getConfig(
                        org.mockito.ArgumentMatchers.eq("flexlb-test"),
                        org.mockito.ArgumentMatchers.eq("FLEXLB_GROUP"),
                        org.mockito.ArgumentMatchers.eq(3000L)))
                .thenReturn("{\"enableQueueing\":true}");
        new EnvironmentVariables(
                        NACOS_SERVER_ADDR,
                        "127.0.0.1:8848",
                        NACOS_DATA_ID,
                        "flexlb-test",
                        NACOS_GROUP,
                        "FLEXLB_GROUP",
                        WHALE_BIZ_NAME,
                        "dash_pd",
                        WHALE_DEPLOYMENT_NAME,
                        "flexlb-test",
                        WHALE_ZONE_NAME,
                        "master",
                        CONFIG_SCHEMA_VERSION_ENV,
                        "0")
                .execute(
                        () -> {
                            NacosConfigSource source =
                                    new NacosConfigSource(new DeploymentIdentity());
                            ReflectionTestUtils.setField(source, "client", client);
                            source.initialize();
                            assertThat(source.name()).isEqualTo("Nacos");
                            verify(client)
                                    .addListener(
                                            org.mockito.ArgumentMatchers.eq("flexlb-test"),
                                            org.mockito.ArgumentMatchers.eq("FLEXLB_GROUP"),
                                            listenerCaptor.capture());

                            listenerCaptor
                                    .getValue()
                                    .receiveConfigInfo(
                                            """
                            {"schemaVersion":4,"requestLifecycle":{"request":{"timeoutMs":60000}},"scheduler":{"type":"QUEUE"}}
                            """);

                            assertThat(source.name()).isEqualTo("Nacos");
                            assertThat(source.loadConfig().sourceSchemaVersion()).isEqualTo(4);
                            ConfigService configService =
                                    new ConfigService(
                                            List.of(
                                                    new StandardConfigDocumentParser(),
                                                    new V0ConfigDocumentParser()));
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
                .thenReturn(
                        "{\"schemaVersion\":4,\"observability\":{\"logging\":{\"level\":\"warn\"}}}");
        NacosConfigSource source = createSource(client, "test-namespace");

        source.initialize();
        verify(client)
                .addListener(
                        org.mockito.ArgumentMatchers.eq("flexlb-test"),
                        org.mockito.ArgumentMatchers.eq("FLEXLB_GROUP"),
                        listenerCaptor.capture());
        ConfigService configService =
                new ConfigService(
                        List.of(new StandardConfigDocumentParser(), new V0ConfigDocumentParser()));

        assertThat(configService.loadBalanceConfig().getObservability().getLogging().getLevel())
                .isEqualTo(LogLevel.WARN);
        listenerCaptor
                .getValue()
                .receiveConfigInfo(
                        "{\"schemaVersion\":4,\"observability\":{\"logging\":{\"level\":\"error\"}}}");
        configService.close();

        assertThat(configService.loadBalanceConfig().getObservability().getLogging().getLevel())
                .isEqualTo(LogLevel.ERROR);
        verify(client).removeListener("flexlb-test", "FLEXLB_GROUP", listenerCaptor.getValue());
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
                .thenReturn(
                        "{\"schemaVersion\":4,\"observability\":{\"logging\":{\"level\":\"warn\"}}}");
        NacosConfigSource source = createSource(client, "");
        source.initialize();
        ConfigService configService =
                new ConfigService(
                        List.of(new StandardConfigDocumentParser(), new V0ConfigDocumentParser()));
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

    private NacosConfigSource createSource(
            com.alibaba.nacos.api.config.ConfigService client, String namespace) throws Exception {
        NacosConfigSource source =
                new EnvironmentVariables(
                                NACOS_SERVER_ADDR,
                                "127.0.0.1:8848",
                                NACOS_DATA_ID,
                                "flexlb-test",
                                NACOS_GROUP,
                                "FLEXLB_GROUP",
                                NACOS_NAMESPACE,
                                namespace,
                                WHALE_BIZ_NAME,
                                "dash_pd",
                                WHALE_DEPLOYMENT_NAME,
                                "flexlb-test",
                                WHALE_ZONE_NAME,
                                "master")
                        .execute(() -> new NacosConfigSource(new DeploymentIdentity()));
        ReflectionTestUtils.setField(source, "client", client);
        return source;
    }
}
