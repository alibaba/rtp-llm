package org.flexlb.service.config.source;

import com.alibaba.nacos.api.NacosFactory;
import com.alibaba.nacos.api.config.listener.Listener;
import org.flexlb.config.ConfigService;
import org.flexlb.config.DeploymentIdentity;
import org.flexlb.service.config.parser.StandardConfigDocumentParser;
import org.flexlb.service.config.parser.V0ConfigDocumentParser;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.NullAndEmptySource;
import org.junit.jupiter.params.provider.ValueSource;
import org.mockito.MockedStatic;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import uk.org.webcompere.systemstubs.environment.EnvironmentVariables;

import java.util.Properties;

import static org.assertj.core.api.Assertions.assertThat;
import static org.flexlb.constant.DeploymentIdentityConstants.SPECTRUM_APPLICATION_NAME;
import static org.flexlb.constant.DeploymentIdentityConstants.SPECTRUM_DEPLOYMENT_NAME;
import static org.flexlb.constant.DeploymentIdentityConstants.SPECTRUM_WORKSPACE_ID;
import static org.flexlb.constant.DeploymentIdentityConstants.WHALE_BIZ_NAME;
import static org.flexlb.constant.DeploymentIdentityConstants.WHALE_DEPLOYMENT_NAME;
import static org.flexlb.constant.DeploymentIdentityConstants.WHALE_ZONE_NAME;
import static org.flexlb.constant.NacosConfigConstants.DEFAULT_NACOS_GROUP;
import static org.flexlb.constant.NacosConfigConstants.NACOS_DATA_ID;
import static org.flexlb.constant.NacosConfigConstants.NACOS_GROUP;
import static org.flexlb.constant.NacosConfigConstants.NACOS_SERVER_ADDR;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class ConfigSourceSelectionTest {

    @ParameterizedTest
    @NullAndEmptySource
    @ValueSource(strings = {"   "})
    void startupFailsWhenOnlyTheHippoRoleConfigExistsAfterRuntimeUpgrade(String missingConfig) throws Exception {
        com.alibaba.nacos.api.config.ConfigService client = mock(com.alibaba.nacos.api.config.ConfigService.class);
        String deploymentId = "dash_pd:ea118_RTX_PRO_5000_72GB:master";
        when(client.getConfig(anyString(), eq(DEFAULT_NACOS_GROUP), eq(3000L)))
                .thenAnswer(invocation -> "legacy-role".equals(invocation.getArgument(0))
                        ? "{\"schemaVersion\":3,\"enableFallback\":true}" : missingConfig);

        try (MockedStatic<NacosFactory> factory = mockStatic(NacosFactory.class)) {
            factory.when(() -> NacosFactory.createConfigService(any(Properties.class))).thenReturn(client);
            new EnvironmentVariables(
                    "FLEXLB_UNICONF_ENABLE", "false",
                    NACOS_SERVER_ADDR, "127.0.0.1:8848",
                    "HIPPO_ROLE", "legacy-role",
                    "BIZ_NAME", "dash_pd",
                    "DEPLOYMENT_NAME", "ea118_RTX_PRO_5000_72GB",
                    "ZONE_NAME", "master",
                    "FLEXLB_CONFIG", "{\"schemaVersion\":3,\"enableFallback\":true}")
                    .remove(NACOS_DATA_ID)
                    .remove(NACOS_GROUP)
                    .remove(SPECTRUM_WORKSPACE_ID)
                    .remove(SPECTRUM_APPLICATION_NAME)
                    .remove(SPECTRUM_DEPLOYMENT_NAME)
                    .execute(() -> new ApplicationContextRunner()
                            .withUserConfiguration(DeploymentIdentity.class, NacosConfigSource.class)
                            .run(context -> {
                                assertThat(context).hasFailed();
                                assertThat(context.getStartupFailure())
                                        .hasRootCauseInstanceOf(IllegalStateException.class)
                                        .hasStackTraceContaining(deploymentId)
                                        .hasStackTraceContaining("Nacos configuration is missing or blank");
                            }));
        }
        verify(client).getConfig(deploymentId, DEFAULT_NACOS_GROUP, 3000L);
        verify(client, never()).getConfig("legacy-role", DEFAULT_NACOS_GROUP, 3000L);
        verify(client, never()).addListener(anyString(), anyString(), any(Listener.class));
        verify(client).shutDown();
    }

    @Test
    void springInitializesSourcesBeforeLoadingTheEnvironmentFallback() throws Exception {
        new EnvironmentVariables(
                "FLEXLB_UNICONF_ENABLE", "false",
                "UNICONF_ENABLE", "true",
                WHALE_BIZ_NAME, "dash_pd", WHALE_DEPLOYMENT_NAME, "flexlb-test", WHALE_ZONE_NAME, "master",
                "FLEXLB_CONFIG", "{\"schemaVersion\":3,\"enableFallback\":true,"
                        + "\"fallbackBatchTokenCapacity\":1048576}")
                .remove(NACOS_SERVER_ADDR)
                .remove(SPECTRUM_WORKSPACE_ID)
                .remove(SPECTRUM_APPLICATION_NAME)
                .remove(SPECTRUM_DEPLOYMENT_NAME)
                .remove("MODEL_SERVICE_CONFIG")
                .execute(() -> new ApplicationContextRunner()
                        .withUserConfiguration(ConfigService.class, DeploymentIdentity.class,
                                EnvironmentConfigSource.class, NacosConfigSource.class, UniConfigConfigSource.class,
                                StandardConfigDocumentParser.class, V0ConfigDocumentParser.class)
                        .run(context -> {
                            assertThat(context).hasNotFailed();
                            assertThat(context.getBean(ConfigService.class).loadBalanceConfig().isEnableFallback()).isTrue();
                        }));
    }

    @Test
    void environmentSourceDoesNotRequireDeploymentIdentity() throws Exception {
        new EnvironmentVariables(
                "FLEXLB_UNICONF_ENABLE", "false",
                "FLEXLB_CONFIG", "{\"schemaVersion\":3,\"enableFallback\":true}")
                .remove(NACOS_SERVER_ADDR)
                .remove(WHALE_BIZ_NAME)
                .remove(WHALE_DEPLOYMENT_NAME)
                .remove(WHALE_ZONE_NAME)
                .remove(SPECTRUM_WORKSPACE_ID)
                .remove(SPECTRUM_APPLICATION_NAME)
                .remove(SPECTRUM_DEPLOYMENT_NAME)
                .remove("MODEL_SERVICE_CONFIG")
                .execute(() -> new ApplicationContextRunner()
                        .withUserConfiguration(ConfigService.class, DeploymentIdentity.class,
                                EnvironmentConfigSource.class, NacosConfigSource.class,
                                UniConfigConfigSource.class,
                                StandardConfigDocumentParser.class,
                                V0ConfigDocumentParser.class)
                        .run(context -> assertThat(context).hasNotFailed()));
    }

    @ParameterizedTest
    @CsvSource({
            "true, false, nacos:8848, UNICONFIG",
            "' TRUE ', , , UNICONFIG",
            "false, true, nacos:8848, NACOS",
            ", true, ' nacos:8848 ', NACOS",
            "false, true, , ENVIRONMENT",
            ", true, '   ', ENVIRONMENT",
            ", , , ENVIRONMENT"
    })
    void selectsOneBehaviorSource(String flexlbUniconfEnable, String turboUniconfEnable, String nacosAddress,
                                 ConfigSourceSelection expected) throws Exception {
        new EnvironmentVariables()
                .set("FLEXLB_UNICONF_ENABLE", flexlbUniconfEnable)
                .set("UNICONF_ENABLE", turboUniconfEnable)
                .set(NACOS_SERVER_ADDR, nacosAddress)
                .execute(() -> assertThat(ConfigSourceSelection.fromEnvironment()).isEqualTo(expected));
    }
}
