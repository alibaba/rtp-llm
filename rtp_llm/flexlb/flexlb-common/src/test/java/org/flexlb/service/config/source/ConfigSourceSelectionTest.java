package org.flexlb.service.config.source;

import org.flexlb.config.ConfigService;
import org.flexlb.config.DeploymentIdentity;
import org.flexlb.service.config.parser.StandardConfigDocumentParser;
import org.flexlb.service.config.parser.V0ConfigDocumentParser;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import uk.org.webcompere.systemstubs.environment.EnvironmentVariables;

import static org.assertj.core.api.Assertions.assertThat;
import static org.flexlb.constant.DeploymentIdentityConstants.HIPPO_ROLE;
import static org.flexlb.constant.DeploymentIdentityConstants.SPECTRUM_APPLICATION_NAME;
import static org.flexlb.constant.DeploymentIdentityConstants.SPECTRUM_DEPLOYMENT_NAME;
import static org.flexlb.constant.DeploymentIdentityConstants.SPECTRUM_WORKSPACE_ID;
import static org.flexlb.constant.NacosConfigConstants.NACOS_SERVER_ADDR;

class ConfigSourceSelectionTest {

    @Test
    void springInitializesSourcesBeforeLoadingTheEnvironmentFallback() throws Exception {
        new EnvironmentVariables(
                "FLEXLB_UNICONF_ENABLE", "false",
                "UNICONF_ENABLE", "true",
                HIPPO_ROLE, "flexlb-test",
                "FLEXLB_CONFIG", "{\"schemaVersion\":1,\"enableFallback\":true}")
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
