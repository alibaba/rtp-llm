package org.flexlb.config;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;
import uk.org.webcompere.systemstubs.environment.EnvironmentVariables;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.flexlb.constant.DeploymentIdentityConstants.SPECTRUM_APPLICATION_NAME;
import static org.flexlb.constant.DeploymentIdentityConstants.SPECTRUM_DEPLOYMENT_NAME;
import static org.flexlb.constant.DeploymentIdentityConstants.SPECTRUM_WORKSPACE_ID;
import static org.flexlb.constant.DeploymentIdentityConstants.WHALE_BIZ_NAME;
import static org.flexlb.constant.DeploymentIdentityConstants.WHALE_DEPLOYMENT_NAME;
import static org.flexlb.constant.DeploymentIdentityConstants.WHALE_ZONE_NAME;

class DeploymentIdentityTest {

    @Test
    void usesSpectrumIdentityWhenAllSpectrumFieldsAreConfigured() throws Exception {
        DeploymentIdentity identity = runtimeEnvironment()
                .set(SPECTRUM_WORKSPACE_ID, "df4a7748")
                .set(SPECTRUM_APPLICATION_NAME, "flexlb-test")
                .set(SPECTRUM_DEPLOYMENT_NAME, "flexlb-test-wlcb")
                .execute(DeploymentIdentity::new);

        assertThat(identity.isSpectrum()).isTrue();
        assertThat(identity.getDeploymentId())
                .isEqualTo("spectrum:df4a7748:flexlb-test:flexlb-test-wlcb");
    }

    @Test
    void joinsBizDeploymentAndZoneWhenSpectrumFieldsAreNotConfigured() throws Exception {
        DeploymentIdentity identity = runtimeEnvironment()
                .set(WHALE_BIZ_NAME, " dash_pd ")
                .set(WHALE_DEPLOYMENT_NAME, " ea118_RTX_PRO_5000_72GB ")
                .set(WHALE_ZONE_NAME, " master ")
                .execute(DeploymentIdentity::new);

        assertThat(identity.isSpectrum()).isFalse();
        assertThat(identity.getDeploymentId()).isEqualTo("dash_pd:ea118_RTX_PRO_5000_72GB:master");
    }

    @ParameterizedTest
    @ValueSource(strings = {SPECTRUM_WORKSPACE_ID, SPECTRUM_APPLICATION_NAME, SPECTRUM_DEPLOYMENT_NAME})
    void usesRuntimeIdentityWhenSpectrumIdentityIsIncomplete(String missingField) throws Exception {
        DeploymentIdentity identity = runtimeEnvironment()
                .set(SPECTRUM_WORKSPACE_ID, "df4a7748")
                .set(SPECTRUM_APPLICATION_NAME, "flexlb-test")
                .set(SPECTRUM_DEPLOYMENT_NAME, "flexlb-test-wlcb")
                .set(missingField, null)
                .execute(DeploymentIdentity::new);

        assertThat(identity.isSpectrum()).isFalse();
        assertThat(identity.getDeploymentId()).isEqualTo("dash_pd:ea118_RTX_PRO_5000_72GB:master");
    }

    @Test
    void reportsBothIdentitiesWhenNeitherTripletIsComplete() {
        EnvironmentVariables environment = runtimeEnvironment()
                .set(SPECTRUM_WORKSPACE_ID, "df4a7748")
                .set(WHALE_ZONE_NAME, null);

        assertThatThrownBy(() -> environment.execute(
                () -> new DeploymentIdentity().getDeploymentId()))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining(SPECTRUM_WORKSPACE_ID + "=df4a7748")
                .hasMessageContaining(SPECTRUM_APPLICATION_NAME + "=null")
                .hasMessageContaining(SPECTRUM_DEPLOYMENT_NAME + "=null")
                .hasMessageContaining(WHALE_BIZ_NAME + "=dash_pd")
                .hasMessageContaining(WHALE_DEPLOYMENT_NAME + "=ea118_RTX_PRO_5000_72GB")
                .hasMessageContaining(WHALE_ZONE_NAME + "=null");
    }

    @ParameterizedTest
    @CsvSource({
            "BIZ_NAME,",
            "DEPLOYMENT_NAME,",
            "ZONE_NAME,",
            "BIZ_NAME, ' '",
            "DEPLOYMENT_NAME, ' '",
            "ZONE_NAME, ' '"
    })
    void rejectsIncompleteRuntimeIdentity(String field, String value) {
        EnvironmentVariables environment = runtimeEnvironment().set(field, value);

        assertThatThrownBy(() -> environment.execute(
                () -> new DeploymentIdentity().getDeploymentId()))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining(field);
    }

    @Test
    void rejectsHippoRoleAsTheOnlyDeploymentIdentity() {
        EnvironmentVariables environment = runtimeEnvironment()
                .remove(WHALE_BIZ_NAME)
                .remove(WHALE_DEPLOYMENT_NAME)
                .remove(WHALE_ZONE_NAME);

        assertThatThrownBy(() -> environment.execute(
                () -> new DeploymentIdentity().getDeploymentId()))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining(WHALE_BIZ_NAME)
                .hasMessageContaining(WHALE_DEPLOYMENT_NAME)
                .hasMessageContaining(WHALE_ZONE_NAME);
    }

    private EnvironmentVariables runtimeEnvironment() {
        return new EnvironmentVariables(
                WHALE_BIZ_NAME, "dash_pd",
                WHALE_DEPLOYMENT_NAME, "ea118_RTX_PRO_5000_72GB",
                WHALE_ZONE_NAME, "master",
                "HIPPO_ROLE", "legacy-role")
                .set(SPECTRUM_WORKSPACE_ID, null)
                .set(SPECTRUM_APPLICATION_NAME, null)
                .set(SPECTRUM_DEPLOYMENT_NAME, null);
    }
}
