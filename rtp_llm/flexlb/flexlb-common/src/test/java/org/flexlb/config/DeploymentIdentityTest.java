package org.flexlb.config;

import org.junit.jupiter.api.Test;
import uk.org.webcompere.systemstubs.environment.EnvironmentVariables;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.flexlb.constant.DeploymentIdentityConstants.HIPPO_APP;
import static org.flexlb.constant.DeploymentIdentityConstants.HIPPO_ROLE;
import static org.flexlb.constant.DeploymentIdentityConstants.SPECTRUM_APPLICATION_NAME;
import static org.flexlb.constant.DeploymentIdentityConstants.SPECTRUM_DEPLOYMENT_NAME;
import static org.flexlb.constant.DeploymentIdentityConstants.SPECTRUM_WORKSPACE_ID;

class DeploymentIdentityTest {

    @Test
    void usesSpectrumIdentityWhenAllSpectrumFieldsAreConfigured() throws Exception {
        DeploymentIdentity identity = new EnvironmentVariables(
                SPECTRUM_WORKSPACE_ID, "df4a7748",
                SPECTRUM_APPLICATION_NAME, "flexlb-test",
                SPECTRUM_DEPLOYMENT_NAME, "flexlb-test-wlcb",
                HIPPO_APP, "legacy-app",
                HIPPO_ROLE, "legacy-role")
                .execute(DeploymentIdentity::new);

        assertThat(identity.isSpectrum()).isTrue();
        assertThat(identity.getDeploymentId())
                .isEqualTo("spectrum:df4a7748:flexlb-test:flexlb-test-wlcb");
    }

    @Test
    void fallsBackToHippoIdentityWhenSpectrumFieldsAreNotConfigured() throws Exception {
        DeploymentIdentity identity = withoutSpectrumFields()
                .set(HIPPO_APP, "flexlb-app")
                .set(HIPPO_ROLE, "flexlb-role")
                .execute(DeploymentIdentity::new);

        assertThat(identity.isSpectrum()).isFalse();
        assertThat(identity.getHippoApp()).isEqualTo("flexlb-app");
        assertThat(identity.getDeploymentId()).isEqualTo("flexlb-role");
    }

    @Test
    void failsFastWhenSpectrumIdentityIsIncomplete() {
        EnvironmentVariables environment = new EnvironmentVariables(
                SPECTRUM_WORKSPACE_ID, "df4a7748",
                SPECTRUM_APPLICATION_NAME, "flexlb-test")
                .remove(SPECTRUM_DEPLOYMENT_NAME);

        assertThatThrownBy(() -> environment.execute(DeploymentIdentity::new))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining(SPECTRUM_DEPLOYMENT_NAME);
    }

    @Test
    void failsFastWhenNoDeploymentIdentityIsConfigured() {
        EnvironmentVariables environment = withoutSpectrumFields()
                .remove(HIPPO_APP)
                .remove(HIPPO_ROLE);

        assertThatThrownBy(() -> environment.execute(DeploymentIdentity::new))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining(HIPPO_ROLE);
    }

    private EnvironmentVariables withoutSpectrumFields() {
        return new EnvironmentVariables()
                .remove(SPECTRUM_WORKSPACE_ID)
                .remove(SPECTRUM_APPLICATION_NAME)
                .remove(SPECTRUM_DEPLOYMENT_NAME);
    }
}
