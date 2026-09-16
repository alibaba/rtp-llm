package org.flexlb.config;

import lombok.Getter;
import org.apache.commons.lang3.StringUtils;
import org.springframework.stereotype.Component;

import static org.flexlb.constant.DeploymentIdentityConstants.SPECTRUM_APPLICATION_NAME;
import static org.flexlb.constant.DeploymentIdentityConstants.SPECTRUM_DEPLOYMENT_NAME;
import static org.flexlb.constant.DeploymentIdentityConstants.SPECTRUM_WORKSPACE_ID;
import static org.flexlb.constant.DeploymentIdentityConstants.WHALE_BIZ_NAME;
import static org.flexlb.constant.DeploymentIdentityConstants.WHALE_DEPLOYMENT_NAME;
import static org.flexlb.constant.DeploymentIdentityConstants.WHALE_ZONE_NAME;

@Getter
@Component
public class DeploymentIdentity {

    private static final String SPECTRUM_IDENTITY_PREFIX = "spectrum:";

    private final String spectrumWorkspaceId;
    private final String spectrumDeploymentName;
    private final String deploymentId;

    public DeploymentIdentity() {
        String spectrumWorkspaceId = StringUtils.trimToNull(System.getenv(SPECTRUM_WORKSPACE_ID));
        String applicationName = StringUtils.trimToNull(System.getenv(SPECTRUM_APPLICATION_NAME));
        String spectrumDeploymentName = StringUtils.trimToNull(System.getenv(SPECTRUM_DEPLOYMENT_NAME));

        if (spectrumWorkspaceId != null && applicationName != null && spectrumDeploymentName != null) {
            this.spectrumWorkspaceId = spectrumWorkspaceId;
            this.spectrumDeploymentName = spectrumDeploymentName;
            deploymentId = SPECTRUM_IDENTITY_PREFIX + spectrumWorkspaceId + ":" + applicationName + ":"
                    + spectrumDeploymentName;
            return;
        }

        String bizName = StringUtils.trimToNull(System.getenv(WHALE_BIZ_NAME));
        String runtimeDeploymentName = StringUtils.trimToNull(System.getenv(WHALE_DEPLOYMENT_NAME));
        String zoneName = StringUtils.trimToNull(System.getenv(WHALE_ZONE_NAME));

        if (bizName != null && runtimeDeploymentName != null && zoneName != null) {
            this.spectrumWorkspaceId = null;
            this.spectrumDeploymentName = null;
            deploymentId = bizName + ":" + runtimeDeploymentName + ":" + zoneName;
            return;
        }

        throw new IllegalStateException("Deployment identity requires a complete Spectrum or runtime triplet: "
                + SPECTRUM_WORKSPACE_ID + "=" + spectrumWorkspaceId + ", "
                + SPECTRUM_APPLICATION_NAME + "=" + applicationName + ", "
                + SPECTRUM_DEPLOYMENT_NAME + "=" + spectrumDeploymentName + "; "
                + WHALE_BIZ_NAME + "=" + bizName + ", " + WHALE_DEPLOYMENT_NAME + "=" + runtimeDeploymentName + ", "
                + WHALE_ZONE_NAME + "=" + zoneName);
    }

    public boolean isSpectrum() {
        return spectrumWorkspaceId != null;
    }
}
