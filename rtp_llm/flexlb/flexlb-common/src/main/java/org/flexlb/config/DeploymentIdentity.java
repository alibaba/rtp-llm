package org.flexlb.config;

import lombok.Getter;
import org.apache.commons.lang3.StringUtils;
import org.springframework.stereotype.Component;

import static org.flexlb.constant.DeploymentIdentityConstants.HIPPO_APP;
import static org.flexlb.constant.DeploymentIdentityConstants.HIPPO_ROLE;
import static org.flexlb.constant.DeploymentIdentityConstants.SPECTRUM_APPLICATION_NAME;
import static org.flexlb.constant.DeploymentIdentityConstants.SPECTRUM_DEPLOYMENT_NAME;
import static org.flexlb.constant.DeploymentIdentityConstants.SPECTRUM_WORKSPACE_ID;

@Getter
@Component
public class DeploymentIdentity {

    private static final String SPECTRUM_IDENTITY_PREFIX = "spectrum:";

    private final String workspaceId;
    private final String applicationName;
    private final String deploymentName;
    private final String hippoApp;
    private final String hippoRole;
    private final String deploymentId;

    public DeploymentIdentity() {
        workspaceId = StringUtils.trimToNull(System.getenv(SPECTRUM_WORKSPACE_ID));
        applicationName = StringUtils.trimToNull(System.getenv(SPECTRUM_APPLICATION_NAME));
        deploymentName = StringUtils.trimToNull(System.getenv(SPECTRUM_DEPLOYMENT_NAME));
        hippoApp = StringUtils.trimToNull(System.getenv(HIPPO_APP));
        hippoRole = StringUtils.trimToNull(System.getenv(HIPPO_ROLE));

        boolean hasSpectrumField = workspaceId != null || applicationName != null || deploymentName != null;
        if (hasSpectrumField && (workspaceId == null || applicationName == null || deploymentName == null)) {
            throw new IllegalStateException(
                    SPECTRUM_WORKSPACE_ID
                            + ", "
                            + SPECTRUM_APPLICATION_NAME
                            + " and "
                            + SPECTRUM_DEPLOYMENT_NAME
                            + " must be configured together");
        }
        if (hasSpectrumField) {
            deploymentId = SPECTRUM_IDENTITY_PREFIX + workspaceId + ":" + applicationName + ":" + deploymentName;
        } else if (hippoRole != null) {
            deploymentId = hippoRole;
        } else {
            throw new IllegalStateException(
                    "Spectrum deployment identity or environment variable " + HIPPO_ROLE + " must be configured");
        }
    }

    public boolean isSpectrum() {
        return workspaceId != null;
    }
}
