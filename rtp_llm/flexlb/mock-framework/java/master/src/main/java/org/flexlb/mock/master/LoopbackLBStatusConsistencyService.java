package org.flexlb.mock.master;

import org.flexlb.config.ConfigService;
import org.flexlb.config.DeploymentIdentity;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.consistency.ZookeeperMasterElectService;
import org.springframework.core.env.Environment;

/**
 * Uses the local Mock Master's loopback address for consistency status and the forwarding guard.
 */
public final class LoopbackLBStatusConsistencyService extends LBStatusConsistencyService {
    public LoopbackLBStatusConsistencyService(
            ZookeeperMasterElectService zookeeperMasterElectService,
            Environment environment,
            ConfigService configService,
            DeploymentIdentity deploymentIdentity) {
        super(zookeeperMasterElectService, environment, configService, deploymentIdentity);
    }

    @Override
    protected String resolveLocalIp(Environment environment) {
        return environment.getRequiredProperty("server.address");
    }
}
