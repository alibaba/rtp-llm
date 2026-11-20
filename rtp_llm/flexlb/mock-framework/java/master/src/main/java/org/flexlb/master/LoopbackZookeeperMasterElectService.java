package org.flexlb.master;

import org.flexlb.config.ConfigService;
import org.flexlb.config.service.DeploymentIdentity;
import org.flexlb.util.network.GeneralHttpNettyService;
import org.springframework.core.env.Environment;

/**
 * Supplies the local Mock Master's loopback address before the real ZooKeeper election initializes.
 * The base service owns election, notifications, and shutdown.
 */
public final class LoopbackZookeeperMasterElectService extends ZookeeperMasterElectService {
    public LoopbackZookeeperMasterElectService(
            GeneralHttpNettyService generalHttpNettyService,
            MasterMetrics masterMetrics,
            Environment environment,
            ConfigService configService,
            DeploymentIdentity deploymentIdentity) {
        super(
                generalHttpNettyService,
                masterMetrics,
                environment,
                configService,
                deploymentIdentity,
                environment.getRequiredProperty("server.address"));
    }
}
