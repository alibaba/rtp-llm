package org.flexlb.mock.master;

import org.flexlb.config.ConfigService;
import org.flexlb.config.DeploymentIdentity;
import org.flexlb.consistency.ZookeeperMasterElectService;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.transport.GeneralHttpNettyService;
import org.springframework.core.env.Environment;

/**
 * Supplies the local Mock Master's loopback address before the real ZooKeeper election initializes.
 * The base service owns election, notifications, and shutdown.
 */
public final class LoopbackZookeeperMasterElectService extends ZookeeperMasterElectService {
    public LoopbackZookeeperMasterElectService(
            GeneralHttpNettyService generalHttpNettyService,
            EngineHealthReporter engineHealthReporter,
            Environment environment,
            ConfigService configService,
            DeploymentIdentity deploymentIdentity) {
        super(
                generalHttpNettyService,
                engineHealthReporter,
                environment,
                configService,
                deploymentIdentity,
                environment.getRequiredProperty("server.address"));
    }
}
