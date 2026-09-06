package org.flexlb.consistency;

import org.flexlb.config.ConfigService;
import org.flexlb.config.DeploymentIdentity;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.domain.consistency.MasterChangeNotifyReq;
import org.flexlb.domain.consistency.MasterChangeNotifyResp;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.transport.GeneralHttpNettyService;
import org.junit.jupiter.api.Test;
import org.springframework.core.env.Environment;
import org.springframework.test.util.ReflectionTestUtils;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class DeploymentIdentityConsistencyTest {

    @Test
    void uses_deployment_identity_for_master_change_notifications() {
        DeploymentIdentity identity = mock(DeploymentIdentity.class);
        when(identity.getDeploymentId()).thenReturn("spectrum:workspace:application:deployment");

        LBStatusConsistencyService service = new LBStatusConsistencyService(
                mock(ZookeeperMasterElectService.class), mock(Environment.class), configService(), identity);
        MasterChangeNotifyReq request = new MasterChangeNotifyReq();
        request.setRoleId("spectrum:workspace:application:deployment");

        MasterChangeNotifyResp response = service.handleMasterChange(request);

        assertThat(response.isSuccess()).isTrue();
    }

    @Test
    void uses_deployment_identity_for_zookeeper_election_path() {
        DeploymentIdentity identity = mock(DeploymentIdentity.class);
        when(identity.getDeploymentId()).thenReturn("spectrum:workspace:application:deployment");
        ZookeeperMasterElectService service = new ZookeeperMasterElectService(
                mock(GeneralHttpNettyService.class), mock(EngineHealthReporter.class),
                mock(Environment.class), configService(), identity);

        ReflectionTestUtils.invokeMethod(service, "initializeRoleId");

        assertThat(ReflectionTestUtils.getField(service, "roleId"))
                .isEqualTo("spectrum:workspace:application:deployment");
    }

    private ConfigService configService() {
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(new FlexlbConfig());
        return configService;
    }
}
