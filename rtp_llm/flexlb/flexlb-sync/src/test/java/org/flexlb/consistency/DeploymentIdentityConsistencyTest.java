package org.flexlb.consistency;

import org.flexlb.config.ConfigService;
import org.flexlb.config.DeploymentIdentity;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.domain.consistency.MasterChangeNotifyReq;
import org.flexlb.domain.consistency.MasterChangeNotifyResp;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.transport.GeneralHttpNettyService;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.core.env.Environment;
import org.springframework.test.util.ReflectionTestUtils;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class DeploymentIdentityConsistencyTest {

    @ParameterizedTest
    @ValueSource(strings = {"spectrum:workspace:application:deployment", "dash_pd:deployment:master"})
    void uses_deployment_identity_for_master_change_notifications(String deploymentId) {
        DeploymentIdentity identity = mock(DeploymentIdentity.class);
        when(identity.getDeploymentId()).thenReturn(deploymentId);

        LBStatusConsistencyService service = new LBStatusConsistencyService(
                mock(ZookeeperMasterElectService.class), mock(Environment.class), configService(), identity);
        MasterChangeNotifyReq request = new MasterChangeNotifyReq();
        request.setRoleId(deploymentId);

        MasterChangeNotifyResp response = service.handleMasterChange(request);

        assertThat(response.isSuccess()).isTrue();
    }

    @ParameterizedTest
    @ValueSource(strings = {"spectrum:workspace:application:deployment", "dash_pd:deployment:master"})
    void uses_deployment_identity_for_zookeeper_election_path(String deploymentId) {
        DeploymentIdentity identity = mock(DeploymentIdentity.class);
        when(identity.getDeploymentId()).thenReturn(deploymentId);
        ZookeeperMasterElectService service = new ZookeeperMasterElectService(
                mock(GeneralHttpNettyService.class), mock(EngineHealthReporter.class),
                mock(Environment.class), configService(), identity);

        ReflectionTestUtils.invokeMethod(service, "initializeRoleId");

        assertThat(ReflectionTestUtils.getField(service, "roleId"))
                .isEqualTo(deploymentId);
    }

    private ConfigService configService() {
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(new FlexlbConfig());
        return configService;
    }
}
