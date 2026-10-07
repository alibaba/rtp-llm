package org.flexlb.consistency;

import org.apache.curator.framework.CuratorFramework;
import org.flexlb.config.ConfigService;
import org.flexlb.config.DeploymentIdentity;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.ZookeeperConsistencyConfig;
import org.flexlb.domain.consistency.MasterChangeNotifyReq;
import org.flexlb.domain.consistency.MasterChangeNotifyResp;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.transport.GeneralHttpNettyService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.NullSource;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import org.springframework.core.env.Environment;
import org.springframework.test.util.ReflectionTestUtils;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class DeploymentIdentityConsistencyTest {

    @ParameterizedTest
    @ValueSource(strings = {"spectrum:workspace:application:deployment", "dash_pd:deployment:master"})
    void uses_deployment_identity_for_master_change_notifications(String deploymentId) {
        DeploymentIdentity identity = mock(DeploymentIdentity.class);
        when(identity.getDeploymentId()).thenReturn(deploymentId);
        ZookeeperMasterElectService electionService = mock(ZookeeperMasterElectService.class);

        LBStatusConsistencyService service = new LBStatusConsistencyService(
                electionService, mock(Environment.class),
                configService(true), identity);
        MasterChangeNotifyReq request = new MasterChangeNotifyReq();
        request.setRoleId(deploymentId);

        MasterChangeNotifyResp response = service.handleMasterChange(request);

        assertThat(response.isSuccess()).isTrue();
        verify(electionService).updateLatestMaster();
    }

    @ParameterizedTest
    @NullSource
    @ValueSource(strings = {"other-deployment"})
    void disabledConsistencyRejectsMasterChangeNotifications(String notifiedRoleId) {
        DeploymentIdentity identity = mock(DeploymentIdentity.class);
        ZookeeperMasterElectService electionService = mock(ZookeeperMasterElectService.class);
        LBStatusConsistencyService service = new LBStatusConsistencyService(
                electionService, mock(Environment.class), configService(false), identity);
        MasterChangeNotifyReq request = new MasterChangeNotifyReq();
        request.setRoleId(notifiedRoleId);

        MasterChangeNotifyResp response = service.handleMasterChange(request);

        assertThat(response.isSuccess()).isFalse();
        verifyNoInteractions(identity, electionService);
    }

    @ParameterizedTest
    @NullSource
    @ValueSource(strings = {"other-deployment"})
    void enabledConsistencyRejectsMissingOrDifferentNotificationRoleId(String notifiedRoleId) {
        DeploymentIdentity identity = mock(DeploymentIdentity.class);
        when(identity.getDeploymentId()).thenReturn("current-deployment");
        ZookeeperMasterElectService electionService = mock(ZookeeperMasterElectService.class);
        LBStatusConsistencyService service = new LBStatusConsistencyService(
                electionService, mock(Environment.class), configService(true), identity);
        MasterChangeNotifyReq request = new MasterChangeNotifyReq();
        request.setRoleId(notifiedRoleId);

        MasterChangeNotifyResp response = service.handleMasterChange(request);

        assertThat(response.isSuccess()).isFalse();
        verifyNoInteractions(electionService);
    }

    @ParameterizedTest
    @ValueSource(strings = {"spectrum:workspace:application:deployment", "dash_pd:deployment:master"})
    void uses_deployment_identity_for_zookeeper_election_path(String deploymentId) {
        DeploymentIdentity identity = mock(DeploymentIdentity.class);
        when(identity.getDeploymentId()).thenReturn(deploymentId);
        ZookeeperMasterElectService service = new ZookeeperMasterElectService(
                mock(GeneralHttpNettyService.class), mock(EngineHealthReporter.class),
                mock(Environment.class), configService(false), identity);

        ReflectionTestUtils.invokeMethod(service, "initializeRoleId");

        assertThat(ReflectionTestUtils.getField(service, "roleId"))
                .isEqualTo(deploymentId);
    }

    @Test
    void disabledConsistencyDoesNotRequireDeploymentIdentity() {
        DeploymentIdentity identity = mock(DeploymentIdentity.class);

        new LBStatusConsistencyService(
                mock(ZookeeperMasterElectService.class), mock(Environment.class),
                configService(false), identity);

        verifyNoInteractions(identity);
    }

    @Test
    void closingSpringContextDestroysZookeeperElection() {
        CuratorFramework curatorClient = mock(CuratorFramework.class);
        ZookeeperMasterElectService electionService = new ZookeeperMasterElectService(
                mock(GeneralHttpNettyService.class), mock(EngineHealthReporter.class),
                mock(Environment.class), configService(false), mock(DeploymentIdentity.class));
        ReflectionTestUtils.setField(electionService, "client", curatorClient);
        try (AnnotationConfigApplicationContext context = new AnnotationConfigApplicationContext()) {
            context.registerBean(LBStatusConsistencyService.class,
                    () -> new LBStatusConsistencyService(electionService, mock(Environment.class),
                            configService(true), mock(DeploymentIdentity.class)));
            context.refresh();
            verifyNoInteractions(curatorClient);
        }

        verify(curatorClient).close();
    }

    private ConfigService configService(boolean consistencyEnabled) {
        ConfigService configService = mock(ConfigService.class);
        FlexlbConfig config = new FlexlbConfig();
        if (consistencyEnabled) {
            config.setConsistency(new ZookeeperConsistencyConfig());
        }
        when(configService.loadBalanceConfig()).thenReturn(config);
        return configService;
    }
}
