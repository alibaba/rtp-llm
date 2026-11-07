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
    @ValueSource(
            strings = {"spectrum:workspace:application:deployment", "dash_pd:deployment:master"})
    void uses_deployment_identity_for_master_change_notifications(String deploymentId) {
        DeploymentIdentity identity = mock(DeploymentIdentity.class);
        when(identity.getDeploymentId()).thenReturn(deploymentId);
        ZookeeperMasterElectService electionService = mock(ZookeeperMasterElectService.class);

        MasterStatusService service = masterStatus(electionService, identity, true);
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
        MasterStatusService service = masterStatus(electionService, identity, false);
        MasterChangeNotifyReq request = new MasterChangeNotifyReq();
        request.setRoleId(notifiedRoleId);

        MasterChangeNotifyResp response = service.handleMasterChange(request);

        assertThat(response.isSuccess()).isFalse();
        verifyNoInteractions(identity);
        org.mockito.Mockito.verify(electionService, org.mockito.Mockito.never())
                .updateLatestMaster();
    }

    @ParameterizedTest
    @NullSource
    @ValueSource(strings = {"other-deployment"})
    void enabledConsistencyRejectsMissingOrDifferentNotificationRoleId(String notifiedRoleId) {
        DeploymentIdentity identity = mock(DeploymentIdentity.class);
        when(identity.getDeploymentId()).thenReturn("current-deployment");
        ZookeeperMasterElectService electionService = mock(ZookeeperMasterElectService.class);
        MasterStatusService service = masterStatus(electionService, identity, true);
        MasterChangeNotifyReq request = new MasterChangeNotifyReq();
        request.setRoleId(notifiedRoleId);

        MasterChangeNotifyResp response = service.handleMasterChange(request);

        assertThat(response.isSuccess()).isFalse();
        org.mockito.Mockito.verify(electionService, org.mockito.Mockito.never())
                .updateLatestMaster();
    }

    @ParameterizedTest
    @ValueSource(
            strings = {"spectrum:workspace:application:deployment", "dash_pd:deployment:master"})
    void uses_deployment_identity_for_zookeeper_election_path(String deploymentId) {
        DeploymentIdentity identity = mock(DeploymentIdentity.class);
        when(identity.getDeploymentId()).thenReturn(deploymentId);
        ZookeeperMasterElectService service =
                new ZookeeperMasterElectService(
                        mock(GeneralHttpNettyService.class),
                        mock(EngineHealthReporter.class),
                        mock(Environment.class),
                        configService(false),
                        identity);

        ReflectionTestUtils.setField(
                service, "consistencyConfig", new ZookeeperConsistencyConfig());
        assertThat(service.localNodeIdentity().roleId()).isEqualTo(deploymentId);
    }

    @Test
    void disabledConsistencyDoesNotRequireDeploymentIdentity() {
        DeploymentIdentity identity = mock(DeploymentIdentity.class);

        var election =
                new ZookeeperMasterElectService(
                        mock(GeneralHttpNettyService.class),
                        mock(EngineHealthReporter.class),
                        mock(Environment.class),
                        configService(false),
                        identity);
        new MasterStatusService(election);

        verifyNoInteractions(identity);
    }

    @Test
    void closingSpringContextDestroysZookeeperElection() {
        CuratorFramework curatorClient = mock(CuratorFramework.class);
        ZookeeperMasterElectService electionService =
                new ZookeeperMasterElectService(
                        mock(GeneralHttpNettyService.class),
                        mock(EngineHealthReporter.class),
                        mock(Environment.class),
                        configService(false),
                        mock(DeploymentIdentity.class));
        ReflectionTestUtils.setField(electionService, "client", curatorClient);
        ReflectionTestUtils.setField(
                electionService, "consistencyConfig", new ZookeeperConsistencyConfig());
        ReflectionTestUtils.setField(
                electionService,
                "localNode",
                new ZookeeperMasterElectService.LocalNodeIdentity("127.0.0.1", "7001", "test"));
        try (AnnotationConfigApplicationContext context =
                new AnnotationConfigApplicationContext()) {
            context.registerBean(
                    MasterStatusService.class, () -> new MasterStatusService(electionService));
            context.refresh();
            verifyNoInteractions(curatorClient);
        }

        verify(curatorClient).close();
    }

    private MasterStatusService masterStatus(
            ZookeeperMasterElectService election, DeploymentIdentity identity, boolean enabled) {
        when(election.isConsistencyEnabled()).thenReturn(enabled);
        String roleId = enabled ? identity.getDeploymentId() : null;
        when(election.localNodeIdentity())
                .thenReturn(
                        new ZookeeperMasterElectService.LocalNodeIdentity(
                                "127.0.0.1", "7001", roleId));
        org.mockito.Mockito.clearInvocations(election);
        return new MasterStatusService(election);
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
