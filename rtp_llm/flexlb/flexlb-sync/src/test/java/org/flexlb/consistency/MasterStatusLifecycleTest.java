package org.flexlb.consistency;

import org.flexlb.domain.consistency.LBConsistencyConfig;
import org.flexlb.domain.consistency.MasterChangeNotifyReq;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.transport.GeneralHttpNettyService;
import org.flexlb.util.JsonUtils;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import org.springframework.core.env.Environment;
import org.springframework.mock.env.MockEnvironment;
import org.springframework.test.util.ReflectionTestUtils;

import java.net.InetAddress;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class MasterStatusLifecycleTest {

    @ParameterizedTest
    @ValueSource(booleans = {true, false})
    void contextShutdownClosesOnlyEnabledElection(boolean enabled) {
        var election = mock(ZookeeperMasterElectService.class);
        var config = new LBConsistencyConfig();
        config.setNeedConsistency(enabled);
        when(election.getLbConsistencyConfig()).thenReturn(config);
        when(election.localNodeIdentity()).thenReturn(
                new ZookeeperMasterElectService.LocalNodeIdentity("127.0.0.1", "7001", "master-role"));
        try (var context = new AnnotationConfigApplicationContext()) {
            context.registerBean(MasterStatusService.class,
                    () -> new MasterStatusService(election));
            context.refresh();
            verify(election, times(0)).destroy();
        }
        verify(election, times(enabled ? 1 : 0)).destroy();
    }

    @ParameterizedTest
    @CsvSource({"true,true", "true,false", "false,true", "false,false"})
    void masterChangeRefreshesOnlyAnEnabledMatchingRole(boolean enabled, boolean matchingRole) {
        var election = mock(ZookeeperMasterElectService.class);
        var config = new LBConsistencyConfig();
        config.setNeedConsistency(enabled);
        when(election.getLbConsistencyConfig()).thenReturn(config);
        when(election.localNodeIdentity()).thenReturn(
                new ZookeeperMasterElectService.LocalNodeIdentity("127.0.0.1", "7001", "master-role"));
        var service = new MasterStatusService(election);
        var request = new MasterChangeNotifyReq();
        request.setRoleId(matchingRole ? "master-role" : "other-role");

        assertEquals(enabled && matchingRole, service.handleMasterChange(request).isSuccess());
        verify(election, times(enabled && matchingRole ? 1 : 0)).updateLatestMaster();
    }

    @Test
    void disabledElectionDoesNotResolveIdentityOrParsePort() {
        Environment environment = mock(Environment.class);
        try (var hostLookup = mockStatic(InetAddress.class)) {
            var election = new ZookeeperMasterElectService(mock(GeneralHttpNettyService.class),
                    mock(EngineHealthReporter.class), environment);

            assertFalse(election.getLbConsistencyConfig().isNeedConsistency());
            assertNull(ReflectionTestUtils.getField(election, "localNode"));
            hostLookup.verifyNoInteractions();
            verifyNoInteractions(environment);
        }
    }

    @Test
    void standaloneStatusCapturesIdentityOnceAndKeepsTheRawPort() throws Exception {
        var environment = new MockEnvironment().withProperty("server.port", "not-a-number");
        var localAddress = mock(InetAddress.class);
        when(localAddress.getHostAddress()).thenReturn("127.0.0.7");
        try (var hostLookup = mockStatic(InetAddress.class)) {
            hostLookup.when(InetAddress::getLocalHost).thenReturn(localAddress);
            var election = new ZookeeperMasterElectService(mock(GeneralHttpNettyService.class),
                    mock(EngineHealthReporter.class), environment);
            hostLookup.verifyNoInteractions();

            var service = new MasterStatusService(election);
            var identity = election.localNodeIdentity();
            assertSame(identity, ReflectionTestUtils.getField(service, "localNode"));
            environment.setProperty("server.port", "changed-after-capture");
            assertSame(identity, election.localNodeIdentity());
            Map<?, ?> snapshot = JsonUtils.toObject(service.dumpLBStatus().getLbStatus(), Map.class);
            assertEquals("127.0.0.7", snapshot.get("local_host"));
            assertEquals("not-a-number", snapshot.get("server_port"));
            assertEquals(false, snapshot.get("consistency_enabled"));
            assertEquals(false, snapshot.get("master"));
            assertNull(snapshot.get("master_host"));
            assertEquals("127.0.0.7", service.getLocalHostIp());
            hostLookup.verify(InetAddress::getLocalHost, times(1));
        }
    }

    @Test
    void electedStatusUsesTheIdentityAlreadyCapturedByElection() {
        var environment = mock(Environment.class);
        var election = new ZookeeperMasterElectService(mock(GeneralHttpNettyService.class),
                mock(EngineHealthReporter.class), environment);
        var identity = new ZookeeperMasterElectService.LocalNodeIdentity("127.0.0.8", "07001", "master-role");
        ReflectionTestUtils.setField(election, "localNode", identity);
        election.getLbConsistencyConfig().setNeedConsistency(true);
        ReflectionTestUtils.setField(election, "isMaster", true);
        try (var hostLookup = mockStatic(InetAddress.class)) {
            var service = new MasterStatusService(election);

            assertSame(identity, ReflectionTestUtils.getField(service, "localNode"));
            assertEquals("127.0.0.8:07001", service.getMasterHostIpPort());
            hostLookup.verifyNoInteractions();
            verifyNoInteractions(environment);
        }
    }
}
