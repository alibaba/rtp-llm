package org.flexlb.master;

import org.apache.curator.framework.recipes.leader.LeaderSelector;
import org.flexlb.util.network.GeneralHttpNettyService;
import org.flexlb.worker.status.ServiceMetricsReporter;
import org.junit.jupiter.api.Test;
import org.springframework.core.env.Environment;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.List;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Checks shutdown signals when a node goes offline during master notification.
 */
class ZkLeadershipShutdownRaceTest {
    @Test
    void offlineDuringLeadershipNotificationCannotLoseShutdownSignal() throws Exception {
        var reporter = mock(ServiceMetricsReporter.class);
        var service =
                new ZookeeperMasterElectService(
                        mock(GeneralHttpNettyService.class),
                        reporter,
                        mock(Environment.class),
                        org.flexlb.balance.schedule.SchedulingTestConfig.configService(
                                new org.flexlb.config.FlexlbConfig()),
                        org.mockito.Mockito.mock(
                                org.flexlb.config.service.DeploymentIdentity.class));
        var selector = mock(LeaderSelector.class);
        when(selector.getParticipants()).thenReturn(List.of());
        ReflectionTestUtils.setField(service, "leaderSelector", selector);
        ReflectionTestUtils.setField(
                service,
                "localNode",
                new ZookeeperMasterElectService.LocalNodeIdentity("127.0.0.1", "7001", null));
        doAnswer(
                        call -> {
                            service.offline();
                            return null;
                        })
                .when(reporter)
                .reportPrefillBalanceMasterEvent(ZkMasterEvent.MASTER_TAKE_LEADERSHIP);
        var failure = new AtomicReference<Throwable>();
        Thread leader =
                new Thread(
                        () -> {
                            try {
                                service.takeLeadership(null);
                            } catch (Throwable unexpected) {
                                failure.set(unexpected);
                            }
                        },
                        "leadership-shutdown-race");
        leader.start();
        try {
            leader.join(2_000L);
            assertFalse(
                    leader.isAlive(), "offline must release a leadership callback still starting");
            assertNull(failure.get());
            assertFalse(service.isMaster());
            verify(selector).close();
        } finally {
            leader.interrupt();
            leader.join(2_000L);
        }
    }
}
