package org.flexlb.consistency;

import org.apache.curator.framework.recipes.leader.CancelLeadershipException;
import org.apache.curator.framework.recipes.leader.LeaderSelector;
import org.apache.curator.framework.recipes.leader.Participant;
import org.apache.curator.framework.state.ConnectionState;
import org.flexlb.constant.ZkMasterEvent;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.transport.GeneralHttpNettyService;
import org.junit.jupiter.api.Test;
import org.springframework.core.env.Environment;

import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class ZkLeadershipLifecycleTest {
    @Test
    void olderLeaderLookupCannotOverwriteANewerLookup() throws Exception {
        var service = new ZookeeperMasterElectService(mock(GeneralHttpNettyService.class),
                mock(EngineHealthReporter.class), mock(Environment.class));
        var selector = mock(LeaderSelector.class);
        var oldLeader = mock(Participant.class);
        var newLeader = mock(Participant.class);
        when(oldLeader.getId()).thenReturn("127.0.0.9");
        when(newLeader.getId()).thenReturn("127.0.0.10");
        CountDownLatch entered = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        var lookups = new AtomicInteger();
        when(selector.getLeader()).thenAnswer(call -> {
            if (lookups.incrementAndGet() == 1) {
                entered.countDown();
                assertTrue(release.await(3, TimeUnit.SECONDS));
                return oldLeader;
            }
            return newLeader;
        });
        org.springframework.test.util.ReflectionTestUtils.setField(service, "leaderSelector", selector);
        var executor = Executors.newFixedThreadPool(2);
        var lookup = executor.submit(service::updateLatestMaster);
        try {
            assertTrue(entered.await(2, TimeUnit.SECONDS));
            var newerLookup = executor.submit(service::updateLatestMaster);
            release.countDown();
            lookup.get(2, TimeUnit.SECONDS);
            newerLookup.get(2, TimeUnit.SECONDS);
            assertEquals("127.0.0.10", service.getMasterHostIp());
        } finally {
            release.countDown();
            executor.shutdownNow();
            assertTrue(executor.awaitTermination(3, TimeUnit.SECONDS));
        }
    }

    @Test
    void oldLeaderLookupCannotCrossAConnectionRevision() throws Exception {
        var service = new ZookeeperMasterElectService(mock(GeneralHttpNettyService.class),
                mock(EngineHealthReporter.class), mock(Environment.class));
        var selector = mock(LeaderSelector.class);
        var oldLeader = mock(Participant.class);
        when(oldLeader.getId()).thenReturn("127.0.0.9");
        CountDownLatch entered = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        when(selector.getLeader()).thenAnswer(call -> {
            entered.countDown();
            assertTrue(release.await(3, TimeUnit.SECONDS));
            return oldLeader;
        });
        org.springframework.test.util.ReflectionTestUtils.setField(service, "leaderSelector", selector);
        var executor = Executors.newFixedThreadPool(2);
        var lookup = executor.submit(service::updateLatestMaster);
        try {
            assertTrue(entered.await(2, TimeUnit.SECONDS));
            executor.submit(() -> assertThrows(CancelLeadershipException.class,
                    () -> service.stateChanged(null, ConnectionState.SUSPENDED)))
                    .get(500, TimeUnit.MILLISECONDS);
            service.stateChanged(null, ConnectionState.RECONNECTED);
            release.countDown();
            lookup.get(2, TimeUnit.SECONDS);
            assertNull(service.getMasterHostIp(), "the pre-disconnect lookup cannot restore the old leader");
        } finally {
            release.countDown();
            executor.shutdownNow();
            assertTrue(executor.awaitTermination(3, TimeUnit.SECONDS));
        }
    }

    @Test
    @org.junit.jupiter.api.Timeout(5)
    void suspendedSessionCannotReenterLeadershipBeforeReconnection() {
        EngineHealthReporter reporter = mock(EngineHealthReporter.class);
        ZookeeperMasterElectService service = new ZookeeperMasterElectService(
                mock(GeneralHttpNettyService.class), reporter, mock(Environment.class));
        assertThrows(CancelLeadershipException.class,
                () -> service.stateChanged(null, ConnectionState.SUSPENDED));
        service.takeLeadership(null);
        assertFalse(service.isMaster());
    }

    @Test
    void offlineCannotLoseTheLeadershipReleaseSignal() throws Exception {
        exercise(true);
    }

    @Test
    void suspendedConnectionWithdrawsLeadershipBeforeTheCallbackExits() throws Exception {
        exercise(false);
    }

    private void exercise(boolean offline) throws Exception {
        EngineHealthReporter reporter = mock(EngineHealthReporter.class);
        ZookeeperMasterElectService service = new ZookeeperMasterElectService(
                mock(GeneralHttpNettyService.class), reporter, mock(Environment.class));
        org.springframework.test.util.ReflectionTestUtils.setField(service, "localNode",
                new ZookeeperMasterElectService.LocalNodeIdentity("127.0.0.1", "7001", "test-role"));
        LeaderSelector selector = mock(LeaderSelector.class);
        when(selector.getParticipants()).thenReturn(List.of());
        org.springframework.test.util.ReflectionTestUtils.setField(service, "leaderSelector", selector);
        CountDownLatch published = new CountDownLatch(1);
        CountDownLatch resume = new CountDownLatch(1);
        doAnswer(call -> {
            published.countDown();
            assertTrue(resume.await(3, TimeUnit.SECONDS));
            return null;
        }).when(reporter).reportPrefillBalanceMasterEvent(eq(ZkMasterEvent.MASTER_TAKE_LEADERSHIP));
        var executor = Executors.newSingleThreadExecutor();
        var leadership = executor.submit(() -> service.takeLeadership(null));
        try {
            assertTrue(published.await(2, TimeUnit.SECONDS));
            assertTrue(service.isMaster());
            if (offline) {
                service.offline();
                resume.countDown();
                leadership.get(500, TimeUnit.MILLISECONDS);
                assertFalse(service.isMaster());
            } else {
                assertThrows(CancelLeadershipException.class,
                        () -> service.stateChanged(null, ConnectionState.SUSPENDED));
                assertFalse(service.isMaster(), "a suspended ZK session cannot authorize local master scheduling");
            }
        } finally {
            resume.countDown();
            leadership.cancel(true);
            executor.shutdownNow();
            assertTrue(executor.awaitTermination(3, TimeUnit.SECONDS));
        }
    }
}
