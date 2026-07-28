package org.flexlb.cache.match;

import org.flexlb.cache.domain.CacheMatchSource;
import org.flexlb.config.CacheMatchConfiguration;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.kvcm.KvcmHealthSnapshot;
import org.flexlb.dao.kvcm.KvcmHealthState;
import org.flexlb.dao.route.KvcmConfig;
import org.flexlb.dao.route.LocalStandbyConfig;
import org.flexlb.dao.route.ServiceRoute;
import org.flexlb.engine.grpc.client.KvcmGrpcClient;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import java.util.function.Consumer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class CacheMatchFailoverManagerTest {

    @Test
    void automaticallyFollowsKvcmClientHealth() {
        KvcmGrpcClient client = mock(KvcmGrpcClient.class);
        when(client.healthSnapshot()).thenReturn(
                health(KvcmHealthState.HEALTHY, 0, 0, 0, "initial"));
        CacheMatchFailoverManager manager =
                new CacheMatchFailoverManager(configuration(true), client);
        Consumer<KvcmHealthSnapshot> healthSnapshotListener = healthSnapshotListener(client);

        healthSnapshotListener.accept(
                health(KvcmHealthState.UNHEALTHY, 3, 0, 0, "heartbeat failure"));
        assertEquals(CacheMatchSource.LOCAL_STANDBY, manager.activeSource());
        assertEquals("heartbeat failure", manager.lastFailoverReason());

        healthSnapshotListener.accept(
                health(KvcmHealthState.HEALTHY, 0, 3, 0, "heartbeat recovery"));
        assertEquals(CacheMatchSource.KVCM, manager.activeSource());
        assertEquals("KVCM heartbeat recovered", manager.lastFailoverReason());
    }

    @Test
    void keepsKvcmActiveUntilManualFailoverWhenAutoSwitchIsDisabled() {
        KvcmGrpcClient client = mock(KvcmGrpcClient.class);
        when(client.healthSnapshot())
                .thenReturn(health(KvcmHealthState.UNHEALTHY, 3, 0, 10, "query failure"));
        CacheMatchFailoverManager manager =
                new CacheMatchFailoverManager(configuration(false), client);

        assertEquals(CacheMatchSource.KVCM, manager.activeSource());

        manager.activateFallbackManually();
        assertEquals(CacheMatchSource.LOCAL_STANDBY, manager.activeSource());

        assertThrows(IllegalStateException.class, manager::recoverPrimaryManually);
        assertEquals(CacheMatchSource.LOCAL_STANDBY, manager.activeSource());
    }

    @Test
    void manualFallbackRemainsActiveAfterKvcmRecovers() {
        KvcmGrpcClient client = mock(KvcmGrpcClient.class);
        when(client.healthSnapshot())
                .thenReturn(health(KvcmHealthState.HEALTHY, 0, 3, 0, "heartbeat recovery"));
        CacheMatchFailoverManager manager =
                new CacheMatchFailoverManager(configuration(true), client);

        manager.activateFallbackManually();

        assertEquals(CacheMatchSource.LOCAL_STANDBY, manager.activeSource());

        manager.recoverPrimaryManually();
        assertEquals(CacheMatchSource.KVCM, manager.activeSource());
    }

    @Test
    void manualRecoveryDefersToAutomaticFailoverForUnhealthyKvcm() {
        KvcmGrpcClient client = mock(KvcmGrpcClient.class);
        when(client.healthSnapshot())
                .thenReturn(health(KvcmHealthState.UNHEALTHY, 3, 0, 0, "heartbeat failure"));
        CacheMatchFailoverManager manager =
                new CacheMatchFailoverManager(configuration(true), client);
        Consumer<KvcmHealthSnapshot> healthSnapshotListener = healthSnapshotListener(client);

        manager.activateFallbackManually();
        manager.recoverPrimaryManually();

        assertEquals(CacheMatchSource.LOCAL_STANDBY, manager.activeSource());

        healthSnapshotListener.accept(
                health(KvcmHealthState.HEALTHY, 0, 3, 0, "heartbeat recovery"));
        assertEquals(CacheMatchSource.KVCM, manager.activeSource());
    }

    @SuppressWarnings("unchecked")
    private Consumer<KvcmHealthSnapshot> healthSnapshotListener(KvcmGrpcClient client) {
        ArgumentCaptor<Consumer<KvcmHealthSnapshot>> captor =
                ArgumentCaptor.forClass(Consumer.class);
        verify(client).setHealthSnapshotListener(captor.capture());
        return captor.getValue();
    }

    private KvcmHealthSnapshot health(KvcmHealthState state, int heartbeatFailures, int heartbeatSuccesses, int queryFailures, String reason) {
        return new KvcmHealthSnapshot(
                state,
                heartbeatFailures,
                heartbeatSuccesses,
                queryFailures,
                100,
                0,
                reason);
    }

    private CacheMatchConfiguration configuration(boolean autoSwitch) {
        LocalStandbyConfig standby = new LocalStandbyConfig();
        standby.setAutoSwitch(autoSwitch);

        KvcmConfig kvcm = new KvcmConfig();
        kvcm.setEnabled(true);
        kvcm.setLocalStandby(standby);

        ServiceRoute route = new ServiceRoute();
        route.setServiceId("test-service");
        route.setKvcm(kvcm);

        ModelMetaConfig config = new ModelMetaConfig();
        config.putServiceRoute(route.getServiceId(), route);
        return new CacheMatchConfiguration(config);
    }
}
