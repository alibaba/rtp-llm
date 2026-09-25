package org.flexlb.cache.match;

import org.flexlb.cache.domain.CacheMatchSource;
import org.flexlb.cache.telemetry.CacheMetricsReporter;
import org.flexlb.config.CacheMatchConfiguration;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.kvcm.KvcmHealthSnapshot;
import org.flexlb.dao.kvcm.KvcmHealthState;
import org.flexlb.dao.route.KvcmConfig;
import org.flexlb.dao.route.ServiceRoute;
import org.flexlb.engine.grpc.client.KvcmGrpcClient;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;

import static org.flexlb.cache.CacheMatchTestConfigurations.kvcm;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class CacheMatchFailoverManagerTest {

    @Test
    void automaticallyFollowsKvcmClientHealth() {
        KvcmGrpcClient client = mock(KvcmGrpcClient.class);
        CacheMetricsReporter metricsReporter = mock(CacheMetricsReporter.class);
        when(client.healthSnapshot()).thenReturn(
                health(KvcmHealthState.HEALTHY, 0, 0, 0, "initial"));
        CacheMatchFailoverManager manager =
                new CacheMatchFailoverManager(
                        configuration(true), client, metricsReporter);
        Consumer<KvcmHealthSnapshot> healthSnapshotListener = healthSnapshotListener(client);

        healthSnapshotListener.accept(
                health(KvcmHealthState.UNHEALTHY, 3, 0, 0, "heartbeat failure"));
        assertEquals(CacheMatchSource.LOCAL_STANDBY, manager.activeSource());
        assertEquals("heartbeat failure", manager.lastFailoverReason());

        healthSnapshotListener.accept(
                health(KvcmHealthState.HEALTHY, 0, 3, 0, "heartbeat recovery"));
        assertEquals(CacheMatchSource.KVCM, manager.activeSource());
        assertEquals("KVCM heartbeat recovered", manager.lastFailoverReason());
        verify(metricsReporter).reportCacheMatchSourceChange(
                CacheMatchSource.KVCM, CacheMatchSource.LOCAL_STANDBY);
        verify(metricsReporter).reportCacheMatchSourceChange(
                CacheMatchSource.LOCAL_STANDBY, CacheMatchSource.KVCM);
    }

    @Test
    void keepsKvcmActiveUntilManualFailoverWhenAutoSwitchIsDisabled() {
        KvcmGrpcClient client = mock(KvcmGrpcClient.class);
        AtomicReference<KvcmHealthSnapshot> currentHealth = new AtomicReference<>(
                health(KvcmHealthState.UNHEALTHY, 3, 0, 10, "query failure"));
        when(client.healthSnapshot()).thenAnswer(ignored -> currentHealth.get());
        CacheMatchFailoverManager manager =
                new CacheMatchFailoverManager(
                        configuration(false), client, mock(CacheMetricsReporter.class));
        Consumer<KvcmHealthSnapshot> healthSnapshotListener = healthSnapshotListener(client);

        assertEquals(CacheMatchSource.KVCM, manager.activeSource());

        manager.activateFallbackManually();
        assertEquals(CacheMatchSource.LOCAL_STANDBY, manager.activeSource());

        assertThrows(IllegalStateException.class, manager::recoverPrimaryManually);
        assertEquals(CacheMatchSource.LOCAL_STANDBY, manager.activeSource());

        currentHealth.set(
                health(KvcmHealthState.HEALTHY, 0, 3, 0, "heartbeat recovery"));
        healthSnapshotListener.accept(currentHealth.get());
        assertEquals(CacheMatchSource.LOCAL_STANDBY, manager.activeSource());

        manager.recoverPrimaryManually();
        assertEquals(CacheMatchSource.KVCM, manager.activeSource());
    }

    @Test
    void manualFallbackRemainsActiveAfterKvcmRecovers() {
        KvcmGrpcClient client = mock(KvcmGrpcClient.class);
        when(client.healthSnapshot())
                .thenReturn(health(KvcmHealthState.HEALTHY, 0, 3, 0, "heartbeat recovery"));
        CacheMatchFailoverManager manager =
                new CacheMatchFailoverManager(
                        configuration(true), client, mock(CacheMetricsReporter.class));

        manager.activateFallbackManually();

        assertEquals(CacheMatchSource.LOCAL_STANDBY, manager.activeSource());

        manager.recoverPrimaryManually();
        assertEquals(CacheMatchSource.KVCM, manager.activeSource());
    }

    @Test
    void rejectsManualRecoveryForUnhealthyKvcmWhenAutoSwitchIsEnabled() {
        KvcmGrpcClient client = mock(KvcmGrpcClient.class);
        AtomicReference<KvcmHealthSnapshot> currentHealth = new AtomicReference<>(
                health(KvcmHealthState.UNHEALTHY, 3, 0, 0, "heartbeat failure"));
        when(client.healthSnapshot()).thenAnswer(ignored -> currentHealth.get());
        CacheMatchFailoverManager manager =
                new CacheMatchFailoverManager(
                        configuration(true), client, mock(CacheMetricsReporter.class));
        Consumer<KvcmHealthSnapshot> healthSnapshotListener = healthSnapshotListener(client);

        manager.activateFallbackManually();
        assertThrows(IllegalStateException.class, manager::recoverPrimaryManually);
        assertEquals(CacheMatchSource.LOCAL_STANDBY, manager.activeSource());

        currentHealth.set(
                health(KvcmHealthState.HEALTHY, 0, 3, 0, "heartbeat recovery"));
        healthSnapshotListener.accept(currentHealth.get());
        assertEquals(CacheMatchSource.LOCAL_STANDBY, manager.activeSource());

        manager.recoverPrimaryManually();
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
        KvcmConfig kvcmTopology = new KvcmConfig();

        ServiceRoute route = new ServiceRoute();
        route.setServiceId("test-service");
        route.setKvcm(kvcmTopology);

        ModelMetaConfig config = new ModelMetaConfig();
        config.putServiceRoute(route.getServiceId(), route);
        return kvcm(config,
                runtime -> runtime.getLocalStandby().setAutoSwitch(autoSwitch));
    }
}
