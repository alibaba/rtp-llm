package org.flexlb.cache.match;

import org.flexlb.cache.domain.CacheMatchSource;
import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.config.CacheMatchConfiguration;
import org.flexlb.dao.kvcm.KvcmHealthSnapshot;
import org.flexlb.dao.kvcm.KvcmHealthState;
import org.flexlb.engine.grpc.client.KvcmGrpcClient;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

class CacheMatchFailoverManagerTest {
    @Test void healthRecoveryAndManualOverrideHaveDistinctOwners() {
        var config = mock(CacheMatchConfiguration.class);
        when(config.isKvcmEnabled()).thenReturn(true); when(config.isAutoSwitchEnabled()).thenReturn(true);
        var client = mock(KvcmGrpcClient.class);
        when(client.healthSnapshot()).thenReturn(health(true));
        var manager = new CacheMatchFailoverManager(config, client, mock(CacheMetricsReporter.class));
        assertEquals(CacheMatchSource.KVCM, manager.activeSource());
        when(client.healthSnapshot()).thenReturn(health(false)); manager.updateFromKvcmHealth();
        assertEquals(CacheMatchSource.LOCAL_STANDBY, manager.activeSource());
        assertThrows(IllegalStateException.class, manager::recoverPrimaryManually);
        manager.activateFallbackManually();
        when(client.healthSnapshot()).thenReturn(health(true)); manager.updateFromKvcmHealth();
        assertEquals(CacheMatchSource.LOCAL_STANDBY, manager.activeSource());
        manager.recoverPrimaryManually();
        assertEquals(CacheMatchSource.KVCM, manager.activeSource());
    }

    @Test void offDoesNotSubscribeToKvcmHealth() {
        var client = mock(KvcmGrpcClient.class);
        var manager = new CacheMatchFailoverManager(mock(CacheMatchConfiguration.class), client, mock(CacheMetricsReporter.class));
        assertEquals(CacheMatchSource.LOCAL_SYNC, manager.activeSource());
        verifyNoInteractions(client);
    }

    private static KvcmHealthSnapshot health(boolean healthy) {
        return new KvcmHealthSnapshot(healthy ? KvcmHealthState.HEALTHY : KvcmHealthState.UNHEALTHY,
                healthy ? 0 : 3, healthy ? 3 : 0, 0, 0, 0, "test");
    }
}
