package org.flexlb.config;

import org.flexlb.dao.route.KvcmConfig;
import org.flexlb.dao.route.ServiceRoute;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import java.util.List;
import java.util.function.Consumer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class CacheMatchConfigurationTest {

    @Test
    void defaultsToKvcmAndRequiresTopology() {
        FlexlbConfig config = ConfigTestFixtures.parse("{}");
        CacheMatchConfiguration configuration = new CacheMatchConfiguration(new ModelMetaConfig(route(true)), config);
        assertEquals(CacheMatchMode.KVCM, configuration.getConfiguredMode());
        assertTrue(configuration.isKvcmEnabled());
        assertTrue(configuration.isLocalStandbyEnabled());
        assertFalse(configuration.isLocalSyncEnabled());
        assertThrows(IllegalStateException.class,
                () -> new CacheMatchConfiguration(new ModelMetaConfig(route(false)), config));
    }

    @Test
    void usesExplicitLocalSync() {
        FlexlbConfig config = new FlexlbConfig();
        config.setCacheMatching(new LocalSyncCacheMatchingConfig());
        CacheMatchConfiguration configuration = new CacheMatchConfiguration(new ModelMetaConfig(route(false)), config);

        assertFalse(configuration.isKvcmEnabled());
        assertTrue(configuration.isLocalSyncEnabled());
        assertFalse(configuration.isLocalStandbyEnabled());
        assertFalse(configuration.isAutoSwitchEnabled());
        assertNull(configuration.getKvcmConfig());
        assertNull(configuration.getKvcmRuntimeConfig());
        assertNull(configuration.getLocalStandbyConfig());
        assertEquals(CacheMatchMode.LOCAL_SYNC, configuration.getConfiguredMode());
    }

    @Test
    void usesTopologyFromModelConfigAndPolicyFromFlexlbConfig() {
        ServiceRoute route = route(true);
        FlexlbConfig flexlbConfig = kvcmConfig(false);

        CacheMatchConfiguration configuration = new CacheMatchConfiguration(new ModelMetaConfig(route), flexlbConfig);

        assertTrue(configuration.isKvcmEnabled());
        assertFalse(configuration.isLocalSyncEnabled());
        assertTrue(configuration.isLocalStandbyEnabled());
        assertSame(route, configuration.getServiceRoute());
        assertSame(route.getKvcm(), configuration.getKvcmConfig());
        assertSame(flexlbConfig.getKvcmAndStandbyConfig(), configuration.getKvcmRuntimeConfig());
        assertSame(flexlbConfig.getKvcmAndStandbyConfig().getLocalStandby(), configuration.getLocalStandbyConfig());
        assertFalse(configuration.isAutoSwitchEnabled());
        assertEquals(CacheMatchMode.KVCM, configuration.getConfiguredMode());
    }

    @Test
    void readsAutoSwitchFromFlexlbConfig() {
        CacheMatchConfiguration configuration =
                new CacheMatchConfiguration(new ModelMetaConfig(route(true)), kvcmConfig(true));

        assertTrue(configuration.isLocalStandbyEnabled());
        assertTrue(configuration.isAutoSwitchEnabled());
    }

    @Test
    void rejectsKvcmModeWithoutKvcmTopology() {
        IllegalStateException error = assertThrows(IllegalStateException.class,
                () -> new CacheMatchConfiguration(new ModelMetaConfig(route(false)), kvcmConfig(true)));

        assertTrue(error.getMessage().contains(
                "cacheMatching.type=KVCM requires MODEL_SERVICE_CONFIG kvcm topology"));
    }

    @Test
    void standbyRequiresKvcmTopologyForSwitchingBack() {
        FlexlbConfig config = new FlexlbConfig();
        config.setCacheMatching(new LocalStandbyCacheMatchingConfig());
        IllegalStateException error = assertThrows(IllegalStateException.class,
                () -> new CacheMatchConfiguration(new ModelMetaConfig(route(false)), config));
        assertTrue(error.getMessage().contains("cacheMatching.type=LOCAL_STANDBY"));
    }

    @Test
    void publishesKvcmRuntimeConfigUpdatesToReaders() {
        ConfigService configService = mock(ConfigService.class);
        FlexlbConfig initial = kvcmConfig(false);
        when(configService.loadBalanceConfig()).thenReturn(initial);
        @SuppressWarnings("unchecked")
        ArgumentCaptor<Consumer<FlexlbConfig>> listener =
                ArgumentCaptor.forClass(Consumer.class);

        CacheMatchConfiguration configuration =
                new CacheMatchConfiguration(new ModelMetaConfig(route(true)), configService);
        assertSame(initial.getKvcmAndStandbyConfig(), configuration.getKvcmRuntimeConfig());
        verify(configService).addUpdateListener(listener.capture());

        FlexlbConfig updated = kvcmConfig(false);
        updated.getKvcmAndStandbyConfig().setTopKHostCount(7);
        updated.getKvcmAndStandbyConfig()
                .setBackendTypes(List.of(KvcmCacheMatchingConfig.BackendType.ST_EVENT_REPORT_L2));
        updated.getKvcmAndStandbyConfig().setMedium(List.of("kvs"));
        listener.getValue().accept(updated);

        assertNotSame(initial.getKvcmAndStandbyConfig(), configuration.getKvcmRuntimeConfig());
        assertSame(updated.getKvcmAndStandbyConfig(), configuration.getKvcmRuntimeConfig());
        assertEquals(7, configuration.getKvcmRuntimeConfig().getTopKHostCount());
        assertEquals(List.of(KvcmCacheMatchingConfig.BackendType.ST_EVENT_REPORT_L2),
                configuration.getKvcmRuntimeConfig().getBackendTypes());
        assertEquals(List.of("kvs"), configuration.getKvcmRuntimeConfig().getMedium());
        assertEquals(initial.getKvcmAndStandbyConfig().getLocalStandby(), configuration.getLocalStandbyConfig(),
                "local standby stays fixed for the process lifetime");
    }

    @Test
    void keepsKvcmRuntimeConfigWhenAnUpdateActivatesAnotherMode() {
        ConfigService configService = mock(ConfigService.class);
        FlexlbConfig initial = kvcmConfig(false);
        when(configService.loadBalanceConfig()).thenReturn(initial);
        @SuppressWarnings("unchecked")
        ArgumentCaptor<Consumer<FlexlbConfig>> listener =
                ArgumentCaptor.forClass(Consumer.class);

        CacheMatchConfiguration configuration =
                new CacheMatchConfiguration(new ModelMetaConfig(route(true)), configService);
        verify(configService).addUpdateListener(listener.capture());

        FlexlbConfig localSync = new FlexlbConfig();
        localSync.setCacheMatching(new LocalSyncCacheMatchingConfig());
        listener.getValue().accept(localSync);

        assertSame(initial.getKvcmAndStandbyConfig(), configuration.getKvcmRuntimeConfig());
        assertEquals(CacheMatchMode.KVCM, configuration.getConfiguredMode());
    }

    private FlexlbConfig kvcmConfig(boolean autoSwitch) {
        KvcmCacheMatchingConfig kvcm = new KvcmCacheMatchingConfig();
        kvcm.getLocalStandby().setAutoSwitch(autoSwitch);
        FlexlbConfig config = new FlexlbConfig();
        config.setCacheMatching(kvcm);
        return config;
    }

    private ServiceRoute route(boolean hasKvcmTopology) {
        ServiceRoute route = new ServiceRoute();
        if (hasKvcmTopology) {
            route.setKvcm(new KvcmConfig());
        }
        return route;
    }
}
