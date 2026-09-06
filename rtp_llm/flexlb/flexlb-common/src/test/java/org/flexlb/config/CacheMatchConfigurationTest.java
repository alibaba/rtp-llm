package org.flexlb.config;

import org.flexlb.dao.route.KvcmConfig;
import org.flexlb.dao.route.ServiceRoute;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class CacheMatchConfigurationTest {

    @Test
    void usesLocalSyncByDefault() {
        CacheMatchConfiguration configuration = new CacheMatchConfiguration(
                modelMetaConfig(route("service-1", false)),
                new FlexlbConfig());

        assertFalse(configuration.isKvcmEnabled());
        assertTrue(configuration.isLocalSyncEnabled());
        assertFalse(configuration.isLocalStandbyEnabled());
        assertFalse(configuration.isAutoSwitchEnabled());
        assertNull(configuration.getKvcmServiceRoute());
        assertNull(configuration.getKvcmConfig());
        assertNull(configuration.getKvcmRuntimeConfig());
        assertNull(configuration.getLocalStandbyConfig());
        assertEquals(CacheMatchMode.LOCAL_SYNC, configuration.getConfiguredMode());
    }

    @Test
    void usesTopologyFromModelConfigAndPolicyFromFlexlbConfig() {
        ServiceRoute route = route("service-1", true);
        FlexlbConfig flexlbConfig = kvcmConfig(false);

        CacheMatchConfiguration configuration = new CacheMatchConfiguration(
                modelMetaConfig(route), flexlbConfig);

        assertTrue(configuration.isKvcmEnabled());
        assertFalse(configuration.isLocalSyncEnabled());
        assertTrue(configuration.isLocalStandbyEnabled());
        assertSame(route, configuration.getKvcmServiceRoute());
        assertSame(route.getKvcm(), configuration.getKvcmConfig());
        assertSame(flexlbConfig.kvcmCacheMatching(),
                configuration.getKvcmRuntimeConfig());
        assertSame(flexlbConfig.kvcmCacheMatching().getLocalStandby(),
                configuration.getLocalStandbyConfig());
        assertFalse(configuration.isAutoSwitchEnabled());
        assertEquals(CacheMatchMode.KVCM, configuration.getConfiguredMode());
    }

    @Test
    void readsAutoSwitchFromFlexlbConfig() {
        CacheMatchConfiguration configuration = new CacheMatchConfiguration(
                modelMetaConfig(route("service-1", true)),
                kvcmConfig(true));

        assertTrue(configuration.isLocalStandbyEnabled());
        assertTrue(configuration.isAutoSwitchEnabled());
    }

    @Test
    void rejectsKvcmModeWithoutKvcmTopology() {
        IllegalStateException error = assertThrows(IllegalStateException.class,
                () -> new CacheMatchConfiguration(
                        modelMetaConfig(route("service-1", false)),
                        kvcmConfig(true)));

        assertTrue(error.getMessage().contains(
                "cacheMatching.type=KVCM requires MODEL_SERVICE_CONFIG kvcm topology"));
    }

    @Test
    void selectsOneOfMultipleKvcmTopologies() {
        ModelMetaConfig modelMetaConfig = modelMetaConfig(
                route("service-1", true),
                route("service-2", true));

        CacheMatchConfiguration configuration = new CacheMatchConfiguration(
                modelMetaConfig, kvcmConfig(true));
        String selectedServiceId = configuration.getKvcmServiceRoute().getServiceId();

        assertTrue("service-1".equals(selectedServiceId)
                || "service-2".equals(selectedServiceId));
    }

    private FlexlbConfig kvcmConfig(boolean autoSwitch) {
        KvcmCacheMatchingConfig kvcm = new KvcmCacheMatchingConfig();
        kvcm.getLocalStandby().setAutoSwitch(autoSwitch);
        FlexlbConfig config = new FlexlbConfig();
        config.setCacheMatching(kvcm);
        return config;
    }

    private ModelMetaConfig modelMetaConfig(ServiceRoute... routes) {
        ModelMetaConfig modelMetaConfig = new ModelMetaConfig();
        for (ServiceRoute route : routes) {
            modelMetaConfig.putServiceRoute(route.getServiceId(), route);
        }
        return modelMetaConfig;
    }

    private ServiceRoute route(String serviceId, boolean hasKvcmTopology) {
        ServiceRoute route = new ServiceRoute();
        route.setServiceId(serviceId);
        if (hasKvcmTopology) {
            route.setKvcm(new KvcmConfig());
        }
        return route;
    }
}
