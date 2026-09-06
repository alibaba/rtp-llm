package org.flexlb.config;

import org.flexlb.dao.route.KvcmConfig;
import org.flexlb.dao.route.LocalStandbyConfig;
import org.flexlb.dao.route.ServiceRoute;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

class CacheMatchConfigurationTest {

    @Test
    void usesLocalSyncWhenKvcmIsDisabled() {
        CacheMatchConfiguration configuration =
                new CacheMatchConfiguration(
                        modelMetaConfig(route("service-1", false, false)));

        assertFalse(configuration.isKvcmEnabled());
        assertTrue(configuration.isLocalSyncEnabled());
        assertFalse(configuration.isLocalStandbyEnabled());
        assertFalse(configuration.isAutoSwitchEnabled());
        assertNull(configuration.getKvcmServiceRoute());
        assertNull(configuration.getKvcmConfig());
        assertNull(configuration.getLocalStandbyConfig());
        assertEquals(CacheMatchMode.LOCAL_SYNC, configuration.getConfiguredMode());
    }

    @Test
    void alwaysEnablesLocalStandbyInKvcmMode() {
        ServiceRoute route = route("service-1", true, false);
        CacheMatchConfiguration configuration =
                new CacheMatchConfiguration(modelMetaConfig(route));

        assertTrue(configuration.isKvcmEnabled());
        assertFalse(configuration.isLocalSyncEnabled());
        assertTrue(configuration.isLocalStandbyEnabled());
        assertSame(route, configuration.getKvcmServiceRoute());
        assertSame(route.getKvcm(), configuration.getKvcmConfig());
        assertSame(route.getKvcm().getLocalStandby(), configuration.getLocalStandbyConfig());
        assertFalse(configuration.isAutoSwitchEnabled());
        assertEquals(CacheMatchMode.KVCM, configuration.getConfiguredMode());
    }

    @Test
    void readsAutoSwitchPolicyIndependentlyFromStandbyAvailability() {
        CacheMatchConfiguration configuration =
                new CacheMatchConfiguration(
                        modelMetaConfig(route("service-1", true, true)));

        assertTrue(configuration.isLocalStandbyEnabled());
        assertTrue(configuration.isAutoSwitchEnabled());
    }

    @Test
    void suppliesDefaultLocalStandbyConfigurationWhenOmitted() {
        ServiceRoute route = route("service-1", true, true);
        route.getKvcm().setLocalStandby(null);

        CacheMatchConfiguration configuration =
                new CacheMatchConfiguration(modelMetaConfig(route));

        assertTrue(configuration.isLocalStandbyEnabled());
        assertTrue(configuration.isAutoSwitchEnabled());
        assertEquals(
                LocalStandbyConfig.DEFAULT_TTL_MS,
                configuration.getLocalStandbyConfig().getTtlMs());
    }

    @Test
    void selectsOneOfMultipleKvcmEnabledRoutes() {
        ModelMetaConfig modelMetaConfig = modelMetaConfig(
                route("service-1", true, true),
                route("service-2", true, true));

        CacheMatchConfiguration configuration = new CacheMatchConfiguration(modelMetaConfig);
        String selectedServiceId = configuration.getKvcmServiceRoute().getServiceId();

        assertTrue("service-1".equals(selectedServiceId) || "service-2".equals(selectedServiceId));
    }

    private ModelMetaConfig modelMetaConfig(ServiceRoute... routes) {
        ModelMetaConfig modelMetaConfig = new ModelMetaConfig();
        for (ServiceRoute route : routes) {
            modelMetaConfig.putServiceRoute(route.getServiceId(), route);
        }
        return modelMetaConfig;
    }

    private ServiceRoute route(
            String serviceId,
            boolean kvcmEnabled,
            boolean autoSwitch) {
        LocalStandbyConfig localStandby = new LocalStandbyConfig();
        localStandby.setAutoSwitch(autoSwitch);

        KvcmConfig kvcm = new KvcmConfig();
        kvcm.setEnabled(kvcmEnabled);
        kvcm.setLocalStandby(localStandby);

        ServiceRoute route = new ServiceRoute();
        route.setServiceId(serviceId);
        route.setKvcm(kvcm);
        return route;
    }
}
