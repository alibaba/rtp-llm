package org.flexlb.config;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertThrows;

class FlexlbConfigRoleTest {
    @Test
    void vitRoutingLoadsThroughTheRealEnvironmentConfigPath() {
        var defaults = new ConfigService(java.util.Map.of(ConfigService.FLEXLB_CONFIG_ENV,
                "{\"schemaVersion\":3,\"requestLifecycle\":{\"request\":{\"timeoutMs\":60000}}}"))
                .loadBalanceConfig().getRouter().getRoles().getVit();
        assertFalse(defaults.isCacheAffinityEnabled());
        assertEquals(5000L, defaults.getLoadMaxAgeMs());

        var configured = new ConfigService(java.util.Map.of(ConfigService.FLEXLB_CONFIG_ENV, """
                {"schemaVersion":3,"requestLifecycle":{"request":{"timeoutMs":60000}},
                 "router":{"roles":{"vit":{"cacheAffinityEnabled":true,"loadMaxAgeMs":1200}}}}
                """)).loadBalanceConfig().getRouter().getRoles().getVit();
        assertTrue(configured.isCacheAffinityEnabled());
        assertEquals(1200L, configured.getLoadMaxAgeMs());
        assertFalse(ConfigTestFixtures.parse("""
                {"router":{"roles":{"vit":{"cacheAffinityEnabled":false}}}}
                """).getRouter().getRoles().getVit().isCacheAffinityEnabled());
    }

    @Test
    void vitRoutingRejectsInvalidTypesValuesAndWrongNesting() {
        for (String fields : new String[]{
                "\"cacheAffinityEnabled\":\"true\"", "\"cacheAffinityEnabled\":1",
                "\"cacheAffinityEnabled\":null", "\"loadMaxAgeMs\":0",
                "\"loadMaxAgeMs\":-1", "\"loadMaxAgeMs\":1.5",
                "\"loadMaxAgeMs\":\"5000\"", "\"loadMaxAgeMs\":null",
                "\"cacheAffinityEnable\":true"}) {
            String patch = "{\"router\":{\"roles\":{\"vit\":{" + fields + "}}}}";
            assertThrows(ConfigValidationException.class, () -> ConfigTestFixtures.parse(patch), patch);
        }
        assertThrows(ConfigValidationException.class, () -> ConfigTestFixtures.parse("""
                {"router":{"vit":{"cacheAffinityEnabled":true}}}
                """));
    }

    @Test
    void vitDefaultsToLeastLoadWithoutCacheAffinity() {
        var vit = new FlexlbConfig().getRouter().getRoles().getVit();
        assertFalse(vit.isCacheAffinityEnabled());
        assertEquals(5000, vit.getLoadMaxAgeMs());
    }
    @Test
    void request_timeout_is_explicit_and_dispatcher_defaults_are_shared() {
        FlexlbConfig config = new FlexlbConfig();
        assertNull(config.getRequestLifecycle().getRequest().getTimeoutMs());
        assertEquals(2.0, config.getRequestLifecycle().getDecision().getLifetime());
        assertEquals(2, config.getDispatcher().getMaxInflightPerPrefillWorker());
        assertEquals(2, DispatcherConfig.nonBatch().getMaxInflightPerPrefillWorker());
    }
}
