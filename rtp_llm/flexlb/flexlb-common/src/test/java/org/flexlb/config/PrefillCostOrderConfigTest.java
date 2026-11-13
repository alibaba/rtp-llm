package org.flexlb.config;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PrefillCostOrderConfigTest {
    @Test void disabledByDefaultAndRequiresSupportedAdmissionPath() {
        assertFalse(ConfigTestFixtures.parse("{}").getScheduler().getOrdering().isPrefillCostOrder());
        var config = ConfigTestFixtures.parse("""
                {"scheduler":{"decision":{"type":"SINGLE"},"ordering":{"prefillCostOrder":true}},
                 "dispatcher":{"type":"NON_BATCH"},"router":{"roles":{"prefill":{"tokenBalance":{}}}}}
                """);
        assertTrue(config.getScheduler().getOrdering().isPrefillCostOrder());
        for (String body : new String[]{
                "{\"scheduler\":{\"ordering\":{\"prefillCostOrder\":true}}}",
                "{\"scheduler\":{\"type\":\"DIRECT\",\"ordering\":{\"prefillCostOrder\":true}}}"}) {
            assertThrows(RuntimeException.class, () -> ConfigTestFixtures.parse(body));
        }
    }
}
