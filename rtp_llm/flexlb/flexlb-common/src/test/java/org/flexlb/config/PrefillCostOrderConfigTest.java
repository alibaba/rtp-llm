package org.flexlb.config;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PrefillCostOrderConfigTest {
    @Test void disabledByDefaultAndRequiresSupportedAdmissionPath() {
        assertFalse(ConfigTestFixtures.parse("{}").getScheduler().getOrdering().isPrefillCostOrder());
        for (String body : new String[]{
                // Single decision with token-balance, standalone, or cache-affinity selection.
                """
                {"scheduler":{"decision":{"type":"SINGLE"},"ordering":{"prefillCostOrder":true}},
                 "dispatcher":{"type":"NON_BATCH"},"router":{"roles":{"prefill":{"tokenBalance":{}}}}}
                """,
                """
                {"scheduler":{"decision":{"type":"SINGLE"},"ordering":{"prefillCostOrder":true}},
                 "dispatcher":{"type":"NON_BATCH"}}
                """,
                """
                {"scheduler":{"decision":{"type":"SINGLE"},"ordering":{"prefillCostOrder":true}},
                 "dispatcher":{"type":"NON_BATCH"},
                 "router":{"roles":{"prefill":{"cacheAffinity":{"maxExtraTtftMs":1000}}}}}
                """}) {
            assertTrue(ConfigTestFixtures.parse(body)
                    .getScheduler().getOrdering().isPrefillCostOrder());
        }
        for (String body : new String[]{
                // Default FIXED_WINDOW decisions and the default BATCH dispatcher are rejected.
                "{\"scheduler\":{\"ordering\":{\"prefillCostOrder\":true}}}",
                "{\"scheduler\":{\"type\":\"DIRECT\",\"ordering\":{\"prefillCostOrder\":true}}}"}) {
            assertThrows(RuntimeException.class, () -> ConfigTestFixtures.parse(body));
        }
    }
}
