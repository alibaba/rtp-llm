package org.flexlb.config;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class PrefillTokenBalanceConfigTest {
    private FlexlbConfig parse(String balance) {
        return ConfigTestFixtures.parse("{\"router\":{\"roles\":{\"prefill\":{\"tokenBalance\":" + balance + "}}}}");
    }
    @Test void optInDefaultsAndValidation() {
        assertNull(ConfigTestFixtures.parse("{}").getRouter().getRoles().getPrefill().getTokenBalance());
        assertEquals(2_000_000, parse("{}").getRouter().getRoles().getPrefill().getTokenBalance().getMaxPendingComputeTokens());
        for (String value : new String[]{"{\"windowMs\":0}", "{\"maxPendingComputeTokens\":0}",
                "{\"minPrefixHitPercent\":101}", "{\"minPrefixHitPercent\":-1}"}) {
            assertThrows(RuntimeException.class, () -> parse(value));
        }
        assertThrows(RuntimeException.class, () -> ConfigTestFixtures.parse("""
                {"router":{"roles":{"prefill":{"tokenBalance":{},"cacheAffinity":{}}}}}
                """));
    }
    @Test void windowAndEngineQueueAdmissionAreExplicitOptIns() {
        var defaults = parse("{}").getRouter().getRoles().getPrefill().getTokenBalance();
        assertFalse(defaults.isEnableWindow());
        assertEquals(10_000, defaults.getWindowMs());
        assertNull(defaults.getEngineQueueAdmissionThreshold());
        var enabled = parse("{\"enableWindow\":true,\"windowMs\":5000,\"engineQueueAdmissionThreshold\":0}")
                .getRouter().getRoles().getPrefill().getTokenBalance();
        assertTrue(enabled.isEnableWindow());
        assertEquals(5000, enabled.getWindowMs());
        assertEquals(0L, enabled.getEngineQueueAdmissionThreshold());
        assertThrows(RuntimeException.class, () -> parse("{\"engineQueueAdmissionThreshold\":-1}"));
    }

    @Test void validatesBoundedHotspotRescue() {
        var rescue = parse("{\"hotspotRescue\":{}}").getRouter().getRoles().getPrefill().getTokenBalance().getHotspotRescue();
        assertEquals(5000, rescue.getMinWaitMs());
        assertEquals(0, rescue.getMaxIdlePendingTokens());
        assertEquals(1000, rescue.getMigrationIntervalMs());
        assertEquals(1, rescue.getMaxMigrationsPerRound());
        var configured = parse("""
                {"hotspotRescue":{"maxIdlePendingTokens":32000,"migrationIntervalMs":2000,"maxMigrationsPerRound":2}}
                """).getRouter().getRoles().getPrefill().getTokenBalance().getHotspotRescue();
        assertEquals(32000, configured.getMaxIdlePendingTokens());
        assertEquals(2000, configured.getMigrationIntervalMs());
        assertEquals(2, configured.getMaxMigrationsPerRound());
        for (String value : new String[]{"{\"minWaitMs\":0}", "{\"maxInputTokens\":0}",
                "{\"migrationIntervalMs\":0}", "{\"migrationIntervalMs\":-1}",
                "{\"migrationIntervalMs\":9223372036854775807}",
                "{\"maxMigrationsPerRound\":0}", "{\"maxMigrationsPerRound\":-1}",
                "{\"maxIdlePendingTokens\":-1}", "{\"maxIdlePendingTokens\":2000000}"}) {
            assertThrows(RuntimeException.class, () -> parse("{\"hotspotRescue\":" + value + "}"));
        }
    }

    @Test void batchSheddingRequiresTokenBackpressure() {
        assertNotNull(ConfigTestFixtures.parse("""
                {"scheduler":{"type":"QUEUE","decision":{"type":"FIXED_WINDOW"},"prefillQueueShedding":{}},
                 "dispatcher":{"type":"BATCH"},"router":{"roles":{"prefill":{"tokenBalance":{}}}}}
                """));
    }
}
