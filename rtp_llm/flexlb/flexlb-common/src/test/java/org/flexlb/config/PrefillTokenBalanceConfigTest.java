package org.flexlb.config;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

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
    }

    @Test void windowIsOptInAndEngineQueueAdmissionDefaultsToTen() {
        var defaults = parse("{}").getRouter().getRoles().getPrefill().getTokenBalance();
        assertFalse(defaults.isEnableWindow());
        assertEquals(10_000, defaults.getWindowMs());
        assertEquals(10L, defaults.getEngineQueueAdmissionThreshold());
        assertEquals(5_000L, defaults.getEngineQueueOverloadDurationMs());
        var configured = parse("{\"enableWindow\":true,\"windowMs\":5000,\"engineQueueAdmissionThreshold\":30,"
                + "\"engineQueueOverloadDurationMs\":2000}")
                .getRouter().getRoles().getPrefill().getTokenBalance();
        assertTrue(configured.isEnableWindow());
        assertEquals(5000, configured.getWindowMs());
        assertEquals(30L, configured.getEngineQueueAdmissionThreshold());
        assertEquals(2000L, configured.getEngineQueueOverloadDurationMs());
        assertThrows(RuntimeException.class, () -> parse("{\"engineQueueAdmissionThreshold\":-1}"));
        assertThrows(RuntimeException.class, () -> parse("{\"engineQueueOverloadDurationMs\":-1}"));
    }

    @Test void hotspotRescueIsOptInWithBoundedValidation() {
        assertNull(parse("{}").getRouter().getRoles().getPrefill().getTokenBalance().getHotspotRescue());
        var defaults = parse("{\"hotspotRescue\":{}}").getRouter().getRoles().getPrefill().getTokenBalance().getHotspotRescue();
        assertEquals(5_000L, defaults.getMinWaitMs());
        assertEquals(30L, defaults.getEngineQueueMigrationThreshold());
        assertEquals(32_768L, defaults.getMaxInputTokens());
        assertEquals(0L, defaults.getMaxIdlePendingTokens());
        assertEquals(1_000L, defaults.getMigrationIntervalMs());
        assertEquals(1, defaults.getMaxMigrationsPerRound());
        var configured = parse("""
                {"hotspotRescue":{"engineQueueMigrationThreshold":60,"maxIdlePendingTokens":32000,"migrationIntervalMs":2000,"maxMigrationsPerRound":2}}
                """).getRouter().getRoles().getPrefill().getTokenBalance().getHotspotRescue();
        assertEquals(60L, configured.getEngineQueueMigrationThreshold());
        assertEquals(32_000L, configured.getMaxIdlePendingTokens());
        assertEquals(2_000L, configured.getMigrationIntervalMs());
        assertEquals(2, configured.getMaxMigrationsPerRound());
        assertEquals(0L, parse("{\"hotspotRescue\":{\"engineQueueMigrationThreshold\":0}}")
                .getRouter().getRoles().getPrefill().getTokenBalance().getHotspotRescue().getEngineQueueMigrationThreshold());
        for (String value : new String[]{"{\"engineQueueMigrationThreshold\":-1}", "{\"minWaitMs\":0}", "{\"maxInputTokens\":0}",
                "{\"migrationIntervalMs\":0}", "{\"migrationIntervalMs\":-1}",
                "{\"migrationIntervalMs\":9223372036854775807}",
                "{\"maxMigrationsPerRound\":0}", "{\"maxMigrationsPerRound\":-1}",
                "{\"maxIdlePendingTokens\":-1}", "{\"maxIdlePendingTokens\":2000000}"}) {
            assertThrows(RuntimeException.class, () -> parse("{\"hotspotRescue\":" + value + "}"));
        }
    }
}
