package org.flexlb.config;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class PrefillTokenBalanceConfigTest {
    private FlexlbConfig parse(String balance) {
        return ConfigTestFixtures.parse("{\"router\":{\"roles\":{\"prefill\":{\"tokenBalance\":" + balance + "}}}}");
    }
    @Test void optInDefaultsAndValidation() {
        assertNull(ConfigTestFixtures.parse("{}").getRouter().getRoles().getPrefill().getTokenBalance());
        assertEquals(5.0, parse("{}").getRouter().getRoles().getPrefill().getTokenBalance().getMinPrefixHitPercent());
        assertEquals(50.0, parse("{\"minPrefixHitPercent\":50}")
                .getRouter().getRoles().getPrefill().getTokenBalance().getMinPrefixHitPercent());
        for (String value : new String[]{"{\"minPrefixHitPercent\":101}", "{\"minPrefixHitPercent\":-1}"}) {
            assertThrows(RuntimeException.class, () -> parse(value));
        }
        assertThrows(RuntimeException.class, () -> ConfigTestFixtures.parse("""
                {"router":{"roles":{"prefill":{"tokenBalance":{},"cacheAffinity":{}}}}}
                """));
    }
    @Test void engineQueueAdmissionDefaultsToTen() {
        var defaults = parse("{}").getRouter().getRoles().getPrefill().getTokenBalance();
        assertEquals(10L, defaults.getEngineQueueAdmissionThreshold());
        assertEquals(5000L, defaults.getEngineQueueOverloadDurationMs());
        var enabled = parse("{\"engineQueueAdmissionThreshold\":0}")
                .getRouter().getRoles().getPrefill().getTokenBalance();
        assertEquals(0L, enabled.getEngineQueueAdmissionThreshold());
        assertThrows(RuntimeException.class, () -> parse("{\"engineQueueAdmissionThreshold\":-1}"));
        assertThrows(RuntimeException.class, () -> parse("{\"engineQueueOverloadDurationMs\":-1}"));
    }

    @Test void hotspotRescueRateIsANonNegativeInteger() {
        var defaults = parse("{}").getRouter().getRoles().getPrefill().getTokenBalance();
        assertEquals(0, defaults.getHotspotRescue());
        assertEquals(0L, defaults.hotspotRescueIntervalNanos());
        var balance = parse("""
                {"hotspotRescue":3}
                """).getRouter().getRoles().getPrefill().getTokenBalance();
        assertEquals(3, balance.getHotspotRescue());
        assertEquals(333_333_334L, balance.hotspotRescueIntervalNanos());
        for (String value : new String[]{"-1", "1.5", "true", "{\"enabled\":true}"}) {
            assertThrows(ConfigValidationException.class, () -> parse("{\"hotspotRescue\":" + value + "}"));
        }
    }

    @Test void removedRescueAndHistoryOptionsAreRejected() {
        for (String value : new String[]{"{\"hotspotRescue\":{\"minWaitMs\":5000}}", "{\"enableWindow\":false}",
                "{\"enableWindow\":true}", "{\"windowMs\":10000}"}) {
            assertThrows(ConfigValidationException.class, () -> parse(value));
        }
    }
}
