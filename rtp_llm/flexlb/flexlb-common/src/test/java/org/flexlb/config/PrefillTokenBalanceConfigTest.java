package org.flexlb.config;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class PrefillTokenBalanceConfigTest {
    private FlexlbConfig parse(String balance) {
        return ConfigTestFixtures.parse("{\"router\":{\"roles\":{\"prefill\":{\"tokenBalance\":" + balance + "}}}}");
    }
    @Test void optInDefaultsAndValidation() {
        assertNull(ConfigTestFixtures.parse("{}").getRouter().getRoles().getPrefill().getTokenBalance());
        assertEquals(10000, parse("{}").getRouter().getRoles().getPrefill().getTokenBalance().getWindowMs());
        for (String value : new String[]{"{\"windowMs\":0}", "{\"maxPendingComputeTokens\":0}",
                "{\"minPrefixHitPercent\":101}", "{\"minPrefixHitPercent\":-1}"}) {
            assertThrows(RuntimeException.class, () -> parse(value));
        }
        assertThrows(RuntimeException.class, () -> ConfigTestFixtures.parse("""
                {"router":{"roles":{"prefill":{"tokenBalance":{},"cacheAffinity":{}}}}}
                """));
    }
    @Test void validatesBoundedHotspotRescue() {
        var rescue = parse("{\"hotspotRescue\":{}}").getRouter().getRoles().getPrefill().getTokenBalance().getHotspotRescue();
        assertEquals(5000, rescue.getMinWaitMs());
        for (String value : new String[]{"{\"minWaitMs\":0}", "{\"maxInputTokens\":0}",
                "{\"maxIdlePendingTokens\":-1}", "{\"maxIdlePendingTokens\":140000}"}) {
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
