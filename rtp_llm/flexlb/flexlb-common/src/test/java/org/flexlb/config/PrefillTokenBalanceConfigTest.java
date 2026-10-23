package org.flexlb.config;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class PrefillTokenBalanceConfigTest {
    private FlexlbConfig parse(String balance) {
        return ConfigTestFixtures.parse("{\"router\":{\"roles\":{\"prefill\":{\"tokenBalance\":" + balance + "}}}}");
    }
    @Test void optInDefaultsAndValidation() {
        assertNull(ConfigTestFixtures.parse("{}").getRouter().getRoles().getPrefill().getTokenBalance());
        assertEquals(0, parse("{}").getRouter().getRoles().getPrefill().getTokenBalance().getWindowMs());
        assertEquals(RoutingConfig.EstimatorType.LEARNING,
                parse("{}").getRouter().getRoles().getPrefill().getExecutionTimeEstimator().getType());
        for (String value : new String[]{"{\"windowMs\":-1}", "{\"maxPendingComputeTokens\":-1}",
                "{\"minPrefixHitPercent\":101}", "{\"minPrefixHitPercent\":-1}"}) {
            assertThrows(RuntimeException.class, () -> parse(value));
        }
        assertThrows(RuntimeException.class, () -> ConfigTestFixtures.parse("""
                {"router":{"roles":{"prefill":{"tokenBalance":{},"cacheAffinity":{}}}}}
                """));
    }
    @Test void validatesBoundedHotspotRescue() {
        var rescue = parse("{\"hotspotRescue\":{}}").getRouter().getRoles().getPrefill().getTokenBalance().getHotspotRescue();
        assertEquals(0, rescue.getMinWaitMs());
        for (String value : new String[]{"{\"minWaitMs\":-1}", "{\"maxInputTokens\":-1}",
                "{\"maxIdlePendingTokens\":-1}", "{\"maxIdlePendingTokens\":140000}"}) {
            assertThrows(RuntimeException.class, () -> parse("{\"maxPendingComputeTokens\":140000,\"hotspotRescue\":" + value + "}"));
        }
    }

    @Test void batchSheddingRequiresTokenBackpressure() {
        assertNotNull(ConfigTestFixtures.parse("""
                {"scheduler":{"type":"QUEUE","decision":{"type":"FIXED_WINDOW"},"prefillQueueShedding":{}},
                 "dispatcher":{"type":"BATCH"},"router":{"roles":{"prefill":{"tokenBalance":{}}}}}
                """));
    }
}
