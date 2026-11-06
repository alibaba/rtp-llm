package org.flexlb.config;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class PrefillQueueSheddingConfigTest {
    private FlexlbConfig parse(String body) {
        return ConfigTestFixtures.parse("{\"scheduler\":{\"type\":\"QUEUE\","
                + "\"decision\":{\"type\":\"SINGLE\"},\"prefillQueueShedding\":" + body
                + "},\"dispatcher\":{\"type\":\"NON_BATCH\"}}");
    }

    @Test void sustainedProtectionDefaultsOnAndCanBeDisabled() {
        assertTrue(parse("{}").queueScheduler().getPrefillQueueShedding().isSustainedOverloadProtection());
        assertFalse(parse("{\"sustainedOverloadProtection\":false}").queueScheduler()
                .getPrefillQueueShedding().isSustainedOverloadProtection());
    }

    @Test void acceptingWorkerPercentDefaultsToTenAndAcceptsOverrides() {
        assertEquals(10.0, parse("{}").queueScheduler().getPrefillQueueShedding().getMinAcceptingWorkerPercent());
        for (double percent : new double[]{0.1, 10.0, 25.5, 100.0}) {
            assertEquals(percent, parse("{\"minAcceptingWorkerPercent\":" + percent + "}")
                    .queueScheduler().getPrefillQueueShedding().getMinAcceptingWorkerPercent());
        }
    }

    @Test void rejectsInvalidAcceptingWorkerPercent() {
        for (String percent : new String[]{"0", "-1", "100.01", "1e309", "\"NaN\"", "null"}) {
            assertThrows(RuntimeException.class, () -> parse("{\"minAcceptingWorkerPercent\":" + percent + "}"), percent);
        }
    }

    @Test void appliesDefaultThresholdsAndHardDeadline() {
        var config = parse("{}");
        assertEquals(30_000L, config.queueScheduler().getPrefillQueueShedding().getExpectedWaitThresholdMs());
        assertEquals(60_000L, config.queueScheduler().getPrefillQueueShedding().getMaxWaitMs());
        assertEquals(61_000L, config.resolveExpiresAtMs(1_000L));
        config.queueScheduler().setQueueTimeoutMs(100L);
        assertEquals(1_100L, config.resolveExpiresAtMs(1_000L));
        assertEquals(Long.MAX_VALUE, config.resolveExpiresAtMs(Long.MAX_VALUE - 1));
    }

    @Test void acceptsOnlyTimeThresholdConfiguration() {
        var config = parse("{\"expectedWaitThresholdMs\":10000,\"maxWaitMs\":20000}");
        assertEquals(10_000L, config.queueScheduler().getPrefillQueueShedding().getExpectedWaitThresholdMs());
        assertEquals(21_000L, config.resolveExpiresAtMs(1_000L));
        assertThrows(RuntimeException.class, () -> parse("{\"mode\":\"HIT_LENGTH\"}"));
        assertThrows(RuntimeException.class, () -> parse("{\"mode\":\"HIT_RATE\"}"));
    }

    @Test void rejectsInvalidThresholdsAndInactiveSchedulers() {
        assertThrows(RuntimeException.class, () -> parse("{\"expectedWaitThresholdMs\":0}"));
        assertThrows(RuntimeException.class, () -> parse("{\"maxWaitMs\":29999}"));
        assertThrows(RuntimeException.class, () -> parse("{\"mode\":null}"));
        assertThrows(RuntimeException.class, () -> parse("{\"mode\":\"RANDOM\"}"));
        assertThrows(RuntimeException.class, () -> ConfigTestFixtures.parse("""
                {"scheduler":{"type":"DIRECT","prefillQueueShedding":{}},"dispatcher":{"type":"NON_BATCH"}}
                """));
        assertThrows(RuntimeException.class, () -> ConfigTestFixtures.parse("""
                {"scheduler":{"type":"QUEUE","decision":{"type":"SINGLE"},"prefillQueueShedding":{}},"dispatcher":{"type":"BATCH"}}
                """));
    }

    @Test void validatesRecoveryWithoutExtendingHardDeadline() {
        var config = parse("{\"recovery\":{}}");
        assertEquals(10000, config.queueScheduler().getPrefillQueueShedding().getRecovery().getMaxGraceMs());
        assertEquals(61000, config.resolveExpiresAtMs(1000));
        for (String value : new String[]{"{\"windowMs\":0}", "{\"maxGraceMs\":0}",
                "{\"maxGraceMs\":60001}", "{\"waitMultiplier\":0.9}", "{\"maxEvictionFraction\":0}",
                "{\"minArrivalDropFraction\":1.1}", "{\"minQueueDrainFraction\":0}"}) {
            assertThrows(RuntimeException.class, () -> parse("{\"recovery\":" + value + "}"));
        }
    }

    @Test void remainsDisabledByDefault() {
        var config = ConfigTestFixtures.parse("{\"scheduler\":{\"type\":\"QUEUE\"},\"dispatcher\":{\"type\":\"NON_BATCH\"}}");
        assertNull(config.queueScheduler().getPrefillQueueShedding());
        assertEquals(3_601_000L, config.resolveExpiresAtMs(1_000L));
    }

    @Test void deadlineBudgetRequiresExplicitCalibrationAndTokenAccounting() {
        assertNull(parse("{}").queueScheduler().getPrefillQueueShedding().getDeadlineBudget());
        assertThrows(RuntimeException.class, () -> parse("{\"deadlineBudget\":{\"tokensPerSecond\":70000}}"));
        String prefix = "{\"router\":{\"roles\":{\"prefill\":{\"tokenBalance\":{}}}},"
                + "\"scheduler\":{\"type\":\"QUEUE\",\"prefillQueueShedding\":{\"deadlineBudget\":";
        var parsed = ConfigTestFixtures.parse(prefix + "{\"tokensPerSecond\":70000}}}}");
        var budget = parsed.queueScheduler().getPrefillQueueShedding().getDeadlineBudget();
        assertEquals(70_000, budget.getTokensPerSecond());
        assertEquals(0.9, budget.getCapacityFactor());
        for (String invalid : new String[]{"{}", "{\"tokensPerSecond\":0}", "{\"tokensPerSecond\":-1}",
                "{\"tokensPerSecond\":70000,\"capacityFactor\":1.1}",
                "{\"tokensPerSecond\":70000,\"saturationHoldMs\":0}",
                "{\"tokensPerSecond\":70000,\"saturatedWorkerPercent\":0}",
                "{\"tokensPerSecond\":70000,\"dispatchSlackMs\":60000}"}) {
            assertThrows(RuntimeException.class, () -> ConfigTestFixtures.parse(prefix + invalid + "}}}"));
        }
    }
}
