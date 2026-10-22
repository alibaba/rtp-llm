package org.flexlb.balance.scheduler;

import org.flexlb.config.PrefillQueueSheddingConfig;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class PrefillTrafficTrendTest {
    private static PrefillQueueSheddingConfig.RecoveryConfig config() {
        var config = new PrefillQueueSheddingConfig.RecoveryConfig();
        config.setWindowMs(1000);
        config.setMaxGraceMs(1500);
        config.setBurstProbeMs(1);
        return config;
    }
    private static void arrivals(PrefillTrafficTrend trend, int count, long tokens) {
        for (int i = 0; i < count; i++) { trend.arrive(tokens); }
    }
    @Test void sustainedRampNeverGetsGrace() {
        var trend = new PrefillTrafficTrend(config());
        trend.observe(0, 31000, 30000, 60000);
        for (int i = 1; i <= 20; i++) {
            arrivals(trend, i * 10, 100000);
            assertFalse(trend.observe(i * 500L, 31000 + i * 1000, 30000, 60000).grace());
        }
    }
    @Test void decreasingBurstGetsOneBoundedGraceAndCannotResetWhileOverloaded() {
        var trend = new PrefillTrafficTrend(config());
        trend.observe(0, 58000, 30000, 60000);
        for (int i = 1; i <= 4; i++) {
            arrivals(trend, 100, 100000);
            trend.observe(i * 500L, 58000, 30000, 60000);
        }
        for (int i = 5; i <= 6; i++) {
            arrivals(trend, 10, 100000);
            var observed = trend.observe(i * 500L, 58000 - (i - 4) * 4000, 30000, 60000);
            assertTrue(observed.grace());
            assertEquals(PrefillTrafficTrend.State.RECOVERING, observed.state());
        }
        for (int i = 7; i <= 14; i++) {
            arrivals(trend, 1, 100000);
            var observed = trend.observe(i * 500L, 50000 - (i - 6) * 2000, 30000, 60000);
            if (i >= 8) { assertFalse(observed.grace(), "finite grace cannot extend the deadline"); }
        }
    }
    @Test void evictionIsNotNaturalDrainAndLongerInputsAreNotTrafficRecovery() {
        for (boolean eviction : new boolean[]{true, false}) {
            var trend = new PrefillTrafficTrend(config());
            trend.observe(0, 50000, 30000, 60000);
            for (int i = 1; i <= 4; i++) {
                arrivals(trend, 100, 1000);
                trend.observe(i * 500L, 50000, 30000, 60000);
            }
            for (int i = 5; i <= 6; i++) {
                arrivals(trend, 10, eviction ? 1000 : 100000);
                if (eviction) { trend.discard(5000); }
                assertFalse(trend.observe(i * 500L, 50000 - (i - 4) * 5000, 30000, 60000).grace());
            }
        }
    }
    @Test void hardOverloadAndMissingHistoryNeverRelax() {
        var trend = new PrefillTrafficTrend(config());
        assertFalse(trend.observe(0, 90000, 30000, 60000).grace());
        for (int i = 1; i <= 4; i++) {
            arrivals(trend, 100, 1000);
            trend.observe(i * 500L, 90000, 30000, 60000);
        }
        assertFalse(trend.observe(3000, 65000, 30000, 60000).grace());
        assertThrows(IllegalArgumentException.class, () -> trend.observe(2000, 1, 30000, 60000));
    }
    @Test void abruptBurstGetsFiniteObservationButUnchangingBacklogCannotKeepIt() {
        var recovery = new PrefillQueueSheddingConfig.RecoveryConfig();
        var trend = new PrefillTrafficTrend(recovery);
        arrivals(trend, 1000, 100000);
        var first = trend.observe(0, 50000, 30000, 60000);
        assertEquals(PrefillTrafficTrend.State.BURST_PROBE, first.state());
        assertTrue(first.grace());
        for (int i = 1; i <= 10; i++) {
            var observed = trend.observe(i * 1000L, 50000, 30000, 60000);
            if (i >= 4) { assertFalse(observed.grace()); }
        }
    }

    @Test void softProbeNeverOverridesHardCapacityOrEvictionPriority() {
        var config = new PrefillQueueSheddingConfig();
        config.setRecovery(new PrefillQueueSheddingConfig.RecoveryConfig());
        var trend = new PrefillTrafficTrend(config.getRecovery());
        var queue = java.util.List.of(
                new PrefillQueueShedding.Candidate<>("low", 10, 1,
                        new PrefillQueueShedding.Estimate(100, 90, 25000, 1)),
                new PrefillQueueShedding.Candidate<>("high", 90, 2,
                        new PrefillQueueShedding.Estimate(100, 0, 25000, 1)));
        var probe = trend.observe(0, 50000, 30000, 60000);
        assertTrue(PrefillQueueShedding.adaptiveVictims(queue, config, probe).isEmpty());
        var hard = trend.observe(100, 65000, 30000, 60000);
        assertFalse(hard.grace());
        assertEquals(java.util.List.of("low"), PrefillQueueShedding.adaptiveVictims(queue, config, hard));
        assertEquals(java.util.List.of("low"), PrefillQueueShedding.adaptiveVictims(queue, config,
                trend.observe(5000, 50000, 30000, 60000)));
    }

}
