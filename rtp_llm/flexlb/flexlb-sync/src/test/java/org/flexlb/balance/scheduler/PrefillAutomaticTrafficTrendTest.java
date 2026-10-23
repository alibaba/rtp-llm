package org.flexlb.balance.scheduler;

import org.flexlb.config.PrefillQueueSheddingConfig;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import static org.junit.jupiter.api.Assertions.*;

class PrefillAutomaticTrafficTrendTest {
    private final PrefillTrafficTrend trend = new PrefillTrafficTrend(new PrefillQueueSheddingConfig.RecoveryConfig());

    @ParameterizedTest @ValueSource(longs = {100, 1_000, 30_000, 120_000})
    void drainingBurstUsesSloBudgetAndEventuallyReturnsToNormal(long soft) {
        long step = Math.max(1, Math.min(2_000, soft / 10));
        double queue = soft * 1.8;
        assertTrue(trend.observe(0, queue, soft, soft * 2).grace());
        for (int i = 1; i <= 10; i++) {
            queue -= soft * .09;
            var observed = trend.observe(i * step, Math.max(0, queue), soft, soft * 2);
            if (queue > soft) { assertTrue(observed.grace(), "measured drain fits the remaining bounded budget"); }
            else { assertEquals(PrefillTrafficTrend.State.NORMAL, observed.state()); }
        }
    }

    @Test void sustainedGrowthLosesGraceAfterObservationAndRetriesNeverExtendBudget() {
        for (int i = 0; i < 50; i++) {
            for (int j = 0; j <= i; j++) { trend.arrive(100_000); }
            var observed = trend.observe(i * 500L, 35_000 + i * 100, 30_000, 60_000);
            assertEquals(i < 8, observed.grace());
            if (i >= 8) { assertEquals(PrefillTrafficTrend.State.GROWING, observed.state()); }
        }
    }

    @Test void evictionAndTimeoutCannotImpersonateDrain() {
        trend.observe(0, 55_000, 30_000, 60_000);
        for (int i = 1; i <= 20; i++) {
            trend.discard(500);
            var observed = trend.observe(i * 500L, 55_000 - i * 500, 30_000, 60_000);
            if (i >= 8) { assertFalse(observed.grace()); }
        }
    }

    @Test void shortDipBelowThresholdCannotRearmAnEpisode() {
        trend.observe(0, 50_000, 30_000, 60_000);
        assertFalse(trend.observe(31_000, 50_000, 30_000, 60_000).grace());
        trend.observe(31_100, 29_000, 30_000, 60_000);
        assertFalse(trend.observe(31_200, 50_000, 30_000, 60_000).grace());
        trend.observe(32_000, 10_000, 30_000, 60_000);
        trend.observe(36_000, 10_000, 30_000, 60_000);
        assertTrue(trend.observe(36_100, 50_000, 30_000, 60_000).grace());
    }

    @Test void hardLimitAndZeroHeadroomNeverGrantGrace() {
        assertFalse(trend.observe(0, 60_000, 30_000, 60_000).grace());
        assertFalse(trend.observe(500, 35_000, 30_000, 30_000).grace());
        assertThrows(IllegalArgumentException.class, () -> trend.observe(400, 0, 30_000, 60_000));
    }

    @Test void slowDrainingBacklogMustFitBudgetBeforeGettingMoreGrace() {
        trend.observe(0, 58_000, 30_000, 60_000);
        for (int i = 1; i <= 20; i++) {
            var observed = trend.observe(i * 500, 58_000 - i * 10, 30_000, 60_000);
            if (i >= 8) { assertFalse(observed.grace(), "a tiny negative slope is insufficient evidence of recovery"); }
        }
    }

    @Test void longPrefillGetsTimeToCompleteBeforeFlatQueueIsClassifiedAsGrowth() {
        assertTrue(trend.observe(0, 4_000, 3_000, 6_000, 1_000).grace());
        assertTrue(trend.observe(700, 4_000, 3_000, 6_000, 1_000).grace());
        assertTrue(trend.observe(1_000, 3_800, 3_000, 6_000, 1_000).grace());
        assertTrue(trend.observe(1_500, 3_400, 3_000, 6_000, 1_000).grace());
        assertFalse(trend.observe(3_001, 3_300, 3_000, 6_000, 10_000).grace(),
                "later longer queries cannot renew the original episode budget");
    }
}
