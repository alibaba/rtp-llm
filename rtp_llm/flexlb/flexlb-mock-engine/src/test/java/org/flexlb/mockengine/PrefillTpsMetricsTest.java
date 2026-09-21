package org.flexlb.mockengine;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class PrefillTpsMetricsTest {
    @Test
    void ratioOfSumsCountsBatchDurationOnceAndSeparatesWallClock() {
        var metrics = new PrefillTpsMetrics(0);
        metrics.finish(metrics.begin(800, 1000, 0), 100_000_000L);
        metrics.finish(metrics.begin(400, 800, 100_000_000L), 400_000_000L);
        var rates = new PrefillTpsMetrics.Reader().sample(metrics.snapshot(), 2_000_000_000L);
        // C++ addTokenSize(800,1000,...,100000), then (400,800,...,300000).
        assertEquals(3000.0, rates.get("context_tps"));
        assertEquals(4500.0, rates.get("context_tps_with_cache"));
        assertEquals(600.0, rates.get("context_wall_tps"));
        assertEquals(900.0, rates.get("context_wall_tps_with_cache"));
        assertEquals(2_000_000.0, rates.get("wall_tps_report_interval_us"));
    }

    @Test
    void fullyCachedBatchDoesNotDiluteComputeDenominator() {
        var metrics = new PrefillTpsMetrics(0);
        metrics.finish(metrics.begin(800, 1000, 0), 100_000_000L);
        metrics.finish(metrics.begin(0, 1000, 100_000_000L), 300_000_000L);
        var rates = new PrefillTpsMetrics.Reader().sample(metrics.snapshot(), 1_000_000_000L);
        assertEquals(8000.0, rates.get("context_tps"));
        assertEquals(2000 * 1e6 / 300000, rates.get("context_tps_with_cache"));
        assertEquals(800.0, rates.get("context_wall_tps"));
        assertEquals(2000.0, rates.get("context_wall_tps_with_cache"));
    }

    @Test
    void longStepIsSilentAndItsWallWindowIsNotTruncated() {
        var metrics = new PrefillTpsMetrics(0);
        var reader = new PrefillTpsMetrics.Reader();
        var batch = metrics.begin(400, 1000, 0);
        assertTrue(reader.sample(metrics.snapshot(), 1_000_000_000L).isEmpty());
        assertTrue(reader.sample(metrics.snapshot(), 2_000_000_000L).isEmpty());
        metrics.finish(batch, 3_000_000_000L);
        var rates = reader.sample(metrics.snapshot(), 4_000_000_000L);
        assertEquals(400 / 3.0, rates.get("context_tps"));
        assertEquals(250.0, rates.get("context_wall_tps_with_cache"));
        assertEquals(4_000_000.0, rates.get("wall_tps_report_interval_us"));
        var idle = reader.sample(metrics.snapshot(), 5_000_000_000L);
        assertEquals(0.0, idle.get("context_tps"));
        assertEquals(0.0, idle.get("context_tps_with_cache"));
    }

    @Test
    void observersDoNotDrainEachOtherOrLoseTheFirstBatch() {
        var metrics = new PrefillTpsMetrics(0);
        var http = new PrefillTpsMetrics.Reader();
        var whale = new PrefillTpsMetrics.Reader();
        metrics.finish(metrics.begin(100, 200, 0), 100_000_000L);
        assertEquals(http.sample(metrics.snapshot(), 1_000_000_000L),
                whale.sample(metrics.snapshot(), 1_000_000_000L));
        assertEquals(0.0, http.sample(metrics.snapshot(), 2_000_000_000L).get("context_tps"));
        metrics.finish(metrics.begin(300, 600, 2_000_000_000L), 2_100_000_000L);
        var a = http.sample(metrics.snapshot(), 3_000_000_000L);
        var b = whale.sample(metrics.snapshot(), 3_000_000_000L);
        assertEquals(a.get("context_tps"), b.get("context_tps"));
        assertEquals(300.0, a.get("context_wall_tps"));
        assertEquals(150.0, b.get("context_wall_tps"));
    }

    @Test
    void restartRejectsOldCallbacksAndRebasesReaders() {
        var metrics = new PrefillTpsMetrics(0);
        var reader = new PrefillTpsMetrics.Reader();
        var stale = metrics.begin(999, 999, 0);
        reader.sample(metrics.snapshot(), 1_000_000_000L);
        metrics.reset(2_000_000_000L);
        metrics.finish(stale, 2_100_000_000L);
        metrics.finish(metrics.begin(100, 200, 2_000_000_000L), 2_100_000_000L);
        var rates = reader.sample(metrics.snapshot(), 3_000_000_000L);
        assertEquals(1000.0, rates.get("context_tps"));
        assertEquals(200.0, rates.get("context_wall_tps_with_cache"));
        assertEquals(0, metrics.snapshot().active());
    }

    @Test
    void zeroDurationAndEmptyComputeFollowRealCollectorGates() {
        var metrics = new PrefillTpsMetrics(0);
        metrics.finish(metrics.begin(100, 100, 0), 0);
        metrics.finish(metrics.begin(0, 100, 0), 100_000_000L);
        var rates = new PrefillTpsMetrics.Reader().sample(metrics.snapshot(), 1_000_000_000L);
        assertFalse(rates.containsKey("context_tps"));
        assertEquals(1000.0, rates.get("context_tps_with_cache"));
        assertEquals(0.0, rates.get("context_wall_tps"));
    }
}
