package org.flexlb.mockengine;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class CounterRateMetricsTest {
    @Test
    void ratesUseElapsedSecondsAndIndependentObservers() {
        var http = new CounterRateMetrics.Reader();
        var whale = new CounterRateMetrics.Reader();
        var initial = new CounterRateMetrics.Snapshot(0, 0, 100);
        assertEquals(50.0, http.sample(initial, 2_000_000_000L));
        assertEquals(50.0, whale.sample(initial, 2_000_000_000L));
        assertEquals(0.0, http.sample(initial, 3_000_000_000L));
        var next = new CounterRateMetrics.Snapshot(0, 0, 160);
        assertEquals(60.0, http.sample(next, 4_000_000_000L));
        assertEquals(30.0, whale.sample(next, 4_000_000_000L));
    }

    @Test
    void restartRebasesAndRepeatedTimestampsDoNotConsumeTokens() {
        var reader = new CounterRateMetrics.Reader();
        assertEquals(100.0, reader.sample(new CounterRateMetrics.Snapshot(0, 0, 100), 1_000_000_000L));
        assertEquals(100.0, reader.sample(new CounterRateMetrics.Snapshot(0, 0, 120), 1_000_000_000L));
        assertEquals(20.0, reader.sample(new CounterRateMetrics.Snapshot(0, 0, 120), 2_000_000_000L));
        assertEquals(5.0, reader.sample(new CounterRateMetrics.Snapshot(1, 3_000_000_000L, 10), 5_000_000_000L));
        reader.reset();
        assertEquals(0.0, reader.last());
        assertEquals(2.0, reader.sample(new CounterRateMetrics.Snapshot(2, 6_000_000_000L, 4), 8_000_000_000L));
    }
}
