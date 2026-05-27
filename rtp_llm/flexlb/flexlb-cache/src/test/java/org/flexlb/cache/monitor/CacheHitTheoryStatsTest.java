package org.flexlb.cache.monitor;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CacheHitTheoryStatsTest {

    @Test
    void should_keep_all_time_and_rolling_window_sums() {
        CacheHitTheoryStats stats = new CacheHitTheoryStats(() -> 0L);

        stats.record(1L, 4L, 0L);
        stats.record(2L, 6L, 30_000L);
        CacheHitTheoryStats.Snapshot snapshot = stats.record(3L, 10L, 70_000L);

        assertEquals(6L, snapshot.getAllHitCount());
        assertEquals(20L, snapshot.getAllTotalCount());
        assertEquals(5L, snapshot.getWindow1m().getHitCount());
        assertEquals(16L, snapshot.getWindow1m().getTotalCount());
        assertEquals(6L, snapshot.getWindow5m().getHitCount());
        assertEquals(20L, snapshot.getWindow5m().getTotalCount());
    }
}
