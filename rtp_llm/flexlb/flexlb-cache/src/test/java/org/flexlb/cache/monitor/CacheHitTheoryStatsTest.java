package org.flexlb.cache.monitor;

import org.junit.jupiter.api.Test;

import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CacheHitTheoryStatsTest {

    @Test
    void should_keep_all_time_token_sums() {
        CacheHitTheoryStats stats = new CacheHitTheoryStats(() -> 0L);

        stats.record(1L, 4L, 0L);
        stats.record(2L, 6L, 30_000L);
        CacheHitTheoryStats.Snapshot snapshot = stats.record(3L, 10L, 70_000L);

        assertEquals(6L, snapshot.getAllHitCount());
        assertEquals(20L, snapshot.getAllTotalCount());
        assertEquals(3L, snapshot.getRequestHitCount());
        assertEquals(10L, snapshot.getRequestTotalCount());
    }

    @Test
    void should_keep_exact_sums_after_concurrent_updates() {
        CacheHitTheoryStats stats = new CacheHitTheoryStats(() -> 0L);

        IntStream.range(0, 100_000).parallel()
                .forEach(ignored -> stats.record(1L, 4L, 0L));
        CacheHitTheoryStats.Snapshot snapshot = stats.record(0L, 0L, 0L);

        assertEquals(100_000L, snapshot.getAllHitCount());
        assertEquals(400_000L, snapshot.getAllTotalCount());
    }
}
