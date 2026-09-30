package org.flexlb.cache.telemetry;

import org.junit.jupiter.api.Test;

import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.assertEquals;

class TheoryCacheHitStatsTest {

    @Test
    void should_keep_all_time_token_sums() {
        TheoryCacheHitStats stats = new TheoryCacheHitStats();

        stats.record(1L, 4L);
        stats.record(2L, 6L);
        TheoryCacheHitStats.Snapshot snapshot = stats.record(3L, 10L);

        assertEquals(6L, snapshot.getAllHitCount());
        assertEquals(20L, snapshot.getAllTotalCount());
        assertEquals(3L, snapshot.getRequestHitCount());
        assertEquals(10L, snapshot.getRequestTotalCount());
    }

    @Test
    void should_keep_exact_sums_after_concurrent_updates() {
        TheoryCacheHitStats stats = new TheoryCacheHitStats();

        IntStream.range(0, 100_000).parallel()
                .forEach(ignored -> {
                    TheoryCacheHitStats.Snapshot snapshot = stats.record(1L, 4L);
                    assertEquals(snapshot.getAllHitCount() * 4L, snapshot.getAllTotalCount());
                });
        TheoryCacheHitStats.Snapshot snapshot = stats.record(0L, 0L);

        assertEquals(100_000L, snapshot.getAllHitCount());
        assertEquals(400_000L, snapshot.getAllTotalCount());
    }
}
