package org.flexlb.cache.core;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.jupiter.api.Assertions.assertEquals;

class ShardedRecentCacheKeyWindowTest {

    @Test
    void should_count_hits_across_requests_with_different_request_ids() {
        AtomicLong now = new AtomicLong(0L);
        ShardedRecentCacheKeyWindow window =
                new ShardedRecentCacheKeyWindow(1000L, 320L, now::get);

        RecentCacheKeyWindow.Snapshot first = window.record(1L, List.of(11L, 22L, 33L));
        assertEquals(0L, first.getRequestHitOccurrences());

        now.set(10L);
        RecentCacheKeyWindow.Snapshot second = window.record(2L, List.of(11L, 22L, 44L));
        assertEquals(3L, second.getRequestOccurrences());
        assertEquals(2L, second.getRequestHitOccurrences());
    }

    @Test
    void should_count_all_request_occurrences() {
        AtomicLong now = new AtomicLong(0L);
        ShardedRecentCacheKeyWindow window =
                new ShardedRecentCacheKeyWindow(1000L, 40L, now::get);

        window.record(7L, List.of(1L, 2L, 3L, 4L, 5L));
        now.set(10L);
        RecentCacheKeyWindow.Snapshot snapshot =
                window.record(99L, List.of(1L, 2L, 3L, 4L, 5L, 6L));

        assertEquals(6L, snapshot.getRequestOccurrences());
        assertEquals(5L, snapshot.getRequestHitOccurrences());
    }
}
