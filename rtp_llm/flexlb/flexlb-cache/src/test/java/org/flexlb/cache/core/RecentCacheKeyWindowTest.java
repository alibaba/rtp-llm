package org.flexlb.cache.core;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.jupiter.api.Assertions.assertEquals;

class RecentCacheKeyWindowTest {

    @Test
    void should_count_request_hits_against_prior_pool() {
        AtomicLong now = new AtomicLong(0L);
        RecentCacheKeyWindow window = new RecentCacheKeyWindow(1000L, 10L, now::get);

        RecentCacheKeyWindow.Snapshot first = window.record(List.of(1L, 2L, 3L));
        assertEquals(3L, first.getRequestOccurrences());
        assertEquals(0L, first.getRequestHitOccurrences());

        now.set(10L);
        RecentCacheKeyWindow.Snapshot second = window.record(List.of(2L, 3L, 4L));
        assertEquals(3L, second.getRequestOccurrences());
        assertEquals(2L, second.getRequestHitOccurrences());
    }

    @Test
    void should_expire_entries_by_time_window_before_matching() {
        AtomicLong now = new AtomicLong(0L);
        RecentCacheKeyWindow window = new RecentCacheKeyWindow(1000L, 10L, now::get);

        window.record(List.of(1L, 2L, 3L));
        now.set(1001L);
        RecentCacheKeyWindow.Snapshot snapshot = window.record(List.of(1L, 2L, 3L));

        assertEquals(3L, snapshot.getRequestOccurrences());
        assertEquals(0L, snapshot.getRequestHitOccurrences());
    }

    @Test
    void should_not_count_repeated_keys_in_one_request_as_self_hits() {
        AtomicLong now = new AtomicLong(0L);
        RecentCacheKeyWindow window = new RecentCacheKeyWindow(1000L, 10L, now::get);

        RecentCacheKeyWindow.Snapshot first = window.record(List.of(7L, 7L, 7L));
        assertEquals(3L, first.getRequestOccurrences());
        assertEquals(0L, first.getRequestHitOccurrences());

        now.set(10L);
        RecentCacheKeyWindow.Snapshot second = window.record(List.of(7L, 7L));
        assertEquals(2L, second.getRequestOccurrences());
        assertEquals(2L, second.getRequestHitOccurrences());
    }

    @Test
    void should_bound_pool_by_cache_key_capacity() {
        AtomicLong now = new AtomicLong(0L);
        RecentCacheKeyWindow window = new RecentCacheKeyWindow(60_000L, 3L, now::get);

        window.record(List.of(1L, 2L));
        now.set(1L);
        window.record(List.of(3L, 4L));
        now.set(2L);
        RecentCacheKeyWindow.Snapshot snapshot = window.record(List.of(1L, 3L, 4L));

        assertEquals(3L, snapshot.getRequestOccurrences());
        assertEquals(2L, snapshot.getRequestHitOccurrences());
    }

    @Test
    void should_ignore_null_keys() {
        AtomicLong now = new AtomicLong(0L);
        RecentCacheKeyWindow window = new RecentCacheKeyWindow(1000L, 10L, now::get);

        RecentCacheKeyWindow.Snapshot snapshot = window.record(java.util.Arrays.asList(1L, null, 1L));

        assertEquals(2L, snapshot.getRequestOccurrences());
        assertEquals(0L, snapshot.getRequestHitOccurrences());
    }
}
