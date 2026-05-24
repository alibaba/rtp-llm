package org.flexlb.cache.core;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.jupiter.api.Assertions.assertEquals;

class RecentCacheKeyWindowTest {

    @Test
    void should_count_request_hits_against_prior_window() {
        AtomicLong now = new AtomicLong(0L);
        RecentCacheKeyWindow window = new RecentCacheKeyWindow(1000L, now::get);

        RecentCacheKeyWindow.Snapshot first = window.record(List.of(1L, 2L, 3L));
        assertEquals(3L, first.getRequestOccurrences());
        assertEquals(0L, first.getRequestHitOccurrences());
        assertEquals(0.0, first.getRequestHitRatio(), 1e-9);
        assertEquals(3L, first.getRetainedOccurrences());
        assertEquals(3L, first.getRetainedUniqueCacheKeys());

        now.set(10L);
        RecentCacheKeyWindow.Snapshot second = window.record(List.of(2L, 3L, 4L));
        assertEquals(3L, second.getRequestOccurrences());
        assertEquals(2L, second.getRequestHitOccurrences());
        assertEquals(2.0 / 3.0, second.getRequestHitRatio(), 1e-9);
        assertEquals(6L, second.getRetainedOccurrences());
        assertEquals(4L, second.getRetainedUniqueCacheKeys());
    }

    @Test
    void should_decrement_counts_and_remove_keys_when_entries_expire_before_matching() {
        AtomicLong now = new AtomicLong(0L);
        RecentCacheKeyWindow window = new RecentCacheKeyWindow(1000L, now::get);

        window.record(List.of(1L, 2L, 3L));
        now.set(10L);
        window.record(List.of(2L, 3L, 4L));

        now.set(1001L);
        RecentCacheKeyWindow.Snapshot snapshot = window.record(List.of(1L, 2L, 3L, 4L));

        assertEquals(4L, snapshot.getRequestOccurrences());
        assertEquals(3L, snapshot.getRequestHitOccurrences());
        assertEquals(3.0 / 4.0, snapshot.getRequestHitRatio(), 1e-9);
        assertEquals(7L, snapshot.getRetainedOccurrences());
        assertEquals(4L, snapshot.getRetainedUniqueCacheKeys());
    }

    @Test
    void should_not_count_repeated_keys_in_one_request_as_self_hits() {
        AtomicLong now = new AtomicLong(0L);
        RecentCacheKeyWindow window = new RecentCacheKeyWindow(1000L, now::get);

        RecentCacheKeyWindow.Snapshot first = window.record(List.of(7L, 7L, 7L));

        assertEquals(3L, first.getRequestOccurrences());
        assertEquals(0L, first.getRequestHitOccurrences());
        assertEquals(0.0, first.getRequestHitRatio(), 1e-9);
        assertEquals(3L, first.getRetainedOccurrences());
        assertEquals(1L, first.getRetainedUniqueCacheKeys());

        now.set(10L);
        RecentCacheKeyWindow.Snapshot second = window.record(List.of(7L, 7L));

        assertEquals(2L, second.getRequestOccurrences());
        assertEquals(2L, second.getRequestHitOccurrences());
        assertEquals(1.0, second.getRequestHitRatio(), 1e-9);
    }

    @Test
    void should_ignore_null_keys_and_support_clear() {
        AtomicLong now = new AtomicLong(0L);
        RecentCacheKeyWindow window = new RecentCacheKeyWindow(1000L, now::get);

        RecentCacheKeyWindow.Snapshot snapshot = window.record(java.util.Arrays.asList(1L, null, 1L));
        assertEquals(2L, snapshot.getRequestOccurrences());
        assertEquals(0L, snapshot.getRequestHitOccurrences());
        assertEquals(2L, snapshot.getRetainedOccurrences());
        assertEquals(1L, snapshot.getRetainedUniqueCacheKeys());

        RecentCacheKeyWindow.Snapshot cleared = window.clear();
        assertEquals(0L, cleared.getRequestOccurrences());
        assertEquals(0L, cleared.getRequestHitOccurrences());
        assertEquals(0L, cleared.getRetainedOccurrences());
        assertEquals(0L, cleared.getRetainedUniqueCacheKeys());
        assertEquals(0.0, cleared.getRequestHitRatio(), 1e-9);
    }
}
