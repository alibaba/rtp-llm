package org.flexlb.cache.core;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
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

    @Test
    void should_expire_shared_history_before_matching() {
        AtomicLong now = new AtomicLong(0L);
        ShardedRecentCacheKeyWindow window =
                new ShardedRecentCacheKeyWindow(1000L, 40L, now::get);

        window.record(1L, List.of(11L));
        now.set(999L);
        assertEquals(1L, window.record(2L, List.of(11L)).getRequestHitOccurrences());
        now.set(1999L);
        assertEquals(0L, window.record(3L, List.of(11L)).getRequestHitOccurrences());
    }

    @Test
    void should_apply_capacity_to_the_global_history() {
        ShardedRecentCacheKeyWindow window =
                new ShardedRecentCacheKeyWindow(1000L, 4L, () -> 0L);

        window.record(1L, List.of(11L, 22L));
        window.record(2L, List.of(33L, 44L));
        assertEquals(1L, window.record(3L, List.of(33L)).getRequestHitOccurrences());
        // The oldest whole request was evicted to retain the preceding record.
        assertEquals(2L, window.record(4L, List.of(11L, 22L, 33L, 44L))
                .getRequestHitOccurrences());
    }

    @Test
    void should_not_count_duplicate_keys_as_hits_within_the_same_request() {
        ShardedRecentCacheKeyWindow window =
                new ShardedRecentCacheKeyWindow(1000L, 40L, () -> 0L);

        assertEquals(0L, window.record(1L, List.of(11L, 11L)).getRequestHitOccurrences());
        assertEquals(2L, window.record(2L, List.of(11L, 11L)).getRequestHitOccurrences());
    }

    @Test
    void should_atomically_match_and_retain_concurrent_requests() throws Exception {
        int requestCount = 32;
        ShardedRecentCacheKeyWindow window =
                new ShardedRecentCacheKeyWindow(1000L, requestCount * 3L, () -> 0L);
        CountDownLatch start = new CountDownLatch(1);
        List<Future<RecentCacheKeyWindow.Snapshot>> results = new ArrayList<>();
        var executor = Executors.newFixedThreadPool(8);
        try {
            for (long requestId = 0; requestId < requestCount; requestId++) {
                long id = requestId;
                results.add(executor.submit(() -> {
                    start.await();
                    return window.record(id, List.of(11L, 22L, 11L));
                }));
            }
            start.countDown();
            int coldRequests = 0;
            for (Future<RecentCacheKeyWindow.Snapshot> result : results) {
                RecentCacheKeyWindow.Snapshot snapshot = result.get(5, TimeUnit.SECONDS);
                assertEquals(3L, snapshot.getRequestOccurrences());
                if (snapshot.getRequestHitOccurrences() == 0L) {
                    coldRequests++;
                } else {
                    assertEquals(3L, snapshot.getRequestHitOccurrences());
                }
            }
            assertEquals(1, coldRequests);
        } finally {
            start.countDown();
            executor.shutdownNow();
        }
    }
}
