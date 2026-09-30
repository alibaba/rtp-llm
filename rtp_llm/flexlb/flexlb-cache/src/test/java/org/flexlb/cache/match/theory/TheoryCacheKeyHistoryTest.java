package org.flexlb.cache.match.theory;

import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusProvider;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.LongSupplier;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class TheoryCacheKeyHistoryTest {

    @Test
    void should_limit_history_to_ten_times_prefill_block_capacity() {
        FlexlbConfig config = new FlexlbConfig();
        config.getObservability().getCacheHit().getRecentKeyWindow().setMaxKeyOccurrences(100L);
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        TheoryCacheKeyHistory window = new TheoryCacheKeyHistory(
                configService, prefillCapacity(100L, 100L));
        window.record(List.of(1L, 2L, 3L, 4L, 5L));
        window.record(List.of(6L, 7L, 8L, 9L, 10L));
        window.record(List.of(11L));
        assertEquals(0L, window.record(List.of(1L)).getRequestHitOccurrences());
        assertEquals(1L, window.record(List.of(6L)).getRequestHitOccurrences());
    }

    @Test
    void should_wait_for_prefill_capacity_before_recording_history() {
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(new FlexlbConfig());
        WorkerStatusProvider provider = mock(WorkerStatusProvider.class);
        when(provider.getWorkerStatuses(RoleType.PREFILL, null)).thenReturn(List.of());
        TheoryCacheKeyHistory window = new TheoryCacheKeyHistory(configService, provider);
        assertNull(window.record(List.of(11L)));
        WorkerStatusProvider ready = prefillCapacity(100L, 100L);
        var readyWorkers = ready.getWorkerStatuses(RoleType.PREFILL, null);
        when(provider.getWorkerStatuses(RoleType.PREFILL, null)).thenReturn(readyWorkers);
        assertEquals(0L, window.record(List.of(11L)).getRequestHitOccurrences());
    }

    @Test
    void should_read_one_prefill_capacity_and_multiply_by_worker_count() {
        ConfigService configService = mock(ConfigService.class);
        FlexlbConfig config = new FlexlbConfig();
        config.getObservability().getCacheHit().getRecentKeyWindow().setMaxKeyOccurrences(100L);
        when(configService.loadBalanceConfig()).thenReturn(config);
        var firstWorkers = prefillCapacity(200L, 100L).getWorkerStatuses(RoleType.PREFILL, null);
        WorkerStatus secondWorker = mock(WorkerStatus.class);
        List<WorkerStatus> prefillWorkers = new ArrayList<>(firstWorkers);
        prefillWorkers.add(secondWorker);
        WorkerStatusProvider provider = mock(WorkerStatusProvider.class);
        when(provider.getWorkerStatuses(RoleType.PREFILL, null)).thenReturn(prefillWorkers);
        TheoryCacheKeyHistory window = new TheoryCacheKeyHistory(configService, provider);
        List<Long> firstKeys = java.util.stream.LongStream.range(0L, 20L).boxed().toList();
        List<Long> secondKeys = java.util.stream.LongStream.range(20L, 40L).boxed().toList();
        window.record(firstKeys);
        window.record(secondKeys);
        window.record(List.of(40L));
        assertEquals(0L, window.record(List.of(0L)).getRequestHitOccurrences());
        assertEquals(1L, window.record(List.of(20L)).getRequestHitOccurrences());
        verifyNoInteractions(secondWorker);
    }

    private static TheoryCacheKeyHistory historyWithWindow(long durationMs, long maxKeys, LongSupplier clock)
            throws Exception {
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(new FlexlbConfig());
        TheoryCacheKeyHistory history = new TheoryCacheKeyHistory(configService, mock(WorkerStatusProvider.class));
        Field window = TheoryCacheKeyHistory.class.getDeclaredField("historyWindow");
        window.setAccessible(true);
        window.set(history, new RecentCacheKeyWindow(durationMs, maxKeys, clock));
        return history;
    }

    private static WorkerStatusProvider prefillCapacity(long totalKvCacheTokens, long blockSize) {
        WorkerStatusProvider provider = mock(WorkerStatusProvider.class);
        WorkerStatus worker = mock(WorkerStatus.class);
        WorkerStatus.EngineObservation capacity = mock(WorkerStatus.EngineObservation.class);
        when(provider.getWorkerStatuses(RoleType.PREFILL, null)).thenReturn(List.of(worker));
        when(worker.committedEngineObservation()).thenReturn(capacity);
        when(capacity.totalKvCacheTokens()).thenReturn(totalKvCacheTokens);
        when(capacity.blockSize()).thenReturn(blockSize);
        return provider;
    }

    @Test
    void should_read_configured_capacity_when_master_window_is_created() {
        FlexlbConfig config = new FlexlbConfig();
        config.getObservability().getCacheHit().getRecentKeyWindow().setMaxKeyOccurrences(2L);
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        TheoryCacheKeyHistory small = new TheoryCacheKeyHistory(configService, prefillCapacity(1000L, 100L));
        small.record(List.of(11L, 22L));
        small.record(List.of(33L));
        assertEquals(0L, small.record(List.of(11L)).getRequestHitOccurrences());

        config.getObservability().getCacheHit().getRecentKeyWindow().setMaxKeyOccurrences(4L);
        TheoryCacheKeyHistory restarted = new TheoryCacheKeyHistory(configService, prefillCapacity(1000L, 100L));
        restarted.record(List.of(11L, 22L));
        restarted.record(List.of(33L));
        assertEquals(1L, restarted.record(List.of(11L)).getRequestHitOccurrences());
    }

    @Test
    void should_count_hits_across_requests() throws Exception {
        AtomicLong now = new AtomicLong(0L);
        TheoryCacheKeyHistory window =
                historyWithWindow(1000L, 320L, now::get);

        RecentCacheKeyWindow.Snapshot first = window.record(List.of(11L, 22L, 33L));
        assertEquals(0L, first.getRequestHitOccurrences());

        now.set(10L);
        RecentCacheKeyWindow.Snapshot second = window.record(List.of(11L, 22L, 44L));
        assertEquals(3L, second.getRequestOccurrences());
        assertEquals(2L, second.getRequestHitOccurrences());
    }

    @Test
    void should_count_all_request_occurrences() throws Exception {
        AtomicLong now = new AtomicLong(0L);
        TheoryCacheKeyHistory window =
                historyWithWindow(1000L, 40L, now::get);

        window.record(List.of(1L, 2L, 3L, 4L, 5L));
        now.set(10L);
        RecentCacheKeyWindow.Snapshot snapshot =
                window.record(List.of(1L, 2L, 3L, 4L, 5L, 6L));

        assertEquals(6L, snapshot.getRequestOccurrences());
        assertEquals(5L, snapshot.getRequestHitOccurrences());
    }

    @Test
    void should_expire_shared_history_before_matching() throws Exception {
        AtomicLong now = new AtomicLong(0L);
        TheoryCacheKeyHistory window =
                historyWithWindow(1000L, 40L, now::get);

        window.record(List.of(11L));
        now.set(999L);
        assertEquals(1L, window.record(List.of(11L)).getRequestHitOccurrences());
        now.set(1999L);
        assertEquals(0L, window.record(List.of(11L)).getRequestHitOccurrences());
    }

    @Test
    void should_apply_capacity_to_the_global_history() throws Exception {
        TheoryCacheKeyHistory window =
                historyWithWindow(1000L, 4L, () -> 0L);

        window.record(List.of(11L, 22L));
        window.record(List.of(33L, 44L));
        assertEquals(1L, window.record(List.of(33L)).getRequestHitOccurrences());
        // The oldest whole request was evicted to retain the preceding record.
        assertEquals(2L, window.record(List.of(11L, 22L, 33L, 44L))
                .getRequestHitOccurrences());
    }

    @Test
    void should_not_count_duplicate_keys_as_hits_within_the_same_request() throws Exception {
        TheoryCacheKeyHistory window =
                historyWithWindow(1000L, 40L, () -> 0L);

        assertEquals(0L, window.record(List.of(11L, 11L)).getRequestHitOccurrences());
        assertEquals(2L, window.record(List.of(11L, 11L)).getRequestHitOccurrences());
    }

    @Test
    void should_atomically_match_and_retain_concurrent_requests() throws Exception {
        int requestCount = 32;
        TheoryCacheKeyHistory window =
                historyWithWindow(1000L, requestCount * 3L, () -> 0L);
        CountDownLatch start = new CountDownLatch(1);
        List<Future<RecentCacheKeyWindow.Snapshot>> results = new ArrayList<>();
        var executor = Executors.newFixedThreadPool(8);
        try {
            for (int i = 0; i < requestCount; i++) {
                results.add(executor.submit(() -> {
                    start.await();
                    return window.record(List.of(11L, 22L, 11L));
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
