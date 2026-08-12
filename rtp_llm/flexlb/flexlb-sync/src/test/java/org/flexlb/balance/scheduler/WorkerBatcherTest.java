package org.flexlb.balance.scheduler;

import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.ScheduleBudget;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;

/**
 * Tests for {@link WorkerBatcher#queueSizeByPriority()}: per-priority
 * bucketing, the legacy priority-0 bucket, and the present-only empty-bucket
 * behavior (drained priorities disappear from the snapshot — same convention
 * as the batch wait-time-by-priority series).
 *
 * <p>Same construction pattern as {@link PrefillQueueManagerTest}: the
 * {@code fixed_window} algorithm needs no predictor, and the batcher is
 * never started so the queue content is fully deterministic.
 */
class WorkerBatcherTest {

    private FlexlbConfig config;

    @BeforeEach
    void setUp() {
        config = new FlexlbConfig();
        config.setFlexlbBatchAlgorithm("fixed_window");
        config.setAutoTpmEnabled(true);
    }

    private WorkerBatcher newBatcher() {
        return new WorkerBatcher("test-worker", null, config,
                mock(BatchDecisionHandler.class), mock(BatchSchedulerReporter.class));
    }

    @Test
    void queue_size_by_priority_buckets_multiple_priorities() {
        WorkerBatcher batcher = newBatcher();
        long now = System.currentTimeMillis();

        assertTrue(batcher.tryOffer(item(1, 70, now)));
        assertTrue(batcher.tryOffer(item(2, 50, now)));
        assertTrue(batcher.tryOffer(item(3, 50, now)));
        assertTrue(batcher.tryOffer(item(4, 30, now)));

        Map<Integer, Integer> buckets = batcher.queueSizeByPriority();
        assertEquals(Map.of(70, 1, 50, 2, 30, 1), buckets);
        // Bucket sum matches the global queue size
        assertEquals(batcher.queueSize(), buckets.values().stream().mapToInt(Integer::intValue).sum());
    }

    @Test
    void legacy_items_without_budget_fall_into_priority_zero_bucket() {
        config.setAutoTpmEnabled(false);
        WorkerBatcher batcher = newBatcher();
        long now = System.currentTimeMillis();

        assertTrue(batcher.tryOffer(legacyItem(1, now)));
        assertTrue(batcher.tryOffer(legacyItem(2, now)));

        assertEquals(Map.of(0, 2), batcher.queueSizeByPriority());
    }

    @Test
    void empty_queue_returns_empty_map() {
        assertEquals(Map.of(), newBatcher().queueSizeByPriority());
    }

    @Test
    void drained_priorities_disappear_from_snapshot() {
        WorkerBatcher batcher = newBatcher();
        long now = System.currentTimeMillis();

        assertTrue(batcher.tryOffer(item(1, 70, now)));
        assertTrue(batcher.tryOffer(item(2, 50, now)));

        // Drain the P70 item: its bucket drops out (present-only, no zero-fill
        // — same empty-bucket behavior as wait-time-by-priority)
        List<BatchItem> removed = batcher.tryRemoveNoVersion(List.of(1L), "test-drain");
        assertEquals(1, removed.size());

        assertEquals(Map.of(50, 1), batcher.queueSizeByPriority());

        // Fully drained queue reports no buckets at all
        assertEquals(1, batcher.tryRemoveNoVersion(List.of(2L), "test-drain").size());
        assertEquals(Map.of(), batcher.queueSizeByPriority());
    }

    // ==================== helpers ====================

    private static BatchItem item(long requestId, int priority, long enqueuedAtMs) {
        BalanceContext ctx = newContext(requestId, priority);
        ctx.setBudget(ScheduleBudget.forDeadline(priority, enqueuedAtMs, enqueuedAtMs + 5_000));
        return new BatchItem(ctx, new CompletableFuture<>(), null,
                null, null, null, null, enqueuedAtMs);
    }

    /** Legacy path: budget = null, so {@link BatchItem#priority()} returns 0. */
    private static BatchItem legacyItem(long requestId, long enqueuedAtMs) {
        return new BatchItem(newContext(requestId, 0), new CompletableFuture<>(), null,
                null, null, null, null, enqueuedAtMs);
    }

    private static BalanceContext newContext(long requestId, int priority) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(128);
        request.setPriority(priority);
        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(request);
        return ctx;
    }
}
