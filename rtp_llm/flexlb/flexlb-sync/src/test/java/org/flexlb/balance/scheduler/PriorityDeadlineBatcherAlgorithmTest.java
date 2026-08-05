package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.strategy.PrefillTimePredictor;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import java.util.Comparator;
import java.util.List;
import java.util.Set;
import java.util.concurrent.PriorityBlockingQueue;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class PriorityDeadlineBatcherAlgorithmTest {

    private static final long DEADLINE_MULT = 10_000_000_000_000L;

    // ==================== Sorting tests ====================

    @Test
    void prioritySorting_p70BeforeP50BeforeP30() {
        FlexlbConfig config = new FlexlbConfig();
        config.setFlexlbBatchAlgorithm("priority_deadline");
        BatcherContext ctx = context("test", null, config, null,
                queueWith(), mock(BatchSchedulerReporter.class));

        PriorityDeadlineBatcherAlgorithm algo = new PriorityDeadlineBatcherAlgorithm();
        long deadline = 10_000_000L; // same deadline for all

        BatchItem p70 = priorityItem(1L, 70, deadline);
        BatchItem p50 = priorityItem(2L, 50, deadline);
        BatchItem p30 = priorityItem(3L, 30, deadline);

        long key70 = algo.computeSortKey(ctx, p70);
        long key50 = algo.computeSortKey(ctx, p50);
        long key30 = algo.computeSortKey(ctx, p30);

        assertTrue(key70 < key50, "P70 sortKey must be less than P50");
        assertTrue(key50 < key30, "P50 sortKey must be less than P30");
    }

    @Test
    void deadlineSorting_samePriorityEarlierDeadlineFirst() {
        FlexlbConfig config = new FlexlbConfig();
        BatcherContext ctx = context("test", null, config, null,
                queueWith(), mock(BatchSchedulerReporter.class));

        PriorityDeadlineBatcherAlgorithm algo = new PriorityDeadlineBatcherAlgorithm();
        long earlyDeadline = 5_000_000L;
        long lateDeadline = 8_000_000L;

        BatchItem early = priorityItem(1L, 50, earlyDeadline);
        BatchItem late = priorityItem(2L, 50, lateDeadline);

        long keyEarly = algo.computeSortKey(ctx, early);
        long keyLate = algo.computeSortKey(ctx, late);

        assertTrue(keyEarly < keyLate, "Earlier deadline must sort first within same priority");
    }

    @Test
    void priorityOverridesDeadline_p30EarlierDeadlineStillAfterP70() {
        FlexlbConfig config = new FlexlbConfig();
        BatcherContext ctx = context("test", null, config, null,
                queueWith(), mock(BatchSchedulerReporter.class));

        PriorityDeadlineBatcherAlgorithm algo = new PriorityDeadlineBatcherAlgorithm();
        // P30 with very early deadline vs P70 with very late deadline
        BatchItem p30Early = priorityItem(1L, 30, 1_000_000L);
        BatchItem p70Late = priorityItem(2L, 70, 9_000_000L);

        long keyP30 = algo.computeSortKey(ctx, p30Early);
        long keyP70 = algo.computeSortKey(ctx, p70Late);

        assertTrue(keyP70 < keyP30, "P70 must always sort before P30 regardless of deadline");
    }

    // ==================== No silent drop test ====================

    @Test
    void deadlineExpired_returnsToSchedulerNotSilentDrop() throws InterruptedException {
        FlexlbConfig config = priorityConfig();
        config.setFlexlbBatchFixedWaitMs(0);
        config.setFlexlbBatchSizeMax(8);

        PrefillEndpoint endpoint = mockEndpoint(1000);
        BatchDecisionHandler handler = mock(BatchDecisionHandler.class);

        long pastDeadline = System.currentTimeMillis() - 10_000; // 10s ago
        BatchItem expired = priorityItem(1L, 50, pastDeadline);
        expired.setSortKey(computeSortKeyRaw(50, pastDeadline));

        BatcherContext ctx = context("test", endpoint, config, handler,
                queueWith(expired), mock(BatchSchedulerReporter.class));

        new PriorityDeadlineBatcherAlgorithm().processQueue(ctx);

        // Must call onDeadlineExceeded, NOT onExpired
        verify(handler).onDeadlineExceeded(expired);
        verify(handler, never()).onExpired(any());
        verify(handler, never()).onBatchReady(anyList(), any());
        assertEquals(0, ctx.size(), "Expired item must be removed from queue");
    }

    // ==================== CAS API tests ====================

    @Test
    void tryRemove_correctVersionSucceeds() {
        BatchItem item1 = priorityItem(1L, 50, 10_000_000L);
        BatchItem item2 = priorityItem(2L, 70, 9_000_000L);
        item1.setSortKey(computeSortKeyRaw(50, 10_000_000L));
        item2.setSortKey(computeSortKeyRaw(70, 9_000_000L));

        BatcherContext ctx = context("test", null, new FlexlbConfig(), null,
                queueWith(item1, item2), mock(BatchSchedulerReporter.class));

        assertEquals(2, ctx.size());

        QueueSnapshot snapshot = ctx.snapshot();
        long version = snapshot.version();
        assertEquals(2, snapshot.queueSize());

        List<BatchItem> removed = ctx.tryRemove(Set.of(1L), version);
        assertNotNull(removed, "tryRemove should return non-null list with correct version");
        assertEquals(1, removed.size(), "exactly one victim should be removed");
        assertEquals(1L, removed.get(0).requestId());
        assertEquals(1, ctx.size(), "Queue should have 1 item after removing 1");
    }

    @Test
    void tryRemove_wrongVersionFails() {
        BatchItem item1 = priorityItem(1L, 50, 10_000_000L);
        item1.setSortKey(computeSortKeyRaw(50, 10_000_000L));

        BatcherContext ctx = context("test", null, new FlexlbConfig(), null,
                queueWith(item1), mock(BatchSchedulerReporter.class));

        QueueSnapshot snapshot = ctx.snapshot();
        long correctVersion = snapshot.version();

        // Use wrong version → must return null
        List<BatchItem> removed = ctx.tryRemove(Set.of(1L), correctVersion + 999);
        assertNull(removed, "tryRemove should return null with wrong version");
        assertEquals(1, ctx.size(), "Queue should remain unchanged after failed CAS");
    }

    @Test
    void tryOffer_correctVersionSucceeds() {
        BatchItem item1 = priorityItem(1L, 50, 10_000_000L);
        item1.setSortKey(computeSortKeyRaw(50, 10_000_000L));

        BatcherContext ctx = context("test", null, new FlexlbConfig(), null,
                queueWith(item1), mock(BatchSchedulerReporter.class));

        QueueSnapshot snapshot = ctx.snapshot();
        long version = snapshot.version();

        BatchItem newItem = priorityItem(2L, 70, 9_000_000L);
        newItem.setSortKey(computeSortKeyRaw(70, 9_000_000L));

        boolean result = ctx.tryOffer(newItem, version);
        assertTrue(result, "tryOffer should succeed with correct version");
        assertEquals(2, ctx.size(), "Queue should have 2 items after offer");
    }

    @Test
    void tryOffer_wrongVersionFails() {
        BatcherContext ctx = context("test", null, new FlexlbConfig(), null,
                queueWith(), mock(BatchSchedulerReporter.class));

        QueueSnapshot snapshot = ctx.snapshot();
        long correctVersion = snapshot.version();

        BatchItem newItem = priorityItem(1L, 70, 9_000_000L);
        newItem.setSortKey(computeSortKeyRaw(70, 9_000_000L));

        boolean result = ctx.tryOffer(newItem, correctVersion + 999);
        assertFalse(result, "tryOffer should fail with wrong version");
        assertEquals(0, ctx.size(), "Queue should remain empty after failed CAS");
    }

    // ==================== Snapshot test ====================

    @Test
    void snapshot_returnsCorrectVersionAndItems() {
        BatchItem item1 = priorityItem(1L, 70, 9_000_000L);
        BatchItem item2 = priorityItem(2L, 50, 10_000_000L);
        item1.setSortKey(computeSortKeyRaw(70, 9_000_000L));
        item2.setSortKey(computeSortKeyRaw(50, 10_000_000L));

        BatcherContext ctx = context("test", null, new FlexlbConfig(), null,
                queueWith(item1, item2), mock(BatchSchedulerReporter.class));

        QueueSnapshot snapshot = ctx.snapshot();
        assertEquals(2, snapshot.queueSize());
        assertEquals(2, snapshot.items().size());

        // Verify item summaries
        QueueSnapshot.ItemSummary first = snapshot.items().get(0);
        assertEquals(1L, first.requestId(), "First item should be P70 (lower sortKey)");
        assertEquals(70, first.priority());

        QueueSnapshot.ItemSummary second = snapshot.items().get(1);
        assertEquals(2L, second.requestId());
        assertEquals(50, second.priority());
    }

    @Test
    void versionBumpsOnEveryMutation() {
        BatchItem item1 = priorityItem(1L, 50, 10_000_000L);
        item1.setSortKey(computeSortKeyRaw(50, 10_000_000L));

        BatcherContext ctx = context("test", null, new FlexlbConfig(), null,
                queueWith(item1), mock(BatchSchedulerReporter.class));

        long v0 = ctx.version();
        ctx.remove(item1);
        long v1 = ctx.version();
        assertTrue(v1 > v0, "Version must bump after remove");

        // tryOffer bumps version
        boolean offered = ctx.tryOffer(item1, v1);
        assertTrue(offered, "tryOffer should succeed with correct version");
        long v2 = ctx.version();
        assertTrue(v2 > v1, "Version must bump after tryOffer");

        // tryRemove bumps version
        ctx.tryRemove(Set.of(1L), v2);
        long v3 = ctx.version();
        assertTrue(v3 > v2, "Version must bump after tryRemove");
    }

    // ==================== Dispatch test ====================

    @Test
    void dispatchesBatchWhenFull() throws InterruptedException {
        int batchSizeMax = 2;
        FlexlbConfig config = priorityConfig();
        config.setFlexlbBatchSizeMax(batchSizeMax);
        config.setFlexlbBatchWindowMs(10_000);

        PrefillEndpoint endpoint = mockEndpoint(1000);
        BatchDecisionHandler handler = mock(BatchDecisionHandler.class);

        long futureDeadline = System.currentTimeMillis() + 60_000;
        BatchItem item1 = priorityItem(1L, 70, futureDeadline);
        BatchItem item2 = priorityItem(2L, 50, futureDeadline + 1000);
        item1.setSortKey(computeSortKeyRaw(70, futureDeadline));
        item2.setSortKey(computeSortKeyRaw(50, futureDeadline + 1000));

        BatcherContext ctx = context("test", endpoint, config, handler,
                queueWith(item1, item2), mock(BatchSchedulerReporter.class));

        new PriorityDeadlineBatcherAlgorithm().processQueue(ctx);

        ArgumentCaptor<List<BatchItem>> dispatched = ArgumentCaptor.forClass(List.class);
        verify(handler).onBatchReady(dispatched.capture(), any(DispatchMeta.class));

        List<BatchItem> batch = dispatched.getValue();
        assertEquals(2, batch.size());
        assertEquals(1L, batch.get(0).requestId(), "P70 must be first in dispatched batch");
        assertEquals(2L, batch.get(1).requestId(), "P50 must be second");
    }

    // ==================== estimateWait test ====================

    @Test
    void estimateWait_countsAheadItems() {
        FlexlbConfig config = new FlexlbConfig();
        long deadline = 10_000_000L;
        BatchItem p70_1 = priorityItem(1L, 70, deadline);
        BatchItem p70_2 = priorityItem(2L, 70, deadline + 100);
        BatchItem p70_3 = priorityItem(3L, 70, deadline + 200);
        p70_1.setSortKey(computeSortKeyRaw(70, deadline));
        p70_2.setSortKey(computeSortKeyRaw(70, deadline + 100));
        p70_3.setSortKey(computeSortKeyRaw(70, deadline + 200));

        BatcherContext ctx = context("test", null, config, null,
                queueWith(p70_1, p70_2, p70_3), mock(BatchSchedulerReporter.class));

        PriorityDeadlineBatcherAlgorithm algo = new PriorityDeadlineBatcherAlgorithm();
        BatchItem incoming = priorityItem(99L, 50, deadline);
        long wait = algo.estimateWait(ctx, incoming);

        // 3 P70 items ahead of P50 incoming, each ~50ms
        assertEquals(3 * 50L, wait);
    }

    // ==================== Helpers ====================

    private static FlexlbConfig priorityConfig() {
        FlexlbConfig config = new FlexlbConfig();
        config.setFlexlbBatchAlgorithm("priority_deadline");
        config.setFlexlbBatchWindowMs(300);
        config.setFlexlbBatchMinSize(1);
        config.setFlexlbBatchSizeMax(8);
        config.setFlexlbBatchEmergencyBudgetMs(150);
        config.setFlexlbBatchDispatchGuardMs(40);
        config.setFlexlbBatchSloMaxInflightBatches(0);
        config.setFlexlbBatchScanAhead(64);
        config.setFlexlbBatchMaxCapacity(1_048_576);
        return config;
    }

    private static long computeSortKeyRaw(int priority, long deadlineMs) {
        return (long) (70 - priority) * DEADLINE_MULT + deadlineMs;
    }

    private static BatchItem priorityItem(long requestId, int priority, long deadlineMs) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(100);
        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(request);
        ctx.setPriority(priority);
        ctx.setDeadlineMs(deadlineMs);

        BatchItem item = new BatchItem(ctx, null, null, null, null, null, null,
                System.currentTimeMillis());
        item.setPriority(priority);
        item.setDeadlineMs(deadlineMs);
        return item;
    }

    private static PrefillEndpoint mockEndpoint(int inflightBatches) {
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getInflightBatchCount()).thenReturn(inflightBatches);
        when(endpoint.getIp()).thenReturn("127.0.0.1");
        when(endpoint.ipPort()).thenReturn("127.0.0.1:61000");

        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        when(predictor.estimateMs(anyLong(), anyLong())).thenReturn(50L);
        when(predictor.predictBatchMs(anyList())).thenReturn(50.0);
        when(predictor.predictBatchMsUncached(anyList())).thenReturn(50.0);
        when(endpoint.getPredictor()).thenReturn(predictor);

        WorkerStatus status = new WorkerStatus();
        status.setMaxBatchTokensSize(1_048_576);
        status.setMaxSeqLen(131_072L);
        when(endpoint.getStatus()).thenReturn(status);

        return endpoint;
    }

    private static PriorityBlockingQueue<BatchItem> queueWith(BatchItem... items) {
        PriorityBlockingQueue<BatchItem> queue =
                new PriorityBlockingQueue<>(11, Comparator.comparingLong(BatchItem::sortKey));
        for (BatchItem item : items) {
            queue.add(item);
        }
        return queue;
    }

    private static BatcherContext context(String key, PrefillEndpoint endpoint,
                                           FlexlbConfig config, BatchDecisionHandler handler,
                                           PriorityBlockingQueue<BatchItem> queue,
                                           BatchSchedulerReporter reporter) {
        return new BatcherContext(key, endpoint, config, handler, queue,
                new AtomicInteger(queue.size()), reporter);
    }
}
