package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;

/**
 * Tests for {@link WorkerBatcher#peekBatchItems()}.
 *
 * <p>Verifies the {@code remaining = queueSize % batchMaxCount} logic that
 * determines which queue items a new request would join in the same batch.
 */
class WorkerBatcherTest {

    @Test
    void peekBatchItemsEmptyQueueReturnsEmptyList() {
        WorkerBatcher batcher = createBatcher(8);
        try {
            List<BatchItem> result = batcher.peekBatchItems();
            assertTrue(result.isEmpty(), "empty queue should return empty list");
        } finally {
            batcher.shutdown();
        }
    }

    @Test
    void peekBatchItemsShortQueueReturnsAllItems() {
        // 3 items, batchMaxCount=8: remaining = 3 % 8 = 3 → return all 3
        WorkerBatcher batcher = createBatcher(8);
        try {
            batcher.offer(enqueuedItem(1, 100));
            batcher.offer(enqueuedItem(2, 101));
            batcher.offer(enqueuedItem(3, 102));

            List<BatchItem> result = batcher.peekBatchItems();
            assertEquals(3, result.size(), "short queue should return all items");
        } finally {
            batcher.shutdown();
        }
    }

    @Test
    void peekBatchItemsLongQueueReturnsTailItems() {
        // 10 items, batchMaxCount=8: remaining = 10 % 8 = 2 → return last 2 sorted items
        WorkerBatcher batcher = createBatcher(8);
        try {
            for (int i = 0; i < 10; i++) {
                batcher.offer(enqueuedItem(i + 1, 100 + i));
            }

            List<BatchItem> result = batcher.peekBatchItems();
            assertEquals(2, result.size(), "long queue should return tail 2 items");
            // The tail items should have the highest sortKeys (108, 109)
            assertEquals(108, result.get(0).sortKey());
            assertEquals(109, result.get(1).sortKey());
        } finally {
            batcher.shutdown();
        }
    }

    @Test
    void peekBatchItemsRemainingZeroReturnsEmptyList() {
        // 8 items, batchMaxCount=8: remaining = 8 % 8 = 0 → empty list
        WorkerBatcher batcher = createBatcher(8);
        try {
            for (int i = 0; i < 8; i++) {
                batcher.offer(enqueuedItem(i + 1, 100 + i));
            }

            List<BatchItem> result = batcher.peekBatchItems();
            assertTrue(result.isEmpty(), "remaining=0 should return empty list");
        } finally {
            batcher.shutdown();
        }
    }

    // ---- helpers ----

    private static WorkerBatcher createBatcher(int batchSizeMax) {
        FlexlbConfig config = new FlexlbConfig();
        config.setFlexlbBatchAlgorithm("fixed_window");
        config.setFlexlbBatchSizeMax(batchSizeMax);
        // queue max size default (1024) is sufficient; set explicitly for clarity
        config.setFlexlbBatchQueueMaxSize(1024);
        PrefillEndpoint prefillEp = mock(PrefillEndpoint.class);
        BatchDecisionHandler handler = mock(BatchDecisionHandler.class);
        BatchSchedulerReporter reporter = mock(BatchSchedulerReporter.class);
        return new WorkerBatcher("test", prefillEp, config, handler, reporter);
    }

    private static BatchItem enqueuedItem(long requestId, long enqueuedAtMs) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(100);
        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(request);
        // sortKey is overwritten by WorkerBatcher.offer() via computeSortKey,
        // so the initial value 0 here is irrelevant for FixedWindow algorithm.
        return new BatchItem(ctx, null, null, null, null, null, null, 0, enqueuedAtMs);
    }
}
