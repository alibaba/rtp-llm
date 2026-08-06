package org.flexlb.balance.scheduler;

import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.junit.jupiter.api.Test;

import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link PriorityYieldBatcherAlgorithm}.
 *
 * <p>Covers: sorting correctness (priority desc → deadline asc → arrival asc → requestId asc),
 * same-priority FIFO, yield skip-but-leave-in-queue behavior.
 */
class PriorityYieldBatcherAlgorithmTest {

    @Test
    void sortOrder_priorityDescending() {
        // Use a high SLO to prevent yield from triggering
        FlexlbConfig cfg = defaultConfig();
        cfg.setFlexlbBatchFixedWaitMs(0);
        cfg.setCostSloMs(100_000); // high SLO → no yield
        PriorityYieldBatcherAlgorithm alg = createAlgorithm(cfg);
        long now = System.currentTimeMillis();

        alg.offer(itemWithPriority(1, 30, now - 1000));
        alg.offer(itemWithPriority(2, 70, now - 1000));
        alg.offer(itemWithPriority(3, 50, now - 1000));

        BatchDecision decision = alg.decide();
        assertInstanceOf(BatchDecision.Dispatch.class, decision);
        BatchDecision.Dispatch dispatch = (BatchDecision.Dispatch) decision;
        // Should be sorted: priority 70, 50, 30
        assertEquals(3, dispatch.items().size());
        assertEquals(70, dispatch.items().get(0).priority());
        assertEquals(50, dispatch.items().get(1).priority());
        assertEquals(30, dispatch.items().get(2).priority());
    }

    @Test
    void sortOrder_samePriority_fifoByEnqueueTime() {
        FlexlbConfig cfg = defaultConfig();
        cfg.setFlexlbBatchFixedWaitMs(0);
        PriorityYieldBatcherAlgorithm alg = createAlgorithm(cfg);
        long now = System.currentTimeMillis();

        alg.offer(itemWithPriority(10, 50, now - 200));
        alg.offer(itemWithPriority(20, 50, now - 100));
        alg.offer(itemWithPriority(30, 50, now - 300));

        BatchDecision decision = alg.decide();
        assertInstanceOf(BatchDecision.Dispatch.class, decision);
        BatchDecision.Dispatch dispatch = (BatchDecision.Dispatch) decision;
        // Same priority → FIFO by enqueuedAtMs (ascending)
        assertEquals(30, dispatch.items().get(0).requestId()); // enqueuedAt -300
        assertEquals(10, dispatch.items().get(1).requestId()); // enqueuedAt -200
        assertEquals(20, dispatch.items().get(2).requestId()); // enqueuedAt -100
    }

    @Test
    void sortOrder_samePriorityAndTime_byRequestId() {
        FlexlbConfig cfg = defaultConfig();
        cfg.setFlexlbBatchFixedWaitMs(0);
        PriorityYieldBatcherAlgorithm alg = createAlgorithm(cfg);
        long now = System.currentTimeMillis();
        long enqueueTime = now - 500;

        alg.offer(itemWithPriority(99, 50, enqueueTime));
        alg.offer(itemWithPriority(11, 50, enqueueTime));
        alg.offer(itemWithPriority(55, 50, enqueueTime));

        BatchDecision decision = alg.decide();
        assertInstanceOf(BatchDecision.Dispatch.class, decision);
        BatchDecision.Dispatch dispatch = (BatchDecision.Dispatch) decision;
        // Same priority + same enqueuedAtMs → ascending requestId
        assertEquals(11, dispatch.items().get(0).requestId());
        assertEquals(55, dispatch.items().get(1).requestId());
        assertEquals(99, dispatch.items().get(2).requestId());
    }

    @Test
    void yieldSkipsLowPriorityButKeepsInQueue() {
        FlexlbConfig cfg = defaultConfig();
        cfg.setFlexlbBatchFixedWaitMs(0);
        // Set SLO to a value that will trigger yield (elapsed > sloMs / 2)
        cfg.setCostSloMs(100); // head waited > 50ms will trigger yield
        cfg.setFlexlbBatchSizeMax(32);
        PriorityYieldBatcherAlgorithm alg = createAlgorithm(cfg);
        long now = System.currentTimeMillis();

        // Head is high priority, enqueued long enough to trigger SLO risk
        alg.offer(itemWithPriority(1, 70, now - 200)); // elapsed 200 > sloMs/2=50
        alg.offer(itemWithPriority(2, 30, now - 100)); // low priority, should be skipped
        alg.offer(itemWithPriority(3, 70, now - 50));  // same priority as head, should be picked

        BatchDecision decision = alg.decide();
        assertInstanceOf(BatchDecision.Dispatch.class, decision);
        BatchDecision.Dispatch dispatch = (BatchDecision.Dispatch) decision;

        // Only items with priority >= 70 are picked
        assertEquals(2, dispatch.items().size());
        assertEquals(1, dispatch.items().get(0).requestId());
        assertEquals(3, dispatch.items().get(1).requestId());

        // The low-priority item (req 2) stays in queue
        assertEquals(1, alg.size());
    }

    @Test
    void noYieldWhenSloNotAtRisk() {
        FlexlbConfig cfg = defaultConfig();
        cfg.setFlexlbBatchFixedWaitMs(0);
        cfg.setCostSloMs(10000); // Very long SLO → elapsed won't exceed sloMs/2
        PriorityYieldBatcherAlgorithm alg = createAlgorithm(cfg);
        long now = System.currentTimeMillis();

        alg.offer(itemWithPriority(1, 70, now - 10)); // elapsed 10 < sloMs/2=5000
        alg.offer(itemWithPriority(2, 30, now - 5));

        BatchDecision decision = alg.decide();
        assertInstanceOf(BatchDecision.Dispatch.class, decision);
        BatchDecision.Dispatch dispatch = (BatchDecision.Dispatch) decision;

        // No yield: both items dispatched
        assertEquals(2, dispatch.items().size());
        assertEquals(0, alg.size());
    }

    @Test
    void drainTo_emptiesQueue() {
        PriorityYieldBatcherAlgorithm alg = createAlgorithm(defaultConfig());
        long now = System.currentTimeMillis();
        alg.offer(itemWithPriority(1, 50, now));
        alg.offer(itemWithPriority(2, 60, now));

        java.util.List<BatchItem> dst = new java.util.ArrayList<>();
        alg.drainTo(dst);

        assertEquals(2, dst.size());
        assertEquals(0, alg.size());
    }

    // ---- D12: 0-sentinel items take the legacy path ----

    @Test
    void zeroSentinel_neverYieldSkipped_dispatchedAlongsideHead() {
        FlexlbConfig cfg = defaultConfig();
        cfg.setFlexlbBatchFixedWaitMs(0);
        cfg.setCostSloMs(100); // head elapsed 200 > sloMs/2=50 → yield active
        PriorityYieldBatcherAlgorithm alg = createAlgorithm(cfg);
        long now = System.currentTimeMillis();

        alg.offer(itemWithPriority(1, 70, now - 200)); // head with SLO risk
        alg.offer(itemWithPriority(2, 0, now - 100));  // 0 sentinel: legacy, never skipped
        alg.offer(itemWithPriority(3, 30, now - 100)); // low priority: yield-skipped

        BatchDecision decision = alg.decide();
        assertInstanceOf(BatchDecision.Dispatch.class, decision);
        BatchDecision.Dispatch dispatch = (BatchDecision.Dispatch) decision;

        // The 0-sentinel item is dispatched with the head; only P30 is skipped.
        assertEquals(2, dispatch.items().size());
        assertEquals(1, dispatch.items().get(0).requestId());
        assertEquals(2, dispatch.items().get(1).requestId());
        assertEquals(1, alg.size());
    }

    @Test
    void zeroSentinel_expired_clearedAsLegacyQueueDeadline_notYielded() {
        FlexlbConfig cfg = defaultConfig();
        cfg.setFlexlbBatchFixedWaitMs(300);
        cfg.setFlexlbBatchEnqueueDeadlineMs(1_000);
        PriorityYieldBatcherAlgorithm alg = createAlgorithm(cfg);
        long now = System.currentTimeMillis();

        alg.offer(itemWithPriority(1, 70, now));           // fresh head
        alg.offer(itemWithPriority(2, 0, now - 2_000));    // expired 0 sentinel

        BatchDecision decision = alg.decide();
        assertInstanceOf(BatchDecision.Drop.class, decision);
        BatchDecision.Drop drop = (BatchDecision.Drop) decision;

        // Legacy queue-deadline attribution (BATCH_SLO_EXPIRED path),
        // never the yielded 8400 attribution.
        assertEquals(2, drop.item().requestId());
        assertEquals(BatchDecision.DropCause.QUEUE_DEADLINE_EXCEEDED, drop.cause());
        assertEquals(0, drop.yieldedForPriority());
        assertEquals(1, alg.size());
    }

    // ---- helpers ----

    private static FlexlbConfig defaultConfig() {
        FlexlbConfig cfg = new FlexlbConfig();
        cfg.setFlexlbBatchFixedWaitMs(300);
        cfg.setFlexlbBatchSizeMax(32);
        cfg.setFlexlbBatchFixedMaxInflightBatches(0);
        cfg.setFlexlbBatchEnqueueDeadlineMs(10_000);
        cfg.setCostSloMs(500);
        cfg.setAutoTpmEnabled(true);
        cfg.setAutoTpmQueueYieldEnabled(true);
        return cfg;
    }

    private static PriorityYieldBatcherAlgorithm createAlgorithm(FlexlbConfig cfg) {
        return new PriorityYieldBatcherAlgorithm(cfg, null);
    }

    private static BatchItem itemWithPriority(long requestId, int priority, long enqueuedAtMs) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(100);
        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(request);
        ctx.setPriority(priority);
        return new BatchItem(ctx, new CompletableFuture<>(),
                null, null, null, null, null, enqueuedAtMs);
    }
}
