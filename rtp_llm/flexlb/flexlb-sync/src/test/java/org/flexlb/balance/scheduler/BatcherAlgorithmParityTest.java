package org.flexlb.balance.scheduler;

import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.junit.jupiter.api.Test;

import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Parity test: when autoTpmQueueYieldEnabled is false (or autoTpmEnabled is false),
 * PriorityYieldBatcherAlgorithm should produce the same output as
 * FixedWindowBatcherAlgorithm for the same input sequence.
 *
 * <p>Since yield is only active when SLO risk exists AND items have different
 * priorities, this test uses same-priority items to verify identical behavior.
 */
class BatcherAlgorithmParityTest {

    @Test
    void samePriority_yieldAlgorithm_matchesFixedWindow_dispatchOrder() {
        FlexlbConfig cfg = parityCfg();
        FixedWindowBatcherAlgorithm fixed = new FixedWindowBatcherAlgorithm(cfg, null);
        PriorityYieldBatcherAlgorithm priority = new PriorityYieldBatcherAlgorithm(cfg, null);

        long now = System.currentTimeMillis();
        // All same priority (50), same seqLen → FIFO should be identical
        for (int i = 1; i <= 5; i++) {
            BatchItem item = makeItem(i, 50, now - (5 - i) * 100L);
            fixed.offer(item);
            priority.offer(makeItem(i, 50, now - (5 - i) * 100L));
        }

        BatchDecision fixedDec = fixed.decide();
        BatchDecision priorityDec = priority.decide();

        assertNotNull(fixedDec);
        assertNotNull(priorityDec);
        assertInstanceOf(BatchDecision.Dispatch.class, fixedDec);
        assertInstanceOf(BatchDecision.Dispatch.class, priorityDec);

        BatchDecision.Dispatch fDispatch = (BatchDecision.Dispatch) fixedDec;
        BatchDecision.Dispatch pDispatch = (BatchDecision.Dispatch) priorityDec;

        assertEquals(fDispatch.items().size(), pDispatch.items().size());
        for (int i = 0; i < fDispatch.items().size(); i++) {
            assertEquals(fDispatch.items().get(i).requestId(),
                    pDispatch.items().get(i).requestId(),
                    "Item " + i + " should match between algorithms");
        }
    }

    @Test
    void emptyQueue_bothReturnNull() {
        FlexlbConfig cfg = parityCfg();
        FixedWindowBatcherAlgorithm fixed = new FixedWindowBatcherAlgorithm(cfg, null);
        PriorityYieldBatcherAlgorithm priority = new PriorityYieldBatcherAlgorithm(cfg, null);

        assertNull(fixed.decide());
        assertNull(priority.decide());
    }

    // ---- helpers ----

    private static FlexlbConfig parityCfg() {
        FlexlbConfig cfg = new FlexlbConfig();
        cfg.setFlexlbBatchFixedWaitMs(0); // immediate dispatch
        cfg.setFlexlbBatchSizeMax(32);
        cfg.setFlexlbBatchFixedMaxInflightBatches(0);
        cfg.setFlexlbBatchEnqueueDeadlineMs(10_000);
        cfg.setCostSloMs(500);
        cfg.setAutoTpmEnabled(true);
        cfg.setAutoTpmQueueYieldEnabled(true);
        return cfg;
    }

    private static BatchItem makeItem(long requestId, int priority, long enqueuedAtMs) {
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
