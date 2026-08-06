package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Field;
import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

/**
 * Switch matrix behavior assertions:
 * - Off state: no PriorityYield logic is invoked, FixedWindow is used exclusively
 * - On state: PriorityYield is active
 */
class AutoTpmSwitchMatrixTest {

    @Test
    void autoTpmDisabled_usesFixedWindow_noNewLogicInvolved() throws Exception {
        FlexlbConfig cfg = new FlexlbConfig();
        cfg.setAutoTpmEnabled(false);
        cfg.setAutoTpmQueueYieldEnabled(false);

        WorkerBatcher batcher = createBatcher(cfg);
        Object algorithm = getAlgorithm(batcher);

        assertInstanceOf(FixedWindowBatcherAlgorithm.class, algorithm);
        // Verify the algorithm is FixedWindow: offer and decide work without priority
        FixedWindowBatcherAlgorithm fixedAlg = (FixedWindowBatcherAlgorithm) algorithm;
        BatchItem item = makeItem(1, 0); // zero priority = not set
        fixedAlg.offer(item);
        assertEquals(1, fixedAlg.size());
    }

    @Test
    void autoTpmEnabled_yieldDisabled_usesFixedWindow() throws Exception {
        FlexlbConfig cfg = new FlexlbConfig();
        cfg.setAutoTpmEnabled(true);
        cfg.setAutoTpmQueueYieldEnabled(false);

        WorkerBatcher batcher = createBatcher(cfg);
        Object algorithm = getAlgorithm(batcher);

        assertInstanceOf(FixedWindowBatcherAlgorithm.class, algorithm);
    }

    @Test
    void autoTpmEnabled_yieldEnabled_usesPriorityYield() throws Exception {
        FlexlbConfig cfg = new FlexlbConfig();
        cfg.setAutoTpmEnabled(true);
        cfg.setAutoTpmQueueYieldEnabled(true);

        WorkerBatcher batcher = createBatcher(cfg);
        Object algorithm = getAlgorithm(batcher);

        assertInstanceOf(PriorityYieldBatcherAlgorithm.class, algorithm);
    }

    @Test
    void switchOff_fixedWindowBehavior_fullyPreserved() throws Exception {
        FlexlbConfig cfg = new FlexlbConfig();
        cfg.setAutoTpmEnabled(false);
        cfg.setFlexlbBatchFixedWaitMs(0); // immediate dispatch
        cfg.setFlexlbBatchSizeMax(10);
        cfg.setFlexlbBatchEnqueueDeadlineMs(10_000);

        WorkerBatcher batcher = createBatcher(cfg);
        FixedWindowBatcherAlgorithm fixedAlg =
                (FixedWindowBatcherAlgorithm) getAlgorithm(batcher);

        long now = System.currentTimeMillis();
        fixedAlg.offer(makeItem(1, 70, now - 100));
        fixedAlg.offer(makeItem(2, 30, now - 50));

        BatchDecision decision = fixedAlg.decide();
        assertInstanceOf(BatchDecision.Dispatch.class, decision);
        BatchDecision.Dispatch dispatch = (BatchDecision.Dispatch) decision;
        // FixedWindow doesn't care about priority — FIFO by enqueue order
        assertEquals(2, dispatch.items().size());
        assertEquals(1, dispatch.items().get(0).requestId()); // enqueued first
        assertEquals(2, dispatch.items().get(1).requestId());
    }

    // ---- helpers ----

    private static WorkerBatcher createBatcher(FlexlbConfig cfg) {
        PrefillEndpoint ep = mock(PrefillEndpoint.class);
        BatchSchedulerReporter reporter = mock(BatchSchedulerReporter.class);
        return new WorkerBatcher("test", ep, cfg, reporter);
    }

    private static Object getAlgorithm(WorkerBatcher batcher) throws Exception {
        Field field = WorkerBatcher.class.getDeclaredField("algorithm");
        field.setAccessible(true);
        return field.get(batcher);
    }

    private static BatchItem makeItem(long requestId, int priority) {
        return makeItem(requestId, priority, System.currentTimeMillis());
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
