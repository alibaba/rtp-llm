package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.master.WorkerStatus;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Auto-TPM priority pick-order tests for {@link FixedWindowBatcherAlgorithm}
 * with {@code autoTpmPriorityQueueEnabled=true}. The switch-off baseline is
 * covered by the untouched {@link FixedWindowBatcherAlgorithmTest}; this
 * class only adds the switch-on behavior plus an explicit off-switch
 * FIFO-order regression.
 */
class FixedWindowBatcherAlgorithmPriorityTest {

    @Test
    void laterHighPriorityItemIsPickedFirstWhenEnabled() {
        FixedWindowBatcherAlgorithm algorithm =
                new FixedWindowBatcherAlgorithm(priorityConfig(), null);
        long now = System.currentTimeMillis() - 1_000;
        algorithm.offer(item(1, now, 10, 50));
        algorithm.offer(item(2, now + 1, 10, 70));
        algorithm.offer(item(3, now + 2, 10, 30));

        BatchDecision.Dispatch dispatch = assertInstanceOf(
                BatchDecision.Dispatch.class, algorithm.decide());

        assertEquals(List.of(2L, 1L, 3L),
                dispatch.items().stream().map(BatchItem::requestId).toList());
        assertEquals(0, algorithm.size());
    }

    @Test
    void samePriorityKeepsFifoOrderWhenEnabled() {
        FixedWindowBatcherAlgorithm algorithm =
                new FixedWindowBatcherAlgorithm(priorityConfig(), null);
        long now = System.currentTimeMillis() - 1_000;
        algorithm.offer(item(1, now, 10, 50));
        algorithm.offer(item(2, now + 1, 10, 50));
        algorithm.offer(item(3, now + 2, 10, 50));

        BatchDecision.Dispatch dispatch = assertInstanceOf(
                BatchDecision.Dispatch.class, algorithm.decide());

        assertEquals(List.of(1L, 2L, 3L),
                dispatch.items().stream().map(BatchItem::requestId).toList());
    }

    @Test
    void fifoOrderIsPreservedWhenDisabled() {
        FlexlbConfig config = priorityConfig();
        config.setAutoTpmPriorityQueueEnabled(false);
        FixedWindowBatcherAlgorithm algorithm =
                new FixedWindowBatcherAlgorithm(config, null);
        long now = System.currentTimeMillis() - 1_000;
        algorithm.offer(item(1, now, 10, 50));
        algorithm.offer(item(2, now + 1, 10, 70));

        BatchDecision.Dispatch dispatch = assertInstanceOf(
                BatchDecision.Dispatch.class, algorithm.decide());

        assertEquals(List.of(1L, 2L),
                dispatch.items().stream().map(BatchItem::requestId).toList());
    }

    @Test
    void oversizedPickOrderHeadIsDroppedAndDoesNotBlockDispatch() {
        FlexlbConfig config = priorityConfig();
        WorkerStatus status = new WorkerStatus();
        status.setMaxBatchTokensSize(100);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getStatus()).thenReturn(status);

        FixedWindowBatcherAlgorithm algorithm =
                new FixedWindowBatcherAlgorithm(config, endpoint);
        long now = System.currentTimeMillis() - 1_000;
        // Oldest item is a fitting P50; a newer oversized P70 jumps to the
        // front of the pick order and must be rejected there, not left to
        // block dispatch until it drifts to the FIFO head.
        algorithm.offer(item(1, now, 50, 50));
        algorithm.offer(item(2, now + 1, 100, 70));

        BatchDecision.Drop drop = assertInstanceOf(
                BatchDecision.Drop.class, algorithm.decide());
        assertEquals(BatchDecision.DropCause.EXCEEDS_BATCH_TOKEN_CAPACITY, drop.cause());
        assertEquals(2L, drop.item().requestId());
        assertEquals(1, algorithm.size());

        // Next cycle dispatches the fitting item — no blocking
        BatchDecision.Dispatch dispatch = assertInstanceOf(
                BatchDecision.Dispatch.class, algorithm.decide());
        assertEquals(List.of(1L),
                dispatch.items().stream().map(BatchItem::requestId).toList());
    }

    @Test
    void oldestLowPriorityItemStillExpiresAtFifoHead() {
        FlexlbConfig config = priorityConfig();
        config.setFlexlbBatchEnqueueDeadlineMs(100);
        FixedWindowBatcherAlgorithm algorithm =
                new FixedWindowBatcherAlgorithm(config, null);
        long now = System.currentTimeMillis();
        // FIFO head is an expired P30; a fresh P70 leads the pick order but
        // must not shadow the head expiry check.
        algorithm.offer(item(1, now - 1_000, 10, 30));
        algorithm.offer(item(2, now, 10, 70));

        BatchDecision.Drop drop = assertInstanceOf(
                BatchDecision.Drop.class, algorithm.decide());
        assertEquals(BatchDecision.DropCause.QUEUE_DEADLINE_EXCEEDED, drop.cause());
        assertEquals(1L, drop.item().requestId());
        assertEquals(1, algorithm.size());
    }

    @Test
    void windowTimingStaysAnchoredToOldestRequest() {
        FlexlbConfig config = priorityConfig();
        config.setFlexlbBatchFixedWaitMs(160);
        FixedWindowBatcherAlgorithm algorithm =
                new FixedWindowBatcherAlgorithm(config, null);
        long now = System.currentTimeMillis();
        // Oldest P30 has exceeded the window; the fresh P70 pick-order head
        // has not. Dispatch must fire (anchored to the oldest request) and
        // pick the P70 first.
        algorithm.offer(item(1, now - 170, 10, 30));
        algorithm.offer(item(2, now, 10, 70));

        BatchDecision.Dispatch dispatch = assertInstanceOf(
                BatchDecision.Dispatch.class, algorithm.decide());
        assertEquals("fixed_window_timeout", dispatch.reason());
        assertTrue(dispatch.headWaitMs() >= 170);
        assertEquals(List.of(2L, 1L),
                dispatch.items().stream().map(BatchItem::requestId).toList());
    }

    @Test
    void estimateWaitMsGrowsWithHigherPriorityItemsAhead() {
        FlexlbConfig config = priorityConfig();
        config.setFlexlbBatchSizeMax(2);
        config.setFlexlbBatchFixedWaitMs(100);
        FixedWindowBatcherAlgorithm algorithm =
                new FixedWindowBatcherAlgorithm(config, null);
        long now = System.currentTimeMillis();

        assertEquals(100, algorithm.estimateWaitMs(50));

        algorithm.offer(item(1, now, 10, 70));
        algorithm.offer(item(2, now + 1, 10, 70));
        assertEquals(200, algorithm.estimateWaitMs(50));

        algorithm.offer(item(3, now + 2, 10, 70));
        algorithm.offer(item(4, now + 3, 10, 70));
        assertEquals(300, algorithm.estimateWaitMs(50));

        // Lower-priority items do not precede a P50 arrival in pick order
        algorithm.offer(item(5, now + 4, 10, 30));
        algorithm.offer(item(6, now + 5, 10, 30));
        assertEquals(300, algorithm.estimateWaitMs(50));
    }

    @Test
    void depthByPriorityCountsAllValidLevels() {
        FixedWindowBatcherAlgorithm algorithm =
                new FixedWindowBatcherAlgorithm(priorityConfig(), null);
        long now = System.currentTimeMillis();
        algorithm.offer(item(1, now, 10, 70));
        algorithm.offer(item(2, now + 1, 10, 70));
        algorithm.offer(item(3, now + 2, 10, 50));

        Map<Integer, Integer> depth = algorithm.depthByPriority();

        assertEquals(Map.of(30, 0, 40, 0, 50, 1, 60, 0, 70, 2), depth);
    }

    // ---- helpers ----

    private static FlexlbConfig priorityConfig() {
        FlexlbConfig config = new FlexlbConfig();
        config.setAutoTpmPriorityQueueEnabled(true);
        config.setFlexlbBatchPredictThresholdMs(0);
        config.setFlexlbBatchFixedWaitMs(0);
        config.setFlexlbBatchSizeMax(32);
        config.setFlexlbBatchFixedMaxInflightBatches(0);
        config.setFlexlbBatchEnqueueDeadlineMs(10_000);
        return config;
    }

    private static BatchItem item(long requestId, long enqueuedAtMs, long seqLen, int priority) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(seqLen);
        request.setPriority(priority);
        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setRequest(request);
        return new BatchItem(
                balanceContext, new CompletableFuture<>(),
                null, null, null, null, null, enqueuedAtMs);
    }
}
