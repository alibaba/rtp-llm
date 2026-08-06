package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.master.WorkerStatus;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.fail;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Multi-node scenarios for the Auto-TPM priority pick order: each Prefill
 * worker owns an independent {@link FixedWindowBatcherAlgorithm} instance, so
 * the switch state and pick order must apply per node without any cross-node
 * effect, and {@link FixedWindowBatcherAlgorithm#queueWaitMs} (the routing
 * score input) must be byte-identical between switch states for the same
 * queue content.
 *
 * <p>Single-node pick-order basics are covered by
 * {@link FixedWindowBatcherAlgorithmPriorityTest}; this class only adds the
 * multi-instance angle plus capacity-constraint gaps (maxCount cutoff, break
 * semantics, KV budget under priority order).
 */
class MultiEndpointPrioritySchedulingTest {

    /** Large fixed window so queueWaitMs returns a non-trivial positive value. */
    private static final long PARITY_FIXED_WAIT_MS = 3_600_000L;

    // ==================== queueWaitMs parity (routing score input) ====================

    @Test
    void queueWaitMsIsIdenticalAcrossSwitchStatesOnEveryNode() {
        long now = System.currentTimeMillis();
        // Per-node queue contents: {requestId, enqueuedAtMs, seqLen, priority}.
        // Node 1: mixed priorities with a late high-priority arrival.
        // Node 2: all high priority. Node 3: empty queue.
        List<long[][]> nodeContents = List.of(
                new long[][] {{1, now - 50, 10, 30}, {2, now - 40, 10, 70}, {3, now - 30, 10, 50}},
                new long[][] {{11, now - 500, 10, 70}, {12, now - 400, 10, 70}},
                new long[][] {});

        for (long[][] content : nodeContents) {
            FixedWindowBatcherAlgorithm enabled = node(true, PARITY_FIXED_WAIT_MS);
            FixedWindowBatcherAlgorithm disabled = node(false, PARITY_FIXED_WAIT_MS);
            for (long[] row : content) {
                enabled.offer(item(row[0], row[1], row[2], (int) row[3]));
                disabled.offer(item(row[0], row[1], row[2], (int) row[3]));
            }
            assertSameTickQueueWaitMs(enabled, disabled);
        }
    }

    @Test
    void laterHighPriorityArrivalDoesNotChangeQueueWaitMs() {
        FixedWindowBatcherAlgorithm enabled = node(true, PARITY_FIXED_WAIT_MS);
        long now = System.currentTimeMillis();
        enabled.offer(item(1, now - 100, 10, 30));

        // Sample queueWaitMs before and after a high-priority offer within the
        // same millisecond tick: the value is anchored to the FIFO head, so a
        // pick-order-leading arrival must not move it.
        long nextRequestId = 100;
        for (int attempt = 0; attempt < 1_000; attempt++) {
            long tick = System.currentTimeMillis();
            long before = enabled.queueWaitMs();
            enabled.offer(item(nextRequestId++, System.currentTimeMillis(), 10, 70));
            long after = enabled.queueWaitMs();
            if (System.currentTimeMillis() == tick) {
                assertEquals(before, after,
                        "queueWaitMs must stay anchored to the FIFO head");
                return;
            }
        }
        fail("could not sample queueWaitMs twice within a single millisecond tick");
    }

    // ==================== per-node independence ====================

    @Test
    void eachNodeAppliesItsOwnPickOrderIndependently() {
        // Node A: low-priority backlog plus one late high-priority arrival.
        FixedWindowBatcherAlgorithm nodeA = node(true, 0);
        // Node B: all low priority — must keep pure FIFO order.
        FixedWindowBatcherAlgorithm nodeB = node(true, 0);
        long now = System.currentTimeMillis() - 1_000;
        nodeA.offer(item(1, now, 10, 30));
        nodeA.offer(item(2, now + 1, 10, 30));
        nodeA.offer(item(3, now + 2, 10, 70));
        nodeB.offer(item(11, now, 10, 30));
        nodeB.offer(item(12, now + 1, 10, 30));
        nodeB.offer(item(13, now + 2, 10, 30));

        BatchDecision.Dispatch dispatchA = assertInstanceOf(
                BatchDecision.Dispatch.class, nodeA.decide());
        assertEquals(List.of(3L, 1L, 2L),
                dispatchA.items().stream().map(BatchItem::requestId).toList());
        // Node A's dispatch must not touch node B's queue
        assertEquals(3, nodeB.size());

        BatchDecision.Dispatch dispatchB = assertInstanceOf(
                BatchDecision.Dispatch.class, nodeB.decide());
        assertEquals(List.of(11L, 12L, 13L),
                dispatchB.items().stream().map(BatchItem::requestId).toList());
    }

    @Test
    void switchStateIsFrozenPerNodeAtConstruction() {
        // Two nodes built from the same config object at different switch
        // states: each instance freezes the state it saw at construction.
        FlexlbConfig shared = config(true, 0);
        FixedWindowBatcherAlgorithm priorityNode = new FixedWindowBatcherAlgorithm(shared, null);
        shared.setAutoTpmPriorityQueueEnabled(false);
        FixedWindowBatcherAlgorithm fifoNode = new FixedWindowBatcherAlgorithm(shared, null);

        long now = System.currentTimeMillis() - 1_000;
        priorityNode.offer(item(1, now, 10, 30));
        priorityNode.offer(item(2, now + 1, 10, 70));
        fifoNode.offer(item(1, now, 10, 30));
        fifoNode.offer(item(2, now + 1, 10, 70));

        BatchDecision.Dispatch priorityDispatch = assertInstanceOf(
                BatchDecision.Dispatch.class, priorityNode.decide());
        assertEquals(List.of(2L, 1L),
                priorityDispatch.items().stream().map(BatchItem::requestId).toList());

        BatchDecision.Dispatch fifoDispatch = assertInstanceOf(
                BatchDecision.Dispatch.class, fifoNode.decide());
        assertEquals(List.of(1L, 2L),
                fifoDispatch.items().stream().map(BatchItem::requestId).toList());
    }

    // ==================== capacity-constrained pick correctness ====================

    @Test
    void highPriorityWinsBatchSlotsUnderMaxCountLimit() {
        FlexlbConfig config = config(true, 0);
        config.setFlexlbBatchSizeMax(2);
        FixedWindowBatcherAlgorithm algorithm = new FixedWindowBatcherAlgorithm(config, null);
        long now = System.currentTimeMillis() - 1_000;
        algorithm.offer(item(1, now, 10, 30));
        algorithm.offer(item(2, now + 1, 10, 30));
        algorithm.offer(item(3, now + 2, 10, 70));

        // maxCount=2: the late P70 takes a slot, the oldest P30 fills the rest
        BatchDecision.Dispatch first = assertInstanceOf(
                BatchDecision.Dispatch.class, algorithm.decide());
        assertEquals(List.of(3L, 1L),
                first.items().stream().map(BatchItem::requestId).toList());
        assertEquals(1, algorithm.size());

        BatchDecision.Dispatch second = assertInstanceOf(
                BatchDecision.Dispatch.class, algorithm.decide());
        assertEquals(List.of(2L),
                second.items().stream().map(BatchItem::requestId).toList());
        assertEquals(0, algorithm.size());
    }

    @Test
    void unfittingHighPriorityDoesNotLetLaterItemsBypassBreak() {
        FixedWindowBatcherAlgorithm algorithm =
                new FixedWindowBatcherAlgorithm(config(true, 0), endpointWithBatchTokenLimit(100));
        long now = System.currentTimeMillis() - 1_000;
        // Pick order: P70(60), P70(60), P50(10). The second P70 does not fit
        // the padded shape (60*2=120 > 100) and must stop the pick — the
        // fitting P50 behind it must not jump the break.
        algorithm.offer(item(1, now, 60, 70));
        algorithm.offer(item(2, now + 1, 60, 70));
        algorithm.offer(item(3, now + 2, 10, 50));

        BatchDecision.Dispatch first = assertInstanceOf(
                BatchDecision.Dispatch.class, algorithm.decide());
        assertEquals(List.of(1L),
                first.items().stream().map(BatchItem::requestId).toList());
        assertEquals(2, algorithm.size());

        // Same break semantics on the next cycle: P70(60) alone, P50 waits
        BatchDecision.Dispatch second = assertInstanceOf(
                BatchDecision.Dispatch.class, algorithm.decide());
        assertEquals(List.of(2L),
                second.items().stream().map(BatchItem::requestId).toList());

        BatchDecision.Dispatch third = assertInstanceOf(
                BatchDecision.Dispatch.class, algorithm.decide());
        assertEquals(List.of(3L),
                third.items().stream().map(BatchItem::requestId).toList());
        assertEquals(0, algorithm.size());
    }

    @Test
    void kvBudgetLimitsAdditionalMembersInPriorityPickOrder() {
        WorkerStatus status = new WorkerStatus();
        status.setMaxBatchTokensSize(1_000);
        status.getTotalKvCacheTokens().set(100);
        status.getAvailableKvCacheTokens().set(70);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getStatus()).thenReturn(status);

        FixedWindowBatcherAlgorithm algorithm =
                new FixedWindowBatcherAlgorithm(config(true, 0), endpoint);
        long now = System.currentTimeMillis() - 1_000;
        // The newer P70 leads the pick order and is KV-exempt as the head;
        // the older P50 would push combined KV to 120 > 70 and must wait.
        algorithm.offer(item(1, now, 60, 50));
        algorithm.offer(item(2, now + 1, 60, 70));

        BatchDecision.Dispatch first = assertInstanceOf(
                BatchDecision.Dispatch.class, algorithm.decide());
        assertEquals(List.of(2L),
                first.items().stream().map(BatchItem::requestId).toList());
        assertEquals(1, algorithm.size());

        // Next cycle the P50 is the head itself and stays KV-exempt
        BatchDecision.Dispatch second = assertInstanceOf(
                BatchDecision.Dispatch.class, algorithm.decide());
        assertEquals(List.of(1L),
                second.items().stream().map(BatchItem::requestId).toList());
        assertEquals(0, algorithm.size());
    }

    // ---- helpers ----

    /**
     * Assert that the enabled / disabled twin algorithms return the same
     * queueWaitMs. Both calls are sampled within one millisecond tick so the
     * shared {@code System.currentTimeMillis()} anchor cannot skew the
     * comparison.
     */
    private static void assertSameTickQueueWaitMs(FixedWindowBatcherAlgorithm enabled,
                                                  FixedWindowBatcherAlgorithm disabled) {
        for (int attempt = 0; attempt < 1_000; attempt++) {
            long tick = System.currentTimeMillis();
            long enabledWait = enabled.queueWaitMs();
            long disabledWait = disabled.queueWaitMs();
            if (System.currentTimeMillis() == tick) {
                assertEquals(enabledWait, disabledWait,
                        "queueWaitMs must be identical across switch states for the same queue content");
                return;
            }
        }
        fail("could not sample queueWaitMs twice within a single millisecond tick");
    }

    private static FixedWindowBatcherAlgorithm node(boolean priorityEnabled, long fixedWaitMs) {
        return new FixedWindowBatcherAlgorithm(config(priorityEnabled, fixedWaitMs), null);
    }

    private static FlexlbConfig config(boolean priorityEnabled, long fixedWaitMs) {
        FlexlbConfig config = new FlexlbConfig();
        config.setAutoTpmPriorityQueueEnabled(priorityEnabled);
        config.setFlexlbBatchPredictThresholdMs(0);
        config.setFlexlbBatchFixedWaitMs(fixedWaitMs);
        config.setFlexlbBatchSizeMax(32);
        config.setFlexlbBatchFixedMaxInflightBatches(0);
        config.setFlexlbBatchEnqueueDeadlineMs(10_000);
        return config;
    }

    private static PrefillEndpoint endpointWithBatchTokenLimit(long maxBatchTokensSize) {
        WorkerStatus status = new WorkerStatus();
        status.setMaxBatchTokensSize(maxBatchTokensSize);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getStatus()).thenReturn(status);
        return endpoint;
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
