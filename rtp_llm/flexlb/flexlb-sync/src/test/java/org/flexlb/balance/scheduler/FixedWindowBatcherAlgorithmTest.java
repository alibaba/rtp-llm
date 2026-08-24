package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.strategy.PrefillTimePredictor;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Tests for {@link FixedWindowBatcherAlgorithm}.
 *
 * <p>The algorithm owns its queue, so every test enqueues items via
 * {@link FixedWindowBatcherAlgorithm#offer} and asserts the returned
 * {@link BatchDecision} (and resulting queue state) from
 * {@link FixedWindowBatcherAlgorithm#decide}. No external context or
 * queue mock is needed.
 */
class FixedWindowBatcherAlgorithmTest {

    @Test
    void offerAndSizeTrackQueueState() {
        FixedWindowBatcherAlgorithm algorithm =
                new FixedWindowBatcherAlgorithm(sloCaseConfig(), null);

        assertEquals(0, algorithm.size());
        algorithm.offer(enqueuedItem(1, 100L));
        assertEquals(1, algorithm.size());
        algorithm.offer(enqueuedItem(2, 200L));
        assertEquals(2, algorithm.size());

        algorithm.shutdown();
        assertEquals(0, algorithm.size());
    }

    @Test
    void emptyQueueYieldsNullDecision() {
        FixedWindowBatcherAlgorithm algorithm =
                new FixedWindowBatcherAlgorithm(sloCaseConfig(), null);

        assertNull(algorithm.decide());
    }

    @Test
    void sloCaseDispatchesWhenPredictionReachesThreshold() {
        FlexlbConfig config = sloCaseConfig();
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        when(endpoint.getPredictor()).thenReturn(predictor);
        when(predictor.predictBatchMs(anyList())).thenReturn(500.0);

        FixedWindowBatcherAlgorithm algorithm =
                new FixedWindowBatcherAlgorithm(config, endpoint);
        algorithm.offer(enqueuedItem(1, System.currentTimeMillis()));
        algorithm.offer(enqueuedItem(2, System.currentTimeMillis()));

        BatchDecision decision = algorithm.decide();

        BatchDecision.Dispatch dispatch = assertInstanceOf(BatchDecision.Dispatch.class, decision);
        assertEquals(2, dispatch.items().size());
        assertEquals("predict_threshold", dispatch.reason());
        assertEquals(2, dispatch.queueSizeBefore());
        // Algorithm removed the picked items from its queue
        assertEquals(0, algorithm.size());
    }

    @Test
    void sloCaseDispatchesAtFixedWindowWhenPredictionIsBelowThreshold() {
        FlexlbConfig config = sloCaseConfig();
        FixedWindowBatcherAlgorithm algorithm =
                new FixedWindowBatcherAlgorithm(config, null);
        algorithm.offer(enqueuedItem(1, System.currentTimeMillis() - 170));

        BatchDecision decision = algorithm.decide();

        BatchDecision.Dispatch dispatch = assertInstanceOf(BatchDecision.Dispatch.class, decision);
        assertEquals("fixed_window_timeout", dispatch.reason());
        assertEquals(1, dispatch.items().size());
        assertTrue(dispatch.headWaitMs() >= 170);
        assertEquals(0, algorithm.size());
    }

    @Test
    void sloCaseDispatchesWhenBatchReachesMaxSize() {
        FlexlbConfig config = sloCaseConfig();
        FixedWindowBatcherAlgorithm algorithm =
                new FixedWindowBatcherAlgorithm(config, null);
        long now = System.currentTimeMillis() - 1_000;
        for (int index = 0; index < 32; index++) {
            algorithm.offer(enqueuedItem(index + 1, now));
        }

        BatchDecision decision = algorithm.decide();

        BatchDecision.Dispatch dispatch = assertInstanceOf(BatchDecision.Dispatch.class, decision);
        assertEquals(32, dispatch.items().size());
        assertEquals("batch_full", dispatch.reason());
        assertEquals(32, dispatch.queueSizeBefore());
        assertEquals(0, algorithm.size());
    }

    @Test
    void backpressureYieldsNullParkDecision() {
        FlexlbConfig config = sloCaseConfig();
        config.setFlexlbBatchFixedMaxInflightBatches(1);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.prefillActiveRequestCount()).thenReturn(1);

        FixedWindowBatcherAlgorithm algorithm =
                new FixedWindowBatcherAlgorithm(config, endpoint);
        algorithm.offer(enqueuedItem(1, System.currentTimeMillis() - 1_000));

        assertNull(algorithm.decide());
        // Park decision: item remains in queue
        assertEquals(1, algorithm.size());
    }

    @Test
    void deadlineExceededHeadYieldsDropDecision() {
        FlexlbConfig config = sloCaseConfig();
        config.setFlexlbBatchEnqueueDeadlineMs(100);

        FixedWindowBatcherAlgorithm algorithm =
                new FixedWindowBatcherAlgorithm(config, null);
        BatchItem head = enqueuedItem(1, System.currentTimeMillis() - 1_000, 10);
        algorithm.offer(head);

        BatchDecision decision = algorithm.decide();

        BatchDecision.Drop drop = assertInstanceOf(BatchDecision.Drop.class, decision);
        assertEquals(BatchDecision.DropCause.QUEUE_DEADLINE_EXCEEDED, drop.cause());
        assertEquals(head, drop.item());
        assertTrue(drop.detail().contains("deadline_ms=100"));
        // Algorithm removed the dropped item from its queue
        assertEquals(0, algorithm.size());
        // Settlement happens in the batcher, not the algorithm
        assertFalse(head.future().isDone());
    }

    @Test
    void fixedWindowBatchUsesEnginePaddedTokenCost() {
        FlexlbConfig config = sloCaseConfig();
        config.setFlexlbBatchFixedWaitMs(0);
        config.setFlexlbBatchMaxCapacity(1_000);

        WorkerStatus status = new WorkerStatus();
        status.setMaxSeqLen(200);
        status.setMaxBatchTokensSize(100);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getStatus()).thenReturn(status);

        FixedWindowBatcherAlgorithm algorithm =
                new FixedWindowBatcherAlgorithm(config, endpoint);
        long now = System.currentTimeMillis() - 1_000;
        algorithm.offer(enqueuedItem(1, now, 60));
        algorithm.offer(enqueuedItem(2, now + 1, 50));
        algorithm.offer(enqueuedItem(3, now + 2, 30));

        BatchDecision decision = algorithm.decide();

        BatchDecision.Dispatch dispatch = assertInstanceOf(BatchDecision.Dispatch.class, decision);
        assertEquals(List.of(1L), dispatch.items().stream().map(BatchItem::requestId).toList());
        assertEquals(60L, dispatch.items().stream().mapToLong(BatchItem::seqLen).sum());
        // Only the picked item was removed; others remain queued
        assertEquals(2, algorithm.size());
    }

    @Test
    void largeMrcrRequestIsDispatchedAloneWhenPaddedBatchWouldOverflow() {
        final int engineBatchTokenLimit = 1_048_576;

        FlexlbConfig config = sloCaseConfig();
        config.setFlexlbBatchFixedWaitMs(0);
        config.setFlexlbBatchSizeMax(13);
        config.setFlexlbBatchMaxCapacity(engineBatchTokenLimit);

        WorkerStatus status = new WorkerStatus();
        status.setMaxSeqLen(engineBatchTokenLimit);
        status.setMaxBatchTokensSize(engineBatchTokenLimit);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getStatus()).thenReturn(status);

        FixedWindowBatcherAlgorithm algorithm =
                new FixedWindowBatcherAlgorithm(config, endpoint);
        long now = System.currentTimeMillis() - 1_000;
        algorithm.offer(enqueuedItem(1L, now, 929_760L));
        for (int index = 1; index < 13; index++) {
            algorithm.offer(enqueuedItem(index + 1L, now + index, 9_192L));
        }

        BatchDecision decision = algorithm.decide();

        BatchDecision.Dispatch dispatch = assertInstanceOf(BatchDecision.Dispatch.class, decision);
        assertEquals(List.of(1L), dispatch.items().stream().map(BatchItem::requestId).toList());
        assertEquals(12, algorithm.size());
    }

    @Test
    void dynamicKvBudgetLimitsOnlyAdditionalBatchMembers() {
        FlexlbConfig config = sloCaseConfig();
        config.setFlexlbBatchFixedWaitMs(0);

        WorkerStatus status = new WorkerStatus();
        status.setMaxBatchTokensSize(1_000);
        status.getTotalKvCacheTokens().set(100);
        status.getAvailableKvCacheTokens().set(70);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getStatus()).thenReturn(status);

        FixedWindowBatcherAlgorithm algorithm =
                new FixedWindowBatcherAlgorithm(config, endpoint);
        long now = System.currentTimeMillis() - 1_000;
        algorithm.offer(enqueuedItem(1, now, 60));
        algorithm.offer(enqueuedItem(2, now + 1, 20));
        algorithm.offer(enqueuedItem(3, now + 2, 5));

        BatchDecision decision = algorithm.decide();

        BatchDecision.Dispatch dispatch = assertInstanceOf(BatchDecision.Dispatch.class, decision);
        assertEquals(List.of(1L), dispatch.items().stream().map(BatchItem::requestId).toList());
    }

    @Test
    void everyDispatchedMrcrBatchSatisfiesEngineStrictTokenAdmission() {
        final int requestCount = 32;
        final long seqLen = 32_769L;
        final int engineBatchTokenLimit = 1_048_576;

        FlexlbConfig config = sloCaseConfig();
        config.setFlexlbBatchSizeMax(requestCount);
        config.setFlexlbBatchMaxCapacity(engineBatchTokenLimit);
        config.setFlexlbBatchFixedWaitMs(0);

        WorkerStatus status = new WorkerStatus();
        status.setMaxSeqLen(131_072L);
        status.setMaxBatchTokensSize(engineBatchTokenLimit);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getStatus()).thenReturn(status);

        FixedWindowBatcherAlgorithm algorithm =
                new FixedWindowBatcherAlgorithm(config, endpoint);
        long now = System.currentTimeMillis() - 1_000;
        for (int index = 0; index < requestCount; index++) {
            algorithm.offer(enqueuedItem(index + 1L, now + index, seqLen));
        }

        // Decision cycle 1 — algorithm removes the picked items automatically
        BatchDecision.Dispatch first = assertInstanceOf(
                BatchDecision.Dispatch.class, algorithm.decide());

        // Decision cycle 2 — queue now has only the remaining items
        BatchDecision.Dispatch second = assertInstanceOf(
                BatchDecision.Dispatch.class, algorithm.decide());

        List<List<BatchItem>> batches = List.of(first.items(), second.items());
        assertEquals(List.of(31, 1), batches.stream().map(List::size).toList());
        assertEquals(requestCount, batches.stream().mapToInt(List::size).sum());
        for (List<BatchItem> batch : batches) {
            long totalTokens = batch.stream().mapToLong(BatchItem::seqLen).sum();
            assertTrue(totalTokens < engineBatchTokenLimit,
                    "Engine would reject batch with total_tokens=" + totalTokens);
        }
        assertEquals(0, algorithm.size());
    }

    @Test
    void maxSeqLenIsUsedWhenWorkerDoesNotReportBatchTokenLimit() {
        FlexlbConfig config = sloCaseConfig();
        config.setFlexlbBatchFixedWaitMs(0);
        config.setFlexlbBatchMaxCapacity(1_000);

        WorkerStatus status = new WorkerStatus();
        status.setMaxSeqLen(100);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getStatus()).thenReturn(status);

        FixedWindowBatcherAlgorithm algorithm =
                new FixedWindowBatcherAlgorithm(config, endpoint);
        long now = System.currentTimeMillis();
        algorithm.offer(enqueuedItem(1, now, 60));
        algorithm.offer(enqueuedItem(2, now + 1, 40));

        BatchDecision decision = algorithm.decide();

        BatchDecision.Dispatch dispatch = assertInstanceOf(BatchDecision.Dispatch.class, decision);
        assertEquals(List.of(1L), dispatch.items().stream().map(BatchItem::requestId).toList());
    }

    @Test
    void requestAtEngineTokenLimitIsRejectedBeforeDispatch() {
        // Covers the decide() fallback path: batchTokenCapacity is partly
        // worker-reported and may shrink after the offer-time check() admitted
        // the item, so an oversized head must still be rejected here.
        FlexlbConfig config = sloCaseConfig();
        config.setFlexlbBatchFixedWaitMs(0);
        config.setFlexlbBatchMaxCapacity(1_000);

        WorkerStatus status = new WorkerStatus();
        status.setMaxBatchTokensSize(100);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getStatus()).thenReturn(status);

        FixedWindowBatcherAlgorithm algorithm =
                new FixedWindowBatcherAlgorithm(config, endpoint);
        BatchItem item = enqueuedItem(1, 1, 100);
        algorithm.offer(item);

        BatchDecision decision = algorithm.decide();

        BatchDecision.Drop drop = assertInstanceOf(BatchDecision.Drop.class, decision);
        assertEquals(BatchDecision.DropCause.EXCEEDS_BATCH_TOKEN_CAPACITY, drop.cause());
        assertEquals(item, drop.item());
        assertTrue(drop.detail().contains("seq_len=100"));
        assertTrue(drop.detail().contains("capacity=100"));
        // Algorithm removed the dropped item from its queue
        assertEquals(0, algorithm.size());
        // Settlement happens in the batcher, not the algorithm
        assertFalse(item.future().isDone());
    }

    @Test
    void offerRejectsOversizedRequestViaCheckWithoutEnqueue() {
        FlexlbConfig config = sloCaseConfig();
        config.setFlexlbBatchMaxCapacity(100);

        WorkerBatcher batcher = new WorkerBatcher(
                "test", null, config, mock(BatchSchedulerReporter.class));

        // seqLen == capacity → padded shape does not fit the strict limit
        BatchItem oversized = enqueuedItem(1, System.currentTimeMillis(), 100);
        batcher.offer(oversized);

        assertTrue(oversized.future().isDone());
        Response response = oversized.future().join();
        assertFalse(response.isSuccess());
        assertTrue(response.getErrorMessage()
                .contains("cannot fit strict padded batch token capacity"));
        assertEquals(0, batcher.queueSize());
    }

    @Test
    void offerCheckRejectionDoesNotLeakQueueDepth() {
        FlexlbConfig config = sloCaseConfig();
        config.setFlexlbBatchMaxCapacity(100);
        config.setFlexlbBatchQueueMaxSize(1);

        WorkerBatcher batcher = new WorkerBatcher(
                "test", null, config, mock(BatchSchedulerReporter.class));

        for (long id = 1; id <= 3; id++) {
            BatchItem oversized = enqueuedItem(id, System.currentTimeMillis(), 200);
            batcher.offer(oversized);
            assertTrue(oversized.future().isDone());
        }
        assertEquals(0, batcher.queueSize());

        // Rejected offers must not consume the single queue slot
        BatchItem admitted = enqueuedItem(9, System.currentTimeMillis(), 60);
        batcher.offer(admitted);
        assertFalse(admitted.future().isDone());
        assertEquals(1, batcher.queueSize());
    }

    @Test
    void offerAdmitsRequestWithinCapacityAndEnqueues() {
        FlexlbConfig config = sloCaseConfig();
        config.setFlexlbBatchMaxCapacity(100);

        WorkerBatcher batcher = new WorkerBatcher(
                "test", null, config, mock(BatchSchedulerReporter.class));

        BatchItem item = enqueuedItem(1, System.currentTimeMillis(), 60);
        batcher.offer(item);

        assertFalse(item.future().isDone());
        assertEquals(1, batcher.queueSize());
    }

    @Test
    void defaultCheckAdmitsWithinCapacityAndRejectsOversized() {
        FlexlbConfig config = sloCaseConfig();
        config.setFlexlbBatchMaxCapacity(100);
        FixedWindowBatcherAlgorithm algorithm =
                new FixedWindowBatcherAlgorithm(config, null);

        assertNull(algorithm.check(enqueuedItem(1, 1, 60)));

        String reason = algorithm.check(enqueuedItem(2, 2, 100));
        assertEquals("request seq_len=100 cannot fit strict padded batch token capacity=100", reason);
    }

    // ---- helpers ----

    private static FlexlbConfig sloCaseConfig() {
        FlexlbConfig config = new FlexlbConfig();
        config.setFlexlbBatchPredictThresholdMs(500);
        config.setFlexlbBatchFixedWaitMs(160);
        config.setFlexlbBatchSizeMax(32);
        config.setFlexlbBatchFixedMaxInflightBatches(0);
        config.setFlexlbBatchEnqueueDeadlineMs(10_000);
        return config;
    }

    private static BatchItem enqueuedItem(long requestId, long enqueuedAtMs) {
        return new BatchItem(null, new CompletableFuture<>(),
                null, null, null, null, null, enqueuedAtMs);
    }

    private static BatchItem enqueuedItem(long requestId, long enqueuedAtMs, long seqLen) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(seqLen);
        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setRequest(request);
        return new BatchItem(
                balanceContext, new CompletableFuture<>(),
                null, null, null, null, null, enqueuedAtMs);
    }
}
