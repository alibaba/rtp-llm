package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;

/**
 * Concurrency stress tests for the Auto-TPM priority queue: N producer
 * threads offer mixed-priority requests into one {@link WorkerBatcher} while
 * its own run loop consumes via {@link FixedWindowBatcherAlgorithm#decide}
 * and executes dispatch / drop side effects, exactly as in production.
 *
 * <p>Dispatch is intercepted at {@link PrefillEndpoint#submitBatch}: the mock
 * records every dispatched requestId and settles the item's future, so the
 * exactly-once accounting can be asserted end to end:
 * <ul>
 *   <li>no loss — every offered request reaches exactly one terminal state
 *       (dispatch success or drop error), offered = dispatched + dropped</li>
 *   <li>no duplicate dispatch — requestIds recorded at submitBatch are
 *       unique and disjoint from the dropped set</li>
 *   <li>queue depth returns to zero (total and per-priority gauges)</li>
 * </ul>
 *
 * <p>All workloads are deterministic (modulo-derived priorities and sizes,
 * {@link CountDownLatch} start gates, no sleeps) and run in well under a
 * minute. Lifecycle races (offer vs shutdown etc.) are covered separately by
 * {@link ConcurrencyStressTest}.
 */
class ConcurrentPriorityBatcherStressTest {

    private static final int[] PRIORITIES = {30, 40, 50, 60, 70};
    private static final int PRODUCERS = 4;
    private static final int ITEMS_PER_PRODUCER = 100;
    private static final long BATCH_TOKEN_CAPACITY = 100_000;
    private static final long FITTING_SEQ_LEN = 10;
    private static final long OVERSIZED_SEQ_LEN = 200_000;
    /** Every 25th item per producer is oversized and must be dropped. */
    private static final int OVERSIZED_STRIDE = 25;

    // ==================== mixed-priority drain, switch on/off ====================

    @Test
    @Timeout(30)
    void mixedPriorityStressDrainsExactlyOnceWithSwitchOn() throws Exception {
        runMixedPriorityStress(true);
    }

    @Test
    @Timeout(30)
    void mixedPriorityStressDrainsExactlyOnceWithSwitchOff() throws Exception {
        runMixedPriorityStress(false);
    }

    /**
     * Shared stress routine — both switch states must reach the identical
     * conclusion: all fitting requests dispatched exactly once, all oversized
     * requests dropped exactly once, queue fully drained.
     */
    private void runMixedPriorityStress(boolean priorityEnabled) throws Exception {
        FlexlbConfig config = stressConfig(priorityEnabled);
        List<Long> dispatched = new CopyOnWriteArrayList<>();
        WorkerBatcher batcher = new WorkerBatcher("stress", dispatchRecordingEndpoint(dispatched),
                config, mock(BatchSchedulerReporter.class));
        Map<Long, CompletableFuture<Response>> futures = new ConcurrentHashMap<>();
        Set<Long> oversizedIds = ConcurrentHashMap.newKeySet();
        ExecutorService producers = Executors.newFixedThreadPool(PRODUCERS);
        CountDownLatch startGate = new CountDownLatch(1);
        CountDownLatch producersDone = new CountDownLatch(PRODUCERS);
        try {
            batcher.start();
            for (int p = 0; p < PRODUCERS; p++) {
                final int producer = p;
                producers.execute(() -> {
                    try {
                        startGate.await();
                        for (int i = 0; i < ITEMS_PER_PRODUCER; i++) {
                            long requestId = producer * 1_000L + i + 1;
                            boolean oversized = i % OVERSIZED_STRIDE == 0;
                            if (oversized) {
                                oversizedIds.add(requestId);
                            }
                            BatchItem item = item(requestId,
                                    oversized ? OVERSIZED_SEQ_LEN : FITTING_SEQ_LEN,
                                    PRIORITIES[i % PRIORITIES.length]);
                            futures.put(requestId, item.future());
                            batcher.offer(item);
                        }
                    } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();
                    } finally {
                        producersDone.countDown();
                    }
                });
            }
            startGate.countDown();
            assertTrue(producersDone.await(10, TimeUnit.SECONDS), "producers did not finish");

            int total = PRODUCERS * ITEMS_PER_PRODUCER;
            assertEquals(total, futures.size(), "producers must offer every request exactly once");
            CompletableFuture.allOf(futures.values().toArray(new CompletableFuture[0]))
                    .get(15, TimeUnit.SECONDS);

            // No duplicate dispatch: requestIds recorded at submitBatch are unique
            Set<Long> dispatchedIds = new HashSet<>(dispatched);
            assertEquals(dispatched.size(), dispatchedIds.size(), "duplicate dispatch detected");

            // Each future settled exactly once with a definite terminal state
            Set<Long> succeededIds = new HashSet<>();
            Set<Long> droppedIds = new HashSet<>();
            for (Map.Entry<Long, CompletableFuture<Response>> entry : futures.entrySet()) {
                Response response = entry.getValue().get();
                if (response.isSuccess()) {
                    succeededIds.add(entry.getKey());
                } else {
                    droppedIds.add(entry.getKey());
                }
            }
            // No loss: offered = dispatched + dropped, and the sets are disjoint
            assertEquals(total, succeededIds.size() + droppedIds.size());
            assertEquals(dispatchedIds, succeededIds,
                    "every dispatched request must succeed and vice versa");
            assertEquals(oversizedIds, droppedIds,
                    "exactly the oversized requests must be dropped");

            // Queue depth returns to zero, total and per-priority
            assertEquals(0, batcher.queueSize());
            batcher.depthByPriority().forEach((priority, depth) ->
                    assertEquals(0, depth, "queue depth for priority " + priority + " must drain to zero"));
        } finally {
            producers.shutdownNow();
            batcher.shutdown();
        }
    }

    // ==================== deadline expiry under sustained high-priority inflow ====================

    /**
     * Decision consistency under starvation pressure: with the switch on,
     * {@code flexlbBatchSizeMax=1} and high-priority requests flowing in
     * concurrently, the oldest low-priority item still reaches its
     * {@code flexlbBatchEnqueueDeadlineMs} terminal state (expired, not
     * dispatched, never stuck in the queue without a terminal state). The
     * deadline check is anchored to the FIFO head, so priority pick order
     * cannot shadow it.
     */
    @Test
    @Timeout(30)
    void oldestLowPriorityStillExpiresUnderSustainedHighPriorityInflow() throws Exception {
        FlexlbConfig config = stressConfig(true);
        config.setFlexlbBatchSizeMax(1);
        config.setFlexlbBatchEnqueueDeadlineMs(200);

        List<Long> dispatched = new CopyOnWriteArrayList<>();
        WorkerBatcher batcher = new WorkerBatcher("starvation", dispatchRecordingEndpoint(dispatched),
                config, mock(BatchSchedulerReporter.class));
        Map<Long, CompletableFuture<Response>> highPriFutures = new ConcurrentHashMap<>();
        ExecutorService producers = Executors.newFixedThreadPool(2);
        try {
            // Phase 1: concurrent high-priority burst queued ahead of the
            // low-priority item (batcher not started yet, queue only fills)
            CountDownLatch burstDone = new CountDownLatch(2);
            for (int p = 0; p < 2; p++) {
                final long idBase = 10_000L + p * 1_000L;
                producers.execute(() ->
                        offerHighPriority(batcher, highPriFutures, idBase, 30, burstDone));
            }
            assertTrue(burstDone.await(5, TimeUnit.SECONDS), "burst producers did not finish");

            // The low-priority item joins behind the burst, already past its
            // deadline — it must expire once it reaches the FIFO head instead
            // of being dispatched or starving without a terminal state
            BatchItem lowPri = item(1L, System.currentTimeMillis() - 10_000, FITTING_SEQ_LEN, 30);
            batcher.offer(lowPri);

            // Phase 2: start draining while more high-priority load flows in
            batcher.start();
            CountDownLatch inflowDone = new CountDownLatch(2);
            for (int p = 0; p < 2; p++) {
                final long idBase = 20_000L + p * 1_000L;
                producers.execute(() ->
                        offerHighPriority(batcher, highPriFutures, idBase, 50, inflowDone));
            }
            assertTrue(inflowDone.await(5, TimeUnit.SECONDS), "inflow producers did not finish");

            Response lowPriResponse = lowPri.future().get(10, TimeUnit.SECONDS);
            assertFalse(lowPriResponse.isSuccess(), "expired low-priority item must not succeed");
            assertEquals(StrategyErrorType.BATCH_SLO_EXPIRED.getErrorCode(), lowPriResponse.getCode());
            assertFalse(dispatched.contains(1L), "expired item must never be dispatched");

            // All high-priority requests dispatch normally, exactly once each
            CompletableFuture.allOf(highPriFutures.values().toArray(new CompletableFuture[0]))
                    .get(15, TimeUnit.SECONDS);
            for (CompletableFuture<Response> future : highPriFutures.values()) {
                assertTrue(future.get().isSuccess());
            }
            Set<Long> dispatchedIds = new HashSet<>(dispatched);
            assertEquals(dispatched.size(), dispatchedIds.size(), "duplicate dispatch detected");
            assertEquals(highPriFutures.keySet(), dispatchedIds);

            assertEquals(0, batcher.queueSize());
            batcher.depthByPriority().forEach((priority, depth) ->
                    assertEquals(0, depth, "queue depth for priority " + priority + " must drain to zero"));
        } finally {
            producers.shutdownNow();
            batcher.shutdown();
        }
    }

    // ---- helpers ----

    private static void offerHighPriority(WorkerBatcher batcher,
                                          Map<Long, CompletableFuture<Response>> futures,
                                          long idBase,
                                          int count,
                                          CountDownLatch done) {
        try {
            for (int i = 0; i < count; i++) {
                BatchItem item = item(idBase + i, FITTING_SEQ_LEN, 70);
                futures.put(item.requestId(), item.future());
                batcher.offer(item);
            }
        } finally {
            done.countDown();
        }
    }

    private static FlexlbConfig stressConfig(boolean priorityEnabled) {
        FlexlbConfig config = new FlexlbConfig();
        config.setAutoTpmPriorityQueueEnabled(priorityEnabled);
        config.setFlexlbBatchPredictThresholdMs(0);
        config.setFlexlbBatchFixedWaitMs(0);
        config.setFlexlbBatchSizeMax(8);
        config.setFlexlbBatchFixedMaxInflightBatches(0);
        config.setFlexlbBatchEnqueueDeadlineMs(60_000);
        config.setFlexlbBatchQueueMaxSize(0);
        config.setFlexlbBatchMaxCapacity((int) BATCH_TOKEN_CAPACITY);
        return config;
    }

    /**
     * {@link PrefillEndpoint} mock whose {@code submitBatch} records every
     * dispatched requestId and settles the item's future, standing in for the
     * engine ACK path so the batcher's exactly-once contract is observable.
     */
    private static PrefillEndpoint dispatchRecordingEndpoint(List<Long> dispatched) {
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        doAnswer(invocation -> {
            List<BatchItem> items = invocation.getArgument(0);
            for (BatchItem item : items) {
                dispatched.add(item.requestId());
                item.completeSuccess(0);
            }
            return null;
        }).when(endpoint).submitBatch(anyList(), any());
        return endpoint;
    }

    private static BatchItem item(long requestId, long seqLen, int priority) {
        return item(requestId, System.currentTimeMillis(), seqLen, priority);
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
                new Response(), null, null, null, null, enqueuedAtMs);
    }
}
