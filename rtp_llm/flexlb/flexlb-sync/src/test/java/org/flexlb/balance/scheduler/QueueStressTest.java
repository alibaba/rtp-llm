package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.strategy.PrefillTimePredictor;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.ScheduleBudget;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.RoutingQueueReporter;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.time.Duration;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * PR-B unit tests for the {@link QueueManager} {@link java.util.concurrent.PriorityBlockingQueue}
 * backed by {@link org.flexlb.util.PriorityOrdering#STRICT}.
 *
 * <p>Covers three core invariants:
 * <ol>
 *   <li><b>Same-priority FIFO</b> — items with equal priority are dequeued
 *       in enqueue-sequence order.</li>
 *   <li><b>Re-enqueue position preservation</b> — a re-offered item (via
 *       {@link QueueManager#offerToHead}) keeps its original sequenceId and
 *       sorts back to the front of its same-priority group.</li>
 *   <li><b>Capacity rejection</b> — when the external size guard fires,
 *       the offer returns {@link StrategyErrorType#QUEUE_FULL}.</li>
 * </ol>
 */
class QueueStressTest {

    private FlexlbConfig config;
    private ConfigService configService;
    private RoutingQueueReporter reporter;

    @BeforeEach
    void setUp() {
        config = new FlexlbConfig();
        configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        reporter = mock(RoutingQueueReporter.class);
    }

    private QueueManager newQueueManager(int maxQueueSize) {
        config.setMaxQueueSize(maxQueueSize);
        return new QueueManager(reporter, configService);
    }

    private static BalanceContext ctx(long requestId, int priority) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(128);
        request.setPriority(priority);
        request.setGenerateTimeout(600_000);  // 10 min — no timeout interference
        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(request);
        ctx.setBudget(ScheduleBudget.forDeadline(priority, System.currentTimeMillis(),
                System.currentTimeMillis() + 300_000));
        return ctx;
    }

    // ==================== 1. Same-priority FIFO ====================

    @Test
    void same_priority_items_are_dequeued_in_enqueue_order() {
        QueueManager qm = newQueueManager(100);

        BalanceContext a = ctx(1, 50);
        BalanceContext b = ctx(2, 50);
        BalanceContext c = ctx(3, 50);

        // Intentionally submit in A-B-C order
        qm.tryRouteAsync(a);
        qm.tryRouteAsync(b);
        qm.tryRouteAsync(c);

        assertEquals(3, qm.queueSize());

        // Same priority → strict FIFO by sequenceId
        assertEquals(1, qm.takeRequest(100).getRequestId());
        assertEquals(2, qm.takeRequest(100).getRequestId());
        assertEquals(3, qm.takeRequest(100).getRequestId());
        assertEquals(0, qm.queueSize());
    }

    // ==================== 2. Re-enqueue position preservation ====================

    @Test
    void re_offered_item_keeps_original_sequence_and_returns_to_front() {
        QueueManager qm = newQueueManager(100);

        BalanceContext a = ctx(1, 50);
        BalanceContext b = ctx(2, 50);
        BalanceContext c = ctx(3, 50);

        qm.tryRouteAsync(a);
        qm.tryRouteAsync(b);
        qm.tryRouteAsync(c);
        assertEquals(3, qm.queueSize());

        // Dequeue the head (A, oldest sequenceId)
        BalanceContext dequeued = qm.takeRequest(100);
        assertEquals(1, dequeued.getRequestId());
        assertEquals(2, qm.queueSize());

        // Re-offer A — it keeps its original (oldest) sequenceId, so it
        // should sort back to the front of the same-priority group.
        qm.offerToHead(dequeued);
        assertEquals(3, qm.queueSize());

        // A must come out first again
        assertEquals(1, qm.takeRequest(100).getRequestId());
        assertEquals(2, qm.takeRequest(100).getRequestId());
        assertEquals(3, qm.takeRequest(100).getRequestId());
    }

    @Test
    void re_offered_higher_priority_item_jumps_ahead_of_lower_priority() {
        QueueManager qm = newQueueManager(100);

        BalanceContext low = ctx(1, 30);
        BalanceContext mid = ctx(2, 50);

        qm.tryRouteAsync(low);
        qm.tryRouteAsync(mid);

        // Dequeue mid (higher priority, comes out first)
        BalanceContext dequeued = qm.takeRequest(100);
        assertEquals(2, dequeued.getRequestId());

        // Re-offer mid — its sequenceId is still newer than low's, but
        // priority dominates: mid (p50) should come out before low (p30).
        qm.offerToHead(dequeued);
        assertEquals(2, qm.queueSize());

        assertEquals(2, qm.takeRequest(100).getRequestId());
        assertEquals(1, qm.takeRequest(100).getRequestId());
    }

    // ==================== 3. Capacity rejection ====================

    @Test
    void offer_beyond_capacity_returns_queue_full() {
        QueueManager qm = newQueueManager(2);

        // Fill the queue
        BalanceContext a = ctx(1, 50);
        BalanceContext b = ctx(2, 50);
        qm.tryRouteAsync(a);
        qm.tryRouteAsync(b);
        assertEquals(2, qm.queueSize());

        // Third offer should be rejected immediately
        BalanceContext c = ctx(3, 50);
        Response response = qm.tryRouteAsync(c).block(Duration.ofMillis(500));

        assertNotNull(response);
        assertEquals(StrategyErrorType.QUEUE_FULL.getErrorCode(), response.getCode());
        assertEquals(2, qm.queueSize(), "rejected item must not be enqueued");
    }

    @Test
    void offer_to_head_beyond_capacity_completes_future_with_queue_full() {
        QueueManager qm = newQueueManager(1);

        BalanceContext a = ctx(1, 50);
        qm.tryRouteAsync(a);
        assertEquals(1, qm.queueSize());

        // Try to re-offer when queue is full — future must be set first
        // (offerToHead completes it with QUEUE_FULL on capacity rejection).
        BalanceContext b = ctx(2, 50);
        b.setFuture(new CompletableFuture<>());
        qm.offerToHead(b);

        assertTrue(b.getFuture().isDone());
        Response response = b.getFuture().getNow(null);
        assertNotNull(response);
        assertEquals(StrategyErrorType.QUEUE_FULL.getErrorCode(), response.getCode());
        assertEquals(1, qm.queueSize(), "rejected re-offer must not be enqueued");
    }

    // ==================== Bonus: priority ordering ====================

    @Test
    void mixed_priority_items_are_dequeued_priority_desc_then_fifo() {
        QueueManager qm = newQueueManager(100);

        // Submit interleaved: p50, p70, p50, p30
        BalanceContext a = ctx(1, 50);
        BalanceContext b = ctx(2, 70);
        BalanceContext c = ctx(3, 50);
        BalanceContext d = ctx(4, 30);

        qm.tryRouteAsync(a);
        qm.tryRouteAsync(b);
        qm.tryRouteAsync(c);
        qm.tryRouteAsync(d);

        // Priority desc: p70 first, then p50 items FIFO (a before c), then p30
        assertEquals(2, qm.takeRequest(100).getRequestId());
        assertEquals(1, qm.takeRequest(100).getRequestId());
        assertEquals(3, qm.takeRequest(100).getRequestId());
        assertEquals(4, qm.takeRequest(100).getRequestId());
    }

    @Test
    void empty_queue_take_returns_null_after_timeout() {
        QueueManager qm = newQueueManager(100);
        assertNull(qm.takeRequest(50));
    }

    // ==================== P0-2: WorkerBatcher concurrent overshoot (PR-B) ====================

    /**
     * N threads concurrently tryOffer into a queue with maxSize=10. The CAS-based
     * {@code reserveQueueSlot} must guarantee that {@code queueDepth} never exceeds
     * maxSize — no unbounded overshoot is allowed.
     */
    @Test
    void concurrentOffer_queueDepthNeverExceedsMaxSize() throws Exception {
        int maxSize = 10;
        int threadCount = 50;

        FlexlbConfig config = new FlexlbConfig();
        config.setFlexlbBatchAlgorithm("slo_budget");
        config.setFlexlbBatchQueueMaxSize(maxSize);

        PrefillEndpoint prefillEp = mock(PrefillEndpoint.class);
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        when(prefillEp.getPredictor()).thenReturn(predictor);
        when(prefillEp.realWaitTimeMs()).thenReturn(0L);
        when(prefillEp.getInflightBatchCount()).thenReturn(0);
        when(predictor.estimateMs(anyLong(), anyLong())).thenReturn(0L);

        BatchDecisionHandler handler = mock(BatchDecisionHandler.class);
        BatchSchedulerReporter reporter = mock(BatchSchedulerReporter.class);

        WorkerBatcher batcher = new WorkerBatcher("stress-test", prefillEp, config, handler, reporter);

        CountDownLatch start = new CountDownLatch(1);
        AtomicInteger successCount = new AtomicInteger(0);
        AtomicInteger failureCount = new AtomicInteger(0);

        Thread[] threads = new Thread[threadCount];
        for (int i = 0; i < threadCount; i++) {
            final long requestId = i + 1;
            threads[i] = new Thread(() -> {
                try {
                    start.await();
                    BatchItem item = batchItem(requestId);
                    if (batcher.tryOffer(item)) {
                        successCount.incrementAndGet();
                    } else {
                        failureCount.incrementAndGet();
                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            });
            threads[i].start();
        }

        start.countDown();
        for (Thread t : threads) {
            t.join(5_000);
        }

        // CAS-based reserveQueueSlot guarantees no overshoot
        assertTrue(batcher.queueSize() <= maxSize,
                "queueSize=" + batcher.queueSize() + " must never exceed maxSize=" + maxSize);
        assertEquals(maxSize, successCount.get(),
                "exactly maxSize offers should succeed");
        assertEquals(threadCount - maxSize, failureCount.get(),
                "remaining offers should be rejected");

        batcher.shutdown();
    }

    private static BatchItem batchItem(long requestId) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(128);
        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(request);
        ctx.setBudget(ScheduleBudget.forDeadline(50, System.currentTimeMillis(),
                System.currentTimeMillis() + 300_000));
        return new BatchItem(ctx, new CompletableFuture<>(), null, null, null, null, null,
                System.currentTimeMillis());
    }

    // ==================== P1-5: low-priority retry preempted by new high-priority (PR-B) ====================

    /**
     * A re-offered low-priority retry (offerToHead) is preempted by a newly arrived
     * high-priority request: the queue is priority-ordered, not insertion-ordered,
     * so the high-priority item is dequeued first even though the retry was re-offered
     * earlier in wall-clock time.
     */
    @Test
    void lowPriorityRetry_isPreemptedByNewHighPriorityArrival() {
        QueueManager qm = newQueueManager(100);

        // 1. Queue a low-priority item
        BalanceContext low = ctx(1, 30);
        qm.tryRouteAsync(low);

        // 2. Dequeue and re-offer (retry) — low-priority goes back to queue
        BalanceContext dequeued = qm.takeRequest(100);
        assertEquals(1, dequeued.getRequestId());
        qm.offerToHead(dequeued);
        assertEquals(1, qm.queueSize());

        // 3. New high-priority arrival
        BalanceContext high = ctx(2, 70);
        qm.tryRouteAsync(high);
        assertEquals(2, qm.queueSize());

        // 4. High-priority is dequeued first despite low-priority retry being re-offered earlier
        assertEquals(2, qm.takeRequest(100).getRequestId());
        assertEquals(1, qm.takeRequest(100).getRequestId());
    }
}
