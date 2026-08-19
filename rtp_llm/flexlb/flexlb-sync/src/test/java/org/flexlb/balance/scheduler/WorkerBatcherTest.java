package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.ScheduleBudget;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.PriorityBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.ReentrantLock;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;

/**
 * Tests for {@link WorkerBatcher#queueSizeByPriority()}: per-priority
 * bucketing, the legacy priority-0 bucket, and the present-only empty-bucket
 * behavior (drained priorities disappear from the snapshot — same convention
 * as the batch wait-time-by-priority series).
 *
 * <p>Same construction pattern as {@link PrefillQueueManagerTest}: the
 * {@code fixed_window} algorithm needs no predictor, and the batcher is
 * never started so the queue content is fully deterministic.
 */
class WorkerBatcherTest {

    private FlexlbConfig config;

    @BeforeEach
    void setUp() {
        config = new FlexlbConfig();
        config.setFlexlbBatchAlgorithm("fixed_window");
        config.setAutoTpmEnabled(true);
    }

    private WorkerBatcher newBatcher() {
        return new WorkerBatcher("test-worker", null, config,
                mock(BatchDecisionHandler.class), mock(BatchSchedulerReporter.class));
    }

    @Test
    void queue_size_by_priority_buckets_multiple_priorities() {
        WorkerBatcher batcher = newBatcher();
        long now = System.currentTimeMillis();

        assertTrue(batcher.tryOffer(item(1, 70, now)));
        assertTrue(batcher.tryOffer(item(2, 50, now)));
        assertTrue(batcher.tryOffer(item(3, 50, now)));
        assertTrue(batcher.tryOffer(item(4, 30, now)));

        Map<Integer, Integer> buckets = batcher.queueSizeByPriority();
        assertEquals(Map.of(70, 1, 50, 2, 30, 1), buckets);
        // Bucket sum matches the global queue size
        assertEquals(batcher.queueSize(), buckets.values().stream().mapToInt(Integer::intValue).sum());
    }

    @Test
    void legacy_items_without_budget_fall_into_priority_zero_bucket() {
        config.setAutoTpmEnabled(false);
        WorkerBatcher batcher = newBatcher();
        long now = System.currentTimeMillis();

        assertTrue(batcher.tryOffer(legacyItem(1, now)));
        assertTrue(batcher.tryOffer(legacyItem(2, now)));

        assertEquals(Map.of(0, 2), batcher.queueSizeByPriority());
    }

    @Test
    void empty_queue_returns_empty_map() {
        assertEquals(Map.of(), newBatcher().queueSizeByPriority());
    }

    @Test
    void drained_priorities_disappear_from_snapshot() {
        WorkerBatcher batcher = newBatcher();
        long now = System.currentTimeMillis();

        assertTrue(batcher.tryOffer(item(1, 70, now)));
        assertTrue(batcher.tryOffer(item(2, 50, now)));

        // Drain the P70 item: its bucket drops out (present-only, no zero-fill
        // — same empty-bucket behavior as wait-time-by-priority)
        List<BatchItem> removed = batcher.tryRemoveNoVersion(List.of(1L), "test-drain");
        assertEquals(1, removed.size());

        assertEquals(Map.of(50, 1), batcher.queueSizeByPriority());

        // Fully drained queue reports no buckets at all
        assertEquals(1, batcher.tryRemoveNoVersion(List.of(2L), "test-drain").size());
        assertEquals(Map.of(), batcher.queueSizeByPriority());
    }

    @Test
    void dispatchCallbackFailure_restoresOnlyStagedItemsWithoutDepthLeak() {
        PriorityBlockingQueue<BatchItem> queue = new PriorityBlockingQueue<>(
                11, WorkerBatcher.AUTO_TPM_QUEUE_ORDER);
        BatchItem first = item(1, 50, 100);
        BatchItem second = item(2, 50, 200);
        queue.add(first);
        queue.add(second);
        AtomicInteger depth = new AtomicInteger(2);
        BatcherContext ctx = context(queue, depth, new BatchDecisionHandler() {
            @Override
            public void onExpired(BatchItem head) {
            }

            @Override
            public void onBatchReady(List<BatchItem> items, DispatchMeta meta) {
                throw new IllegalStateException("test callback failure");
            }

            @Override
            public void onOfferFailure(BatchItem item, Throwable error) {
            }
        });

        assertThrows(IllegalStateException.class,
                () -> ctx.dispatch(List.of(first, second), new DispatchMeta("test", 0)));

        assertEquals(2, depth.get());
        assertEquals(0, ctx.dispatchPendingSize());
        List<BatchItem> restored = ctx.sortedItems();
        assertEquals(List.of(1L, 2L), restored.stream().map(BatchItem::requestId).toList());
        assertSame(first, restored.get(0));
        assertSame(second, restored.get(1));
    }

    @Test
    void successfulLegacyCallbackConsumesDistinctItemsSharingRequestId() {
        PriorityBlockingQueue<BatchItem> queue = new PriorityBlockingQueue<>(
                11, WorkerBatcher.AUTO_TPM_QUEUE_ORDER);
        BatchItem first = item(0, 50, 100);
        BatchItem second = item(0, 50, 200);
        queue.add(first);
        queue.add(second);
        AtomicInteger depth = new AtomicInteger(2);
        AtomicInteger callbackMembers = new AtomicInteger();
        BatcherContext ctx = context(queue, depth, new BatchDecisionHandler() {
            @Override
            public void onExpired(BatchItem head) {
            }

            @Override
            public void onBatchReady(List<BatchItem> items, DispatchMeta meta) {
                callbackMembers.set(items.size());
            }

            @Override
            public void onOfferFailure(BatchItem item, Throwable error) {
            }
        });

        ctx.dispatch(List.of(first, second), new DispatchMeta("test", 0));

        assertEquals(2, callbackMembers.get());
        assertEquals(0, depth.get(),
                "a successful legacy callback consumes unclaimed staged members");
        assertEquals(0, ctx.dispatchPendingSize());
        assertTrue(ctx.sortedItems().isEmpty());
    }

    @Test
    void claimedDispatchCompletesOnce_andFinallyCannotRequeueIt() {
        PriorityBlockingQueue<BatchItem> queue = new PriorityBlockingQueue<>(
                11, WorkerBatcher.AUTO_TPM_QUEUE_ORDER);
        BatchItem item = item(7, 50, 100);
        queue.add(item);
        AtomicInteger depth = new AtomicInteger(1);
        AtomicInteger dispatchCalls = new AtomicInteger();
        AtomicReference<BatcherContext> owner = new AtomicReference<>();
        BatcherContext ctx = context(queue, depth, new BatchDecisionHandler() {
            @Override
            public void onExpired(BatchItem head) {
            }

            @Override
            public void onBatchReady(List<BatchItem> items, DispatchMeta meta) {
                dispatchCalls.incrementAndGet();
                long stagedVersion = owner.get().queueVersionValue();
                assertEquals(BatcherContext.PendingClaimResult.CLAIMED,
                        owner.get().claimPendingDispatch(item));
                assertEquals(stagedVersion + 1, owner.get().queueVersionValue(),
                        "queue-to-dispatch ownership must invalidate versioned plans");
                long claimedVersion = owner.get().queueVersionValue();
                assertTrue(owner.get().completePendingDispatch(item));
                assertEquals(claimedVersion + 1, owner.get().queueVersionValue(),
                        "releasing charged capacity must invalidate versioned offers");
            }

            @Override
            public void onOfferFailure(BatchItem item, Throwable error) {
            }
        });
        owner.set(ctx);

        ctx.dispatch(List.of(item), new DispatchMeta("test", 0));
        ctx.dispatch(List.of(item), new DispatchMeta("test", 0));

        assertEquals(1, dispatchCalls.get(), "a claimed member must not be dispatched twice");
        assertEquals(0, depth.get());
        assertEquals(0, ctx.dispatchPendingSize());
        assertTrue(ctx.sortedItems().isEmpty());
    }

    @Test
    void claimedCallbackFailure_isTerminatedWithoutPendingLeakOrRequeue() {
        PriorityBlockingQueue<BatchItem> queue = new PriorityBlockingQueue<>(
                11, WorkerBatcher.AUTO_TPM_QUEUE_ORDER);
        BatchItem item = item(8, 50, 100);
        queue.add(item);
        AtomicInteger depth = new AtomicInteger(1);
        AtomicInteger offerFailures = new AtomicInteger();
        AtomicReference<BatcherContext> owner = new AtomicReference<>();
        BatcherContext ctx = context(queue, depth, new BatchDecisionHandler() {
            @Override
            public void onExpired(BatchItem head) {
            }

            @Override
            public void onBatchReady(List<BatchItem> items, DispatchMeta meta) {
                assertEquals(BatcherContext.PendingClaimResult.CLAIMED,
                        owner.get().claimPendingDispatch(item));
                throw new IllegalStateException("failed after claim");
            }

            @Override
            public void onOfferFailure(BatchItem failed, Throwable error) {
                assertSame(item, failed);
                assertEquals("failed after claim", error.getMessage());
                offerFailures.incrementAndGet();
            }
        });
        owner.set(ctx);

        assertThrows(IllegalStateException.class,
                () -> ctx.dispatch(List.of(item), new DispatchMeta("test", 0)));
        assertEquals(1, offerFailures.get());
        assertEquals(0, depth.get());
        assertEquals(0, ctx.dispatchPendingSize());
        assertTrue(ctx.sortedItems().isEmpty());
    }

    @Test
    void shutdownDrainWinsStagedItemExactlyOnce() throws Exception {
        PriorityBlockingQueue<BatchItem> queue = new PriorityBlockingQueue<>(
                11, WorkerBatcher.AUTO_TPM_QUEUE_ORDER);
        BatchItem item = item(9, 50, 100);
        queue.add(item);
        AtomicInteger depth = new AtomicInteger(1);
        AtomicInteger offerFailures = new AtomicInteger();
        CountDownLatch callbackEntered = new CountDownLatch(1);
        CountDownLatch callbackMayReturn = new CountDownLatch(1);
        BatcherContext ctx = context(queue, depth, new BatchDecisionHandler() {
            @Override
            public void onExpired(BatchItem head) {
            }

            @Override
            public void onBatchReady(List<BatchItem> items, DispatchMeta meta) {
                callbackEntered.countDown();
                try {
                    assertTrue(callbackMayReturn.await(2, TimeUnit.SECONDS));
                } catch (InterruptedException e) {
                    throw new IllegalStateException(e);
                }
            }

            @Override
            public void onOfferFailure(BatchItem failed, Throwable error) {
                assertSame(item, failed);
                offerFailures.incrementAndGet();
            }
        });

        CompletableFuture<Void> dispatch = CompletableFuture.runAsync(() ->
                ctx.dispatch(List.of(item), new DispatchMeta("test", 0)));
        assertTrue(callbackEntered.await(2, TimeUnit.SECONDS));
        long stagedVersion = ctx.queueVersionValue();
        List<BatchItem> drained = new java.util.ArrayList<>();
        ctx.stopAndDrainTo(drained);
        assertEquals(List.of(item), drained);
        assertEquals(stagedVersion + 1, ctx.queueVersionValue(),
                "shutdown releasing a staged capacity slot must invalidate versioned offers");
        callbackMayReturn.countDown();
        dispatch.get(2, TimeUnit.SECONDS);

        assertEquals(0, depth.get());
        assertEquals(0, ctx.dispatchPendingSize());
        assertTrue(ctx.sortedItems().isEmpty());
        assertEquals(0, offerFailures.get(),
                "shutdown owns the drained item; callback finally must not deliver it twice");
    }

    @Test
    void dispatchPendingItemKeepsCapacityChargedUntilCallbackResolves() throws Exception {
        config.setFlexlbBatchQueueMaxSize(1);
        config.setFlexlbBatchSizeMax(1);
        config.setFlexlbBatchFixedWaitMs(0);
        CountDownLatch callbackEntered = new CountDownLatch(1);
        CountDownLatch callbackMayReturn = new CountDownLatch(1);
        BatchDecisionHandler handler = new BatchDecisionHandler() {
            @Override
            public void onExpired(BatchItem head) {
            }

            @Override
            public void onBatchReady(List<BatchItem> items, DispatchMeta meta) {
                callbackEntered.countDown();
                try {
                    assertTrue(callbackMayReturn.await(2, TimeUnit.SECONDS));
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    throw new IllegalStateException(e);
                }
            }

            @Override
            public void onOfferFailure(BatchItem item, Throwable error) {
            }
        };
        WorkerBatcher batcher = new WorkerBatcher(
                "capacity-test", mock(PrefillEndpoint.class),
                config, handler, mock(BatchSchedulerReporter.class));
        batcher.start();
        try {
            assertTrue(batcher.tryOffer(item(20, 50, System.currentTimeMillis())));
            assertTrue(callbackEntered.await(2, TimeUnit.SECONDS));
            assertTrue(batcher.queueManager().snapshot().items().isEmpty(),
                    "staged item is no longer a live queue member");
            assertEquals(1, batcher.queueSize(),
                    "staged callback ownership must retain the charged slot");
            assertFalse(batcher.tryOffer(item(21, 50, System.currentTimeMillis())),
                    "charged staged capacity must reject a second offer");

            callbackMayReturn.countDown();
            long deadlineNanos = System.nanoTime() + TimeUnit.SECONDS.toNanos(2);
            while (batcher.queueSize() != 0 && System.nanoTime() < deadlineNanos) {
                Thread.sleep(1);
            }
            assertEquals(0, batcher.queueSize());
        } finally {
            callbackMayReturn.countDown();
            batcher.shutdown();
        }
    }

    @Test
    void dropHeadOnlyNotifiesExpiryWhenTheItemWasActuallyRemoved() {
        PriorityBlockingQueue<BatchItem> queue = new PriorityBlockingQueue<>(
                11, WorkerBatcher.AUTO_TPM_QUEUE_ORDER);
        BatchItem queued = item(30, 50, 100);
        queue.add(queued);
        AtomicInteger depth = new AtomicInteger(1);
        AtomicInteger expired = new AtomicInteger();
        BatcherContext ctx = context(queue, depth, new BatchDecisionHandler() {
            @Override
            public void onExpired(BatchItem head) {
                assertSame(queued, head);
                expired.incrementAndGet();
            }

            @Override
            public void onBatchReady(List<BatchItem> items, DispatchMeta meta) {
            }

            @Override
            public void onOfferFailure(BatchItem item, Throwable error) {
            }
        });

        ctx.dropHead(item(31, 50, 100));
        assertEquals(0, expired.get());
        assertEquals(1, depth.get());

        ctx.dropHead(queued);
        assertEquals(1, expired.get());
        assertEquals(0, depth.get());
    }

    // ==================== helpers ====================

    private static BatchItem item(long requestId, int priority, long enqueuedAtMs) {
        BalanceContext ctx = newContext(requestId, priority);
        ctx.setBudget(ScheduleBudget.forDeadline(priority, enqueuedAtMs, enqueuedAtMs + 5_000));
        return new BatchItem(ctx, new CompletableFuture<>(), null,
                null, null, null, null, enqueuedAtMs);
    }

    /** Legacy path: budget = null, so {@link BatchItem#priority()} returns 0. */
    private static BatchItem legacyItem(long requestId, long enqueuedAtMs) {
        return new BatchItem(newContext(requestId, 0), new CompletableFuture<>(), null,
                null, null, null, null, enqueuedAtMs);
    }

    private static BalanceContext newContext(long requestId, int priority) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(128);
        request.setPriority(priority);
        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(request);
        return ctx;
    }

    private BatcherContext context(PriorityBlockingQueue<BatchItem> queue,
                                   AtomicInteger depth,
                                   BatchDecisionHandler handler) {
        return new BatcherContext("test-worker", null, config, handler, queue, depth,
                new AtomicLong(), new ReentrantLock(), WorkerBatcher.AUTO_TPM_QUEUE_ORDER,
                mock(BatchSchedulerReporter.class));
    }
}
